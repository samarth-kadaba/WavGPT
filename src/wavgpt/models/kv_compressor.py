"""KV-cache compressor.

Maps a prefix's (K, V) of length T to a fixed-size cache of length K_slots via
a learned cross-attention with K_slots queries, biased by an SSM-derived
per-position importance score. Fully differentiable; trained end-to-end on
LM cross-entropy on a held-out continuation.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import CompressorConfig
from wavgpt.models.ssm import SSMBackbone


class CompressorOutput(NamedTuple):
    K_out: torch.Tensor              # (B, n_layers, n_heads, K_slots, head_dim)
    V_out: torch.Tensor              # (B, n_layers, n_heads, K_slots, head_dim)
    mixing_weights: torch.Tensor     # (B, K_slots, T) — last sample's W
    importance: torch.Tensor         # (B, T)


class KVCompressor(nn.Module):
    """SSM importance + learnable slot queries + cross-attention mixing."""

    def __init__(
        self,
        config: CompressorConfig,
        pretrained_dim: int,
    ):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        self.K_slots = config.max_kv_slots

        self.input_proj = nn.Linear(pretrained_dim, config.compress_dim)
        self.input_norm = nn.LayerNorm(config.compress_dim)

        self.backbone = SSMBackbone(
            d_model=config.compress_dim,
            n_layers=config.n_ssm_layers,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        # Per-slot learned queries.
        self.slot_queries = nn.Parameter(torch.randn(self.K_slots, config.compress_dim) * 0.02)

        # Cross-attention projections (kept low-dim in compress_dim space).
        self.q_proj = nn.Linear(config.compress_dim, config.compress_dim, bias=False)
        self.k_proj = nn.Linear(config.compress_dim, config.compress_dim, bias=False)

        self.importance_head = nn.Sequential(
            nn.Linear(config.compress_dim, config.compress_dim),
            nn.GELU(),
            nn.Linear(config.compress_dim, 1),
        )

        self.log_importance_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def importance_temperature(self) -> torch.Tensor:
        return self.log_importance_temp.exp().clamp(
            self.config.importance_temp_min, self.config.importance_temp_max
        )

    def compute_mixing_weights(
        self,
        hidden_states: torch.Tensor,                # (B, T, pretrained_dim)
        attention_mask: Optional[torch.Tensor] = None,
        gumbel_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns W (B, K_slots, T) and importance (B, T)."""
        x = self.input_norm(self.input_proj(hidden_states))
        h = self.backbone(x)  # (B, T, compress_dim)

        importance = self.importance_head(h).squeeze(-1)  # (B, T)
        if gumbel_noise and self.training:
            g = -torch.empty_like(importance).exponential_().log()  # Gumbel(0,1)
            importance = importance + g

        # Cross-attention: K_slots queries attend over T positions.
        q = self.q_proj(self.slot_queries)            # (K_slots, compress_dim)
        k = self.k_proj(h)                            # (B, T, compress_dim)

        scale = 1.0 / math.sqrt(self.config.compress_dim)
        logits = torch.einsum("kd,btd->bkt", q, k) * scale
        logits = logits + (importance / self.importance_temperature).unsqueeze(1)

        if attention_mask is not None:
            logits = logits.masked_fill(
                (attention_mask == 0).unsqueeze(1), float("-inf"),
            )

        W = F.softmax(logits, dim=-1)
        return W, importance

    def apply_mixing(
        self,
        W: torch.Tensor,                                  # (B, K_slots, T)
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Apply the same W to every layer's (K, V).

        Each layer's K, V are (B, n_heads, T, head_dim). The mixing produces
        (B, n_heads, K_slots, head_dim) per layer.
        """
        compressed = []
        # W: (B, K_slots, T) -> broadcast over heads as (B, 1, K_slots, T)
        W_b = W.unsqueeze(1)
        for K, V in past_key_values:
            K_out = torch.matmul(W_b, K)  # (B, n_heads, K_slots, head_dim)
            V_out = torch.matmul(W_b, V)
            compressed.append((K_out, V_out))
        return tuple(compressed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        attention_mask: Optional[torch.Tensor] = None,
        gumbel_noise: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor, torch.Tensor]:
        """Returns (compressed_past_kv, W, importance)."""
        W, importance = self.compute_mixing_weights(
            hidden_states, attention_mask=attention_mask, gumbel_noise=gumbel_noise,
        )
        compressed = self.apply_mixing(W, past_key_values)
        return compressed, W, importance


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def coverage_loss(W: torch.Tensor) -> torch.Tensor:
    """Encourage different slots to attend to different positions.

    Penalises the off-diagonal of W @ W.T (slot-slot similarity)."""
    K_slots = W.size(1)
    sim = torch.matmul(W, W.transpose(1, 2))  # (B, K_slots, K_slots)
    eye = torch.eye(K_slots, device=W.device, dtype=W.dtype)
    off_diag = sim - sim * eye
    return (off_diag ** 2).mean()


def sparsity_loss(W: torch.Tensor) -> torch.Tensor:
    """Encourage each slot's mixing to be peaked (low entropy)."""
    eps = 1e-9
    entropy = -(W * (W + eps).log()).sum(dim=-1)  # (B, K_slots)
    return entropy.mean()
