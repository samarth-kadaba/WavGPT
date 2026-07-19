"""Cross-attention KV compressor: mix N cache entries into learned slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CompressorConfig:
    d_model: int
    n_heads: int
    max_slots: int = 512
    compress_dim: int = 256
    occupancy_beta: float = 0.75
    occupancy_scale: float = 4096.0
    importance_temp: float = 1.0


def active_slot_mask(cfg: CompressorConfig, tokens_seen: float, device) -> torch.Tensor:
    """Soft mask over slots; active count grows to beta*max_slots, never reaching it."""
    target = cfg.occupancy_beta * cfg.max_slots * (1.0 - math.exp(-tokens_seen / cfg.occupancy_scale))
    return torch.sigmoid(target - torch.arange(cfg.max_slots, device=device).float())


def fold_heads(kv: torch.Tensor) -> torch.Tensor:
    B, H, N, hd = kv.shape
    return kv.transpose(1, 2).reshape(B, N, H * hd)


class KVCompressor(nn.Module):
    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.compress_dim
        self.slots = nn.Parameter(torch.randn(cfg.max_slots, c) * 0.02)
        self.q_proj = nn.Linear(c, c, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, c, bias=False)
        self.importance = nn.Sequential(nn.Linear(cfg.d_model, c), nn.GELU(), nn.Linear(c, 1))
        self.log_temp = nn.Parameter(torch.tensor(math.log(cfg.importance_temp)))

    def mixing_weights(self, candidates, active_mask=None):
        c = self.cfg.compress_dim
        q = self.q_proj(self.slots)
        k = self.k_proj(candidates)
        s = self.importance(candidates).squeeze(-1)
        logits = torch.einsum("mc,bnc->bmn", q, k) / math.sqrt(c) + (s / self.log_temp.exp()).unsqueeze(1)
        A = F.softmax(logits, dim=-1)
        if active_mask is not None:
            A = A * active_mask.to(A.dtype).view(1, -1, 1)
        return A

    def forward(self, K, V, active_mask: Optional[torch.Tensor] = None):
        A = self.mixing_weights(fold_heads(K), active_mask).unsqueeze(1)
        return A @ K, A @ V
