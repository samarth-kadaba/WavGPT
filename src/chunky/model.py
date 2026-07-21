"""Pre-norm transformer: RMSNorm, SwiGLU, RoPE, weight-tied embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ffn_mult: float = 3.0
    max_seq_len: int = 16384  # RoPE cache; training uses seq_len, val extrapolates beyond
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


SCALES = {
    "xs": ModelConfig(n_layers=8, d_model=512, n_heads=8),
    "s": ModelConfig(n_layers=12, d_model=768, n_heads=12),
    "m": ModelConfig(n_layers=24, d_model=1024, n_heads=16),
    "l": ModelConfig(n_layers=36, d_model=1280, n_heads=20),
    "xl": ModelConfig(n_layers=48, d_model=1600, n_heads=25),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.n_heads * self.head_dim, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x, cos, sin, attn_mask):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))

    def forward_streaming(self, x, cos, sin, mem_k, mem_v, mem_active):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        K = k if mem_k is None else torch.cat((mem_k, k), dim=2)
        V = v if mem_v is None else torch.cat((mem_v, v), dim=2)
        m = 0 if mem_k is None else mem_k.size(2)
        mask = x.new_zeros(T, m + T)
        # Exclude inactive (zero-padded) memory slots so they don't steal softmax mass.
        if m and mem_active is not None:
            mask[:, :m].masked_fill_(~mem_active.view(1, m), float("-inf"))
        mask[:, m:].masked_fill_(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1)), k, v


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_streaming(self, x, cos, sin, mem_k, mem_v, mem_active):
        attn_out, k, v = self.attn.forward_streaming(self.attn_norm(x), cos, sin, mem_k, mem_v, mem_active)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), k, v


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, attn_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        T = input_ids.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        logits = self.lm_head(self.norm(x))
        loss = _lm_loss(logits, labels) if labels is not None else None
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.embed.weight.numel()
        return n


def _lm_loss(logits, labels):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
