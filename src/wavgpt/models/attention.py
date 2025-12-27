from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import InfiniteContextConfig


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = query.shape
        _, S, _ = key.shape

        q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal_mask = torch.triu(torch.ones(T, S, device=query.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.out_proj(out)
