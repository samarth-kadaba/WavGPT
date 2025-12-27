from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class InfiniteContextConfig:
    """Configuration for Infinite Context Transformer."""

    # GPT-2 Small equivalent (~120M params)
    vocab_size: int = 50257
    hidden_size: int = 768  # GPT-2: 768
    n_heads: int = 12  # GPT-2: 12 (head_dim=64)
    head_dim: int = 64

    # SSM configuration
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2

    # Chunk configuration
    min_chunk_size: int = 32  # Minimum tokens per chunk (larger = fewer chunks = faster)
    max_chunks: int = 256  # Maximum chunks (context window for transformer)

    # Soft assignment temperature for differentiable chunking
    soft_assign_temperature: float = 0.5

    # Compression regularization - balances LM loss vs chunk count
    compression_weight: float = 0.0  # Weight for compression loss (start low, let LM loss dominate)

    # Layer counts
    n_boundary_layers: int = 2  # Boundary detection SSM
    n_chunk_ssm_layers: int = 2  # Chunk compression SSM
    n_chunk_transformer_layers: int = 8  # Main compute (transformer)

    # Architecture
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Gumbel-Softmax temperature for boundary sampling (learnable)
    gumbel_temperature_init: float = 1.0

    # Memory optimization
    gradient_checkpointing: bool = False

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.n_heads
        assert self.hidden_size % self.n_heads == 0


@dataclass
class GenerationState:
    """Maintains state for incremental generation."""

    committed_chunk_embeds: List[torch.Tensor] = field(default_factory=list)
    committed_chunk_contextualized: Optional[torch.Tensor] = None

    chunk_conv_states: Optional[List[torch.Tensor]] = None
    chunk_ssm_states: Optional[List[torch.Tensor]] = None
    current_ssm_output: Optional[torch.Tensor] = None
    current_chunk_size: int = 0

    boundary_conv_states: Optional[List[torch.Tensor]] = None
    boundary_ssm_states: Optional[List[torch.Tensor]] = None
    boundary_prev_hidden: Optional[torch.Tensor] = None
    boundary_prev_avg_log_prob: Optional[torch.Tensor] = None
    boundary_token_count: int = 0
