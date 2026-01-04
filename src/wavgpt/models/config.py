from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InfiniteContextConfig:
    """Configuration for Infinite Context Transformer."""

    vocab_size: int = 50257
    hidden_size: int = 768
    n_heads: int = 12
    head_dim: int = 64

    # SSM configuration
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2

    max_chunks: int = 256

    n_boundary_layers: int = 2
    n_chunk_ssm_layers: int = 2
    n_chunk_transformer_layers: int = 8

    mlp_ratio: float = 4.0
    dropout: float = 0.1

    boundary_temperature_init: float = 1.0
    entropy_weight: float = 0.1
    sparsity_weight: float = 0.5

    gradient_checkpointing: bool = False

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.n_heads
        assert self.hidden_size % self.n_heads == 0
