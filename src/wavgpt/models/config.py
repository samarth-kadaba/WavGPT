"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressorConfig:
    pretrained_model_name: str = "gpt2"
    hidden_size: int = 768  # overridden from pretrained config

    # Compressed cache size (K_slots in the math).
    max_kv_slots: int = 128

    # SSM scoring backbone.
    compress_dim: int = 256
    ssm_d_state: int = 64
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    n_ssm_layers: int = 4

    importance_temp_min: float = 0.1
    importance_temp_max: float = 10.0

    # Initialization knobs (default to "useful starting point").
    init_slot_queries_orthogonal: bool = True
    initial_importance_temperature: float = 0.5
    initial_importance_bias: float = 0.1

    # Auxiliary loss weights.
    coverage_loss_weight: float = 0.0   # encourage slot diversity
    sparsity_loss_weight: float = 0.0   # encourage peaked mixing

    dropout: float = 0.1
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.max_kv_slots <= 0:
            raise ValueError("max_kv_slots must be positive")
        if self.compress_dim % 2 != 0:
            raise ValueError("compress_dim must be even for sinusoidal pos enc compatibility")


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)

    # Training sequence is split into prefix (compressed) + continuation (loss target).
    max_seq_length: int = 1024
    min_continuation_length: int = 64
    max_continuation_length: int = 256

    # Gumbel-noise on importance for exploration during training.
    use_gumbel_noise: bool = True

    num_epochs: int = 3
    max_steps: Optional[int] = None

    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100

    use_amp: bool = True
    device: str = "cuda"

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
