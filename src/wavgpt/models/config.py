"""Configuration for Context Extension via Learned Chunking.

This module provides configuration for fine-tuning pretrained transformers
with learned chunk boundaries via GRPO (Group Relative Policy Optimization).

KEY CONSTRAINT: num_chunks + num_unchunked_tokens <= max_context

The policy learns TWO decisions per token:
    1. boundary_prob: Should we end a chunk here?
    2. keep_prob: Should this token be kept at full fidelity (for retrieval)?

UNIFIED ARCHITECTURE: Policy and Compressor share the same SSM backbone,
enabling end-to-end credit assignment via difficulty scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextExtenderConfig:
    """
    Configuration for Context Extension with Unified Policy-Compressor.
    
    Architecture:
        - Pretrained transformer (frozen or fine-tuned with KL penalty)
        - UNIFIED SSM backbone for policy + compression (shared representations)
        - Policy heads: boundary + keep decisions
        - Compression head: chunk embeddings + difficulty scores
    
    KEY INSIGHT: Sharing the SSM backbone enables credit assignment.
    Difficulty scores tell the policy which boundary placements make
    compression hard vs easy, providing direct gradient signal.
    """
    
    # Pretrained model settings
    pretrained_model_name: str = "gpt2"
    hidden_size: int = 768  # Overridden by pretrained model
    freeze_pretrained: bool = False
    kl_penalty_weight: float = 0.1
    
    # Context constraint: chunks + kept tokens must fit
    max_context: int = 1024
    
    @property
    def max_chunks(self) -> int:
        return self.max_context
    
    # Chunk/policy dimension (shared by policy and compression)
    chunk_dim: int = 256
    
    # SSM backbone settings (SHARED between policy and compression)
    ssm_d_state: int = 64
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    n_ssm_layers: int = 4
    
    # Policy settings
    policy_hidden_dim: int = 256
    initial_boundary_bias: float = -2.0  # ~12% boundary probability
    initial_keep_bias: float = -1.0      # ~27% keep probability
    
    # Entropy bonus (prevents policy collapse)
    entropy_bonus_weight: float = 0.05
    
    # Difficulty-based credit assignment
    # Penalizes high difficulty when performance is bad
    difficulty_loss_weight: float = 0.1
    
    # Budget penalty (encourages staying under max_context)
    budget_penalty_weight: float = 0.1
    
    # GRPO settings
    grpo_num_samples: int = 4
    grpo_temperature: float = 1.0
    grpo_kl_coef: float = 0.01
    grpo_clip_range: float = 0.2
    
    # Loss scaling
    policy_loss_scale: float = 1000.0
    
    # Training settings
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_context > 0, "max_context must be positive"
        assert self.grpo_num_samples >= 2, "GRPO needs at least 2 samples for variance"


@dataclass  
class TrainingConfig:
    """Training configuration for GRPO-based context extension."""
    
    # Optimization
    learning_rate: float = 5e-5
    policy_lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)
    
    # Sequence settings
    max_seq_length: int = 4096
    
    # Training duration
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    
    # Mixed precision
    use_amp: bool = True
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
