"""Configuration for Context Extension via Learned Chunking.

This module provides configuration for fine-tuning pretrained transformers
with learned chunk boundaries via GRPO (Group Relative Policy Optimization).

KEY CONSTRAINT: num_chunks + num_unchunked_tokens <= max_context

The policy learns TWO decisions per token:
    1. boundary_prob: Should we end a chunk here?
    2. keep_prob: Should this token be kept at full fidelity (for retrieval)?

This enables:
    - Compressing less important context into chunks
    - Keeping important tokens (entities, numbers, key facts) verbatim
    - Better retrieval of specific information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextExtenderConfig:
    """
    Configuration for Context Extension with Learned Boundaries and Selective Retention.
    
    Architecture:
        - Pretrained transformer (frozen or fine-tuned with KL penalty)
        - SSM backbone for boundary decisions and compression
        - Policy: learns BOTH where to chunk AND which tokens to keep verbatim
        - Chunk compressor (trained via standard gradients)
    
    KEY CONSTRAINT:
        num_chunks + num_kept_tokens <= max_context
        
        Policy learns:
        - Which segments to compress into chunks (boundary_prob)
        - Which tokens to keep at full fidelity for retrieval (keep_prob)
        
        This enables selective compression where important tokens
        (entities, numbers, key facts) remain accessible at full fidelity.
    """
    
    # Pretrained model settings
    pretrained_model_name: str = "gpt2"
    hidden_size: int = 768  # Will be overridden by pretrained model
    freeze_pretrained: bool = False  # Train full model with KL penalty
    kl_penalty_weight: float = 0.1  # Penalize divergence from original model
    
    # THE ONLY CONSTRAINT: base model's context window
    # Chunks + current window tokens must fit in this
    max_context: int = 1024  # GPT-2 default; Llama-2 = 4096, etc.
    
    # max_chunks is derived from max_context (no separate limit)
    # This is just for tensor allocation - actual chunks are dynamically determined
    @property
    def max_chunks(self) -> int:
        return self.max_context
    
    # Chunk embedding dimension
    chunk_dim: int = 256
    
    # SSM backbone settings (for processing and compression)
    ssm_d_state: int = 64
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    n_ssm_layers: int = 4
    
    # Policy settings
    policy_hidden_dim: int = 256
    initial_boundary_bias: float = 0.0  # Start at 50% for balanced exploration
    initial_keep_bias: float = -1.0  # Start at ~27% for keep decisions
    
    # Entropy bonus (CRITICAL: prevents policy collapse)
    entropy_bonus_weight: float = 0.01  # Encourages exploration, prevents all-0 collapse
    
    # Minimum context usage (prevents degenerate "do nothing" solutions)
    min_context_usage: float = 0.1  # At least 10% of max_context should be used
    min_usage_penalty_weight: float = 1.0  # Penalty for using less than minimum
    
    # Confidence regularization (pushes probabilities toward 0 or 1)
    # WARNING: Can cause collapse if too high - use with entropy bonus
    confidence_loss_weight: float = 0.0  # Disabled by default - can cause collapse
    
    # GRPO settings
    grpo_num_samples: int = 4  # Number of boundary configurations to sample (G)
    grpo_temperature: float = 1.0  # Sampling temperature
    grpo_kl_coef: float = 0.01  # KL penalty coefficient for policy
    grpo_clip_range: float = 0.2  # PPO-style ratio clipping
    
    # Budget constraint (soft penalty - optional, can be 0)
    target_chunks_per_1k_tokens: float = 8.0  # Target ~8 chunks per 1000 tokens
    budget_penalty_weight: float = 0.0  # Set to 0 to disable
    
    # Training settings
    dropout: float = 0.1
    gradient_checkpointing: bool = True  # CRITICAL for memory - enables for SSM AND pretrained model

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_context > 0, "max_context must be positive"
        assert self.grpo_num_samples >= 2, "GRPO needs at least 2 samples for variance"


@dataclass  
class TrainingConfig:
    """Training configuration for GRPO-based context extension."""
    
    # Optimization
    learning_rate: float = 5e-5  # For compressor/injector
    policy_lr: float = 1e-5  # Separate LR for policy (RL is sensitive)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)
    
    # Sequence settings (for data loading)
    max_seq_length: int = 4096  # Maximum sequence length to train on
    
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
