"""Context Extension via Learned Chunking with GRPO.

This package extends pretrained transformer context windows by learning
chunk boundaries via Group Relative Policy Optimization (GRPO).

UNIFIED ARCHITECTURE:
    - PolicyCompressor: Single network with shared SSM backbone
      - Policy heads: boundary + keep decisions
      - Compression head: chunk embeddings + difficulty scores
    - ContextExtender: Main model wrapping pretrained transformer
    - GRPOTrainer: Training loop for GRPO-based learning
"""

__version__ = "5.0.0"

import torch

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from wavgpt.models import (
    # Configuration
    ContextExtenderConfig,
    TrainingConfig,
    # SSM components
    SelectiveSSM,
    SSMLayer,
    SSMBackbone,
    # Policy-Compressor (unified)
    PolicyCompressor,
    PolicyCompressorWithProjection,
    PolicySample,
    PolicyOutput,
    # Injector
    ChunkInjector,
    # Main model
    ContextExtender,
    ContextExtenderOutput,
)

from wavgpt.training import (
    GRPOTrainer,
    create_grpo_trainer,
)

__all__ = [
    # Version
    "__version__",
    # Device
    "DEVICE",
    # Configuration
    "ContextExtenderConfig",
    "TrainingConfig",
    # SSM components
    "SelectiveSSM",
    "SSMLayer",
    "SSMBackbone",
    # Policy-Compressor (unified)
    "PolicyCompressor",
    "PolicyCompressorWithProjection",
    "PolicySample",
    "PolicyOutput",
    # Injector
    "ChunkInjector",
    # Main model
    "ContextExtender",
    "ContextExtenderOutput",
    # Training
    "GRPOTrainer",
    "create_grpo_trainer",
]
