"""CHUNKY: Contextual Hierarchical Understanding via Neural K-boundarYing.

A differentiable SSM-based compressor for frozen-transformer KV caches.
"""

__version__ = "6.0.0"

from wavgpt.models import (
    CompressorConfig,
    TrainingConfig,
    SelectiveSSM,
    SSMLayer,
    SSMBackbone,
    KVCompressor,
    CompressorOutput,
    KVExtender,
    KVExtenderOutput,
)
from wavgpt.training import CompressorTrainer, create_trainer, split_prefix_continuation

__all__ = [
    "__version__",
    "CompressorConfig",
    "TrainingConfig",
    "SelectiveSSM",
    "SSMLayer",
    "SSMBackbone",
    "KVCompressor",
    "CompressorOutput",
    "KVExtender",
    "KVExtenderOutput",
    "CompressorTrainer",
    "create_trainer",
    "split_prefix_continuation",
]
