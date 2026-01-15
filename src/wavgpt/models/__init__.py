"""Context Extension Models.

This package provides models for extending pretrained transformer context
windows via learned chunk boundaries (GRPO) and compression.

Main components:
    - ContextExtender: Wraps pretrained transformer with learned chunking
    - BoundaryPolicy: Learns chunk boundaries via GRPO
    - ChunkCompressor: Compresses token chunks into fixed-size vectors
    - SSMBackbone: State space model for processing sequences
"""

from wavgpt.models.config import ContextExtenderConfig, TrainingConfig
from wavgpt.models.ssm import SelectiveSSM, SSMLayer, SSMBackbone
from wavgpt.models.policy import BoundaryPolicy, BoundaryPolicyWithProjection, BoundarySample
from wavgpt.models.compressor import ChunkCompressor, ChunkInjector
from wavgpt.models.context_extender import ContextExtender, ContextExtenderOutput

__all__ = [
    # Config
    "ContextExtenderConfig",
    "TrainingConfig",
    # SSM
    "SelectiveSSM",
    "SSMLayer",
    "SSMBackbone",
    # Policy
    "BoundaryPolicy",
    "BoundaryPolicyWithProjection",
    "BoundarySample",
    # Compressor
    "ChunkCompressor",
    "ChunkInjector",
    # Main model
    "ContextExtender",
    "ContextExtenderOutput",
]
