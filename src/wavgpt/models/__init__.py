"""Context Extension Models.

This package provides models for extending pretrained transformer context
windows via learned chunk boundaries (GRPO) and compression.

UNIFIED ARCHITECTURE:
    - PolicyCompressor: Single network with shared SSM backbone
      - Policy heads: boundary + keep decisions
      - Compression head: chunk embeddings + difficulty scores
    - ContextExtender: Wraps pretrained transformer with learned chunking
    - ChunkInjector: Projects chunks to virtual tokens
    - SSMBackbone: State space model for processing sequences
"""

from wavgpt.models.config import ContextExtenderConfig, TrainingConfig
from wavgpt.models.ssm import SelectiveSSM, SSMLayer, SSMBackbone
from wavgpt.models.policy import (
    PolicyCompressor,
    PolicyCompressorWithProjection,
    PolicySample,
    PolicyOutput,
)
from wavgpt.models.compressor import ChunkInjector
from wavgpt.models.context_extender import ContextExtender, ContextExtenderOutput

__all__ = [
    # Config
    "ContextExtenderConfig",
    "TrainingConfig",
    # SSM
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
]
