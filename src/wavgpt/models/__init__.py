"""Model components for Infinite Context Transformer."""

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.s4 import SelectiveSSM, SSMLayer
from wavgpt.models.attention import MultiHeadAttention
from wavgpt.models.boundary import BoundaryDetector
from wavgpt.models.compressor import ChunkCompressor
from wavgpt.models.transformer import TransformerLayer, ChunkTransformer, TokenPredictor
from wavgpt.models.core import InfiniteContextTransformer, create_model

__all__ = [
    "InfiniteContextConfig",
    "SelectiveSSM",
    "SSMLayer",
    "MultiHeadAttention",
    "BoundaryDetector",
    "ChunkCompressor",
    "TransformerLayer",
    "ChunkTransformer",
    "TokenPredictor",
    "InfiniteContextTransformer",
    "create_model",
]
