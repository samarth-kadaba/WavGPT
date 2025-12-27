"""
Infinite Context Transformer with SSM-Guided Chunking.

This package provides efficient long-context modeling through:
1. Boundary SSM: Global pass to detect semantic chunk boundaries
2. Chunk SSM: Fresh per-chunk compression (no cross-contamination)
3. Chunk Transformer: O(chunks²) causal attention over chunk embeddings
4. Token Predictor: Combines global (chunks) + local (within-chunk) context

Key insight: By chunking at semantic boundaries and applying
attention over chunks (not tokens), we achieve 100K+ token
context with O(T) + O(chunks²) complexity.

Each chunk is compressed INDEPENDENTLY, maintaining maximum
resolution even for chunks late in the sequence.
"""

__version__ = "2.0.0"

from wavgpt.models import (
    # Configuration
    InfiniteContextConfig,
    # Core components
    SelectiveSSM,
    SSMLayer,
    BoundaryDetector,
    ChunkCompressor,
    ChunkTransformer,
    TokenPredictor,
    # Generation
    GenerationState,
    # Main model
    InfiniteContextTransformer,
    # Utilities
    create_model,
)

from wavgpt.config import DEVICE, VAL_RATIO, TEST_RATIO, VAL_INTERVAL

from wavgpt.data import create_dataloader, create_dataloaders

from wavgpt.training import train, validate, create_optimizer, create_scheduler

__all__ = [
    # Configuration
    "InfiniteContextConfig",
    # Core components
    "SelectiveSSM",
    "SSMLayer",
    "BoundaryDetector",
    "ChunkCompressor",
    "ChunkTransformer",
    "TokenPredictor",
    # Generation
    "GenerationState",
    # Main model
    "InfiniteContextTransformer",
    # Data loading
    "create_dataloader",
    "create_dataloaders",
    # Training
    "train",
    "validate",
    "create_optimizer",
    "create_scheduler",
    # Config
    "create_model",
    "DEVICE",
    "VAL_RATIO",
    "TEST_RATIO",
    "VAL_INTERVAL",
]
