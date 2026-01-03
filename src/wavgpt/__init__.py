"""Infinite Context Transformer with Learnable Chunking."""

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
