"""Configuration for Infinite Context Transformer."""

import torch

# =============================================================================
# Device Configuration
# =============================================================================

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# =============================================================================
# Model Configuration
# =============================================================================

# Model dimensions (GPT-2 Small equivalent ~120M params)
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size
HIDDEN_SIZE = 768  # Model hidden dimension (GPT-2: 768)
N_HEADS = 12  # Number of attention heads (GPT-2: 12, head_dim=64)

# Layer counts (new architecture)
N_BOUNDARY_LAYERS = 2  # Boundary detection SSM layers
N_CHUNK_SSM_LAYERS = 2  # Per-chunk compression SSM layers
N_CHUNK_TRANSFORMER_LAYERS = 8  # Chunk transformer layers (main compute)

# =============================================================================
# SSM Configuration
# =============================================================================

SSM_D_STATE = 16  # State dimension
SSM_D_CONV = 4  # Convolution kernel size
SSM_EXPAND = 2  # Expansion factor

# =============================================================================
# Chunking Configuration
# =============================================================================

MIN_CHUNK_SIZE = 32  # Larger chunks = fewer chunks = MUCH faster training
MAX_CHUNKS = 256  # 8192/32 = 256 max chunks (was 1024, overkill)

# Gumbel-Softmax temperature for boundary classifier
GUMBEL_TEMPERATURE_INIT = 1.0

# =============================================================================
# Dataset Configuration
# =============================================================================

# Available: "c4", "wikitext", "wikipedia", "gutenberg", "code", "arxiv"
DATASET_NAME = "c4"  # C4 - diverse web text (750GB, streaming)
CONCAT_DOCUMENTS = True  # Concatenate docs to fill context
MIN_SEQ_LENGTH = 64  # Minimum sequence length

# =============================================================================
# Training Configuration
# =============================================================================

BATCH_SIZE = 2  # Larger batch = better GPU utilization
MAX_LENGTH = 8192  # Maximum sequence length - can go higher!
LEARNING_RATE = 1e-4  # Lower LR = more stable training
NUM_EPOCHS = 3  # Training epochs
WARMUP_RATIO = 0.1  # Longer warmup = more stable early training
MAX_GRAD_NORM = 1.0  # Gradient clipping
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 * 4 = 16 (same)
DROPOUT = 0.1  # Dropout rate

# =============================================================================
# Data Split Configuration
# =============================================================================

VAL_RATIO = 0.05  # 5% of data for validation
TEST_RATIO = 0.05  # 5% of data for testing

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_INTERVAL = 50  # More frequent logging for faster feedback
SAVE_INTERVAL = 500  # Save more often to avoid losing progress
VAL_INTERVAL = 200  # Validate every N steps
WANDB_PROJECT = "infinite-context-transformer"

# =============================================================================
# Paths
# =============================================================================

CHECKPOINT_DIR = "checkpoints"
