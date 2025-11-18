"""Configuration constants for WavGPT training and inference."""

import torch

# Model configuration
MODEL_NAME = "gpt2-large"  # or "gpt2-medium", "gpt2-large", etc.
BLOCK_SIZE = 256  # sequence length (must be power of 2 for wavelets)
BATCH_SIZE = 8
KEEP_RATIO = 1 / BLOCK_SIZE  # keep ratio of wavelet coefficients
LEARNING_RATE = 5e-4
NUM_EPOCHS = 3
LOG_INTERVAL = 100  # log every N steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "invertible-gpt-refinement"
TEMPERATURE = 2.0  # for knowledge distillation

HIDDEN_SIZE = 1280  # Hidden dimension size
WAVELET_LEVELS = None  # Auto-compute from BLOCK_SIZE (or set to specific int)
REFINE_N_LAYERS = 3  # Number of transformer layers in refinement
REFINE_N_HEADS = 8  # Number of attention heads
REFINE_DIM_FEEDFORWARD = None  # Feedforward dimension (None = 4 * HIDDEN_SIZE)
USE_TEMPORAL_ATTENTION = True  # Whether to use temporal attention

WARMUP_RATIO = 0.05

