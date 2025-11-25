"""Configuration constants for WavGPT training and inference."""

import torch

# Model configuration
MODEL_NAME = "bert-base-uncased"  # BERT for bidirectional embeddings (bert-base: 768, bert-large: 1024)
BLOCK_SIZE = 256  # sequence length (must be power of 2 for wavelets)
BATCH_SIZE = 8
KEEP_RATIO = 0.01  # keep ratio of wavelet coefficients
LEARNING_RATE = 5e-4
NUM_EPOCHS = 3
LOG_INTERVAL = 100  # log every N steps
# set DEVICE, cuda for GPU, mps for M1/M2 Macs, cpu for CPU
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
WANDB_PROJECT = "invertible-bert-refinement"
TEMPERATURE = 2.0  # for knowledge distillation

HIDDEN_SIZE = 768  # Hidden dimension size (bert-base: 768, bert-large: 1024)
WAVELET_LEVELS = None  # Auto-compute from BLOCK_SIZE (or set to specific int)
REFINE_N_LAYERS = 3  # Number of transformer layers in refinement
REFINE_N_HEADS = 8  # Number of attention heads
REFINE_DIM_FEEDFORWARD = None  # Feedforward dimension (None = 4 * HIDDEN_SIZE)
USE_TEMPORAL_ATTENTION = True  # Whether to use temporal attention

WARMUP_RATIO = 0.05

# Checkpoint resumption
CHECKPOINT_PATH = None # Set to checkpoint path to resume training
# NOTE: /home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.00390625_step9000.pt is CORRUPTED

