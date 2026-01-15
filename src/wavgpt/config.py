"""Configuration for Infinite Context Transformer."""

import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

VOCAB_SIZE = 50257
HIDDEN_SIZE = 768
N_HEADS = 12

N_BOUNDARY_LAYERS = 2
N_CHUNK_TRANSFORMER_LAYERS = 8

SSM_D_STATE = 16
SSM_D_CONV = 4
SSM_EXPAND = 2

MAX_CHUNKS = 256

DATASET_NAME = "c4"
CONCAT_DOCUMENTS = True
MIN_SEQ_LENGTH = 64

BATCH_SIZE = 2
MAX_LENGTH = 8192
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
DROPOUT = 0.1

VAL_RATIO = 0.05
TEST_RATIO = 0.05

LOG_INTERVAL = 50
SAVE_INTERVAL = 500
VAL_INTERVAL = 200
WANDB_PROJECT = "infinite-context-transformer"

CHECKPOINT_DIR = "checkpoints"
