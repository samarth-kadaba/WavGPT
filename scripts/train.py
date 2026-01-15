#!/usr/bin/env python3
"""
Training script for Context Extension via GRPO.

KEY DESIGN: No distinction between "past" and "current" tokens.
The model learns to dynamically place boundaries anywhere in the sequence.
Tokens after the last boundary become the current window for prediction.

FRONTIER LAB TRAINING:
    - Variable-length sequences sampled from log-uniform distribution
    - Enables robust generalization across sequence lengths
    - Mixed precision training with gradient checkpointing

Usage:
    python scripts/train.py --model gpt2 --min-seq-length 256 --max-seq-length 4096
    python scripts/train.py --model gpt2 --debug  # Quick test
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Disable FX graph cache to avoid disk space issues
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.logging_config import configure_logging  # noqa: E402
from wavgpt.models.config import ContextExtenderConfig, TrainingConfig  # noqa: E402
from wavgpt.models.context_extender import ContextExtender  # noqa: E402
from wavgpt.training.grpo import GRPOTrainer  # noqa: E402

configure_logging(use_json=False)
logger = structlog.get_logger()


def sample_length_log_uniform(min_len: int, max_len: int) -> int:
    """
    Sample a sequence length from a log-uniform distribution.
    
    This gives equal probability to each order of magnitude,
    ensuring the model sees both short and long sequences.
    Used by frontier labs for robust length generalization.
    """
    log_min = math.log(min_len)
    log_max = math.log(max_len)
    log_len = random.uniform(log_min, log_max)
    return int(math.exp(log_len))


class VariableLengthDataset(Dataset):
    """
    Dataset for context extension training with VARIABLE-LENGTH sequences.
    
    Frontier lab training technique:
        - Store long token sequences (up to max_seq_length)
        - At each __getitem__, sample a random length from log-uniform distribution
        - This ensures the model generalizes across all sequence lengths
    
    Each item contains:
        - input_ids: Variable-length sequence tokens
        - labels: Same as input_ids (for next-token prediction)
    """
    
    def __init__(
        self,
        tokenizer,
        texts: list,
        min_seq_length: int = 256,
        max_seq_length: int = 4096,
        stride: int = 512,
    ):
        self.tokenizer = tokenizer
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.stride = stride
        
        # Store long token sequences (at least max_seq_length)
        self.samples = []
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Create sliding windows using max_seq_length
            for start in range(0, len(tokens) - max_seq_length + 1, stride):
                end = start + max_seq_length
                self.samples.append(tokens[start:end])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Sample a random length for this sequence
        seq_length = sample_length_log_uniform(self.min_seq_length, self.max_seq_length)
        
        # Random start position within the stored tokens
        max_start = len(tokens) - seq_length
        if max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = 0
        
        sampled_tokens = tokens[start:start + seq_length]
        
        return {
            "input_ids": torch.tensor(sampled_tokens, dtype=torch.long),
            "labels": torch.tensor(sampled_tokens, dtype=torch.long),
        }


class PG19VariableLengthDataset(Dataset):
    """
    Dataset from PG19 (Project Gutenberg books) with VARIABLE-LENGTH sequences.
    
    Frontier lab training technique:
        - Store long token sequences (up to max_seq_length)
        - At each __getitem__, sample a random length from log-uniform distribution
        - Achieves robust generalization across all sequence lengths
    """
    
    def __init__(
        self,
        tokenizer,
        min_seq_length: int = 256,
        max_seq_length: int = 4096,
        num_samples: int = 10000,
        stride: int = 512,
    ):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.stride = stride
        
        logger.info(
            "loading_pg19_variable_length",
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            num_samples=num_samples,
        )
        
        # Load PG19 with streaming
        ds = load_dataset("emozilla/pg19", split="train", streaming=True)
        
        # Collect samples from books (store at max_seq_length for flexibility)
        self.samples = []
        books_processed = 0
        
        for item in ds:
            text = item["text"]
            if len(text) < 1000:  # Skip very short texts
                continue
                
            # Tokenize the book
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Create sliding windows using max_seq_length
            for start in range(0, len(tokens) - max_seq_length + 1, stride):
                end = start + max_seq_length
                self.samples.append(tokens[start:end])
                
                if len(self.samples) >= num_samples:
                    break
            
            books_processed += 1
            if len(self.samples) >= num_samples:
                break
            
            if books_processed % 10 == 0:
                logger.info("pg19_progress", books=books_processed, samples=len(self.samples))
        
        logger.info("pg19_loaded", books=books_processed, samples=len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Sample a random length for this sequence (LOG-UNIFORM distribution)
        seq_length = sample_length_log_uniform(self.min_seq_length, self.max_seq_length)
        
        # Random start position within the stored tokens
        max_start = len(tokens) - seq_length
        if max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = 0
        
        sampled_tokens = tokens[start:start + seq_length]
        
        return {
            "input_ids": torch.tensor(sampled_tokens, dtype=torch.long),
            "labels": torch.tensor(sampled_tokens, dtype=torch.long),
        }


def variable_length_collate_fn(batch, pad_token_id: int = 0):
    """
    Collate function for variable-length sequences.
    
    Pads sequences to the longest in the batch (not a fixed max).
    This is more memory-efficient than padding to global max.
    
    Args:
        batch: List of dicts with 'input_ids' and 'labels'
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dict with padded tensors and attention_mask
    """
    # Find the max length in this batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 = ignore in loss
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def load_data(tokenizer, args):
    """Load dataset for training with variable-length sequences."""
    
    if args.debug:
        # Small synthetic dataset for debugging
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 200,
            "In machine learning, neural networks learn representations. " * 200,
            "The history of computing spans from ancient times to today. " * 200,
        ]
        
        dataset = VariableLengthDataset(
            tokenizer=tokenizer,
            texts=texts,
            min_seq_length=128,
            max_seq_length=512,  # Small for debug
            stride=64,
        )
    else:
        # Use PG19 dataset with variable-length sampling
        dataset = PG19VariableLengthDataset(
            tokenizer=tokenizer,
            min_seq_length=args.min_seq_length,
            max_seq_length=args.max_seq_length,
            num_samples=args.num_samples,
            stride=args.stride,
        )
    
    if len(dataset) == 0:
        raise ValueError(
            f"No samples created! Try reducing max_seq_length={args.max_seq_length}"
        )
    
    logger.info(
        "dataset_ready",
        num_samples=len(dataset),
        min_seq_length=args.min_seq_length if not args.debug else 128,
        max_seq_length=args.max_seq_length if not args.debug else 512,
        sampling="log-uniform",
    )
    
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Context Extension via GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode with small model and data"
    )

    # Sequence settings - VARIABLE LENGTH (frontier lab training!)
    # MEMORY NOTE: For 15GB GPU (T4), use max-seq-length <= 1024 with batch-size=2
    parser.add_argument(
        "--min-seq-length", type=int, default=128,
        help="Minimum sequence length (sampled from log-uniform distribution)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="Maximum sequence length (log-uniform sampling). Reduce for OOM."
    )
    parser.add_argument(
        "--max-context", type=int, default=512,
        help="Base model's context limit (chunks + kept tokens must fit)"
    )
    parser.add_argument(
        "--chunk-dim", type=int, default=128,
        help="Dimension of compressed chunks (smaller = less memory)"
    )

    # Training
    # MEMORY NOTE: Effective batch = batch_size * grad_accum. Use small batch with accumulation.
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Training batch size (reduce for OOM, compensate with grad-accum)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate for compressor"
    )
    parser.add_argument(
        "--policy-lr", type=float, default=1e-5,
        help="Learning rate for policy (RL is sensitive)"
    )
    parser.add_argument(
        "--grpo-samples", type=int, default=2,
        help="Number of boundary samples for GRPO (2 is minimum, more = more memory)"
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Stride for sliding window"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of training samples to collect"
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Steps between logging"
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging"
    )

    # Hardware
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed precision"
    )

    # Model training mode
    parser.add_argument(
        "--freeze-pretrained", action="store_true",
        help="Freeze pretrained model (only train policy/compressor)"
    )
    parser.add_argument(
        "--kl-weight", type=float, default=0.1,
        help="KL penalty weight (prevents drift from original model)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(
        "training_config",
        model=args.model,
        device=device,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        max_context=args.max_context,
        grpo_samples=args.grpo_samples,
        sampling="log-uniform (frontier lab style)",
        debug=args.debug,
    )

    # Initialize W&B
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project="context-extension", config=vars(args))

    # Load tokenizer
    logger.info("loading_tokenizer", model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create configs
    model_config = ContextExtenderConfig(
        pretrained_model_name=args.model,
        max_context=args.max_context,
        chunk_dim=args.chunk_dim,
        grpo_num_samples=args.grpo_samples,
        freeze_pretrained=args.freeze_pretrained,
        kl_penalty_weight=args.kl_weight if not args.freeze_pretrained else 0.0,
    )
    
    train_config = TrainingConfig(
        learning_rate=args.lr,
        policy_lr=args.policy_lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        log_interval=args.log_interval,
        use_amp=not args.no_amp,
        device=device,
    )

    # Create model
    logger.info("creating_model", pretrained=args.model)
    
    if args.debug:
        # For debug, use smaller settings
        model_config.n_ssm_layers = 2
        model_config.chunk_dim = 128
        model_config.grpo_num_samples = 2
    
    model = ContextExtender.from_pretrained(
        args.model,
        config=model_config,
    )
    
    logger.info(
        "model_created",
        trainable_params=model.get_trainable_params(),
        total_params=model.get_num_params(trainable_only=False),
        )
    
    # Load data
    logger.info("loading_data")
    dataset = load_data(tokenizer, args)
    
    # Create collate function with tokenizer's pad token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = lambda batch: variable_length_collate_fn(batch, pad_token_id=pad_token_id)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_fn,
    )
    
    logger.info("data_loaded", num_samples=len(dataset), variable_length=True)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        train_config=train_config,
        model_config=model_config,
        use_wandb=use_wandb,
        save_dir=args.save_dir,
    )

    # Train
    logger.info("starting_training")
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.epochs,
    )

    logger.info("training_complete")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
