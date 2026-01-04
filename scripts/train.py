#!/usr/bin/env python3
"""Training script for Infinite Context Transformer with Learnable Chunking."""

import argparse
import os
import sys
from pathlib import Path

import structlog
import torch
from transformers import AutoTokenizer

# Configure torch.compile cache to avoid filling disk
# Use a temp directory that can be cleaned, and disable FX graph cache
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")  # Disable persistent FX cache

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification (ruff: noqa: E402)
from wavgpt.logging_config import configure_logging  # noqa: E402
from wavgpt import InfiniteContextConfig, InfiniteContextTransformer, DEVICE  # noqa: E402
from wavgpt.data import create_dataloader, create_dataloaders  # noqa: E402
from wavgpt.training import train, create_optimizer, create_scheduler, load_checkpoint  # noqa: E402
from wavgpt.config import (  # noqa: E402
    HIDDEN_SIZE,
    N_HEADS,
    N_BOUNDARY_LAYERS,
    N_CHUNK_SSM_LAYERS,
    N_CHUNK_TRANSFORMER_LAYERS,
    MAX_CHUNKS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    LOG_INTERVAL,
    SAVE_INTERVAL,
    VAL_INTERVAL,
    GRADIENT_ACCUMULATION_STEPS,
    CHECKPOINT_DIR,
    WANDB_PROJECT,
    DROPOUT,
    DATASET_NAME,
    MIN_SEQ_LENGTH,
    VAL_RATIO,
    TEST_RATIO,
)

# Configure structlog for console output (after imports)
configure_logging(use_json=False)
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Infinite Context Transformer with Learnable Chunking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH, help="Maximum sequence length"
    )

    # Model parameters
    parser.add_argument(
        "--hidden-size", type=int, default=HIDDEN_SIZE, help="Model hidden dimension"
    )
    parser.add_argument(
        "--n-heads", type=int, default=N_HEADS, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-boundary-layers",
        type=int,
        default=N_BOUNDARY_LAYERS,
        help="Number of boundary detection SSM layers",
    )
    parser.add_argument(
        "--n-chunk-ssm-layers",
        type=int,
        default=N_CHUNK_SSM_LAYERS,
        help="Number of per-chunk compression SSM layers",
    )
    parser.add_argument(
        "--n-chunk-transformer-layers",
        type=int,
        default=N_CHUNK_TRANSFORMER_LAYERS,
        help="Number of chunk transformer layers",
    )

    # Chunking parameters (budget constraint)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=MAX_CHUNKS,
        help="Maximum number of chunks (budget constraint K)",
    )
    # Logging and saving
    parser.add_argument(
        "--log-interval", type=int, default=LOG_INTERVAL, help="Steps between logging"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=VAL_INTERVAL,
        help="Steps between validation runs",
    )
    parser.add_argument(
        "--val-batches", type=int, default=50, help="Number of batches for validation"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-validation", action="store_true", help="Disable validation during training"
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save best model (saves disk space)",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        choices=["c4", "wikitext", "wikipedia", "gutenberg", "code", "arxiv"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--no-concat", action="store_true", help="Disable document concatenation"
    )
    parser.add_argument(
        "--fixed-length",
        action="store_true",
        help="Use fixed sequence length instead of variable",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Fraction of data for testing",
    )

    # Memory optimization
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory, slower training)",
    )

    # Speed optimization
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision (FP16)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (enabled by default for speed)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for parallel data loading",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Log configuration
    logger.info(
        "training_config",
        device=DEVICE,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_boundary_layers=args.n_boundary_layers,
        n_chunk_ssm_layers=args.n_chunk_ssm_layers,
        n_chunk_transformer_layers=args.n_chunk_transformer_layers,
        max_chunks=args.max_chunks,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
        learning_rate=args.lr,
        epochs=args.epochs,
        dataset=args.dataset,
        concat_documents=not args.no_concat,
        variable_length=not args.fixed_length,
        train_ratio=f"{1 - args.val_ratio - args.test_ratio:.0%}",
        val_ratio=f"{args.val_ratio:.0%}",
        test_ratio=f"{args.test_ratio:.0%}",
        validation_enabled=not args.no_validation,
        val_interval=args.val_interval if not args.no_validation else None,
        save_mode="best_only" if args.save_best_only else "all_checkpoints",
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=not args.no_compile,
        num_workers=args.num_workers,
    )

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            config=vars(args),
        )

    # Load tokenizer
    logger.info("loading_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("tokenizer_loaded", vocab_size=tokenizer.vocab_size)

    # Create model config
    config = InfiniteContextConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_boundary_layers=args.n_boundary_layers,
        n_chunk_ssm_layers=args.n_chunk_ssm_layers,
        n_chunk_transformer_layers=args.n_chunk_transformer_layers,
        max_chunks=args.max_chunks,
        dropout=DROPOUT,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Create model
    logger.info("creating_model")
    model = InfiniteContextTransformer(config)

    # Count parameters
    total_params = model.get_num_params()
    logger.info("model_created", parameters=total_params)

    # Prepare dataloaders (train + validation)
    logger.info("preparing_datasets")
    use_validation = not args.no_validation

    if use_validation:
        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_length=args.max_length,
            min_length=MIN_SEQ_LENGTH,
            concat_documents=not args.no_concat,
            variable_length=not args.fixed_length,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            include_test=False,
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        logger.info("dataloaders_created", train=True, validation=True)
    else:
        train_loader = create_dataloader(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_length=args.max_length,
            min_length=MIN_SEQ_LENGTH,
            concat_documents=not args.no_concat,
            variable_length=not args.fixed_length,
            num_workers=args.num_workers,
            split="train",
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        val_loader = None
        logger.info("dataloaders_created", train=True, validation=False)

    # Create optimizer
    optimizer = create_optimizer(model, lr=args.lr)
    logger.info("optimizer_created", optimizer="AdamW", learning_rate=args.lr)

    # Create scheduler
    estimated_steps = 100000 // (args.batch_size * args.grad_accum)
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=estimated_steps,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info("resuming_from_checkpoint", checkpoint=args.resume)
        checkpoint_info = load_checkpoint(model, optimizer, scheduler, args.resume)
        logger.info(
            "checkpoint_loaded",
            epoch=checkpoint_info["epoch"],
            step=checkpoint_info["step"],
        )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Clear CUDA cache before training
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        logger.info("cuda_memory", allocated_gb=torch.cuda.memory_allocated() / 1e9)

    # Train
    logger.info("starting_training")

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=DEVICE,
        scheduler=scheduler,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        gradient_accumulation_steps=args.grad_accum,
        use_wandb=use_wandb,
        save_dir=str(save_dir),
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
        val_loader=val_loader,
        val_batches=args.val_batches,
        save_best_only=args.save_best_only,
    )

    logger.info("training_complete")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
