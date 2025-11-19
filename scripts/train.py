#!/usr/bin/env python3
"""Training script for WavGPT."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM
import wandb

from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.data import prepare_dataset, IterableDatasetWrapper
from wavgpt.training import train_model, create_scheduler
from wavgpt.config import (
    MODEL_NAME,
    BLOCK_SIZE,
    BATCH_SIZE,
    KEEP_RATIO,
    LEARNING_RATE,
    NUM_EPOCHS,
    LOG_INTERVAL,
    DEVICE,
    WANDB_PROJECT,
    TEMPERATURE,
    HIDDEN_SIZE,
    WAVELET_LEVELS,
    REFINE_N_LAYERS,
    REFINE_N_HEADS,
    REFINE_DIM_FEEDFORWARD,
    USE_TEMPORAL_ATTENTION,
    WARMUP_RATIO,
)


def main():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    print(f"Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Keep ratio: {KEEP_RATIO}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Temperature: {TEMPERATURE}")

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        config={
            'model_name': MODEL_NAME,
            'block_size': BLOCK_SIZE,
            'batch_size': BATCH_SIZE,
            'keep_ratio': KEEP_RATIO,
            'learning_rate': LEARNING_RATE,
            'temperature': TEMPERATURE,
            'num_epochs': NUM_EPOCHS,
        }
    )

    # Load tokenizer and BERT model
    print("\nLoading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # BERT has native pad token support
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token

    lm_model = BertForMaskedLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    ).to(DEVICE)
    lm_model.eval()
    for p in lm_model.parameters():
        p.requires_grad = False  # Freeze the BERT model

    hidden_size = lm_model.config.hidden_size
    print(f"Loaded model: {MODEL_NAME}")
    print(f"Hidden size: {hidden_size}")
    print(f"Note: Using BERT (bidirectional) for natural h[i]→token[i] alignment")

    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset, num_rows = prepare_dataset(tokenizer, BLOCK_SIZE)
    train_dataset_wrapped = IterableDatasetWrapper(train_dataset, num_rows)
    train_loader = DataLoader(train_dataset_wrapped, batch_size=BATCH_SIZE)

    # Create our hybrid model
    print("\nInitializing hybrid wavelet refinement model...")
    # Initialize model
    model = HybridWaveletRefinementModel(
        seq_len=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        keep_ratio=KEEP_RATIO,
        wavelet_levels=WAVELET_LEVELS,
        refine_n_layers=REFINE_N_LAYERS,
        refine_n_heads=REFINE_N_HEADS,
        refine_dim_feedforward=REFINE_DIM_FEEDFORWARD,
        use_temporal_attention=USE_TEMPORAL_ATTENTION,
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = create_scheduler(optimizer, NUM_EPOCHS, train_loader, warmup_ratio=WARMUP_RATIO)

    # Train
    print("\nStarting training...")
    train_model(
        model=model,
        lm_model=lm_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        scheduler=scheduler,
        log_interval=LOG_INTERVAL,
        temperature=TEMPERATURE,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save model
    save_path = f"hybrid_wavelet_model_ratio{KEEP_RATIO}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'seq_len': BLOCK_SIZE,
            'hidden_size': hidden_size,
            'keep_ratio': KEEP_RATIO,
        }
    }, save_path)
    print(f"Model saved to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

