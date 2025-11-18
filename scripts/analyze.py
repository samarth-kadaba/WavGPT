#!/usr/bin/env python3
"""Analysis script for WavGPT."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.analysis import full_filter_analysis
from wavgpt.config import MODEL_NAME, DEVICE


def main():
    """Main analysis function."""
    # Load model checkpoint
    checkpoint_path = "hybrid_wavelet_model_ratio0.00390625.pt"  # Update with your checkpoint path
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']
    
    # Load tokenizer and base model
    print("\nLoading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    ).to(DEVICE)
    lm_model.eval()
    
    # Initialize WavGPT model
    print("\nInitializing WavGPT model...")
    model = HybridWaveletRefinementModel(
        seq_len=config['seq_len'],
        hidden_size=config['hidden_size'],
        keep_ratio=config['keep_ratio'],
    ).to(DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # TODO: Load your dataloader here
    # For now, this is a placeholder - you'll need to provide a dataloader
    # dataloader = ...
    # full_filter_analysis(model, lm_model, dataloader, n_batches=10, device=DEVICE)
    
    print("\nNote: To run full analysis, provide a dataloader to the full_filter_analysis function.")


if __name__ == "__main__":
    main()

