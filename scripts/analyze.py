#!/usr/bin/env python3
"""Analysis script for WavGPT."""

from transformers import AutoTokenizer, BertForMaskedLM

from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.analysis import full_filter_analysis
from wavgpt.config import MODEL_NAME, DEVICE


def main():
    """Main analysis function."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkpoint_path = os.path.join(project_root, "checkpoints", "hybrid_wavelet_model_ratio0.00390625.pt")
    
    # Load model and checkpoint info
    model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
    
    # Load BERT model
    print("\nLoading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    lm_model = BertForMaskedLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    ).to(DEVICE)
    lm_model.eval()
    
    print("Model loaded successfully!")
    print(f"Checkpoint metrics: {info['metrics']}")
    
    # TODO: Load your dataloader here
    # dataloader = ...
    # full_filter_analysis(model, lm_model, dataloader, n_batches=10, device=DEVICE)
    
    print("\nNote: To run full analysis, provide a dataloader to the full_filter_analysis function.")


if __name__ == "__main__":
    main()

