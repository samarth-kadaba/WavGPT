"""Example script to run WavGPT model analysis."""

import sys
import torch
from transformers import BertForMaskedLM, AutoTokenizer

from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.config import DEVICE, MODEL_NAME, BLOCK_SIZE
from wavgpt.data import prepare_dataset, IterableDatasetWrapper, create_collate_fn
from wavgpt.analysis import full_filter_analysis


def main(checkpoint_path: str, output_dir: str = None, n_batches: int = 10):
    """
    Run comprehensive analysis on a trained WavGPT model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory for outputs (default: analysis_outputs/)
        n_batches: Number of batches to analyze
    """
    print("\n" + "="*80)
    print("🔬 WavGPT Model Analysis")
    print("="*80)
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
    print(f"✓ Model loaded from step {info.get('global_step', 'unknown')}")
    
    # Load BERT
    print(f"\nLoading BERT ({MODEL_NAME})...")
    lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lm_model.eval()
    print("✓ BERT loaded")
    
    # Prepare data
    print(f"\nPreparing data...")
    tokenized_dataset, length = prepare_dataset(tokenizer, BLOCK_SIZE)
    dataset = IterableDatasetWrapper(tokenized_dataset, length)
    collate_fn = create_collate_fn(tokenizer, BLOCK_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
    )
    
    # Run analysis
    freq_stats, selection_stats = full_filter_analysis(
        model=model,
        lm_model=lm_model,
        dataloader=dataloader,
        n_batches=n_batches,
        device=DEVICE,
        output_dir=output_dir
    )
    
    return freq_stats, selection_stats


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        checkpoint = '/home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.01_step36500.pt'
    
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(checkpoint, output)

