"""Full analysis runner for WavGPT models."""

import torch

from wavgpt.models import HybridWaveletRefinementModel
from .utils import ensure_analysis_dir
from .frequency_importance import analyze_frequency_importance
from .coefficient_selection import analyze_coefficient_selection
from .frequency_content import analyze_frequency_content


def full_filter_analysis(
    model: HybridWaveletRefinementModel,
    lm_model,
    dataloader,
    n_batches: int = 10,
    device=None,
    output_dir: str = None
):
    """
    Run complete analysis suite and save all plots.
    
    Args:
        model: Trained WavGPT model
        lm_model: BERT model for generating hidden states
        dataloader: Data loader for samples
        n_batches: Number of batches to analyze
        device: torch device
        output_dir: Directory to save all plots (default: analysis_outputs/)
        
    Returns:
        Tuple of (freq_stats, selection_stats)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure output directory exists
    output_path = ensure_analysis_dir(output_dir)
    
    print("🔬 STARTING COMPREHENSIVE FILTER BANK ANALYSIS")
    print("="*80)
    print(f"📁 Outputs will be saved to: {output_path}/")
    print("="*80)

    # Collect sample hidden states
    hidden_states_list = []
    print("\n📥 Collecting hidden states from data...")
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            h_orig = outputs.hidden_states[-1]  # (B, T, d)
        hidden_states_list.append(h_orig)
        if (i + 1) % 5 == 0:
            print(f"   Processed {i+1}/{n_batches} batches...")

    hidden_states = torch.cat(hidden_states_list, dim=0)
    print(f"✓ Collected {hidden_states.shape[0]} samples")

    # 1. Learned frequency importance
    print("\n📊 PART 1: Analyzing learned frequency importance...")
    freq_stats = analyze_frequency_importance(
        model, 
        save_path=output_path / "1_frequency_importance.png"
    )

    # 2. Actual coefficient selection patterns
    print("\n📊 PART 2: Analyzing coefficient selection patterns...")
    selection_stats = analyze_coefficient_selection(
        model, 
        hidden_states[:100],
        save_path=output_path / "2_coefficient_selection.png"
    )

    # 3. Frequency domain analysis
    print("\n📊 PART 3: Analyzing frequency content preservation...")
    analyze_frequency_content(
        model, 
        hidden_states[:10],
        save_path=output_path / "3_frequency_content.png"
    )

    print("\n✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All plots saved to: {output_path}/")
    print("  - 1_frequency_importance.png")
    print("  - 2_coefficient_selection.png")
    print("  - 3_frequency_content.png")
    print("="*80)
    
    return freq_stats, selection_stats

