"""Analysis of coefficient selection patterns in WavGPT models."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from wavgpt.models import HybridWaveletRefinementModel
from .utils import ensure_analysis_dir


def analyze_coefficient_selection(
    model: HybridWaveletRefinementModel,
    hidden_states: torch.Tensor,
    n_samples: int = 100,
    save_path: str = None,
    output_dir: str = None
):
    """
    Analyze which coefficients are actually being selected in practice.

    Args:
        model: Trained WavGPT model
        hidden_states: (B, T, d) - sample hidden states from your dataset
        n_samples: number of samples to analyze
        save_path: Specific path to save plot (optional, overrides output_dir)
        output_dir: Directory to save plots (default: analysis_outputs/)
        
    Returns:
        Dictionary with selection statistics
    """
    # Setup output path
    if save_path is None:
        output_path = ensure_analysis_dir(output_dir)
        save_path = output_path / "coefficient_selection.png"
    
    model.eval()

    # Run through model
    with torch.no_grad():
        coeffs = model.wavelet_module._dwt_lifting_1d(hidden_states)
        coeffs_sparse, mask_kept, importance_map = model.frequency_filter(
            coeffs,
            training=False,
            hard_threshold=True
        )

    # Analyze selection patterns
    mask_np = mask_kept.cpu().numpy()  # (B, T, d)
    coeffs_np = coeffs.cpu().numpy()
    importance_np = importance_map.cpu().numpy()

    B, T, d = mask_np.shape

    # === Figure: Selection patterns ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Selection rate per position (averaged across dimensions)
    selection_per_pos = mask_np.mean(axis=(0, 2))  # (T,)

    axes[0, 0].bar(range(T), selection_per_pos, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Sequence Position', fontsize=12)
    axes[0, 0].set_ylabel('Selection Rate', fontsize=12)
    axes[0, 0].set_title('Coefficient Selection Rate per Position', fontsize=14, fontweight='bold')

    # Add band boundaries
    for band in model.frequency_filter.band_layout:
        start = band['start']
        axes[0, 0].axvline(x=start, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[0, 0].text(
            start + band['length']//2,
            selection_per_pos.max() * 0.9,
            f"{band['kind'][:3]}\nL{band['level']}",
            ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    axes[0, 0].grid(axis='y', alpha=0.3)

    # Plot 2: Selection rate per dimension (averaged across positions)
    selection_per_dim = mask_np.mean(axis=(0, 1))  # (d,)

    axes[0, 1].plot(selection_per_dim, linewidth=1, color='steelblue')
    axes[0, 1].fill_between(range(d), selection_per_dim, alpha=0.3)
    axes[0, 1].set_xlabel('Hidden Dimension', fontsize=12)
    axes[0, 1].set_ylabel('Selection Rate', fontsize=12)
    axes[0, 1].set_title('Coefficient Selection Rate per Dimension', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(
        y=selection_per_dim.mean(),
        color='red', linestyle='--',
        label=f'Mean={selection_per_dim.mean():.3f}'
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: 2D heatmap of selection patterns (position × dimension)
    # Average across batch
    selection_2d = mask_np.mean(axis=0)  # (T, d)

    im = axes[1, 0].imshow(
        selection_2d.T,
        aspect='auto',
        cmap='hot',
        interpolation='nearest'
    )
    axes[1, 0].set_xlabel('Sequence Position', fontsize=12)
    axes[1, 0].set_ylabel('Hidden Dimension', fontsize=12)
    axes[1, 0].set_title('2D Selection Pattern (Position × Dimension)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], label='Selection Rate')

    # Add band boundaries
    for band in model.frequency_filter.band_layout:
        axes[1, 0].axvline(x=band['start'], color='cyan', linestyle='--', alpha=0.7, linewidth=1)

    # Plot 4: Histogram of importance scores for kept vs dropped coefficients
    kept_importance = importance_np[mask_np.astype(bool)]
    dropped_importance = importance_np[~mask_np.astype(bool)]

    axes[1, 1].hist(kept_importance, bins=50, alpha=0.6, label='Kept', color='green', density=True)
    axes[1, 1].hist(dropped_importance, bins=50, alpha=0.6, label='Dropped', color='red', density=True)
    axes[1, 1].set_xlabel('Importance Score', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Importance Distribution: Kept vs Dropped', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {save_path}")
    plt.close('all')  # Clean up

    # === Per-band statistics ===
    print("\n" + "="*80)
    print("COEFFICIENT SELECTION STATISTICS PER FREQUENCY BAND")
    print("="*80)

    for band_idx, band in enumerate(model.frequency_filter.band_layout):
        start = band['start']
        end = start + band['length']

        # Get selection rate for this band
        band_mask = mask_np[:, start:end, :]
        selection_rate = band_mask.mean()

        # Get average importance for this band
        band_importance = importance_np[:, start:end, :]
        avg_importance = band_importance.mean()

        # Get average coefficient magnitude in this band
        band_coeffs = np.abs(coeffs_np[:, start:end, :])
        avg_magnitude = band_coeffs.mean()

        print(f"\n📍 Band {band_idx}: {band['kind']} at level {band['level']}")
        print(f"   Frequency order: {band['frequency_order']} (0=lowest freq)")
        print(f"   Position range: [{start}, {end})")
        print(f"   Selection rate: {selection_rate:.4f} ({selection_rate*100:.2f}%)")
        print(f"   Avg importance: {avg_importance:.4f}")
        print(f"   Avg magnitude:  {avg_magnitude:.4f}")
        print(f"   Kept coeffs:    {int(band_mask.sum())} / {band_mask.size}")

    print("\n" + "="*80)

    return {
        'selection_per_pos': selection_per_pos,
        'selection_per_dim': selection_per_dim,
        'selection_2d': selection_2d,
        'kept_importance': kept_importance,
        'dropped_importance': dropped_importance
    }

