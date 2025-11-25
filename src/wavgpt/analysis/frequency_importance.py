"""Analysis of learned frequency importance in WavGPT models."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from wavgpt.models import HybridWaveletRefinementModel
from .utils import ensure_analysis_dir


def analyze_frequency_importance(
    model: HybridWaveletRefinementModel,
    save_path: str = None,
    output_dir: str = None
):
    """
    Comprehensive visualization of learned frequency priorities.
    
    Args:
        model: Trained WavGPT model
        save_path: Specific path to save plot (optional, overrides output_dir)
        output_dir: Directory to save plots (default: analysis_outputs/)
        
    Returns:
        Dictionary with frequency importance statistics
    """
    # Setup output path
    if save_path is None:
        output_path = ensure_analysis_dir(output_dir)
        save_path = output_path / "frequency_importance.png"
    
    filter_bank = model.frequency_filter
    band_layout = filter_bank.band_layout

    # Get learned importance scores
    band_importance = torch.sigmoid(filter_bank.band_importance).detach().cpu().numpy()
    dim_importance = torch.sigmoid(filter_bank.dim_importance).detach().cpu().numpy()

    n_bands = len(band_layout)
    hidden_size = band_importance.shape[1]

    # === Figure 1: Overall frequency response ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Average importance per frequency band
    avg_importance_per_band = band_importance.mean(axis=1)
    freq_labels = [
        f"{band['kind'][:3]}_L{band['level']}\n(freq_order={band['frequency_order']})"
        for band in band_layout
    ]

    axes[0, 0].bar(range(n_bands), avg_importance_per_band, color='steelblue')
    axes[0, 0].set_xlabel('Frequency Band', fontsize=12)
    axes[0, 0].set_ylabel('Average Importance', fontsize=12)
    axes[0, 0].set_title('Learned Frequency Band Importance', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(n_bands))
    axes[0, 0].set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add frequency order annotation
    for i, band in enumerate(band_layout):
        freq_order = band['frequency_order']
        axes[0, 0].text(
            i, avg_importance_per_band[i] + 0.02,
            f"f={freq_order}",
            ha='center', fontsize=7, color='red'
        )

    # Plot 2: Importance vs Frequency Order (shows low-pass/high-pass/band-pass nature)
    freq_orders = [band['frequency_order'] for band in band_layout]
    axes[0, 1].scatter(freq_orders, avg_importance_per_band, s=100, alpha=0.6, c='steelblue')
    axes[0, 1].plot(freq_orders, avg_importance_per_band, '--', alpha=0.3, c='steelblue')
    axes[0, 1].set_xlabel('Frequency Order (0=lowest, higher=higher freq)', fontsize=12)
    axes[0, 1].set_ylabel('Average Importance', fontsize=12)
    axes[0, 1].set_title('Frequency Response Curve', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Annotate with band names
    for i, (fo, imp) in enumerate(zip(freq_orders, avg_importance_per_band)):
        axes[0, 1].annotate(
            f"B{i}", (fo, imp),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.7
        )

    # Plot 3: Heatmap of importance across dimensions
    im = axes[1, 0].imshow(
        band_importance.T,
        aspect='auto',
        cmap='hot',
        interpolation='nearest'
    )
    axes[1, 0].set_xlabel('Frequency Band', fontsize=12)
    axes[1, 0].set_ylabel('Hidden Dimension', fontsize=12)
    axes[1, 0].set_title('Importance Heatmap (Band × Dimension)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(n_bands))
    axes[1, 0].set_xticklabels([f"B{i}" for i in range(n_bands)], fontsize=8)
    plt.colorbar(im, ax=axes[1, 0], label='Importance')

    # Plot 4: Per-dimension importance modulation
    axes[1, 1].plot(dim_importance, linewidth=1, color='steelblue')
    axes[1, 1].fill_between(range(hidden_size), dim_importance, alpha=0.3)
    axes[1, 1].set_xlabel('Hidden Dimension Index', fontsize=12)
    axes[1, 1].set_ylabel('Dimension Importance', fontsize=12)
    axes[1, 1].set_title('Per-Dimension Importance Modulation', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=dim_importance.mean(), color='red', linestyle='--',
                       label=f'Mean={dim_importance.mean():.3f}')
    axes[1, 1].legend()

    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {save_path}")
    plt.close('all')  # Clean up

    # === Print interpretation ===
    print("\n" + "="*80)
    print("LEARNED FILTER BANK INTERPRETATION")
    print("="*80)

    # Classify filter type
    low_freq_importance = avg_importance_per_band[:3].mean() if n_bands >= 3 else avg_importance_per_band[0]
    high_freq_importance = avg_importance_per_band[-3:].mean() if n_bands >= 3 else avg_importance_per_band[-1]

    print(f"\n📊 Filter Type Classification:")
    print(f"   Low frequency importance:  {low_freq_importance:.4f}")
    print(f"   High frequency importance: {high_freq_importance:.4f}")

    if low_freq_importance > high_freq_importance * 1.5:
        print(f"   → LOW-PASS FILTER: Model prioritizes smooth, global patterns")
    elif high_freq_importance > low_freq_importance * 1.5:
        print(f"   → HIGH-PASS FILTER: Model prioritizes fine-grained details")
    else:
        print(f"   → BAND-PASS FILTER: Model prioritizes mid-range frequencies")

    # Most important bands
    top_k = min(5, n_bands)
    top_bands = np.argsort(avg_importance_per_band)[-top_k:][::-1]

    print(f"\n🎯 Top {top_k} Most Important Frequency Bands:")
    for rank, band_idx in enumerate(top_bands, 1):
        band = band_layout[band_idx]
        imp = avg_importance_per_band[band_idx]
        print(f"   {rank}. Band {band_idx} ({band['kind']}, level {band['level']}, "
              f"freq_order={band['frequency_order']}): importance={imp:.4f}")
        print(f"      → Spans positions {band['start']} to {band['start'] + band['length']}")

    # Dimension analysis
    print(f"\n🔢 Dimension Importance Analysis:")
    print(f"   Mean dimension importance: {dim_importance.mean():.4f}")
    print(f"   Std dimension importance:  {dim_importance.std():.4f}")

    top_dims = np.argsort(dim_importance)[-10:][::-1]
    print(f"   Top 10 most important dimensions: {top_dims.tolist()}")

    bottom_dims = np.argsort(dim_importance)[:10]
    print(f"   Bottom 10 least important dimensions: {bottom_dims.tolist()}")

    print("\n" + "="*80)

    return {
        'band_importance': band_importance,
        'dim_importance': dim_importance,
        'avg_per_band': avg_importance_per_band,
        'freq_orders': freq_orders,
        'top_bands': top_bands
    }

