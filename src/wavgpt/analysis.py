"""Analysis and visualization utilities for WavGPT."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from wavgpt.models import HybridWaveletRefinementModel


def analyze_frequency_importance(
    model: HybridWaveletRefinementModel,
    save_path: str = None
):
    """
    Comprehensive visualization of learned frequency priorities.
    """
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

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


def analyze_coefficient_selection(
    model: HybridWaveletRefinementModel,
    hidden_states: torch.Tensor,
    n_samples: int = 100
):
    """
    Analyze which coefficients are actually being selected in practice.

    Args:
        hidden_states: (B, T, d) - sample hidden states from your dataset
        n_samples: number of samples to analyze
    """
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
    plt.show()

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


def analyze_frequency_content(
    model: HybridWaveletRefinementModel,
    hidden_states: torch.Tensor,
    token_idx: int = None
):
    """
    Analyze the frequency content of hidden states and what the filter keeps.

    This helps you understand: "What temporal patterns does the model preserve?"
    """
    model.eval()

    with torch.no_grad():
        coeffs = model.wavelet_module._dwt_lifting_1d(hidden_states)
        coeffs_sparse, mask_kept, _ = model.frequency_filter(
            coeffs,
            training=False,
            hard_threshold=True
        )

        # Reconstruct
        h_reconstructed = model.wavelet_module._idwt_lifting_1d(coeffs_sparse)

    # Pick a specific sample and dimension to analyze
    b_idx = 0
    d_idx = hidden_states.shape[-1] // 2  # Middle dimension

    # Get 1D signals
    original_signal = hidden_states[b_idx, :, d_idx].cpu().numpy()
    reconstructed_signal = h_reconstructed[b_idx, :, d_idx].cpu().numpy()

    # Compute FFT for frequency analysis
    fft_original = np.fft.fft(original_signal)
    fft_reconstructed = np.fft.fft(reconstructed_signal)

    freqs = np.fft.fftfreq(len(original_signal))

    # Power spectral density
    psd_original = np.abs(fft_original) ** 2
    psd_reconstructed = np.abs(fft_reconstructed) ** 2

    # === Visualization ===
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Plot 1: Time domain - original signal
    axes[0, 0].plot(original_signal, linewidth=1, label='Original', alpha=0.8)
    axes[0, 0].set_xlabel('Token Position', fontsize=11)
    axes[0, 0].set_ylabel('Activation Value', fontsize=11)
    axes[0, 0].set_title(f'Original Hidden State (dim {d_idx})', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Time domain - comparison
    axes[0, 1].plot(original_signal, linewidth=1, label='Original', alpha=0.7)
    axes[0, 1].plot(reconstructed_signal, linewidth=1, label='Reconstructed', alpha=0.7, linestyle='--')
    axes[0, 1].set_xlabel('Token Position', fontsize=11)
    axes[0, 1].set_ylabel('Activation Value', fontsize=11)
    axes[0, 1].set_title('Original vs Reconstructed', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Compute error
    error = np.abs(original_signal - reconstructed_signal)
    mse = np.mean(error ** 2)
    axes[0, 1].text(
        0.02, 0.98, f'MSE: {mse:.6f}',
        transform=axes[0, 1].transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Plot 3: Frequency domain - power spectrum
    # Only plot positive frequencies
    pos_mask = freqs >= 0
    axes[1, 0].semilogy(freqs[pos_mask], psd_original[pos_mask],
                        linewidth=1.5, label='Original', alpha=0.8)
    axes[1, 0].semilogy(freqs[pos_mask], psd_reconstructed[pos_mask],
                        linewidth=1.5, label='Reconstructed', alpha=0.8, linestyle='--')
    axes[1, 0].set_xlabel('Normalized Frequency', fontsize=11)
    axes[1, 0].set_ylabel('Power Spectral Density (log)', fontsize=11)
    axes[1, 0].set_title('Frequency Content (Power Spectrum)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Frequency preservation ratio
    preservation_ratio = psd_reconstructed / (psd_original + 1e-10)
    axes[1, 1].plot(freqs[pos_mask], preservation_ratio[pos_mask], linewidth=1.5, color='purple')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect preservation')
    axes[1, 1].set_xlabel('Normalized Frequency', fontsize=11)
    axes[1, 1].set_ylabel('Reconstruction / Original Power', fontsize=11)
    axes[1, 1].set_title('Frequency Preservation Ratio', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.5])

    # Plot 5: Wavelet coefficients - original
    coeffs_np = coeffs[b_idx, :, d_idx].cpu().numpy()
    axes[2, 0].stem(coeffs_np, linefmt='steelblue', markerfmt='o', basefmt=' ')
    axes[2, 0].set_xlabel('Coefficient Index', fontsize=11)
    axes[2, 0].set_ylabel('Coefficient Value', fontsize=11)
    axes[2, 0].set_title('Original Wavelet Coefficients', fontsize=13, fontweight='bold')

    # Add band boundaries
    for band in model.frequency_filter.band_layout:
        axes[2, 0].axvline(x=band['start'], color='red', linestyle='--', alpha=0.5)
        axes[2, 0].text(
            band['start'] + band['length']//2,
            coeffs_np.max() * 0.8,
            f"{band['kind'][:1]}{band['level']}",
            ha='center', fontsize=8
        )
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Wavelet coefficients - after sparsification
    coeffs_sparse_np = coeffs_sparse[b_idx, :, d_idx].cpu().numpy()
    mask_np = mask_kept[b_idx, :, d_idx].cpu().numpy()

    # Convert to boolean explicitly
    mask_bool = mask_np.astype(bool)

    # Color code: green for kept, red for dropped
    axes[2, 1].stem(coeffs_sparse_np, linefmt='green', markerfmt='o', basefmt=' ')

    # Overlay dropped locations
    dropped_indices = np.where(~mask_bool)[0]
    if len(dropped_indices) > 0:
        axes[2, 1].scatter(
            dropped_indices,
            np.zeros_like(dropped_indices),
            color='red', marker='x', s=20, alpha=0.5, label='Dropped'
        )

    axes[2, 1].set_xlabel('Coefficient Index', fontsize=11)
    axes[2, 1].set_ylabel('Coefficient Value', fontsize=11)
    axes[2, 1].set_title('Sparse Wavelet Coefficients (Green=Kept)', fontsize=13, fontweight='bold')
    axes[2, 1].legend()

    # Add band boundaries
    for band in model.frequency_filter.band_layout:
        axes[2, 1].axvline(x=band['start'], color='red', linestyle='--', alpha=0.5)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # === Frequency band analysis ===
    print("\n" + "="*80)
    print(f"FREQUENCY CONTENT ANALYSIS (Sample {b_idx}, Dimension {d_idx})")
    print("="*80)

    # Compute energy per frequency band
    for band in model.frequency_filter.band_layout:
        start = band['start']
        end = start + band['length']

        # Energy in this band (original)
        band_coeffs_orig = coeffs_np[start:end]
        energy_orig = np.sum(band_coeffs_orig ** 2)

        # Energy in this band (reconstructed)
        band_coeffs_sparse = coeffs_sparse_np[start:end]
        energy_reconstructed = np.sum(band_coeffs_sparse ** 2)

        # Preservation ratio
        preservation = energy_reconstructed / (energy_orig + 1e-10)

        # Number of kept coefficients
        band_mask = mask_bool[start:end]
        n_kept = band_mask.sum()
        n_total = len(band_mask)

        print(f"\n📊 {band['kind']} (level {band['level']}, freq_order={band['frequency_order']})")
        print(f"   Original energy:      {energy_orig:.4f}")
        print(f"   Reconstructed energy: {energy_reconstructed:.4f}")
        print(f"   Energy preservation:  {preservation:.2%}")
        print(f"   Coefficients kept:    {n_kept}/{n_total} ({n_kept/n_total:.2%})")

    print("\n" + "="*80)


def full_filter_analysis(
    model: HybridWaveletRefinementModel,
    lm_model,
    dataloader,
    n_batches: int = 10,
    device=None
):
    """Run complete analysis suite"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔬 STARTING COMPREHENSIVE FILTER BANK ANALYSIS")
    print("="*80)

    # Collect sample hidden states
    hidden_states_list = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        # Assuming batch contains hidden states or you can extract them
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

    hidden_states = torch.cat(hidden_states_list, dim=0)

    # 1. Learned frequency importance
    print("\n📊 PART 1: Analyzing learned frequency importance...")
    freq_stats = analyze_frequency_importance(model, save_path='frequency_importance.png')

    # 2. Actual coefficient selection patterns
    print("\n📊 PART 2: Analyzing coefficient selection patterns...")
    selection_stats = analyze_coefficient_selection(model, hidden_states[:100])

    # 3. Frequency domain analysis
    print("\n📊 PART 3: Analyzing frequency content preservation...")
    analyze_frequency_content(model, hidden_states[:10])

    print("\n✅ ANALYSIS COMPLETE!")
    print("="*80)

