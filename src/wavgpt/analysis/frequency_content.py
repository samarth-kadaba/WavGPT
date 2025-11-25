"""Analysis of frequency content preservation in WavGPT models."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from wavgpt.models import HybridWaveletRefinementModel
from .utils import ensure_analysis_dir


def analyze_frequency_content(
    model: HybridWaveletRefinementModel,
    hidden_states: torch.Tensor,
    token_idx: int = None,
    save_path: str = None,
    output_dir: str = None
):
    """
    Analyze the frequency content of hidden states and what the filter keeps.

    This helps you understand: "What temporal patterns does the model preserve?"
    
    Args:
        model: Trained WavGPT model
        hidden_states: (B, T, d) - sample hidden states
        token_idx: Specific token index to analyze (optional)
        save_path: Specific path to save plot (optional, overrides output_dir)
        output_dir: Directory to save plots (default: analysis_outputs/)
    """
    # Setup output path
    if save_path is None:
        output_path = ensure_analysis_dir(output_dir)
        save_path = output_path / "frequency_content.png"
    
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
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {save_path}")
    plt.close('all')  # Clean up

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

