"""Model definitions for WavGPT."""

import math
import torch
from torch import nn


class LearnedFrequencyFilterBank(nn.Module):
    """
    Learns which frequency components (in wavelet space) are important
    for minimizing KL divergence between original and reconstructed logits.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        levels: int,
        target_sparsity: float = 0.5,
        init_strategy: str = 'dct_like'  # 'dct_like', 'uniform', 'lowpass'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.levels = levels
        self.target_sparsity = target_sparsity

        # Compute coefficient layout
        approx_len = seq_len // (2 ** levels)
        self.band_layout = self._compute_band_layout(seq_len, levels, approx_len)

        # Learn importance scores for each frequency band and dimension
        # Shape: (n_bands, hidden_size) where each band is a different frequency
        n_bands = len(self.band_layout)

        if init_strategy == 'dct_like':
            # Initialize to favor low frequencies (like DCT energy distribution)
            importance_init = self._dct_like_init(n_bands, hidden_size)
        elif init_strategy == 'lowpass':
            # Strong low-pass filter initialization
            importance_init = self._lowpass_init(n_bands, hidden_size)
        else:
            # Uniform initialization
            importance_init = torch.randn(n_bands, hidden_size) * 0.1

        self.band_importance = nn.Parameter(importance_init)

        # Learn per-dimension importance modulation
        self.dim_importance = nn.Parameter(torch.ones(hidden_size))

        # Temperature for soft thresholding (starts high, anneals)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _compute_band_layout(self, seq_len, levels, approx_len):
        """Compute the band structure like in your original code"""
        bands = []
        start = 0

        # Approximation band (lowest frequency)
        bands.append({
            "kind": "approx",
            "level": levels,
            "start": start,
            "length": approx_len,
            "frequency_order": 0  # Lowest frequency
        })
        start += approx_len
        curr_len = approx_len

        # Detail bands (higher frequencies)
        for level in range(levels - 1, -1, -1):
            bands.append({
                "kind": "detail",
                "level": level,
                "start": start,
                "length": curr_len,
                "frequency_order": levels - level  # Higher = higher frequency
            })
            start += curr_len
            curr_len *= 2

        return bands

    def _dct_like_init(self, n_bands, hidden_size):
        """
        Initialize importance scores similar to DCT energy distribution.
        Low frequencies get higher initial importance.
        """
        importance = torch.zeros(n_bands, hidden_size)

        for band_idx, band in enumerate(self.band_layout):
            freq_order = band['frequency_order']
            # Exponential decay with frequency (like natural signals)
            # Lower frequency_order = higher importance
            decay_factor = 0.5 ** (freq_order / 2)
            importance[band_idx, :] = torch.randn(hidden_size) * 0.1 + decay_factor

        return importance

    def _lowpass_init(self, n_bands, hidden_size):
        """Strong low-pass filter: only keep lowest frequencies initially"""
        importance = torch.ones(n_bands, hidden_size) * -2.0  # Low sigmoid value
        # Only approximation band starts with high importance
        importance[0, :] = 2.0  # High sigmoid value
        return importance

    def forward(
        self,
        coeffs: torch.Tensor,
        training: bool = True,
        hard_threshold: bool = False
    ):
        """
        Apply learned frequency filter to wavelet coefficients.

        Args:
            coeffs: (B, T, d) - wavelet coefficients
            training: whether in training mode (affects temperature)
            hard_threshold: if True, use hard thresholding (inference)

        Returns:
            coeffs_filtered: (B, T, d) - filtered coefficients
            mask: (B, T, d) - selection mask
            importance_map: (B, T, d) - importance scores for each coeff
        """
        B, T, d = coeffs.shape
        device = coeffs.device

        # Compute importance score for each coefficient
        importance_map = torch.zeros(B, T, d, device=device)

        for band_idx, band in enumerate(self.band_layout):
            start = band['start']
            length = band['length']
            end = start + length

            # Get learned importance for this band
            band_imp = torch.sigmoid(self.band_importance[band_idx])  # (d,)

            # Modulate by dimension-specific importance
            dim_imp = torch.sigmoid(self.dim_importance)  # (d,)
            combined_imp = band_imp * dim_imp  # (d,)

            # Broadcast to batch and sequence positions in this band
            importance_map[:, start:end, :] = combined_imp.unsqueeze(0).unsqueeze(0)

        # Compute coefficient-specific importance
        # Multiply learned importance by magnitude
        coeff_importance = importance_map * coeffs.abs()

        if training and not hard_threshold:
            # Soft thresholding with temperature (differentiable)
            # Compute global threshold to achieve target sparsity
            k = int(T * d * self.target_sparsity)

            # Get threshold value (k-th largest importance)
            flat_importance = coeff_importance.view(B, -1)
            threshold_vals = torch.kthvalue(
                flat_importance,
                flat_importance.shape[1] - k + 1,
                dim=1,
                keepdim=True
            )[0]  # (B, 1)

            threshold_vals = threshold_vals.view(B, 1, 1)

            # Soft mask using sigmoid with temperature
            soft_mask = torch.sigmoid(
                (coeff_importance - threshold_vals) / self.temperature.abs()
            )

            coeffs_filtered = coeffs * soft_mask
            mask = soft_mask

        else:
            # Hard thresholding (inference or explicit request)
            k = int(T * d * self.target_sparsity)

            masks = []
            for b in range(B):
                flat_imp = coeff_importance[b].view(-1)
                threshold = torch.kthvalue(
                    flat_imp,
                    flat_imp.numel() - k + 1
                )[0]

                hard_mask = (coeff_importance[b] >= threshold).float()
                masks.append(hard_mask)

            mask = torch.stack(masks, dim=0)
            coeffs_filtered = coeffs * mask

        return coeffs_filtered, mask, importance_map

    def get_band_statistics(self):
        """Return learned importance per frequency band (for analysis)"""
        stats = {}
        for band_idx, band in enumerate(self.band_layout):
            band_imp = torch.sigmoid(self.band_importance[band_idx])
            stats[f"band_{band_idx}_{band['kind']}_level_{band['level']}"] = {
                'mean_importance': band_imp.mean().item(),
                'std_importance': band_imp.std().item(),
                'frequency_order': band['frequency_order']
            }
        return stats


class CompressedWaveletEmbedding(nn.Module):
    """
    Wavelet-based compression with embedded-zerotree sparsification.
    Uses learnable lifting scheme (initialized to Haar) instead of fixed wavelets.
    This module only handles compression/decompression - refinement is separate.
    """

    def __init__(self, seq_len: int, hidden_size: int, keep_ratio: float = 0.5, levels: int = None):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        self.keep_ratio = keep_ratio

        # Decide DWT levels
        if levels is None:
            L = 0
            while seq_len % (2 ** (L + 1)) == 0:
                L += 1
            levels = L

        if levels < 1:
            raise ValueError("CompressedWaveletEmbedding requires at least 1 level.")
        if seq_len % (2 ** levels) != 0:
            raise ValueError(
                f"seq_len={seq_len} is not divisible by 2**levels={2**levels}. "
                "Choose a different seq_len or reduce levels."
            )

        self.levels = levels
        self.sqrt2 = math.sqrt(2.0)

        # Learnable lifting parameters (one set per level and per hidden dimension)
        # Initialize to Haar wavelet transform
        # Haar: approx = (even + odd)/sqrt(2), detail = (even - odd)/sqrt(2)
        # In lifting: detail = odd - predict(even), approx = even + update(detail)
        # For Haar: predict = 1.0, update = 0.5, normalize = 1/sqrt(2)

        self.predict_weights = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size) * 1.0) for _ in range(levels)
        ])
        self.update_weights = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size) * 0.5) for _ in range(levels)
        ])
        self.normalize_scale = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size) / self.sqrt2) for _ in range(levels)
        ])
        self.normalize_detail = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size) / self.sqrt2) for _ in range(levels)
        ])

    def _dwt_lifting_1d(self, h: torch.Tensor) -> torch.Tensor:
        """
        Multi-level 1D learnable lifting DWT along time dimension.

        h: (B, T, d)
        returns: coeffs (B, T, d) laid out as [approx_L, detail_{L-1}, ..., detail_0]
        """
        B, T, d = h.shape
        x = h
        coeffs_details = []
        length = T

        for level in range(self.levels):
            if length % 2 != 0:
                raise RuntimeError("Length not divisible by 2 at some DWT level")

            curr = x[:, :length, :]
            even = curr[:, 0::2, :]  # (B, length//2, d)
            odd = curr[:, 1::2, :]   # (B, length//2, d)

            # Lifting scheme:
            # 1. Predict step: detail = odd - predict(even)
            predict_weight = self.predict_weights[level].view(1, 1, d)
            detail = odd - predict_weight * even

            # 2. Update step: approx = even + update(detail)
            update_weight = self.update_weights[level].view(1, 1, d)
            approx = even + update_weight * detail

            # 3. Normalization
            norm_scale = self.normalize_scale[level].view(1, 1, d)
            norm_detail = self.normalize_detail[level].view(1, 1, d)
            approx = approx * norm_scale
            detail = detail * norm_detail

            coeffs_details.append(detail)
            x = x.clone()
            new_len = length // 2
            x[:, :new_len, :] = approx
            length = new_len

        approx_L = x[:, :length, :]
        all_coeffs = [approx_L] + coeffs_details[::-1]
        coeffs = torch.cat(all_coeffs, dim=1)
        return coeffs

    def _idwt_lifting_1d(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Inverse multi-level 1D learnable lifting DWT.

        coeffs: (B, T, d) laid out as [approx_L, detail_{L-1}, ..., detail_0]
        returns: h_hat (B, T, d)
        """
        B, T, d = coeffs.shape
        approx_len = self.seq_len // (2 ** self.levels)

        approx = coeffs[:, :approx_len, :]
        idx = approx_len

        for level in range(self.levels):
            detail = coeffs[:, idx:idx + approx_len, :]
            idx += approx_len

            # Inverse lifting scheme (reverse order of forward):
            # 1. Inverse normalization
            norm_scale = self.normalize_scale[level].view(1, 1, d)
            norm_detail = self.normalize_detail[level].view(1, 1, d)
            approx = approx / norm_scale
            detail = detail / norm_detail

            # 2. Inverse update: even = approx - update(detail)
            update_weight = self.update_weights[level].view(1, 1, d)
            even = approx - update_weight * detail

            # 3. Inverse predict: odd = detail + predict(even)
            predict_weight = self.predict_weights[level].view(1, 1, d)
            odd = detail + predict_weight * even

            # Interleave even and odd
            approx = torch.stack((even, odd), dim=2).reshape(B, approx_len * 2, d)
            approx_len *= 2

        return approx

    def _dwt_haar_1d(self, h: torch.Tensor) -> torch.Tensor:
        """Alias for backward compatibility"""
        return self._dwt_lifting_1d(h)

    def _idwt_haar_1d(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Alias for backward compatibility"""
        return self._idwt_lifting_1d(coeffs)


class HiddenStateRefinementNetwork(nn.Module):
    """
    Refines approximate hidden states after lossy wavelet reconstruction.
    Operates entirely in hidden space, not wavelet space.
    """

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        use_temporal_attention: bool = True,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.use_temporal_attention = use_temporal_attention

        # Default dim_feedforward to 4x hidden_size if not specified
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4

        # Per-token refinement network (MLP)
        self.token_refine = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Temporal refinement to restore coherence across sequence
        if use_temporal_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.temporal_refine = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )

        # Optional: learn a residual scaling factor
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, h_approx: torch.Tensor, mask_kept: torch.Tensor = None) -> torch.Tensor:
        """
        Refine approximate hidden states.

        Args:
            h_approx: (B, T, d) - lossy reconstruction from wavelets
            mask_kept: (B, T, d) - optional mask showing which coefficients were kept

        Returns:
            h_refined: (B, T, d) - refined hidden states
        """
        # Per-token MLP refinement
        residual = self.token_refine(h_approx)  # (B, T, d)
        h_intermediate = h_approx + self.residual_scale * residual

        # Temporal attention refinement
        if self.use_temporal_attention:
            h_refined = self.temporal_refine(h_intermediate)
        else:
            h_refined = h_intermediate

        return h_refined


class HybridWaveletRefinementModel(nn.Module):
    """
    Complete model: learnable wavelet compression with learned frequency filtering
    + hidden state refinement, optimized for KL divergence on logits.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        keep_ratio: float = 0.5,
        wavelet_levels: int = None,
        refine_n_layers: int = 2,
        refine_n_heads: int = 8,
        refine_dim_feedforward: int = None,
        use_temporal_attention: bool = True,
        filter_init_strategy: str = 'dct_like'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.keep_ratio = keep_ratio

        # Determine wavelet levels
        if wavelet_levels is None:
            L = 0
            while seq_len % (2 ** (L + 1)) == 0:
                L += 1
            wavelet_levels = L
        self.wavelet_levels = wavelet_levels

        # Learnable wavelet transform (lifting scheme)
        self.wavelet_module = CompressedWaveletEmbedding(
            seq_len=seq_len,
            hidden_size=hidden_size,
            keep_ratio=keep_ratio,
            levels=wavelet_levels,
        )

        # Learned frequency filter bank (REPLACES zerotrees)
        self.frequency_filter = LearnedFrequencyFilterBank(
            seq_len=seq_len,
            hidden_size=hidden_size,
            levels=wavelet_levels,
            target_sparsity=keep_ratio,
            init_strategy=filter_init_strategy
        )

        # Refinement network
        self.refinement_network = HiddenStateRefinementNetwork(
            hidden_size=hidden_size,
            seq_len=seq_len,
            use_temporal_attention=use_temporal_attention,
            n_layers=refine_n_layers,
            n_heads=refine_n_heads,
            dim_feedforward=refine_dim_feedforward
        )

    def forward(self, h_orig: torch.Tensor, training: bool = True):
        """
        Compress and refine hidden states.

        Args:
            h_orig: (B, T, d) - original hidden states from LM
            training: whether in training mode

        Returns:
            h_refined: (B, T, d) - refined hidden states
            h_approx: (B, T, d) - approximate reconstruction (before refinement)
            coeffs_sparse: (B, T, d) - sparse wavelet coefficients
            mask_kept: (B, T, d) - mask of kept coefficients
            importance_map: (B, T, d) - learned importance scores
        """
        # Forward learnable lifting wavelet transform
        coeffs = self.wavelet_module._dwt_lifting_1d(h_orig)

        # Apply learned frequency filter (REPLACES zerotree_sparsify)
        coeffs_sparse, mask_kept, importance_map = self.frequency_filter(
            coeffs,
            training=training,
            hard_threshold=not training
        )

        # Inverse wavelet transform (lossy reconstruction)
        h_approx = self.wavelet_module._idwt_lifting_1d(coeffs_sparse)

        # Refine in hidden space
        h_refined = self.refinement_network(h_approx, mask_kept)

        return h_refined, h_approx, coeffs_sparse, mask_kept, importance_map

    def get_compression_stats(self, mask_kept: torch.Tensor):
        """Get compression statistics"""
        total_coeffs = mask_kept.numel()
        kept_coeffs = mask_kept.sum().item()
        compression_ratio = total_coeffs / kept_coeffs if kept_coeffs > 0 else float('inf')

        # Also get frequency band statistics
        band_stats = self.frequency_filter.get_band_statistics()

        return {
            'total_coefficients': total_coeffs,
            'kept_coefficients': kept_coeffs,
            'sparsity': 1.0 - (kept_coeffs / total_coeffs),
            'compression_ratio': compression_ratio,
            'target_keep_ratio': self.keep_ratio,
            'band_importance': band_stats
        }

