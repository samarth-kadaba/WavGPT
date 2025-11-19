"""Load a model from a checkpoint."""

import torch
from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.config import DEVICE
from wavgpt.config import BLOCK_SIZE, HIDDEN_SIZE, KEEP_RATIO, REFINE_N_LAYERS, REFINE_N_HEADS, USE_TEMPORAL_ATTENTION, WAVELET_LEVELS, REFINE_DIM_FEEDFORWARD


def load_model_from_checkpoint(pt_path: str, device=DEVICE):
    """Load model from checkpoint, handling architecture mismatches."""
    checkpoint = torch.load(pt_path, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    device = torch.device(device)
    # Create model
    model = HybridWaveletRefinementModel(
        seq_len=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        keep_ratio=KEEP_RATIO,
        wavelet_levels=WAVELET_LEVELS,
        refine_n_layers=REFINE_N_LAYERS,
        refine_n_heads=REFINE_N_HEADS,
        refine_dim_feedforward=REFINE_DIM_FEEDFORWARD,
        use_temporal_attention=USE_TEMPORAL_ATTENTION,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


if __name__ == "__main__":
    model, config = load_model_from_checkpoint("/Users/samkadaba/Desktop/WavGPT/hybrid_wavelet_model_ratio0.00390625.pt")
    print(model)