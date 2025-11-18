"""WavGPT: Decodable Embeddings via Wavelet Compression and ANN Refinement."""

__version__ = "0.1.0"

from wavgpt.models import (
    HybridWaveletRefinementModel,
    CompressedWaveletEmbedding,
    HiddenStateRefinementNetwork,
    LearnedFrequencyFilterBank,
)

__all__ = [
    "HybridWaveletRefinementModel",
    "CompressedWaveletEmbedding",
    "HiddenStateRefinementNetwork",
    "LearnedFrequencyFilterBank",
]

