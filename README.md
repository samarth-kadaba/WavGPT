# WavGPT

Decodable Embeddings via Wavelet Compression and ANN Refinement

## Overview

WavGPT is a research project that explores wavelet-based compression techniques for transformer language models. The project implements a hybrid approach combining learnable wavelet transforms with learned frequency filtering and hidden state refinement networks.

## Features

- **Learnable Wavelet Transforms**: Uses a learnable lifting scheme initialized to Haar wavelets
- **Learned Frequency Filtering**: Automatically learns which frequency components are important
- **Hidden State Refinement**: Refines compressed hidden states using transformer-based refinement networks
- **Comprehensive Analysis Tools**: Includes visualization and analysis utilities for understanding learned frequency patterns

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) installed

### Setup

1. Clone the repository:
```bash
git clone https://github.com/samarth-kadaba/WavGPT.git
cd WavGPT
```

2. Install the package in editable mode with dependencies using uv:
```bash
uv sync
```

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies
- Install the package in editable mode so changes are reflected immediately

3. Activate the virtual environment (if not already activated by uv):
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Training

Train a WavGPT model:

```bash
python scripts/train.py
```

Or use the installed command:
```bash
wavgpt-train
```

### Configuration

Edit `src/wavgpt/config.py` to modify training hyperparameters:

- `MODEL_NAME`: Base language model (default: "bert-large-uncased")
- `BLOCK_SIZE`: Sequence length (must be power of 2, default: 256)
- `BATCH_SIZE`: Training batch size (default: 8)
- `KEEP_RATIO`: Fraction of wavelet coefficients to keep (default: 1/256)
- `LEARNING_RATE`: Learning rate (default: 5e-4)
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `TEMPERATURE`: Temperature for knowledge distillation (default: 2.0)

**Note**: WavGPT uses BERT (bidirectional model) instead of GPT-2 (causal model) to enable true decodable embeddings. With BERT, hidden state `h[i]` represents token `[i]` directly, allowing perfect reconstruction without needing to store the first token separately. 

### Analysis

Run analysis on a trained model:

```bash
python scripts/analyze.py
```

Or use the installed command:
```bash
wavgpt-analyze
```

## Project Structure

```
WavGPT/
├── src/
│   └── wavgpt/
│       ├── __init__.py
│       ├── models.py          # Model definitions
│       ├── data.py             # Data loading utilities
│       ├── training.py         # Training functions
│       ├── analysis.py          # Analysis and visualization
│       └── config.py           # Configuration constants
├── scripts/
│   ├── train.py                # Training script
│   └── analyze.py              # Analysis script
├── tests/                      # Test files (to be added)
├── pyproject.toml              # Project configuration and dependencies
└── README.md                   # This file
```

## Model Architecture

The model consists of three main components:

1. **CompressedWaveletEmbedding**: Performs learnable wavelet transform on hidden states
2. **LearnedFrequencyFilterBank**: Learns which frequency components to keep
3. **HiddenStateRefinementNetwork**: Refines compressed hidden states using transformer layers

### Why BERT Instead of GPT-2?

WavGPT uses BERT (bidirectional) rather than GPT-2 (causal) for a fundamental architectural reason:

- **GPT-2 (Causal)**: Hidden state `h[i]` predicts token `[i+1]` (next token). Cannot reconstruct the current sequence without storing the first token separately.
- **BERT (Bidirectional)**: Hidden state `h[i]` represents token `[i]` directly. Allows perfect reconstruction from compressed hidden states alone.

This makes BERT the natural choice for decodable embeddings where you want to compress text → hidden states → wavelet coefficients and then decompress back to the original text.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wavgpt,
  title={WavGPT: Decodable Embeddings via Wavelet Compression and ANN Refinement},
  author={Your Name},
  year={2024},
  url={https://github.com/samarth-kadaba/WavGPT}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original research on wavelet compression for transformers
- HuggingFace Transformers library
- PyTorch team
