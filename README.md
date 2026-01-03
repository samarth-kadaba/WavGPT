# WavGPT

Infinite Context Transformer with Learnable Chunking

## Overview

WavGPT implements an efficient long-context transformer that learns to segment sequences into semantic chunks, enabling processing of 100K+ token contexts with O(T) + O(K²) complexity, where T is sequence length and K is the maximum number of chunks.

## Features

- **Learnable Boundary Detection**: O(T) learned value function for optimal chunk segmentation
- **SSM-Based Compression**: State Space Models compress tokens within chunks
- **Chunk Transformer**: O(K²) causal attention over chunk embeddings (the only quadratic operation)
- **Efficient Generation**: Amortized boundary predictor for O(1) per-token decisions

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

Train a model:

```bash
python scripts/train.py
```

With custom parameters:

```bash
python scripts/train.py --epochs 5 --batch-size 4 --max-length 16384
python scripts/train.py --hidden-size 768 --n-heads 12 --max-chunks 512
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt
python scripts/evaluate.py --generate --prompt "Your prompt here"
```

### Configuration

Edit `src/wavgpt/config.py` to modify default hyperparameters:

- `HIDDEN_SIZE`: Model hidden dimension (default: 768)
- `N_HEADS`: Number of attention heads (default: 12)
- `MAX_CHUNKS`: Maximum number of chunks K (default: 256)
- `BATCH_SIZE`: Training batch size (default: 2)
- `MAX_LENGTH`: Maximum sequence length (default: 8192)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_EPOCHS`: Number of training epochs (default: 3)

## Project Structure

```
WavGPT/
├── src/
│   └── wavgpt/
│       ├── __init__.py
│       ├── config.py           # Configuration constants
│       ├── models/              # Model components
│       │   ├── core.py          # Main transformer model
│       │   ├── boundary.py      # Boundary detection
│       │   ├── compressor.py    # Chunk compression
│       │   ├── transformer.py   # Chunk transformer
│       │   └── s4.py            # SSM layers
│       ├── data/                # Data loading
│       └── training/            # Training utilities
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── tests/                       # Test files
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Model Architecture

The model consists of four main components:

1. **BoundaryDetector**: Learns to detect semantic chunk boundaries using O(T) learned value function
2. **ChunkCompressor**: Compresses tokens within chunks using SSM layers
3. **ChunkTransformer**: Applies causal attention over chunk embeddings (O(K²))
4. **TokenPredictor**: Combines global (chunk) and local (SSM) context for token prediction

### Complexity

- Boundary detection: O(T) via learned value function
- Chunk compression: O(T) via SSM processing
- Chunk transformer: O(K²) causal attention
- **Total**: O(T) + O(K²) where K ≤ max_chunks

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wavgpt,
  title={WavGPT: Infinite Context Transformer with Learnable Chunking},
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

- HuggingFace Transformers library
- PyTorch team