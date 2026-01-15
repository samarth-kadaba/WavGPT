# Context Extension via GRPO

Extend pretrained transformer context windows through learned chunk boundaries using **Group Relative Policy Optimization (GRPO)**.

## Key Insight

Traditional approaches to context extension (RoPE scaling, sliding window, etc.) have limitations. This approach learns **where to place chunk boundaries** to optimally compress past context, using the language modeling loss as the reward signal.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Pretrained Transformer (frozen)                            │
│  - Processes: [compressed_chunks | current_window]         │
└─────────────────────────────────────────────────────────────┘
                              ↑
              compressed chunks (K vectors)
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  ChunkCompressor (trainable)                                │
│  - SSM compresses each chunk into a fixed-size vector      │
└─────────────────────────────────────────────────────────────┘
                              ↑
           boundaries (discrete, from policy)
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  BoundaryPolicy (trained via GRPO)                          │
│  - Learns where to place chunk boundaries                  │
│  - Reward = negative language modeling loss                │
└─────────────────────────────────────────────────────────────┘
                              ↑
                    past tokens
```

## GRPO Training

**Group Relative Policy Optimization** is a baseline-free policy gradient method:

1. Sample G boundary configurations from the policy
2. For each: compress chunks → run transformer → compute LM loss
3. Rewards = -LM_loss
4. Advantages = (rewards - mean) / std (per-sequence normalization)
5. Update policy: maximize E[advantages × log_prob]

The key insight: by comparing multiple boundary placements on the **same sequence**, we can learn which placements lead to better predictions without needing a value network.

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/WavGPT.git
cd WavGPT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets structlog wandb
pip install mamba-ssm  # Optional: for faster SSM on GPU

# Install package
pip install -e .
```

## Quick Start

### Training

```bash
# Quick test with GPT-2
python scripts/train.py --model gpt2 --debug

# Full training with larger model
python scripts/train.py \
    --model meta-llama/Llama-2-7b-hf \
    --past-length 4096 \
    --current-length 512 \
    --max-chunks 64 \
    --grpo-samples 4 \
    --epochs 3
```

### Evaluation

```bash
# Evaluate checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Quick test with generation
python scripts/evaluate.py --model gpt2 --generate --prompt "The future of AI"
```

### Python API

```python
from wavgpt import ContextExtender, ContextExtenderConfig

# Create model
config = ContextExtenderConfig(
    pretrained_model_name="gpt2",
    max_chunks=64,
    chunk_dim=256,
)
model = ContextExtender.from_pretrained("gpt2", config=config)

# Forward pass with past context
output = model(
    input_ids=current_tokens,      # Current window
    past_token_ids=past_tokens,    # Past context to compress
    labels=current_tokens,
)

print(f"Loss: {output.loss}")
print(f"Boundaries: {output.boundaries}")
```

## Configuration

### Model Config (`ContextExtenderConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_model_name` | "gpt2" | HuggingFace model to extend |
| `max_chunks` | 128 | Maximum compressed chunks |
| `chunk_dim` | 256 | Dimension of compressed chunks |
| `n_ssm_layers` | 4 | SSM backbone layers |
| `grpo_num_samples` | 4 | Boundary configurations per batch |
| `freeze_pretrained` | True | Keep pretrained model frozen |

### Training Config (`TrainingConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate for compressor |
| `policy_lr` | 1e-5 | Learning rate for policy (RL) |
| `batch_size` | 4 | Training batch size |
| `grpo_num_samples` | 4 | GRPO samples per sequence |

## Components

### BoundaryPolicy
- SSM backbone processes input sequence
- Policy head outputs boundary probability at each position
- Trained via GRPO (policy gradient with group-relative advantages)

### ChunkCompressor  
- SSM accumulates information within each chunk
- Outputs fixed-size vector per chunk
- Trained with standard gradients (differentiable)

### ChunkInjector
- Projects compressed chunks to transformer dimension
- Adds positional embeddings
- Creates "virtual tokens" prepended to input

## Mathematical Foundation

### GRPO Objective

For sequence x with G sampled boundary configurations:

```
L_policy = -E_{B~π}[A(B) · log π(B|x)]

where:
  A(B) = (r(B) - μ_r) / σ_r  (group-relative advantage)
  r(B) = -LM_loss(B)          (reward)
```

### Why GRPO over Gumbel-Softmax?

1. **No train/test gap**: Both use discrete boundaries
2. **Better credit assignment**: Compares outcomes directly
3. **Baseline-free**: No value network needed
4. **Explores discrete space**: Samples actual configurations

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_context_extension.py::TestBoundaryPolicy -v
```

## License

MIT

## Citation

```bibtex
@software{context_extension_grpo,
  title = {Context Extension via GRPO},
  year = {2026},
  url = {https://github.com/your-repo/WavGPT}
}
```
