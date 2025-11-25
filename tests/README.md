# WavGPT Test Suite

This directory contains comprehensive tests for the WavGPT wavelet-based text compression model.

## Test Files

### Core Functionality Tests

**`test_embeddings_roundtrip.py`** - Embedding & Decoding Tests
- Tests text → sparse coefficients → text roundtrip
- Validates coefficient preservation
- Measures reconstruction quality (positional accuracy, token recall/precision)
- Usage: `python tests/test_embeddings_roundtrip.py [checkpoint_path]`

### Similarity & Semantic Tests

**`test_coeff_similarities.py`** - Comprehensive Similarity Analysis ⭐
- **8 different similarity metrics** tested on 70 text pairs
- Compares BERT baselines vs sparse methods vs decoded methods vs frequency methods
- Statistical analysis with rankings and significance testing
- **Most important test** for understanding semantic preservation
- Usage: `python tests/test_coeff_similarities.py`
- See `SIMILARITY_TEST_README.md` for details

### Model Behavior Tests

**`test_direct_model.py`** - Direct Model Forward Pass
- Tests model forward pass without training loop
- Validates compression ratios and sparsity

**`test_gradient_flow.py`** - Gradient Flow Analysis
- Checks gradient propagation through model
- Validates learnable parameters

**`test_direct_coeffs.py`** - Direct Coefficient Handling
- Tests coefficient extraction and manipulation

**`test_without_compression.py`** - Non-compressed Baseline
- Tests model behavior without sparsification

**`hidden_states.py`** - Hidden State Analysis
- Analyzes hidden state transformations

## Quick Start

### Run the comprehensive similarity test (recommended):
```bash
cd /home/ubuntu/WavGPT
. .venv/bin/activate
python tests/test_coeff_similarities.py
```

### Run embedding roundtrip test:
```bash
python tests/test_embeddings_roundtrip.py [checkpoint_path]
```

## Test Organization

- ✅ **Embeddings functionality** → `embeddings.py` contains ONLY the class, NO test code
- ✅ **All test code** → Located in `tests/` directory
- ✅ **Modular design** → Each test focuses on a specific aspect

## Key Insights from Tests

The similarity tests reveal:
1. Whether sparse coefficients preserve semantic similarity
2. Whether decoded representations recover semantic structure
3. Which similarity metric works best for retrieval
4. Whether frequency-based methods capture semantic patterns

These insights inform whether to add similarity distillation to training.

