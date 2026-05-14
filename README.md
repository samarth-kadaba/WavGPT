# CHUNKY

**C**ontextual **H**ierarchical **U**nderstanding via **N**eural **K**-boundar**Y**ing.

A small SSM-based compressor that learns to shrink a frozen transformer's
KV cache to a fixed budget by (1) scoring per-position importance and
(2) mixing entries into `K` learned anchor slots, trained end-to-end on
LM cross-entropy of a held-out continuation.

## Hypothesis

A frozen language model's KV cache contains substantial redundancy. A
small auxiliary network can learn — purely from next-token loss on a
withheld continuation — to compress the cache to a fixed budget `K` while
preserving prediction quality, and the same compressor can be applied
recursively without catastrophic loss.

## Method

For a prefix of `T` tokens, run the frozen LM once with `use_cache=True`
to get per-layer `(K_l, V_l)` and the last-layer hidden states
`h ∈ ℝ^{T × d}`.

1. **Importance.** An SSM processes `h` (linear time in `T`) and a linear
   head produces a per-position score `s_i ∈ ℝ`.
2. **Slot queries.** `K` learnable queries `q_k ∈ ℝ^{d'}` represent the
   compressed cache slots.
3. **Mixing.** Cross-attention with importance bias gives mixing weights
   ```
   logits[k, j]  =  (q_k · proj(h_j)) / √d  +  s_j / τ
   W[k, j]       =  softmax_j(logits[k, j])
   ```
4. **Apply.** The same `W` is applied to every layer:
   `K'_l = W K_l`, `V'_l = W V_l`.
   The compressed cache has shape `(B, n_heads, K, head_dim)` per layer.
5. **Decode.** Run the frozen LM on a continuation of `M` tokens with
   `past_key_values = (K', V')`. Cross-entropy on the continuation tokens
   is the training loss.

Optional Gumbel noise on `s_i` adds exploration during training; the
softmax mixing is smooth in `s` so the gradient is well-defined. At
inference, Gumbel is off.

Everything backpropagates through the continuation forward, the
cross-attention, and the SSM in a single graph. The LM's parameters
have `requires_grad=False` so only the compressor is updated; gradient
still flows through the LM's operations to reach the compressor inputs.

## Why this framing

- The compressor operates on the **actual KV cache**, the object that
  long-context inference is bottlenecked on — not on a re-embedded
  approximation that requires a second LM forward pass to evaluate.
- The mixing weights are **distributional over the entire prefix**, so
  the compressor can express "merge these three entries into one anchor"
  rather than only "drop entry j" as heuristic eviction methods do.
- The compressed cache is **still a KV cache**, so the same compressor
  composes with itself. Compression depth becomes a controllable
  experimental axis.
- The training signal is **differentiable**, so each step costs one
  forward + one backward. Cheap enough to sweep model scales.

## Repository layout

```
src/wavgpt/
  models/
    config.py            CompressorConfig, TrainingConfig
    ssm.py               SelectiveSSM (Mamba kernel when available) + SSMBackbone
    kv_compressor.py     Importance scoring + slot queries + cross-attention mixing
    kv_extender.py       Wraps an HF causal LM and its past_key_values
  training/
    trainer.py           Supervised LM-loss trainer (single rollout per step)

scripts/
  train.py               Train on PG19 windows or a synthetic debug corpus
  evaluate.py            Continuation PPL + mixing-weight analysis + recursive PPL
  benchmark.py           Compression-ratio sweep vs full attention + sliding window
  eval_niah.py           Needle-in-Haystack accuracy across (depth × prefix length)

tests/
  test_compression.py    Unit tests
```

## Quick start

```bash
pip install -e .
pip install mamba-ssm  # optional; faster SSM on CUDA

# Smoke-test the training pipeline.
python scripts/train.py --model gpt2 --debug --epochs 1

# Train on PG19 (frozen GPT-2 decoder).
python scripts/train.py \
    --model gpt2 \
    --max-seq-length 1024 \
    --max-kv-slots 128 \
    --epochs 3

# Benchmark vs baselines on long-document continuation PPL.
python scripts/benchmark.py \
    --checkpoint checkpoints/best_model.pt \
    --base-model gpt2 \
    --prefix-length 512 \
    --continuation-length 128

# Needle-in-Haystack across depth and prefix length.
python scripts/eval_niah.py \
    --checkpoint checkpoints/best_model.pt \
    --prefix-lengths 256 512 1024 \
    --depths 0.1 0.3 0.5 0.7 0.9
```

## Out of scope

- **No reinforcement learning.** LM cross-entropy is differentiable; we
  use that. RL would only be justified by a non-differentiable downstream
  metric (e.g. exact-match accuracy on a retrieval task), which is not
  the objective here.
- **No token-level insertion.** The compressor never produces "virtual
  tokens" in the embedding space. The pretrained LM only ever sees its
  own (compressed) KV cache.
- **No heuristic-eviction baseline as the method.** H2O- and
  StreamingLLM-style truncation appear in `benchmark.py` as comparison
  points, not as the model under study.

## Status

- [x] Compressor + KV-cache hooks + differentiable training loop
- [x] Unit tests for shape, simplex, padding, gradient flow, aux losses
- [x] Benchmark script: full attention, sliding window, learned compressor
- [x] NIAH eval script over a (depth × prefix length) grid
- [ ] Recursive-compression experiments
- [ ] Cross-model-scale results (GPT-2 S/M/L, Pythia)
- [ ] Writeup

## License

MIT
