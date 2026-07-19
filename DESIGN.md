# Design: streaming LM with learned chunking

## Thesis

Jointly pretrain an LM and a KV-cache compressor from scratch, in a single
autoregressive pass. As the persistent KV cache fills toward a budget `M`, the
compressor mixes tokens into `D` slots, where `D = g(occupancy/M)` asymptotes below `M` so memory is bounded and compression grows with context.

The compressor is entered as the method under test against a `Standard`
transformer baseline with identical backbone, params, and hyperparameters,
differing only in the KV path.

## Architecture

Shared backbone: pre-norm, RMSNorm, SwiGLU FFN (`3d`), RoPE, no linear biases, weight-tied unembedding, context `4096`.

| Scale | L | d | H | Params |
|-------|---|-----|----|--------|
| XS | 8 | 512 | 8 | 53M |
| S | 12 | 768 | 12 | 131M |
| M | 24 | 1024 | 16 | 379M |
| L | 36 | 1280 | 20 | 831M |
| XL | 48 | 1600 | 25 | 1.678B |

**Compressor** (per-layer, cache-only, cross-attention). For layer `ℓ`, fold heads of its keys → candidates `C_ℓ`. `M` learned slot queries soft-masked to the active `D`:

```
A_ℓ = softmax_N( (S W_q)(C_ℓ W_k)ᵀ / √c + bias_ℓ / τ )   # (D, N)
K'_ℓ = A_ℓ K_ℓ ,  V'_ℓ = A_ℓ V_ℓ
```

One cross-attention = routing + mixing. Modules shared across layers.

**Streaming.** Process in chunks; carry compressed memory as the KV cache. `D` set by occupancy via soft masking, targeting `βM (β<1)` so occupancy asymptotes.

## Data

FineWeb-Edu, GPT-2 tokenizer. Pack across documents with `<|endoftext|>`; mask cross-document attention; exclude the post-EOS prediction from the loss.

## Training

AdamW β=(0.9, 0.95), wd=0.1, grad-clip 1.0, peak LR `6e-4`, bf16. Global batch ≈ `393k` tokens/update via grad accumulation. Warmup → constant → linear decay over the final 10%. 20B tokens (XS ≈ compute-optimal).

## Eval

- Validation NLL on FineWeb-Edu.
- OOD corpus NLL: WikiText, C4, Books3, GovReport.
- Zero-shot accuracy (LM Eval Harness): ARC-Easy, HellaSwag, PIQA, SciQ, LAMBADA.
- NIAH (depth × length)

## Open questions

- RoPE over compressed memory: mixed keys keep their source rotation. Validate loss and deep-needle NIAH before committing large runs.
- Cross-doc masking at scale: swap the dense `T×T` mask for flash-attn varlen.
