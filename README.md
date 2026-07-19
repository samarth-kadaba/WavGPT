# CHUNKY

A streaming language model with **learned KV-cache compression**. As the KV
cache fills toward a fixed budget `M`, a per-layer cross-attention compressor
mixes tokens into a bounded set of slots — so memory stays bounded while context
grows. Trained from scratch, jointly with the LM, in a single pass.

## How it works

The sequence is processed in chunks, carrying a compressed KV memory per layer.
After each chunk, the shared compressor refolds `memory + chunk` back into the
budget via one cross-attention (routing + mixing), with the number of active
slots growing toward `βM` and never reaching it. Training is ordinary next-token
prediction over the whole pass.

We benchmark against a `Standard` transformer with an identical backbone
(RMSNorm, SwiGLU, RoPE, weight-tied embeddings) — the only difference is the KV
path.

See [`DESIGN.md`](DESIGN.md) for the full design.

## Layout

```
src/chunky/
  model.py        transformer backbone + Standard baseline
  compressor.py   cross-attention KV compressor
  streaming.py    single-pass streaming compressed LM
  data.py         packing, cross-doc masking, corpora
  pretrain.py     training loop (WSD schedule, DDP, W&B)
scripts/pretrain.py   torchrun entry
modal_app.py          Modal launcher
```

## Install

```bash
pip install -e .
```

## Run

Training runs on Modal (2×H100, data-parallel). Both variants log to the same
W&B project (`chunky`) and record train/val loss.

```bash
# validate both variants (tiny, fast)
modal run modal_app.py::smoke

# launch the parallel study (only after smoke is green)
modal run --detach modal_app.py::train --variant standard
modal run --detach modal_app.py::train --variant ours
```

Locally with torchrun:

```bash
torchrun --nproc_per_node=2 scripts/pretrain.py --variant ours --scale xs
```

## Model scales

| Scale | Layers | Width | Heads | Params |
|-------|--------|-------|-------|--------|
| xs | 8 | 512 | 8 | 53M |
| s | 12 | 768 | 12 | 131M |
| m | 24 | 1024 | 16 | 379M |
| l | 36 | 1280 | 20 | 831M |
| xl | 48 | 1600 | 25 | 1.678B |

## License

MIT
