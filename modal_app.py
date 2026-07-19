"""Modal launcher.

    modal run modal_app.py::smoke                      # validate both variants
    modal run --detach modal_app.py::train --variant standard
    modal run --detach modal_app.py::train --variant ours

Launch both variants for the parallel study. Run `smoke` green before a full run.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "wandb", "numpy")
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App("chunky", image=image)
volume = modal.Volume.from_name("chunky-ckpts", create_if_missing=True)
data_volume = modal.Volume.from_name("chunky-data", create_if_missing=True)
DATA_DIR = "/data/fineweb-edu"


@app.function(volumes={"/data": data_volume}, timeout=3600)
def prepare_data() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "HuggingFaceFW/fineweb-edu", repo_type="dataset",
        allow_patterns="sample/10BT/*.parquet", local_dir=DATA_DIR,
    )
    data_volume.commit()
    print("data ready at", DATA_DIR)


@app.function(gpu="A10G", timeout=900)
def smoke() -> None:
    import sys

    import torch

    sys.path.insert(0, "/root/src")
    from chunky.compressor import CompressorConfig
    from chunky.model import ModelConfig, Transformer
    from chunky.streaming import CompressedTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(vocab_size=512, n_layers=2, d_model=64, n_heads=4, max_seq_len=64)
    ids = torch.randint(0, 512, (2, 32), device=dev)

    variants = {
        "standard": Transformer(cfg),
        "ours": CompressedTransformer(cfg, CompressorConfig(d_model=64, n_heads=4, max_slots=16), chunk_size=8),
    }
    for name, model in variants.items():
        model = model.to(dev)
        _, loss = model(ids, labels=ids)
        loss.backward()
        assert torch.isfinite(loss), f"{name}: non-finite loss"
        print(f"{name}: loss={loss.item():.4f} params={model.num_params():,}  OK")


@app.function(gpu="H100", timeout=1800)
def profile(seq_len: int = 4096, steps: int = 6, warmup: int = 2) -> None:
    import subprocess
    import sys
    import threading
    import time

    import torch

    sys.path.insert(0, "/root/src")
    from chunky.compressor import CompressorConfig
    from chunky.data import cross_doc_attn_mask
    from chunky.model import SCALES, Transformer
    from chunky.streaming import CompressedTransformer

    dev = "cuda"
    mcfg = SCALES["xs"]

    def sample_util(stop, out):
        while not stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            try:
                out.append(int(r.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.05)

    def run(variant, mb):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if variant == "standard":
            model = Transformer(mcfg).to(dev)
            mask = cross_doc_attn_mask(torch.zeros(mb, seq_len, dtype=torch.long, device=dev))
        else:
            comp = CompressorConfig(d_model=mcfg.d_model, n_heads=mcfg.n_heads, max_slots=512)
            model = CompressedTransformer(mcfg, comp, chunk_size=512).to(dev)
            mask = None
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ids = torch.randint(0, mcfg.vocab_size, (mb, seq_len), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, attn_mask=mask, labels=ids)
            loss.backward()
            opt.step()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        stop, utils = threading.Event(), []
        t = threading.Thread(target=sample_util, args=(stop, utils))
        t.start()
        t0 = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        stop.set()
        t.join()
        toks = mb * seq_len * steps / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        util = sum(utils) / len(utils) if utils else -1
        del model, opt, ids, mask
        return toks, peak, util

    print(f"{'variant':>9} {'mb':>4} {'tok/s':>10} {'peakGB':>8} {'util%':>6}")
    for variant in ["standard", "ours"]:
        for mb in [8, 16, 24, 32, 48, 64]:
            try:
                toks, peak, util = run(variant, mb)
                print(f"{variant:>9} {mb:>4} {toks:>10.0f} {peak:>8.1f} {util:>6.0f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{variant:>9} {mb:>4}  OOM")
                    torch.cuda.empty_cache()
                    break
                raise


@app.function(gpu="H100:2", timeout=24 * 3600,
              volumes={"/ckpt": volume, "/data": data_volume},
              secrets=[modal.Secret.from_name("wandb")], retries=3)
def train(variant: str = "standard", scale: str = "xs",
          total_tokens: int = 20_000_000_000, micro_batch: int = 8) -> None:
    import os
    import subprocess
    import sys
    import time

    volume.reload()  # pick up checkpoints from a previous (preempted) attempt
    data_volume.reload()
    proc = subprocess.Popen(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
         "/root/scripts/pretrain.py",
         "--variant", variant, "--scale", scale,
         "--total-tokens", str(total_tokens), "--micro-batch", str(micro_batch),
         "--out-dir", f"/ckpt/{variant}_{scale}"],
        cwd="/root", env={**os.environ, "PYTHONPATH": "/root/src", "CHUNKY_DATA_DIR": DATA_DIR},
    )
    while proc.poll() is None:  # persist checkpoints periodically for resumability
        time.sleep(300)
        volume.commit()
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"training exited with code {proc.returncode}")


@app.local_entrypoint()
def main(variant: str = "both") -> None:
    """Spawn training async so it runs to completion in the cloud.

    Launch with: modal run --detach modal_app.py --variant both
    """
    variants = ["standard", "ours"] if variant == "both" else [variant]
    for v in variants:
        handle = train.spawn(variant=v)
        print(f"spawned {v}: {handle.object_id}")
