#!/usr/bin/env python3
"""torchrun entry: pretrain the Standard baseline or the streaming compressor."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer  # noqa: E402

from chunky.pretrain import TrainConfig, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--variant", choices=["standard", "ours"], default="standard")
    p.add_argument("--scale", default="xs")
    p.add_argument("--dataset", default="fineweb-edu")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--total-tokens", type=int, default=20_000_000_000)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = TrainConfig(
        variant=args.variant, scale=args.scale, dataset=args.dataset,
        seq_len=args.seq_len, total_tokens=args.total_tokens, micro_batch=args.micro_batch,
        budget=args.budget, chunk_size=args.chunk_size, out_dir=args.out_dir,
        ckpt_every=args.ckpt_every, log_every=args.log_every,
    )
    train(cfg, tokenizer)


if __name__ == "__main__":
    main()
    os._exit(0)  # skip finalization; avoids native teardown crash from streaming threads
