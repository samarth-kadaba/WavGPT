#!/usr/bin/env python3
"""Compression-ratio sweep: continuation perplexity vs prefix length / K_slots."""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.models.config import CompressorConfig
from wavgpt.models.kv_extender import KVExtender


@dataclass
class Row:
    method: str
    prefix_length: int
    compressed_length: int
    compression_ratio: float
    perplexity: float
    n_tokens: int


def load_docs(tokenizer, num_docs: int, prefix_length: int, cont_length: int, device: str):
    needed = prefix_length + cont_length
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    docs = []
    for item in tqdm(ds, total=num_docs * 3, desc="Loading"):
        toks = tokenizer.encode(item["text"], add_special_tokens=False)
        if len(toks) >= needed:
            docs.append(torch.tensor(toks[:needed], device=device))
            if len(docs) >= num_docs:
                break
    return docs


@torch.no_grad()
def eval_full_attention(base_model, docs: List[torch.Tensor], prefix_length: int) -> Row:
    """No compression: full attention on prefix + continuation."""
    total_loss, total_n = 0.0, 0
    for doc in tqdm(docs, desc="Full"):
        ids = doc.unsqueeze(0)
        out = base_model(input_ids=ids, return_dict=True)
        logits = out.logits[:, prefix_length - 1:-1, :]
        labels = ids[:, prefix_length:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_n += labels.numel()
    avg = total_loss / max(total_n, 1)
    return Row("Full attention", prefix_length, prefix_length, 1.0, math.exp(avg), total_n)


@torch.no_grad()
def eval_sliding_window(base_model, docs: List[torch.Tensor], prefix_length: int,
                       window: int) -> Row:
    """Truncate prefix to last `window` tokens (StreamingLLM-like)."""
    total_loss, total_n = 0.0, 0
    for doc in tqdm(docs, desc=f"Window={window}"):
        ids = doc.unsqueeze(0)
        kept_prefix = ids[:, max(0, prefix_length - window):prefix_length]
        cont = ids[:, prefix_length:]
        merged = torch.cat([kept_prefix, cont], dim=1)
        out = base_model(input_ids=merged, return_dict=True)
        L_kept = kept_prefix.size(1)
        logits = out.logits[:, L_kept - 1:-1, :]
        labels = cont
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_n += labels.numel()
    avg = total_loss / max(total_n, 1)
    return Row(f"Sliding window={window}", prefix_length, window,
               prefix_length / max(window, 1), math.exp(avg), total_n)


@torch.no_grad()
def eval_compressed(model: KVExtender, docs: List[torch.Tensor], prefix_length: int) -> Row:
    total_loss, total_n = 0.0, 0
    for doc in tqdm(docs, desc=f"Compressed K={model.config.max_kv_slots}"):
        ids = doc.unsqueeze(0)
        prefix = ids[:, :prefix_length]
        cont = ids[:, prefix_length:]
        out = model.forward(prefix_ids=prefix, continuation_ids=cont,
                            gumbel_noise=False, return_aux=False)
        n = cont.size(1) - 1
        total_loss += float(out.lm_loss) * n
        total_n += n
    avg = total_loss / max(total_n, 1)
    K_slots = model.config.max_kv_slots
    return Row(f"Compressor K={K_slots}", prefix_length, K_slots,
               prefix_length / max(K_slots, 1), math.exp(avg), total_n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, help="Path to trained compressor checkpoint")
    p.add_argument("--base-model", type=str, default="gpt2")
    p.add_argument("--num-docs", type=int, default=10)
    p.add_argument("--prefix-length", type=int, default=512)
    p.add_argument("--continuation-length", type=int, default=128)
    p.add_argument("--windows", type=int, nargs="*", default=[64, 128, 256])
    p.add_argument("--output", type=str)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
    ).to(args.device).eval()

    docs = load_docs(tokenizer, args.num_docs, args.prefix_length,
                     args.continuation_length, args.device)
    if not docs:
        print("No documents loaded; reduce --prefix-length / --continuation-length")
        return

    rows: List[Row] = []
    rows.append(eval_full_attention(base_model, docs, args.prefix_length))
    for w in args.windows:
        rows.append(eval_sliding_window(base_model, docs, args.prefix_length, w))

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        config = ckpt.get("model_config", CompressorConfig(pretrained_model_name=args.base_model))
        comp_model = KVExtender.from_pretrained(
            config.pretrained_model_name, config=config,
        ).to(args.device)
        if "compressor_state_dict" in ckpt:
            comp_model.compressor.load_state_dict(ckpt["compressor_state_dict"])
        comp_model.eval()
        rows.append(eval_compressed(comp_model, docs, args.prefix_length))

    print("\nContinuation perplexity by method (lower is better):")
    print(f"{'method':<30} {'comp.ratio':>10} {'ppl':>10}")
    print("-" * 54)
    for r in rows:
        print(f"{r.method:<30} {r.compression_ratio:>9.1f}x {r.perplexity:>10.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
