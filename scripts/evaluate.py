#!/usr/bin/env python3
"""Evaluate the trained KV compressor: perplexity, mixing analysis, generation."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.models.config import CompressorConfig
from wavgpt.models.kv_extender import KVExtender
from wavgpt.training.trainer import split_prefix_continuation


def load_model(checkpoint_path: str, device: str = "cuda"):
    torch.serialization.add_safe_globals([CompressorConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("model_config", CompressorConfig())
    model = KVExtender.from_pretrained(config.pretrained_model_name, config=config).to(device)
    if "compressor_state_dict" in ckpt:
        model.compressor.load_state_dict(ckpt["compressor_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, config


@torch.no_grad()
def eval_perplexity(model, tokenizer, text: str, prefix_frac: float = 0.75,
                    device: str = "cuda"):
    """Compress `prefix_frac` of `text`, predict the rest."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    T = input_ids.size(1)
    T_prefix = int(T * prefix_frac)
    prefix = input_ids[:, :T_prefix]
    cont = input_ids[:, T_prefix:]
    out = model.forward(
        prefix_ids=prefix, continuation_ids=cont,
        gumbel_noise=False, return_aux=False,
    )
    return {
        "perplexity": float(torch.exp(out.lm_loss)),
        "loss": float(out.lm_loss),
        "prefix_length": out.prefix_length,
        "continuation_length": out.continuation_length,
        "compressed_length": out.compressed_length,
        "compression_ratio": out.prefix_length / max(out.compressed_length, 1),
    }


@torch.no_grad()
def analyze_mixing(model, tokenizer, text: str, device: str = "cuda", max_slots: int = 8):
    """Show which prefix positions each compressed slot pulls from."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    T = input_ids.size(1)

    _, W, importance = model.compress_cache(input_ids)
    W = W[0]            # (K_slots, T)
    importance = importance[0]
    K_slots = W.size(0)

    print(f"\nTokens: {T}, K_slots: {K_slots}, compression ratio: {T / K_slots:.2f}x")
    print(f"Importance: mean={float(importance.mean()):.3f}, "
          f"std={float(importance.std()):.3f}, "
          f"range=[{float(importance.min()):.2f}, {float(importance.max()):.2f}]")

    eff_per_slot = torch.exp(-(W * (W + 1e-9).log()).sum(dim=-1))
    print(f"Effective source positions per slot: mean={float(eff_per_slot.mean()):.1f}, "
          f"min={float(eff_per_slot.min()):.1f}, max={float(eff_per_slot.max()):.1f}")

    print(f"\nTop-3 source positions for the first {min(max_slots, K_slots)} slots:")
    for k in range(min(max_slots, K_slots)):
        top_vals, top_idx = W[k].topk(min(3, T))
        snippets = []
        for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
            snippet = tokenizer.decode(tokens[max(0, idx - 1):idx + 2]).replace("\n", " ")[:30]
            snippets.append(f"pos {idx} ({v:.2f}): {snippet!r}")
        print(f"  slot {k:>3}: " + " | ".join(snippets))


@torch.no_grad()
def eval_recursive(model, tokenizer, text: str, num_rounds: int = 4, device: str = "cuda"):
    """Recursive compression: compress, then compress the decompressed continuation, etc.

    Simpler proxy: at each round, increase prefix_frac and measure continuation PPL."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    T = input_ids.size(1)
    print(f"\nRecursive perplexity (single-round w/ growing prefix), T={T}:")
    print(f"{'round':>5} {'prefix':>7} {'cont':>5} {'ratio':>7} {'ppl':>8}")
    for r in range(1, num_rounds + 1):
        prefix_frac = min(0.9, 0.4 + 0.1 * r)
        T_prefix = int(T * prefix_frac)
        cont_len = max(32, T - T_prefix)
        if T_prefix < 32 or cont_len < 8:
            continue
        prefix = input_ids[:, :T_prefix]
        cont = input_ids[:, T_prefix:T_prefix + cont_len]
        out = model.forward(prefix_ids=prefix, continuation_ids=cont,
                            gumbel_noise=False, return_aux=False)
        ppl = float(torch.exp(out.lm_loss))
        ratio = T_prefix / out.compressed_length
        print(f"{r:>5} {T_prefix:>7} {cont_len:>5} {ratio:>6.1f}x {ppl:>8.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--text", type=str)
    p.add_argument("--prefix-frac", type=float, default=0.75)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--recursive", action="store_true")
    args = p.parse_args()

    model, tokenizer, config = load_model(args.checkpoint, args.device)
    print(f"Pretrained: {config.pretrained_model_name}   K_slots: {config.max_kv_slots}")

    text = args.text or (
        "The history of artificial intelligence began in antiquity with myths and stories "
        "of artificial beings endowed with intelligence by master craftsmen. The seeds of "
        "modern AI were planted by philosophers describing thinking as mechanical symbol "
        "manipulation, culminating in the programmable digital computer of the 1940s. "
        "Alan Turing proposed the imitation game as a test of machine intelligence in 1950. "
        "John McCarthy organised the Dartmouth workshop in 1956, coining the term "
        "artificial intelligence and launching the field as an academic discipline."
    )

    r = eval_perplexity(model, tokenizer, text, prefix_frac=args.prefix_frac, device=args.device)
    print(f"\nPPL on continuation: {r['perplexity']:.2f}  "
          f"loss: {r['loss']:.4f}  "
          f"compression ratio: {r['compression_ratio']:.1f}x")

    if args.analyze:
        analyze_mixing(model, tokenizer, text, device=args.device)
    if args.recursive:
        eval_recursive(model, tokenizer, text, device=args.device)


if __name__ == "__main__":
    main()
