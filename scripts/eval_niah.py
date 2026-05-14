#!/usr/bin/env python3
"""Needle-in-a-Haystack eval for the compressed KV cache.

Builds long synthetic prefixes from a haystack corpus, inserts a needle
factoid at varying depths, then asks the model to retrieve the needle's
payload conditioned on the compressed prefix. Reports exact-match accuracy
(via constrained next-token logits) over a (depth × prefix_length) grid.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.models.config import CompressorConfig
from wavgpt.models.kv_extender import KVExtender


HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. The moon is white. "
    "Cats are mammals. Dogs are mammals. Birds can fly. Fish swim in water. "
    "Mountains are tall. Oceans are deep. Forests are dense with trees. "
    "Rivers flow downhill toward the sea. Deserts are dry and hot during the day. "
)

NEEDLE_TEMPLATE = "The secret passcode is {}."
QUESTION = " The secret passcode is"   # we score the token following this.


@dataclass
class NiahRow:
    prefix_length: int
    depth_frac: float
    compression_ratio: float
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


def make_needle(rng: random.Random) -> str:
    # Single-token-ish numeric payload makes scoring trivial: just score next token.
    return str(rng.randint(1000, 9999))


def build_prompt(tokenizer, prefix_length: int, depth_frac: float, needle: str) -> torch.Tensor:
    needle_text = NEEDLE_TEMPLATE.format(needle)
    haystack_tokens = tokenizer.encode(HAYSTACK, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

    # Repeat haystack until we have enough room around the needle.
    target_padding = prefix_length - len(needle_tokens)
    if target_padding <= 0:
        target_padding = max(0, prefix_length // 2)
    reps = (target_padding // len(haystack_tokens)) + 2
    pad_pool = haystack_tokens * reps

    insert_at = int(target_padding * depth_frac)
    before = pad_pool[:insert_at]
    after = pad_pool[:target_padding - insert_at]
    prefix = before + needle_tokens + after
    prefix = prefix[:prefix_length]
    return torch.tensor(prefix, dtype=torch.long)


@torch.no_grad()
def evaluate_compressor(
    model: KVExtender,
    tokenizer,
    prefix_lengths: List[int],
    depths: List[float],
    trials: int,
    device: str,
) -> List[NiahRow]:
    rows: List[NiahRow] = []
    rng = random.Random(0)

    for prefix_length in prefix_lengths:
        for depth in depths:
            correct = 0
            total = 0
            for _ in tqdm(range(trials), desc=f"L={prefix_length} d={depth:.2f}"):
                needle = make_needle(rng)
                needle_ids = tokenizer.encode(needle, add_special_tokens=False)
                if not needle_ids:
                    continue
                target_id = needle_ids[0]

                prefix = build_prompt(tokenizer, prefix_length, depth, needle).unsqueeze(0).to(device)
                question_ids = torch.tensor(
                    [tokenizer.encode(QUESTION, add_special_tokens=False)],
                    dtype=torch.long, device=device,
                )

                # Forward: compress prefix, then score the first continuation token after `question`.
                full_cont = torch.cat([question_ids, torch.tensor([[target_id]], device=device)], dim=1)
                out = model.forward(
                    prefix_ids=prefix, continuation_ids=full_cont,
                    gumbel_noise=False, return_aux=False,
                )
                # cont_logits[:, t] predicts continuation[:, t+1].
                # Token we want is the one AFTER the question, i.e. position len(question_ids)-1 in cont_logits.
                pred_idx = question_ids.size(1) - 1
                pred = int(out.cont_logits[0, pred_idx].argmax().item())
                if pred == target_id:
                    correct += 1
                total += 1

            ratio = prefix_length / max(model.config.max_kv_slots, 1)
            rows.append(NiahRow(prefix_length, depth, ratio, correct, total))
    return rows


def print_grid(rows: List[NiahRow]):
    by_L = {}
    by_d = sorted({r.depth_frac for r in rows})
    for r in rows:
        by_L.setdefault(r.prefix_length, {})[r.depth_frac] = r.accuracy
    print(f"\nNIAH accuracy (rows = prefix length, cols = depth fraction):\n")
    header = f"{'L':>6} | " + " ".join(f"{d:>6.2f}" for d in by_d)
    print(header)
    print("-" * len(header))
    for L in sorted(by_L):
        row = f"{L:>6} | " + " ".join(f"{by_L[L].get(d, 0.0):>6.2f}" for d in by_d)
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prefix-lengths", type=int, nargs="*", default=[256, 512, 1024])
    p.add_argument("--depths", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str)
    args = p.parse_args()

    torch.serialization.add_safe_globals([CompressorConfig])
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt.get("model_config", CompressorConfig())
    model = KVExtender.from_pretrained(config.pretrained_model_name, config=config).to(args.device)
    if "compressor_state_dict" in ckpt:
        model.compressor.load_state_dict(ckpt["compressor_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = evaluate_compressor(
        model, tokenizer,
        prefix_lengths=args.prefix_lengths,
        depths=args.depths,
        trials=args.trials,
        device=args.device,
    )
    print_grid(rows)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
