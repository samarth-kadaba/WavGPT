#!/usr/bin/env python3
"""
Benchmark: Long Context Perplexity Evaluation

Compares:
  1. Base model (sliding window - loses past context)
  2. Extended model with learned chunking (our method)

KEY INSIGHT: No past/current split. The model sees all tokens up to position t
and internally decides how to chunk them.

Usage:
    # Base model only
    python scripts/benchmark.py --base-only --num-docs 5

    # Compare base vs trained model
    python scripts/benchmark.py --checkpoint checkpoints/best.pt

    # Compare boundary strategies
    python scripts/benchmark.py --checkpoint checkpoints/best.pt --compare-boundaries
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.models.context_extender import ContextExtender
from wavgpt.models.config import ContextExtenderConfig


class BoundaryStrategy(Enum):
    """How to determine chunk boundaries."""
    LEARNED = "learned"   # Use trained policy
    FIXED = "fixed"       # Every N tokens
    RANDOM = "random"     # Random placement


@dataclass
class PerplexityResult:
    """Perplexity at a position range."""
    position_start: int
    position_end: int
    perplexity: float
    num_tokens: int


@dataclass 
class BenchmarkResults:
    """Results for one model/strategy."""
    name: str
    by_position: Dict[str, PerplexityResult] = field(default_factory=dict)
    overall_ppl: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "overall_ppl": self.overall_ppl,
            "by_position": {
                k: {"ppl": v.perplexity, "tokens": v.num_tokens}
                for k, v in self.by_position.items()
            }
        }


class LongContextBenchmark:
    """
    Benchmark long document perplexity.
    
    For position t in a document, we compare:
    - Base model: only sees last 1024 tokens (sliding window)
    - Our model: sees all tokens 0 to t-1 (via learned chunking)
    """
    
    RANGES = [
        ("0-1K", 0, 1024),
        ("1K-2K", 1024, 2048),
        ("2K-4K", 2048, 4096),
        ("4K-8K", 4096, 8192),
    ]
    
    def __init__(self, base_model: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.base_model_name = base_model
        self.context_window = 1024
        
        print(f"Loading {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        
        self.extended_model: Optional[ContextExtender] = None
        
    def load_extended_model(self, checkpoint_path: str):
        """Load trained context extender."""
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("model_config", ContextExtenderConfig())
        self.extended_model = ContextExtender.from_pretrained(
            config.pretrained_model_name,
            config=config,
        ).to(self.device)
        if "model_state_dict" in ckpt:
            self.extended_model.load_state_dict(ckpt["model_state_dict"])
        self.extended_model.eval()
        
    def load_documents(self, num_docs: int = 10, min_length: int = 4096) -> List[torch.Tensor]:
        """Load long documents from PG19."""
        print(f"Loading {num_docs} documents (min {min_length} tokens)...")
        docs = []
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        
        for item in tqdm(ds, total=num_docs * 3, desc="Loading"):
            tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
            if len(tokens) >= min_length:
                docs.append(torch.tensor(tokens[:min_length], device=self.device))
                if len(docs) >= num_docs:
                    break
        return docs
    
    @torch.no_grad()
    def evaluate_base(self, docs: List[torch.Tensor], max_pos: int) -> BenchmarkResults:
        """
        Evaluate base model (sliding window).
        
        At position t, base model only sees tokens [t-1024:t].
        Earlier tokens are lost.
        """
        print("\n" + "=" * 50)
        print("BASE MODEL (sliding window)")
        print("=" * 50)
        
        losses = {name: (0.0, 0) for name, _, _ in self.RANGES}
        total_loss, total_n = 0.0, 0
        
        for doc in tqdm(docs, desc="Base"):
            for name, start, end in self.RANGES:
                if start >= min(len(doc), max_pos):
                    continue
                for pos in range(max(start, self.context_window), min(end, len(doc), max_pos)):
                    # Sliding window: only see last context_window tokens
                    window = doc[pos - self.context_window:pos].unsqueeze(0)
                    target = doc[pos].unsqueeze(0)
                    logits = self.base_model(window).logits[0, -1]
                    loss = F.cross_entropy(logits, target).item()
                    losses[name] = (losses[name][0] + loss, losses[name][1] + 1)
                    total_loss += loss
                    total_n += 1
        
        results = BenchmarkResults(name="Base (sliding window)")
        for name, start, end in self.RANGES:
            if losses[name][1] > 0:
                ppl = math.exp(losses[name][0] / losses[name][1])
                results.by_position[name] = PerplexityResult(start, end, ppl, losses[name][1])
        results.overall_ppl = math.exp(total_loss / total_n) if total_n > 0 else float('inf')
        return results
    
    @torch.no_grad()
    def evaluate_extended(
        self, 
        docs: List[torch.Tensor], 
        max_pos: int,
    ) -> BenchmarkResults:
        """
        Evaluate extended model with learned chunking.
        
        At position t, model sees ALL tokens [0:t] and internally
        decides how to chunk them for compression.
        """
        if self.extended_model is None:
            raise ValueError("Call load_extended_model() first")
            
        print("\n" + "=" * 50)
        print("EXTENDED MODEL (learned chunking)")
        print("=" * 50)
        
        losses = {name: (0.0, 0) for name, _, _ in self.RANGES}
        total_loss, total_n = 0.0, 0
        
        for doc in tqdm(docs, desc="Extended"):
            for name, start, end in self.RANGES:
                if start >= min(len(doc), max_pos):
                    continue
                for pos in range(max(start, 100), min(end, len(doc), max_pos)):
                    # Model sees ALL tokens up to position
                    # Internally handles chunking
                    input_ids = doc[:pos].unsqueeze(0)
                    target = doc[pos].unsqueeze(0)
                    
                    # Forward pass (model decides chunking)
                    outputs = self.extended_model(
                        input_ids=input_ids,
                        use_deterministic_boundaries=True,
                    )
                    
                    # Get last position logits
                    logits = outputs.logits[0, -1]
                    loss = F.cross_entropy(logits, target).item()
                    
                    losses[name] = (losses[name][0] + loss, losses[name][1] + 1)
                    total_loss += loss
                    total_n += 1
        
        results = BenchmarkResults(name="Extended (learned chunking)")
        for name, start, end in self.RANGES:
            if losses[name][1] > 0:
                ppl = math.exp(losses[name][0] / losses[name][1])
                results.by_position[name] = PerplexityResult(start, end, ppl, losses[name][1])
        results.overall_ppl = math.exp(total_loss / total_n) if total_n > 0 else float('inf')
        return results


def print_comparison(results: List[BenchmarkResults]):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("RESULTS: Long Context Perplexity (lower is better)")
    print("=" * 70)
    
    # Header
    header = f"{'Position':<10}"
    for r in results:
        header += f"{r.name[:25]:>25}"
    print(header)
    print("-" * (10 + 25 * len(results)))
    
    # By position
    for range_name in ["0-1K", "1K-2K", "2K-4K", "4K-8K"]:
        row = f"{range_name:<10}"
        for r in results:
            if range_name in r.by_position:
                ppl = r.by_position[range_name].perplexity
                row += f"{ppl:>25.2f}"
            else:
                row += f"{'--':>25}"
        print(row)
    
    # Overall
    print("-" * (10 + 25 * len(results)))
    row = f"{'Overall':<10}"
    for r in results:
        row += f"{r.overall_ppl:>25.2f}"
    print(row)
    
    # Analysis
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        base = results[0]
        for r in results[1:]:
            improvement = ((base.overall_ppl - r.overall_ppl) / base.overall_ppl) * 100
            symbol = "✓" if improvement > 0 else "✗"
            print(f"\n{symbol} {r.name}: {improvement:+.1f}% vs base")
            
            # Show where it helps most
            if improvement > 0:
                best_range = None
                best_delta = 0
                for name in ["1K-2K", "2K-4K", "4K-8K"]:
                    if name in base.by_position and name in r.by_position:
                        delta = base.by_position[name].perplexity - r.by_position[name].perplexity
                        if delta > best_delta:
                            best_delta = delta
                            best_range = name
                if best_range:
                    print(f"  Best improvement at {best_range}: -{best_delta:.2f} PPL")


def main():
    parser = argparse.ArgumentParser(description="Benchmark long context perplexity")
    parser.add_argument("--checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--num-docs", type=int, default=5)
    parser.add_argument("--max-position", type=int, default=4096)
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    benchmark = LongContextBenchmark(args.base_model, args.device)
    docs = benchmark.load_documents(args.num_docs, args.max_position)
    
    if not docs:
        print("Error: No documents loaded")
        return
    
    all_results = []
    
    # Always evaluate base
    all_results.append(benchmark.evaluate_base(docs, args.max_position))
    
    # Evaluate extended model if checkpoint provided
    if args.checkpoint and not args.base_only:
        benchmark.load_extended_model(args.checkpoint)
        all_results.append(benchmark.evaluate_extended(docs, args.max_position))
    
    # Print comparison
    print_comparison(all_results)
    
    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
