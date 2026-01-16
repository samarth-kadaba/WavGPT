#!/usr/bin/env python3
"""
Evaluation script for Context Extension model.

Tests:
  1. Perplexity on long documents
  2. Text generation with extended context
  3. Boundary placement analysis

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --generate --prompt "Once upon a time"
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.models.context_extender import ContextExtender
from wavgpt.models.config import ContextExtenderConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Allow our custom config class for safe loading
    torch.serialization.add_safe_globals([ContextExtenderConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("model_config", ContextExtenderConfig())
    
    model = ContextExtender.from_pretrained(
        config.pretrained_model_name,
        config=config,
    ).to(device)
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # Handle torch.compile _orig_mod prefix
        # Strip "_orig_mod." from keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, config


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, text: str, device: str = "cuda"):
    """Compute perplexity on a text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    
    # Get model output
    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        use_deterministic_boundaries=True,
    )
    
    if outputs.loss is not None:
        ppl = torch.exp(outputs.loss).item()
    else:
        ppl = float('inf')
    
    return {
        "perplexity": ppl,
        "loss": outputs.loss.item() if outputs.loss is not None else None,
        "num_chunks": outputs.num_chunks,
        "num_kept_tokens": outputs.num_kept_tokens,
        "num_boundaries": outputs.boundaries.sum().item(),
    }


@torch.no_grad()
def generate_text(
    model, 
    tokenizer, 
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """Generate text with extended context."""
    print(f"\n{'='*60}")
    print("TEXT GENERATION")
    print(f"{'='*60}")
    print(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt length: {input_ids.size(1)} tokens")
    
    # Generate
    print("\nGenerating...")
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    new_text = tokenizer.decode(generated[0, input_ids.size(1):], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print("GENERATED TEXT")
    print(f"{'='*60}")
    print(f"\n{generated_text}")
    print(f"\n{'='*60}")
    print(f"Generated {generated.size(1) - input_ids.size(1)} new tokens")
    
    return {
        "prompt": prompt,
        "generated": generated_text,
        "new_tokens": new_text,
        "prompt_length": input_ids.size(1),
        "total_length": generated.size(1),
    }


@torch.no_grad()
def analyze_chunks(model, tokenizer, text: str, device: str = "cuda", max_display: int = 10):
    """
    Comprehensive chunk analysis with ASCII visualization.
    
    Shows:
    - Visual diagram of chunk decomposition
    - Chunk length statistics
    - Kept tokens vs compressed tokens
    - Content preview of each chunk
    """
    print(f"\n{'='*70}")
    print("  CHUNK DECOMPOSITION ANALYSIS")
    print(f"{'='*70}")
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    T = len(tokens)
    
    # Get embeddings
    embeddings = model.get_embeddings(input_ids)
    
    # Get policy decisions using the sample() method
    samples, policy_output = model.policy.sample(embeddings, num_samples=1, deterministic=False)
    sample = samples[0]
    
    boundaries = sample.boundaries[0]  # (T,)
    keep_mask = sample.keep_mask[0]  # (T,)
    boundary_probs = policy_output.boundary_probs[0]  # (T,)
    keep_probs = policy_output.keep_probs[0]  # (T,)
    
    # Compute expected values for reference
    expected_boundaries = boundary_probs.sum().item()
    expected_kept = keep_probs.sum().item()
    
    # Extract chunk info
    boundary_positions = boundaries.nonzero(as_tuple=True)[0].tolist()
    kept_positions = keep_mask.nonzero(as_tuple=True)[0].tolist()
    
    # Build chunks
    chunks = []
    prev_pos = 0
    for pos in boundary_positions:
        chunks.append((prev_pos, pos + 1))  # inclusive end
        prev_pos = pos + 1
    # Last chunk
    if prev_pos < T:
        chunks.append((prev_pos, T))
    
    num_chunks = len(chunks)
    num_kept = len(kept_positions)
    
    # ════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    print(f"\n┌{'─'*68}┐")
    print(f"│{'OVERVIEW':^68}│")
    print(f"├{'─'*68}┤")
    print(f"│  Total tokens:        {T:<44}│")
    print(f"│  Compressed chunks:   {num_chunks:<6} (expected: {expected_boundaries:.0f}){' '*(30-len(str(num_chunks)))}│")
    print(f"│  Kept tokens:         {num_kept:<6} (expected: {expected_kept:.0f}){' '*(30-len(str(num_kept)))}│")
    print(f"│  Context used:        {num_chunks + num_kept} / {model.config.max_context} ({100*(num_chunks+num_kept)/model.config.max_context:.1f}%){' '*(28-len(str(num_chunks+num_kept))-len(str(model.config.max_context)))}│")
    print(f"│  Compression ratio:   {T:.0f} → {num_chunks + num_kept} ({T/(num_chunks+num_kept) if num_chunks+num_kept > 0 else 0:.1f}x){' '*(35-len(str(T))-len(str(num_chunks+num_kept)))}│")
    print(f"└{'─'*68}┘")
    
    # ════════════════════════════════════════════════════════════════════
    # ASCII DIAGRAM
    # ════════════════════════════════════════════════════════════════════
    print(f"\n┌{'─'*68}┐")
    print(f"│{'SEQUENCE VISUALIZATION':^68}│")
    print(f"├{'─'*68}┤")
    
    # Create a visual representation
    # Scale to fit ~60 chars
    scale = min(1.0, 60 / T)
    visual_len = max(60, int(T * scale))
    
    # Build the visual string
    visual = ['░'] * visual_len  # Default: compressed
    
    # Mark kept tokens
    for pos in kept_positions:
        scaled_pos = int(pos * scale * visual_len / T)
        if scaled_pos < visual_len:
            visual[scaled_pos] = '█'
    
    # Mark boundaries
    for pos in boundary_positions:
        scaled_pos = int(pos * scale * visual_len / T)
        if scaled_pos < visual_len:
            visual[scaled_pos] = '│'
    
    visual_str = ''.join(visual[:60])
    print(f"│  {visual_str}      │")
    print(f"│                                                                    │")
    print(f"│  Legend: ░ = compressed  █ = kept  │ = chunk boundary              │")
    print(f"└{'─'*68}┘")
    
    # ════════════════════════════════════════════════════════════════════
    # CHUNK LENGTH STATISTICS
    # ════════════════════════════════════════════════════════════════════
    if chunks:
        chunk_lengths = [end - start for start, end in chunks]
        
        print(f"\n┌{'─'*68}┐")
        print(f"│{'CHUNK LENGTH STATISTICS':^68}│")
        print(f"├{'─'*68}┤")
        print(f"│  Mean:    {sum(chunk_lengths)/len(chunk_lengths):>6.1f} tokens{' '*45}│")
        print(f"│  Median:  {sorted(chunk_lengths)[len(chunk_lengths)//2]:>6} tokens{' '*45}│")
        print(f"│  Min:     {min(chunk_lengths):>6} tokens{' '*45}│")
        print(f"│  Max:     {max(chunk_lengths):>6} tokens{' '*45}│")
        print(f"│  Std:     {(sum((x-sum(chunk_lengths)/len(chunk_lengths))**2 for x in chunk_lengths)/len(chunk_lengths))**0.5:>6.1f} tokens{' '*45}│")
        print(f"└{'─'*68}┘")
        
        # ════════════════════════════════════════════════════════════════════
        # CHUNK LENGTH HISTOGRAM
        # ════════════════════════════════════════════════════════════════════
        print(f"\n┌{'─'*68}┐")
        print(f"│{'CHUNK LENGTH DISTRIBUTION':^68}│")
        print(f"├{'─'*68}┤")
        
        # Create histogram buckets
        max_len = max(chunk_lengths)
        min_len = min(chunk_lengths)
        num_buckets = min(10, max_len - min_len + 1)
        if num_buckets > 0:
            bucket_size = max(1, (max_len - min_len + 1) // num_buckets)
            buckets = {}
            for length in chunk_lengths:
                bucket = ((length - min_len) // bucket_size) * bucket_size + min_len
                buckets[bucket] = buckets.get(bucket, 0) + 1
            
            max_count = max(buckets.values())
            for bucket in sorted(buckets.keys())[:8]:
                count = buckets[bucket]
                bar_len = int(40 * count / max_count) if max_count > 0 else 0
                bar = '▓' * bar_len
                label = f"{bucket:>3}-{bucket+bucket_size-1:<3}"
                print(f"│  {label} │{bar:<40} {count:>3}  │")
        
        print(f"└{'─'*68}┘")
    
    # ════════════════════════════════════════════════════════════════════
    # CHUNK CONTENTS
    # ════════════════════════════════════════════════════════════════════
    print(f"\n┌{'─'*68}┐")
    print(f"│{'CHUNK CONTENTS (first {})'.format(min(max_display, len(chunks))):^68}│")
    print(f"├{'─'*68}┤")
    
    for i, (start, end) in enumerate(chunks[:max_display]):
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        length = end - start
        
        # Truncate for display
        display_text = chunk_text[:50].replace('\n', '↵')
        if len(chunk_text) > 50:
            display_text += '...'
        
        print(f"│  Chunk {i+1:>2} │ pos {start:>4}-{end:<4} │ {length:>3} tok │ {display_text:<25}│")
    
    if len(chunks) > max_display:
        print(f"│  ... and {len(chunks) - max_display} more chunks{' '*44}│")
    
    print(f"└{'─'*68}┘")
    
    # ════════════════════════════════════════════════════════════════════
    # KEPT TOKENS
    # ════════════════════════════════════════════════════════════════════
    if kept_positions:
        print(f"\n┌{'─'*68}┐")
        print(f"│{'KEPT TOKENS (first {})'.format(min(20, len(kept_positions))):^68}│")
        print(f"├{'─'*68}┤")
        
        # Show kept tokens in groups
        kept_display = kept_positions[:20]
        kept_tokens_text = [tokenizer.decode([tokens[p]]) for p in kept_display]
        
        line = "│  "
        for i, (pos, tok) in enumerate(zip(kept_display, kept_tokens_text)):
            tok_display = tok.replace('\n', '↵')[:10]
            item = f"[{pos}:'{tok_display}'] "
            if len(line) + len(item) > 67:
                print(f"{line:<68}│")
                line = "│  "
            line += item
        if len(line) > 3:
            print(f"{line:<68}│")
        
        if len(kept_positions) > 20:
            print(f"│  ... and {len(kept_positions) - 20} more kept tokens{' '*38}│")
        
        print(f"└{'─'*68}┘")
    
    # ════════════════════════════════════════════════════════════════════
    # PROBABILITY STATISTICS
    # ════════════════════════════════════════════════════════════════════
    print(f"\n┌{'─'*68}┐")
    print(f"│{'POLICY STATISTICS':^68}│")
    print(f"├{'─'*68}┤")
    
    # Get importance and threshold info
    importance = policy_output.boundary_importance[0]
    threshold = policy_output.boundary_threshold[0].item()
    
    print(f"│  Boundary probs:  mean={boundary_probs.mean().item():.3f}  std={boundary_probs.std().item():.3f}  max={boundary_probs.max().item():.3f}{' '*12}│")
    print(f"│  Keep probs:      mean={keep_probs.mean().item():.3f}  std={keep_probs.std().item():.3f}  max={keep_probs.max().item():.3f}{' '*12}│")
    print(f"│  Importance:      mean={importance.mean().item():.3f}  std={importance.std().item():.3f}  range=[{importance.min().item():.2f}, {importance.max().item():.2f}]│")
    print(f"│  Threshold:       {threshold:.3f} (positions above this → boundary){' '*19}│")
    print(f"└{'─'*68}┘")
    
    return {
        "num_chunks": num_chunks,
        "num_kept": num_kept,
        "chunk_lengths": [end - start for start, end in chunks],
        "kept_positions": kept_positions,
        "boundary_probs_mean": boundary_probs.mean().item(),
        "keep_probs_mean": keep_probs.mean().item(),
        "compression_ratio": T / (num_chunks + num_kept) if num_chunks + num_kept > 0 else 0,
    }


@torch.no_grad()
def analyze_boundaries(model, tokenizer, text: str, device: str = "cuda"):
    """Legacy boundary analysis (use analyze_chunks for detailed view)."""
    return analyze_chunks(model, tokenizer, text, device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Context Extension model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluation options
    parser.add_argument("--perplexity", action="store_true", help="Compute perplexity")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--analyze", action="store_true", help="Analyze boundary placement")
    
    # Generation options
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away,")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    
    # Test text
    parser.add_argument("--text", type=str, default=None, help="Text for perplexity/analysis")
    
    # Display options
    parser.add_argument("--max-display", type=int, default=10, help="Max chunks to display (use -1 for all)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    
    print(f"\nModel config:")
    print(f"  Pretrained: {config.pretrained_model_name}")
    print(f"  Max context: {config.max_context} (chunks + current window)")
    print(f"  Chunk dim: {config.chunk_dim}")
    
    # Default test text
    test_text = args.text or (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors "
        "of artificial beings endowed with intelligence or consciousness by master craftsmen. "
        "The seeds of modern AI were planted by philosophers who attempted to describe the process "
        "of human thinking as the mechanical manipulation of symbols. This work culminated in the "
        "invention of the programmable digital computer in the 1940s, a machine based on the abstract "
        "essence of mathematical reasoning. This device and the ideas behind it inspired a handful "
        "of scientists to begin seriously discussing the possibility of building an electronic brain."
    )
    
    # Run evaluations
    if args.perplexity or (not args.generate and not args.analyze):
        print(f"\n{'='*60}")
        print("PERPLEXITY EVALUATION")
        print(f"{'='*60}")
        result = evaluate_perplexity(model, tokenizer, test_text, args.device)
        print(f"\nPerplexity: {result['perplexity']:.2f}")
        print(f"Loss: {result['loss']:.4f}")
        print(f"Chunks used: {result['num_chunks']}")
        print(f"Kept tokens: {result['num_kept_tokens']} tokens")
    
    if args.analyze:
        max_disp = args.max_display if args.max_display >= 0 else 999999
        analyze_chunks(model, tokenizer, test_text, args.device, max_display=max_disp)
    
    if args.generate:
        generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
        )


if __name__ == "__main__":
    main()
