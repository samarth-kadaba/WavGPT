#!/usr/bin/env python3
"""
Evaluation script for Infinite Context Transformer.

Tests the model's ability to:
1. Detect meaningful chunk boundaries
2. Handle long sequences efficiently
3. Generate coherent text

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint path/to/model.pt
    python scripts/evaluate.py --generate --prompt "Once upon a time"
"""

import argparse
import sys
from pathlib import Path

import structlog
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification (ruff: noqa: E402)
from wavgpt import InfiniteContextConfig, InfiniteContextTransformer, DEVICE  # noqa: E402
from wavgpt.logging_config import configure_logging  # noqa: E402

# Configure structlog for console output
configure_logging(use_json=False)
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Infinite Context Transformer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Model hidden size (if no checkpoint)"
    )
    parser.add_argument("--generate", action="store_true", help="Run text generation")
    parser.add_argument(
        "--prompt", type=str, default="The quick brown fox", help="Generation prompt"
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument(
        "--efficient", action="store_true", help="Use efficient generation with state caching"
    )
    return parser.parse_args()


def compute_perplexity(model, tokenizer, texts, device, max_length=512):
    """Compute perplexity on a list of texts."""
    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            tokens = tokenizer.encode(
                text, return_tensors="pt", max_length=max_length, truncation=True
            )
            tokens = tokens.to(device)

            if tokens.size(1) < 2:
                continue

            labels = tokens.clone()
            labels[:, 0] = -100

            outputs = model(input_ids=tokens, labels=labels)
            loss = outputs["loss"]

            if loss is not None:
                total_loss += loss.item() * (tokens.size(1) - 1)
                total_tokens += tokens.size(1) - 1

    if total_tokens == 0:
        return float("inf")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def analyze_chunks(model, tokenizer, text, device, max_length=1024):
    """Analyze detected chunk boundaries for a text."""
    tokens = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
    tokens = tokens.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=tokens)

    n_chunks = outputs["n_chunks"]  # This is an int, not a list
    chunk_ranges = outputs["chunk_ranges"][0]
    boundary_probs = outputs["boundary_probs"]

    return {
        "n_chunks": n_chunks,
        "chunk_ranges": chunk_ranges,
        "boundary_probs": boundary_probs,
        "tokens": tokens,
    }


def generate_text(model, tokenizer, prompt, device, max_new_tokens=100, efficient=False):
    """Generate text from a prompt."""
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if efficient:
        generated = model.generate_efficient(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
    else:
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    args = parse_args()

    logger.info("evaluation_start", device=DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create/load model
    if args.checkpoint:
        logger.info("loading_checkpoint", checkpoint=args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        config = checkpoint.get("config")
        if config is None:
            config = InfiniteContextConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=args.hidden_size,
            )
        model = InfiniteContextTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.info("creating_new_model")
        config = InfiniteContextConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
        )
        model = InfiniteContextTransformer(config)

    model = model.to(DEVICE)
    model.eval()

    logger.info("model_loaded", parameters=model.get_num_params())

    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "In machine learning, neural networks learn representations of data.",
        "Climate change is one of the most pressing challenges facing humanity.",
        "The history of computing spans from ancient abacuses to quantum computers.",
    ]

    # Compute perplexity
    logger.info("perplexity_evaluation")
    ppl = compute_perplexity(model, tokenizer, test_texts, DEVICE)
    logger.info("perplexity_result", perplexity=ppl)

    # Analyze chunks
    logger.info("chunk_analysis")
    for text in test_texts[:2]:
        result = analyze_chunks(model, tokenizer, text, DEVICE)
        logger.info(
            "chunk_analysis_result",
            text_preview=text[:50],
            n_chunks=result["n_chunks"],
            chunk_ranges=result["chunk_ranges"][:5],
        )

    # Boundary parameters
    logger.info("boundary_parameters")
    detector = model.boundary_detector
    logger.info(
        "boundary_params",
        decision_rule="surprisal-based: boundary when likelihood decreases",
    )

    # Generation
    if args.generate:
        logger.info("text_generation", prompt=args.prompt, efficient=args.efficient)
        generated = generate_text(
            model, tokenizer, args.prompt, DEVICE, args.max_tokens, efficient=args.efficient
        )
        logger.info("generated_text", text=generated)

    # Summary
    logger.info(
        "evaluation_summary",
        perplexity=ppl,
        parameters=model.get_num_params(),
    )


if __name__ == "__main__":
    main()
