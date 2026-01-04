#!/usr/bin/env python3
"""Quick test script to check if the model generates coherent text."""

import argparse
import sys
from pathlib import Path

import structlog
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification (ruff: noqa: E402)
from wavgpt import InfiniteContextTransformer  # noqa: E402
from wavgpt.logging_config import configure_logging  # noqa: E402

# Configure structlog for console output
configure_logging(use_json=False)
logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Test generation from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    args = parser.parse_args()

    # Load checkpoint
    logger.info("loading_checkpoint", checkpoint=args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Create model
    model = InfiniteContextTransformer(config)

    # Handle torch.compile prefix in state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("stripping_torch_compile_prefix")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(
        "model_loaded",
        device=device,
        parameters=model.get_num_params(),
        step=checkpoint.get("step", "N/A"),
        tokens_millions=checkpoint.get("total_tokens", 0) / 1e6,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant kingdom",
            "The capital of France is",
            "def fibonacci(n):",
            "In the year 2050, humans",
        ]

    logger.info(
        "generation_test_start",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # Uses full forward pass (matches training exactly)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info("generated_text", prompt=prompt, generated=generated)

    logger.info("boundary_analysis_start")

    # Analyze boundaries on a sample
    test_text = "The quick brown fox jumps over the lazy dog. Meanwhile, in a galaxy far away, scientists discovered a new planet."
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        x = model.token_embed(input_ids)
        x = model.embed_dropout(x)
        # boundary_detector now returns 6 values (removed distill_loss)
        (
            boundary_probs,
            boundary_decisions,
            ssm_output,
            expected_chunks,
            entropy_loss,
            sparsity_loss,
        ) = model.boundary_detector(x)

    # Show boundary probabilities
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    probs = boundary_probs[0].cpu().numpy()

    boundary_info = []
    for tok, prob in zip(tokens[1:], probs[1:]):
        is_boundary = prob > 0.5
        boundary_info.append(
            {
                "token": tok,
                "prob": float(prob),
                "is_boundary": bool(is_boundary),
            }
        )

    logger.info(
        "boundary_analysis",
        mean_probability=float(probs.mean()),
        boundaries_detected=int((probs > 0.5).sum()),
        total_positions=len(probs),
        boundary_details=boundary_info[:20],  # Limit to first 20 for readability
    )

    logger.info("generation_test_complete")


if __name__ == "__main__":
    main()
