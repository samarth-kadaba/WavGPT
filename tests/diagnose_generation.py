#!/usr/bin/env python3
"""Diagnostic script to check for generation bugs."""

import sys
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification (ruff: noqa: E402)
from wavgpt import InfiniteContextTransformer  # noqa: E402
from wavgpt.logging_config import configure_logging  # noqa: E402

# Configure structlog for console output
configure_logging(use_json=False)
logger = structlog.get_logger()


def main():
    checkpoint_path = "checkpoints/best_model.pt"
    logger.info("loading_checkpoint", checkpoint=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = InfiniteContextTransformer(config)

    # Handle torch.compile prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("stripping_torch_compile_prefix")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("model_loaded", device=device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The capital of France is"
    logger.info("diagnostic_start", prompt=prompt)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logger.info(
        "input_info",
        shape=list(input_ids.shape),
        tokens=tokenizer.convert_ids_to_tokens(input_ids[0]),
    )

    # Run forward pass
    with torch.no_grad():
        outputs = model.forward(input_ids)

    logits = outputs["logits"]
    logger.info(
        "logits_shape",
        actual_shape=list(logits.shape),
        expected_shape=[1, input_ids.shape[1], config.vocab_size],
    )

    # Get logits for last position (next token prediction)
    last_logits = logits[:, -1, :]  # (1, vocab_size)
    logger.info("last_position_logits_shape", shape=list(last_logits.shape))

    # Check for NaN/Inf
    logger.info(
        "logits_stats",
        min=last_logits.min().item(),
        max=last_logits.max().item(),
        mean=last_logits.mean().item(),
        std=last_logits.std().item(),
        has_nan=torch.isnan(last_logits).any().item(),
        has_inf=torch.isinf(last_logits).any().item(),
    )

    # Apply softmax to get probabilities
    probs = F.softmax(last_logits, dim=-1)

    logger.info(
        "probability_stats",
        min=probs.min().item(),
        max=probs.max().item(),
        sum=probs.sum().item(),
        expected_sum=1.0,
    )

    # Get top-20 tokens
    top_k = 20
    top_probs, top_indices = torch.topk(probs[0], top_k)

    top_tokens = []
    cumulative = 0.0
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx.item()])
        token_repr = repr(token)
        logit = last_logits[0, idx].item()
        cumulative += prob.item()
        top_tokens.append(
            {
                "rank": i + 1,
                "token": token_repr,
                "prob": prob.item(),
                "logit": logit,
            }
        )

    logger.info(
        "top_tokens",
        top_k=top_k,
        tokens=top_tokens,
        cumulative_prob=cumulative,
    )

    # Check if distribution is too peaked
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_entropy = torch.log(torch.tensor(config.vocab_size)).item()
    normalized_entropy = entropy / max_entropy

    logger.info(
        "entropy_stats",
        entropy=entropy,
        max_entropy=max_entropy,
        normalized_entropy=normalized_entropy,
    )

    if top_probs[0].item() > 0.5:
        logger.warning(
            "high_top_token_prob",
            prob=top_probs[0].item(),
            message="Top token has >50% probability - distribution is very peaked!",
        )
    if entropy < 1.0:
        logger.warning(
            "low_entropy",
            entropy=entropy,
            message="Very low entropy - model is almost deterministic!",
        )

    # Test greedy vs sampled
    samples = []
    for i in range(5):
        sampled = torch.multinomial(probs, num_samples=1)
        token = tokenizer.decode([sampled[0, 0].item()])
        prob = probs[0, sampled[0, 0]].item()
        samples.append({"sample": i + 1, "token": repr(token), "prob": prob})

    logger.info("sampling_test", temperature=1.0, samples=samples)

    # Check different positions
    position_logits = []
    for pos in range(min(5, input_ids.shape[1])):
        pos_logits = logits[:, pos, :]
        pos_probs = F.softmax(pos_logits, dim=-1)
        top_prob, top_idx = pos_probs[0].max(dim=0)
        token = tokenizer.decode([top_idx.item()])
        position_logits.append(
            {
                "position": pos,
                "top_token": repr(token),
                "prob": top_prob.item(),
            }
        )

    logger.info("position_logits", positions=position_logits)
    logger.info("diagnostic_complete")


if __name__ == "__main__":
    main()
