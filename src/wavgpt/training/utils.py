from __future__ import annotations

from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from wavgpt.config import LEARNING_RATE, WARMUP_RATIO, DEVICE


def create_optimizer(model, lr: float = LEARNING_RATE, weight_decay: float = 0.01):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def create_scheduler(optimizer, num_epochs: int, steps_per_epoch: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    path: str,
    total_tokens: int = 0,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "total_tokens": total_tokens,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    import structlog

    logger = structlog.get_logger()
    logger.info("saved_checkpoint", path=path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path: str,
    device: str = DEVICE,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    import structlog

    logger = structlog.get_logger()
    logger.info("loaded_checkpoint", path=path)
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "total_tokens": checkpoint.get("total_tokens", 0),
    }
