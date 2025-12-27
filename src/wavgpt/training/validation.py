from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from wavgpt.config import DEVICE


@torch.no_grad()
def validate(
    model,
    val_loader,
    device: str = DEVICE,
    max_batches: int = 50,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Run validation and compute metrics.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run on
        max_batches: Maximum number of batches to validate (for speed)
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with validation metrics (val_loss, val_lm_loss, etc.)
    """
    model.eval()
    use_amp = use_amp and device == "cuda"

    total_loss = 0.0
    total_lm_loss = 0.0
    total_comp_loss = 0.0
    total_chunks = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids.clone()).to(device)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                labels = labels.masked_fill(attention_mask == 0, -100)

            # Forward pass with optional mixed precision
            if use_amp:
                with autocast(device_type="cuda"):
                    outputs = model(input_ids=input_ids, labels=labels)
            else:
                outputs = model(input_ids=input_ids, labels=labels)

            total_loss += outputs["loss"].item()

            if outputs.get("lm_loss") is not None:
                total_lm_loss += outputs["lm_loss"].item()
            else:
                total_lm_loss += outputs["loss"].item()

            if outputs.get("compression_loss") is not None:
                total_comp_loss += outputs["compression_loss"].item()

            n_chunks = outputs["n_chunks"]
            if isinstance(n_chunks, torch.Tensor):
                total_chunks += n_chunks.item()
            elif isinstance(n_chunks, list):
                total_chunks += sum(n_chunks) / len(n_chunks)
            else:
                total_chunks += n_chunks

            num_batches += 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    model.train()

    if num_batches == 0:
        return {"val_loss": float("inf"), "val_lm_loss": float("inf")}

    return {
        "val_loss": total_loss / num_batches,
        "val_lm_loss": total_lm_loss / num_batches,
        "val_comp_loss": total_comp_loss / num_batches,
        "val_avg_chunks": total_chunks / num_batches,
    }
