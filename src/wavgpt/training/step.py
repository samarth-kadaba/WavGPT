from __future__ import annotations

from typing import Dict

import torch


def train_step(
    model,
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Single training step."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels", input_ids.clone()).to(device)

    # Mask out padding in labels
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        labels = labels.masked_fill(attention_mask == 0, -100)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)

    return outputs
