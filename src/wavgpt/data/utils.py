from typing import Dict, Literal

import torch
from torch.utils.data import DataLoader

from wavgpt.data.config import DATASET_CONFIGS

SplitType = Literal["train", "val", "test"]


def create_collate_fn(max_length: int):
    """Create collate function with dynamic padding."""

    def collate_fn(batch):
        # Filter empty items
        batch = [b for b in batch if len(b["input_ids"]) > 0]
        if not batch:
            return {
                "input_ids": torch.zeros(1, 64, dtype=torch.long),
                "attention_mask": torch.zeros(1, 64, dtype=torch.long),
                "labels": torch.full((1, 64), -100, dtype=torch.long),
            }

        # Pad to max length in batch (dynamic padding)
        max_len = min(max(len(b["input_ids"]) for b in batch), max_length)

        input_ids = []
        attention_mask = []

        for b in batch:
            ids = b["input_ids"][:max_len]
            mask = b["attention_mask"][:max_len]

            # Pad if needed
            if len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

            input_ids.append(ids)
            attention_mask.append(mask)

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)

        # Labels are input_ids with padding masked
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


def create_dataloader(
    tokenizer,
    dataset_name: str = "pile",
    batch_size: int = 2,
    max_length: int = 4096,
    min_length: int = 64,
    concat_documents: bool = True,
    variable_length: bool = True,
    num_workers: int = 4,
    split: SplitType = "train",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> DataLoader:
    """
    Create dataloader for a specific split.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Dataset to use (pile, wikitext, pg19, redpajama)
        batch_size: Batch size
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        concat_documents: Concatenate short docs to fill context
        variable_length: Sample random lengths between min and max
        num_workers: DataLoader workers
        split: Which split to create ("train", "val", or "test")
        val_ratio: Fraction of data for validation (default 5%)
        test_ratio: Fraction of data for testing (default 5%)

    Returns:
        DataLoader for the specified split
    """
    import structlog

    logger = structlog.get_logger()
    logger.info(
        "loading_dataset",
        dataset=dataset_name,
        split=split,
        description=DATASET_CONFIGS[dataset_name]["description"],
        max_length=max_length,
        min_length=min_length,
        concat_documents=concat_documents,
        variable_length=variable_length,
        train_ratio=f"{1 - val_ratio - test_ratio:.0%}",
        val_ratio=f"{val_ratio:.0%}",
        test_ratio=f"{test_ratio:.0%}",
    )

    # Lazy import to avoid circular dependency
    from wavgpt.data.data import StreamingTextDataset

    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        concat_documents=concat_documents,
        variable_length=variable_length,
        split=split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    collate_fn = create_collate_fn(max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_dataloaders(
    tokenizer,
    dataset_name: str = "pile",
    batch_size: int = 2,
    max_length: int = 4096,
    min_length: int = 64,
    concat_documents: bool = True,
    variable_length: bool = True,
    num_workers: int = 4,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    include_test: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and optionally test dataloaders.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Dataset to use
        batch_size: Batch size
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        concat_documents: Concatenate short docs to fill context
        variable_length: Sample random lengths between min and max
        num_workers: DataLoader workers
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        include_test: Whether to create test dataloader

    Returns:
        Dictionary with "train", "val", and optionally "test" dataloaders
    """
    common_kwargs = {
        "tokenizer": tokenizer,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "min_length": min_length,
        "concat_documents": concat_documents,
        "variable_length": variable_length,
        "num_workers": num_workers,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    loaders = {
        "train": create_dataloader(**common_kwargs, split="train"),
        "val": create_dataloader(**common_kwargs, split="val"),
    }

    if include_test:
        loaders["test"] = create_dataloader(**common_kwargs, split="test")

    return loaders


def list_datasets():
    """List available datasets."""
    import structlog

    logger = structlog.get_logger()
    logger.info(
        "available_datasets",
        datasets={name: config["description"] for name, config in DATASET_CONFIGS.items()},
    )


def test_dataset(dataset_name: str = "pile", num_samples: int = 3):
    """Test a dataset by loading a few samples."""
    import structlog
    from transformers import AutoTokenizer

    logger = structlog.get_logger()
    logger.info("testing_dataset", dataset=dataset_name, num_samples=num_samples)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Lazy import to avoid circular dependency
    from wavgpt.data.data import StreamingTextDataset

    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=2048,
        min_length=64,
        concat_documents=True,
    )

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        seq_len = len(sample["input_ids"])
        text_preview = tokenizer.decode(sample["input_ids"][:100])

        logger.info(
            "dataset_sample",
            sample=i + 1,
            length=seq_len,
            preview=text_preview[:200],
        )

    logger.info("dataset_test_complete")
