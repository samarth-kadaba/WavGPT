"""
Data loading utilities for Infinite Context Transformer.

Supports multiple datasets:
- The Pile: Diverse 886GB dataset (streaming) - long AND short context
- WikiText-103: Wikipedia articles (fallback)
- PG-19: Full books for very long context testing

Supports train/val/test splits for streaming datasets using hash-based partitioning.
"""

import hashlib
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Iterator, Dict, Literal

from wavgpt.data.config import DATASET_CONFIGS

SplitType = Literal["train", "val", "test"]


def _hash_text(text: str) -> float:
    """
    Hash text to a float in [0, 1) for deterministic split assignment.
    This ensures the same document always goes to the same split.
    """
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text corpora.

    Supports multiple datasets with automatic length filtering
    and optional document concatenation for long-context training.

    Supports train/val/test splits via hash-based partitioning.
    """

    def __init__(
        self,
        dataset_name: str = "pile",
        tokenizer=None,
        max_length: int = 4096,
        min_length: int = 64,
        concat_documents: bool = True,
        buffer_size: int = 10000,
        variable_length: bool = True,  # NEW: sample variable lengths
        split: SplitType = "train",  # Which split to use
        val_ratio: float = 0.05,  # Fraction for validation
        test_ratio: float = 0.05,  # Fraction for testing
    ):
        """
        Args:
            dataset_name: Name from DATASET_CONFIGS
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            min_length: Minimum sequence length (skip shorter)
            concat_documents: Whether to concatenate short docs
            buffer_size: Shuffle buffer size
            variable_length: If True, randomly sample lengths between min and max
            split: Which split to use ("train", "val", or "test")
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.concat_documents = concat_documents
        self.buffer_size = buffer_size
        self.variable_length = variable_length
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Get dataset config
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Choose from: {list(DATASET_CONFIGS.keys())}"
            )

        self.config = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self._dataset = None

    def _load_dataset(self):
        """Lazily load dataset."""
        if self._dataset is None:
            load_kwargs = {
                "path": self.config["path"],
                "split": self.config["split"],
                "streaming": True,
            }
            if "name" in self.config:
                load_kwargs["name"] = self.config["name"]

            try:
                self._dataset = load_dataset(**load_kwargs)
            except Exception as e:
                import structlog

                logger = structlog.get_logger()
                logger.warning(
                    "dataset_load_failed",
                    path=self.config["path"],
                    error=str(e),
                    fallback="wikitext",
                )
                self._dataset = load_dataset(
                    "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
                )
                self.config["text_field"] = "text"

            # Only shuffle for training split
            if self.split == "train":
                self._dataset = self._dataset.shuffle(seed=42, buffer_size=self.buffer_size)

        return self._dataset

    def _belongs_to_split(self, text: str) -> bool:
        """
        Determine if a document belongs to the current split.

        Uses hash-based partitioning for deterministic splits:
        - [0, test_ratio): test
        - [test_ratio, test_ratio + val_ratio): val
        - [test_ratio + val_ratio, 1.0): train
        """
        h = _hash_text(text)

        if self.split == "test":
            return h < self.test_ratio
        elif self.split == "val":
            return self.test_ratio <= h < (self.test_ratio + self.val_ratio)
        else:  # train
            return h >= (self.test_ratio + self.val_ratio)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = self._load_dataset()
        text_field = self.config["text_field"]

        if self.concat_documents:
            yield from self._iter_concatenated(dataset, text_field)
        else:
            yield from self._iter_single(dataset, text_field)

    def _iter_single(self, dataset, text_field: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over single documents."""
        for item in dataset:
            text = item.get(text_field, "")
            if not text or len(text.strip()) < 10:
                continue

            # Check if this document belongs to our split
            if not self._belongs_to_split(text):
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )

            seq_len = tokens["input_ids"].size(1)
            if seq_len < self.min_length:
                continue

            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

    def _iter_concatenated(self, dataset, text_field: str) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Concatenate documents to fill sequences.

        This creates long sequences from multiple short documents,
        allowing the model to learn cross-document patterns.

        With variable_length=True, randomly samples sequence lengths
        to help the model generalize across different context sizes.
        """
        import random

        buffer_ids = []
        buffer_mask = []

        for item in dataset:
            text = item.get(text_field, "")
            if not text or len(text.strip()) < 10:
                continue

            # Check if this document belongs to our split
            if not self._belongs_to_split(text):
                continue

            # Tokenize this document
            tokens = self.tokenizer(
                text,
                truncation=False,  # Don't truncate - we'll handle it
                padding=False,
                add_special_tokens=True,
            )

            doc_ids = tokens["input_ids"]
            doc_mask = [1] * len(doc_ids)

            # Add to buffer
            buffer_ids.extend(doc_ids)
            buffer_mask.extend(doc_mask)

            # Yield complete sequences
            while len(buffer_ids) >= self.max_length:
                # Variable length: sample between min_length and max_length
                if self.variable_length:
                    # Bias toward longer sequences (more informative)
                    # Use log-uniform distribution for better coverage of all scales
                    import math

                    log_min = max(
                        1, int(self.min_length * 4)
                    )  # At least 4x min_length for useful training
                    log_max = self.max_length
                    # Actual log-uniform sampling
                    target_len = int(math.exp(random.uniform(math.log(log_min), math.log(log_max))))
                    # Round to multiple of 64 for efficiency
                    target_len = min(((target_len + 63) // 64) * 64, self.max_length)
                else:
                    target_len = self.max_length

                yield {
                    "input_ids": torch.tensor(buffer_ids[:target_len], dtype=torch.long),
                    "attention_mask": torch.tensor(buffer_mask[:target_len], dtype=torch.long),
                }
                # Keep some overlap for context continuity
                overlap = target_len // 4
                buffer_ids = buffer_ids[target_len - overlap :]
                buffer_mask = buffer_mask[target_len - overlap :]

        # Yield remaining if long enough
        if len(buffer_ids) >= self.min_length:
            final_len = min(len(buffer_ids), self.max_length)
            yield {
                "input_ids": torch.tensor(buffer_ids[:final_len], dtype=torch.long),
                "attention_mask": torch.tensor(buffer_mask[:final_len], dtype=torch.long),
            }
