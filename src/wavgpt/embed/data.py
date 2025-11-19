"""Datasets for evaluating embeddings from a trained WavGPT model."""

import torch
from datasets import load_dataset, load_dataset_builder

from wavgpt.config import DEVICE


def prepare_dataset(tokenizer, block_size):
    """Load and tokenize dataset."""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    text_column = "text"
    ds_builder = load_dataset_builder("wikitext", "wikitext-103-raw-v1")
    length = ds_builder.info.splits['train'].num_examples