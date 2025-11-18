"""Data loading and preprocessing utilities for WavGPT."""

import torch
from datasets import load_dataset, load_dataset_builder


def prepare_dataset(tokenizer, block_size):
    """Load and tokenize dataset."""
    # Use a dataset that works with current HuggingFace datasets library
    # Options:
    # 1. "Skylion007/openwebtext" - direct replacement for openwebtext
    # 2. "wikitext" - smaller, faster for testing
    # 3. "bookcorpus" - another good option
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    text_column = "text"
    ds_builder = load_dataset_builder("wikitext", "wikitext-103-raw-v1")
    length = ds_builder.info.splits['train'].num_examples

    def tokenize_function(examples):
        # Filter out empty texts
        texts = [t for t in examples[text_column] if t and len(t.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_attention_mask=True,
        )
        return tokenized

    # Take a subset for faster iteration (remove .take() for full dataset)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column]
    )  # Adjust as needed

    print(f"Tokenized dataset length: {length}")

    return tokenized_dataset, length


class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """Wrapper to convert HuggingFace streaming dataset to proper PyTorch IterableDataset."""
    def __init__(self, hf_dataset, length):
        self.dataset = hf_dataset
        # randomize the dataset
        self.dataset = self.dataset.shuffle()
        self.length = length

    def __iter__(self):
        for item in self.dataset:
            # Convert lists to tensors
            yield {
                "input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"])
            }

    def __len__(self):
        return self.length


def decode_trimmed(tokenizer, ids_tensor):
    """Helper to decode tokens, trimming at pad/eos."""
    ids_list = ids_tensor.cpu().tolist()
    if tokenizer.pad_token_id in ids_list:
        first_pad = ids_list.index(tokenizer.pad_token_id)
        ids_list = ids_list[:first_pad]
    return tokenizer.decode(ids_list, skip_special_tokens=True)

