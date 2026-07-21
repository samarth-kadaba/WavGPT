"""Corpora with document packing, cross-document masking, and loss masking."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

FINEWEB = "HuggingFaceFW/fineweb-edu"


def _open_stream(corpus: "Corpus"):
    """Local parquet if CHUNKY_DATA_DIR is set (robust, no flaky HTTP), else hub."""
    from datasets import load_dataset

    data_dir = os.environ.get("CHUNKY_DATA_DIR")
    if data_dir and corpus.path == FINEWEB:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
        if files:
            return load_dataset("parquet", data_files=files, split="train", streaming=True)
    return load_dataset(corpus.path, name=corpus.name, split=corpus.split, streaming=True)


@dataclass(frozen=True)
class Corpus:
    path: str
    name: Optional[str] = None
    split: str = "train"
    text_key: str = "text"


CORPORA = {
    "fineweb-edu": Corpus("HuggingFaceFW/fineweb-edu", name="sample-10BT"),
    "pg19": Corpus("emozilla/pg19"),  # long books, for long-context validation
    "wikitext": Corpus("Salesforce/wikitext", name="wikitext-103-raw-v1", split="test"),
    "c4": Corpus("allenai/c4", name="en", split="validation"),
    "books3": Corpus("SaylorTwift/the_pile_books3_minus_gutenberg"),
    "govreport": Corpus("ccdv/govreport-summarization", split="test", text_key="report"),
}


def pack_documents(docs: Iterable[List[int]], seq_length: int, eos_id: int, max_sequences: int):
    """Pack tokenized docs into fixed EOS-delimited windows.

    Returns parallel lists of (input_ids, segment_ids); segment ids mark each
    token's source document within a window for cross-document masking.
    """
    ids: list[torch.Tensor] = []
    segs: list[torch.Tensor] = []
    tok_buf: List[int] = []
    seg_buf: List[int] = []
    seg = 0
    for doc in docs:
        tok_buf.extend(doc)
        tok_buf.append(eos_id)
        seg_buf.extend([seg] * (len(doc) + 1))
        seg += 1
        while len(tok_buf) >= seq_length:
            ids.append(torch.tensor(tok_buf[:seq_length]))
            segs.append(torch.tensor(seg_buf[:seq_length]))
            del tok_buf[:seq_length]
            del seg_buf[:seq_length]
            if len(ids) >= max_sequences:
                return ids, segs
    return ids, segs


def _token_docs(stream, tokenizer, text_key: str, min_chars: int) -> Iterator[List[int]]:
    for item in stream:
        text = item.get(text_key)
        if text and len(text) >= min_chars:
            yield tokenizer.encode(text, add_special_tokens=False)


class PackedCorpus(Dataset):
    """Materialized packed windows; for validation and OOD evaluation."""

    def __init__(self, tokenizer, corpus: Corpus, seq_length: int, num_sequences: int,
                 skip_docs: int = 0, min_doc_chars: int = 200):
        eos_id = tokenizer.eos_token_id
        stream = _open_stream(corpus)
        if skip_docs:
            stream = stream.skip(skip_docs)
        docs = _token_docs(stream, tokenizer, corpus.text_key, min_doc_chars)
        self.ids, self.segs = pack_documents(docs, seq_length, eos_id, num_sequences)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {"input_ids": self.ids[idx], "seg_ids": self.segs[idx]}


class StreamingPacked(IterableDataset):
    """On-the-fly packing for large-scale training; loops the source stream."""

    def __init__(self, tokenizer, corpus: Corpus, seq_length: int, skip_docs: int = 0,
                 min_doc_chars: int = 200, rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.seq_length = seq_length
        self.skip_docs = skip_docs
        self.min_doc_chars = min_doc_chars
        self.rank = rank
        self.world_size = world_size
        self.eos_id = tokenizer.eos_token_id
        self.docs_consumed = 0

    def __iter__(self):
        from datasets.distributed import split_dataset_by_node

        tok_buf: List[int] = []
        seg_buf: List[int] = []
        seg = 0
        while True:
            stream = _open_stream(self.corpus)
            if self.skip_docs:
                stream = stream.skip(self.skip_docs)
            if self.world_size > 1:
                stream = split_dataset_by_node(stream, self.rank, self.world_size)
            for doc in _token_docs(stream, self.tokenizer, self.corpus.text_key, self.min_doc_chars):
                self.docs_consumed += 1
                tok_buf.extend(doc)
                tok_buf.append(self.eos_id)
                seg_buf.extend([seg] * (len(doc) + 1))
                seg += 1
                while len(tok_buf) >= self.seq_length:
                    yield {
                        "input_ids": torch.tensor(tok_buf[: self.seq_length]),
                        "seg_ids": torch.tensor(seg_buf[: self.seq_length]),
                    }
                    del tok_buf[: self.seq_length]
                    del seg_buf[: self.seq_length]


def held_out_val(tokenizer, corpus: Corpus, seq_length: int, n_docs: int, max_windows: int = 200):
    """Pack the first n_docs documents into up to max_windows validation windows."""
    stream = _open_stream(corpus).take(n_docs)
    docs = _token_docs(stream, tokenizer, corpus.text_key, min_chars=200)
    ids, segs = pack_documents(docs, seq_length, tokenizer.eos_token_id, max_sequences=max_windows)
    return [{"input_ids": i, "seg_ids": s} for i, s in zip(ids, segs)]


def sliding_window_mask(seq_len: int, window: int, device) -> torch.Tensor:
    """Additive causal mask (1, 1, T, T) restricting each query to the last `window` keys."""
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    keep = (j <= i) & (i - j < window)
    mask = torch.zeros(seq_len, seq_len, device=device)
    return mask.masked_fill(~keep, float("-inf")).view(1, 1, seq_len, seq_len)


def masked_labels(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Next-token labels with the token following an EOS set to -100."""
    labels = input_ids.clone()
    after_eos = torch.zeros_like(labels, dtype=torch.bool)
    after_eos[:, 1:] = input_ids[:, :-1] == eos_id
    labels[after_eos] = -100
    return labels
