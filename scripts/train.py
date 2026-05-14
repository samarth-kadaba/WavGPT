#!/usr/bin/env python3
"""Train the CHUNKY KV-cache compressor."""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavgpt.logging_config import configure_logging  # noqa: E402
from wavgpt.models.config import CompressorConfig, TrainingConfig  # noqa: E402
from wavgpt.models.kv_extender import KVExtender  # noqa: E402
from wavgpt.training.trainer import CompressorTrainer  # noqa: E402

configure_logging(use_json=False)
logger = structlog.get_logger()


def sample_length_log_uniform(min_len: int, max_len: int) -> int:
    return int(math.exp(random.uniform(math.log(min_len), math.log(max_len))))


class FixedTextDataset(Dataset):
    """Long synthetic text repeated and windowed (used for --debug)."""

    def __init__(self, tokenizer, texts: list, max_seq_length: int = 1024, stride: int = 256):
        self.max_seq_length = max_seq_length
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(tokens) - max_seq_length + 1, stride):
                self.samples.append(tokens[start:start + max_seq_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        toks = self.samples[idx]
        return {"input_ids": torch.tensor(toks, dtype=torch.long)}


class PG19Dataset(Dataset):
    """Streaming windows from PG19. Fixed length per window."""

    def __init__(self, tokenizer, max_seq_length: int = 1024, num_samples: int = 5000,
                 stride: int = 512):
        from datasets import load_dataset

        self.max_seq_length = max_seq_length
        logger.info("loading_pg19", max_seq_length=max_seq_length, num_samples=num_samples)
        ds = load_dataset("emozilla/pg19", split="train", streaming=True)
        self.samples = []
        for item in ds:
            text = item["text"]
            if len(text) < 1000:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(tokens) - max_seq_length + 1, stride):
                self.samples.append(tokens[start:start + max_seq_length])
                if len(self.samples) >= num_samples:
                    break
            if len(self.samples) >= num_samples:
                break
        logger.info("pg19_loaded", samples=len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        toks = self.samples[idx]
        return {"input_ids": torch.tensor(toks, dtype=torch.long)}


def collate_fn(batch, pad_token_id: int = 0):
    max_len = max(item["input_ids"].size(0) for item in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--min-continuation", type=int, default=64)
    p.add_argument("--max-continuation", type=int, default=256)
    p.add_argument("--max-kv-slots", type=int, default=128)
    p.add_argument("--compress-dim", type=int, default=256)
    p.add_argument("--n-ssm-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--coverage-weight", type=float, default=0.0)
    p.add_argument("--sparsity-weight", type=float, default=0.0)
    p.add_argument("--no-gumbel", action="store_true")
    p.add_argument("--unfreeze-lm", action="store_true")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("config", model=args.model, device=device,
                max_seq_length=args.max_seq_length, max_kv_slots=args.max_kv_slots,
                compression_target=args.max_seq_length // args.max_kv_slots)

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project="kv-compression", config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = CompressorConfig(
        pretrained_model_name=args.model,
        max_kv_slots=args.max_kv_slots,
        compress_dim=args.compress_dim,
        n_ssm_layers=args.n_ssm_layers,
        coverage_loss_weight=args.coverage_weight,
        sparsity_loss_weight=args.sparsity_weight,
    )
    train_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        min_continuation_length=args.min_continuation,
        max_continuation_length=args.max_continuation,
        use_gumbel_noise=not args.no_gumbel,
        log_interval=args.log_interval,
        use_amp=not args.no_amp,
        device=device,
    )

    model = KVExtender.from_pretrained(
        args.model, config=model_config, freeze_pretrained=not args.unfreeze_lm,
    )
    logger.info("model_created", trainable_params=model.get_trainable_params())

    if args.debug:
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 400,
            "In machine learning neural networks learn representations. " * 400,
            "The history of computing spans from ancient times to today. " * 400,
        ]
        dataset = FixedTextDataset(tokenizer, texts, max_seq_length=args.max_seq_length,
                                   stride=args.stride)
    else:
        dataset = PG19Dataset(tokenizer, max_seq_length=args.max_seq_length,
                              num_samples=args.num_samples, stride=args.stride)

    if len(dataset) == 0:
        raise ValueError(f"Empty dataset; reduce max_seq_length (got {args.max_seq_length})")
    logger.info("dataset_ready", num_samples=len(dataset))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
    )

    trainer = CompressorTrainer(
        model=model, train_config=train_config, model_config=model_config,
        use_wandb=use_wandb, save_dir=args.save_dir,
    )
    trainer.train(train_loader=train_loader, num_epochs=args.epochs)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
