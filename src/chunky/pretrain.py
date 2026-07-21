"""Pretrainer for the Standard baseline and the streaming compressor."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

POS_BINS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30)]

from chunky.compressor import CompressorConfig
from chunky.data import (
    CORPORA,
    StreamingPacked,
    held_out_val,
    masked_labels,
    sliding_window_mask,
)
from chunky.model import SCALES, Transformer
from chunky.streaming import CompressedTransformer


@dataclass
class TrainConfig:
    scale: str = "xs"
    variant: str = "standard"
    dataset: str = "fineweb-edu"
    seq_len: int = 4096
    budget: int = 512
    chunk_size: int = 512
    total_tokens: int = 20_000_000_000
    global_batch_tokens: int = 393_216
    micro_batch: int = 8
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.02
    decay_frac: float = 0.10
    grad_clip: float = 1.0
    val_docs: int = 40
    val_every: int = 500
    val_batches: int = 20
    val_len: int = 16384        # single long val sequence; loss is binned by context length
    val_corpus: str = "pg19"    # long books so far-context bins populate
    log_every: int = 20
    ckpt_every: int = 2000
    out_dir: str = "checkpoints"
    wandb_project: str = "chunky"
    run_tag: str = "v3"
    seed: int = 0
    grad_accum: int = field(init=False)

    def __post_init__(self):
        self.grad_accum = max(1, self.global_batch_tokens // (self.micro_batch * self.seq_len))


def build_model(cfg: TrainConfig) -> Transformer:
    model_cfg = SCALES[cfg.scale]
    if cfg.variant in ("standard", "window"):
        return Transformer(model_cfg)
    comp = CompressorConfig(d_model=model_cfg.d_model, n_heads=model_cfg.n_heads, max_slots=cfg.budget)
    return CompressedTransformer(model_cfg, comp, chunk_size=cfg.chunk_size)


def wsd_lr(step: int, total_steps: int, cfg: TrainConfig) -> float:
    warmup, decay = int(cfg.warmup_frac * total_steps), int(cfg.decay_frac * total_steps)
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    if step > total_steps - decay:
        return cfg.lr * max(0.0, (total_steps - step) / decay)
    return cfg.lr


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


def save_ckpt(path: Path, model, opt, step: int, docs_trained: int, cfg: TrainConfig) -> None:
    total_steps = cfg.total_tokens // cfg.global_batch_tokens
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "lr": wsd_lr(step, total_steps, cfg),  # recomputed from step on resume; saved for visibility
        "docs_trained": docs_trained,
        "rng": torch.get_rng_state(),
        "cfg": vars(cfg),
    }, tmp)
    tmp.replace(path)


def load_ckpt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def within_doc_offset(seg_ids: torch.Tensor) -> torch.Tensor:
    """Same-document context length available at each position (tokens since doc start)."""
    B, T = seg_ids.shape
    idx = torch.arange(T, device=seg_ids.device).expand(B, T)
    is_start = torch.ones_like(seg_ids, dtype=torch.bool)
    is_start[:, 1:] = seg_ids[:, 1:] != seg_ids[:, :-1]
    start = torch.cummax(torch.where(is_start, idx, torch.zeros_like(idx)), dim=1).values
    return idx - start


@torch.no_grad()
def evaluate(model, val, device, eos_id, attn_mask, batch_size, max_batches):
    """Returns (overall_loss, {bin: loss}) with loss binned by same-doc context length."""
    model.eval()
    bin_sum = [0.0] * len(POS_BINS)
    bin_cnt = [0] * len(POS_BINS)
    for start in range(0, min(len(val), max_batches * batch_size), batch_size):
        batch = val[start:start + batch_size]
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        seg_ids = torch.stack([b["seg_ids"] for b in batch]).to(device)
        labels = masked_labels(input_ids, eos_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits, _ = model(input_ids, attn_mask=attn_mask)
        tok_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none",
        )
        offset = within_doc_offset(seg_ids)[:, :-1].reshape(-1)
        valid = labels[:, 1:].reshape(-1) != -100
        for k, (lo, hi) in enumerate(POS_BINS):
            m = valid & (offset >= lo) & (offset < hi)
            bin_sum[k] += float(tok_loss[m].sum())
            bin_cnt[k] += int(m.sum())
    model.train()
    def _label(lo, hi):
        return f"{lo}_{hi if hi < (1 << 30) else 'inf'}"

    bins = {_label(lo, hi): bin_sum[k] / bin_cnt[k] for k, (lo, hi) in enumerate(POS_BINS) if bin_cnt[k]}
    overall = sum(bin_sum) / max(sum(bin_cnt), 1)
    return overall, bins


def train(cfg: TrainConfig, tokenizer, log=print) -> None:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    cfg.grad_accum = max(1, cfg.global_batch_tokens // (world * cfg.micro_batch * cfg.seq_len))
    if ddp:
        torch.distributed.init_process_group("nccl")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed + rank)

    model = build_model(cfg).to(device)
    opt = make_optimizer(model, cfg)
    total_steps = cfg.total_tokens // cfg.global_batch_tokens
    eos_id = tokenizer.eos_token_id
    corpus = CORPORA[cfg.dataset]
    # window: sliding-window attention capped at the same budget as ours; else full/streaming.
    attn_mask = sliding_window_mask(cfg.seq_len, cfg.budget, device) if cfg.variant == "window" else None

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / f"{cfg.variant}_{cfg.scale}_{cfg.run_tag}_latest.pt"

    ckpt = load_ckpt(latest)
    start_step, docs_prev = 0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        torch.set_rng_state(ckpt["rng"])
        start_step, docs_prev = ckpt["step"] + 1, ckpt["docs_trained"]
        log(f"resumed from step {ckpt['step']} (docs_trained={docs_prev})")

    engine = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if ddp else model

    # Val uses a separate corpus (pg19), so no train/val overlap and no skip needed.
    # We also do NOT skip docs_prev on resume (skipping millions of docs stalls startup).
    dataset = StreamingPacked(tokenizer, corpus, cfg.seq_len, skip_docs=0,
                              rank=rank, world_size=world)
    batches = iter(DataLoader(dataset, batch_size=cfg.micro_batch, num_workers=0, pin_memory=True))

    run = None
    val = []
    if rank == 0:
        try:
            val = held_out_val(tokenizer, CORPORA[cfg.val_corpus], cfg.val_len, cfg.val_docs)
        except Exception as e:
            log(f"val set unavailable ({e}); continuing without validation")
        log(f"model={cfg.variant}/{cfg.scale} params={model.num_params():,} steps={total_steps} "
            f"val_windows={len(val)}@{cfg.val_len}")
        try:
            import wandb
            name = f"{cfg.variant}-{cfg.scale}-{cfg.run_tag}"
            run = wandb.init(project=cfg.wandb_project, name=name, config=vars(cfg),
                             resume="allow", id=name)
        except Exception as e:
            log(f"wandb disabled: {e}")

    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = wsd_lr(step, total_steps, cfg)

        loss_sum = 0.0
        for micro in range(cfg.grad_accum):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = masked_labels(input_ids, eos_id)
            sync = (micro == cfg.grad_accum - 1) or not ddp
            with (engine.no_sync() if (ddp and not sync) else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    _, loss = engine(input_ids, attn_mask=attn_mask, labels=labels)
                    loss = loss / cfg.grad_accum
                loss.backward()
            loss_sum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if rank == 0 and step % cfg.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            log(f"step={step}/{total_steps} lr={lr:.2e} loss={loss_sum:.4f} ppl={math.exp(min(loss_sum, 20)):.2f}")
            if run:
                run.log({"train/loss": loss_sum, "train/ppl": math.exp(min(loss_sum, 20)), "lr": lr}, step=step)

        if rank == 0 and val and step > 0 and step % cfg.val_every == 0:
            vmask = sliding_window_mask(cfg.val_len, cfg.budget, device) if cfg.variant == "window" else None
            try:
                vloss, bins = evaluate(model, val, device, eos_id, vmask, 1, cfg.val_batches)
            except RuntimeError as e:  # e.g. OOM on very long val; don't kill training
                torch.cuda.empty_cache()
                log(f"  val skipped: {e}")
            else:
                log(f"  val nll={vloss:.4f} ppl={math.exp(min(vloss, 20)):.2f} "
                    + " ".join(f"[{b}]={v:.3f}" for b, v in bins.items()))
                if run:
                    metrics = {"val/nll": vloss, "val/ppl": math.exp(min(vloss, 20))}
                    for b, v in bins.items():
                        metrics[f"val/nll_ctx_{b}"] = v
                        metrics[f"val/ppl_ctx_{b}"] = math.exp(min(v, 20))
                    run.log(metrics, step=step)

        if rank == 0 and step > 0 and step % cfg.ckpt_every == 0:
            docs_trained = docs_prev + world * dataset.docs_consumed
            save_ckpt(latest, model, opt, step, docs_trained, cfg)
            log(f"  saved checkpoint step={step} docs_trained={docs_trained}")

    if rank == 0 and total_steps > start_step:  # only if we actually trained
        save_ckpt(latest, model, opt, total_steps - 1, docs_prev + world * dataset.docs_consumed, cfg)
    if run:
        run.finish()
    if ddp:
        torch.distributed.destroy_process_group()
