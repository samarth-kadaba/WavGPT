"""Supervised LM-loss trainer for the KV compressor."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Optional, Dict

import structlog
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from wavgpt.models.config import CompressorConfig, TrainingConfig
from wavgpt.models.kv_extender import KVExtender

logger = structlog.get_logger()


def split_prefix_continuation(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    min_cont: int,
    max_cont: int,
) -> Dict[str, torch.Tensor]:
    """Sample a prefix / continuation split per batch.

    Uses a single split point for the whole batch (simplifies attention masking
    and keeps prefix lengths equal across the batch)."""
    B, T = input_ids.shape
    # Effective length per batch item (skip padded tail).
    if attention_mask is not None:
        # Use the shortest non-padded length so every item has a valid continuation.
        valid_lens = attention_mask.sum(dim=1)
        eff_T = int(valid_lens.min().item())
    else:
        eff_T = T

    eff_T = max(eff_T, min_cont + 8)
    cont_len = random.randint(min_cont, min(max_cont, eff_T - 8))
    prefix_len = eff_T - cont_len

    prefix_ids = input_ids[:, :prefix_len]
    cont_ids = input_ids[:, prefix_len:prefix_len + cont_len]

    prefix_mask = attention_mask[:, :prefix_len] if attention_mask is not None else None

    return {
        "prefix_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "continuation_ids": cont_ids,
    }


class CompressorTrainer:
    def __init__(
        self,
        model: KVExtender,
        train_config: TrainingConfig,
        model_config: CompressorConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = True,
        save_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config.device
        self.use_wandb = use_wandb and HAS_WANDB
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(train_config.warmup_steps, 1)),
        )

        self.use_amp = train_config.use_amp and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.global_step = 0
        self.best_loss = float("inf")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        groups = [{
            "params": [p for p in self.model.compressor.parameters() if p.requires_grad],
            "lr": self.train_config.learning_rate,
            "name": "compressor",
        }]
        pretrained_params = [p for p in self.model.pretrained.parameters() if p.requires_grad]
        if pretrained_params:
            groups.append({
                "params": pretrained_params,
                "lr": self.train_config.learning_rate * 0.1,
                "name": "pretrained",
            })
        return torch.optim.AdamW(
            groups, betas=(0.9, 0.95), eps=1e-5,
            weight_decay=self.train_config.weight_decay,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_accumulating: bool = False,
        grad_accum_steps: int = 1,
    ) -> Optional[Dict[str, float]]:
        self.model.train()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        split = split_prefix_continuation(
            input_ids, attention_mask,
            self.train_config.min_continuation_length,
            self.train_config.max_continuation_length,
        )
        prefix_ids = split["prefix_ids"]
        prefix_mask = split["prefix_attention_mask"]
        cont_ids = split["continuation_ids"]

        with autocast(device_type="cuda", enabled=self.use_amp):
            out = self.model.forward(
                prefix_ids=prefix_ids,
                continuation_ids=cont_ids,
                prefix_attention_mask=prefix_mask,
                gumbel_noise=self.train_config.use_gumbel_noise,
                return_aux=True,
            )
            loss = out.loss / grad_accum_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = None
        if not is_accumulating:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                W = out.mixing_weights
                # Effective number of source positions per slot = exp(entropy(W_k))
                eff_pos = torch.exp(-(W * (W + 1e-9).log()).sum(dim=-1)).mean()

            metrics = {
                "loss": float(out.loss),
                "lm_loss": float(out.lm_loss),
                "aux_loss": float(out.aux_loss),
                "perplexity": float(torch.exp(out.lm_loss)),
                "prefix_length": float(out.prefix_length),
                "continuation_length": float(out.continuation_length),
                "compression_ratio": float(out.prefix_length / max(out.compressed_length, 1)),
                "effective_positions_per_slot": float(eff_pos),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 1,
    ):
        logger.info(
            "training_start",
            device=self.device,
            trainable_params=self.model.get_trainable_params(),
            num_epochs=num_epochs,
            K_slots=self.model_config.max_kv_slots,
        )

        for epoch in range(num_epochs):
            epoch_metrics = self._train_epoch(train_loader, epoch)
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                epoch_metrics.update(val_metrics)
                if val_metrics.get("val_loss", float("inf")) < self.best_loss:
                    self.best_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
            logger.info("epoch_complete", epoch=epoch + 1, **epoch_metrics)
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            if (self.train_config.max_steps is not None
                    and self.global_step >= self.train_config.max_steps):
                break

        logger.info("training_complete", global_step=self.global_step)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses = []
        accumulated: Dict[str, list] = {}
        grad_accum = self.train_config.gradient_accumulation_steps

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attn = batch.get("attention_mask")
                else:
                    input_ids = batch[0]
                    attn = batch[1] if len(batch) > 1 else None

                is_accum = (batch_idx + 1) % grad_accum != 0
                metrics = self.train_step(
                    input_ids=input_ids, attention_mask=attn,
                    is_accumulating=is_accum, grad_accum_steps=grad_accum,
                )

                if not is_accum:
                    self.global_step += 1

                if metrics is not None:
                    losses.append(metrics["lm_loss"])
                    for k, v in metrics.items():
                        accumulated.setdefault(k, []).append(v)
                    pbar.set_postfix({
                        "ppl": f"{metrics['perplexity']:.2f}",
                        "ratio": f"{metrics['compression_ratio']:.1f}x",
                    })

                if (self.global_step > 0
                        and self.global_step % self.train_config.log_interval == 0
                        and accumulated):
                    avg = {k: sum(v) / len(v) for k, v in accumulated.items()}
                    logger.info("training_step", step=self.global_step, **avg)
                    if self.use_wandb:
                        wandb.log(avg, step=self.global_step)
                    accumulated = {}

                if (self.global_step > 0
                        and self.global_step % self.train_config.save_interval == 0):
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")

                if (self.train_config.max_steps is not None
                        and self.global_step >= self.train_config.max_steps):
                    break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("oom_error", batch=batch_idx)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                raise

        return {"train_lm_loss": sum(losses) / len(losses) if losses else float("inf")}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        self.model.eval()
        total, n = 0.0, 0

        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    attn = batch.get("attention_mask")
                    if attn is not None:
                        attn = attn.to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    attn = batch[1].to(self.device) if len(batch) > 1 else None

                split = split_prefix_continuation(
                    input_ids, attn,
                    self.train_config.min_continuation_length,
                    self.train_config.max_continuation_length,
                )
                out = self.model.forward(
                    prefix_ids=split["prefix_ids"],
                    continuation_ids=split["continuation_ids"],
                    prefix_attention_mask=split["prefix_attention_mask"],
                    gumbel_noise=False,
                    return_aux=False,
                )
                total += float(out.lm_loss)
                n += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        self.model.train()
        if n == 0:
            return {"val_loss": float("inf"), "val_perplexity": float("inf")}
        return {"val_loss": total / n, "val_perplexity": float(torch.exp(torch.tensor(total / n)))}

    def save_checkpoint(self, filename: str):
        path = self.save_dir / filename
        if filename.startswith("checkpoint_"):
            for old in glob.glob(str(self.save_dir / "checkpoint_*.pt")):
                if old != str(path):
                    try:
                        Path(old).unlink()
                    except OSError:
                        pass

        ckpt = {
            "global_step": self.global_step,
            "compressor_state_dict": self.model.compressor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config,
            "best_loss": self.best_loss,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        logger.info("checkpoint_saved", path=str(path))

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.compressor.load_state_dict(ckpt["compressor_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info("checkpoint_loaded", path=path, step=self.global_step)


def create_trainer(
    model: KVExtender,
    train_config: Optional[TrainingConfig] = None,
    model_config: Optional[CompressorConfig] = None,
    **kwargs,
) -> CompressorTrainer:
    return CompressorTrainer(
        model=model,
        train_config=train_config or TrainingConfig(),
        model_config=model_config or model.config,
        **kwargs,
    )
