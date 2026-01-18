"""GRPO Training for Context Extension with Unified Policy-Compressor.

Group Relative Policy Optimization (GRPO) training loop for learning
chunk boundaries with credit assignment via difficulty scores.

KEY INSIGHT: The unified policy-compressor shares an SSM backbone,
enabling end-to-end credit assignment. Difficulty scores tell the
policy which boundary placements make compression hard vs easy.

Algorithm:
    For each batch:
        1. Sample G boundary configurations from policy
        2. For each: compress chunks (reusing hidden states!), forward through transformer
        3. Rewards = -LM_loss
        4. Advantages = (rewards - mean) / std
        5. Policy loss = GRPO with difficulty-based credit assignment
        6. Update with gradient descent
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import structlog

from wavgpt.models.config import ContextExtenderConfig, TrainingConfig
from wavgpt.models.context_extender import ContextExtender

logger = structlog.get_logger()


class GRPOTrainer:
    """
    GRPO Trainer for unified policy-compressor architecture.
    
    Handles training with:
        - GRPO policy gradient with difficulty-based credit assignment
        - Shared SSM backbone between policy and compression
        - Reference policy for importance sampling
    """
    
    def __init__(
        self,
        model: ContextExtender,
        train_config: TrainingConfig,
        model_config: ContextExtenderConfig,
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
        
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / 100)
            )
        else:
            self.scheduler = scheduler
        
        self.scaler = GradScaler() if train_config.use_amp else None
        self.use_amp = train_config.use_amp and self.device == "cuda"
        
        # Reference policy (frozen copy for importance sampling)
        self.ref_policy = copy.deepcopy(model.policy)
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        self.global_step = 0
        self.best_loss = float("inf")
        
        if hasattr(self.model, 'compile_modules'):
            self.model.compile_modules()
    
    def _update_ref_policy(self):
        """Update reference policy to current policy."""
        current_state = self.model.policy.state_dict()
        
        # Handle torch.compile prefix
        cleaned_state = {}
        for key, value in current_state.items():
            clean_key = key.replace('._orig_mod.', '.').replace('_orig_mod.', '')
            cleaned_state[clean_key] = value
        
        ref_keys = set(self.ref_policy.state_dict().keys())
        current_cleaned_keys = set(cleaned_state.keys())
        
        if ref_keys == current_cleaned_keys:
            self.ref_policy.load_state_dict(cleaned_state)
        else:
            try:
                self.ref_policy.load_state_dict(cleaned_state, strict=False)
            except RuntimeError as e:
                logger.warning("ref_policy_update_failed", error=str(e)[:200])
                self.ref_policy = copy.deepcopy(self.model.policy)
                if hasattr(self.ref_policy, '_orig_mod'):
                    self.ref_policy = self.ref_policy._orig_mod
                for param in self.ref_policy.parameters():
                    param.requires_grad = False
        
        self.ref_policy.eval()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate LRs for policy and injector."""
        policy_params = list(self.model.policy.parameters())
        injector_params = list(self.model.injector.parameters())
        
        param_groups = [
            {
                "params": policy_params,
                "lr": self.train_config.policy_lr,
                "name": "policy",
            },
            {
                "params": injector_params,
                "lr": self.train_config.learning_rate,
                "name": "injector",
            },
        ]
        
        if not self.model_config.freeze_pretrained:
            pretrained_params = [
                p for p in self.model.pretrained.parameters()
                if p.requires_grad
            ]
            if pretrained_params:
                param_groups.append({
                    "params": pretrained_params,
                    "lr": self.train_config.learning_rate * 0.1,
                    "name": "pretrained",
                })
        
        return torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=self.train_config.weight_decay,
        )
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step with GRPO."""
        self.model.train()
        
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with autocast(device_type="cuda", enabled=self.use_amp):
            grpo_batch = self.model.forward_grpo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_samples=self.model_config.grpo_num_samples,
                temperature=self.model_config.grpo_temperature,
            )
            
            # Reference policy output for importance ratio
            with torch.no_grad():
                ref_policy_output = self.ref_policy.forward(
                    grpo_batch.hidden_states,
                    attention_mask=attention_mask,
                )
            
            # GRPO loss with difficulty-based credit assignment
            policy_loss, policy_metrics = self.model.compute_grpo_loss(
                grpo_batch, ref_policy_output
            )
            
            # Compressor loss (from first sample)
            compressor_loss = self.model.compute_compressor_loss(grpo_batch)
            
            # Scale policy loss
            policy_loss_scale = getattr(self.model_config, 'policy_loss_scale', 1000.0)
            scaled_policy_loss = policy_loss * policy_loss_scale
            
            total_loss = scaled_policy_loss + compressor_loss
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        metrics = {
            "total_loss": total_loss.item(),
            "compressor_loss": compressor_loss.item(),
            **policy_metrics,
            "lr/policy": self.optimizer.param_groups[0]["lr"],
            "lr/injector": self.optimizer.param_groups[1]["lr"],
        }
        
        if len(self.optimizer.param_groups) > 2:
            metrics["lr/pretrained"] = self.optimizer.param_groups[2]["lr"]
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 1,
    ):
        """Full training loop."""
        logger.info(
            "training_start",
            device=self.device,
            trainable_params=self.model.get_trainable_params(),
            num_epochs=num_epochs,
            grpo_samples=self.model_config.grpo_num_samples,
        )
        
        for epoch in range(num_epochs):
            logger.info("epoch_start", epoch=epoch + 1, total_epochs=num_epochs)
            
            epoch_metrics = self._train_epoch(train_loader, epoch)
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                epoch_metrics.update(val_metrics)
                
                if val_metrics.get("val_loss", float("inf")) < self.best_loss:
                    self.best_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
                    logger.info("best_model_saved", val_loss=self.best_loss)
            
            logger.info("epoch_complete", epoch=epoch + 1, **epoch_metrics)
            
            self._update_ref_policy()
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")
            
            if (
                self.train_config.max_steps is not None and
                self.global_step >= self.train_config.max_steps
            ):
                break
        
        logger.info("training_complete", global_step=self.global_step)
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        
        epoch_losses = []
        accumulated_metrics = {}
        grad_accum_steps = self.train_config.gradient_accumulation_steps
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    labels = batch.get("labels", batch["input_ids"])
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch[0]
                    labels = batch[1] if len(batch) > 1 else batch[0]
                    attention_mask = batch[2] if len(batch) > 2 else None
                
                is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
                metrics = self.train_step_with_accumulation(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    is_accumulating=is_accumulating,
                    grad_accum_steps=grad_accum_steps,
                )
                
                if not is_accumulating:
                    self.global_step += 1
                
                if metrics is not None:
                    epoch_losses.append(metrics["lm_loss"])
                    
                    for k, v in metrics.items():
                        if k not in accumulated_metrics:
                            accumulated_metrics[k] = []
                        accumulated_metrics[k].append(v)
                    
                    pbar.set_postfix({
                        "lm_loss": f"{metrics['lm_loss']:.4f}",
                        "chunks": f"{metrics.get('policy/mean_chunks', 0):.1f}",
                        "difficulty": f"{metrics.get('policy/difficulty_loss', 0):.4f}",
                    })
                
                if self.global_step > 0 and self.global_step % self.train_config.log_interval == 0 and accumulated_metrics:
                    avg_metrics = {
                        k: sum(v) / len(v) for k, v in accumulated_metrics.items()
                    }
                    logger.info("training_step", step=self.global_step, **avg_metrics)
                    
                    if self.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    accumulated_metrics = {}
                
                if self.global_step > 0 and self.global_step % self.train_config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                if (
                    self.train_config.max_steps is not None and
                    self.global_step >= self.train_config.max_steps
                ):
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("oom_error", batch=batch_idx, seq_len=input_ids.shape[1] if hasattr(input_ids, 'shape') else 'unknown')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                raise
        
        avg_lm_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        return {"train_lm_loss": avg_lm_loss}
    
    def train_step_with_accumulation(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_accumulating: bool = False,
        grad_accum_steps: int = 1,
    ) -> Optional[Dict[str, float]]:
        """Training step with gradient accumulation."""
        self.model.train()
        
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with autocast(device_type="cuda", enabled=self.use_amp):
            grpo_batch = self.model.forward_grpo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_samples=self.model_config.grpo_num_samples,
                temperature=self.model_config.grpo_temperature,
            )
            
            with torch.no_grad():
                ref_policy_output = self.ref_policy.forward(
                    grpo_batch.hidden_states,
                    attention_mask=attention_mask,
                )
            
            policy_loss, policy_metrics = self.model.compute_grpo_loss(
                grpo_batch, ref_policy_output
            )
            compressor_loss = self.model.compute_compressor_loss(grpo_batch)
            
            policy_loss_scale = getattr(self.model_config, 'policy_loss_scale', 1000.0)
            scaled_policy_loss = policy_loss * policy_loss_scale
            
            total_loss = (scaled_policy_loss + compressor_loss) / grad_accum_steps
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        if not is_accumulating:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
            
            mean_reward = grpo_batch.rewards.mean().item()
            lm_loss = -mean_reward
            
            return {
                "total_loss": total_loss.item() * grad_accum_steps,
                "lm_loss": lm_loss,
                "compressor_loss": compressor_loss.item(),
                **policy_metrics,
                "lr/policy": self.optimizer.param_groups[0]["lr"],
                "lr/injector": self.optimizer.param_groups[1]["lr"],
            }
        
        return None
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        total_reward = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch.get("labels", batch["input_ids"]).to(self.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
                    attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
                
                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    use_deterministic_boundaries=True,
                )
                
                if output.loss is not None:
                    total_loss += output.loss.item()
                    total_reward += -output.loss.item()
                    num_batches += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
        
        self.model.train()
        
        if num_batches == 0:
            return {"val_loss": float("inf"), "val_reward": float("-inf")}
        
        return {
            "val_loss": total_loss / num_batches,
            "val_reward": total_reward / num_batches,
        }
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint, keeping only the latest step checkpoint."""
        path = self.save_dir / filename
        
        # Delete old step checkpoints (keep only latest)
        if filename.startswith("checkpoint_"):
            import glob
            old_checkpoints = glob.glob(str(self.save_dir / "checkpoint_*.pt"))
            for old_ckpt in old_checkpoints:
                if old_ckpt != str(path):
                    try:
                        Path(old_ckpt).unlink()
                        logger.debug("deleted_old_checkpoint", path=old_ckpt)
                    except OSError:
                        pass
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config,
            "best_loss": self.best_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info("checkpoint_saved", path=str(path))
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info("checkpoint_loaded", path=path, step=self.global_step)


def create_grpo_trainer(
    model: ContextExtender,
    train_config: Optional[TrainingConfig] = None,
    model_config: Optional[ContextExtenderConfig] = None,
    **kwargs,
) -> GRPOTrainer:
    """Convenience function to create a GRPO trainer."""
    if train_config is None:
        train_config = TrainingConfig()
    if model_config is None:
        model_config = model.config
    
    return GRPOTrainer(
        model=model,
        train_config=train_config,
        model_config=model_config,
        **kwargs,
    )
