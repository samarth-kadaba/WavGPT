"""GRPO Training for Context Extension.

Group Relative Policy Optimization (GRPO) training loop for learning
chunk boundaries. GRPO is a baseline-free policy gradient method that
uses the group mean as the baseline.

Key insight: By sampling multiple boundary configurations and comparing
their rewards (negative LM loss), we can learn which boundary placements
lead to better language modeling without needing a value network.

Algorithm:
    For each batch:
        1. Sample G boundary configurations from policy
        2. For each: compress chunks, forward through transformer, get LM loss
        3. Rewards = -LM_loss
        4. Advantages = (rewards - mean) / std (per-sequence normalization)
        5. Policy loss = -mean(advantages * log_probs)
        6. Compressor loss = mean(LM_loss) over all samples
        7. Update policy with policy gradient
        8. Update compressor with standard gradients
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
    GRPO Trainer for context extension.
    
    Handles the two-phase training:
        1. Policy: GRPO (policy gradient with group-relative advantages)
        2. Compressor: Standard gradient descent on LM loss
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
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Create warmup scheduler if not provided
        if scheduler is None:
            # Warmup for first 100 steps, then constant
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / 100)  # Linear warmup
            )
        else:
            self.scheduler = scheduler
        
        # Mixed precision
        self.scaler = GradScaler() if train_config.use_amp else None
        self.use_amp = train_config.use_amp and self.device == "cuda"
        
        # Reference policy for importance sampling ratio (required for PPO-style clipping)
        # This is a frozen copy used to compute π_old in the ratio r = π_new / π_old
        # Use the full policy wrapper (with projection) since embeddings are in pretrained dim
        self.ref_policy = copy.deepcopy(model.policy)
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        # Tracking
        self.global_step = 0
        self.best_loss = float("inf")
    
    def _update_ref_policy(self):
        """Update reference policy to current policy (for importance sampling)."""
        self.ref_policy.load_state_dict(self.model.policy.state_dict())
        self.ref_policy.eval()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate learning rates for policy and compressor."""
        # Group parameters
        policy_params = list(self.model.policy.parameters())
        compressor_params = list(self.model.compressor.parameters())
        injector_params = list(self.model.injector.parameters())
        
        param_groups = [
            {
                "params": policy_params,
                "lr": self.train_config.policy_lr,
                "name": "policy",
            },
            {
                "params": compressor_params + injector_params,
                "lr": self.train_config.learning_rate,
                "name": "compressor",
            },
        ]
        
        # Add pretrained params if not frozen
        if not self.model_config.freeze_pretrained:
            pretrained_params = [
                p for p in self.model.pretrained.parameters()
                if p.requires_grad
            ]
            if pretrained_params:
                param_groups.append({
                    "params": pretrained_params,
                    "lr": self.train_config.learning_rate * 0.1,  # Lower LR
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
        """
        Single training step with GRPO.
        
        Args:
            input_ids: (B, T) full sequence tokens (model decides chunking)
            labels: (B, T) labels for LM loss
            attention_mask: (B, T) attention mask
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Move to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward with GRPO sampling (model internally handles chunking)
        with autocast(device_type="cuda", enabled=self.use_amp):
            grpo_batch = self.model.forward_grpo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_samples=self.model_config.grpo_num_samples,
                temperature=self.model_config.grpo_temperature,
            )
            
            # Compute reference policy output for importance ratio
            # CRITICAL: Use ref policy's OWN logits for stable ratio computation
            with torch.no_grad():
                ref_policy_output = self.ref_policy.forward(
                    grpo_batch.embeddings,
                    attention_mask=attention_mask,
                )
            
            # Compute policy loss (GRPO)
            policy_loss, policy_metrics = self.model.compute_grpo_loss(
                grpo_batch, ref_policy_output
            )
            
            # Compute compressor loss
            compressor_loss = self.model.compute_compressor_loss(grpo_batch)
            
            # Get KL penalty (prevents drift from original model)
            kl_penalty = grpo_batch.kl_penalty if grpo_batch.kl_penalty is not None else torch.tensor(0.0, device=self.device)
            kl_weight = self.model_config.kl_penalty_weight if hasattr(self.model_config, 'kl_penalty_weight') else 0.0
            
            # Total loss (policy + compressor + KL penalty)
            total_loss = policy_loss + compressor_loss + kl_weight * kl_penalty
        
        # Backward pass
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
        
        # Compile metrics
        metrics = {
            "total_loss": total_loss.item(),
            "compressor_loss": compressor_loss.item(),
            "kl_penalty": kl_penalty.item() if torch.is_tensor(kl_penalty) else kl_penalty,
            **policy_metrics,
            "lr/policy": self.optimizer.param_groups[0]["lr"],
            "lr/compressor": self.optimizer.param_groups[1]["lr"],
        }
        
        # Add pretrained LR if training full model
        if len(self.optimizer.param_groups) > 2:
            metrics["lr/pretrained"] = self.optimizer.param_groups[2]["lr"]
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 1,
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            num_epochs: Number of epochs (or use max_steps)
        """
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
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                epoch_metrics.update(val_metrics)
                
                # Save best model
                if val_metrics.get("val_loss", float("inf")) < self.best_loss:
                    self.best_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
                    logger.info("best_model_saved", val_loss=self.best_loss)
            
            logger.info("epoch_complete", epoch=epoch + 1, **epoch_metrics)
            
            # Update reference policy at end of epoch (standard PPO practice)
            # This ensures the importance ratio stays close to 1
            self._update_ref_policy()
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")
            
            # Check max steps
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
                # Extract batch components
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    labels = batch.get("labels", batch["input_ids"])
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch[0]
                    labels = batch[1] if len(batch) > 1 else batch[0]
                    attention_mask = batch[2] if len(batch) > 2 else None
                
                # Training step with gradient accumulation
                is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
                metrics = self.train_step_with_accumulation(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    is_accumulating=is_accumulating,
                    grad_accum_steps=grad_accum_steps,
                )
                
                # Only count as step when we actually update
                if not is_accumulating:
                    self.global_step += 1
                
                if metrics is not None:
                    epoch_losses.append(metrics["total_loss"])
                    
                    # Accumulate metrics
                    for k, v in metrics.items():
                        if k not in accumulated_metrics:
                            accumulated_metrics[k] = []
                        accumulated_metrics[k].append(v)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{metrics['total_loss']:.4f}",
                        "reward": f"{metrics.get('policy/mean_reward', 0):.4f}",
                    })
                
                # Logging
                if self.global_step > 0 and self.global_step % self.train_config.log_interval == 0 and accumulated_metrics:
                    avg_metrics = {
                        k: sum(v) / len(v) for k, v in accumulated_metrics.items()
                    }
                    logger.info("training_step", step=self.global_step, **avg_metrics)
                    
                    if self.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    accumulated_metrics = {}
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.train_config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                # Max steps check
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
                    # Skip this batch entirely
                    self.optimizer.zero_grad()
                    continue
                raise
        
        return {"train_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0}
    
    def train_step_with_accumulation(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_accumulating: bool = False,
        grad_accum_steps: int = 1,
    ) -> Optional[Dict[str, float]]:
        """
        Training step with gradient accumulation support.
        
        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) labels
            attention_mask: (B, T) attention mask
            is_accumulating: If True, don't step optimizer yet
            grad_accum_steps: Number of accumulation steps (for loss scaling)
            
        Returns:
            metrics: Dict of metrics (only on optimizer step), else None
        """
        self.model.train()
        
        # Move to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward with GRPO sampling
        with autocast(device_type="cuda", enabled=self.use_amp):
            grpo_batch = self.model.forward_grpo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_samples=self.model_config.grpo_num_samples,
                temperature=self.model_config.grpo_temperature,
            )
            
            # Compute reference policy output for importance ratio
            with torch.no_grad():
                ref_policy_output = self.ref_policy.forward(
                    grpo_batch.embeddings,
                    attention_mask=attention_mask,
                )
            
            policy_loss, policy_metrics = self.model.compute_grpo_loss(
                grpo_batch, ref_policy_output
            )
            compressor_loss = self.model.compute_compressor_loss(grpo_batch)
            
            kl_penalty = grpo_batch.kl_penalty if grpo_batch.kl_penalty is not None else torch.tensor(0.0, device=self.device)
            kl_weight = getattr(self.model_config, 'kl_penalty_weight', 0.0)
            
            # Scale loss by accumulation steps
            total_loss = (policy_loss + compressor_loss + kl_weight * kl_penalty) / grad_accum_steps
        
        # Backward (accumulates gradients)
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Only step optimizer when not accumulating
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
            
            self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Return metrics only when we step
            return {
                "total_loss": total_loss.item() * grad_accum_steps,  # Unscale for logging
                "compressor_loss": compressor_loss.item(),
                "kl_penalty": kl_penalty.item() if torch.is_tensor(kl_penalty) else kl_penalty,
                **policy_metrics,
                "lr/policy": self.optimizer.param_groups[0]["lr"],
                "lr/compressor": self.optimizer.param_groups[1]["lr"],
            }
        
        return None  # Still accumulating
    
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
                    past_token_ids = batch.get("past_token_ids", batch["input_ids"]).to(self.device)
                    labels = batch.get("labels", batch["input_ids"]).to(self.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    past_token_ids = batch[1].to(self.device) if len(batch) > 1 else input_ids
                    labels = batch[2].to(self.device) if len(batch) > 2 else input_ids
                    attention_mask = batch[3].to(self.device) if len(batch) > 3 else None
                
                # Forward with deterministic boundaries
                output = self.model(
                    input_ids=input_ids,
                    past_token_ids=past_token_ids,
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
        """Save training checkpoint."""
        path = self.save_dir / filename
        
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

