"""Training utilities for Infinite Context Transformer."""

from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import structlog

from wavgpt.config import (
    NUM_EPOCHS,
    LOG_INTERVAL,
    SAVE_INTERVAL,
    VAL_INTERVAL,
    MAX_GRAD_NORM,
    GRADIENT_ACCUMULATION_STEPS,
    DEVICE,
    CHECKPOINT_DIR,
)
from wavgpt.training.utils import save_checkpoint
from wavgpt.training.validation import validate

logger = structlog.get_logger()


def train(
    model,
    train_loader,
    optimizer,
    num_epochs: int = NUM_EPOCHS,
    device: str = DEVICE,
    scheduler=None,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    val_interval: int = VAL_INTERVAL,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    use_wandb: bool = True,
    save_dir: str = CHECKPOINT_DIR,
    use_amp: bool = True,  # Mixed precision training
    compile_model: bool = False,  # torch.compile
    val_loader=None,  # Validation dataloader
    val_batches: int = 50,  # Number of validation batches
    save_best_only: bool = False,  # Only save best model (saves disk space)
):
    """
    Train the Infinite Context Transformer.

    Args:
        model: InfiniteContextTransformer model
        train_loader: DataLoader
        optimizer: Optimizer
        num_epochs: Number of epochs
        device: Device to train on
        scheduler: Learning rate scheduler
        log_interval: Steps between logging
        save_interval: Steps between checkpoints
        val_interval: Steps between validation runs
        gradient_accumulation_steps: Gradient accumulation steps
        use_wandb: Whether to log to W&B
        save_dir: Directory for saving checkpoints
        use_amp: Use automatic mixed precision (FP16)
        compile_model: Use torch.compile for optimization
        val_loader: Optional validation DataLoader
        val_batches: Number of batches to use for validation
        save_best_only: If True, only save best model based on val loss (saves disk)
    """
    use_wandb = use_wandb and HAS_WANDB
    global_step = 0
    total_tokens = 0  # Track total training tokens seen

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Setup mixed precision training
    scaler = GradScaler() if use_amp and device == "cuda" else None
    use_amp = use_amp and device == "cuda"

    # Optional: compile model for faster execution
    # Note: Use 'default' mode to avoid CUDAGraphs issues with dynamic chunk sizes
    if compile_model and hasattr(torch, "compile"):
        logger.info("compiling_model", mode="default")
        model = torch.compile(model, mode="default", fullgraph=False)

    logger.info(
        "training_start",
        device=device,
        parameters=model.get_num_params(),
        gradient_accumulation=gradient_accumulation_steps,
        mixed_precision=use_amp,
        torch_compile=compile_model,
        validation_enabled=val_loader is not None,
        val_interval=val_interval if val_loader is not None else None,
        save_mode="best_only" if save_best_only else "all_checkpoints",
    )

    # Track best validation LM loss for model selection (not total loss)
    best_val_lm_loss = float("inf")

    model.to(device)

    for epoch in range(num_epochs):
        logger.info("epoch_start", epoch=epoch + 1, total_epochs=num_epochs)

        model.train()
        epoch_losses = []
        accumulated_loss = 0.0
        accumulation_counter = 0

        # Use total=None for streaming datasets that don't have a length
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None  # IterableDataset doesn't support len()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=total_batches)

        for batch_idx, batch in enumerate(pbar):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels", input_ids.clone()).to(device)

                # Count tokens in this batch
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    labels = labels.masked_fill(attention_mask == 0, -100)

                # Forward pass with optional mixed precision
                if use_amp:
                    with autocast(device_type="cuda"):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs["loss"] / gradient_accumulation_steps
                else:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / gradient_accumulation_steps

                # NaN detection: skip batch if loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "nan_loss_detected",
                        batch=batch_idx,
                        step=global_step,
                        loss=loss.item() if not torch.isnan(loss) else "NaN",
                    )
                    optimizer.zero_grad()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item()
                accumulation_counter += 1

                # Optimizer step
                if accumulation_counter >= gradient_accumulation_steps:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    
                    # Check for NaN gradients before clipping
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        logger.warning("nan_gradient_detected", step=global_step)
                        optimizer.zero_grad()
                        if use_amp:
                            scaler.update()  # Still update scaler to adjust scale factor
                        accumulated_loss = 0.0
                        accumulation_counter = 0
                        continue

                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1
                    epoch_losses.append(accumulated_loss)

                    # Get average chunks (n_chunks is now a tensor)
                    n_chunks = outputs["n_chunks"]
                    if isinstance(n_chunks, torch.Tensor):
                        avg_chunks = n_chunks.item()
                    elif isinstance(n_chunks, list):
                        avg_chunks = sum(n_chunks) / len(n_chunks)
                    else:
                        avg_chunks = n_chunks

                    # Get boundary statistics
                    boundary_probs = outputs.get("boundary_probs")
                    if boundary_probs is not None:
                        mean_boundary_prob = boundary_probs.mean().item()
                    else:
                        mean_boundary_prob = 0.0

                    pbar.set_postfix(
                        {
                            "loss": f"{accumulated_loss:.4f}",
                            "chunks": f"{avg_chunks:.1f}",
                            "b_prob": f"{mean_boundary_prob:.2f}",
                        }
                    )

                    # Logging
                    if global_step % log_interval == 0:
                        # Get LM loss
                        lm_loss = outputs.get("lm_loss")
                        lm_loss_val = lm_loss.item() if lm_loss is not None else accumulated_loss
                        
                        expected_chunks = outputs.get("expected_chunks")
                        expected_chunks_val = expected_chunks.item() if expected_chunks is not None else avg_chunks

                        logger.info(
                            "training_step",
                            step=global_step,
                            loss=accumulated_loss,
                            lm_loss=lm_loss_val,
                            avg_chunks=avg_chunks,
                            expected_chunks=expected_chunks_val,
                            boundary_mean_prob=mean_boundary_prob,
                            tokens_millions=total_tokens / 1e6,
                        )

                        if use_wandb:
                            # Get auxiliary losses
                            entropy_loss = outputs.get("entropy_loss")
                            sparsity_loss = outputs.get("sparsity_loss")
                            distill_loss = outputs.get("distillation_loss")
                            
                            log_dict = {
                                "loss": accumulated_loss,
                                "lm_loss": lm_loss_val,
                                "avg_chunks": avg_chunks,
                                "expected_chunks": expected_chunks_val,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "boundary/mean_prob": mean_boundary_prob,
                                "boundary/entropy_loss": entropy_loss.item() if entropy_loss is not None else 0.0,
                                "boundary/sparsity_loss": sparsity_loss.item() if sparsity_loss is not None else 0.0,
                                "boundary/distill_loss": distill_loss.item() if distill_loss is not None else 0.0,
                                "total_tokens": total_tokens,
                                "tokens_millions": total_tokens / 1e6,
                            }
                            wandb.log(log_dict, step=global_step)

                    # Save checkpoint (skip if save_best_only is enabled)
                    if not save_best_only and global_step % save_interval == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            global_step,
                            save_path / f"checkpoint_{global_step}.pt",
                            total_tokens=total_tokens,
                        )

                    # Validation
                    if val_loader is not None and global_step % val_interval == 0:
                        val_metrics = validate(
                            model,
                            val_loader,
                            device=device,
                            max_batches=val_batches,
                            use_amp=use_amp,
                        )

                        logger.info(
                            "validation",
                            step=global_step,
                            val_loss=val_metrics["val_loss"],
                            val_lm_loss=val_metrics["val_lm_loss"],
                            val_avg_chunks=val_metrics["val_avg_chunks"],
                            val_expected_chunks=val_metrics["val_expected_chunks"],
                        )

                        if use_wandb:
                            wandb.log(val_metrics, step=global_step)

                        # Save best model (based on LM loss, not total loss)
                        if val_metrics["val_lm_loss"] < best_val_lm_loss:
                            best_val_lm_loss = val_metrics["val_lm_loss"]

                            # Delete old checkpoints to free disk space
                            for old_ckpt in save_path.glob("*.pt"):
                                try:
                                    old_ckpt.unlink()
                                except Exception:
                                    pass

                            save_checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                epoch,
                                global_step,
                                save_path / "best_model.pt",
                                total_tokens=total_tokens,
                            )
                            logger.info("best_model_saved", val_lm_loss=best_val_lm_loss)

                    accumulated_loss = 0.0
                    accumulation_counter = 0

                    # Clear CUDA cache periodically
                    if global_step % 100 == 0 and device == "cuda":
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("oom_error", batch=batch_idx, error=str(e))
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error("training_error", batch=batch_idx, error=str(e))
                optimizer.zero_grad()
                continue

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info("epoch_complete", epoch=epoch + 1, avg_loss=avg_loss)

        # End-of-epoch validation
        if val_loader is not None:
            val_metrics = validate(
                model,
                val_loader,
                device=device,
                max_batches=val_batches * 2,  # More batches for epoch validation
                use_amp=use_amp,
            )

            logger.info(
                "epoch_validation",
                epoch=epoch + 1,
                val_loss=val_metrics["val_loss"],
                val_lm_loss=val_metrics["val_lm_loss"],
            )

            if use_wandb:
                epoch_val_metrics = {f"epoch_{k}": v for k, v in val_metrics.items()}
                epoch_val_metrics["epoch"] = epoch + 1
                wandb.log(epoch_val_metrics, step=global_step)

            # Update best model (based on LM loss, not total loss)
            if val_metrics["val_lm_loss"] < best_val_lm_loss:
                best_val_lm_loss = val_metrics["val_lm_loss"]

                # Delete old checkpoints to free disk space
                for old_ckpt in save_path.glob("*.pt"):
                    try:
                        old_ckpt.unlink()
                    except Exception:
                        pass

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    save_path / "best_model.pt",
                    total_tokens=total_tokens,
                )
                logger.info("best_model_saved", val_lm_loss=best_val_lm_loss)

        # Save epoch checkpoint (skip if save_best_only is enabled)
        if not save_best_only:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                save_path / f"epoch_{epoch + 1}.pt",
                total_tokens=total_tokens,
            )
