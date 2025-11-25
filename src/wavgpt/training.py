"""Training utilities for WavGPT."""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from wavgpt.config import KEEP_RATIO, BLOCK_SIZE, HIDDEN_SIZE, WAVELET_LEVELS, REFINE_N_LAYERS, REFINE_N_HEADS, REFINE_DIM_FEEDFORWARD, USE_TEMPORAL_ATTENTION, WARMUP_RATIO
from wavgpt.data import decode_trimmed
from wavgpt.utils.save_checkpoint import save_checkpoint


def compute_loss(h_orig, h_refined, h_approx, logits_orig, logits_refined,
                 input_ids, attention_mask, tokenizer, temperature=2.0):
    """
    Compute multi-component loss for training with attention masking.
    
    KEY CHANGE: Uses attention_mask to ignore padding tokens in loss computation.
    This allows training on variable-length sequences without learning padding artifacts.
    
    Note: For BERT (bidirectional model), no label shifting is needed.
    h[i] represents token[i], so logits[i] predicts token[i] directly.
    """
    # Convert attention_mask to float for masking operations
    mask_float = attention_mask.float()  # (B, T)
    num_real_tokens = attention_mask.sum()  # Total real tokens in batch
    
    # 1. Hidden state reconstruction loss (L2) - masked
    mask_3d = mask_float.unsqueeze(-1)  # (B, T, 1) for broadcasting
    loss_hidden_refined = ((h_refined - h_orig) ** 2 * mask_3d).sum() / num_real_tokens / h_orig.size(-1)
    loss_hidden_approx = ((h_approx - h_orig) ** 2 * mask_3d).sum() / num_real_tokens / h_orig.size(-1)

    # 2. Logits reconstruction loss - masked
    logits_diff = ((logits_refined - logits_orig) ** 2).sum(dim=-1)  # (B, T)
    loss_logits = (logits_diff * mask_float).sum() / num_real_tokens

    # 3. Cross-entropy loss on actual tokens - already handles padding via ignore_index
    loss_ce = F.cross_entropy(
        logits_refined.view(-1, logits_refined.size(-1)),
        input_ids.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='mean'
    )

    # 4. Knowledge distillation loss (KL divergence) - masked
    # Compute KL for each token position
    kl_per_token = F.kl_div(
        F.log_softmax(logits_refined / temperature, dim=-1),
        F.softmax(logits_orig.detach() / temperature, dim=-1),
        reduction='none',
        log_target=False
    ).sum(dim=-1)  # (B, T)
    
    # Mask padding and average over real tokens only
    loss_kl = (kl_per_token * mask_float).sum() / num_real_tokens * (temperature ** 2)

    # Combine losses with weights
    # Weight output space (logits, CE, KL) more than hidden space
    loss_total = (
        0.0 * loss_hidden_refined +
        0.0 * loss_hidden_approx +
        0.0 * loss_logits +
        0.10 * loss_ce +
        0.90 * loss_kl
    )

    return {
        'loss_total': loss_total,
        'loss_hidden_refined': loss_hidden_refined,
        'loss_hidden_approx': loss_hidden_approx,
        'loss_logits': loss_logits,
        'loss_ce': loss_ce,
        'loss_kl': loss_kl,
    }


def create_scheduler(optimizer, num_epochs, train_loader, warmup_ratio=0.05):
    """Create learning rate scheduler with warmup and cosine annealing."""
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    # Create warmup scheduler (linear increase from 0 to initial lr)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # starts at 0.1 * initial_lr
        end_factor=1.0,    # ends at 1.0 * initial_lr
        total_iters=warmup_steps
    )

    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,  # remaining epochs after warmup
        eta_min=1e-6  # minimum learning rate
    )

    # Combine them sequentially
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler


def train_model(model, lm_model, tokenizer, train_loader, optimizer, num_epochs, device, scheduler, log_interval=100, temperature=2.0, start_epoch=0, start_global_step=0):
    """Main training loop."""
    global_step = start_global_step

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get original hidden states and logits (frozen BERT)
            with torch.no_grad():
                outputs = lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                h_orig = outputs.hidden_states[-1]  # (B, T, d)
                # BERT uses cls.predictions for the MLM head
                logits_orig = lm_model.cls.predictions(h_orig)  # (B, T, vocab)

            # Forward through our model
            h_refined, h_approx, coeffs_sparse, mask_kept, importance_map = model(h_orig, training=True)

            # Get logits from refined hidden states using BERT's MLM head
            logits_refined = lm_model.cls.predictions(h_refined)

            # Compute losses (now with attention_mask for proper masking)
            losses = compute_loss(
                h_orig, h_refined, h_approx, logits_orig, logits_refined,
                input_ids, attention_mask, tokenizer, temperature=temperature
            )
            loss = losses['loss_total']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            global_step += 1
            epoch_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{losses['loss_ce'].item():.4f}",
            })

            # Detailed logging
            if global_step % log_interval == 0:
                # Compute perplexity (no shifting needed for BERT)
                with torch.no_grad():
                    ce_orig = F.cross_entropy(
                        logits_orig.view(-1, logits_orig.size(-1)),
                        input_ids.view(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction='mean'
                    )
                    ce_refined = F.cross_entropy(
                        logits_refined.view(-1, logits_refined.size(-1)),
                        input_ids.view(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction='mean'
                    )

                    ppl_orig = torch.exp(ce_orig)
                    ppl_refined = torch.exp(ce_refined)

                    # Compression stats - verify hard masking produces true sparsity
                    num_kept = mask_kept.sum().item()
                    num_total = mask_kept.numel()
                    actual_keep_ratio = num_kept / num_total
                    
                    # Check for true sparsity (hard masks should give binary 0/1)
                    num_nonzero_coeffs = (coeffs_sparse != 0).sum().item()
                    # For hard masks, num_nonzero should ≈ num_kept
                    # For soft masks, num_nonzero would be much larger (~115k vs ~1k)

                # Log to wandb
                log_dict = {
                    'loss_total': losses['loss_total'].item(),
                    'loss_hidden_refined': losses['loss_hidden_refined'].item(),
                    'loss_hidden_approx': losses['loss_hidden_approx'].item(),
                    'loss_logits': losses['loss_logits'].item(),
                    'loss_ce': losses['loss_ce'].item(),
                    'loss_kl': losses['loss_kl'].item(),
                    'ppl_original': ppl_orig.item(),
                    'ppl_refined': ppl_refined.item(),
                    'ppl_degradation': ppl_refined.item() - ppl_orig.item(),
                    'compression_ratio': actual_keep_ratio,
                    'sparsity/num_kept_mask': num_kept,
                    'sparsity/num_nonzero_coeffs': num_nonzero_coeffs,
                    'sparsity/true_sparsity_ratio': num_nonzero_coeffs / num_total,
                    'sparsity/mask_vs_coeffs_diff': abs(num_nonzero_coeffs - num_kept),
                    'epoch': epoch,
                }

                # Sample text comparison
                with torch.no_grad():
                    i = 0  # First sample in batch
                    orig_text = decode_trimmed(tokenizer, input_ids[i])
                    pred_orig_ids = torch.argmax(logits_orig[i], dim=-1)
                    pred_orig_text = decode_trimmed(tokenizer, pred_orig_ids)
                    pred_refined_ids = torch.argmax(logits_refined[i], dim=-1)
                    pred_refined_text = decode_trimmed(tokenizer, pred_refined_ids)

                    log_dict.update({
                        'text/original_input': orig_text,
                        'text/from_orig_logits': pred_orig_text,
                        'text/from_refined_logits': pred_refined_text,
                    })

                    # Print sample
                    if global_step % (log_interval * 5) == 0:
                        print("\n" + "="*60)
                        print(f"Step {global_step} - Text Comparison")
                        print("="*60)
                        print(f"Original Input:\n{orig_text[:200]}...\n")
                        print(f"From Original Logits:\n{pred_orig_text[:200]}...\n")
                        print(f"From Refined Logits:\n{pred_refined_text[:200]}...\n")
                        print(f"PPL Original: {ppl_orig.item():.2f}")
                        print(f"PPL Refined: {ppl_refined.item():.2f}")
                        print(f"PPL Degradation: {ppl_refined.item() - ppl_orig.item():.2f}")
                        print(f"\nSparsity Check (verifying hard masks):")
                        print(f"  Mask entries kept: {num_kept:,} / {num_total:,} ({actual_keep_ratio:.2%})")
                        print(f"  Nonzero coeffs:    {num_nonzero_coeffs:,} / {num_total:,} ({num_nonzero_coeffs/num_total:.2%})")
                        print(f"  Difference:        {abs(num_nonzero_coeffs - num_kept):,.0f} (should be ~0 for hard masks)")
                        if abs(num_nonzero_coeffs - num_kept) / num_total > 0.01:
                            print(f"  ⚠️  WARNING: Large difference suggests soft masking!")
                        else:
                            print(f"  ✓ Hard masking confirmed (true sparsity)")
                        print("="*60 + "\n")

                        # Save checkpoint at project root level
                        import os
                        # From /home/ubuntu/WavGPT/src/wavgpt/training.py -> /home/ubuntu/WavGPT
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        save_path = os.path.join(project_root, "checkpoints", f"hybrid_wavelet_model_ratio{KEEP_RATIO}_step{global_step}.pt")
                        
                        checkpoint_config = {
                            'seq_len': BLOCK_SIZE,
                            'hidden_size': HIDDEN_SIZE,
                            'keep_ratio': KEEP_RATIO,
                            'wavelet_levels': WAVELET_LEVELS,
                            'refine_n_layers': REFINE_N_LAYERS,
                            'refine_n_heads': REFINE_N_HEADS,
                            'refine_dim_feedforward': REFINE_DIM_FEEDFORWARD,
                            'use_temporal_attention': USE_TEMPORAL_ATTENTION,
                            'temperature': temperature,
                            'warmup_ratio': WARMUP_RATIO,
                        }
                        
                        checkpoint_metrics = {
                            'ppl_original': ppl_orig.item(),
                            'ppl_refined': ppl_refined.item(),
                            'ppl_degradation': ppl_refined.item() - ppl_orig.item(),
                            'compression_ratio': actual_keep_ratio,
                        }
                        
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            losses=losses,
                            metrics=checkpoint_metrics,
                            config=checkpoint_config,
                            save_path=save_path,
                        )

                wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

