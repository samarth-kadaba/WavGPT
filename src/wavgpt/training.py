"""Training utilities for WavGPT."""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from wavgpt.data import decode_trimmed


def compute_loss(h_orig, h_refined, h_approx, logits_orig, logits_refined,
                 input_ids, tokenizer, temperature=2.0):
    """
    Compute multi-component loss for training.
    """
    # 1. Hidden state reconstruction loss (L2)
    loss_hidden_refined = F.mse_loss(h_refined, h_orig)
    loss_hidden_approx = F.mse_loss(h_approx, h_orig)

    # 2. Logits reconstruction loss
    loss_logits = F.mse_loss(logits_refined, logits_orig)

    # 3. Cross-entropy loss on actual tokens
    # Shift for causal LM
    shift_logits = logits_refined[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='mean'
    )

    # 4. Knowledge distillation loss (KL divergence)
    loss_kl = F.kl_div(
        F.log_softmax(logits_refined / temperature, dim=-1),
        F.softmax(logits_orig.detach() / temperature, dim=-1),
        reduction='batchmean',
        log_target=False
    ) * (temperature ** 2)

    # Combine losses with weights
    # Weight output space (logits, CE, KL) more than hidden space
    loss_total = (
        0.0 * loss_hidden_refined +
        0.0 * loss_hidden_approx +
        0.0 * loss_logits +
        0.00 * loss_ce +
        1.00 * loss_kl
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


def train_model(model, lm_model, tokenizer, train_loader, optimizer, num_epochs, device, scheduler, log_interval=100, temperature=2.0):
    """Main training loop."""
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get original hidden states and logits (frozen LM)
            with torch.no_grad():
                outputs = lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                h_orig = outputs.hidden_states[-1]  # (B, T, d)
                logits_orig = lm_model.lm_head(h_orig)  # (B, T, vocab)

            # Forward through our model
            h_refined, h_approx, coeffs_sparse, mask_kept, importance_map = model(h_orig, training=True)

            # Get logits from refined hidden states
            logits_refined = lm_model.lm_head(h_refined)

            # Compute losses
            losses = compute_loss(
                h_orig, h_refined, h_approx, logits_orig, logits_refined,
                input_ids, tokenizer, temperature=temperature
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
                # Compute perplexity
                with torch.no_grad():
                    shift_logits_orig = logits_orig[:, :-1, :].contiguous()
                    shift_logits_refined = logits_refined[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    ce_orig = F.cross_entropy(
                        shift_logits_orig.view(-1, shift_logits_orig.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction='mean'
                    )
                    ce_refined = F.cross_entropy(
                        shift_logits_refined.view(-1, shift_logits_refined.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction='mean'
                    )

                    ppl_orig = torch.exp(ce_orig)
                    ppl_refined = torch.exp(ce_refined)

                    # Compression stats
                    num_kept = mask_kept.sum().item()
                    num_total = mask_kept.numel()
                    actual_keep_ratio = num_kept / num_total

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
                        print("="*60 + "\n")

                wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

