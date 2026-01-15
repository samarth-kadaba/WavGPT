"""Chunking Policy for GRPO-based boundary and retention learning.

This module implements the policy network that learns:
    1. WHERE to place chunk boundaries
    2. WHICH tokens to keep at full fidelity (for retrieval)

The policy outputs TWO decisions per token:
    - boundary_prob: probability of ending a chunk at this position
    - keep_prob: probability of keeping this token verbatim (not compressed)

GRPO Algorithm:
    1. Sample G (boundary, keep) configurations from policy
    2. Compute reward (negative LM loss) for each
    3. Compute group-relative advantage: A_i = (r_i - mean) / std
    4. Update policy to maximize advantage-weighted log probability

Constraint: num_chunks + num_kept_tokens <= max_context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import ContextExtenderConfig


class PolicySample(NamedTuple):
    """A sampled configuration with boundaries and kept tokens."""
    boundaries: torch.Tensor      # (B, T) binary: 1 = end chunk here
    keep_mask: torch.Tensor       # (B, T) binary: 1 = keep this token verbatim
    log_probs: torch.Tensor       # (B,) log probability of this configuration
    boundary_logits: torch.Tensor # (B, T) boundary logits (for stable log-prob)
    keep_logits: torch.Tensor     # (B, T) keep logits (for stable log-prob)
    boundary_probs: torch.Tensor  # (B, T) boundary probability at each position
    keep_probs: torch.Tensor      # (B, T) keep probability at each position


# Alias for backwards compatibility
BoundarySample = PolicySample


@dataclass
class PolicyOutput:
    """Output from policy forward pass."""
    boundary_logits: torch.Tensor   # (B, T) raw boundary logits
    boundary_probs: torch.Tensor    # (B, T) boundary probabilities
    keep_logits: torch.Tensor       # (B, T) raw keep logits
    keep_probs: torch.Tensor        # (B, T) keep probabilities
    hidden_states: torch.Tensor     # (B, T, D) SSM hidden states


class BoundaryPolicy(nn.Module):
    """
    Policy network for learning chunk boundaries AND token retention via GRPO.
    
    Architecture:
        1. SSM backbone processes input embeddings → hidden states
        2. Boundary head: hidden states → boundary logits (where to chunk)
        3. Keep head: hidden states → keep logits (what to keep verbatim)
        4. During training: sample from Bernoulli for both decisions
        5. During inference: threshold at 0.5
    
    The policy outputs TWO independent decisions per token:
        π(B, K|x) = ∏_t π(b_t | h_t) * π(k_t | h_t)
    
    Constraint enforced during sampling: num_chunks + num_kept <= max_context
    """
    
    def __init__(self, config: ContextExtenderConfig):
        super().__init__()
        self.config = config
        
        # Import here to avoid circular dependency
        from wavgpt.models.ssm import SSMBackbone
        
        # SSM backbone for processing context
        self.backbone = SSMBackbone(
            d_model=config.chunk_dim,
            n_layers=config.n_ssm_layers,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        
        # Boundary head: hidden state → boundary logit (where to chunk)
        self.boundary_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Keep head: hidden state → keep logit (what to retain verbatim)
        self.keep_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Initialize biases
        with torch.no_grad():
            # Start at 50% for better exploration (was -2.0 → 12%)
            self.boundary_head[-1].bias.fill_(config.initial_boundary_bias)
            # Start conservative for keep decisions
            initial_keep = getattr(config, 'initial_keep_bias', -2.0)
            self.keep_head[-1].bias.fill_(initial_keep)
        
        # Learnable temperature for sampling (optional)
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Compute boundary and keep probabilities for a sequence.
        
        Args:
            x: (B, T, D) input embeddings (projected to chunk_dim)
            attention_mask: (B, T) optional mask for valid tokens (1=valid, 0=padding)
            
        Returns:
            PolicyOutput with boundary/keep logits, probs, and hidden states
        """
        # Process through backbone
        hidden = self.backbone(x)  # (B, T, D)
        
        # Compute boundary logits
        boundary_logits = self.boundary_head(hidden).squeeze(-1)  # (B, T)
        
        # First position is never a boundary - use safe mask value
        # -20 gives sigmoid(-20) ≈ 2e-9, effectively 0 but numerically safe
        SAFE_MASK_VALUE = -20.0
        boundary_logits = torch.cat([
            torch.full((boundary_logits.size(0), 1), SAFE_MASK_VALUE, 
                       device=boundary_logits.device, dtype=boundary_logits.dtype),
            boundary_logits[:, 1:]
        ], dim=1)
        
        # Compute keep logits (which tokens to retain verbatim)
        keep_logits = self.keep_head(hidden).squeeze(-1)  # (B, T)
        
        # Mask out padded positions with safe value
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
            boundary_logits = boundary_logits.masked_fill(padding_mask, SAFE_MASK_VALUE)
            keep_logits = keep_logits.masked_fill(padding_mask, SAFE_MASK_VALUE)
        
        # Compute probabilities
        boundary_probs = torch.sigmoid(boundary_logits)
        keep_probs = torch.sigmoid(keep_logits)
        
        return PolicyOutput(
            boundary_logits=boundary_logits,
            boundary_probs=boundary_probs,
            keep_logits=keep_logits,
            keep_probs=keep_probs,
            hidden_states=hidden,
        )
    
    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
        deterministic: bool = False,
        max_context: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list, PolicyOutput]:
        """
        Sample (boundary, keep) configurations from the policy.
        
        Args:
            x: (B, T, D) input embeddings
            num_samples: Number of configurations to sample (G in GRPO)
            temperature: Sampling temperature (higher = more random)
            deterministic: If True, use threshold instead of sampling
            max_context: Maximum total context (num_chunks + num_kept <= max_context)
            attention_mask: (B, T) optional mask for valid tokens (1=valid, 0=padding)
            
        Returns:
            samples: List of PolicySample, one per sample
            policy_output: PolicyOutput from forward pass
        """
        if max_context is None:
            max_context = self.config.max_context
            
        policy_output = self.forward(x, attention_mask=attention_mask)
        B, T = policy_output.boundary_probs.shape
        device = x.device
        
        samples = []
        
        for _ in range(num_samples):
            if deterministic:
                # Hard threshold at 0.5 (logit > 0)
                boundaries = (policy_output.boundary_logits > 0).float()
                keep_mask = (policy_output.keep_logits > 0).float()
            else:
                # Sample from Bernoulli with temperature
                # Apply temperature to logits (more stable than to probs)
                scaled_boundary_logits = policy_output.boundary_logits / temperature
                scaled_keep_logits = policy_output.keep_logits / temperature
                boundary_probs = torch.sigmoid(scaled_boundary_logits)
                keep_probs = torch.sigmoid(scaled_keep_logits)
                boundaries = torch.bernoulli(boundary_probs)
                keep_mask = torch.bernoulli(keep_probs)
            
            # Enforce basic constraints
            boundaries = self._apply_boundary_constraints(boundaries)
            
            # Enforce context limit: num_chunks + num_kept <= max_context
            boundaries, keep_mask = self._apply_context_limit(
                boundaries, keep_mask, max_context
            )
            
            # Compute log probability using LOGITS (numerically stable)
            log_probs = self._compute_log_prob_from_logits(
                boundaries, keep_mask,
                policy_output.boundary_logits, policy_output.keep_logits
            )
            
            samples.append(PolicySample(
                boundaries=boundaries,
                keep_mask=keep_mask,
                log_probs=log_probs,
                boundary_logits=policy_output.boundary_logits,
                keep_logits=policy_output.keep_logits,
                boundary_probs=policy_output.boundary_probs,
                keep_probs=policy_output.keep_probs,
            ))
        
        return samples, policy_output
    
    def _apply_boundary_constraints(
        self,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply minimal boundary constraints.
        First position is never a boundary.
        """
        boundaries[:, 0] = 0
        return boundaries
    
    def _apply_context_limit(
        self,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        max_context: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enforce: num_chunks + num_kept_tokens <= max_context
        
        Strategy: If over budget, reduce kept tokens first (they're optional),
        then reduce chunks if still needed.
        
        OPTIMIZED: Still has loop over batch but minimized work inside loop.
        Full vectorization is complex due to variable sparsity patterns.
        """
        B, T = boundaries.shape
        device = boundaries.device
        
        # Count chunks and kept tokens (vectorized)
        num_chunks = boundaries.sum(dim=-1) + 1  # (B,) +1 for initial segment
        num_kept = keep_mask.sum(dim=-1)  # (B,)
        total = num_chunks + num_kept  # (B,)
        
        # Quick check: if no batch element exceeds limit, return early
        over_budget = total > max_context
        if not over_budget.any():
            return boundaries, keep_mask
        
        # Only process batch elements that are over budget
        over_indices = over_budget.nonzero(as_tuple=True)[0]
        
        for b in over_indices.tolist():
            excess = int(total[b].item() - max_context)
            
            # Reduce kept tokens first
            if num_kept[b] > 0 and excess > 0:
                kept_indices = keep_mask[b].nonzero(as_tuple=True)[0]
                num_to_remove = min(excess, len(kept_indices))
                # Remove from the end (arbitrary but consistent)
                remove_indices = kept_indices[-num_to_remove:]
                keep_mask[b, remove_indices] = 0
                excess -= num_to_remove
            
            # If still over, remove boundaries
            if excess > 0 and num_chunks[b] > 1:
                boundary_indices = boundaries[b].nonzero(as_tuple=True)[0]
                num_to_remove = min(excess, len(boundary_indices))
                remove_indices = boundary_indices[-num_to_remove:]
                boundaries[b, remove_indices] = 0
        
        return boundaries, keep_mask
    
    def _compute_log_prob_from_logits(
        self,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        boundary_logits: torch.Tensor,
        keep_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of a (boundary, keep) configuration.
        
        NUMERICALLY STABLE: Uses logsigmoid instead of log(sigmoid(x)).
        
        log π(B, K|x) = Σ_t [log π(b_t|x) + log π(k_t|x)]
        
        For Bernoulli with logits:
            log P(y=1) = logsigmoid(logit) = -softplus(-logit)
            log P(y=0) = logsigmoid(-logit) = -softplus(logit)
        
        Args:
            boundaries: (B, T) binary boundary indicators
            keep_mask: (B, T) binary keep indicators
            boundary_logits: (B, T) raw boundary logits (NOT probabilities!)
            keep_logits: (B, T) raw keep logits (NOT probabilities!)
            
        Returns:
            log_probs: (B,) log probability per sequence
        """
        # Clamp logits to prevent extreme values (±20 is plenty for sigmoid)
        boundary_logits = boundary_logits.clamp(-20.0, 20.0)
        keep_logits = keep_logits.clamp(-20.0, 20.0)
        
        # NUMERICALLY STABLE log-probability computation
        # logsigmoid(x) = log(sigmoid(x)) = -softplus(-x), never overflows
        # logsigmoid(-x) = log(1 - sigmoid(x)) = -softplus(x)
        
        # Log probability of boundary decisions: 
        # P(b=1) when boundaries=1, P(b=0) when boundaries=0
        boundary_log_probs = (
            boundaries * F.logsigmoid(boundary_logits) +
            (1 - boundaries) * F.logsigmoid(-boundary_logits)
        )
        
        # Log probability of keep decisions
        keep_log_probs = (
            keep_mask * F.logsigmoid(keep_logits) +
            (1 - keep_mask) * F.logsigmoid(-keep_logits)
        )
        
        # Skip first position for boundaries (always 0, so log P = 0)
        # Sum both to get total log prob
        log_probs = (
            boundary_log_probs[:, 1:].sum(dim=-1) +
            keep_log_probs.sum(dim=-1)
        )
        
        # Normalize by sequence length for stability
        T = boundaries.size(1)
        log_probs = log_probs / max(T, 1)
        
        return log_probs
    
    def compute_grpo_loss(
        self,
        samples: list,
        rewards: torch.Tensor,
        ref_policy_output: Optional['PolicyOutput'] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO policy gradient loss with PPO-style ratio clipping.
        
        NUMERICALLY STABLE: Uses logits directly via logsigmoid.
        
        GRPO uses group-relative advantages (no value network needed):
            A_i = (r_i - mean(r)) / (std(r) + eps)
        
        PPO clipped objective:
            L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            
        where r_t = π(a|s) / π_old(a|s) is the importance sampling ratio.
        
        Args:
            samples: List of PolicySample from self.sample()
            rewards: (G, B) rewards for each sample (negative LM loss)
            ref_policy_output: PolicyOutput from reference policy forward pass
                              (with boundary_logits and keep_logits for same input)
            
        Returns:
            loss: Scalar GRPO loss
            metrics: Dictionary of training metrics
        """
        G = len(samples)
        B = rewards.size(1) if rewards.dim() > 1 else 1
        device = rewards.device
        
        # Reshape rewards to (G, B) if needed
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        
        # Validate rewards don't contain NaN (fail fast, don't mask!)
        assert not torch.isnan(rewards).any(), \
            f"NaN detected in rewards! This indicates upstream numerical issues."
        
        # Compute group-relative advantages
        mean_r = rewards.mean(dim=0, keepdim=True)  # (1, B)
        std_r = rewards.std(dim=0, keepdim=True)  # (1, B)
        
        # Handle case where all rewards are identical (std=0)
        # Use unbiased=False for single-sample std to avoid NaN
        if G == 1:
            advantages = torch.zeros_like(rewards)
        else:
            std_r = std_r.clamp(min=1e-6)
            advantages = (rewards - mean_r) / std_r  # (G, B)
        
        # Clamp advantages for stability (10 std is already extreme)
        advantages = advantages.clamp(-10.0, 10.0)
        
        # Collect current log probabilities (already computed stably in sample())
        log_probs = torch.stack([s.log_probs for s in samples], dim=0)  # (G, B)
        
        # Get clip range from config
        clip_eps = self.config.grpo_clip_range  # Typically 0.2
        
        # Compute importance sampling ratio
        if ref_policy_output is not None:
            # Compute reference policy log probs using REFERENCE policy's logits
            # This is the CORRECT way - use ref policy's own probabilities!
            ref_log_probs = []
            for sample in samples:
                ref_lp = self._compute_log_prob_from_logits(
                    sample.boundaries, sample.keep_mask,
                    ref_policy_output.boundary_logits,  # Reference policy's logits!
                    ref_policy_output.keep_logits,       # Reference policy's logits!
                )
                ref_log_probs.append(ref_lp)
            ref_log_probs = torch.stack(ref_log_probs, dim=0)  # (G, B)
            
            # Importance ratio in log space (numerically stable)
            # r = exp(log_new - log_old), but we clamp the log difference first
            log_ratio = (log_probs - ref_log_probs).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)  # (G, B)
            
            # PPO clipped objective
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            pg_loss = -torch.min(surr1, surr2).mean()
            
            # KL divergence approximation (using log ratio directly)
            kl_approx = ((ratio - 1) - log_ratio).mean()
        else:
            # No reference policy: vanilla policy gradient (REINFORCE style)
            pg_loss = -(advantages * log_probs).mean()
            ratio = torch.ones_like(log_probs)
            kl_approx = torch.tensor(0.0, device=device)
        
        # KL penalty to reference policy
        kl_loss = torch.tensor(0.0, device=device)
        if ref_policy_output is not None and self.config.grpo_kl_coef > 0:
            kl_loss = kl_approx * self.config.grpo_kl_coef
        
        # ENTROPY BONUS: Critical for preventing policy collapse!
        # Maximizing entropy encourages exploration and prevents all-0 collapse
        entropy_bonus = torch.tensor(0.0, device=device)
        entropy_weight = getattr(self.config, 'entropy_bonus_weight', 0.01)
        if entropy_weight > 0:
            # Use LOGITS for numerically stable entropy computation
            # Binary entropy: H = -[p*log(p) + (1-p)*log(1-p)]
            #                   = -[sigmoid(x)*logsigmoid(x) + sigmoid(-x)*logsigmoid(-x)]
            # But simpler: H = log(1 + exp(x)) + log(1 + exp(-x)) - x*sigmoid(x)
            # Actually, even simpler using softplus:
            # H = softplus(logit) + softplus(-logit) - |logit| * 0 ... 
            # Let's use the stable formulation:
            # H(p) = -p*log(p) - (1-p)*log(1-p) = BCE_with_logits(logit, sigmoid(logit))
            # For Bernoulli: H = softplus(x) * sigmoid(-x) + softplus(-x) * sigmoid(x)
            # Or equivalently: H = log(1 + exp(-|x|)) + |x| * sigmoid(-|x|)
            
            boundary_logits = samples[0].boundary_logits.clamp(-20.0, 20.0)
            keep_logits = samples[0].keep_logits.clamp(-20.0, 20.0)
            
            # Stable binary entropy from logits:
            # H = -p*log(p) - (1-p)*log(1-p)
            #   = -sigmoid(x)*logsigmoid(x) - sigmoid(-x)*logsigmoid(-x)
            bp = torch.sigmoid(boundary_logits)
            kp = torch.sigmoid(keep_logits)
            
            boundary_entropy = -(bp * F.logsigmoid(boundary_logits) + 
                                 (1 - bp) * F.logsigmoid(-boundary_logits))
            keep_entropy = -(kp * F.logsigmoid(keep_logits) + 
                            (1 - kp) * F.logsigmoid(-keep_logits))
            
            # We MAXIMIZE entropy by SUBTRACTING it from loss
            entropy_bonus = (boundary_entropy.mean() + keep_entropy.mean()) / 2
        
        # MINIMUM CONTEXT USAGE: Prevent degenerate "do nothing" solutions
        min_usage_penalty = torch.tensor(0.0, device=device)
        min_usage = getattr(self.config, 'min_context_usage', 0.1)
        min_usage_weight = getattr(self.config, 'min_usage_penalty_weight', 1.0)
        if min_usage_weight > 0:
            for sample in samples:
                n_chunks = sample.boundaries.sum(dim=-1) + 1  # +1 for initial segment
                n_kept = sample.keep_mask.sum(dim=-1)
                total_context = n_chunks + n_kept
                usage_ratio = total_context / self.config.max_context
                
                # Penalize if usage is below minimum
                shortfall = torch.clamp(min_usage - usage_ratio, min=0)
                min_usage_penalty = min_usage_penalty + shortfall.mean()
            min_usage_penalty = min_usage_penalty / G
        
        # Budget penalty: encourage target number of context slots used
        budget_loss = torch.tensor(0.0, device=device)
        if self.config.budget_penalty_weight > 0:
            for sample in samples:
                T = sample.boundaries.size(1)
                n_chunks = sample.boundaries.sum(dim=-1) + 1
                n_kept = sample.keep_mask.sum(dim=-1)
                total_context = n_chunks + n_kept
                target = self.config.max_context * 0.8
                budget_violation = (total_context - target).abs()
                budget_loss = budget_loss + budget_violation.mean()
            budget_loss = budget_loss / G
        
        # Confidence loss: push probabilities toward 0 or 1 (minimize entropy)
        # WARNING: Can cause collapse if used without entropy bonus
        confidence_loss = torch.tensor(0.0, device=device)
        confidence_weight = getattr(self.config, 'confidence_loss_weight', 0.0)
        if confidence_weight > 0:
            # Reuse entropy computation from above if available
            if entropy_weight > 0:
                # Already computed boundary_entropy and keep_entropy
                confidence_loss = (boundary_entropy.mean() + keep_entropy.mean()) / 2
            else:
                # Compute stable entropy from logits
                boundary_logits = samples[0].boundary_logits.clamp(-20.0, 20.0)
                keep_logits = samples[0].keep_logits.clamp(-20.0, 20.0)
                bp = torch.sigmoid(boundary_logits)
                kp = torch.sigmoid(keep_logits)
                
                b_ent = -(bp * F.logsigmoid(boundary_logits) + 
                         (1 - bp) * F.logsigmoid(-boundary_logits))
                k_ent = -(kp * F.logsigmoid(keep_logits) + 
                         (1 - kp) * F.logsigmoid(-keep_logits))
                confidence_loss = (b_ent.mean() + k_ent.mean()) / 2
        
        # Total loss: pg_loss + penalties - entropy_bonus (we MAXIMIZE entropy)
        total_loss = (
            pg_loss 
            + self.config.budget_penalty_weight * budget_loss 
            + confidence_weight * confidence_loss
            + min_usage_weight * min_usage_penalty
            - entropy_weight * entropy_bonus  # Subtract to maximize entropy
        )
        
        # Validate no NaN in final loss (fail fast!)
        assert not torch.isnan(total_loss), \
            f"NaN in policy loss! pg={pg_loss.item():.4f}, budget={budget_loss.item():.4f}, " \
            f"entropy={entropy_bonus.item():.4f}. Check upstream computations."
        
        # Compute clipping fraction for monitoring
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
        
        # Metrics
        mean_chunks = torch.stack([s.boundaries.sum(dim=-1) + 1 for s in samples]).float().mean().item()
        mean_kept = torch.stack([s.keep_mask.sum(dim=-1) for s in samples]).float().mean().item()
        
        # Average tokens per chunk: (seq_len - kept_tokens) / num_chunks
        seq_len = samples[0].boundaries.size(1)
        tokens_in_chunks = seq_len - mean_kept
        avg_tokens_per_chunk = tokens_in_chunks / max(mean_chunks, 1.0)  # Avoid div by zero
        
        metrics = {
            'policy/pg_loss': pg_loss.item(),
            'policy/kl_approx': kl_approx.item() if torch.is_tensor(kl_approx) else kl_approx,
            'policy/budget_loss': budget_loss.item(),
            'policy/confidence_loss': confidence_loss.item(),
            'policy/entropy_bonus': entropy_bonus.item(),
            'policy/min_usage_penalty': min_usage_penalty.item(),
            'policy/total_loss': total_loss.item(),
            'policy/mean_reward': rewards.mean().item(),
            'policy/std_reward': rewards.std().item(),
            'policy/ratio_mean': ratio.mean().item(),
            'policy/ratio_std': ratio.std().item(),
            'policy/clip_fraction': clip_fraction.item(),
            'policy/mean_chunks': mean_chunks,
            'policy/mean_kept_tokens': mean_kept,
            'policy/avg_tokens_per_chunk': avg_tokens_per_chunk,
            'policy/context_utilization': (mean_chunks + mean_kept) / self.config.max_context,
            'policy/mean_advantage': advantages.mean().item(),
            'policy/boundary_prob_mean': samples[0].boundary_probs.mean().item(),
            'policy/boundary_prob_std': samples[0].boundary_probs.std().item(),
            'policy/keep_prob_mean': samples[0].keep_probs.mean().item(),
        }
        
        return total_loss, metrics
    
    def get_boundaries_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic boundaries for inference.
        
        Args:
            x: (B, T, D) input embeddings
            
        Returns:
            boundaries: (B, T) binary boundary indicators
        """
        samples, _ = self.sample(x, num_samples=1, deterministic=True)
        return samples[0].boundaries


class BoundaryPolicyWithProjection(nn.Module):
    """
    Boundary policy with input projection from pretrained model dimension.
    
    This wraps BoundaryPolicy with a projection layer to handle the dimension
    mismatch between the pretrained model's hidden size and the policy's chunk_dim.
    """
    
    def __init__(self, config: ContextExtenderConfig, pretrained_dim: int):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        
        # Project from pretrained dim to chunk dim
        self.input_proj = nn.Linear(pretrained_dim, config.chunk_dim)
        self.input_norm = nn.LayerNorm(config.chunk_dim)
        
        # Core policy
        self.policy = BoundaryPolicy(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """Forward pass with projection."""
        x = self.input_proj(x)
        x = self.input_norm(x)
        return self.policy.forward(x, attention_mask=attention_mask)
    
    def sample(self, x: torch.Tensor, **kwargs) -> Tuple[list, PolicyOutput]:
        """Sample with projection."""
        x = self.input_proj(x)
        x = self.input_norm(x)
        return self.policy.sample(x, **kwargs)
    
    def compute_grpo_loss(self, *args, **kwargs):
        """Delegate to policy."""
        return self.policy.compute_grpo_loss(*args, **kwargs)
    
    def _apply_context_limit(self, *args, **kwargs):
        """Delegate to policy."""
        return self.policy._apply_context_limit(*args, **kwargs)
    
    def get_boundaries_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Get deterministic boundaries with projection."""
        x = self.input_proj(x)
        x = self.input_norm(x)
        return self.policy.get_boundaries_deterministic(x)

