"""Unified Policy-Compressor Network with Importance-Threshold Selection.

This module implements a policy that learns:
    1. IMPORTANCE scores for each position (how valuable is a boundary here?)
    2. THRESHOLD for boundary selection (what's "important enough"?)
    3. HOW to compress chunks (compression head)

The key insight: boundaries COMPETE for limited slots. Instead of independent
probabilities, we learn a ranking. Positions above the threshold become boundaries,
with a hard cap at max_context.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Shared SSM Backbone + Position Encoding                        │
    └─────────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
    ┌───────────┐       ┌───────────┐        ┌─────────────┐
    │ Importance│       │ Threshold │        │ Compression │
    │   Head    │       │   Head    │        │    Head     │
    └───────────┘       └───────────┘        └─────────────┘
         ↓                    ↓                    ↓
    importance[t]        threshold           chunk_embeds
    
    boundary[t] = importance[t] > threshold (with top-K cap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import ContextExtenderConfig


class PolicySample(NamedTuple):
    """A sampled boundary configuration."""
    boundaries: torch.Tensor      # (B, T) binary: 1 = boundary here
    keep_mask: torch.Tensor       # (B, T) binary: 1 = keep this token
    log_probs: torch.Tensor       # (B,) log probability of this sample
    boundary_probs: torch.Tensor  # (B, T) boundary probabilities
    keep_probs: torch.Tensor      # (B, T) keep probabilities


@dataclass
class PolicyOutput:
    """Output from policy forward pass."""
    boundary_importance: torch.Tensor  # (B, T) importance scores
    boundary_threshold: torch.Tensor   # (B,) per-sequence threshold
    boundary_probs: torch.Tensor       # (B, T) soft selection probabilities
    keep_importance: torch.Tensor      # (B, T) keep importance scores
    keep_threshold: torch.Tensor       # (B,) per-sequence keep threshold
    keep_probs: torch.Tensor           # (B, T) soft keep probabilities
    hidden_states: torch.Tensor        # (B, T, D) for compression
    difficulty_scores: torch.Tensor    # (B, T) compression difficulty


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding with normalized positions [0, 1]."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Precompute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (T,) or (B, T) normalized positions in [0, 1]
        Returns:
            (T, D) or (B, T, D) position embeddings
        """
        # Scale to reasonable range for sinusoids
        positions = positions * 1000  # Scale up for frequency variation
        
        if positions.dim() == 1:
            # (T,) -> (T, D//2)
            sinusoid_inp = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        else:
            # (B, T) -> (B, T, D//2)
            sinusoid_inp = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        
        # Interleave sin and cos
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class PolicyCompressor(nn.Module):
    """
    Policy network using importance-threshold selection.
    
    Key features:
    1. Position-aware: knows where each token is in the sequence
    2. Budget-aware: learns threshold to control boundary count
    3. At most K: hard cap ensures never exceeds budget
    4. Adaptive: threshold varies per sequence based on content
    """
    
    def __init__(self, config: ContextExtenderConfig):
        super().__init__()
        self.config = config
        
        from wavgpt.models.ssm import SSMBackbone
        
        # Shared SSM backbone
        self.backbone = SSMBackbone(
            d_model=config.chunk_dim,
            n_layers=config.n_ssm_layers,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        
        # Position encoding
        self.pos_encoder = SinusoidalPositionEncoding(config.chunk_dim)
        self.pos_proj = nn.Linear(config.chunk_dim, config.chunk_dim)
        
        # Importance head: how important is a boundary at each position?
        self.importance_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Threshold head: what's the bar for "important enough"?
        # Takes global representation -> per-sequence threshold
        self.threshold_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Keep token heads (same structure)
        self.keep_importance_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        self.keep_threshold_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Compression head
        self.compression_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.chunk_dim),
            nn.LayerNorm(config.chunk_dim),
        )
        
        # Difficulty prediction for credit assignment
        self.difficulty_head = nn.Sequential(
            nn.Linear(config.chunk_dim, config.policy_hidden_dim),
            nn.GELU(),
            nn.Linear(config.policy_hidden_dim, 1),
        )
        
        # Temperature for soft selection (learnable)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        
        # Lagrangian price for budget constraint (learnable)
        # λ = exp(log_λ) ensures λ > 0
        # Cost = λ * expected_boundaries penalizes using budget
        # λ self-adjusts: increases when over budget, decreases when under
        self.log_lambda = nn.Parameter(torch.tensor(0.0))  # Start at λ=1
        
        # Initialize thresholds to reasonable values
        self._init_thresholds()
    
    def _init_thresholds(self):
        """Initialize threshold heads to produce reasonable initial thresholds."""
        # Start with threshold ≈ 0, so about half of positions are above
        with torch.no_grad():
            self.threshold_head[-1].bias.fill_(0.0)
            self.keep_threshold_head[-1].bias.fill_(1.0)  # Keep tokens more conservatively
    
    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature for numerical stability."""
        return self.log_temperature.exp().clamp(0.1, 10.0)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Forward pass: compute importance scores and thresholds.
        
        Args:
            x: (B, T, D) input embeddings
            attention_mask: (B, T) optional mask for valid tokens
            
        Returns:
            PolicyOutput with importance, threshold, and probs
        """
        B, T, D = x.shape
        device = x.device
        
        # Process through backbone
        hidden = self.backbone(x)  # (B, T, D)
        
        # Add position encoding (normalized positions [0, 1])
        positions = torch.arange(T, device=device, dtype=x.dtype) / max(T - 1, 1)
        pos_emb = self.pos_encoder(positions)  # (T, D)
        pos_emb = self.pos_proj(pos_emb)  # (T, D)
        
        hidden_with_pos = hidden + pos_emb.unsqueeze(0)  # (B, T, D)
        
        # Compute importance scores
        boundary_importance = self.importance_head(hidden_with_pos).squeeze(-1)  # (B, T)
        keep_importance = self.keep_importance_head(hidden_with_pos).squeeze(-1)  # (B, T)
        
        # Compute per-sequence thresholds from global representation
        global_repr = hidden.mean(dim=1)  # (B, D)
        boundary_threshold = self.threshold_head(global_repr).squeeze(-1)  # (B,)
        keep_threshold = self.keep_threshold_head(global_repr).squeeze(-1)  # (B,)
        
        # Soft boundary probabilities: sigmoid((importance - threshold) / temperature)
        temp = self.temperature
        boundary_diff = boundary_importance - boundary_threshold.unsqueeze(1)  # (B, T)
        boundary_probs = torch.sigmoid(boundary_diff / temp)
        
        keep_diff = keep_importance - keep_threshold.unsqueeze(1)
        keep_probs = torch.sigmoid(keep_diff / temp)
        
        # Mask first position (never a boundary) and padded positions
        boundary_probs = boundary_probs.clone()
        boundary_probs[:, 0] = 0.0
        
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
            boundary_probs = boundary_probs.masked_fill(padding_mask, 0.0)
            keep_probs = keep_probs.masked_fill(padding_mask, 0.0)
        
        # Difficulty scores
        difficulty_scores = self.difficulty_head(hidden).squeeze(-1)
        
        return PolicyOutput(
            boundary_importance=boundary_importance,
            boundary_threshold=boundary_threshold,
            boundary_probs=boundary_probs,
            keep_importance=keep_importance,
            keep_threshold=keep_threshold,
            keep_probs=keep_probs,
            hidden_states=hidden,
            difficulty_scores=difficulty_scores,
        )
    
    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
        deterministic: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list, PolicyOutput]:
        """
        Sample boundary configurations.
        
        Args:
            x: (B, T, D) input embeddings
            num_samples: Number of configurations to sample
            temperature: Additional temperature scaling
            deterministic: If True, use threshold directly (no sampling)
            attention_mask: (B, T) optional mask
            
        Returns:
            samples: List of PolicySample
            policy_output: PolicyOutput from forward pass
        """
        policy_output = self.forward(x, attention_mask=attention_mask)
        B, T = policy_output.boundary_probs.shape
        
        # Scale probs by additional temperature
        if temperature != 1.0:
            boundary_probs = self._rescale_probs(
                policy_output.boundary_importance,
                policy_output.boundary_threshold,
                temperature * self.temperature
            )
            keep_probs = self._rescale_probs(
                policy_output.keep_importance,
                policy_output.keep_threshold,
                temperature * self.temperature
            )
            boundary_probs[:, 0] = 0.0
        else:
            boundary_probs = policy_output.boundary_probs
            keep_probs = policy_output.keep_probs
        
        samples = []
        for _ in range(num_samples):
            if deterministic:
                # Hard threshold: importance > threshold
                boundaries = (policy_output.boundary_importance > 
                            policy_output.boundary_threshold.unsqueeze(1)).float()
                keep_mask = (policy_output.keep_importance > 
                           policy_output.keep_threshold.unsqueeze(1)).float()
                boundaries[:, 0] = 0.0
            else:
                # Sample from soft probabilities
                boundaries = torch.bernoulli(boundary_probs)
                keep_mask = torch.bernoulli(keep_probs)
            
            # Apply budget constraint: at most max_context total
            boundaries, keep_mask = self._apply_budget_constraint(
                boundaries, keep_mask, 
                policy_output.boundary_importance,
                policy_output.keep_importance,
            )
            
            # Compute log probability
            log_probs = self._compute_log_prob(
                boundaries, keep_mask,
                boundary_probs, keep_probs,
            )
            
            samples.append(PolicySample(
                boundaries=boundaries,
                keep_mask=keep_mask,
                log_probs=log_probs,
                boundary_probs=boundary_probs,
                keep_probs=keep_probs,
            ))
        
        return samples, policy_output
    
    def _rescale_probs(
        self,
        importance: torch.Tensor,
        threshold: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute probs with different temperature."""
        diff = importance - threshold.unsqueeze(1)
        return torch.sigmoid(diff / temperature)
    
    def _apply_budget_constraint(
        self,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        boundary_importance: torch.Tensor,
        keep_importance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure total context (boundaries + kept tokens) <= max_context.
        
        Uses importance-based selection when over budget:
        highest importance positions are kept.
        """
        B, T = boundaries.shape
        max_context = self.config.max_context
        
        for b in range(B):
            # Count current usage
            num_boundaries = boundaries[b].sum().int().item()
            num_kept = keep_mask[b].sum().int().item()
            # Each boundary creates a chunk, so num_chunks = num_boundaries + 1
            total_context = (num_boundaries + 1) + num_kept
            
            if total_context <= max_context:
                continue
            
            excess = total_context - max_context
            
            # First, reduce kept tokens (they're optional)
            if num_kept > 0 and excess > 0:
                kept_idx = keep_mask[b].nonzero(as_tuple=True)[0]
                kept_importance_vals = keep_importance[b, kept_idx]
                
                # Remove lowest-importance kept tokens
                num_to_remove = min(excess, len(kept_idx))
                _, remove_order = kept_importance_vals.sort()
                remove_idx = kept_idx[remove_order[:num_to_remove]]
                keep_mask[b, remove_idx] = 0
                excess -= num_to_remove
            
            # Then, reduce boundaries if still over
            if excess > 0 and num_boundaries > 0:
                boundary_idx = boundaries[b].nonzero(as_tuple=True)[0]
                boundary_importance_vals = boundary_importance[b, boundary_idx]
                
                # Remove lowest-importance boundaries
                num_to_remove = min(excess, len(boundary_idx))
                _, remove_order = boundary_importance_vals.sort()
                remove_idx = boundary_idx[remove_order[:num_to_remove]]
                boundaries[b, remove_idx] = 0
        
        return boundaries, keep_mask
    
    def _compute_log_prob(
        self,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        boundary_probs: torch.Tensor,
        keep_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of a (boundary, keep) configuration."""
        # Clamp probs for numerical stability
        eps = 1e-7
        bp = boundary_probs.clamp(eps, 1 - eps)
        kp = keep_probs.clamp(eps, 1 - eps)
        
        # Log prob under Bernoulli
        boundary_log_probs = (
            boundaries * bp.log() + (1 - boundaries) * (1 - bp).log()
        )
        keep_log_probs = (
            keep_mask * kp.log() + (1 - keep_mask) * (1 - kp).log()
        )
        
        # Sum over positions (skip first position for boundaries)
        log_probs = boundary_log_probs[:, 1:].sum(dim=-1) + keep_log_probs.sum(dim=-1)
        
        # Normalize by sequence length for stability
        T = boundaries.size(1)
        log_probs = log_probs / max(T, 1)
        
        return log_probs
    
    def compress(
        self,
        hidden_states: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress tokens into chunk embeddings at boundary positions.
        
        Args:
            hidden_states: (B, T, D) from forward()
            boundaries: (B, T) binary boundary indicators
            attention_mask: (B, T) optional
            keep_mask: (B, T) tokens to exclude (unused, kept for interface)
            
        Returns:
            chunk_embeddings: (B, K, D)
            chunk_mask: (B, K)
            difficulty: (B, K)
        """
        B, T, D = hidden_states.shape
        K = self.config.max_chunks
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Project through compression head
        compressed = self.compression_head(hidden_states)
        difficulty_scores = self.difficulty_head(hidden_states).squeeze(-1)
        
        # Extract embeddings at boundary positions
        chunk_embeddings = torch.zeros(B, K, D, device=device, dtype=dtype)
        chunk_mask = torch.zeros(B, K, device=device, dtype=dtype)
        chunk_difficulty = torch.zeros(B, K, device=device, dtype=dtype)
        
        for b in range(B):
            boundary_positions = boundaries[b].nonzero(as_tuple=True)[0]
            
            # Add final position as implicit boundary
            final_pos = torch.tensor([T - 1], device=device)
            if len(boundary_positions) == 0:
                all_boundaries = final_pos
            elif boundary_positions[-1] != T - 1:
                all_boundaries = torch.cat([boundary_positions, final_pos])
            else:
                all_boundaries = boundary_positions
            
            num_chunks = min(len(all_boundaries), K)
            chunk_embeddings[b, :num_chunks] = compressed[b, all_boundaries[:num_chunks]]
            chunk_mask[b, :num_chunks] = 1.0
            chunk_difficulty[b, :num_chunks] = difficulty_scores[b, all_boundaries[:num_chunks]]
        
        chunk_mask[:, 0] = 1.0  # Always have at least one chunk
        
        return chunk_embeddings, chunk_mask, chunk_difficulty
    
    def compute_grpo_loss(
        self,
        samples: list,
        rewards: torch.Tensor,
        ref_policy_output: Optional[PolicyOutput] = None,
        chunk_difficulties: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO policy gradient loss.
        
        Args:
            samples: List of PolicySample
            rewards: (G, B) rewards for each sample
            ref_policy_output: Optional reference policy for KL
            chunk_difficulties: (G, B, K) difficulty scores
        """
        G = len(samples)
        B = rewards.size(1) if rewards.dim() > 1 else 1
        device = rewards.device
        
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        
        # Group-relative advantages
        mean_r = rewards.mean(dim=0, keepdim=True)
        std_r = rewards.std(dim=0, keepdim=True).clamp(min=1e-6)
        
        if G == 1:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = ((rewards - mean_r) / std_r).clamp(-10.0, 10.0)
        
        # Collect log probabilities
        log_probs = torch.stack([s.log_probs for s in samples], dim=0)
        
        # Policy gradient loss
        if ref_policy_output is not None:
            # With reference policy (PPO-style clipping)
            ref_log_probs = []
            for sample in samples:
                ref_bp = self._compute_ref_probs(
                    sample.boundaries,
                    ref_policy_output.boundary_importance,
                    ref_policy_output.boundary_threshold,
                )
                ref_kp = self._compute_ref_probs(
                    sample.keep_mask,
                    ref_policy_output.keep_importance,
                    ref_policy_output.keep_threshold,
                )
                ref_lp = self._compute_log_prob(
                    sample.boundaries, sample.keep_mask, ref_bp, ref_kp
                )
                ref_log_probs.append(ref_lp)
            ref_log_probs = torch.stack(ref_log_probs, dim=0)
            
            log_ratio = (log_probs - ref_log_probs).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)
            
            clip_eps = self.config.grpo_clip_range
            clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            kl_approx = ((ratio - 1) - log_ratio).mean()
        else:
            pg_loss = -(advantages * log_probs).mean()
            ratio = torch.ones_like(log_probs)
            kl_approx = torch.tensor(0.0, device=device)
        
        # Entropy bonus (encourages exploration)
        entropy_bonus = torch.tensor(0.0, device=device)
        entropy_weight = getattr(self.config, 'entropy_bonus_weight', 0.01)
        if entropy_weight > 0:
            bp = samples[0].boundary_probs.clamp(1e-7, 1 - 1e-7)
            kp = samples[0].keep_probs.clamp(1e-7, 1 - 1e-7)
            boundary_entropy = -(bp * bp.log() + (1 - bp) * (1 - bp).log())
            keep_entropy = -(kp * kp.log() + (1 - kp) * (1 - kp).log())
            entropy_bonus = (boundary_entropy.mean() + keep_entropy.mean()) / 2
        
        # ================================================================
        # LOG BARRIER BUDGET CONSTRAINT
        # ================================================================
        # Uses interior point method: -log(K - expected) creates infinite
        # penalty as expected approaches K, keeping the model in the
        # feasible region (expected < K) at all times.
        #
        # Combined with Lagrangian price λ for smooth optimization.
        # ================================================================
        
        λ = self.log_lambda.exp()  # Barrier strength (self-adjusting)
        max_context = self.config.max_context
        
        # Compute expected context usage across samples
        expected_total = torch.tensor(0.0, device=device)
        for sample in samples:
            expected_boundaries = sample.boundary_probs.sum(dim=-1)  # (B,)
            expected_kept = sample.keep_probs.sum(dim=-1)
            expected_total = expected_total + (expected_boundaries + expected_kept).mean()
        expected_total = expected_total / G
        
        # Log barrier: -log(slack) where slack = K - expected
        # Goes to infinity as expected approaches K from below
        # We use a small buffer (0.95 * K) to keep some margin
        effective_limit = max_context * 0.95  # Target 95% utilization max
        slack = effective_limit - expected_total
        
        # Clamp slack to prevent log(0) or log(negative)
        # If over budget, slack is negative -> use large penalty instead
        min_slack = 1.0  # Minimum slack to prevent log explosion
        
        if slack > min_slack:
            # Interior: use log barrier
            # Normalized so penalty is ~0 when well under budget
            log_barrier = -λ * torch.log(slack / effective_limit)
        else:
            # Exterior (over budget): massive penalty to push back
            # Linear extrapolation from the barrier at min_slack
            overshoot = min_slack - slack
            barrier_at_min = -λ * torch.log(torch.tensor(min_slack / effective_limit, device=device))
            gradient_at_min = λ / min_slack  # Derivative of -log(x)
            log_barrier = barrier_at_min + gradient_at_min * overshoot * 10.0  # 10x steeper outside
        
        # Dual update: adjust λ based on constraint satisfaction
        # Higher λ = stronger barrier = stay further from limit
        with torch.no_grad():
            utilization = expected_total / max_context
            if utilization > 0.9:  # Getting close to limit
                self.log_lambda.add_(0.05)  # Strengthen barrier
            elif utilization < 0.5 and self.log_lambda.exp() > 0.5:
                self.log_lambda.sub_(0.02)  # Relax barrier
        
        # Total loss
        total_loss = (
            pg_loss 
            - entropy_weight * entropy_bonus 
            + log_barrier
        )
        
        # Metrics
        mean_chunks = sum((s.boundaries.sum(dim=-1) + 1).float().mean().item() for s in samples) / G
        mean_kept = sum(s.keep_mask.sum(dim=-1).float().mean().item() for s in samples) / G
        
        metrics = {
            'policy/pg_loss': pg_loss.item(),
            'policy/entropy_bonus': entropy_bonus.item(),
            'policy/log_barrier': log_barrier.item(),
            'policy/barrier_strength': λ.item(),
            'policy/expected_context': expected_total.item(),
            'policy/budget_slack': slack.item() if torch.is_tensor(slack) else slack,
            'policy/total_loss': total_loss.item(),
            'policy/mean_reward': rewards.mean().item(),
            'policy/mean_chunks': mean_chunks,
            'policy/mean_kept_tokens': mean_kept,
            'policy/context_utilization': (mean_chunks + mean_kept) / self.config.max_context,
            'policy/temperature': self.temperature.item(),
            'policy/kl_approx': kl_approx.item() if torch.is_tensor(kl_approx) else kl_approx,
        }
        
        return total_loss, metrics
    
    def _compute_ref_probs(
        self,
        decisions: torch.Tensor,
        importance: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        """Compute probabilities under reference policy."""
        diff = importance - threshold.unsqueeze(1)
        return torch.sigmoid(diff / self.temperature)


class PolicyCompressorWithProjection(nn.Module):
    """
    PolicyCompressor with input projection from pretrained model dimension.
    """
    
    def __init__(self, config: ContextExtenderConfig, pretrained_dim: int):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        
        self.input_proj = nn.Linear(pretrained_dim, config.chunk_dim)
        self.input_norm = nn.LayerNorm(config.chunk_dim)
        self.core = PolicyCompressor(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        x = self.input_proj(x)
        x = self.input_norm(x)
        return self.core.forward(x, attention_mask=attention_mask)
    
    def compress(
        self,
        hidden_states: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.core.compress(hidden_states, boundaries, attention_mask, keep_mask)
    
    def compress_from_embeddings(
        self,
        embeddings: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(embeddings)
        x = self.input_norm(x)
        hidden = self.core.backbone(x)
        return self.core.compress(hidden, boundaries, attention_mask, keep_mask)
    
    def sample(self, x: torch.Tensor, **kwargs) -> Tuple[list, PolicyOutput]:
        x = self.input_proj(x)
        x = self.input_norm(x)
        return self.core.sample(x, **kwargs)
    
    def compute_grpo_loss(self, *args, **kwargs):
        return self.core.compute_grpo_loss(*args, **kwargs)
