"""Learnable chunk boundary detection with O(T) learned value function."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.s4 import SSMLayer


class BoundaryScorer(nn.Module):
    """Scores potential boundary positions based on SSM state transitions."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(config.boundary_temperature_init).log()
        )

    def forward(self, ssm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute boundary scores from SSM state transitions.
        
        Args:
            ssm_outputs: (B, T, D) SSM hidden states
            
        Returns:
            raw_probs: (B, T) raw boundary probabilities (before budget constraint)
            temperature: Current temperature value
        """
        B, T, D = ssm_outputs.shape
        device = ssm_outputs.device

        # Concatenate current and previous state to detect transitions
        prev_states = torch.cat([
            torch.zeros(B, 1, D, device=device, dtype=ssm_outputs.dtype),
            ssm_outputs[:, :-1, :]
        ], dim=1)

        state_pairs = torch.cat([ssm_outputs, prev_states], dim=-1)  # (B, T, 2D)
        raw_scores = self.scorer(state_pairs).squeeze(-1)  # (B, T)

        raw_scores = raw_scores.clone()
        raw_scores[:, 0] = -1e4

        temperature = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        raw_probs = torch.sigmoid(raw_scores / temperature)

        return raw_probs, temperature


class LearnedValueBackward(nn.Module):
    """
    O(T) boundary detection using learned value function.
    
    Numerically stable implementation with:
    - LayerNorm for stable value network outputs
    - Clamped logits to prevent overflow
    - Residual connection for gradient flow
    """

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Value network with LayerNorm for stability
        self.input_norm = nn.LayerNorm(config.hidden_size + 2)
        self.value_net = nn.Sequential(
            nn.Linear(config.hidden_size + 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        # Learnable mixing weight (initialized small for stability)
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))  # Start with alpha ~0.37

    def forward(
        self,
        raw_probs: torch.Tensor,
        ssm_states: torch.Tensor,
        max_chunks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute budget-constrained boundary posteriors via learned value function.
        
        Args:
            raw_probs: (B, T) raw boundary probabilities from scorer
            ssm_states: (B, T, D) SSM hidden states
            max_chunks: K, maximum number of chunks
            
        Returns:
            boundary_probs: (B, T) budget-aware boundary probabilities
            expected_chunks: (B,) expected number of chunks per sequence
        """
        B, T, D = ssm_states.shape
        device = ssm_states.device
        dtype = ssm_states.dtype
        max_boundaries = max(max_chunks - 1, 1)

        # Clamp raw probs for numerical stability
        p = raw_probs.float().clamp(min=1e-4, max=1 - 1e-4)
        
        # === Forward pass: O(T) - track expected boundaries ===
        cumsum_p = torch.cumsum(p, dim=-1)
        expected_k = cumsum_p.clamp(max=max_boundaries)
        
        # === Backward pass: O(T) - learned value function ===
        budget_frac = 1.0 - (expected_k / max_boundaries).clamp(0, 1)
        position_frac = torch.arange(T, device=device, dtype=torch.float32) / max(T - 1, 1)
        position_frac = position_frac.unsqueeze(0).expand(B, -1)
            
        # Normalize SSM states to prevent large activations
        ssm_normalized = ssm_states.float()
        
        features = torch.cat([
            ssm_normalized,
            budget_frac.unsqueeze(-1),
            position_frac.unsqueeze(-1),
        ], dim=-1)
        
        # Apply input normalization for stability
        features = self.input_norm(features)

        # Value network output (clamped for stability)
        value_logits = self.value_net(features).squeeze(-1)
        value_logits = value_logits.clamp(min=-10, max=10)  # Prevent explosion
        
        # === Combine raw probs with learned value ===
        # Use a more stable formulation: work directly with probabilities
        alpha = self.log_alpha.exp().clamp(min=0.01, max=2.0)  # Smaller range
        
        # Convert raw probs to logits safely
        raw_logits = torch.logit(p, eps=1e-4).clamp(min=-10, max=10)

        # Combine with clamping
        combined_logits = raw_logits + alpha * value_logits
        combined_logits = combined_logits.clamp(min=-15, max=15)

        # === Budget-aware masking ===
        budget_headroom = (max_boundaries - expected_k).clamp(min=0)
        budget_gate = torch.sigmoid(budget_headroom * 3.0 - 0.5)  # Softer gating
        
        # Final boundary probabilities
        boundary_probs = torch.sigmoid(combined_logits) * budget_gate
        
        # First position is never a boundary
        boundary_probs = torch.cat([
            torch.zeros(B, 1, device=device, dtype=torch.float32),
            boundary_probs[:, 1:]
        ], dim=1)
        
        # Handle any remaining NaN/Inf (safety net)
        boundary_probs = torch.where(
            torch.isnan(boundary_probs) | torch.isinf(boundary_probs),
            torch.zeros_like(boundary_probs),
            boundary_probs
        )

        boundary_probs = boundary_probs.clamp(min=0, max=1)
        expected_chunks = 1 + boundary_probs.sum(dim=-1)

        return boundary_probs.to(dtype), expected_chunks


class BoundaryDetector(nn.Module):
    """Learnable boundary detection with O(T) learned value function."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        self.ssm_layers = nn.ModuleList([
            SSMLayer(config) for _ in range(config.n_boundary_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.scorer = BoundaryScorer(config)
        self.learned_backward = LearnedValueBackward(config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect boundaries with budget-constrained chunking.
        
        Args:
            x: Token embeddings (B, T, D)
            
        Returns:
            boundary_probs: (B, T) boundary probabilities
            boundary_decisions: (B, T) hard decisions
            ssm_output: (B, T, D) SSM hidden states
            expected_chunks: (B,) expected number of chunks
            entropy_loss: Scalar entropy loss
            sparsity_loss: Scalar sparsity loss
        """
        B, T, D = x.shape
        device = x.device

        # Run SSM layers
        h = x
        for layer in self.ssm_layers:
            h, _ = layer(h, return_all_states=False)
        ssm_output = self.norm(h)

        # Get raw boundary probabilities
        raw_probs, temperature = self.scorer(ssm_output)

        # Learned value backward for budget-constrained posteriors
        boundary_probs, expected_chunks = self.learned_backward(
            raw_probs, ssm_output, self.config.max_chunks
        )

        # Hard decisions with straight-through estimator
        boundary_hard = (boundary_probs > 0.5).float()
        boundary_decisions = boundary_hard + boundary_probs - boundary_probs.detach()

        # Entropy loss
        eps = 1e-6
        probs_for_entropy = boundary_probs[:, 1:]
        entropy = -(
            probs_for_entropy * torch.log(probs_for_entropy + eps) +
            (1 - probs_for_entropy) * torch.log(1 - probs_for_entropy + eps)
        )
        entropy_loss = entropy.mean()

        # Sparsity loss
        k = min(self.config.max_chunks - 1, T - 1)
        if k > 0 and T > 1:
            sorted_probs, _ = torch.sort(probs_for_entropy, dim=-1, descending=True)
            top_k_probs = sorted_probs[:, :k]
            sparsity_loss = torch.relu(0.6 - top_k_probs.mean())
        else:
            sparsity_loss = torch.tensor(0.0, device=x.device)

        return (
            boundary_probs,
            boundary_decisions,
            ssm_output,
            expected_chunks,
            entropy_loss,
            sparsity_loss,
        )

    def compute_chunk_assignments(self, boundary_decisions: torch.Tensor) -> torch.Tensor:
        """Convert boundary decisions to chunk IDs via cumsum."""
        return torch.cumsum(boundary_decisions, dim=-1)
