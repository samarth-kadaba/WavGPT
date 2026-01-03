"""Learnable chunk boundary detection with O(T) learned value function."""

from __future__ import annotations

from typing import List, Tuple

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
    """O(T) boundary detection using learned value function."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.value_net = nn.Sequential(
            nn.Linear(config.hidden_size + 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

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
        
        p = raw_probs.float().clamp(min=1e-6, max=1 - 1e-6)
        
        # Forward pass: track expected boundaries
        cumsum_p = torch.cumsum(p, dim=-1)
        expected_k = cumsum_p.clamp(max=max_boundaries)
        
        # Backward pass: learned value function
        budget_frac = 1.0 - (expected_k / max_boundaries).clamp(0, 1)
        position_frac = torch.arange(T, device=device, dtype=torch.float32) / max(T - 1, 1)
        position_frac = position_frac.unsqueeze(0).expand(B, -1)
        
        features = torch.cat([
            ssm_states.float(),
            budget_frac.unsqueeze(-1),
            position_frac.unsqueeze(-1),
        ], dim=-1)
        
        value_logits = self.value_net(features).squeeze(-1)
        
        # Combine raw probs with learned value
        alpha = self.log_alpha.exp().clamp(min=0.1, max=10.0)
        raw_logits = (p / (1 - p + 1e-8)).log().clamp(min=-20, max=20)
        combined_logits = raw_logits + alpha * value_logits
        
        # Budget-aware masking
        budget_headroom = (max_boundaries - expected_k).clamp(min=0)
        budget_gate = torch.sigmoid(budget_headroom * 5.0 - 1.0)
        
        boundary_probs = torch.sigmoid(combined_logits) * budget_gate
        
        # First position is never a boundary
        boundary_probs = torch.cat([
            torch.zeros(B, 1, device=device, dtype=torch.float32),
            boundary_probs[:, 1:]
        ], dim=1)
        
        boundary_probs = torch.where(
            torch.isnan(boundary_probs) | torch.isinf(boundary_probs),
            torch.zeros_like(boundary_probs),
            boundary_probs
        )
        
        boundary_probs = boundary_probs.clamp(min=0, max=1)
        expected_chunks = 1 + boundary_probs.sum(dim=-1)
        
        return boundary_probs.to(dtype), expected_chunks


class AmortizedBoundaryPredictor(nn.Module):
    """Predicts boundaries using forward-only information for generation."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size + 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(
        self,
        ssm_states: torch.Tensor,
        budget_frac: torch.Tensor,
        position_frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict boundary probabilities from forward-only information.
        
        Args:
            ssm_states: (B, T, D) or (B, D) SSM hidden states
            budget_frac: (B, T) or (B,) remaining budget fraction
            position_frac: (B, T) or (B,) position fraction in sequence
            
        Returns:
            boundary_probs: (B, T) or (B,) predicted boundary probabilities
        """
        # Handle both batched sequence and single-step cases
        if ssm_states.dim() == 2:
            ssm_states = ssm_states.unsqueeze(1)
            budget_frac = budget_frac.unsqueeze(1)
            position_frac = position_frac.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = ssm_states.shape

        features = torch.cat([
            ssm_states,
            budget_frac.unsqueeze(-1),
            position_frac.unsqueeze(-1),
        ], dim=-1)  # (B, T, D+2)

        logits = self.predictor(features).squeeze(-1)
        probs = torch.sigmoid(logits)

        if T > 1:
            mask = torch.ones_like(probs)
            mask[:, 0] = 0.0
            probs = probs * mask

        if squeeze_output:
            probs = probs.squeeze(1)

        return probs

    def distillation_loss(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss for distillation."""
        pred = predicted.clamp(min=1e-7, max=1 - 1e-7)
        tgt = target.detach().clamp(min=1e-7, max=1 - 1e-7)
        bce = -(tgt * pred.log() + (1 - tgt) * (1 - pred).log())
        return bce.mean()


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
        self.amortized = AmortizedBoundaryPredictor(config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect boundaries with budget-constrained chunking.
        
        Args:
            x: Token embeddings (B, T, D)
            
        Returns:
            boundary_probs: (B, T) boundary probabilities
            boundary_decisions: (B, T) hard decisions
            ssm_output: (B, T, D) SSM hidden states
            expected_chunks: (B,) expected number of chunks
            distillation_loss: Scalar distillation loss
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

        # Train amortized predictor via distillation
        cumsum = torch.cumsum(boundary_probs, dim=-1)
        max_boundaries = self.config.max_chunks - 1
        budget_frac = (max_boundaries - cumsum).clamp(min=0) / max(max_boundaries, 1)
        position_frac = torch.arange(T, device=device, dtype=x.dtype) / max(T - 1, 1)
        position_frac = position_frac.unsqueeze(0).expand(B, -1)

        amortized_probs = self.amortized(ssm_output, budget_frac, position_frac)
        distill_loss = self.amortized.distillation_loss(amortized_probs, boundary_probs)

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
            distill_loss,
            entropy_loss,
            sparsity_loss,
        )

    def compute_chunk_assignments(self, boundary_decisions: torch.Tensor) -> torch.Tensor:
        """Convert boundary decisions to chunk IDs via cumsum."""
        return torch.cumsum(boundary_decisions, dim=-1)

    def step(
        self,
        token_embed: torch.Tensor,
        conv_states: List[torch.Tensor],
        ssm_states: List[torch.Tensor],
        n_boundaries_so_far: int,
        position: int,
        expected_length: int,
    ) -> Tuple[bool, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Incremental boundary detection for generation."""
        h = token_embed
        new_conv_states = []
        new_ssm_states = []

        for i, layer in enumerate(self.ssm_layers):
            h, new_conv, new_ssm = layer.step(h, conv_states[i], ssm_states[i])
            new_conv_states.append(new_conv)
            new_ssm_states.append(new_ssm)

        ssm_output = self.norm(h)

        max_boundaries = self.config.max_chunks - 1
        budget_frac = (max_boundaries - n_boundaries_so_far) / max(max_boundaries, 1)
        position_frac = position / max(expected_length - 1, 1)

        budget_tensor = torch.tensor([[budget_frac]], device=token_embed.device, dtype=token_embed.dtype)
        position_tensor = torch.tensor([[position_frac]], device=token_embed.device, dtype=token_embed.dtype)

        boundary_prob = self.amortized(ssm_output, budget_tensor.squeeze(-1), position_tensor.squeeze(-1))

        can_place_boundary = n_boundaries_so_far < max_boundaries
        is_boundary = can_place_boundary and (boundary_prob.item() > 0.5)

        return is_boundary, ssm_output, new_conv_states, new_ssm_states

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get initial SSM states for generation."""
        conv_states = []
        ssm_states = []
        for layer in self.ssm_layers:
            conv, ssm = layer.get_initial_state(batch_size, device)
            conv_states.append(conv)
            ssm_states.append(ssm)
        return conv_states, ssm_states
