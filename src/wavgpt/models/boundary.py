from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.s4 import SSMLayer


class BoundaryDetector(nn.Module):
    """
    Detects chunk boundaries using surprisal-based decisions from SSM outputs.

    The SSM processes all tokens to understand global context, then we compute
    surprisal (negative log likelihood) for each token. Boundaries are detected
    when the average per-token likelihood decreases (surprisal increases) with
    the addition of a token.

    Fully vectorized implementation - no iteration loops!
    """

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        # Global SSM for boundary detection
        self.ssm_layers = nn.ModuleList([SSMLayer(config) for _ in range(config.n_boundary_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        lm_head: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect boundaries from input embeddings using surprisal-based detection.

        Args:
            x: Token embeddings (B, T, D)
            input_ids: Token IDs (B, T)
            lm_head: Language model head to compute logits from SSM outputs

        Returns:
            boundary_logits: Surprisal change signals (B, T) - positive = boundary
            boundary_decisions: Hard boundary decisions (B, T) - differentiable via STE!
            ssm_output: SSM hidden states (B, T, D)
        """
        # Run global SSM
        h = x
        for layer in self.ssm_layers:
            h, _ = layer(h, return_all_states=False)
        h = self.norm(h)

        B, T, D = h.shape

        if T < 2:
            zeros = torch.zeros(B, T, device=x.device, dtype=x.dtype)
            return zeros, zeros, h

        # Compute logits from SSM outputs (vectorized)
        logits = lm_head(h)  # (B, T, vocab_size)

        # Compute log probabilities for actual tokens (vectorized)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, vocab_size)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)

        # Compute cumulative average log probability (likelihood)
        # cumsum gives sum, divide by position+1 to get average
        positions = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, -1)  # (1, T)
        cumsum_log_probs = torch.cumsum(token_log_probs, dim=-1)  # (B, T)
        avg_log_probs = cumsum_log_probs / positions  # (B, T) - average log prob up to each position

        # Compute change in average log probability (likelihood change)
        # Compare each position with the previous position's average
        # Position 0 has no previous, so we start from position 1
        prev_avg_log_probs = torch.cat(
            [torch.zeros(B, 1, device=x.device, dtype=x.dtype), avg_log_probs[:, :-1]], dim=1
        )  # (B, T)
        likelihood_change = avg_log_probs - prev_avg_log_probs  # (B, T)
        # Positive = likelihood increased (surprisal decreased) = no boundary
        # Negative = likelihood decreased (surprisal increased) = boundary

        # Boundary signal: negative likelihood change (surprisal increase)
        boundary_logits = -likelihood_change  # (B, T) - positive = boundary
        boundary_logits[:, 0] = -1e6  # First position is never a boundary (large negative)

        # Compute soft boundary probabilities (differentiable)
        boundary_probs = torch.sigmoid(boundary_logits)  # (B, T)

        # Hard decision: boundary if likelihood decreases (surprisal increases)
        # This will be used for forward pass, but gradients flow through boundary_probs
        boundary_decisions_hard = (likelihood_change < 0).float()  # (B, T)
        boundary_decisions_hard[:, 0] = 0.0  # First position is never a boundary

        # Apply max_chunks constraint in a differentiable way
        # Compute chunk assignments from soft probabilities first
        chunk_ids_soft = torch.cumsum(boundary_probs, dim=-1)  # (B, T) - soft chunk assignments
        
        # Create mask to prevent exceeding max_chunks (differentiable)
        # Use a smooth penalty: sigmoid((chunk_id - max_chunks) / temperature)
        # This creates a soft mask that approaches 0 as chunk_id approaches max_chunks
        temperature = 1.0  # Controls smoothness of the constraint
        excess_penalty = torch.sigmoid((chunk_ids_soft - self.config.max_chunks + 1) / temperature)
        # Invert so that excess_penalty is 0 when chunk_id < max_chunks, and approaches 1 when >= max_chunks
        max_chunks_mask = 1.0 - excess_penalty  # (B, T) - 1.0 when safe, 0.0 when exceeding
        
        # Apply mask to boundary probabilities (differentiable)
        boundary_probs_constrained = boundary_probs * max_chunks_mask

        # For hard decisions, also apply max_chunks constraint
        chunk_ids_hard = torch.cumsum(boundary_decisions_hard, dim=-1)  # (B, T)
        excess_mask_hard = chunk_ids_hard >= self.config.max_chunks  # (B, T)
        boundary_decisions_hard = boundary_decisions_hard * (~excess_mask_hard).float()

        # Straight-through estimator: use hard decision in forward, soft in backward
        # This allows gradients to flow through boundary_probs_constrained
        boundary_decisions = boundary_decisions_hard + boundary_probs_constrained - boundary_probs_constrained.detach()

        return boundary_logits, boundary_decisions, h

    def compute_chunk_assignments(self, boundary_decisions: torch.Tensor) -> torch.Tensor:
        """
        Convert boundary decisions to chunk assignments via cumsum.

        Args:
            boundary_decisions: (B, T) binary boundary indicators

        Returns:
            chunk_ids: (B, T) chunk indices
        """
        return torch.cumsum(boundary_decisions, dim=-1)

    def step(
        self,
        token_embed: torch.Tensor,
        prev_hidden: torch.Tensor,
        conv_states: List[torch.Tensor],
        ssm_states: List[torch.Tensor],
        token_id: torch.Tensor,
        prev_avg_log_prob: Optional[torch.Tensor],
        prev_token_count: int,
        lm_head: nn.Linear,
    ) -> Tuple[bool, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
        """
        Incremental boundary detection for generation using surprisal.

        Args:
            token_embed: Current token embedding (B, D)
            prev_hidden: Previous SSM hidden state (B, D)
            conv_states: Previous conv states
            ssm_states: Previous SSM states
            token_id: Current token ID (B,)
            prev_avg_log_prob: Previous average log probability (B,) or None
            prev_token_count: Previous token count in current chunk
            lm_head: Language model head

        Returns:
            is_boundary: Whether to start a new chunk
            curr_hidden: Current SSM hidden output (B, D)
            new_conv_states: Updated conv states
            new_ssm_states: Updated SSM states
            curr_avg_log_prob: Current average log probability (B,)
            curr_token_count: Current token count
        """
        h = token_embed
        new_conv_states = []
        new_ssm_states = []

        for i, layer in enumerate(self.ssm_layers):
            h, new_conv, new_ssm = layer.step(h, conv_states[i], ssm_states[i])
            new_conv_states.append(new_conv)
            new_ssm_states.append(new_ssm)

        curr_hidden = self.norm(h)

        # Compute logits from SSM output
        logits = lm_head(curr_hidden)  # (B, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)
        token_log_prob = torch.gather(
            log_probs, dim=-1, index=token_id.unsqueeze(-1)
        ).squeeze(-1)  # (B,)

        # Compute new average log probability
        curr_token_count = prev_token_count + 1
        if prev_avg_log_prob is None:
            # First token in chunk
            curr_avg_log_prob = token_log_prob
        else:
            # Update running average: (prev_avg * prev_count + new_log_prob) / curr_count
            curr_avg_log_prob = (prev_avg_log_prob * prev_token_count + token_log_prob) / curr_token_count

        # Check if likelihood decreased (surprisal increased)
        if prev_avg_log_prob is None:
            # First token - no boundary
            is_boundary = False
        else:
            # Boundary if average likelihood decreased
            likelihood_change = curr_avg_log_prob - prev_avg_log_prob
            is_boundary = (likelihood_change < 0).item() if likelihood_change.numel() == 1 else (likelihood_change[0] < 0).item()

        return (
            is_boundary,
            curr_hidden,
            new_conv_states,
            new_ssm_states,
            curr_avg_log_prob,
            curr_token_count,
        )

    def get_initial_state(self, batch_size: int, device: torch.device):
        """Get initial states for incremental boundary detection."""
        conv_states = []
        ssm_states = []
        for layer in self.ssm_layers:
            conv, ssm = layer.get_initial_state(batch_size, device)
            conv_states.append(conv)
            ssm_states.append(ssm)
        return conv_states, ssm_states
