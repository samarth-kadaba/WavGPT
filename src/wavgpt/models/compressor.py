"""Chunk compressor: aggregates tokens into chunk embeddings."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.s4 import SSMLayer


class ChunkCompressor(nn.Module):
    """Compresses tokens into chunk representations using SSM."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        self.ssm_layers = nn.ModuleList([
            SSMLayer(config) for _ in range(config.n_chunk_ssm_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.chunk_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        token_embeds: torch.Tensor,
        chunk_ids: torch.Tensor,
        boundary_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress tokens into chunk embeddings.
        
        Args:
            token_embeds: Token embeddings (B, T, D)
            chunk_ids: Chunk index per token (B, T) from cumsum of boundaries
            boundary_probs: Soft boundary probabilities (B, T)
            
        Returns:
            chunk_embeddings: (B, max_chunks, D)
            chunk_mask: (B, max_chunks) which chunks are active
            ssm_outputs: (B, T, D) per-token SSM outputs for local context
            n_chunks: Average number of active chunks
        """
        B, T, D = token_embeds.shape
        device = token_embeds.device
        dtype = token_embeds.dtype
        n_chunks = self.config.max_chunks

        # Run SSM layers
        h = token_embeds
        for layer in self.ssm_layers:
            h, _ = layer(h, return_all_states=False)
        ssm_outputs = self.norm(h)

        # Aggregate into chunk embeddings
        chunk_embeddings = self._aggregate_to_chunks(
            ssm_outputs, chunk_ids, boundary_probs, n_chunks
        )

        # Create mask
        chunk_indices = torch.arange(n_chunks, device=device).view(1, 1, -1)
        chunk_ids_expanded = chunk_ids.unsqueeze(-1)
        chunk_has_tokens = (chunk_ids_expanded == chunk_indices).any(dim=1).float()
        chunk_mask = chunk_has_tokens

        actual_n_chunks = chunk_mask.sum(dim=1).mean()

        return chunk_embeddings, chunk_mask, ssm_outputs, actual_n_chunks

    def _aggregate_to_chunks(
        self,
        ssm_outputs: torch.Tensor,
        chunk_ids: torch.Tensor,
        boundary_probs: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        """Aggregate SSM outputs into chunk embeddings."""
        B, T, D = ssm_outputs.shape
        device = ssm_outputs.device
        dtype = ssm_outputs.dtype

        is_last_in_chunk = torch.zeros(B, T, device=device, dtype=dtype)
        is_last_in_chunk[:, :-1] = boundary_probs[:, 1:]
        is_last_in_chunk[:, -1] = 1.0

        chunk_indices = torch.arange(n_chunks, device=device, dtype=dtype)
        chunk_ids_expanded = chunk_ids.unsqueeze(-1)
        chunk_indices_expanded = chunk_indices.view(1, 1, -1)

        # Use log-space softmax for numerical stability instead of raw exp
        # This prevents overflow/underflow with large distances
        dist = (chunk_ids_expanded - chunk_indices_expanded).abs()
        log_soft_match = -dist * 3.0  # Reduced from 10 to prevent sharp gradients
        log_soft_match = log_soft_match.clamp(min=-20)  # Prevent underflow
        soft_match = torch.softmax(log_soft_match, dim=-1)  # Stable softmax

        weights = soft_match * is_last_in_chunk.unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        chunk_embeddings = torch.einsum("btd,btc->bcd", ssm_outputs, weights)
        chunk_embeddings = self.chunk_proj(chunk_embeddings)

        return chunk_embeddings

    def step(
        self,
        token_embed: torch.Tensor,
        conv_states: List[torch.Tensor],
        ssm_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Incremental step for generation."""
        h = token_embed
        new_conv_states = []
        new_ssm_states = []

        for i, layer in enumerate(self.ssm_layers):
            h, new_conv, new_ssm = layer.step(h, conv_states[i], ssm_states[i])
            new_conv_states.append(new_conv)
            new_ssm_states.append(new_ssm)

        h = self.norm(h)
        return h, new_conv_states, new_ssm_states

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get initial SSM states."""
        conv_states = []
        ssm_states = []
        for layer in self.ssm_layers:
            conv, ssm = layer.get_initial_state(batch_size, device)
            conv_states.append(conv)
            ssm_states.append(ssm)
        return conv_states, ssm_states
