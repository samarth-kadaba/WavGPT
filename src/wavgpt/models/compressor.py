from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.s4 import SSMLayer


class ChunkCompressor(nn.Module):
    """
    Compresses tokens into chunk representations using boundary-aware SSM.

    Injects boundary information into the SSM input, allowing the selective
    mechanism to learn to reset state at chunk boundaries. This is fully
    vectorized and works with torch.compile.
    """

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        # Boundary signal projection - injects boundary info into hidden dimension
        self.boundary_proj = nn.Linear(1, config.hidden_size)

        # SSM layers process boundary-aware input
        self.ssm_layers = nn.ModuleList(
            [SSMLayer(config) for _ in range(config.n_chunk_ssm_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.chunk_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        token_embeds: torch.Tensor,
        chunk_ids: torch.Tensor,
        boundary_decisions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Compress chunks using boundary-aware SSM (fully vectorized).

        Injects boundary information into the SSM input, allowing the selective
        mechanism to learn chunk-aware processing. No for-loops over timesteps -
        works perfectly with torch.compile!

        Args:
            token_embeds: Token embeddings (B, T, D)
            chunk_ids: Chunk indices per token (B, T)
            boundary_decisions: Boundary indicators (B, T)

        Returns:
            chunk_embeddings: (B, n_chunks, D)
            chunk_mask: (B, n_chunks)
            ssm_outputs: (B, T, D) - per-token SSM outputs
            n_chunks: Number of chunks
        """
        B, T, D = token_embeds.shape
        device = token_embeds.device

        # Use fixed max_chunks to avoid .item() graph break
        # Masking handles variable actual chunk counts
        n_chunks = self.config.max_chunks

        # Inject boundary information into the input
        # This tells the SSM "a new chunk starts here" so it can learn to reset
        boundary_signal = boundary_decisions.unsqueeze(-1)  # (B, T, 1)
        boundary_embed = self.boundary_proj(boundary_signal)  # (B, T, D)

        # Combine token embeddings with boundary signal
        # Use gating so boundary signal modulates the input
        h = token_embeds * (1.0 - boundary_signal * 0.5) + boundary_embed * boundary_signal

        # Run SSM layers - fully vectorized, no for-loops!
        for layer in self.ssm_layers:
            h, _ = layer(h, return_all_states=False)

        ssm_outputs = self.norm(h)

        # Aggregate SSM outputs into chunk embeddings
        chunk_embeddings = self._aggregate_to_chunks(
            ssm_outputs, chunk_ids, boundary_decisions, n_chunks
        )

        # Create mask: 1 for chunks that have tokens, 0 for unused
        # chunk_ids range is [0, max_chunk_id], so chunk c is used if any token has chunk_ids == c
        chunk_indices = (
            torch.arange(n_chunks, device=device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, n_chunks)
        chunk_ids_expanded = chunk_ids.unsqueeze(-1)  # (B, T, 1)
        chunk_has_tokens = (chunk_ids_expanded == chunk_indices).any(dim=1).float()  # (B, n_chunks)
        chunk_mask = chunk_has_tokens

        # Compute actual chunk count from mask (for metrics, not used in computation)
        actual_n_chunks = chunk_mask.sum(dim=1).mean()  # Average across batch

        return chunk_embeddings, chunk_mask, ssm_outputs, actual_n_chunks

    def _aggregate_to_chunks(
        self,
        ssm_outputs: torch.Tensor,
        chunk_ids: torch.Tensor,
        boundary_decisions: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        """
        Aggregate SSM outputs into chunk embeddings.

        For each chunk, take the last token's SSM output as the chunk embedding.
        Fully vectorized using soft assignment weights.

        Args:
            ssm_outputs: (B, T, D) SSM outputs
            chunk_ids: (B, T) chunk index for each token
            boundary_decisions: (B, T) boundary indicators
            n_chunks: Number of chunks

        Returns:
            chunk_embeddings: (B, n_chunks, D)
        """
        B, T, D = ssm_outputs.shape
        device = ssm_outputs.device
        dtype = ssm_outputs.dtype

        # Create "last token in chunk" mask
        # A token is "last in chunk" if the NEXT token is a boundary (or it's the final token)
        is_last_in_chunk = torch.zeros(B, T, device=device, dtype=dtype)
        is_last_in_chunk[:, :-1] = boundary_decisions[:, 1:]  # Next token is boundary
        is_last_in_chunk[:, -1] = 1.0  # Last token is always end of chunk

        # Soft assignment: each token contributes to its chunk weighted by is_last_in_chunk
        chunk_indices = torch.arange(n_chunks, device=device, dtype=dtype)
        chunk_ids_expanded = chunk_ids.unsqueeze(-1)  # (B, T, 1)
        chunk_indices_expanded = chunk_indices.view(1, 1, -1)  # (1, 1, n_chunks)

        # Compute soft distance-based weights for gradient flow
        dist = (chunk_ids_expanded - chunk_indices_expanded).abs()
        temp = self.config.soft_assign_temperature
        soft_match = torch.exp(-dist / temp)  # Soft assignment

        # Combine with last-in-chunk mask
        weights = soft_match * is_last_in_chunk.unsqueeze(-1)  # (B, T, n_chunks)

        # Normalize weights per chunk
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Aggregate
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

    def get_initial_state(self, batch_size: int, device: torch.device):
        """Get initial states for a new chunk."""
        conv_states = []
        ssm_states = []
        for layer in self.ssm_layers:
            conv, ssm = layer.get_initial_state(batch_size, device)
            conv_states.append(conv)
            ssm_states.append(ssm)
        return conv_states, ssm_states
