"""Chunk transformer and token predictor."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.attention import MultiHeadAttention


class TransformerLayer(nn.Module):
    """Standard transformer layer with pre-norm."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        mlp_hidden = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, mask, is_causal)
        x = x + self.mlp(self.norm2(x))
        return x


class ChunkTransformer(nn.Module):
    """Transformer over chunk embeddings with causal attention."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.pos_embed = nn.Embedding(config.max_chunks, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.n_chunk_transformer_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply causal transformer attention over chunks.
        
        Args:
            chunk_embeddings: (B, K, D) chunk representations
            chunk_mask: (B, K) which chunks are active
            
        Returns:
            contextualized_chunks: (B, K, D) chunks with global context
        """
        B, K, D = chunk_embeddings.shape

        positions = torch.arange(K, device=chunk_embeddings.device)
        x = chunk_embeddings + self.pos_embed(positions)
        x = self.dropout(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, chunk_mask, True, use_reentrant=False)
            else:
                x = layer(x, mask=chunk_mask, is_causal=True)

        return self.norm(x)


class TokenPredictor(nn.Module):
    """Predicts tokens by combining global (chunk) and local (SSM) context."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        # NOTE: local_init is kept for backward compatibility with old checkpoints
        # but is no longer used - local_context now uses ssm_outputs directly
        self.local_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        contextualized_chunks: torch.Tensor,
        ssm_outputs: torch.Tensor,
        chunk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level hidden states for LM head.
        
        Args:
            contextualized_chunks: (B, K, D) from ChunkTransformer
            ssm_outputs: (B, T, D) per-token SSM outputs
            chunk_ids: (B, T) chunk index for each token
            
        Returns:
            token_hidden: (B, T, D) combined representations for each token
        """
        B, T, D = ssm_outputs.shape
        K = contextualized_chunks.size(1)
        device = ssm_outputs.device
        dtype = ssm_outputs.dtype

        chunk_indices = torch.arange(K, device=device, dtype=dtype)
        chunk_ids_expanded = chunk_ids.unsqueeze(-1)
        chunk_indices_expanded = chunk_indices.view(1, 1, -1)

        causal_mask = (chunk_indices_expanded < chunk_ids_expanded).float()
        dist = (chunk_ids_expanded - chunk_indices_expanded).abs()
        
        # Use log-space softmax for numerical stability
        log_weights = -dist * 1.5  # Reduced steepness for smoother gradients
        log_weights = log_weights.clamp(min=-20)  # Prevent underflow
        # Apply causal mask in log space (-inf for invalid positions)
        log_weights = log_weights + (causal_mask - 1) * 1e4
        weights = torch.softmax(log_weights, dim=-1) * causal_mask  # Extra mask for safety

        global_context = torch.einsum("btk,bkd->btd", weights, contextualized_chunks)

        # Local context: SSM outputs contain info from tokens 0 to t
        # For predicting token t+1, we need all info up to and including token t
        # NOTE: DO NOT shift! ssm_outputs[t] already represents tokens 0:t+1
        # which is exactly what we need for next-token prediction at position t
        local_context = ssm_outputs

        combined = torch.cat([global_context, local_context], dim=-1)
        token_hidden = self.combine(combined)

        return self.norm(token_hidden)
