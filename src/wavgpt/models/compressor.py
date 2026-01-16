"""Chunk Injector for injecting compressed chunks into pretrained transformer.

This module projects compressed chunk embeddings into the pretrained model's
embedding space as "virtual tokens" that can be prepended to the input.

NOTE: Chunk compression is now handled by PolicyCompressor in policy.py.
This file only contains the ChunkInjector for projecting chunks to virtual tokens.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from wavgpt.models.config import ContextExtenderConfig


class ChunkInjector(nn.Module):
    """
    Injects compressed chunks into the pretrained transformer.
    
    Projects chunk embeddings to the pretrained model dimension
    and adds positional information so they can be used as
    virtual tokens prepended to the input.
    """
    
    def __init__(self, config: ContextExtenderConfig, pretrained_dim: int):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        
        # Project chunk embeddings to pretrained model dimension
        self.chunk_to_token = nn.Sequential(
            nn.Linear(config.chunk_dim, pretrained_dim),
            nn.LayerNorm(pretrained_dim),
            nn.GELU(),
            nn.Linear(pretrained_dim, pretrained_dim),
            nn.LayerNorm(pretrained_dim),
        )
        
        # Learnable position embeddings for chunks
        self.chunk_positions = nn.Embedding(config.max_chunks, pretrained_dim)
    
    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert chunks to virtual tokens for the pretrained model.
        
        Args:
            chunk_embeddings: (B, K, chunk_dim) compressed chunks
            chunk_mask: (B, K) mask for valid chunks
            
        Returns:
            virtual_tokens: (B, K, pretrained_dim) tokens to prepend
            virtual_mask: (B, K) attention mask for virtual tokens
        """
        B, K, _ = chunk_embeddings.shape
        device = chunk_embeddings.device
        
        # Project to pretrained dimension
        virtual_tokens = self.chunk_to_token(chunk_embeddings)
        
        # Add positional embeddings
        positions = torch.arange(K, device=device)
        pos_embed = self.chunk_positions(positions)  # (K, D)
        virtual_tokens = virtual_tokens + pos_embed.unsqueeze(0)
        
        return virtual_tokens, chunk_mask
