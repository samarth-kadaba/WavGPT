"""Chunk Compressor for compressing token sequences into fixed-size vectors.

This module compresses tokens within each chunk (defined by boundaries) into
a single vector that can be used as context for the pretrained transformer.

Unlike the boundary policy (trained via GRPO), the compressor is trained
with standard gradients since compression is a differentiable operation.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import ContextExtenderConfig


class ChunkCompressor(nn.Module):
    """
    Compresses variable-length token chunks into fixed-size vectors.
    
    Architecture:
        1. Project tokens to compression dimension
        2. SSM processes tokens within each chunk
        3. Extract final state as chunk representation
        4. Project to output dimension
    
    The compression is fully differentiable, allowing gradients to flow
    from the language modeling loss back to improve compression quality.
    """
    
    def __init__(self, config: ContextExtenderConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.chunk_dim = config.chunk_dim
        
        # Import here to avoid circular dependency
        from wavgpt.models.ssm import SSMBackbone
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, config.chunk_dim)
        self.input_norm = nn.LayerNorm(config.chunk_dim)
        
        # SSM for processing chunks
        # Uses fewer layers than policy backbone since compression is simpler
        self.ssm = SSMBackbone(
            d_model=config.chunk_dim,
            n_layers=max(1, config.n_ssm_layers // 2),
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        
        # Output projection (to match pretrained model dimension if needed)
        self.output_proj = nn.Linear(config.chunk_dim, config.chunk_dim)
        self.output_norm = nn.LayerNorm(config.chunk_dim)
        
        # Learnable initial state for chunk compression
        self.initial_state = nn.Parameter(torch.zeros(config.chunk_dim))
    
    def forward(
        self,
        tokens: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress tokens into chunk embeddings based on boundaries.
        
        MEMORY-EFFICIENT: Uses O(B * K * D) memory, not O(B * K * T * D).
        
        Each chunk is processed by SSM with FRESH state via sequential
        processing with mini-batching.
        
        Args:
            tokens: (B, T, D) token embeddings from pretrained model
            boundaries: (B, T) binary boundary indicators (1 = boundary after this token)
            attention_mask: (B, T) optional mask for padding
            keep_mask: (B, T) binary mask for tokens to keep (EXCLUDED from chunks)
            
        Returns:
            chunk_embeddings: (B, K, chunk_dim) compressed chunk representations
            chunk_mask: (B, K) mask indicating valid chunks
        """
        B, T, D = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        
        # Create mask for tokens to compress (exclude kept tokens)
        if keep_mask is not None:
            compress_mask = 1.0 - keep_mask
        else:
            compress_mask = torch.ones(B, T, device=device, dtype=dtype)
        
        # Combine with attention mask
        if attention_mask is not None:
            compress_mask = compress_mask * attention_mask.float()
        
        # Count actual number of chunks
        num_boundaries = boundaries.sum(dim=-1)
        actual_max_chunks = int(num_boundaries.max().item()) + 1
        K = min(actual_max_chunks, self.config.max_context)
        K = max(K, 1)
        
        # Project to compression dimension
        x = self.input_proj(tokens)
        x = self.input_norm(x)
        
        # Compute chunk assignments: chunk_ids[t] = which chunk token t belongs to
        shifted_boundaries = F.pad(boundaries[:, :-1], (1, 0), value=0)
        chunk_ids = torch.cumsum(shifted_boundaries, dim=-1).long().clamp(0, K - 1)
        
        # MEMORY-EFFICIENT: Process chunks sequentially with fresh SSM state
        chunk_embeddings, chunk_mask = self._compress_chunks_sequential(
            x, chunk_ids, K, compress_mask
        )
        
        # Output projection
        chunk_embeddings = self.output_proj(chunk_embeddings)
        chunk_embeddings = self.output_norm(chunk_embeddings)
        
        return chunk_embeddings, chunk_mask
    
    def _compress_chunks_sequential(
        self,
        x: torch.Tensor,
        chunk_ids: torch.Tensor,
        K: int,
        compress_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MEMORY-EFFICIENT chunk compression.
        
        Strategy: Process chunks in mini-batches to balance:
        - Chunk independence (fresh SSM state per chunk)
        - Memory efficiency (bounded allocation)
        - GPU efficiency (batched operations)
        
        Memory: O(B * chunk_batch_size * max_chunk_len * D)
        
        Args:
            x: (B, T, D) projected token embeddings
            chunk_ids: (B, T) chunk index for each token (0 to K-1)
            K: number of chunks
            compress_mask: (B, T) mask for tokens to include
            
        Returns:
            chunk_embeddings: (B, K, D) compressed representations
            chunk_mask: (B, K) mask for valid chunks
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # Output tensors
        chunk_embeddings = torch.zeros(B, K, D, device=device, dtype=dtype)
        chunk_counts = torch.zeros(B, K, device=device, dtype=dtype)
        
        # Count tokens per chunk using scatter (memory efficient)
        chunk_counts.scatter_add_(1, chunk_ids, compress_mask)
        
        # Process chunks in mini-batches for memory efficiency
        # Max chunks to process at once (tune based on GPU memory)
        CHUNK_BATCH_SIZE = min(K, 16)  # Process 16 chunks at a time max
        
        for chunk_start in range(0, K, CHUNK_BATCH_SIZE):
            chunk_end = min(chunk_start + CHUNK_BATCH_SIZE, K)
            num_chunks_in_batch = chunk_end - chunk_start
            
            # Find tokens belonging to this chunk range
            in_range = (chunk_ids >= chunk_start) & (chunk_ids < chunk_end)
            valid_in_range = in_range & (compress_mask > 0)
            
            if not valid_in_range.any():
                continue
            
            # Get max tokens per chunk in this batch
            chunk_counts_batch = chunk_counts[:, chunk_start:chunk_end]  # (B, num_chunks_in_batch)
            max_chunk_len = int(chunk_counts_batch.max().item())
            if max_chunk_len == 0:
                continue
            
            # Cap max_chunk_len for memory safety
            max_chunk_len = min(max_chunk_len, 256)
            
            # Build mini-batch tensor: (B, num_chunks_in_batch, max_chunk_len, D)
            chunk_batch = torch.zeros(
                B, num_chunks_in_batch, max_chunk_len, D, 
                device=device, dtype=dtype
            )
            chunk_batch_mask = torch.zeros(
                B, num_chunks_in_batch, max_chunk_len,
                device=device, dtype=dtype
            )
            
            # Compute positions within each chunk efficiently
            # Use cumsum trick: for each token, count how many same-chunk tokens came before
            local_chunk_ids = chunk_ids - chunk_start  # Offset to 0-indexed within batch
            
            # For each batch element, compute positions
            for b in range(B):
                valid_b = valid_in_range[b]  # (T,)
                if not valid_b.any():
                    continue
                    
                positions_b = torch.zeros(T, device=device, dtype=torch.long)
                chunk_pos_counter = torch.zeros(num_chunks_in_batch, device=device, dtype=torch.long)
                
                # Vectorized position computation within batch element
                for t in range(T):
                    if valid_b[t]:
                        k_local = local_chunk_ids[b, t].item()
                        if 0 <= k_local < num_chunks_in_batch:
                            pos = chunk_pos_counter[k_local].item()
                            if pos < max_chunk_len:
                                chunk_batch[b, k_local, pos, :] = x[b, t, :]
                                chunk_batch_mask[b, k_local, pos] = 1.0
                            chunk_pos_counter[k_local] += 1
            
            # Reshape for SSM: (B * num_chunks_in_batch, max_chunk_len, D)
            flat_batch = chunk_batch.view(B * num_chunks_in_batch, max_chunk_len, D)
            
            # Process through SSM - each chunk gets FRESH state
            hidden_flat = self.ssm(flat_batch)  # (B * num_chunks_in_batch, max_chunk_len, D)
            
            # Reshape back: (B, num_chunks_in_batch, max_chunk_len, D)
            hidden = hidden_flat.view(B, num_chunks_in_batch, max_chunk_len, D)
            
            # Mean pool within each chunk (safe division)
            masked_hidden = hidden * chunk_batch_mask.unsqueeze(-1)
            chunk_sums = masked_hidden.sum(dim=2)  # (B, num_chunks_in_batch, D)
            
            # Safe division: use count >= 1 to avoid division by tiny numbers
            # For empty chunks (count=0), output will be 0/1 = 0, which is fine
            counts_safe = chunk_counts_batch.clamp(min=1.0).unsqueeze(-1)
            chunk_emb_batch = chunk_sums / counts_safe
            
            # Store in output
            chunk_embeddings[:, chunk_start:chunk_end, :] = chunk_emb_batch
        
        # Chunk mask: which chunks have at least one token
        chunk_mask = (chunk_counts > 0).float()
        chunk_mask[:, 0] = 1.0  # Ensure chunk 0 is always active
        
        return chunk_embeddings, chunk_mask
    
    def compress_incremental(
        self,
        token: torch.Tensor,
        ssm_state: dict,
        is_boundary: bool,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Incremental compression for generation.
        
        Args:
            token: (B, D) single token embedding
            ssm_state: Current SSM state dict
            is_boundary: Whether to emit a chunk here
            
        Returns:
            chunk_embedding: (B, chunk_dim) if is_boundary else None
            new_state: Updated SSM state
        """
        B = token.size(0)
        device = token.device
        
        # Project token
        x = self.input_proj(token)
        x = self.input_norm(x)
        
        # Get or initialize SSM states
        if 'layer_states' not in ssm_state:
            ssm_state['layer_states'] = [
                layer.get_initial_state(B, device)
                for layer in self.ssm.layers
            ]
        
        # Process through SSM layers
        for i, layer in enumerate(self.ssm.layers):
            conv_state, ssm_st = ssm_state['layer_states'][i]
            x, conv_state, ssm_st = layer.step(x, conv_state, ssm_st)
            ssm_state['layer_states'][i] = (conv_state, ssm_st)
        
        x = self.ssm.norm(x)
        
        # If boundary, emit chunk embedding
        chunk_embedding = None
        if is_boundary:
            chunk_embedding = self.output_proj(x)
            chunk_embedding = self.output_norm(chunk_embedding)
            
            # Reset SSM state for next chunk
            ssm_state['layer_states'] = [
                layer.get_initial_state(B, device)
                for layer in self.ssm.layers
            ]
        
        return chunk_embedding, ssm_state


class ChunkInjector(nn.Module):
    """
    Injects compressed chunks into the pretrained transformer.
    
    Multiple injection strategies:
        1. Soft prompt: Chunks become virtual tokens prepended to input
        2. Cross-attention: Add cross-attention layers to transformer
        3. Memory: Use as external memory for attention
    
    We use the soft prompt approach as it requires no modifications
    to the pretrained transformer architecture.
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

