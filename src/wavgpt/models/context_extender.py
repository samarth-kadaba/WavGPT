"""Context Extender: Extend pretrained transformer context via learned chunking.

This is the main model that wraps a pretrained transformer and extends its
context window using learned chunk boundaries (via GRPO) and compression.

KEY DESIGN PRINCIPLE:
    Policy learns TWO decisions per token:
        1. boundary_prob: Should we end a chunk here?
        2. keep_prob: Should this token be kept at full fidelity?
    
    This enables SELECTIVE RETRIEVAL: 
        - Most tokens get compressed into chunks
        - Important tokens (entities, numbers, key facts) stay verbatim
    
    CONSTRAINT: num_chunks + num_kept_tokens <= max_context

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Pretrained Transformer                                     │
    │  - Processes: [chunks, kept_tokens]                         │
    │  - Each chunk is a compressed summary of a segment          │
    │  - Kept tokens are verbatim (full fidelity for retrieval)  │
    └─────────────────────────────────────────────────────────────┘
                              ↑
                    [chunk_1, ..., chunk_K, kept_1, ..., kept_M]
                              ↑
    ┌─────────────────────────────────────────────────────────────┐
    │  ChunkCompressor (trainable)                                │
    │  - Compresses segments (tokens between boundaries) → chunks │
    └─────────────────────────────────────────────────────────────┘
                              ↑
    ┌─────────────────────────────────────────────────────────────┐
    │  BoundaryPolicy (trained via GRPO)                          │
    │  - boundary_prob: where to place chunk boundaries           │
    │  - keep_prob: which tokens to keep at full fidelity         │
    └─────────────────────────────────────────────────────────────┘
                              ↑
                    All input tokens
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import ContextExtenderConfig
from wavgpt.models.policy import BoundaryPolicy, BoundaryPolicyWithProjection, PolicySample
from wavgpt.models.compressor import ChunkCompressor, ChunkInjector


@dataclass
class ContextExtenderOutput:
    """Output from ContextExtender forward pass."""
    logits: torch.Tensor                    # (B, L, vocab_size) - context tokens
    loss: Optional[torch.Tensor]            # Scalar LM loss
    chunk_embeddings: torch.Tensor          # (B, K, chunk_dim) - compressed chunks
    kept_embeddings: torch.Tensor           # (B, M, hidden_dim) - kept token embeds
    num_chunks: int                         # Number of chunks
    num_kept_tokens: int                    # Number of kept tokens
    boundaries: torch.Tensor                # (B, T) boundary indicators
    keep_mask: torch.Tensor                 # (B, T) keep indicators
    boundary_probs: torch.Tensor            # (B, T) boundary probabilities
    keep_probs: torch.Tensor                # (B, T) keep probabilities


@dataclass  
class GRPOBatch:
    """A batch of GRPO samples with their rewards."""
    samples: List[PolicySample]              # G policy configurations
    rewards: torch.Tensor                    # (G, B) rewards (negative LM loss)
    outputs: List[ContextExtenderOutput]     # Outputs for each sample
    policy_output: Optional[Any] = None      # PolicyOutput for computing ref policy log probs
    embeddings: Optional[torch.Tensor] = None  # (B, T, D) embeddings for ref policy forward
    kl_penalty: Optional[torch.Tensor] = None  # KL divergence from reference model


class ContextExtender(nn.Module):
    """
    Extends pretrained transformer context via learned chunking.
    
    Training Paradigm (GRPO):
        1. Given sequence of T tokens
        2. Policy outputs boundary probs for ALL positions
        3. Sample G boundary configurations
        4. For each: segments → compress → [chunks, current] → transformer
        5. LM loss on current window tokens
        6. GRPO update based on which boundaries gave lower loss
        7. Compressor updated via gradient descent
    
    Inference:
        1. Use deterministic boundaries (threshold 0.5)
        2. Compress segments into chunks
        3. Run transformer on [chunks, current_window]
        4. Predict next token
    
    Generation (Streaming):
        1. Accumulate tokens until policy emits boundary
        2. Compress segment into chunk
        3. Continue with [chunks, new_segment]
    """
    
    def __init__(
        self,
        config: ContextExtenderConfig,
        pretrained_model: nn.Module,
        pretrained_dim: int,
        vocab_size: int,
    ):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        self.vocab_size = vocab_size
        
        # Store pretrained model
        self.pretrained = pretrained_model
        
        # Freeze pretrained model if configured
        if config.freeze_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        # Store frozen reference model for KL penalty (if training full model)
        self.reference_model = None
        if not config.freeze_pretrained and config.kl_penalty_weight > 0:
            self.reference_model = copy.deepcopy(pretrained_model)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
        
        # Boundary policy (trained via GRPO)
        self.policy = BoundaryPolicyWithProjection(config, pretrained_dim)
        
        # Chunk compressor (trained via gradients)
        self.compressor = ChunkCompressor(config, pretrained_dim)
        
        # Chunk injector (projects chunks to transformer input space)
        self.injector = ChunkInjector(config, pretrained_dim)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        config: Optional[ContextExtenderConfig] = None,
        **kwargs,
    ) -> "ContextExtender":
        """Load from a pretrained HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Load pretrained
        hf_config = AutoConfig.from_pretrained(pretrained_model_name)
        pretrained = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **kwargs,
        )
        
        # Create config if not provided
        if config is None:
            config = ContextExtenderConfig(pretrained_model_name=pretrained_model_name)
        
        # Update config with model dimensions
        config.hidden_size = hf_config.hidden_size
        
        # MEMORY OPTIMIZATION: Enable gradient checkpointing for pretrained model
        # This trades compute for memory - essential for training on long sequences
        if config.gradient_checkpointing and hasattr(pretrained, 'gradient_checkpointing_enable'):
            pretrained.gradient_checkpointing_enable()
        
        return cls(
            config=config,
            pretrained_model=pretrained,
            pretrained_dim=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
        )
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from pretrained model."""
        if hasattr(self.pretrained, 'get_input_embeddings'):
            embed_layer = self.pretrained.get_input_embeddings()
            return embed_layer(input_ids)
        elif hasattr(self.pretrained, 'transformer'):
            return self.pretrained.transformer.wte(input_ids)
        elif hasattr(self.pretrained, 'model'):
            return self.pretrained.model.embed_tokens(input_ids)
        raise ValueError("Could not find embedding layer in pretrained model")
    
    def _find_last_boundary(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Find position of last boundary for each sequence in batch."""
        B, T = boundaries.shape
        device = boundaries.device
        
        # positions where boundary = 1
        # We want the LAST such position, or 0 if no boundaries
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # Mask positions where there's a boundary
        boundary_positions = positions * boundaries
        
        # Get max position (last boundary), default to 0
        last_boundary = boundary_positions.max(dim=-1).values  # (B,)
        
        return last_boundary.long()
    
    def _get_segments(
        self, 
        boundaries: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Get segment ranges from boundaries.
        
        A boundary at position i means: "end a chunk at position i"
        Tokens from prev_boundary to this boundary form a segment to compress.
        
        Returns:
            segments: List of (start, end) for each segment to compress
        """
        B, T = boundaries.shape
        
        if B > 1:
            # Use first sequence's boundaries (consistent in GRPO)
            boundaries = boundaries[0:1].expand(B, -1)
        
        # Find boundary positions
        boundary_pos = boundaries[0].nonzero(as_tuple=True)[0]
        
        if len(boundary_pos) == 0:
            # No boundaries - compress entire sequence into one chunk
            return [(0, T)]
        
        # Build segments (each ending at a boundary)
        segments = []
        prev = 0
        for pos in boundary_pos:
            pos = pos.item() + 1  # Include the boundary token in segment
            if pos > prev:
                segments.append((prev, pos))
            prev = pos
        
        # Final segment (after last boundary) - also compress
        if prev < T:
            segments.append((prev, T))
        
        return segments
    
    def _get_kept_indices(
        self,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get indices of tokens to keep at full fidelity.
        
        Args:
            keep_mask: (B, T) binary mask
            
        Returns:
            kept_indices: (B, M) indices of kept tokens (padded with -1)
        """
        B, T = keep_mask.shape
        device = keep_mask.device
        
        # For now, use the first sequence's mask (batch consistency)
        if B > 1:
            keep_mask = keep_mask[0:1].expand(B, -1)
        
        # Find kept positions
        kept_pos = keep_mask[0].nonzero(as_tuple=True)[0]
        
        return kept_pos
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_deterministic_boundaries: bool = True,
    ) -> ContextExtenderOutput:
        """
        Forward pass with dynamic chunking and selective retention.
        
        OPTIMIZED: Uses batched compression instead of Python loops.
        
        The policy outputs TWO decisions per token:
            - boundary_prob: end a chunk here?
            - keep_prob: keep this token verbatim?
        
        Context = interleaved [chunk/kept tokens in original order]
        Constraint: num_chunks + num_kept <= max_context
        
        Args:
            input_ids: (B, T) all token IDs
            attention_mask: (B, T) attention mask
            labels: (B, T) labels for LM loss
            use_deterministic_boundaries: If True, threshold at 0.5
            
        Returns:
            ContextExtenderOutput with logits
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Get embeddings for all tokens
        embeddings = self.get_embeddings(input_ids)  # (B, T, D)
        
        # Get policy decisions (pass attention_mask to mask out padding)
        policy_output = self.policy.forward(embeddings, attention_mask=attention_mask)
        boundary_probs = policy_output.boundary_probs  # (B, T)
        keep_probs = policy_output.keep_probs  # (B, T)
        
        # Get discrete decisions
        if use_deterministic_boundaries:
            boundaries = (boundary_probs > 0.5).float()
            keep_mask = (keep_probs > 0.5).float()
        else:
            boundaries = torch.bernoulli(boundary_probs)
            keep_mask = torch.bernoulli(keep_probs)
        
        # Apply context limit constraint (vectorized)
        boundaries, keep_mask = self._apply_context_limit_vectorized(
            boundaries, keep_mask, boundary_probs, keep_probs, self.config.max_context
        )
        
        # OPTIMIZED: Use batched compression (no Python loop!)
        # Pass keep_mask to exclude kept tokens from chunks (they're handled separately)
        chunks, chunk_mask = self.compressor(embeddings, boundaries, attention_mask, keep_mask)
        K = int(chunk_mask.sum(dim=-1).max().item())  # Actual number of chunks
        if K == 0:
            K = 1
        chunks = chunks[:, :K, :]  # Trim to actual chunks
        chunk_mask = chunk_mask[:, :K]
        
        # Project chunks to virtual tokens
        virtual_tokens, virtual_mask = self.injector(chunks, chunk_mask)
        
        # Get kept token indices and embeddings
        kept_indices = self._get_kept_indices(keep_mask)
        M = len(kept_indices)
        
        if M > 0:
            kept_embeddings = embeddings[:, kept_indices, :]  # (B, M, D)
        else:
            kept_embeddings = torch.zeros(B, 0, self.pretrained_dim, device=device)
        
        # Build interleaved context: [chunk/kept tokens in original order]
        combined_embeddings, combined_mask, context_type = self._build_interleaved_context(
            virtual_tokens, virtual_mask, kept_embeddings, 
            boundaries, keep_mask, kept_indices
        )
        
        total_context = combined_embeddings.size(1)
        
        # Handle edge case: nothing to process
        if total_context == 0:
            return self._forward_simple(
                embeddings, attention_mask, labels, boundaries, keep_mask,
                boundary_probs, keep_probs
            )
        
        # Forward through pretrained
        outputs = self._forward_pretrained(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            num_virtual_tokens=K,
        )
        
        # Get logits
        logits = outputs.logits  # (B, total_context, vocab_size)
        
        # Compute LM loss on kept tokens only
        loss = None
        if labels is not None and M > 1:
            kept_labels = labels[:, kept_indices]  # (B, M)
            
            # Find which positions in combined sequence are kept tokens
            kept_positions = (context_type == 1).nonzero(as_tuple=True)[0]
            num_kept_in_context = len(kept_positions)
            
            if num_kept_in_context > 1:
                kept_logits = logits[:, kept_positions, :]  # (B, num_kept, vocab_size)
                
                # Ensure consistent sizes (use min of both)
                min_len = min(num_kept_in_context, M)
                kept_logits = kept_logits[:, :min_len, :]
                kept_labels = kept_labels[:, :min_len]
                
                shift_logits = kept_logits[:, :-1, :].contiguous()
                shift_labels = kept_labels[:, 1:].contiguous()
                
                if shift_logits.size(1) > 0:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
        
        return ContextExtenderOutput(
            logits=logits,
            loss=loss,
            chunk_embeddings=chunks,
            kept_embeddings=kept_embeddings,
            num_chunks=K,
            num_kept_tokens=M,
            boundaries=boundaries,
            keep_mask=keep_mask,
            boundary_probs=boundary_probs,
            keep_probs=keep_probs,
        )
    
    def _apply_context_limit_vectorized(
        self,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        boundary_probs: torch.Tensor,
        keep_probs: torch.Tensor,
        max_context: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized context limit enforcement.
        
        Strategy: If over budget, reduce kept tokens first (lowest prob),
        then reduce chunks if still needed.
        
        OPTIMIZED: No Python loops over batch dimension.
        """
        B, T = boundaries.shape
        device = boundaries.device
        
        # Count chunks and kept tokens
        num_chunks = boundaries.sum(dim=-1) + 1  # (B,) +1 for final segment
        num_kept = keep_mask.sum(dim=-1)  # (B,)
        total = num_chunks + num_kept  # (B,)
        
        # Check if any batch element exceeds limit
        over_budget = total > max_context
        if not over_budget.any():
            return boundaries, keep_mask
        
        # For each batch: reduce kept tokens first, then boundaries
        # Use probability-based selection: keep highest prob tokens
        
        # Create priority scores for kept tokens (higher = keep)
        # Mask out non-kept positions with -inf
        keep_priority = keep_probs.clone()
        keep_priority[keep_mask == 0] = -float('inf')
        
        # For boundaries, lower probability = more likely to remove
        boundary_priority = boundary_probs.clone()
        boundary_priority[boundaries == 0] = -float('inf')
        
        # Process each batch element that's over budget
        for b in range(B):
            if not over_budget[b]:
                continue
            
            excess = int(total[b].item() - max_context)
            
            # First: remove lowest-priority kept tokens
            if num_kept[b] > 0 and excess > 0:
                # Get keep priorities for this batch
                kp = keep_priority[b]
                kept_indices = (keep_mask[b] == 1).nonzero(as_tuple=True)[0]
                
                if len(kept_indices) > 0:
                    # Sort by priority (ascending - remove lowest first)
                    priorities = kp[kept_indices]
                    sorted_idx = torch.argsort(priorities)
                    
                    # Remove lowest priority kept tokens
                    num_to_remove = min(excess, len(kept_indices))
                    remove_idx = kept_indices[sorted_idx[:num_to_remove]]
                    keep_mask[b, remove_idx] = 0
                    excess -= num_to_remove
            
            # Then: remove lowest-priority boundaries
            if excess > 0 and num_chunks[b] > 1:
                bp = boundary_priority[b]
                boundary_indices = (boundaries[b] == 1).nonzero(as_tuple=True)[0]
                
                if len(boundary_indices) > 0:
                    priorities = bp[boundary_indices]
                    sorted_idx = torch.argsort(priorities)
                    
                    num_to_remove = min(excess, len(boundary_indices))
                    remove_idx = boundary_indices[sorted_idx[:num_to_remove]]
                    boundaries[b, remove_idx] = 0
        
        return boundaries, keep_mask
    
    def _build_interleaved_context(
        self,
        virtual_tokens: torch.Tensor,  # (B, K, D)
        virtual_mask: torch.Tensor,    # (B, K)
        kept_embeddings: torch.Tensor, # (B, M, D)
        boundaries: torch.Tensor,      # (B, T)
        keep_mask: torch.Tensor,       # (B, T)
        kept_indices: torch.Tensor,    # (M,) original positions of kept tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build interleaved context: chunks and kept tokens in original order.
        
        Returns context like: [chunk1, kept1, kept2, chunk2, kept3, ...]
        where positions reflect the original sequence order.
        
        Args:
            virtual_tokens: Projected chunk embeddings
            virtual_mask: Mask for valid chunks
            kept_embeddings: Embeddings of kept tokens
            boundaries: Boundary indicators
            keep_mask: Keep token indicators
            kept_indices: Original positions of kept tokens
            
        Returns:
            combined: (B, K+M, D) interleaved embeddings
            mask: (B, K+M) attention mask
            context_type: (K+M,) 0=chunk, 1=kept token
        """
        B = virtual_tokens.size(0)
        K = virtual_tokens.size(1)
        M = kept_embeddings.size(1)
        D = virtual_tokens.size(2)
        device = virtual_tokens.device
        
        if K == 0 and M == 0:
            return (
                torch.zeros(B, 0, D, device=device),
                torch.zeros(B, 0, device=device),
                torch.zeros(0, device=device, dtype=torch.long),
            )
        
        # Get boundary positions (chunk end positions)
        # Each boundary at position t means chunk ending at t
        # Use first batch element (consistent across batch in GRPO)
        boundary_pos = boundaries[0].nonzero(as_tuple=True)[0]  # (num_boundaries,)
        
        # Add final position as implicit boundary
        T = boundaries.size(1)
        if len(boundary_pos) == 0 or boundary_pos[-1] != T - 1:
            final_pos = torch.tensor([T - 1], device=device)
            boundary_pos = torch.cat([boundary_pos, final_pos])
        
        # Each chunk's "position" is its end boundary
        chunk_positions = boundary_pos[:K]  # (K,)
        
        # Combine chunk and kept positions with their types
        # type 0 = chunk, type 1 = kept
        if M > 0:
            all_positions = torch.cat([chunk_positions, kept_indices.to(device)])
            all_types = torch.cat([
                torch.zeros(K, device=device, dtype=torch.long),
                torch.ones(M, device=device, dtype=torch.long),
            ])
        else:
            all_positions = chunk_positions
            all_types = torch.zeros(K, device=device, dtype=torch.long)
        
        # Sort by position to get original order
        sorted_idx = torch.argsort(all_positions)
        sorted_types = all_types[sorted_idx]
        
        # Build interleaved embeddings
        total_len = K + M
        combined = torch.zeros(B, total_len, D, device=device, dtype=virtual_tokens.dtype)
        
        # Track where chunks and kept tokens go in sorted order
        chunk_idx = 0
        kept_idx = 0
        for i, idx in enumerate(sorted_idx):
            if sorted_types[i] == 0:  # Chunk
                combined[:, i, :] = virtual_tokens[:, chunk_idx, :]
                chunk_idx += 1
            else:  # Kept token
                combined[:, i, :] = kept_embeddings[:, kept_idx, :]
                kept_idx += 1
        
        # Build mask
        combined_mask = torch.ones(B, total_len, device=device)
        
        return combined, combined_mask, sorted_types
    
    def _forward_simple(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        boundary_probs: torch.Tensor,
        keep_probs: torch.Tensor,
    ) -> ContextExtenderOutput:
        """Simple forward without chunking (fallback case)."""
        B, T, D = embeddings.shape
        device = embeddings.device
        
        # Truncate to max context if needed
        if T > self.config.max_context:
            embeddings = embeddings[:, -self.config.max_context:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.max_context:]
            T = self.config.max_context
        
        outputs = self._forward_pretrained(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            num_virtual_tokens=0,
        )
        logits = outputs.logits
        
        loss = None
        if labels is not None:
            trunc_labels = labels[:, -T:] if labels.size(1) > T else labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = trunc_labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return ContextExtenderOutput(
            logits=logits,
            loss=loss,
            chunk_embeddings=torch.zeros(B, 0, self.config.chunk_dim, device=device),
            kept_embeddings=embeddings,
            num_chunks=0,
            num_kept_tokens=T,
            boundaries=boundaries,
            keep_mask=keep_mask,
            boundary_probs=boundary_probs,
            keep_probs=keep_probs,
        )
    
    def _forward_pretrained(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_virtual_tokens: int = 0,
    ):
        """Forward through pretrained model with proper position handling."""
        B, T, D = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Generate position IDs
        # Virtual tokens get positions 0, 1, ..., K-1
        # Kept tokens get positions K, K+1, ..., K+M-1
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # Clamp to max position embeddings
        max_pos = getattr(self.pretrained.config, 'max_position_embeddings', 1024)
        position_ids = position_ids.clamp(0, max_pos - 1)
        
        return self.pretrained(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    
    def forward_grpo(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> GRPOBatch:
        """
        Forward pass for GRPO training with chunking and selective retention.
        
        MEMORY OPTIMIZED:
        - Uses batched compression (no Python loops over segments)
        - Rewards are detached (no grad needed for advantage computation)
        - Logits are NOT stored (only loss scalar matters for rewards)
        - Only policy log_probs and compressor gradients flow through
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        if num_samples is None:
            num_samples = self.config.grpo_num_samples
        
        # Get embeddings (shared across all samples)
        embeddings = self.get_embeddings(input_ids)
        
        # Sample (boundary, keep) configurations (pass attention_mask to mask padding)
        samples, policy_output = self.policy.sample(
            embeddings,
            num_samples=num_samples,
            temperature=temperature,
            max_context=self.config.max_context,
            attention_mask=attention_mask,
        )
        
        # Process each sample - MEMORY EFFICIENT
        # Only keep gradients for FIRST sample (compressor learning)
        # Other samples: no_grad (only need rewards for GRPO advantages)
        outputs = []
        rewards = []
        compressor_loss = None  # Only one loss with gradient
        
        for sample_idx, sample in enumerate(samples):
            # Only compute gradients for first sample (for compressor)
            # Other samples: no_grad (only need scalar rewards)
            grad_context = torch.enable_grad() if sample_idx == 0 else torch.no_grad()
            
            with grad_context:
                # OPTIMIZED: Use batched compression
                # Pass keep_mask to exclude kept tokens from chunks
                chunks, chunk_mask = self.compressor(
                    embeddings, sample.boundaries, attention_mask, sample.keep_mask
                )
                K = int(chunk_mask.sum(dim=-1).max().item())
                if K == 0:
                    K = 1
                chunks = chunks[:, :K, :]
                chunk_mask = chunk_mask[:, :K]
                
                # Get kept indices
                kept_indices = self._get_kept_indices(sample.keep_mask)
                M = len(kept_indices)
                
                # Ensure we have at least some tokens to predict
                if M == 0:
                    M = min(10, T)
                    kept_indices = torch.arange(T - M, T, device=device)
                
                # Get virtual tokens
                if K > 0:
                    virtual_tokens, virtual_mask = self.injector(chunks, chunk_mask)
                else:
                    virtual_tokens = torch.zeros(B, 0, self.pretrained_dim, device=device)
                    virtual_mask = torch.zeros(B, 0, device=device)
                
                kept_embeddings = embeddings[:, kept_indices, :]
                
                # Build interleaved context
                combined, combined_mask, context_type = self._build_interleaved_context(
                    virtual_tokens, virtual_mask, kept_embeddings,
                    sample.boundaries, sample.keep_mask, kept_indices
                )
                
                # Forward through pretrained
                model_outputs = self._forward_pretrained(
                    inputs_embeds=combined,
                    attention_mask=combined_mask,
                    num_virtual_tokens=K,
                )
                
                # Compute loss on kept tokens only
                loss = None
                if labels is not None and M > 1:
                    kept_labels = labels[:, kept_indices]  # (B, M)
                    
                    # Find kept token positions in combined sequence
                    kept_positions = (context_type == 1).nonzero(as_tuple=True)[0]
                    num_kept_in_context = len(kept_positions)
                    
                    if num_kept_in_context > 1:
                        kept_logits = model_outputs.logits[:, kept_positions, :]  # (B, num_kept, vocab)
                        
                        # Ensure consistent sizes (use min of both)
                        min_len = min(num_kept_in_context, M)
                        kept_logits = kept_logits[:, :min_len, :]
                        kept_labels = kept_labels[:, :min_len]
                        
                        shift_logits = kept_logits[:, :-1, :].contiguous()
                        shift_labels = kept_labels[:, 1:].contiguous()
                        
                        if shift_logits.size(1) > 0:
                            loss_per_token = F.cross_entropy(
                                shift_logits.view(-1, self.vocab_size),
                                shift_labels.view(-1),
                                ignore_index=-100,
                                reduction='none',
                            )
                            loss = loss_per_token.view(B, -1).mean(dim=-1)
                        else:
                            loss = torch.zeros(B, device=device)
                    else:
                        loss = torch.zeros(B, device=device)
                else:
                    loss = torch.zeros(B, device=device)
                
                # For first sample: keep gradient for compressor
                if sample_idx == 0 and loss is not None:
                    compressor_loss = loss.mean()
                
                # Rewards always detached
                if loss is not None:
                    rewards.append(-loss.detach())
                else:
                    rewards.append(torch.zeros(B, device=device))
                
                # Clear intermediate tensors
                del combined, combined_mask, model_outputs
            
            # MEMORY OPTIMIZATION: Don't store full logits!
            # Only store minimal info needed
            output = ContextExtenderOutput(
                logits=torch.empty(0, device=device),  # Placeholder - don't store!
                loss=loss.mean().detach() if loss is not None else None,
                chunk_embeddings=chunks.detach(),  # Detach for memory
                kept_embeddings=torch.empty(0, device=device),  # Placeholder
                num_chunks=K,
                num_kept_tokens=M,
                boundaries=sample.boundaries,
                keep_mask=sample.keep_mask,
                boundary_probs=policy_output.boundary_probs,
                keep_probs=policy_output.keep_probs,
            )
            outputs.append(output)
        
        # Store compressor loss for gradient (only first sample has grad)
        self._last_compressor_loss = compressor_loss
        
        # Stack rewards: (G, B)
        rewards = torch.stack(rewards, dim=0)
        
        # KL penalty computed separately if needed (not in this loop to save memory)
        # When freeze_pretrained=True, KL penalty is not needed
        kl_penalty = None
        
        return GRPOBatch(
            samples=samples,
            rewards=rewards,
            outputs=outputs,
            policy_output=policy_output,  # Current policy output
            embeddings=embeddings.detach(),  # Store for ref policy forward (detached)
            kl_penalty=kl_penalty,
        )
    
    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from reference model."""
        if self.reference_model is None:
            return torch.tensor(0.0, device=input_ids.device)
        
        with torch.no_grad():
            ref_outputs = self.reference_model(input_ids)
            ref_logits = ref_outputs.logits
        
        # Align dimensions
        T_cur = current_logits.size(1)
        T_ref = ref_logits.size(1)
        min_T = min(T_cur, T_ref)
        
        current_logits = current_logits[:, :min_T, :]
        ref_logits = ref_logits[:, :min_T, :]
        
        # Compute KL
        current_probs = F.softmax(current_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1).mean()
        
        return kl
    
    def compute_grpo_loss(
        self,
        grpo_batch: GRPOBatch,
        ref_policy_output: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss from a batch of samples.
        
        Args:
            grpo_batch: GRPOBatch from forward_grpo
            ref_policy_output: PolicyOutput from reference policy forward pass
                              (run ref_policy.forward(embeddings) on same input)
        """
        return self.policy.policy.compute_grpo_loss(
            grpo_batch.samples,
            grpo_batch.rewards,
            ref_policy_output,
        )
    
    def compute_compressor_loss(self, grpo_batch: GRPOBatch) -> torch.Tensor:
        """Compute compressor loss (only first sample has gradient)."""
        if hasattr(self, '_last_compressor_loss') and self._last_compressor_loss is not None:
            return self._last_compressor_loss
        return torch.tensor(0.0, device=grpo_batch.rewards.device)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens with dynamic chunking.
        
        Uses streaming approach:
        - Accumulate tokens until policy suggests boundary
        - Compress completed segment into chunk
        - Continue generating
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids.clone()
        chunks = []
        segment_start = 0
        
        for _ in range(max_new_tokens):
            # Get current context
            current_len = generated.size(1)
            current_segment_len = current_len - segment_start
            num_chunks = len(chunks)
            
            # Check if we need to chunk: num_chunks + current_segment > max_context
            if num_chunks + current_segment_len >= self.config.max_context:
                # Compress current segment into chunk
                segment_embeds = self.get_embeddings(generated[:, segment_start:current_len])
                seg_boundaries = torch.zeros(B, current_len - segment_start, device=device)
                seg_boundaries[:, -1] = 1.0
                chunk, _ = self.compressor(segment_embeds, seg_boundaries)
                chunks.append(chunk[:, 0, :])
                segment_start = current_len
            
            # Build context: [chunks, current_segment]
            current_segment = generated[:, segment_start:]
            current_embeds = self.get_embeddings(current_segment)
            
            if chunks:
                chunk_tensor = torch.stack(chunks, dim=1)
                virtual_tokens, virtual_mask = self.injector(
                    chunk_tensor, 
                    torch.ones(B, len(chunks), device=device)
                )
                combined = torch.cat([virtual_tokens, current_embeds], dim=1)
                K = len(chunks)
            else:
                combined = current_embeds
                K = 0
            
            # Get prediction
            outputs = self._forward_pretrained(
                inputs_embeds=combined,
                num_virtual_tokens=K,
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if hasattr(self.pretrained.config, 'eos_token_id'):
                if (next_token == self.pretrained.config.eos_token_id).all():
                    break
        
        return generated
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable params by component."""
        return {
            "policy": sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
            "compressor": sum(p.numel() for p in self.compressor.parameters() if p.requires_grad),
            "injector": sum(p.numel() for p in self.injector.parameters() if p.requires_grad),
            "pretrained": sum(p.numel() for p in self.pretrained.parameters() if p.requires_grad),
        }
