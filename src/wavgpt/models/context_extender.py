"""Context Extender: Extend pretrained transformer context via learned chunking.

This is the main model that wraps a pretrained transformer and extends its
context window using learned chunk boundaries (via GRPO) and compression.

KEY DESIGN PRINCIPLE:
    Policy learns TWO decisions per token:
        1. boundary_prob: Should we end a chunk here?
        2. keep_prob: Should this token be kept at full fidelity?
    
    UNIFIED ARCHITECTURE: Policy and Compressor share the SAME SSM backbone,
    enabling end-to-end credit assignment via difficulty scores.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Pretrained Transformer                                     │
    │  - Processes: [chunks, kept_tokens]                         │
    └─────────────────────────────────────────────────────────────┘
                              ↑
                    [chunk_1, ..., chunk_K, kept_1, ..., kept_M]
                              ↑
    ┌─────────────────────────────────────────────────────────────┐
    │  PolicyCompressor (unified, trained via GRPO + gradients)   │
    │  - Shared SSM backbone                                      │
    │  - Policy heads: boundary + keep decisions                  │
    │  - Compression head: chunk embeddings + difficulty scores   │
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
from wavgpt.models.policy import PolicyCompressorWithProjection, PolicySample, PolicyOutput
from wavgpt.models.compressor import ChunkInjector


@dataclass
class ContextExtenderOutput:
    """Output from ContextExtender forward pass."""
    logits: torch.Tensor                    # (B, L, vocab_size)
    loss: Optional[torch.Tensor]            # Scalar LM loss
    chunk_embeddings: torch.Tensor          # (B, K, chunk_dim)
    kept_embeddings: torch.Tensor           # (B, M, hidden_dim)
    num_chunks: int                         # Number of chunks
    num_kept_tokens: int                    # Number of kept tokens
    boundaries: torch.Tensor                # (B, T) boundary indicators
    keep_mask: torch.Tensor                 # (B, T) keep indicators
    boundary_probs: torch.Tensor            # (B, T) boundary probabilities
    keep_probs: torch.Tensor                # (B, T) keep probabilities
    context_type: Optional[torch.Tensor] = None  # (L,) 0=chunk, 1=kept
    difficulty_scores: Optional[torch.Tensor] = None  # (B, K) chunk difficulties


@dataclass  
class GRPOBatch:
    """A batch of GRPO samples with their rewards."""
    samples: List[PolicySample]
    rewards: torch.Tensor                    # (G, B)
    outputs: List[ContextExtenderOutput]
    policy_output: Optional[PolicyOutput] = None
    hidden_states: Optional[torch.Tensor] = None  # (B, T, D) for ref policy
    chunk_difficulties: Optional[List[torch.Tensor]] = None  # (G,) list of (B, K) tensors


class ContextExtender(nn.Module):
    """
    Extends pretrained transformer context via learned chunking.
    
    Uses UNIFIED PolicyCompressor for both boundary decisions and compression,
    enabling end-to-end credit assignment via shared representations.
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
        
        # Reference model for KL penalty
        self.reference_model = None
        if not config.freeze_pretrained and config.kl_penalty_weight > 0:
            self.reference_model = copy.deepcopy(pretrained_model)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
        
        # UNIFIED policy-compressor (shared SSM backbone!)
        self.policy = PolicyCompressorWithProjection(config, pretrained_dim)
        
        # Chunk injector (projects chunks to transformer space)
        self.injector = ChunkInjector(config, pretrained_dim)
        
        self._compiled = False
    
    def compile_modules(self):
        """Apply torch.compile to key modules."""
        if self._compiled:
            return
        
        try:
            self.policy.core = torch.compile(self.policy.core, mode="default")
            self._compiled = True
            print("Successfully compiled policy-compressor with torch.compile")
        except Exception as e:
            print(f"torch.compile failed (will use eager mode): {e}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        config: Optional[ContextExtenderConfig] = None,
        **kwargs,
    ) -> "ContextExtender":
        """Load from a pretrained HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoConfig
        
        hf_config = AutoConfig.from_pretrained(pretrained_model_name)
        pretrained = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **kwargs,
        )
        
        if config is None:
            config = ContextExtenderConfig(pretrained_model_name=pretrained_model_name)
        
        config.hidden_size = hf_config.hidden_size
        
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
    
    def get_hidden_states(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get LAST HIDDEN STATES from pretrained model."""
        B, T = input_ids.shape
        device = input_ids.device
        
        max_pos = getattr(self.pretrained.config, 'max_position_embeddings', 1024)
        
        if T <= max_pos:
            with torch.set_grad_enabled(not self.config.freeze_pretrained):
                outputs = self.pretrained(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    return outputs.hidden_states[-1]
                elif hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state
                else:
                    raise ValueError("Model doesn't return hidden_states")
        
        # Sliding window for long sequences
        overlap = max_pos // 4
        stride = max_pos - overlap
        
        hidden_states = torch.zeros(B, T, self.pretrained_dim, device=device, dtype=torch.float32)
        hidden_counts = torch.zeros(B, T, 1, device=device, dtype=torch.float32)
        
        for start in range(0, T, stride):
            end = min(start + max_pos, T)
            
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            
            with torch.set_grad_enabled(not self.config.freeze_pretrained):
                outputs = self.pretrained(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    chunk_hidden = outputs.hidden_states[-1]
                else:
                    chunk_hidden = outputs.last_hidden_state
            
            hidden_states[:, start:end, :] += chunk_hidden
            hidden_counts[:, start:end, :] += 1
            
            if end >= T:
                break
        
        hidden_states = hidden_states / hidden_counts.clamp(min=1)
        
        return hidden_states
    
    def _get_kept_indices(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """Get indices of tokens to keep."""
        B, T = keep_mask.shape
        
        if B > 1:
            keep_mask = keep_mask[0:1].expand(B, -1)
        
        kept_pos = keep_mask[0].nonzero(as_tuple=True)[0]
        return kept_pos
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_deterministic_boundaries: bool = True,
    ) -> ContextExtenderOutput:
        """Forward pass with dynamic chunking."""
        B, T = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        embeddings = self.get_embeddings(input_ids)
        
        # Get hidden states for policy (rich contextual features)
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        
        # Get policy decisions using sample() - handles importance/threshold selection
        samples, policy_output = self.policy.sample(
            hidden_states, 
            num_samples=1, 
            deterministic=use_deterministic_boundaries,
            attention_mask=attention_mask,
        )
        sample = samples[0]
        boundaries = sample.boundaries
        keep_mask = sample.keep_mask
        
        # Compress using SHARED hidden states from policy
        chunks, chunk_mask, chunk_difficulty = self.policy.compress(
            policy_output.hidden_states, boundaries, attention_mask, keep_mask
        )
        K = chunks.size(1)
        
        # Project chunks to virtual tokens
        virtual_tokens, virtual_mask = self.injector(chunks, chunk_mask)
        
        # Get kept tokens
        kept_indices = self._get_kept_indices(keep_mask)
        M = len(kept_indices)
        
        if M > 0:
            kept_embeddings = embeddings[:, kept_indices, :]
        else:
            kept_embeddings = torch.zeros(B, 0, self.pretrained_dim, device=device)
        
        # Build interleaved context
        combined_embeddings, combined_mask, context_type = self._build_interleaved_context_vectorized(
            virtual_tokens, virtual_mask, kept_embeddings, 
            boundaries, keep_mask, kept_indices
        )
        
        total_context = combined_embeddings.size(1)
        
        if total_context == 0:
            return self._forward_simple(
                embeddings, attention_mask, labels, boundaries, keep_mask,
                policy_output.boundary_probs, policy_output.keep_probs, chunk_difficulty
            )
        
        outputs = self._forward_pretrained(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            num_virtual_tokens=K,
        )
        
        logits = outputs.logits
        
        # Compute LM loss: predict LITERAL next token (not next kept token)
        # For each kept token at original position i, predict token at position i+1
        loss = None
        if labels is not None and M > 0:
            kept_positions = (context_type == 1).nonzero(as_tuple=True)[1]
            num_kept_in_context = len(kept_positions)
            
            if num_kept_in_context > 0:
                # For each kept index i, we predict token at i+1
                kept_indices_tensor = torch.tensor(kept_indices, device=device)
                next_indices = kept_indices_tensor + 1
                valid_mask = next_indices < T
                
                if valid_mask.sum() > 0:
                    # Get logits at kept positions (predicting next token)
                    kept_logits = logits[:, kept_positions, :]
                    
                    # Only use positions where next token exists
                    valid_logits = kept_logits[:, valid_mask, :]
                    valid_next_indices = next_indices[valid_mask]
                    
                    # Labels are the NEXT tokens in original sequence
                    next_labels = labels[:, valid_next_indices]
                    
                    loss = F.cross_entropy(
                        valid_logits.reshape(-1, self.vocab_size),
                        next_labels.reshape(-1),
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
            boundary_probs=policy_output.boundary_probs,
            keep_probs=policy_output.keep_probs,
            context_type=context_type,
            difficulty_scores=chunk_difficulty,
        )
    
    
    def _build_interleaved_context_vectorized(
        self,
        virtual_tokens: torch.Tensor,
        virtual_mask: torch.Tensor,
        kept_embeddings: torch.Tensor,
        boundaries: torch.Tensor,
        keep_mask: torch.Tensor,
        kept_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build interleaved context using vectorized operations."""
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
        
        boundary_pos = boundaries[0].nonzero(as_tuple=True)[0]
        T = boundaries.size(1)
        
        if len(boundary_pos) == 0 or (len(boundary_pos) > 0 and boundary_pos[-1] != T - 1):
            final_pos = torch.tensor([T - 1], device=device)
            boundary_pos = torch.cat([boundary_pos, final_pos])
        
        if len(boundary_pos) < K:
            padding = torch.full((K - len(boundary_pos),), T - 1, device=device, dtype=boundary_pos.dtype)
            boundary_pos = torch.cat([boundary_pos, padding])
        chunk_positions = boundary_pos[:K]
        
        if M > 0:
            all_positions = torch.cat([chunk_positions, kept_indices.to(device)])
            all_types = torch.cat([
                torch.zeros(K, device=device, dtype=torch.long),
                torch.ones(M, device=device, dtype=torch.long),
            ])
        else:
            all_positions = chunk_positions
            all_types = torch.zeros(K, device=device, dtype=torch.long)
        
        sorted_idx = torch.argsort(all_positions)
        sorted_types = all_types[sorted_idx]
        
        total_len = K + M
        combined = torch.zeros(B, total_len, D, device=device, dtype=virtual_tokens.dtype)
        
        is_chunk = (sorted_types == 0)
        is_kept = (sorted_types == 1)
        
        chunk_indices = torch.cumsum(is_chunk.int(), dim=0) - 1
        kept_indices_mapped = torch.cumsum(is_kept.int(), dim=0) - 1
        
        output_positions = torch.arange(total_len, device=device)
        
        chunk_output_pos = output_positions[is_chunk]
        chunk_src_idx = chunk_indices[is_chunk]
        
        if len(chunk_output_pos) > 0:
            combined[:, chunk_output_pos, :] = virtual_tokens[:, chunk_src_idx, :]
        
        kept_output_pos = output_positions[is_kept]
        kept_src_idx = kept_indices_mapped[is_kept]
        
        if len(kept_output_pos) > 0:
            combined[:, kept_output_pos, :] = kept_embeddings[:, kept_src_idx, :]
        
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
        difficulty_scores: Optional[torch.Tensor] = None,
    ) -> ContextExtenderOutput:
        """Simple forward without chunking (fallback)."""
        B, T, D = embeddings.shape
        device = embeddings.device
        
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
            context_type=torch.ones(T, dtype=torch.long, device=device),  # All kept
            difficulty_scores=difficulty_scores,
        )
    
    def _forward_pretrained(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_virtual_tokens: int = 0,
        output_hidden_states: bool = False,
    ):
        """Forward through pretrained model."""
        B, T, D = inputs_embeds.shape
        device = inputs_embeds.device
        
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        max_pos = getattr(self.pretrained.config, 'max_position_embeddings', 1024)
        position_ids = position_ids.clamp(0, max_pos - 1)
        
        return self.pretrained(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
    
    def forward_grpo(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> GRPOBatch:
        """Forward pass for GRPO training with unified policy-compressor."""
        B, T = input_ids.shape
        device = input_ids.device
        
        if num_samples is None:
            num_samples = self.config.grpo_num_samples
        G = num_samples
        
        # Get embeddings and hidden states
        embeddings = self.get_embeddings(input_ids)
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        
        # Sample configurations from UNIFIED policy
        samples, policy_output = self.policy.sample(
            hidden_states,
            num_samples=G,
            temperature=temperature,
            attention_mask=attention_mask,
        )
        
        # Build contexts for all samples
        all_combined = []
        all_masks = []
        all_context_types = []
        all_K = []
        all_M = []
        all_kept_indices = []
        all_chunk_difficulties = []
        
        for sample_idx, sample in enumerate(samples):
            # Compress using SHARED hidden states
            with torch.no_grad() if sample_idx > 0 else torch.enable_grad():
                chunks, chunk_mask, chunk_difficulty = self.policy.compress(
                    policy_output.hidden_states, sample.boundaries, 
                    attention_mask, sample.keep_mask
                )
            
            K = chunks.size(1)
            all_chunk_difficulties.append(chunk_difficulty)
            
            kept_indices = self._get_kept_indices(sample.keep_mask)
            M = len(kept_indices)
            
            if M == 0:
                M = min(10, T)
                kept_indices = torch.arange(T - M, T, device=device)
            
            if K > 0:
                virtual_tokens, virtual_mask = self.injector(chunks, chunk_mask)
            else:
                virtual_tokens = torch.zeros(B, 0, self.pretrained_dim, device=device)
                virtual_mask = torch.zeros(B, 0, device=device)
            
            kept_embeddings = embeddings[:, kept_indices, :]
            
            combined, combined_mask, context_type = self._build_interleaved_context_vectorized(
                virtual_tokens, virtual_mask, kept_embeddings,
                sample.boundaries, sample.keep_mask, kept_indices
            )
            
            all_combined.append(combined)
            all_masks.append(combined_mask)
            all_context_types.append(context_type)
            all_K.append(K)
            all_M.append(M)
            all_kept_indices.append(kept_indices)
        
        # Pad and batch all samples
        max_len = max(c.size(1) for c in all_combined)
        
        batched_embeds = torch.zeros(G, B, max_len, self.pretrained_dim, device=device, dtype=embeddings.dtype)
        batched_masks = torch.zeros(G, B, max_len, device=device)
        
        for g in range(G):
            L_g = all_combined[g].size(1)
            batched_embeds[g, :, :L_g, :] = all_combined[g]
            batched_masks[g, :, :L_g] = all_masks[g]
        
        batched_embeds = batched_embeds.view(G * B, max_len, self.pretrained_dim)
        batched_masks = batched_masks.view(G * B, max_len)
        
        # ONE forward pass for all samples
        with torch.set_grad_enabled(True):
            outputs = self._forward_pretrained(
                inputs_embeds=batched_embeds,
                attention_mask=batched_masks,
                num_virtual_tokens=0,
            )
        
        all_logits = outputs.logits.view(G, B, max_len, self.vocab_size)
        
        # Compute losses
        rewards = []
        outputs_list = []
        compressor_loss = None
        
        for g in range(G):
            L_g = all_combined[g].size(1)
            sample = samples[g]
            M = all_M[g]
            K = all_K[g]
            kept_indices = all_kept_indices[g]
            context_type = all_context_types[g]
            
            logits_g = all_logits[g, :, :L_g, :]
            
            # Compute loss: predict LITERAL next token (not next kept token)
            # For each kept token at original position i, predict token at position i+1
            loss = None
            if labels is not None and M > 0:
                kept_positions = (context_type == 1).nonzero(as_tuple=True)[0]
                num_kept_in_context = len(kept_positions)
                
                if num_kept_in_context > 0:
                    kept_indices_tensor = torch.tensor(kept_indices, device=device)
                    next_indices = kept_indices_tensor + 1
                    valid_mask = next_indices < T
                    
                    if valid_mask.sum() > 0:
                        kept_logits = logits_g[:, kept_positions, :]
                        valid_logits = kept_logits[:, valid_mask, :]
                        valid_next_indices = next_indices[valid_mask]
                        next_labels = labels[:, valid_next_indices]
                        
                        loss_per_token = F.cross_entropy(
                            valid_logits.reshape(-1, self.vocab_size),
                            next_labels.reshape(-1),
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
            
            if g == 0 and loss is not None:
                compressor_loss = loss.mean()
            
            rewards.append(-loss.detach())
            
            output = ContextExtenderOutput(
                logits=torch.empty(0, device=device),
                loss=loss.mean().detach() if loss is not None else None,
                chunk_embeddings=torch.empty(0, device=device),
                kept_embeddings=torch.empty(0, device=device),
                num_chunks=K,
                num_kept_tokens=M,
                boundaries=sample.boundaries,
                keep_mask=sample.keep_mask,
                boundary_probs=policy_output.boundary_probs,
                keep_probs=policy_output.keep_probs,
                difficulty_scores=all_chunk_difficulties[g],
            )
            outputs_list.append(output)
        
        self._last_compressor_loss = compressor_loss
        
        rewards = torch.stack(rewards, dim=0)
        
        return GRPOBatch(
            samples=samples,
            rewards=rewards,
            outputs=outputs_list,
            policy_output=policy_output,
            hidden_states=hidden_states.detach(),
            chunk_difficulties=all_chunk_difficulties,
        )
    
    def compute_grpo_loss(
        self,
        grpo_batch: GRPOBatch,
        ref_policy_output: Optional[PolicyOutput] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss with difficulty-based credit assignment."""
        return self.policy.core.compute_grpo_loss(
            grpo_batch.samples,
            grpo_batch.rewards,
            ref_policy_output,
            chunk_difficulties=grpo_batch.chunk_difficulties,
        )
    
    def compute_compressor_loss(self, grpo_batch: GRPOBatch) -> torch.Tensor:
        """Get compressor loss (from first sample)."""
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
        Generate tokens using EXACT same flow as training forward().
        
        For each token generation step:
        1. Run forward() to get logits (uses same policy/compression as training)
        2. Sample next token from logits
        3. Append and repeat
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Use forward() for EXACT training parity
            # This runs the full policy + compression + interleaving
            outputs = self.forward(
                input_ids=generated,
                attention_mask=None,
                labels=None,  # No labels for generation
                use_deterministic_boundaries=True,
            )
            
            # Find the last KEPT position (context_type == 1)
            # Training only computes loss on kept positions, so only they predict next tokens
            context_type = outputs.context_type
            kept_positions = (context_type == 1).nonzero(as_tuple=True)[0]
            
            if len(kept_positions) > 0:
                last_kept_idx = kept_positions[-1]
                next_token_logits = outputs.logits[:, last_kept_idx, :]
            else:
                # Fallback: use last position if no kept tokens
                next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if hasattr(self.pretrained.config, 'eos_token_id'):
                if (next_token == self.pretrained.config.eos_token_id).all():
                    break
        
        return generated
    
    @torch.no_grad()
    def _compress_with_policy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Compress input sequence using the learned policy.
        
        THIS MUST MATCH TRAINING EXACTLY:
        1. Get hidden_states (not just embeddings)
        2. Run policy.sample on hidden_states
        3. Compress using policy.compress
        
        Returns:
            chunk_embeddings: (B, K, D) compressed chunk representations
            chunk_mask: (B, K) valid chunk mask
            current_window_start: position where current window begins
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # MATCH TRAINING: Get rich hidden states, not just embeddings
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        
        # Use the learned policy to decide boundaries (deterministic for generation)
        samples, policy_output = self.policy.sample(
            hidden_states,  # MUST be hidden_states to match training
            num_samples=1, 
            deterministic=True,
            attention_mask=attention_mask,
        )
        sample = samples[0]
        boundaries = sample.boundaries  # (B, T)
        keep_mask = sample.keep_mask    # (B, T)
        
        # Find the last boundary position for each sequence
        # Everything after the last boundary is the current window
        last_boundary_pos = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            boundary_positions = boundaries[b].nonzero(as_tuple=True)[0]
            if len(boundary_positions) > 0:
                last_boundary_pos[b] = boundary_positions[-1].item() + 1
            else:
                # No boundaries - everything is current window
                last_boundary_pos[b] = 0
        
        # Use the minimum last boundary position across batch for simplicity
        current_window_start = last_boundary_pos.min().item()
        
        if current_window_start == 0:
            # No compression needed
            return None, None, 0
        
        # Compress the past (before current_window_start)
        past_ids = input_ids[:, :current_window_start]
        past_mask = attention_mask[:, :current_window_start] if attention_mask is not None else None
        
        # MATCH TRAINING: Get hidden states for past portion
        past_hidden_states = self.get_hidden_states(past_ids, past_mask)
        
        # Re-run policy on just the past portion
        past_samples, past_policy_output = self.policy.sample(
            past_hidden_states,  # MUST be hidden_states
            num_samples=1,
            deterministic=True,
            attention_mask=past_mask,
        )
        past_sample = past_samples[0]
        past_boundaries = past_sample.boundaries
        past_keep_mask = past_sample.keep_mask
        
        # MATCH TRAINING: Compress using policy.compress with hidden_states
        chunk_embeddings, chunk_mask, _ = self.policy.compress(
            past_policy_output.hidden_states,
            past_boundaries,
            attention_mask=past_mask,
            keep_mask=past_keep_mask,
        )
        
        return chunk_embeddings, chunk_mask, current_window_start
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable params by component."""
        return {
            "policy": sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
            "injector": sum(p.numel() for p in self.injector.parameters() if p.requires_grad),
            "pretrained": sum(p.numel() for p in self.pretrained.parameters() if p.requires_grad),
        }
