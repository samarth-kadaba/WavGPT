"""
Infinite Context Transformer with SSM-Guided Chunking.

Architecture:
1. Token embeddings (no positional encoding at token level)
2. Boundary SSM: Global O(n) pass to detect semantic chunk boundaries
3. Chunk SSM: Per-chunk compression with boundary gating
4. Chunk Transformer: O(chunks²) causal attention - the ONLY quadratic operation
5. Token Predictor: Combines global (chunk) + local (within-chunk SSM) context
6. Output projection to vocabulary

Key Design Principles:
- Boundary detection uses classifier (2-class: boundary vs no-boundary)
- Differentiable via Gumbel-Softmax with Straight-Through Estimator
- Soft chunk assignments enable gradient flow from LM loss to boundary detector
- Generation uses learned boundaries with incremental state updates
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba  # noqa: F401

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None  # type: ignore

from wavgpt.models.config import InfiniteContextConfig, GenerationState
from wavgpt.models.boundary import BoundaryDetector
from wavgpt.models.compressor import ChunkCompressor
from wavgpt.models.transformer import ChunkTransformer, TokenPredictor


def _can_use_mamba() -> bool:
    """Check if Mamba can be used (requires CUDA)."""
    return HAS_MAMBA and torch.cuda.is_available()


class InfiniteContextTransformer(nn.Module):
    """
    Infinite Context Transformer with SSM-Guided Chunking.

    Architecture:
    1. Token embeddings (no positional encoding)
    2. Boundary detection: Classifier-based with Gumbel-Softmax
    3. Chunk compression: Soft aggregation with boundary gating
    4. Chunk transformer: O(chunks²) causal attention
    5. Token prediction: Global + Local context

    Complexity: O(T) + O(chunks²) where chunks ≤ max_chunks
    """

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.boundary_detector = BoundaryDetector(config)
        self.chunk_compressor = ChunkCompressor(config)
        self.chunk_transformer = ChunkTransformer(config)
        self.token_predictor = TokenPredictor(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying

        self.apply(self._init_weights)

        # No need to initialize classifier bias anymore - using surprisal-based detection

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with fully differentiable boundary learning.

        Args:
            input_ids: Token IDs (B, T)
            labels: Target labels for LM loss (B, T)

        Returns:
            Dictionary with logits, loss, and diagnostics
        """
        B, T = input_ids.shape

        # 1. Token embeddings
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # 2. Boundary detection (surprisal-based, differentiable)
        boundary_logits, boundary_decisions, boundary_hidden = self.boundary_detector(
            x, input_ids, self.lm_head
        )

        # 3. Compute chunk assignments via cumsum
        chunk_ids = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        # 4. Chunk compression with soft assignments
        chunk_embeddings, chunk_mask, ssm_outputs, n_chunks = self.chunk_compressor(
            x, chunk_ids, boundary_decisions
        )

        # 5. Chunk transformer
        contextualized_chunks = self.chunk_transformer(chunk_embeddings, chunk_mask)

        # 6. Token prediction
        token_hidden = self.token_predictor(contextualized_chunks, ssm_outputs, chunk_ids)

        # 7. Output logits
        logits = self.lm_head(token_hidden)

        # Compute loss with compression regularization
        # Boundary detector learns through gradient flow:
        # LM loss → token_hidden → TokenPredictor → soft_assignment_weights → chunk_ids → boundary_decisions
        loss = None
        lm_loss = None
        compression_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Compression loss: directly penalize boundaries
            # boundary_decisions is soft (from Gumbel-Softmax), so gradients flow strongly
            # Higher compression_weight = fewer chunks = more compression
            compression_loss = boundary_decisions.mean()

            loss = lm_loss + self.config.compression_weight * compression_loss

        # Diagnostic info
        chunk_ranges = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "compression_loss": compression_loss,
            "boundary_probs": boundary_decisions,
            "n_chunks": n_chunks,
            "chunk_ranges": chunk_ranges,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """Generate tokens autoregressively (simple version, re-runs full forward)."""
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_efficient(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Efficient generation with state caching and learned boundary detection.

        Uses classifier decision (logit > 0 = boundary) for chunk commits.
        """
        self.eval()
        B = input_ids.shape[0]
        assert B == 1, "Efficient generation currently supports batch_size=1"
        device = input_ids.device

        # Initial forward pass
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Get boundaries and chunk assignments
        boundary_logits, boundary_decisions, boundary_hidden = self.boundary_detector(
            x, input_ids, self.lm_head
        )
        chunk_ids = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        # Initialize state
        state = GenerationState()

        # Process prompt chunks using the regular forward (fully vectorized)
        chunk_embeds, chunk_mask, ssm_outputs, n_chunks = self.chunk_compressor(
            x, chunk_ids, boundary_decisions
        )

        # n_chunks is a tensor (average), get max active chunks from mask for this sample
        active_chunks = int(chunk_mask[0].sum().item())
        state.committed_chunk_embeds = [chunk_embeds[0, i] for i in range(active_chunks)]

        if state.committed_chunk_embeds:
            committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
            mask = torch.ones(1, len(state.committed_chunk_embeds), device=device)
            state.committed_chunk_contextualized = self.chunk_transformer(committed, mask)
        else:
            state.committed_chunk_embeds = [torch.zeros(self.config.hidden_size, device=device)]
            committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
            mask = torch.ones(1, 1, device=device)
            state.committed_chunk_contextualized = self.chunk_transformer(committed, mask)

        # Initialize SSM states
        state.chunk_conv_states, state.chunk_ssm_states = self.chunk_compressor.get_initial_state(
            1, device
        )
        state.current_ssm_output = self.token_predictor.local_init.detach().clone()  # (D,)
        state.current_chunk_size = 0

        state.boundary_conv_states, state.boundary_ssm_states = (
            self.boundary_detector.get_initial_state(1, device)
        )

        # Initialize surprisal tracking state
        state.boundary_prev_hidden = None
        state.boundary_prev_avg_log_prob = None
        state.boundary_token_count = 0

        # Warm up boundary detector with last chunk (if exists)
        if chunk_ids.max() > 0:
            # Find last chunk tokens
            last_chunk_id = chunk_ids[0, -1].item()
            last_chunk_mask = chunk_ids[0] == last_chunk_id
            last_chunk_token_ids = input_ids[0][last_chunk_mask]
            last_chunk_embeds = x[0][last_chunk_mask]

            # Process last chunk tokens to warm up state
            for t in range(last_chunk_embeds.size(0)):
                tok_embed = last_chunk_embeds[t : t + 1]  # (1, D)
                token_id = last_chunk_token_ids[t : t + 1]  # (1,)
                
                if state.boundary_prev_hidden is None:
                    prev_hidden = torch.zeros(1, self.config.hidden_size, device=device)
                else:
                    prev_hidden = state.boundary_prev_hidden

                (
                    _,
                    state.boundary_prev_hidden,
                    state.boundary_conv_states,
                    state.boundary_ssm_states,
                    state.boundary_prev_avg_log_prob,
                    state.boundary_token_count,
                ) = self.boundary_detector.step(
                    tok_embed,
                    prev_hidden,
                    state.boundary_conv_states,
                    state.boundary_ssm_states,
                    token_id,
                    state.boundary_prev_avg_log_prob,
                    state.boundary_token_count,
                    self.lm_head,
                )
        else:
            state.boundary_prev_hidden = torch.zeros(1, self.config.hidden_size, device=device)
            state.boundary_prev_avg_log_prob = None
            state.boundary_token_count = 0

        # Generate tokens
        for _ in range(max_new_tokens):
            global_ctx = state.committed_chunk_contextualized[0, -1]  # (D,)
            local_ctx = state.current_ssm_output  # (D,) or (1, D)

            # Ensure both are 1D tensors of shape (D,)
            if local_ctx.dim() == 2:
                local_ctx = local_ctx.squeeze(0)
            if global_ctx.dim() == 2:
                global_ctx = global_ctx.squeeze(0)

            combined = torch.cat([global_ctx, local_ctx], dim=-1).unsqueeze(0)  # (1, 2D)
            token_hidden = self.token_predictor.combine(combined)
            token_hidden = self.token_predictor.norm(token_hidden)

            logits = self.lm_head(token_hidden).squeeze(0) / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            tok_embed = self.token_embed(next_token.unsqueeze(0)).squeeze(1)

            # Update chunk compressor state
            ssm_out, state.chunk_conv_states, state.chunk_ssm_states = self.chunk_compressor.step(
                tok_embed, state.chunk_conv_states, state.chunk_ssm_states
            )
            # ssm_out is (B, D) where B=1, squeeze to (D,)
            state.current_ssm_output = ssm_out.view(-1) if ssm_out.dim() > 1 else ssm_out
            state.current_chunk_size += 1

            # Check for boundary (surprisal-based decision)
            if state.boundary_prev_hidden is None:
                prev_hidden = torch.zeros(1, self.config.hidden_size, device=device)
            else:
                prev_hidden = state.boundary_prev_hidden

            (
                is_boundary,
                curr_hidden,
                state.boundary_conv_states,
                state.boundary_ssm_states,
                state.boundary_prev_avg_log_prob,
                state.boundary_token_count,
            ) = self.boundary_detector.step(
                tok_embed,
                prev_hidden,
                state.boundary_conv_states,
                state.boundary_ssm_states,
                next_token,
                state.boundary_prev_avg_log_prob,
                state.boundary_token_count,
                self.lm_head,
            )
            state.boundary_prev_hidden = curr_hidden

            # Commit chunk if boundary detected (respecting min_chunk_size)
            should_commit = (
                state.current_chunk_size >= self.config.min_chunk_size
                and is_boundary
                and len(state.committed_chunk_embeds) < self.config.max_chunks
            )

            if should_commit:
                chunk_embed = self.chunk_compressor.chunk_proj(state.current_ssm_output)
                state.committed_chunk_embeds.append(chunk_embed)

                committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
                mask = torch.ones(1, len(state.committed_chunk_embeds), device=device)
                state.committed_chunk_contextualized = self.chunk_transformer(committed, mask)

                state.current_chunk_size = 0
                state.chunk_conv_states, state.chunk_ssm_states = (
                    self.chunk_compressor.get_initial_state(1, device)
                )
                state.current_ssm_output = self.token_predictor.local_init.detach().clone()  # (D,)
                
                # Reset boundary detector state for new chunk
                state.boundary_prev_avg_log_prob = None
                state.boundary_token_count = 0

        return input_ids

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    vocab_size: int = 50257,
    hidden_size: int = 512,
    n_heads: int = 8,
    **kwargs,
) -> InfiniteContextTransformer:
    """Create an Infinite Context Transformer."""
    config = InfiniteContextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        **kwargs,
    )
    return InfiniteContextTransformer(config)
