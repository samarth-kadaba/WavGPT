"""Infinite Context Transformer with Learnable Chunking."""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import InfiniteContextConfig, GenerationState
from wavgpt.models.boundary import BoundaryDetector
from wavgpt.models.compressor import ChunkCompressor
from wavgpt.models.transformer import ChunkTransformer, TokenPredictor


class InfiniteContextTransformer(nn.Module):
    """Infinite Context Transformer with learnable chunking."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Core components
        self.boundary_detector = BoundaryDetector(config)
        self.chunk_compressor = ChunkCompressor(config)
        self.chunk_transformer = ChunkTransformer(config)
        self.token_predictor = TokenPredictor(config)

        # LM head with weight tying
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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
        Forward pass with learnable chunking.
        
        Args:
            input_ids: Token IDs (B, T)
            labels: Target labels for LM loss (B, T)
            
        Returns:
            Dictionary with logits, losses, and diagnostics
        """
        B, T = input_ids.shape

        # 1. Token embeddings
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Boundary detection
        (
            boundary_probs,
            boundary_decisions,
            boundary_ssm_out,
            expected_chunks,
            distill_loss,
            entropy_loss,
            sparsity_loss,
        ) = self.boundary_detector(x)

        # Compute chunk assignments
        chunk_ids = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        # Chunk compression
        chunk_embeddings, chunk_mask, ssm_outputs, n_chunks = self.chunk_compressor(
            x, chunk_ids, boundary_probs
        )

        # Chunk transformer
        contextualized_chunks = self.chunk_transformer(chunk_embeddings, chunk_mask)

        # Token prediction
        token_hidden = self.token_predictor(contextualized_chunks, ssm_outputs, chunk_ids)

        # LM head
        logits = self.lm_head(token_hidden)

        # Compute losses
        loss = None
        lm_loss = None

        if labels is not None:
            # LM loss (shifted)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Total loss
            loss = (
                lm_loss
                + self.config.distillation_weight * distill_loss
                + self.config.entropy_weight * entropy_loss
                + self.config.sparsity_weight * sparsity_loss
            )

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "distillation_loss": distill_loss,
            "entropy_loss": entropy_loss,
            "sparsity_loss": sparsity_loss,
            "boundary_probs": boundary_probs,
            "n_chunks": n_chunks,
            "expected_chunks": expected_chunks.mean() if expected_chunks is not None else n_chunks,
            "chunk_ids": chunk_ids,
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
        """Generate tokens autoregressively."""
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
        """Efficient generation with state caching and amortized boundary predictor."""
        self.eval()
        B = input_ids.shape[0]
        assert B == 1, "Efficient generation currently supports batch_size=1"
        device = input_ids.device
        T_prompt = input_ids.size(1)

        # Process prompt
        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Detect boundaries
        (
            boundary_probs,
            boundary_decisions,
            boundary_ssm_out,
            expected_chunks,
            _,  # distill_loss
            _,  # entropy_loss
            _,  # sparsity_loss
        ) = self.boundary_detector(x)
        chunk_ids = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        # Initialize generation state
        state = GenerationState()

        # Process prompt chunks
        chunk_embeds, chunk_mask, ssm_outputs, n_chunks = self.chunk_compressor(
            x, chunk_ids, boundary_probs
        )

        # Get active chunks
        active_chunks = int(chunk_mask[0].sum().item())
        state.committed_chunk_embeds = [chunk_embeds[0, i] for i in range(active_chunks)]
        state.n_boundaries = active_chunks - 1 if active_chunks > 0 else 0

        # Contextualize committed chunks
        if state.committed_chunk_embeds:
            committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
            mask = torch.ones(1, len(state.committed_chunk_embeds), device=device)
            state.committed_chunk_contextualized = self.chunk_transformer(committed, mask)
        else:
            # Initialize with zeros if no chunks yet
            state.committed_chunk_embeds = [
                torch.zeros(self.config.hidden_size, device=device)
            ]
            committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
            mask = torch.ones(1, 1, device=device)
            state.committed_chunk_contextualized = self.chunk_transformer(committed, mask)

        # Initialize SSM states
        state.chunk_conv_states, state.chunk_ssm_states = (
            self.chunk_compressor.get_initial_state(1, device)
        )
        state.boundary_conv_states, state.boundary_ssm_states = (
            self.boundary_detector.get_initial_state(1, device)
        )
        
        # Find last chunk start position
        last_chunk_id = int(chunk_ids[0, -1].item())
        last_chunk_start = 0
        for t in range(T_prompt):
            if int(chunk_ids[0, t].item()) == last_chunk_id:
                last_chunk_start = t
                break
        
        # Warm up SSM states with last chunk tokens
        state.current_ssm_output = self.token_predictor.local_init.detach().clone()
        state.current_chunk_size = 0
        
        for t in range(last_chunk_start, T_prompt):
            tok_embed = x[0, t:t+1, :]
            
            ssm_out, state.chunk_conv_states, state.chunk_ssm_states = (
                self.chunk_compressor.step(
                    tok_embed, state.chunk_conv_states, state.chunk_ssm_states
                )
            )
            state.current_ssm_output = ssm_out.squeeze(0) if ssm_out.dim() > 1 else ssm_out
            state.current_chunk_size += 1
            
            _, _, state.boundary_conv_states, state.boundary_ssm_states = (
                self.boundary_detector.step(
                    tok_embed,
                    state.boundary_conv_states,
                    state.boundary_ssm_states,
                    state.n_boundaries,
                    t,
                    T_prompt + max_new_tokens,
                )
            )
        
        state.position = T_prompt
        expected_length = T_prompt + max_new_tokens

        # Generate tokens
        for _ in range(max_new_tokens):
            # Get context for prediction
            global_ctx = state.committed_chunk_contextualized[0, -1]
            local_ctx = state.current_ssm_output

            if local_ctx.dim() == 2:
                local_ctx = local_ctx.squeeze(0)
            if global_ctx.dim() == 2:
                global_ctx = global_ctx.squeeze(0)

            # Combine contexts
            combined = torch.cat([global_ctx, local_ctx], dim=-1).unsqueeze(0)
            token_hidden = self.token_predictor.combine(combined)
            token_hidden = self.token_predictor.norm(token_hidden)

            # Sample next token
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

            # Update state with new token
            tok_embed = self.token_embed(next_token.unsqueeze(0)).squeeze(1)

            # Update chunk compressor SSM
            ssm_out, state.chunk_conv_states, state.chunk_ssm_states = (
                self.chunk_compressor.step(
                    tok_embed, state.chunk_conv_states, state.chunk_ssm_states
                )
            )
            state.current_ssm_output = ssm_out.squeeze(0) if ssm_out.dim() > 1 else ssm_out
            state.current_chunk_size += 1
            state.position += 1

            # Check for boundary
            is_boundary, _, state.boundary_conv_states, state.boundary_ssm_states = (
                self.boundary_detector.step(
                    tok_embed,
                    state.boundary_conv_states,
                    state.boundary_ssm_states,
                    state.n_boundaries,
                    state.position,
                    expected_length,
                )
            )

            # Commit chunk if boundary detected
            if is_boundary and len(state.committed_chunk_embeds) < self.config.max_chunks:
                chunk_embed = self.chunk_compressor.chunk_proj(state.current_ssm_output)
                state.committed_chunk_embeds.append(chunk_embed)
                state.n_boundaries += 1

                # Re-contextualize chunks
                committed = torch.stack(state.committed_chunk_embeds).unsqueeze(0)
                mask = torch.ones(1, len(state.committed_chunk_embeds), device=device)
                state.committed_chunk_contextualized = self.chunk_transformer(
                    committed, mask
                )

                # Reset chunk state
                state.current_chunk_size = 0
                state.chunk_conv_states, state.chunk_ssm_states = (
                    self.chunk_compressor.get_initial_state(1, device)
                )
                state.current_ssm_output = self.token_predictor.local_init.detach().clone()

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
