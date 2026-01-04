"""Infinite Context Transformer with Learnable Chunking."""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import InfiniteContextConfig
from wavgpt.models.boundary import BoundaryDetector
from wavgpt.models.compressor import ChunkCompressor
from wavgpt.models.transformer import ChunkTransformer, TokenPredictor


class InfiniteContextTransformer(nn.Module):
    """
    Infinite Context Transformer with learnable chunking.
    
    Generation uses full forward pass to match training exactly.
    No separate "efficient" generation - consistency over speed.
    """

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

        # Final layer norm before LM head (GPT-2 style)
        self.final_norm = nn.LayerNorm(config.hidden_size)

        # LM head with weight tying
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        # Initialize weights and apply residual scaling
        self.apply(self._init_weights)
        self._apply_residual_scaling()

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 style weight initialization with residual scaling.
        
        Key tricks:
        - Normal init with std=0.02 for most weights
        - Residual projections scaled by 1/sqrt(2*N) where N is layer count
        - This prevents activation explosion in deep networks
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _apply_residual_scaling(self) -> None:
        """
        Apply GPT-2 residual scaling: scale output projections by 1/sqrt(2*N).
        
        This prevents gradient/activation explosion in deep residual networks.
        Called after apply(_init_weights).
        """
        import math
        
        # Count total residual layers
        n_layers = (
            self.config.n_boundary_layers +
            self.config.n_chunk_ssm_layers +
            self.config.n_chunk_transformer_layers
        )
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        
        # Scale output projections in transformer layers
        for layer in self.chunk_transformer.layers:
            # Scale attention output projection
            if hasattr(layer.attn, 'out_proj'):
                layer.attn.out_proj.weight.data *= scale
            # Scale MLP output projection (last linear in MLP)
            if hasattr(layer.mlp, '__getitem__'):
                # MLP is Sequential, get last Linear
                for sublayer in reversed(list(layer.mlp.children())):
                    if isinstance(sublayer, nn.Linear):
                        sublayer.weight.data *= scale
                        break

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

        # 2. Boundary detection (returns 6 values after removing distill_loss)
        (
            boundary_probs,
            boundary_decisions,
            boundary_ssm_out,
            expected_chunks,
            entropy_loss,
            sparsity_loss,
        ) = self.boundary_detector(x)

        # 3. Compute chunk assignments
        chunk_ids = self.boundary_detector.compute_chunk_assignments(boundary_decisions)

        # 4. Chunk compression
        chunk_embeddings, chunk_mask, ssm_outputs, n_chunks = self.chunk_compressor(
            x, chunk_ids, boundary_probs
        )

        # 5. Chunk transformer
        contextualized_chunks = self.chunk_transformer(chunk_embeddings, chunk_mask)

        # 6. Token prediction - uses same logic as training for consistency
        token_hidden = self.token_predictor(contextualized_chunks, ssm_outputs, chunk_ids)

        # 7. Final normalization and LM head (GPT-2 style)
        token_hidden = self.final_norm(token_hidden)
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

            # Total loss (removed distillation_loss since AmortizedBoundaryPredictor was removed)
            loss = (
                lm_loss
                + self.config.entropy_weight * entropy_loss
                + self.config.sparsity_weight * sparsity_loss
            )

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
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
        """
        Generate tokens autoregressively using full forward pass.
        
        Uses identical code path to training for consistency.
        This ensures boundary detection, chunking, and token prediction
        all match training exactly.
        
        Args:
            input_ids: Prompt token IDs (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None to disable)
            top_p: Top-p nucleus sampling (None to disable)
            
        Returns:
            Generated token IDs (B, T + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Full forward pass - identical to training
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
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

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

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
