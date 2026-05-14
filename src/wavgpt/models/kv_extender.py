"""KVExtender: a pretrained causal LM + a trainable KV-cache compressor.

Training objective:
    1. Run the LM on a `prefix` with `use_cache=True` to get hidden states
       and per-layer (K, V).
    2. Compress (K, V) to a fixed-size cache via :class:`KVCompressor`.
    3. Run the LM on a `continuation` with the compressed cache as
       `past_key_values`; the LM cross-entropy on continuation tokens is
       the loss.

Everything is differentiable end-to-end; only the compressor's parameters
are trained (and optionally the LM's if `freeze_pretrained=False`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavgpt.models.config import CompressorConfig
from wavgpt.models.kv_compressor import KVCompressor, coverage_loss, sparsity_loss


@dataclass
class KVExtenderOutput:
    loss: Optional[torch.Tensor]
    lm_loss: Optional[torch.Tensor]
    aux_loss: torch.Tensor
    cont_logits: torch.Tensor                  # (B, T_cont, vocab)
    mixing_weights: torch.Tensor               # (B, K_slots, T_prefix)
    importance: torch.Tensor                   # (B, T_prefix)
    prefix_length: int
    continuation_length: int
    compressed_length: int


def _normalize_past_kv(
    past_key_values,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Accept either a legacy tuple-of-tuples or a HF Cache object and return
    a plain tuple-of-(K, V) tensors per layer."""
    if past_key_values is None:
        return tuple()
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            legacy = past_key_values.to_legacy_cache()
            if legacy is not None:
                return tuple(legacy)
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return tuple(zip(past_key_values.key_cache, past_key_values.value_cache))
    return tuple(past_key_values)


def _to_hf_cache(past_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]):
    """Wrap a legacy tuple-of-(K, V) into the HF Cache object the model expects.

    Modern transformers (≥4.55) refuse legacy tuples in forward. We use
    DynamicCache.from_legacy_cache when available; fall back to the legacy
    tuple if the import fails (older transformers)."""
    try:
        from transformers import DynamicCache
    except ImportError:
        return past_kv
    if not past_kv:
        return DynamicCache()
    return DynamicCache.from_legacy_cache(past_kv)


class KVExtender(nn.Module):
    def __init__(
        self,
        config: CompressorConfig,
        pretrained_model: nn.Module,
        pretrained_dim: int,
        vocab_size: int,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        self.config = config
        self.pretrained_dim = pretrained_dim
        self.vocab_size = vocab_size
        self.freeze_pretrained = freeze_pretrained

        self.pretrained = pretrained_model
        if freeze_pretrained:
            for p in self.pretrained.parameters():
                p.requires_grad = False

        self.compressor = KVCompressor(config, pretrained_dim)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        config: Optional[CompressorConfig] = None,
        freeze_pretrained: bool = True,
        **kwargs,
    ) -> "KVExtender":
        from transformers import AutoModelForCausalLM, AutoConfig

        hf_config = AutoConfig.from_pretrained(pretrained_model_name)
        pretrained = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **kwargs)

        if config is None:
            config = CompressorConfig(pretrained_model_name=pretrained_model_name)
        config.hidden_size = hf_config.hidden_size

        return cls(
            config=config,
            pretrained_model=pretrained,
            pretrained_dim=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
            freeze_pretrained=freeze_pretrained,
        )

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------

    def _run_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """One LM forward over the prefix with KV cache + hidden states.

        Note: we do NOT wrap in ``torch.no_grad`` even when the LM is frozen,
        because we still need gradients to flow back through the LM's
        operations into the compressor inputs. ``requires_grad=False`` on the
        LM parameters already prevents updates to them.
        """
        out = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        past_kv = _normalize_past_kv(out.past_key_values)
        return hidden, past_kv

    def _run_continuation(
        self,
        continuation_ids: torch.Tensor,
        compressed_past_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        original_prefix_length: int,
        original_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Continuation forward conditioned on the compressed KV cache."""
        B, T_cont = continuation_ids.shape
        device = continuation_ids.device
        K_slots = compressed_past_kv[0][0].size(2)

        # Position ids for continuation: pretend the compressed prefix lives at
        # contiguous positions [0, K_slots) and continuation starts at K_slots.
        position_ids = torch.arange(K_slots, K_slots + T_cont, device=device).unsqueeze(0).expand(B, -1)

        attn_mask = torch.ones(B, K_slots + T_cont, device=device, dtype=torch.long)

        hf_cache = _to_hf_cache(compressed_past_kv)
        out = self.pretrained(
            input_ids=continuation_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=hf_cache,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        prefix_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        gumbel_noise: bool = False,
        return_aux: bool = True,
    ) -> KVExtenderOutput:
        """Compute LM loss on `continuation_ids` after compressing `prefix_ids`'s KV cache."""
        B, T_prefix = prefix_ids.shape
        T_cont = continuation_ids.size(1)

        # 1) Run LM on prefix.
        hidden, past_kv = self._run_prefix(prefix_ids, prefix_attention_mask)

        # 2) Compress KV cache to K_slots.
        compressed_past_kv, W, importance = self.compressor(
            hidden, past_kv,
            attention_mask=prefix_attention_mask,
            gumbel_noise=gumbel_noise,
        )

        # 3) Decode continuation with the compressed cache.
        cont_logits = self._run_continuation(
            continuation_ids, compressed_past_kv, T_prefix, prefix_attention_mask,
        )

        # 4) Loss: next-token prediction on continuation.
        #    cont_logits[:, t, :] predicts continuation_ids[:, t+1].
        if T_cont >= 2:
            shift_logits = cont_logits[:, :-1, :].contiguous()
            shift_labels = continuation_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            lm_loss = torch.zeros((), device=cont_logits.device, dtype=cont_logits.dtype)

        aux = torch.zeros((), device=cont_logits.device, dtype=cont_logits.dtype)
        if return_aux:
            cw = self.config.coverage_loss_weight
            sw = self.config.sparsity_loss_weight
            if cw > 0:
                aux = aux + cw * coverage_loss(W)
            if sw > 0:
                aux = aux + sw * sparsity_loss(W)

        return KVExtenderOutput(
            loss=lm_loss + aux,
            lm_loss=lm_loss,
            aux_loss=aux,
            cont_logits=cont_logits,
            mixing_weights=W,
            importance=importance,
            prefix_length=T_prefix,
            continuation_length=T_cont,
            compressed_length=self.config.max_kv_slots,
        )

    # ------------------------------------------------------------------
    # Inference / recursive compression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor, torch.Tensor]:
        """Compress a long prefix's KV cache once. Returns (compressed_kv, W, importance)."""
        hidden, past_kv = self._run_prefix(input_ids, attention_mask)
        compressed, W, importance = self.compressor(
            hidden, past_kv, attention_mask=attention_mask, gumbel_noise=False,
        )
        return compressed, W, importance

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        prefix_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        out = self.forward(
            prefix_ids=prefix_ids,
            continuation_ids=continuation_ids,
            prefix_attention_mask=prefix_attention_mask,
            gumbel_noise=False,
            return_aux=False,
        )
        return {
            "lm_loss": float(out.lm_loss),
            "perplexity": float(torch.exp(out.lm_loss)),
            "prefix_length": out.prefix_length,
            "continuation_length": out.continuation_length,
            "compressed_length": out.compressed_length,
            "compression_ratio": out.prefix_length / max(out.compressed_length, 1),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """Generate from a compressed prefix.

        Compress once, then decode token-by-token from the compressed cache."""
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        K_slots = self.config.max_kv_slots

        compressed_kv, _, _ = self.compress_cache(input_ids)
        cache = _to_hf_cache(compressed_kv)

        generated = input_ids.clone()
        next_input = None
        position = K_slots

        for _ in range(max_new_tokens):
            if next_input is None:
                # Prime the LM with a single dummy step? No: we already have
                # a full cache, so the first generation step must use the
                # LAST original token as input to predict the next one.
                next_input = input_ids[:, -1:].clone()

            position_ids = torch.full((B, 1), position, device=device, dtype=torch.long)
            out = self.pretrained(
                input_ids=next_input,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                topk = torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(logits < topk, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = 0
                rm = rm.scatter(1, sorted_idx, rm)
                logits = logits.masked_fill(rm, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            next_input = next_token
            position += 1

            eos = getattr(self.pretrained.config, "eos_token_id", None)
            if eos is not None and (next_token == eos).all():
                break

        return generated

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> Dict[str, int]:
        return {
            "compressor": sum(p.numel() for p in self.compressor.parameters() if p.requires_grad),
            "pretrained": sum(p.numel() for p in self.pretrained.parameters() if p.requires_grad),
        }
