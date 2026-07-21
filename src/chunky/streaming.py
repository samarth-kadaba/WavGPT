"""Streaming LM with learned per-layer KV compression.

Processes the sequence in chunks, carrying a bounded compressed KV memory per
layer. After each chunk the shared compressor refolds memory + chunk back into
budget, with active slots growing toward beta*max_slots. One next-token loss
over the whole pass. Compressed memory keeps the rotation of its source keys.
"""

from __future__ import annotations

from typing import Optional

import torch

from chunky.compressor import CompressorConfig, KVCompressor, active_slot_mask
from chunky.model import ModelConfig, Transformer, _lm_loss


class CompressedTransformer(Transformer):
    def __init__(self, cfg: ModelConfig, comp: CompressorConfig, chunk_size: int = 512, detach_memory: bool = False):
        super().__init__(cfg)
        self.comp_cfg = comp
        self.chunk_size = chunk_size
        self.detach_memory = detach_memory
        self.compressor = KVCompressor(comp)

    def forward(self, input_ids, attn_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        T = input_ids.size(1)
        L = len(self.blocks)
        mem_k = [None] * L
        mem_v = [None] * L
        mem_active = [None] * L  # which memory slots are valid (for attention masking)
        chunks = []

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]
            x = self.embed(input_ids[:, start:end])
            active = active_slot_mask(self.comp_cfg, float(end), x.device)
            active_bool = active > 0.5

            B = x.size(0)
            chunk_valid = x.new_ones(B, x.size(1), dtype=torch.bool)
            for i, block in enumerate(self.blocks):
                x, k, v = block.forward_streaming(x, cos, sin, mem_k[i], mem_v[i], mem_active[i])
                if mem_k[i] is None:
                    pool_k, pool_v, cand_mask = k, v, None
                else:
                    pool_k = torch.cat((mem_k[i], k), dim=2)
                    pool_v = torch.cat((mem_v[i], v), dim=2)
                    cand_mask = torch.cat((mem_active[i].view(1, -1).expand(B, -1), chunk_valid), dim=1)
                nk, nv = self.compressor(pool_k, pool_v, active_mask=active, cand_mask=cand_mask)
                mem_k[i], mem_v[i] = (nk.detach(), nv.detach()) if self.detach_memory else (nk, nv)
                mem_active[i] = active_bool

            chunks.append(self.lm_head(self.norm(x)))

        logits = torch.cat(chunks, dim=1)
        return logits, (_lm_loss(logits, labels) if labels is not None else None)
