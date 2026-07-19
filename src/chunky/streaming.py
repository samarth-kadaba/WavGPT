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
        chunks = []

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]
            x = self.embed(input_ids[:, start:end])
            active = active_slot_mask(self.comp_cfg, float(end), x.device)

            for i, block in enumerate(self.blocks):
                x, k, v = block.forward_streaming(x, cos, sin, mem_k[i], mem_v[i])
                pool_k = k if mem_k[i] is None else torch.cat((mem_k[i], k), dim=2)
                pool_v = v if mem_v[i] is None else torch.cat((mem_v[i], v), dim=2)
                nk, nv = self.compressor(pool_k, pool_v, active_mask=active)
                mem_k[i], mem_v[i] = (nk.detach(), nv.detach()) if self.detach_memory else (nk, nv)

            chunks.append(self.lm_head(self.norm(x)))

        logits = torch.cat(chunks, dim=1)
        return logits, (_lm_loss(logits, labels) if labels is not None else None)
