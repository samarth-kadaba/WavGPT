"""Selective State Space Model with Mamba kernel when available."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


def _can_use_mamba() -> bool:
    return HAS_MAMBA and torch.cuda.is_available()


class SelectiveSSM(nn.Module):
    """Selective SSM. Uses the Mamba CUDA kernel on GPU, else a pure-PyTorch scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        if _can_use_mamba():
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
            return

        self.mamba = None
        self.use_mamba = False

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        with torch.no_grad():
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mamba and x.is_cuda:
            return self.mamba(x)
        return self._forward_pytorch(x)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if self.mamba is not None:
            m = self.mamba
            in_proj_w, in_proj_b = m.in_proj.weight, m.in_proj.bias
            conv_w, conv_b = m.conv1d.weight, m.conv1d.bias
            x_proj_w = m.x_proj.weight
            dt_proj_w, dt_proj_b = m.dt_proj.weight, m.dt_proj.bias
            A_log, D_p = m.A_log, m.D
            out_proj_w = m.out_proj.weight
            out_proj_b = getattr(m.out_proj, "bias", None)
        else:
            in_proj_w, in_proj_b = self.in_proj.weight, getattr(self.in_proj, "bias", None)
            conv_w, conv_b = self.conv1d.weight, self.conv1d.bias
            x_proj_w = self.x_proj.weight
            dt_proj_w, dt_proj_b = self.dt_proj.weight, self.dt_proj.bias
            A_log, D_p = self.A_log, self.D
            out_proj_w = self.out_proj.weight
            out_proj_b = getattr(self.out_proj, "bias", None)

        xz = F.linear(x, in_proj_w, in_proj_b)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = F.conv1d(x_conv, conv_w, conv_b, padding=self.d_conv - 1, groups=self.d_inner)[:, :, :T]
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_dbl = F.linear(x_conv, x_proj_w)
        dt_raw, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(F.linear(dt_raw, dt_proj_w, dt_proj_b)).clamp(max=10.0)
        A = -torch.exp(A_log.float())

        # Sequential selective scan with hidden-state clamping for stability.
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            dt_t = dt[:, t, :, None]
            A_bar = torch.exp(A * dt_t)
            B_bar = B_ssm[:, t, :].unsqueeze(1) * dt_t
            h = (A_bar * h + B_bar * x_conv[:, t, :, None]).clamp(-100.0, 100.0)
            y = (C_ssm[:, t, :].unsqueeze(1) * h).sum(dim=-1) + D_p * x_conv[:, t, :]
            outputs.append(y)

        y_seq = torch.stack(outputs, dim=1) * F.silu(z)
        return F.linear(y_seq, out_proj_w, out_proj_b)


class SSMLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ssm(self.norm(x)))


class SSMBackbone(nn.Module):
    """Stacked SSM layers with optional gradient checkpointing."""

    def __init__(self, d_model: int, n_layers: int = 4, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            [SSMLayer(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)
