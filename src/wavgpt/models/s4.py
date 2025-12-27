from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

from wavgpt.models.config import InfiniteContextConfig


def _can_use_mamba() -> bool:
    """Check if Mamba can be used (requires CUDA)."""
    return HAS_MAMBA and torch.cuda.is_available()


class SelectiveSSM(nn.Module):
    """Selective State Space Model with batch and incremental processing support."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        self._init_pytorch_ssm(d_model, d_state, d_conv, expand)

        if _can_use_mamba():
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
        else:
            self.use_mamba = False

    def _init_pytorch_ssm(self, d_model: int, d_state: int, d_conv: int, expand: int):
        """Initialize pure PyTorch SSM components."""
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batch forward pass."""
        if self.use_mamba and not return_all_states:
            return self.mamba(x), None
        else:
            return self._forward_pytorch(x, return_all_states)

    def _forward_pytorch(
        self, x: torch.Tensor, return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pure PyTorch SSM forward pass."""
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(B_ssm))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        all_states = [] if return_all_states else None

        for t in range(T):
            dt_t = dt[:, t, :, None]
            A_bar = torch.exp(A * dt_t)
            B_bar = B_ssm[:, t, None, :] * dt_t

            h = A_bar * h + B_bar * x_conv[:, t, :, None]
            y = (C_ssm[:, t, None, :] * h).sum(dim=-1) + self.D * x_conv[:, t]
            outputs.append(y)

            if return_all_states:
                all_states.append(h.clone())

        output = torch.stack(outputs, dim=1)
        output = output * F.silu(z)
        output = self.out_proj(output)

        if return_all_states:
            all_states = torch.stack(all_states, dim=1)

        return output, all_states

    def step(
        self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental step for generation."""
        if self.use_mamba:
            x_unsqueezed = x.unsqueeze(1)
            output, new_conv_state, new_ssm_state = self.mamba.step(
                x_unsqueezed, conv_state, ssm_state
            )
            return output.squeeze(1), new_conv_state, new_ssm_state
        else:
            return self._step_pytorch(x, conv_state, ssm_state)

    def _step_pytorch(
        self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch incremental step."""
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        new_conv_state = torch.roll(conv_state, shifts=-1, dims=2)
        new_conv_state[:, :, -1] = x_proj

        # conv1d.weight shape is (d_inner, 1, d_conv) - squeeze the middle dim
        conv_weight = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
        x_conv = (new_conv_state * conv_weight).sum(dim=2)  # (B, d_inner)
        x_conv = F.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(B_ssm))
        A = -torch.exp(self.A_log)

        dt_expanded = dt.unsqueeze(-1)
        A_bar = torch.exp(A * dt_expanded)
        B_bar = B_ssm.unsqueeze(1) * dt_expanded

        new_ssm_state = A_bar * ssm_state + B_bar * x_conv.unsqueeze(-1)
        y = (C_ssm.unsqueeze(1) * new_ssm_state).sum(dim=-1) + self.D * x_conv

        output = y * F.silu(z)
        output = self.out_proj(output)

        return output, new_conv_state, new_ssm_state

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial states for incremental generation."""
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        return conv_state, ssm_state


class SSMLayer(nn.Module):
    """SSM layer with residual connection and normalization."""

    def __init__(self, config: InfiniteContextConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.ssm = SelectiveSSM(
            d_model=config.hidden_size,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed = self.norm(x)
        ssm_out, states = self.ssm(normed, return_all_states)
        return x + self.dropout(ssm_out), states

    def step(
        self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental step."""
        normed = self.norm(x)
        out, new_conv, new_ssm = self.ssm.step(normed, conv_state, ssm_state)
        return x + self.dropout(out), new_conv, new_ssm

    def get_initial_state(self, batch_size: int, device: torch.device):
        return self.ssm.get_initial_state(batch_size, device)
