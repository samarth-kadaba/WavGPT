from __future__ import annotations

import math
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
    """
    Selective State Space Model with shared weights between Mamba and PyTorch implementations.
    
    If Mamba is available (CUDA), we use Mamba's parameters as the source of truth.
    The PyTorch fallback uses the SAME weights, so training with Mamba and inference
    with PyTorch SSM works correctly.
    
    Architecture matches Mamba exactly:
        - in_proj: (d_inner*2, d_model) -> splits into x and z
        - conv1d: depthwise conv on x
        - x_proj: (dt_rank + d_state*2, d_inner) -> outputs dt_raw, B, C
        - dt_proj: (d_inner, dt_rank) -> maps dt_raw to dt
        - A_log: (d_inner, d_state)
        - D: (d_inner,) skip connection
        - out_proj: (d_model, d_inner)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)  # Mamba default

        if _can_use_mamba():
            # Use Mamba - its parameters are the source of truth
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
        else:
            # No Mamba available - create PyTorch SSM parameters matching Mamba architecture
            self.mamba = None
            self.use_mamba = False
            self._init_pytorch_ssm(d_model, d_state, d_conv, expand)

    def _init_pytorch_ssm(self, d_model: int, d_state: int, d_conv: int, expand: int):
        """
        Initialize pure PyTorch SSM with architecture matching Mamba exactly.
        Only called when Mamba is NOT available.
        """
        d_inner = d_model * expand
        dt_rank = math.ceil(d_model / 16)

        # Input projection: d_model -> d_inner * 2 (for x and z)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner, bias=True
        )
        
        # SSM projection: d_inner -> dt_rank + d_state*2 (for dt, B, C)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        
        # Timestep projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # A matrix (log form for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Initialize dt_proj bias (Mamba style)
        dt_min, dt_max = 0.001, 0.1
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

    def _get_params(self):
        """Get parameters from either Mamba or PyTorch SSM."""
        if self.mamba is not None:
            m = self.mamba
            return {
                'in_proj_weight': m.in_proj.weight,
                'in_proj_bias': m.in_proj.bias,
                'conv1d_weight': m.conv1d.weight,
                'conv1d_bias': m.conv1d.bias,
                'x_proj_weight': m.x_proj.weight,
                'dt_proj_weight': m.dt_proj.weight,
                'dt_proj_bias': m.dt_proj.bias,
                'A_log': m.A_log,
                'D': m.D,
                'out_proj_weight': m.out_proj.weight,
                'out_proj_bias': getattr(m.out_proj, 'bias', None),
                'dt_rank': m.dt_rank,
            }
        else:
            return {
                'in_proj_weight': self.in_proj.weight,
                'in_proj_bias': getattr(self.in_proj, 'bias', None),
                'conv1d_weight': self.conv1d.weight,
                'conv1d_bias': self.conv1d.bias,
                'x_proj_weight': self.x_proj.weight,
                'dt_proj_weight': self.dt_proj.weight,
                'dt_proj_bias': self.dt_proj.bias,
                'A_log': self.A_log,
                'D': self.D,
                'out_proj_weight': self.out_proj.weight,
                'out_proj_bias': getattr(self.out_proj, 'bias', None),
                'dt_rank': self.dt_rank,
            }

    def forward(
        self, x: torch.Tensor, return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batch forward pass."""
        if self.use_mamba and x.is_cuda and not return_all_states:
            return self.mamba(x), None
        else:
            return self._forward_pytorch(x, return_all_states)

    def _forward_pytorch(
        self, x: torch.Tensor, return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pure PyTorch SSM forward pass matching Mamba exactly."""
        B, T, D = x.shape
        p = self._get_params()
        
        # 1. Input projection: x -> (x_branch, z)
        xz = F.linear(x, p['in_proj_weight'], p['in_proj_bias'])
        x_branch, z = xz.chunk(2, dim=-1)  # Both (B, T, d_inner)
        
        # 2. Convolution on x_branch
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_conv = F.conv1d(
            x_conv, p['conv1d_weight'], p['conv1d_bias'],
            padding=self.d_conv - 1, groups=self.d_inner
        )[:, :, :T]  # (B, d_inner, T)
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        
        # 3. SSM projection: x_conv -> dt_raw, B_ssm, C_ssm
        x_dbl = F.linear(x_conv, p['x_proj_weight'])  # (B, T, dt_rank + d_state*2)
        dt_rank = p['dt_rank']
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 4. Timestep projection: dt_raw -> dt
        dt = F.linear(dt_raw, p['dt_proj_weight'], p['dt_proj_bias'])  # (B, T, d_inner)
        dt = F.softplus(dt)
        
        # 5. State space recurrence
        A = -torch.exp(p['A_log'].float())  # (d_inner, d_state)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        all_states = [] if return_all_states else None
        
        for t in range(T):
            # Discretize A and B
            dt_t = dt[:, t, :, None]  # (B, d_inner, 1)
            A_bar = torch.exp(A * dt_t)  # (B, d_inner, d_state)
            B_bar = B_ssm[:, t, :].unsqueeze(1) * dt_t  # (B, d_inner, d_state)
            
            # State update
            h = A_bar * h + B_bar * x_conv[:, t, :, None]  # (B, d_inner, d_state)
            
            # Output
            y = (C_ssm[:, t, :].unsqueeze(1) * h).sum(dim=-1)  # (B, d_inner)
            y = y + p['D'] * x_conv[:, t, :]  # Skip connection
            outputs.append(y)
            
            if return_all_states:
                all_states.append(h.clone())
        
        output = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        
        # 6. Gate with z and output projection
        output = output * F.silu(z)
        output = F.linear(output, p['out_proj_weight'], p['out_proj_bias'])
        
        if return_all_states:
            all_states = torch.stack(all_states, dim=1)
        
        return output, all_states

    def step(
        self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental step for generation."""
        if self.use_mamba and x.is_cuda:
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
        """Pure PyTorch incremental step matching Mamba exactly."""
        p = self._get_params()
        
        # 1. Input projection
        xz = F.linear(x, p['in_proj_weight'], p['in_proj_bias'])
        x_branch, z = xz.chunk(2, dim=-1)  # (B, d_inner)
        
        # 2. Update conv state and apply convolution
        new_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        new_conv_state[:, :, -1] = x_branch
        
        # Manual depthwise conv for single step
        conv_weight = p['conv1d_weight'].squeeze(1)  # (d_inner, d_conv)
        x_conv = (new_conv_state * conv_weight).sum(dim=-1)  # (B, d_inner)
        if p['conv1d_bias'] is not None:
            x_conv = x_conv + p['conv1d_bias']
        x_conv = F.silu(x_conv)
        
        # 3. SSM projection
        x_dbl = F.linear(x_conv, p['x_proj_weight'])
        dt_rank = p['dt_rank']
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 4. Timestep projection
        dt = F.linear(dt_raw, p['dt_proj_weight'], p['dt_proj_bias'])
        dt = F.softplus(dt)
        
        # 5. State update
        A = -torch.exp(p['A_log'].float())
        
        dt_expanded = dt.unsqueeze(-1)  # (B, d_inner, 1)
        A_bar = torch.exp(A * dt_expanded)  # (B, d_inner, d_state)
        B_bar = B_ssm.unsqueeze(1) * dt_expanded  # (B, d_inner, d_state)
        
        new_ssm_state = A_bar * ssm_state + B_bar * x_conv.unsqueeze(-1)
        
        # Output
        y = (C_ssm.unsqueeze(1) * new_ssm_state).sum(dim=-1)
        y = y + p['D'] * x_conv
        
        # 6. Gate and output projection
        output = y * F.silu(z)
        output = F.linear(output, p['out_proj_weight'], p['out_proj_bias'])
        
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
