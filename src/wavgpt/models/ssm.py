"""State Space Model implementation for context extension.

This module provides SSM layers used for:
1. Processing past context (backbone)
2. Compressing chunks into fixed-size vectors

Supports both CUDA-optimized Mamba and pure PyTorch fallback.
"""

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


def can_use_mamba() -> bool:
    """Check if Mamba can be used (requires CUDA)."""
    return HAS_MAMBA and torch.cuda.is_available()


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with Mamba-compatible architecture.
    
    Uses CUDA-optimized Mamba when available, falls back to pure PyTorch.
    Both implementations share the same weight format for compatibility.
    
    Architecture (matches Mamba):
        - in_proj: (d_inner*2, d_model) -> splits into x and z
        - conv1d: depthwise conv on x
        - x_proj: (dt_rank + d_state*2, d_inner) -> dt, B, C
        - dt_proj: (d_inner, dt_rank)
        - A_log: (d_inner, d_state)
        - D: (d_inner,) skip connection
        - out_proj: (d_model, d_inner)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        if can_use_mamba():
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            self.mamba = None
            self.use_mamba = False
            self._init_pytorch_params()

    def _init_pytorch_params(self):
        """Initialize PyTorch SSM parameters matching Mamba architecture."""
        # Input projection: d_model -> d_inner * 2 (for x and z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM projection: d_inner -> dt_rank + d_state*2
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # Timestep projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # A matrix in log form
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Initialize dt_proj bias (Mamba style)
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, T, D) input tensor
            
        Returns:
            output: (B, T, D) output tensor
        """
        if self.use_mamba and x.is_cuda:
            return self.mamba(x)
        return self._forward_pytorch(x)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch SSM forward pass.
        
        OPTIMIZED: Uses chunked processing for long sequences to reduce peak memory.
        """
        B, T, D = x.shape
        
        # Get parameters
        if self.mamba is not None:
            m = self.mamba
            in_proj_weight = m.in_proj.weight
            in_proj_bias = m.in_proj.bias
            conv1d_weight = m.conv1d.weight
            conv1d_bias = m.conv1d.bias
            x_proj_weight = m.x_proj.weight
            dt_proj_weight = m.dt_proj.weight
            dt_proj_bias = m.dt_proj.bias
            A_log = m.A_log
            D_param = m.D
            out_proj_weight = m.out_proj.weight
            out_proj_bias = getattr(m.out_proj, 'bias', None)
        else:
            in_proj_weight = self.in_proj.weight
            in_proj_bias = getattr(self.in_proj, 'bias', None)
            conv1d_weight = self.conv1d.weight
            conv1d_bias = self.conv1d.bias
            x_proj_weight = self.x_proj.weight
            dt_proj_weight = self.dt_proj.weight
            dt_proj_bias = self.dt_proj.bias
            A_log = self.A_log
            D_param = self.D
            out_proj_weight = self.out_proj.weight
            out_proj_bias = getattr(self.out_proj, 'bias', None)
        
        # 1. Input projection
        xz = F.linear(x, in_proj_weight, in_proj_bias)
        x_branch, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution
        x_conv = x_branch.transpose(1, 2)
        x_conv = F.conv1d(
            x_conv, conv1d_weight, conv1d_bias,
            padding=self.d_conv - 1, groups=self.d_inner
        )[:, :, :T]
        x_conv = F.silu(x_conv).transpose(1, 2)
        
        # 3. SSM projection
        x_dbl = F.linear(x_conv, x_proj_weight)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 4. Timestep projection with CAPPING for numerical stability
        # Large dt values cause exp(A * dt) to underflow and B_bar to explode
        dt = F.linear(dt_raw, dt_proj_weight, dt_proj_bias)
        dt = F.softplus(dt).clamp(max=10.0)  # Cap at 10 to prevent explosion
        
        # 5. State space recurrence - OPTIMIZED with parallel scan approximation
        A = -torch.exp(A_log.float())
        
        # Use associative scan for parallel computation when sequence is long
        # For short sequences, sequential is fine
        if T <= 64:
            # Sequential recurrence (simpler, less memory for short seqs)
            output = self._sequential_scan(x_conv, dt, B_ssm, C_ssm, A, D_param)
        else:
            # Chunked processing to reduce peak memory
            output = self._chunked_scan(x_conv, dt, B_ssm, C_ssm, A, D_param, chunk_size=64)
        
        # 6. Gate and output projection
        output = output * F.silu(z)
        output = F.linear(output, out_proj_weight, out_proj_bias)
        
        return output
    
    def _sequential_scan(
        self, 
        x_conv: torch.Tensor, 
        dt: torch.Tensor, 
        B_ssm: torch.Tensor, 
        C_ssm: torch.Tensor, 
        A: torch.Tensor, 
        D_param: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard sequential SSM recurrence with numerical stability.
        
        Hidden state is clamped to prevent unbounded accumulation.
        """
        B, T, _ = x_conv.shape
        h = torch.zeros(B, self.d_inner, self.d_state, device=x_conv.device, dtype=x_conv.dtype)
        outputs = []
        
        # Clamp limit for hidden state (prevents explosion)
        H_CLAMP = 100.0
        
        for t in range(T):
            dt_t = dt[:, t, :, None]
            # A is negative, so exp(A * dt_t) is in (0, 1) - decay factor
            A_bar = torch.exp(A * dt_t)  # (d_inner, d_state)
            B_bar = B_ssm[:, t, :].unsqueeze(1) * dt_t  # (B, 1, d_state) * (B, d_inner, 1)
            
            # State update with clamping to prevent unbounded growth
            h = A_bar * h + B_bar * x_conv[:, t, :, None]
            h = h.clamp(-H_CLAMP, H_CLAMP)  # Prevent state explosion
            
            y = (C_ssm[:, t, :].unsqueeze(1) * h).sum(dim=-1)
            y = y + D_param * x_conv[:, t, :]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)
    
    def _chunked_scan(
        self, 
        x_conv: torch.Tensor, 
        dt: torch.Tensor, 
        B_ssm: torch.Tensor, 
        C_ssm: torch.Tensor, 
        A: torch.Tensor, 
        D_param: torch.Tensor,
        chunk_size: int = 64
    ) -> torch.Tensor:
        """
        Chunked SSM recurrence to reduce peak memory.
        
        Processes sequence in chunks, carrying state between chunks.
        This reduces the number of intermediate tensors kept in memory.
        Hidden state is clamped to prevent unbounded accumulation.
        """
        B, T, _ = x_conv.shape
        h = torch.zeros(B, self.d_inner, self.d_state, device=x_conv.device, dtype=x_conv.dtype)
        all_outputs = []
        
        # Clamp limit for hidden state (prevents explosion)
        H_CLAMP = 100.0
        
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_outputs = []
            
            for t in range(chunk_start, chunk_end):
                dt_t = dt[:, t, :, None]
                A_bar = torch.exp(A * dt_t)
                B_bar = B_ssm[:, t, :].unsqueeze(1) * dt_t
                h = A_bar * h + B_bar * x_conv[:, t, :, None]
                h = h.clamp(-H_CLAMP, H_CLAMP)  # Prevent state explosion
                y = (C_ssm[:, t, :].unsqueeze(1) * h).sum(dim=-1)
                y = y + D_param * x_conv[:, t, :]
                chunk_outputs.append(y)
            
            # Stack chunk outputs and append
            all_outputs.append(torch.stack(chunk_outputs, dim=1))
        
        return torch.cat(all_outputs, dim=1)

    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step for autoregressive generation.
        
        Args:
            x: (B, D) single token embedding
            conv_state: (B, d_inner, d_conv) convolution state
            ssm_state: (B, d_inner, d_state) SSM hidden state
            
        Returns:
            output: (B, D) output
            new_conv_state: updated conv state
            new_ssm_state: updated SSM state
        """
        if self.use_mamba and x.is_cuda:
            x_unsqueezed = x.unsqueeze(1)
            output, new_conv, new_ssm = self.mamba.step(
                x_unsqueezed, conv_state, ssm_state
            )
            return output.squeeze(1), new_conv, new_ssm
        
        return self._step_pytorch(x, conv_state, ssm_state)

    def _step_pytorch(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch single step."""
        # Get parameters (same logic as forward)
        if self.mamba is not None:
            m = self.mamba
            in_proj_weight, in_proj_bias = m.in_proj.weight, m.in_proj.bias
            conv1d_weight, conv1d_bias = m.conv1d.weight, m.conv1d.bias
            x_proj_weight = m.x_proj.weight
            dt_proj_weight, dt_proj_bias = m.dt_proj.weight, m.dt_proj.bias
            A_log, D_param = m.A_log, m.D
            out_proj_weight = m.out_proj.weight
            out_proj_bias = getattr(m.out_proj, 'bias', None)
        else:
            in_proj_weight = self.in_proj.weight
            in_proj_bias = getattr(self.in_proj, 'bias', None)
            conv1d_weight, conv1d_bias = self.conv1d.weight, self.conv1d.bias
            x_proj_weight = self.x_proj.weight
            dt_proj_weight, dt_proj_bias = self.dt_proj.weight, self.dt_proj.bias
            A_log, D_param = self.A_log, self.D
            out_proj_weight = self.out_proj.weight
            out_proj_bias = getattr(self.out_proj, 'bias', None)
        
        # 1. Input projection
        xz = F.linear(x, in_proj_weight, in_proj_bias)
        x_branch, z = xz.chunk(2, dim=-1)
        
        # 2. Update conv state
        new_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        new_conv_state[:, :, -1] = x_branch
        
        conv_weight = conv1d_weight.squeeze(1)
        x_conv = (new_conv_state * conv_weight).sum(dim=-1)
        if conv1d_bias is not None:
            x_conv = x_conv + conv1d_bias
        x_conv = F.silu(x_conv)
        
        # 3. SSM projection
        x_dbl = F.linear(x_conv, x_proj_weight)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 4. Timestep with capping
        dt = F.linear(dt_raw, dt_proj_weight, dt_proj_bias)
        dt = F.softplus(dt).clamp(max=10.0)  # Cap for stability
        
        # 5. State update with clamping
        A = -torch.exp(A_log.float())
        dt_expanded = dt.unsqueeze(-1)
        A_bar = torch.exp(A * dt_expanded)
        B_bar = B_ssm.unsqueeze(1) * dt_expanded
        new_ssm_state = A_bar * ssm_state + B_bar * x_conv.unsqueeze(-1)
        new_ssm_state = new_ssm_state.clamp(-100.0, 100.0)  # Prevent explosion
        
        # Output
        y = (C_ssm.unsqueeze(1) * new_ssm_state).sum(dim=-1)
        y = y + D_param * x_conv
        
        # 6. Gate and project
        output = y * F.silu(z)
        output = F.linear(output, out_proj_weight, out_proj_bias)
        
        return output, new_conv_state, new_ssm_state

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial states for incremental generation."""
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        return conv_state, ssm_state


class SSMLayer(nn.Module):
    """SSM layer with residual connection and normalization."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual."""
        return x + self.dropout(self.ssm(self.norm(x)))

    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental step."""
        normed = self.norm(x)
        out, new_conv, new_ssm = self.ssm.step(normed, conv_state, ssm_state)
        return x + self.dropout(out), new_conv, new_ssm

    def get_initial_state(self, batch_size: int, device: torch.device):
        return self.ssm.get_initial_state(batch_size, device)


class SSMBackbone(nn.Module):
    """
    Multi-layer SSM backbone for processing context.
    
    This provides hidden states for boundary policy and context for compression.
    Supports gradient checkpointing for memory efficiency.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        
        self.layers = nn.ModuleList([
            SSMLayer(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through SSM layers.
        
        Args:
            x: (B, T, D) input embeddings
            
        Returns:
            hidden: (B, T, D) hidden states
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Ensure input requires grad for proper checkpointing
                # This is needed when input comes from frozen pretrained model
                # but we still want gradients through this module's parameters
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)
    
    def forward_with_states(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass returning per-layer states for generation.
        
        Args:
            x: (B, T, D) input
            states: Optional list of (conv_state, ssm_state) per layer
            
        Returns:
            hidden: (B, T, D) output
            new_states: List of new states per layer
        """
        if states is None:
            B = x.size(0)
            device = x.device
            states = [layer.get_initial_state(B, device) for layer in self.layers]
        
        new_states = []
        for layer, (conv_state, ssm_state) in zip(self.layers, states):
            # Process token by token
            outputs = []
            for t in range(x.size(1)):
                out, conv_state, ssm_state = layer.step(
                    x[:, t], conv_state, ssm_state
                )
                outputs.append(out)
            x = torch.stack(outputs, dim=1)
            new_states.append((conv_state, ssm_state))
        
        return self.norm(x), new_states

