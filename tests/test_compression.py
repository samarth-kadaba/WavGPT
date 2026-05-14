"""Tests for the CHUNKY KV compressor."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavgpt import (
    CompressorConfig,
    KVCompressor,
    SelectiveSSM,
    SSMBackbone,
)
from wavgpt.models.kv_compressor import coverage_loss, sparsity_loss


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def config():
    return CompressorConfig(
        max_kv_slots=16,
        compress_dim=32,
        n_ssm_layers=2,
        ssm_d_state=8,
    )


class TestSSM:
    def test_forward_shape(self, device):
        ssm = SelectiveSSM(d_model=32, d_state=8).to(device)
        x = torch.randn(2, 16, 32, device=device)
        assert ssm(x).shape == x.shape

    def test_backbone_shape(self, device):
        bb = SSMBackbone(d_model=32, n_layers=2, d_state=8).to(device)
        x = torch.randn(2, 16, 32, device=device)
        assert bb(x).shape == x.shape


class TestKVCompressor:
    def test_mixing_weights_shape_and_simplex(self, config, device):
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        hidden = torch.randn(2, 32, 64, device=device)
        W, importance = comp.compute_mixing_weights(hidden)
        B, K, T = 2, config.max_kv_slots, 32
        assert W.shape == (B, K, T)
        assert importance.shape == (B, T)
        # Each row of W must sum to 1 (it's a softmax over T).
        sums = W.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_apply_mixing_shape(self, config, device):
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        B, n_heads, T, head_dim = 2, 4, 32, 16
        K_slots = config.max_kv_slots
        past_kv = (
            (torch.randn(B, n_heads, T, head_dim, device=device),
             torch.randn(B, n_heads, T, head_dim, device=device)),
            (torch.randn(B, n_heads, T, head_dim, device=device),
             torch.randn(B, n_heads, T, head_dim, device=device)),
        )
        W = torch.softmax(torch.randn(B, K_slots, T, device=device), dim=-1)
        compressed = comp.apply_mixing(W, past_kv)
        assert len(compressed) == 2
        for K, V in compressed:
            assert K.shape == (B, n_heads, K_slots, head_dim)
            assert V.shape == (B, n_heads, K_slots, head_dim)

    def test_attention_mask_zero_padding(self, config, device):
        """Padded positions should receive zero mixing weight."""
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        hidden = torch.randn(1, 32, 64, device=device)
        mask = torch.ones(1, 32, dtype=torch.long, device=device)
        mask[:, 20:] = 0
        W, _ = comp.compute_mixing_weights(hidden, attention_mask=mask)
        assert W[:, :, 20:].abs().max() < 1e-6

    def test_gumbel_noise_affects_W_only_in_training(self, config, device):
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        comp.eval()
        hidden = torch.randn(1, 16, 64, device=device)
        # No gumbel when eval, even if requested.
        W1, _ = comp.compute_mixing_weights(hidden, gumbel_noise=True)
        W2, _ = comp.compute_mixing_weights(hidden, gumbel_noise=True)
        assert torch.allclose(W1, W2, atol=1e-6)

    def test_gradients_flow_to_compressor(self, config, device):
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        hidden = torch.randn(1, 16, 64, device=device, requires_grad=True)
        W, importance = comp.compute_mixing_weights(hidden)
        loss = W.sum() + importance.sum()
        loss.backward()
        # Check critical params have grad.
        assert comp.slot_queries.grad is not None
        assert comp.importance_head[0].weight.grad is not None
        assert not torch.isnan(comp.slot_queries.grad).any()

    def test_gradient_flows_through_K_V_mixing(self, config, device):
        comp = KVCompressor(config, pretrained_dim=64).to(device)
        B, n_heads, T, head_dim = 1, 2, 16, 8
        K_slots = config.max_kv_slots
        K_in = torch.randn(B, n_heads, T, head_dim, device=device, requires_grad=True)
        V_in = torch.randn(B, n_heads, T, head_dim, device=device, requires_grad=True)
        hidden = torch.randn(B, T, 64, device=device)
        W, _ = comp.compute_mixing_weights(hidden)
        compressed = comp.apply_mixing(W, [(K_in, V_in)])
        K_out, V_out = compressed[0]
        (K_out.sum() + V_out.sum()).backward()
        assert K_in.grad is not None and V_in.grad is not None
        assert not torch.isnan(K_in.grad).any()


class TestAuxiliaryLosses:
    def test_coverage_loss_zero_when_disjoint_slots(self, device):
        # One-hot slots that pick distinct positions => zero overlap.
        B, K, T = 1, 4, 8
        W = torch.zeros(B, K, T, device=device)
        for k in range(K):
            W[0, k, 2 * k] = 1.0
        assert float(coverage_loss(W)) < 1e-6

    def test_coverage_loss_positive_when_collapsed(self, device):
        # All slots attend to the same position => high overlap.
        B, K, T = 1, 4, 8
        W = torch.zeros(B, K, T, device=device)
        W[:, :, 0] = 1.0
        assert float(coverage_loss(W)) > 0.1

    def test_sparsity_loss_decreases_with_concentration(self, device):
        B, K, T = 1, 4, 8
        uniform = torch.full((B, K, T), 1.0 / T, device=device)
        peaked = torch.zeros(B, K, T, device=device)
        peaked[:, :, 0] = 1.0
        assert float(sparsity_loss(uniform)) > float(sparsity_loss(peaked))
