#!/usr/bin/env python3
"""
Tests for Context Extension via GRPO.

Run with: python -m pytest tests/test_context_extension.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, "/home/ubuntu/WavGPT/src")

from wavgpt import (
    ContextExtenderConfig,
    SelectiveSSM,
    SSMLayer,
    SSMBackbone,
    BoundaryPolicy,
    ChunkCompressor,
    ChunkInjector,
)
from wavgpt.models.policy import BoundarySample


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def config():
    return ContextExtenderConfig(
        max_context=128,  # Small for tests
        chunk_dim=64,
        n_ssm_layers=2,
        ssm_d_state=8,
        grpo_num_samples=2,
    )


# =============================================================================
# SSM Tests
# =============================================================================


class TestSelectiveSSM:
    """Tests for the SSM implementation."""

    def test_forward_shape(self, device):
        """SSM output should match input shape."""
        ssm = SelectiveSSM(d_model=64, d_state=8).to(device)
        x = torch.randn(2, 32, 64, device=device)
        
        output = ssm(x)
        
        assert output.shape == x.shape

    def test_step(self, device):
        """Incremental step should work correctly."""
        ssm = SelectiveSSM(d_model=64, d_state=8).to(device)
        
        batch_size = 2
        conv_state, ssm_state = ssm.get_initial_state(batch_size, device)
        
        x = torch.randn(batch_size, 64, device=device)
        output, new_conv, new_ssm = ssm.step(x, conv_state, ssm_state)
        
        assert output.shape == (batch_size, 64)
        assert new_conv.shape == conv_state.shape
        assert new_ssm.shape == ssm_state.shape


class TestSSMBackbone:
    """Tests for SSM backbone."""

    def test_forward(self, device):
        """Backbone forward pass."""
        backbone = SSMBackbone(d_model=64, n_layers=2, d_state=8).to(device)
        x = torch.randn(2, 32, 64, device=device)
        
        output = backbone(x)
        
        assert output.shape == x.shape


# =============================================================================
# Policy Tests
# =============================================================================


class TestBoundaryPolicy:
    """Tests for boundary policy."""

    def test_forward(self, config, device):
        """Policy forward pass."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        assert output.boundary_logits.shape == (2, 64)
        assert output.boundary_probs.shape == (2, 64)
        assert output.hidden_states.shape == x.shape

    def test_boundary_probs_range(self, config, device):
        """Boundary probabilities should be in [0, 1]."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        assert (output.boundary_probs >= 0).all()
        assert (output.boundary_probs <= 1).all()

    def test_first_position_no_boundary(self, config, device):
        """First position should never be a boundary."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        # First position probability should be ~0
        assert (output.boundary_probs[:, 0] < 0.01).all()

    def test_sample(self, config, device):
        """Sampling should return valid boundary configurations."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        samples, output = policy.sample(x, num_samples=3)
        
        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, BoundarySample)
            assert sample.boundaries.shape == (2, 64)
            assert sample.log_probs.shape == (2,)
            # Boundaries should be binary
            assert ((sample.boundaries == 0) | (sample.boundaries == 1)).all()

    def test_deterministic_boundaries(self, config, device):
        """Deterministic mode should give consistent results."""
        policy = BoundaryPolicy(config).to(device)
        policy.eval()
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        with torch.no_grad():
            b1 = policy.get_boundaries_deterministic(x)
            b2 = policy.get_boundaries_deterministic(x)
        
        assert (b1 == b2).all()

    def test_grpo_loss(self, config, device):
        """GRPO loss computation."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        samples, _ = policy.sample(x, num_samples=4)
        
        # Fake rewards
        rewards = torch.randn(4, 2, device=device)
        
        loss, metrics = policy.compute_grpo_loss(samples, rewards)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert "policy/pg_loss" in metrics
        assert "policy/mean_reward" in metrics


# =============================================================================
# Compressor Tests
# =============================================================================


class TestChunkCompressor:
    """Tests for chunk compressor."""

    def test_forward(self, config, device):
        """Compressor forward pass."""
        input_dim = 128
        compressor = ChunkCompressor(config, input_dim).to(device)
        
        tokens = torch.randn(2, 64, input_dim, device=device)
        boundaries = torch.zeros(2, 64, device=device)
        # Add some boundaries
        boundaries[:, 15] = 1
        boundaries[:, 31] = 1
        boundaries[:, 47] = 1
        
        chunk_emb, chunk_mask = compressor(tokens, boundaries)
        
        # Now returns dynamic number of chunks, not max_chunks
        num_chunks = int(boundaries.sum(dim=-1).max().item()) + 1  # 3 boundaries + 1 = 4 chunks
        assert chunk_emb.shape == (2, num_chunks, config.chunk_dim)
        assert chunk_mask.shape == (2, num_chunks)

    def test_chunk_mask(self, config, device):
        """Chunk mask should indicate valid chunks."""
        input_dim = 128
        compressor = ChunkCompressor(config, input_dim).to(device)
        
        tokens = torch.randn(2, 64, input_dim, device=device)
        boundaries = torch.zeros(2, 64, device=device)
        boundaries[:, 31] = 1  # Only one boundary
        
        _, chunk_mask = compressor(tokens, boundaries)
        
        # At least chunk 0 and 1 should be active
        assert (chunk_mask[:, 0] > 0.5).all()
        assert (chunk_mask[:, 1] > 0.5).all()

    def test_chunk_independence(self, config, device):
        """
        CRITICAL TEST: Chunks must be processed independently.
        
        Changing tokens in chunk 0 should NOT affect chunk 1's embedding.
        This verifies that the SSM state is reset between chunks.
        """
        input_dim = 128
        compressor = ChunkCompressor(config, input_dim).to(device)
        compressor.eval()
        
        # Create tokens with a boundary at position 31
        tokens_a = torch.randn(1, 64, input_dim, device=device)
        boundaries = torch.zeros(1, 64, device=device)
        boundaries[:, 31] = 1  # Chunk 0: tokens 0-31, Chunk 1: tokens 32-63
        
        # Get chunk embeddings for original tokens
        with torch.no_grad():
            chunks_a, _ = compressor(tokens_a, boundaries)
        
        # Modify ONLY chunk 0 tokens (positions 0-31)
        tokens_b = tokens_a.clone()
        tokens_b[:, 0:16, :] = torch.randn(1, 16, input_dim, device=device)  # Modify first half of chunk 0
        
        with torch.no_grad():
            chunks_b, _ = compressor(tokens_b, boundaries)
        
        # Chunk 0 should be DIFFERENT (we modified its tokens)
        chunk0_diff = (chunks_a[:, 0, :] - chunks_b[:, 0, :]).abs().mean()
        assert chunk0_diff > 0.01, f"Chunk 0 should change when its tokens change, diff={chunk0_diff}"
        
        # Chunk 1 should be IDENTICAL (we didn't modify its tokens)
        chunk1_diff = (chunks_a[:, 1, :] - chunks_b[:, 1, :]).abs().mean()
        assert chunk1_diff < 1e-5, f"Chunk 1 should NOT change when chunk 0 tokens change, diff={chunk1_diff}"

    def test_chunk_independence_multiple_chunks(self, config, device):
        """
        Test chunk independence with multiple chunks.
        
        Modifying tokens in chunk 1 should not affect chunk 0 or chunk 2.
        """
        input_dim = 128
        compressor = ChunkCompressor(config, input_dim).to(device)
        compressor.eval()
        
        # Create 3 chunks: [0-20], [21-40], [41-63]
        tokens_a = torch.randn(1, 64, input_dim, device=device)
        boundaries = torch.zeros(1, 64, device=device)
        boundaries[:, 20] = 1  # End of chunk 0
        boundaries[:, 40] = 1  # End of chunk 1
        # Chunk 2 ends at 63 (implicit)
        
        with torch.no_grad():
            chunks_a, _ = compressor(tokens_a, boundaries)
        
        # Modify ONLY chunk 1 tokens (positions 21-40)
        tokens_b = tokens_a.clone()
        tokens_b[:, 25:35, :] = torch.randn(1, 10, input_dim, device=device)
        
        with torch.no_grad():
            chunks_b, _ = compressor(tokens_b, boundaries)
        
        # Chunk 0 should be IDENTICAL
        chunk0_diff = (chunks_a[:, 0, :] - chunks_b[:, 0, :]).abs().mean()
        assert chunk0_diff < 1e-5, f"Chunk 0 should NOT change, diff={chunk0_diff}"
        
        # Chunk 1 should be DIFFERENT
        chunk1_diff = (chunks_a[:, 1, :] - chunks_b[:, 1, :]).abs().mean()
        assert chunk1_diff > 0.01, f"Chunk 1 should change, diff={chunk1_diff}"
        
        # Chunk 2 should be IDENTICAL
        chunk2_diff = (chunks_a[:, 2, :] - chunks_b[:, 2, :]).abs().mean()
        assert chunk2_diff < 1e-5, f"Chunk 2 should NOT change, diff={chunk2_diff}"


class TestChunkInjector:
    """Tests for chunk injector."""

    def test_forward(self, config, device):
        """Injector forward pass."""
        pretrained_dim = 768
        injector = ChunkInjector(config, pretrained_dim).to(device)
        
        chunks = torch.randn(2, config.max_chunks, config.chunk_dim, device=device)
        mask = torch.ones(2, config.max_chunks, device=device)
        
        virtual_tokens, virtual_mask = injector(chunks, mask)
        
        assert virtual_tokens.shape == (2, config.max_chunks, pretrained_dim)
        assert virtual_mask.shape == mask.shape


# =============================================================================
# Integration Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through the system."""

    def test_policy_gradients(self, config, device):
        """Gradients should flow to policy parameters."""
        policy = BoundaryPolicy(config).to(device)
        x = torch.randn(2, 32, config.chunk_dim, device=device, requires_grad=True)
        
        samples, output = policy.sample(x, num_samples=2)
        
        # Fake rewards
        rewards = torch.randn(2, 2, device=device)
        loss, _ = policy.compute_grpo_loss(samples, rewards)
        
        loss.backward()
        
        # Boundary head should have gradients (renamed from policy_head)
        assert policy.boundary_head[0].weight.grad is not None
        assert not torch.isnan(policy.boundary_head[0].weight.grad).any()

    def test_compressor_gradients(self, config, device):
        """Gradients should flow through compressor."""
        input_dim = 64
        compressor = ChunkCompressor(config, input_dim).to(device)
        
        tokens = torch.randn(2, 32, input_dim, device=device, requires_grad=True)
        boundaries = torch.zeros(2, 32, device=device)
        boundaries[:, 15] = 1
        
        chunks, mask = compressor(tokens, boundaries)
        loss = chunks.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()


# =============================================================================
# GRPO Training Tests
# =============================================================================


class TestGRPOTraining:
    """Tests for GRPO training mechanics."""

    def test_advantage_normalization(self, device):
        """Advantages should be normalized per group."""
        rewards = torch.tensor([
            [1.0, 2.0],  # Sample 1
            [3.0, 4.0],  # Sample 2
            [2.0, 3.0],  # Sample 3
        ], device=device)
        
        # Compute group-relative advantages
        mean_r = rewards.mean(dim=0, keepdim=True)
        std_r = rewards.std(dim=0, keepdim=True) + 1e-8
        advantages = (rewards - mean_r) / std_r
        
        # Mean should be ~0, std should be ~1
        assert advantages.mean(dim=0).abs().max() < 0.1
        assert (advantages.std(dim=0) - 1.0).abs().max() < 0.1


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

