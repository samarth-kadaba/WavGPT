#!/usr/bin/env python3
"""
Tests for Context Extension via GRPO with Unified Policy-Compressor.

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
    PolicyCompressor,
    PolicySample,
    ChunkInjector,
)


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
# Unified Policy-Compressor Tests
# =============================================================================


class TestPolicyCompressor:
    """Tests for unified policy-compressor."""

    def test_forward(self, config, device):
        """PolicyCompressor forward pass."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        assert output.boundary_importance.shape == (2, 64)
        assert output.boundary_probs.shape == (2, 64)
        assert output.boundary_threshold.shape == (2,)
        assert output.keep_importance.shape == (2, 64)
        assert output.keep_probs.shape == (2, 64)
        assert output.keep_threshold.shape == (2,)
        assert output.hidden_states.shape == x.shape
        assert output.difficulty_scores.shape == (2, 64)

    def test_boundary_probs_range(self, config, device):
        """Boundary probabilities should be in [0, 1]."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        assert (output.boundary_probs >= 0).all()
        assert (output.boundary_probs <= 1).all()
        assert (output.keep_probs >= 0).all()
        assert (output.keep_probs <= 1).all()

    def test_first_position_no_boundary(self, config, device):
        """First position should never be a boundary."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        output = policy.forward(x)
        
        # First position probability should be ~0
        assert (output.boundary_probs[:, 0] < 0.01).all()

    def test_sample(self, config, device):
        """Sampling should return valid configurations."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        samples, output = policy.sample(x, num_samples=3)
        
        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, PolicySample)
            assert sample.boundaries.shape == (2, 64)
            assert sample.keep_mask.shape == (2, 64)
            assert sample.log_probs.shape == (2,)
            # Boundaries should be binary
            assert ((sample.boundaries == 0) | (sample.boundaries == 1)).all()
            assert ((sample.keep_mask == 0) | (sample.keep_mask == 1)).all()

    def test_compress(self, config, device):
        """Compression should work from shared hidden states."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        # Forward to get hidden states
        output = policy.forward(x)
        
        # Create boundaries
        boundaries = torch.zeros(2, 64, device=device)
        boundaries[:, 15] = 1
        boundaries[:, 31] = 1
        boundaries[:, 47] = 1
        
        # Compress using shared hidden states
        chunks, chunk_mask, difficulty = policy.compress(
            output.hidden_states, boundaries
        )
        
        assert chunks.shape == (2, config.max_chunks, config.chunk_dim)
        assert chunk_mask.shape == (2, config.max_chunks)
        assert difficulty.shape == (2, config.max_chunks)

    def test_grpo_loss(self, config, device):
        """GRPO loss computation with difficulty."""
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 64, config.chunk_dim, device=device)
        
        samples, output = policy.sample(x, num_samples=4)
        
        # Fake rewards and difficulties
        rewards = torch.randn(4, 2, device=device)
        difficulties = [torch.randn(2, config.max_chunks, device=device) for _ in range(4)]
        
        loss, metrics = policy.compute_grpo_loss(samples, rewards, chunk_difficulties=difficulties)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert "policy/pg_loss" in metrics
        assert "policy/mean_reward" in metrics
        assert "policy/budget_penalty" in metrics
        assert "policy/expected_context" in metrics
        assert "policy/temperature" in metrics


class TestChunkIndependence:
    """Tests for boundary hidden state compression."""

    def test_boundary_state_changes_with_input(self, config, device):
        """
        Test that boundary hidden states change when input tokens change.
        
        With boundary hidden states (not mean pooling), the chunk embedding
        is the SSM hidden state at the boundary position. This state depends
        on all tokens before it, but due to SSM decay, recent tokens have
        more influence than early tokens.
        
        We modify tokens NEAR the boundary to ensure visible change.
        """
        policy = PolicyCompressor(config).to(device)
        policy.eval()
        
        # Create tokens with a boundary at position 31
        tokens_a = torch.randn(1, 64, config.chunk_dim, device=device)
        boundaries = torch.zeros(1, 64, device=device)
        boundaries[:, 31] = 1  # Chunk 0: ends at 31, Chunk 1: ends at 63
        
        # Get chunk embeddings for original tokens
        with torch.no_grad():
            output_a = policy.forward(tokens_a)
            chunks_a, _, _ = policy.compress(output_a.hidden_states, boundaries)
        
        # Modify tokens NEAR the boundary (positions 25-31)
        # These have stronger influence on the boundary hidden state
        tokens_b = tokens_a.clone()
        tokens_b[:, 25:32, :] = torch.randn(1, 7, config.chunk_dim, device=device)
        
        with torch.no_grad():
            output_b = policy.forward(tokens_b)
            chunks_b, _, _ = policy.compress(output_b.hidden_states, boundaries)
        
        # Chunk 0 should change when we modify tokens before its boundary
        chunk0_diff = (chunks_a[:, 0, :] - chunks_b[:, 0, :]).abs().mean()
        assert chunk0_diff > 0.01, f"Chunk 0 should change when tokens near boundary change, diff={chunk0_diff}"

    def test_chunk_count_with_boundaries(self, config, device):
        """
        Test that N boundaries create N+1 chunks.
        
        - Chunk 0 ends at first boundary
        - Chunk 1 ends at second boundary
        - ...
        - Chunk N ends at sequence end (implicit boundary)
        """
        policy = PolicyCompressor(config).to(device)
        policy.eval()
        
        # Create 2 explicit boundaries → should get 3 chunks
        tokens_a = torch.randn(1, 64, config.chunk_dim, device=device)
        boundaries = torch.zeros(1, 64, device=device)
        boundaries[:, 20] = 1  # End of chunk 0
        boundaries[:, 40] = 1  # End of chunk 1
        # Chunk 2 ends at position 63 (implicit)
        
        with torch.no_grad():
            output_a = policy.forward(tokens_a)
            chunks_a, mask_a, diff_a = policy.compress(output_a.hidden_states, boundaries)
        
        # Verify we get 3 chunks marked as valid
        assert mask_a[:, 0:3].sum() == 3, "Should have 3 valid chunks (2 explicit + 1 implicit)"
        
        # Verify chunk embeddings are from correct positions
        # Chunk 0 should be from position 20, Chunk 1 from 40, Chunk 2 from 63
        expected_positions = [20, 40, 63]
        compressed = policy.compression_head(output_a.hidden_states)
        for i, pos in enumerate(expected_positions):
            expected_embed = compressed[0, pos, :]
            actual_embed = chunks_a[0, i, :]
            diff = (expected_embed - actual_embed).abs().mean()
            assert diff < 1e-5, f"Chunk {i} should be from position {pos}, diff={diff}"


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
        policy = PolicyCompressor(config).to(device)
        x = torch.randn(2, 32, config.chunk_dim, device=device, requires_grad=True)
        
        samples, output = policy.sample(x, num_samples=2)
        
        # Fake rewards
        rewards = torch.randn(2, 2, device=device)
        loss, _ = policy.compute_grpo_loss(samples, rewards)
        
        loss.backward()
        
        # Importance head should have gradients
        assert policy.importance_head[0].weight.grad is not None
        assert not torch.isnan(policy.importance_head[0].weight.grad).any()

    def test_compressor_gradients(self, config, device):
        """Gradients should flow through compression."""
        policy = PolicyCompressor(config).to(device)
        
        tokens = torch.randn(2, 32, config.chunk_dim, device=device, requires_grad=True)
        boundaries = torch.zeros(2, 32, device=device)
        boundaries[:, 15] = 1
        
        output = policy.forward(tokens)
        chunks, mask, diff = policy.compress(output.hidden_states, boundaries)
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
