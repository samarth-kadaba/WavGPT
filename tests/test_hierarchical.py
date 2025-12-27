#!/usr/bin/env python3
"""
Tests for SSM-Guided Hierarchical Attention.

Run with: python -m pytest tests/test_hierarchical.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, "/home/ubuntu/WavGPT/src")

from wavgpt import (
    HierarchicalConfig,
    SelectiveSSM,
    BoundaryDetector,
    ChunkAggregator,
    ChunkAttention,
    HierarchicalSSMAttention,
    create_hierarchical_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def config():
    return HierarchicalConfig(
        hidden_size=256,
        n_heads=4,
        n_layers=2,
        ssm_d_state=8,
        min_chunk_size=4,
        max_chunk_size=32,
    )


# =============================================================================
# SSM Tests
# =============================================================================


class TestSelectiveSSM:
    """Tests for the SSM encoder."""

    def test_forward_shape(self, device):
        """SSM output should match input shape."""
        ssm = SelectiveSSM(d_model=256, d_state=16).to(device)
        x = torch.randn(2, 64, 256, device=device)

        output, hidden = ssm(x, return_hidden_states=True)

        assert output.shape == x.shape
        assert hidden is not None

    def test_hidden_states_returned(self, device):
        """Hidden states should be returned when requested."""
        ssm = SelectiveSSM(d_model=256, d_state=16).to(device)
        x = torch.randn(2, 64, 256, device=device)

        _, hidden = ssm(x, return_hidden_states=True)

        assert hidden is not None
        assert hidden.shape[0] == 2  # Batch
        assert hidden.shape[1] == 64  # Sequence


# =============================================================================
# Boundary Detection Tests
# =============================================================================


class TestBoundaryDetector:
    """Tests for boundary detection from SSM dynamics."""

    def test_boundary_probs_shape(self, device):
        """Boundary probabilities should have correct shape."""
        detector = BoundaryDetector(hidden_size=256).to(device)
        hidden_states = torch.randn(2, 64, 256, device=device)

        probs, velocity = detector(hidden_states, return_velocity=True)

        assert probs.shape == (2, 63)  # T-1 boundaries
        assert velocity.shape == (2, 63)

    def test_boundary_probs_range(self, device):
        """Boundary probabilities should be in [0, 1]."""
        detector = BoundaryDetector(hidden_size=256).to(device)
        hidden_states = torch.randn(2, 64, 256, device=device)

        probs, _ = detector(hidden_states)

        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_hard_boundaries_constraints(self, device):
        """Hard boundaries should respect min/max chunk size."""
        detector = BoundaryDetector(hidden_size=256).to(device)
        hidden_states = torch.randn(1, 128, 256, device=device)

        probs, _ = detector(hidden_states)
        boundaries = detector.get_hard_boundaries(
            probs,
            min_chunk_size=8,
            max_chunk_size=32,
        )

        # Check boundaries exist
        assert boundaries.shape == (1, 127)

        # Find boundary positions
        boundary_positions = torch.where(boundaries[0])[0].tolist()

        # Check max chunk size constraint
        last_pos = -1
        for pos in boundary_positions + [127]:
            chunk_size = pos - last_pos
            assert chunk_size <= 32, f"Chunk size {chunk_size} exceeds max 32"
            last_pos = pos

    def test_learnable_parameters(self, device):
        """All boundary detection parameters should be learnable."""
        detector = BoundaryDetector(hidden_size=256).to(device)

        # Check parameters exist and require grad
        assert hasattr(detector, "threshold")
        assert detector.threshold.requires_grad

        assert hasattr(detector, "temperature_logit")
        assert detector.temperature_logit.requires_grad

        assert hasattr(detector, "velocity_weight")
        assert detector.velocity_weight.requires_grad

        # Check get_learned_params returns values
        params = detector.get_learned_params()
        assert "threshold" in params
        assert "temperature" in params
        assert "velocity_mix" in params

    def test_gradient_flow(self, device):
        """Gradients should flow through boundary detection."""
        detector = BoundaryDetector(hidden_size=256).to(device)
        hidden_states = torch.randn(2, 64, 256, device=device, requires_grad=True)

        probs, _ = detector(hidden_states)
        loss = probs.mean()
        loss.backward()

        # Check gradients exist
        assert detector.threshold.grad is not None
        assert detector.temperature_logit.grad is not None
        assert detector.velocity_weight.grad is not None


# =============================================================================
# Chunk Aggregation Tests
# =============================================================================


class TestChunkAggregator:
    """Tests for chunk aggregation."""

    def test_aggregation_output_shape(self, device):
        """Aggregator should produce chunk embeddings."""
        aggregator = ChunkAggregator(hidden_size=256, aggregation="last").to(device)
        hidden_states = torch.randn(2, 64, 256, device=device)

        # Create some boundaries
        boundaries = torch.zeros(2, 63, dtype=torch.bool, device=device)
        boundaries[0, 15] = True  # Boundary at position 16
        boundaries[0, 31] = True  # Boundary at position 32
        boundaries[1, 20] = True  # Boundary at position 21

        chunks, ranges = aggregator(hidden_states, boundaries)

        assert chunks.dim() == 3
        assert chunks.shape[0] == 2  # Batch
        assert chunks.shape[2] == 256  # Hidden size

    def test_aggregation_methods(self, device):
        """Different aggregation methods should work."""
        for method in ["last", "mean", "attention"]:
            aggregator = ChunkAggregator(hidden_size=256, aggregation=method).to(device)
            hidden_states = torch.randn(1, 32, 256, device=device)
            boundaries = torch.zeros(1, 31, dtype=torch.bool, device=device)
            boundaries[0, 15] = True

            chunks, ranges = aggregator(hidden_states, boundaries)

            assert chunks.shape[0] == 1
            assert chunks.shape[2] == 256


# =============================================================================
# Chunk Attention Tests
# =============================================================================


class TestChunkAttention:
    """Tests for chunk-level attention."""

    def test_attention_output_shape(self, device):
        """Attention output should match input shape."""
        attn = ChunkAttention(hidden_size=256, n_heads=4).to(device)
        x = torch.randn(2, 16, 256, device=device)

        output = attn(x)

        assert output.shape == x.shape

    def test_attention_with_mask(self, device):
        """Attention should handle masks correctly."""
        attn = ChunkAttention(hidden_size=256, n_heads=4).to(device)
        x = torch.randn(2, 16, 256, device=device)
        mask = torch.ones(2, 16, device=device)
        mask[0, 8:] = 0  # Mask second half for first batch

        output = attn(x, attention_mask=mask)

        assert output.shape == x.shape


# =============================================================================
# Hierarchical Model Tests
# =============================================================================


class TestHierarchicalSSMAttention:
    """Tests for the full hierarchical model."""

    def test_forward_pass(self, config, device):
        """Model should produce output."""
        model = HierarchicalSSMAttention(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        result = model(x)

        assert "output" in result
        assert result["output"].dim() == 3

    def test_return_boundaries(self, config, device):
        """Model should return boundary info when requested."""
        model = HierarchicalSSMAttention(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        result = model(x, return_boundaries=True)

        assert "boundary_probs" in result
        assert "boundaries" in result
        assert "velocity" in result

    def test_return_chunks(self, config, device):
        """Model should return chunk info when requested."""
        model = HierarchicalSSMAttention(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        result = model(x, return_chunks=True)

        assert "chunk_embeddings" in result
        assert "chunk_ranges" in result
        assert "n_chunks" in result

    def test_variable_sequence_lengths(self, config, device):
        """Model should handle different sequence lengths."""
        model = HierarchicalSSMAttention(config).to(device)

        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(1, seq_len, config.hidden_size, device=device)
            result = model(x)
            assert result["output"].shape[0] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateFunctions:
    """Tests for convenience creation functions."""

    def test_create_hierarchical_model(self, device):
        """create_hierarchical_model should work."""
        model = create_hierarchical_model(
            hidden_size=256,
            n_heads=4,
            n_layers=2,
        ).to(device)

        x = torch.randn(1, 64, 256, device=device)
        result = model(x)

        assert "output" in result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
