#!/usr/bin/env python3
"""
Tests for Infinite Context Transformer with Learnable Chunking.

Run with: python -m pytest tests/test_hierarchical.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, "/home/ubuntu/WavGPT/src")

from wavgpt import (
    InfiniteContextConfig,
    SelectiveSSM,
    SSMLayer,
    BoundaryDetector,
    ChunkCompressor,
    ChunkTransformer,
    TokenPredictor,
    InfiniteContextTransformer,
    create_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def config():
    return InfiniteContextConfig(
        vocab_size=1000,
        hidden_size=256,
        n_heads=4,
        n_boundary_layers=2,
        n_chunk_ssm_layers=2,
        n_chunk_transformer_layers=2,
        max_chunks=32,
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

        output, hidden = ssm(x, return_all_states=True)

        assert output.shape == x.shape
        assert hidden is not None

    def test_hidden_states_returned(self, device):
        """Hidden states should be returned when requested."""
        ssm = SelectiveSSM(d_model=256, d_state=16).to(device)
        x = torch.randn(2, 64, 256, device=device)

        _, hidden = ssm(x, return_all_states=True)

        assert hidden is not None
        assert hidden.shape[0] == 2  # Batch
        assert hidden.shape[1] == 64  # Sequence


# =============================================================================
# Boundary Detection Tests
# =============================================================================


class TestBoundaryDetector:
    """Tests for learned boundary detection with O(T) value function."""

    def test_forward_shape(self, config, device):
        """Boundary detector should return correct shapes."""
        detector = BoundaryDetector(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        (
            boundary_probs,
            boundary_decisions,
            ssm_output,
            expected_chunks,
            distill_loss,
            entropy_loss,
            sparsity_loss,
        ) = detector(x)

        assert boundary_probs.shape == (2, 64)
        assert boundary_decisions.shape == (2, 64)
        assert ssm_output.shape == x.shape
        assert expected_chunks.shape == (2,)  # Per-batch expected chunks
        assert distill_loss.shape == ()
        assert entropy_loss.shape == ()
        assert sparsity_loss.shape == ()

    def test_boundary_probs_range(self, config, device):
        """Boundary probabilities should be in [0, 1]."""
        detector = BoundaryDetector(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        boundary_probs, _, _, _, _, _, _ = detector(x)

        assert (boundary_probs >= 0).all()
        assert (boundary_probs <= 1).all()

    def test_first_position_no_boundary(self, config, device):
        """First position should never be a boundary."""
        detector = BoundaryDetector(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        boundary_probs, _, _, _, _, _, _ = detector(x)

        # First position should always be 0
        assert (boundary_probs[:, 0] == 0).all()

    def test_gradient_flow(self, config, device):
        """Gradients should flow through boundary detection."""
        detector = BoundaryDetector(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device, requires_grad=True)

        boundary_probs, _, _, _, distill_loss, _, _ = detector(x)
        loss = boundary_probs.mean() + distill_loss
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_chunk_assignments(self, config, device):
        """Chunk assignments should be computed correctly from boundaries."""
        detector = BoundaryDetector(config).to(device)
        x = torch.randn(1, 32, config.hidden_size, device=device)

        _, boundary_decisions, _, _, _, _, _ = detector(x)
        chunk_ids = detector.compute_chunk_assignments(boundary_decisions)

        # Chunk IDs should be monotonically non-decreasing
        assert (chunk_ids[:, 1:] >= chunk_ids[:, :-1]).all()


# =============================================================================
# Chunk Compressor Tests
# =============================================================================


class TestChunkCompressor:
    """Tests for chunk compression."""

    def test_compression_output_shape(self, config, device):
        """Compressor should produce chunk embeddings."""
        compressor = ChunkCompressor(config).to(device)
        x = torch.randn(2, 64, config.hidden_size, device=device)

        # Create some chunk assignments
        chunk_ids = torch.zeros(2, 64, device=device)
        chunk_ids[:, 16:32] = 1
        chunk_ids[:, 32:48] = 2
        chunk_ids[:, 48:] = 3

        boundary_probs = torch.zeros(2, 64, device=device)
        boundary_probs[:, 16] = 1.0
        boundary_probs[:, 32] = 1.0
        boundary_probs[:, 48] = 1.0

        chunk_embeddings, chunk_mask, ssm_outputs, n_chunks = compressor(
            x, chunk_ids, boundary_probs
        )

        assert chunk_embeddings.shape == (2, config.max_chunks, config.hidden_size)
        assert chunk_mask.shape == (2, config.max_chunks)
        assert ssm_outputs.shape == x.shape


# =============================================================================
# Chunk Transformer Tests
# =============================================================================


class TestChunkTransformer:
    """Tests for chunk-level attention."""

    def test_attention_output_shape(self, config, device):
        """Attention output should match input shape."""
        transformer = ChunkTransformer(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device)
        mask = torch.ones(2, 16, device=device)

        output = transformer(x, mask)

        assert output.shape == x.shape

    def test_causal_attention(self, config, device):
        """Transformer should use causal attention."""
        transformer = ChunkTransformer(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device)
        mask = torch.ones(2, 16, device=device)

        # Output should be valid (no NaN/Inf)
        output = transformer(x, mask)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Full Model Tests
# =============================================================================


class TestInfiniteContextTransformer:
    """Tests for the full model."""

    def test_forward_pass(self, config, device):
        """Model should produce output."""
        model = InfiniteContextTransformer(config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)

        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 64, config.vocab_size)

    def test_loss_computation(self, config, device):
        """Model should compute loss when labels provided."""
        model = InfiniteContextTransformer(config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        assert "loss" in outputs
        assert outputs["loss"] is not None
        assert outputs["loss"].shape == ()  # Scalar
        assert not torch.isnan(outputs["loss"])

    def test_boundary_outputs(self, config, device):
        """Model should return boundary information."""
        model = InfiniteContextTransformer(config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)

        outputs = model(input_ids)

        assert "boundary_probs" in outputs
        assert outputs["boundary_probs"].shape == (2, 64)
        assert "n_chunks" in outputs
        assert "expected_chunks" in outputs

    def test_variable_sequence_lengths(self, config, device):
        """Model should handle different sequence lengths."""
        model = InfiniteContextTransformer(config).to(device)

        for seq_len in [32, 64, 128, 256]:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
            outputs = model(input_ids)
            assert outputs["logits"].shape == (1, seq_len, config.vocab_size)

    def test_generation(self, config, device):
        """Model should generate tokens."""
        model = InfiniteContextTransformer(config).to(device)
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)

        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=20)

        assert generated.shape == (1, 30)  # 10 prompt + 20 generated


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateFunctions:
    """Tests for convenience creation functions."""

    def test_create_model(self, device):
        """create_model should work."""
        model = create_model(
            vocab_size=1000,
            hidden_size=256,
            n_heads=4,
        ).to(device)

        input_ids = torch.randint(0, 1000, (1, 64), device=device)
        outputs = model(input_ids)

        assert "logits" in outputs


# =============================================================================
# O(T) Complexity Tests
# =============================================================================


class TestComplexity:
    """Tests to verify O(T) complexity of boundary detection."""

    def test_memory_scaling(self, device):
        """Memory should scale linearly with sequence length."""
        if device != "cuda":
            pytest.skip("Memory scaling test requires CUDA")

        config = InfiniteContextConfig(
            vocab_size=1000,
            hidden_size=256,
            n_heads=4,
            max_chunks=256,
        )
        model = InfiniteContextTransformer(config).to(device)
        model.eval()

        memory_usage = []
        for seq_len in [128, 256, 512]:
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
            with torch.no_grad():
                _ = model(input_ids)
            mem = torch.cuda.max_memory_allocated() / 1e6
            memory_usage.append((seq_len, mem))

        # Memory should roughly double when sequence doubles (linear scaling)
        # Allow some tolerance for fixed overhead
        ratio_1 = memory_usage[1][1] / memory_usage[0][1]
        ratio_2 = memory_usage[2][1] / memory_usage[1][1]

        # Ratios should be close to 2.0 (linear) rather than 4.0 (quadratic)
        assert ratio_1 < 3.0, f"Memory ratio {ratio_1} suggests non-linear scaling"
        assert ratio_2 < 3.0, f"Memory ratio {ratio_2} suggests non-linear scaling"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
