# tests/unit/test_pooling.py
"""Tests for the LightAttentionPooler module."""

from __future__ import annotations

import pytest
import torch

from src.models.pooling import LightAttentionPooler


class TestLightAttentionPooler:
    """Tests for trainable light attention pooling."""

    def test_output_shape(self) -> None:
        pooler = LightAttentionPooler(hidden_dim=32)
        hidden = torch.randn(4, 16, 32)
        mask = torch.ones(4, 16, dtype=torch.long)
        out = pooler(hidden, mask)
        assert out.shape == (4, 32)

    def test_respects_padding_mask(self) -> None:
        """Padding tokens should not contribute to the pooled output."""
        pooler = LightAttentionPooler(hidden_dim=8)
        pooler.eval()

        # Same hidden states for two batches, but the second has the
        # tail padded out. With identical real prefixes the outputs
        # should be equal because padding contributes zero.
        torch.manual_seed(0)
        prefix = torch.randn(1, 5, 8)
        suffix = torch.randn(1, 5, 8)

        h1 = torch.cat([prefix, suffix], dim=1)  # length 10
        m1 = torch.ones(1, 10, dtype=torch.long)

        h2 = torch.cat([prefix, suffix], dim=1)  # same content
        m2 = torch.zeros(1, 10, dtype=torch.long)
        m2[0, :5] = 1  # only first 5 are real

        out1 = pooler(h1, m1)
        out2 = pooler(h2, m2)

        # out2 should differ from out1 (different masks → different
        # softmax distribution)
        assert not torch.allclose(out1, out2, atol=1e-5)

        # Sanity: pooling only the real prefix of h2 should equal
        # pooling h2 with the truncated mask
        h_only = prefix
        m_only = torch.ones(1, 5, dtype=torch.long)
        out_truncated = pooler(h_only, m_only)
        assert torch.allclose(out2, out_truncated, atol=1e-5)

    def test_gradient_flow(self) -> None:
        pooler = LightAttentionPooler(hidden_dim=16)
        hidden = torch.randn(2, 10, 16, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.long)
        out = pooler(hidden, mask)
        out.sum().backward()
        assert hidden.grad is not None
        # The attention layer's weight must also have a gradient
        assert pooler.attention.weight.grad is not None

    def test_zero_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            LightAttentionPooler(hidden_dim=0)

    def test_initial_distribution_close_to_uniform(self) -> None:
        """At init, the softmax should be near-uniform so the model
        starts close to mean pooling and the optimizer has a sane
        starting point."""
        torch.manual_seed(0)
        pooler = LightAttentionPooler(hidden_dim=64)
        pooler.eval()

        hidden = torch.randn(1, 20, 64)

        # Recompute the weights manually to inspect them
        scores = pooler.attention(hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        # All weights should be roughly 1/L = 0.05
        uniform = 1.0 / 20
        # Allow up to 3x deviation from uniform per token (loose check)
        assert weights.max().item() < 3 * uniform
        assert weights.min().item() > uniform / 3

    def test_no_nan_with_full_padding_row(self) -> None:
        """Defensive: a row that is fully padded should not produce NaNs."""
        pooler = LightAttentionPooler(hidden_dim=16)
        pooler.eval()

        hidden = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0] = 0  # entire first row is padding

        out = pooler(hidden, mask)
        assert not torch.isnan(out).any()
