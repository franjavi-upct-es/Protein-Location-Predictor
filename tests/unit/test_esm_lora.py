# tests/unit/test_esm_lora.py
"""Tests for ESM-2 backbone utilities (pooling strategies).

Note: Tests that require loading the actual ESM-2 model are marked as 'slow'
and 'gpu' since they need significant resources. Unit tests here focus on
the pooling and utility functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.models.esm_lora import (
    extract_sequence_representation,
    get_embedding_dim,
)
from src.utils.config import DotDict

# ---------------------------------------------------------------------------
# Mock ESM-2 output
# ---------------------------------------------------------------------------


def _mock_model_output(
    batch_size: int = 2, seq_len: int = 10, hidden_dim: int = 16
):
    """Create a mock model output with last_hidden_state."""
    output = MagicMock()
    output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    return output


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------


class TestExtractSequenceRepresentation:
    """Tests for sequence representation extraction."""

    def test_cls_pooling_shape(self) -> None:
        output = _mock_model_output(batch_size=4, seq_len=20, hidden_dim=32)
        mask = torch.ones(4, 20, dtype=torch.long)

        result = extract_sequence_representation(output, mask, pooling="cls")
        assert result.shape == (4, 32)

    def test_mean_pooling_shape(self) -> None:
        output = _mock_model_output(batch_size=4, seq_len=20, hidden_dim=32)
        mask = torch.ones(4, 20, dtype=torch.long)

        result = extract_sequence_representation(output, mask, pooling="mean")
        assert result.shape == (4, 32)

    def test_mean_cls_pooling_shape(self) -> None:
        output = _mock_model_output(batch_size=4, seq_len=20, hidden_dim=32)
        mask = torch.ones(4, 20, dtype=torch.long)

        result = extract_sequence_representation(
            output, mask, pooling="mean_cls"
        )
        assert result.shape == (4, 64)  # 2 * hidden_dim

    def test_mean_pooling_respects_mask(self) -> None:
        """Padding tokens (mask=0) should not affect the mean."""
        B, L, D = 1, 10, 4
        output = _mock_model_output(B, L, D)

        # Only first 5 tokens are real
        mask = torch.zeros(B, L, dtype=torch.long)
        mask[0, :5] = 1

        result = extract_sequence_representation(output, mask, pooling="mean")

        # Manual mean over first 5 tokens
        expected = output.last_hidden_state[0, :5, :].mean(dim=0, keepdim=True)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_cls_pooling_takes_first_token(self) -> None:
        output = _mock_model_output(1, 10, 8)
        mask = torch.ones(1, 10, dtype=torch.long)

        result = extract_sequence_representation(output, mask, pooling="cls")
        expected = output.last_hidden_state[0, 0, :].unsqueeze(0)
        assert torch.allclose(result, expected)

    def test_unknown_pooling_raises(self) -> None:
        output = _mock_model_output(1, 10, 8)
        mask = torch.ones(1, 10, dtype=torch.long)

        with pytest.raises(ValueError, match="Unknown pooling"):
            extract_sequence_representation(output, mask, pooling="max")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestGetEmbeddingDim:
    """Tests for embedding dimension extraction from config."""

    def test_returns_configured_value(self) -> None:
        cfg = DotDict.from_dict(
            {"model": {"backbone": {"embedding_dim": 640}}}
        )
        assert get_embedding_dim(cfg) == 640

    def test_default_value(self) -> None:
        cfg = DotDict.from_dict({"model": {"backbone": {}}})
        assert get_embedding_dim(cfg) == 640
