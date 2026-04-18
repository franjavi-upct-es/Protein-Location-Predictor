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
import torch.nn as nn

from src.models.esm_lora import (
    _resolve_lora_target_modules,
    extract_sequence_representation,
    get_embedding_dim,
)
from src.utils.config import DotDict

# ---------------------------------------------------------------------------
# Mock ESM-2 output
# ---------------------------------------------------------------------------


def _mock_model_output(batch_size: int = 2, seq_len: int = 10, hidden_dim: int = 16):
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

        result = extract_sequence_representation(output, mask, pooling="mean_cls")
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
        cfg = DotDict.from_dict({"model": {"backbone": {"embedding_dim": 640}}})
        assert get_embedding_dim(cfg) == 640

    def test_default_value(self) -> None:
        cfg = DotDict.from_dict({"model": {"backbone": {}}})
        assert get_embedding_dim(cfg) == 640


# ---------------------------------------------------------------------------
# LoRA target resolution
# ---------------------------------------------------------------------------


class _DummyLinear(nn.Module):
    def __init__(self, supports_lora: bool = True) -> None:
        super().__init__()
        self.supports_lora = supports_lora
        self.weight = nn.Parameter(torch.randn(2, 2))


class TestResolveLoraTargetModules:
    """Tests for target-module expansion and filtering."""

    def _make_model(self) -> nn.Module:
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.layer = nn.ModuleList([nn.Module()])
        block = model.encoder.layer[0]

        block.attention = nn.Module()
        block.attention.self = nn.Module()
        block.attention.self.query = _DummyLinear()
        block.attention.self.key = _DummyLinear()
        block.attention.self.value = _DummyLinear()
        block.attention.output = nn.Module()
        block.attention.output.dense = _DummyLinear()

        block.intermediate = nn.Module()
        block.intermediate.dense = _DummyLinear()

        block.output = nn.Module()
        block.output.dense = _DummyLinear()

        model.pooler = nn.Module()
        model.pooler.dense = _DummyLinear(supports_lora=False)

        return model

    def test_non_quantized_mode_keeps_configured_patterns(self) -> None:
        model = self._make_model()
        configured = ["query", "key", "value", "dense"]

        resolved = _resolve_lora_target_modules(
            base_model=model,
            configured_target_modules=configured,
            use_quantization=False,
        )

        assert resolved == configured

    def test_quantized_mode_expands_and_skips_incompatible_modules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = self._make_model()

        monkeypatch.setattr(
            "src.models.esm_lora._supports_peft_bnb_lora_target",
            lambda module: getattr(module, "supports_lora", True),
        )

        resolved = _resolve_lora_target_modules(
            base_model=model,
            configured_target_modules=["query", "key", "value", "dense"],
            use_quantization=True,
        )

        assert "encoder.layer.0.attention.self.query" in resolved
        assert "encoder.layer.0.attention.self.key" in resolved
        assert "encoder.layer.0.attention.self.value" in resolved
        assert "encoder.layer.0.attention.output.dense" in resolved
        assert "encoder.layer.0.intermediate.dense" in resolved
        assert "encoder.layer.0.output.dense" in resolved
        assert "pooler.dense" not in resolved
