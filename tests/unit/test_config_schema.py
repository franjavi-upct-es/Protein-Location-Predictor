# tests/unit/test_config_schema.py
"""Tests for the Pydantic configuration schema."""

from __future__ import annotations

from typing import Any

import pytest

from src.utils.config_schema import (
    ConfigValidationError,
    validate_config,
)


def _minimal_valid_config() -> dict[str, Any]:
    """A minimal config that satisfies every required field in the schema."""
    return {
        "project": {"name": "test", "seed": 42, "log_level": "INFO"},
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "splits_dir": "data/splits",
            "models_dir": "models",
            "reports_dir": "reports",
        },
        "model": {
            "backbone": {
                "name": "facebook/esm2_t30_150M_UR50D",
                "embedding_dim": 640,
                "num_layers": 30,
                "max_position_embeddings": 1024,
            },
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["query", "key", "value", "dense"],
            },
            "classifier": {
                "hidden_dims": [256, 128],
                "dropout": 0.3,
                "activation": "gelu",
            },
        },
        "loss": {
            "focal": {"gamma": 2.0, "alpha": None},
            "hierarchical": {"enabled": True, "weight": 0.1},
        },
        "training": {
            "max_epochs": 30,
            "optimizer": {"name": "adamw", "lr": 2.0e-4},
            "scheduler": {
                "name": "cosine_warmup",
                "warmup_steps_fraction": 0.1,
            },
            "experiment": {
                "name": "test-exp",
                "tracking_uri": "mlruns",
            },
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_minimal_config_validates(self) -> None:
        result = validate_config(_minimal_valid_config())
        assert result["project"]["name"] == "test"
        assert result["model"]["lora"]["rank"] == 8

    def test_runtime_keys_passthrough(self) -> None:
        cfg = _minimal_valid_config()
        cfg["project_root"] = "/tmp/foo"
        cfg["hardware"] = {"profiles": {}, "fallback_tiers": []}
        result = validate_config(cfg)
        assert result["project_root"] == "/tmp/foo"
        assert "hardware" in result

    def test_quantization_defaults(self) -> None:
        result = validate_config(_minimal_valid_config())
        # Quantization is optional with sensible defaults
        assert result["model"]["quantization"]["enabled"] is False
        assert result["model"]["quantization"]["method"] == "nf4"

    def test_serving_defaults_bind_to_loopback(self) -> None:
        cfg = _minimal_valid_config()
        cfg["serving"] = {}
        result = validate_config(cfg)
        assert result["serving"]["host"] == "127.0.0.1"


# ---------------------------------------------------------------------------
# Typo detection — the whole reason this exists
# ---------------------------------------------------------------------------


class TestTypoDetection:
    def test_misspelled_lora_rank_is_caught(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"]["lora"]["rnak"] = 8  # typo
        with pytest.raises(ConfigValidationError, match="Unknown key"):
            validate_config(cfg)

    def test_misspelled_top_level_section_is_caught(self) -> None:
        cfg = _minimal_valid_config()
        cfg["modle"] = cfg.pop("model")  # typo at the top level
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)

    def test_unknown_optimizer_name_is_caught(self) -> None:
        cfg = _minimal_valid_config()
        cfg["training"]["optimizer"]["name"] = "lion"  # not in schema
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------


class TestRangeValidation:
    def test_negative_lr_rejected(self) -> None:
        cfg = _minimal_valid_config()
        cfg["training"]["optimizer"]["lr"] = -1.0
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)

    def test_dropout_above_one_rejected(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"]["classifier"]["dropout"] = 1.5
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)

    def test_lora_rank_zero_rejected(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"]["lora"]["rank"] = 0
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)

    def test_max_epochs_zero_rejected(self) -> None:
        cfg = _minimal_valid_config()
        cfg["training"]["max_epochs"] = 0
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_message_lists_offending_path(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"]["lora"]["rnak"] = 8
        try:
            validate_config(cfg)
        except ConfigValidationError as e:
            assert "lora" in str(e)
            assert "rnak" in str(e)
        else:
            pytest.fail("Expected ConfigValidationError")
