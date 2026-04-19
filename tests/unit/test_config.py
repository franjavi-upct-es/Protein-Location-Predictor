# tests/unit/test_config.py
"""Tests for the configuration loading and merging system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    DotDict,
    _apply_dot_override,
    _deep_merge,
    load_config,
    to_builtin,
)

# ---------------------------------------------------------------------------
# DotDict
# ---------------------------------------------------------------------------


class TestDotDict:
    """Tests for attribute-style access on DotDict."""

    def test_basic_access(self) -> None:
        d = DotDict({"a": 1, "b": "hello"})
        assert d.a == 1
        assert d.b == "hello"

    def test_nested_access(self) -> None:
        d = DotDict.from_dict({"model": {"lora": {"rank": 8}}})
        assert d.model.lora.rank == 8

    def test_missing_key_raises_attribute_error(self) -> None:
        d = DotDict({"a": 1})
        with pytest.raises(AttributeError, match="no key 'missing'"):
            _ = d.missing

    def test_set_attr(self) -> None:
        d = DotDict()
        d.x = 42
        assert d["x"] == 42

    def test_from_dict_with_lists(self) -> None:
        d = DotDict.from_dict({"entries": [{"name": "a"}, {"name": "b"}]})
        assert d.entries[0].name == "a"
        assert d.entries[1].name == "b"

    def test_from_dict_preserves_none(self) -> None:
        d = DotDict.from_dict({"value": None})
        assert d.value is None

    def test_to_builtin_converts_nested_dotdicts(self) -> None:
        d = DotDict.from_dict(
            {
                "model": {"pooling": "mean"},
                "entries": [{"name": "a"}, {"name": "b"}],
            }
        )

        plain = to_builtin(d)

        assert type(plain) is dict
        assert type(plain["model"]) is dict
        assert type(plain["entries"][0]) is dict
        assert plain["entries"][1]["name"] == "b"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Tests for recursive dict merging."""

    def test_flat_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"model": {"backbone": "esm2", "lora": {"rank": 8}}}
        override = {"model": {"lora": {"rank": 16, "alpha": 32}}}
        result = _deep_merge(base, override)
        assert result["model"]["backbone"] == "esm2"
        assert result["model"]["lora"]["rank"] == 16
        assert result["model"]["lora"]["alpha"] == 32

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1


# ---------------------------------------------------------------------------
# Dot override
# ---------------------------------------------------------------------------


class TestDotOverride:
    """Tests for CLI-style dotted key overrides."""

    def test_simple_override(self) -> None:
        cfg: dict = {"training": {"batch_size": 8}}
        _apply_dot_override(cfg, "training.batch_size", "4")
        assert cfg["training"]["batch_size"] == 4

    def test_creates_missing_keys(self) -> None:
        cfg: dict = {}
        _apply_dot_override(cfg, "a.b.c", "hello")
        assert cfg["a"]["b"]["c"] == "hello"

    def test_bool_coercion(self) -> None:
        cfg: dict = {}
        _apply_dot_override(cfg, "flag", "true")
        assert cfg["flag"] is True
        _apply_dot_override(cfg, "flag2", "false")
        assert cfg["flag2"] is False

    def test_null_coercion(self) -> None:
        cfg: dict = {}
        _apply_dot_override(cfg, "value", "null")
        assert cfg["value"] is None
        _apply_dot_override(cfg, "value2", "none")
        assert cfg["value2"] is None

    def test_float_coercion(self) -> None:
        cfg: dict = {}
        _apply_dot_override(cfg, "lr", "0.001")
        assert cfg["lr"] == 0.001

    def test_string_fallback(self) -> None:
        cfg: dict = {}
        _apply_dot_override(cfg, "name", "my_experiment")
        assert cfg["name"] == "my_experiment"


# ---------------------------------------------------------------------------
# Full config loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the full config loading pipeline."""

    @pytest.fixture()
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory with minimal YAML files."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "hardware").mkdir()

        base = {
            "project": {"name": "test", "seed": 42, "log_level": "INFO"},
            "paths": {"data_dir": "data", "models_dir": "models"},
            "model": {"backbone": {"name": "esm2"}, "lora": {"rank": 8}},
        }
        with open(config_dir / "base.yaml", "w") as f:
            yaml.dump(base, f)

        training = {
            "training": {"max_epochs": 10, "batch_size": 4},
        }
        with open(config_dir / "training.yaml", "w") as f:
            yaml.dump(training, f)

        hw = {
            "profiles": {},
            "fallback_tiers": [
                {"vram_up_to_gb": 8, "recommended": {"batch_size": 2}},
            ],
        }
        with open(config_dir / "hardware" / "gpu_profiles.yaml", "w") as f:
            yaml.dump(hw, f)

        # Create a pyproject.toml sentinel so find_project_root works
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "configs").exists()  # already created above

        return config_dir

    def test_loads_base_only(self, temp_config_dir: Path) -> None:
        cfg = load_config(mode="base", config_dir=temp_config_dir)
        assert cfg.project.name == "test"
        assert cfg.model.lora.rank == 8

    def test_merges_training_overlay(self, temp_config_dir: Path) -> None:
        cfg = load_config(mode="training", config_dir=temp_config_dir)
        assert cfg.training.max_epochs == 10
        assert cfg.model.lora.rank == 8  # base value preserved

    def test_cli_overrides(self, temp_config_dir: Path) -> None:
        cfg = load_config(
            mode="training",
            config_dir=temp_config_dir,
            overrides=["model.lora.rank=16", "training.max_epochs=5"],
        )
        assert cfg.model.lora.rank == 16
        assert cfg.training.max_epochs == 5

    def test_env_var_override(self, temp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROT_LOC_MODEL__LORA__RANK", "32")
        cfg = load_config(mode="base", config_dir=temp_config_dir)
        assert cfg.model.lora.rank == 32

    def test_invalid_override_raises(self, temp_config_dir: Path) -> None:
        with pytest.raises(ValueError, match="key=value"):
            load_config(
                mode="base",
                config_dir=temp_config_dir,
                overrides=["bad_format"],
            )

    def test_hardware_profiles_loaded(self, temp_config_dir: Path) -> None:
        cfg = load_config(mode="base", config_dir=temp_config_dir)
        assert "hardware" in cfg
        assert "fallback_tiers" in cfg.hardware
