# src/utils/config.py
"""
Centralized configuration management.

Loads YAML config with a layered merge strategy:
    base.yaml  <--  training.yaml / inference.yaml  <--  CLI overrides

Usage:
    from src.utils.config import load_config
    cfg = load_config(
        overrides=[
            "training.batch_size=4", "model.lora.rank=16"
        ]
    )
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
_SENTINEL_FILES = ("pyproject.toml", "configs")


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward from *start* until a directory containing pyproject.toml
    is found.
    """
    current = Path(start or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        if all((parent / s).exists() for s in _SENTINEL_FILES):
            return Path(parent)
    raise FileExistsError(
        f"Could not locate project root (looker for {_SENTINEL_FILES}) "
        f"starting from {current}"
    )


PROJECT_ROOT = find_project_root()


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    """
    Load a single YAML file, returning an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_dot_override(cfg: dict, key_path: str, value: str) -> None:
    """Apply a dotted key override like 'training.batch_size=4' in-place."""
    keys = key_path.split(".")
    target = cfg
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]

    # Attempt type coercion
    raw = value
    if raw.lower() in ("true", "false"):
        target[keys[-1]] = raw.lower() == "true"
    elif raw.lower() == "null" or raw.lower() == "none":
        target[keys[-1]] = None
    else:
        try:
            target[keys[-1]] = int(raw)
        except ValueError:
            try:
                target[keys[-1]] = float(raw)
            except ValueError:
                target[keys[-1]] = raw


# ---------------------------------------------------------------------------
# DotDict — attribute-style access for config values
# ---------------------------------------------------------------------------


class DotDict(dict):
    """A dict subclass that supports attribute-style access for nested keys.

    Example:
        cfg = DotDict({"model": {"lora": {"rank": 8}}})
        assert cfg.model.lora.rank == 8
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no key '{key}'. "
                f"Available keys: {list(self.keys())}"
            ) from None
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None

    @classmethod
    def from_dict(cls, data: dict) -> DotDict:
        """Recursively convert a nested dict into a DotDict."""
        result = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = cls.from_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    cls.from_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    mode: str = "training",
    config_dir: Path | str | None = None,
    overrides: list[str] | None = None,
) -> DotDict:
    """Load the merged configuration.

    Args:
        mode: Which overlay to apply on top of base.yaml.
              One of "training", "inference", or "base" (no overlay).
        config_dir: Path to the configs/ directory.
                    Defaults to PROJECT_ROOT/configs.
        overrides: List of dotted key=value strings,
                   e.g. ["training.batch_size=4"].

    Returns:
        A DotDict with the full, merged configuration.
    """
    config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "configs"

    # 1. Load base
    cfg = _load_yaml(config_dir / "base.yaml")

    # 2. Merge mode-specific overlay
    if mode != "base":
        overlay = _load_yaml(config_dir / f"{mode}.yaml")
        cfg = _deep_merge(cfg, overlay)

    # 3. Merge hardware profiles (always loaded for reference)
    hw_profiles = _load_yaml(config_dir / "hardware" / "gpu_profiles.yaml")
    cfg = _deep_merge(cfg, {"hardware": hw_profiles})

    # 4. Environment variable overrides (PROT_LOC_ prefix)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("PROT_LOC_"):
            dot_key = env_key[len("PROT_LOC_") :].lower().replace("__", ".")
            _apply_dot_override(cfg, dot_key, env_val)

    # 5. CLI overrides
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: '{item}'")
            key_path, value = item.split("=", 1)
            _apply_dot_override(cfg, key_path, value)

    # 6. Inject project root as an absolute path
    cfg["project_root"] = str(PROJECT_ROOT)

    return DotDict.from_dict(cfg)


def resolve_path(cfg: DotDict, key: str) -> Path:
    """Resolve a config path key relative to the project root.

    Example:
        model_dir = resolve_path(cfg, "paths.models_dir")
    """
    root = Path(cfg.project_root)
    keys = key.split(".")
    value = cfg
    for k in keys:
        value = value[k]
    return Path(root / str(value))
