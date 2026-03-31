# tests/conftest.py
"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture()
def config_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    """Create a minimal temporary config directory for isolated tests."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "hardware").mkdir()

    base = {
        "project": {"name": "test-project", "seed": 42, "log_level": "DEBUG"},
        "paths": {"data_dir": "data", "models_dir": "models"},
        "model": {
            "backbone": {
                "name": "facebook/esm2_t30_150M_UR50D",
                "embedding_dim": 640,
            },
            "lora": {"rank": 8, "alpha": 16},
        },
        "processing": {
            "min_samples_per_class": 10,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
        },
    }
    with open(config_dir / "base.yaml", "w") as f:
        yaml.dump(base, f)

    training = {"training": {"max_epochs": 5, "batch_size": 2}}
    with open(config_dir / "training.yaml", "w") as f:
        yaml.dump(training, f)

    hw = {"profiles": {}, "fallback_tiers": []}
    with open(config_dir / "hardware" / "gpu_profiles.yaml", "w") as f:
        yaml.dump(hw, f)

    # Sentinel file for project root detection
    (tmp_path / "pyproject.toml").touch()

    return config_dir


@pytest.fixture()
def sample_sequences() -> list[str]:
    """Return a small set of valid protein sequences for testing."""
    return [
        "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATIGKLTLKFICTTGKLPVP",
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDKPHKNREYRQVVIDGET",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTL",
        "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEM",
    ]


@pytest.fixture()
def sample_locations() -> list[list[str]]:
    """Return multi-label location annotations matching sample_sequences."""
    return [
        ["Cytoplasm"],
        ["Membrane", "Cytoplasm"],
        ["Nucleus"],
        ["Cytoplasm", "Mitochondrion"],
    ]
