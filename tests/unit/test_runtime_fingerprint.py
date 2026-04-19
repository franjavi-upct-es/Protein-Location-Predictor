# tests/unit/test_runtime_fingerprint.py
"""Tests for RuntimeFingerprintCallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.runtime_fingerprint import (
    RuntimeFingerprintCallback,
    _hash_file,
)
from src.utils.config import DotDict


@pytest.fixture()
def minimal_cfg(tmp_path: Path) -> DotDict:
    """A minimal config sufficient for the callback to run."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.csv").write_text("accession,sequence\nP00001,MSKG\n")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    return DotDict.from_dict(
        {
            "project": {"name": "test", "seed": 42},
            "model": {
                "backbone": {"name": "facebook/esm2_t6_8M_UR50D"},
                "lora": {"rank": 4, "alpha": 8},
            },
            "training": {"batch_size": 2},
            "paths": {
                "splits_dir": str(splits_dir),
                "reports_dir": str(reports_dir),
            },
            "project_root": str(tmp_path),
        }
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class TestFingerprintBuilders:
    def test_build_returns_required_keys(self, minimal_cfg: DotDict) -> None:
        cb = RuntimeFingerprintCallback(minimal_cfg)
        fp = cb._build_fingerprint()

        for key in (
            "timestamp_utc",
            "platform",
            "gpu",
            "packages",
            "git",
            "config_sha256",
            "splits",
            "env",
        ):
            assert key in fp

    def test_platform_info(self, minimal_cfg: DotDict) -> None:
        cb = RuntimeFingerprintCallback(minimal_cfg)
        info = cb._platform_info()
        assert info["python_version"]
        assert info["system"]

    def test_package_versions_includes_known_packages(self, minimal_cfg: DotDict) -> None:
        cb = RuntimeFingerprintCallback(minimal_cfg)
        pkgs = cb._package_versions()
        # torch is a hard dep so it must be there
        assert "torch" in pkgs
        # Each value is either a version string or 'not installed'
        for v in pkgs.values():
            assert isinstance(v, str)

    def test_config_hash_is_stable(self, minimal_cfg: DotDict) -> None:
        cb1 = RuntimeFingerprintCallback(minimal_cfg)
        cb2 = RuntimeFingerprintCallback(minimal_cfg)
        assert cb1._config_hash() == cb2._config_hash()

    def test_split_hashes_picks_up_train_csv(self, minimal_cfg: DotDict) -> None:
        cb = RuntimeFingerprintCallback(minimal_cfg)
        hashes = cb._split_hashes()
        assert hashes.get("train") is not None
        assert hashes.get("val") is None
        assert hashes.get("test") is None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_on_train_start_persists_to_disk(self, minimal_cfg: DotDict, tmp_path: Path) -> None:
        cb = RuntimeFingerprintCallback(minimal_cfg)
        trainer = MagicMock()
        trainer.logger = None  # no MLflow
        pl_module = MagicMock()

        cb.on_train_start(trainer, pl_module)

        assert cb.fingerprint is not None
        out_dir = tmp_path / "reports" / "fingerprints"
        assert out_dir.exists()
        files = list(out_dir.glob("*.json"))
        assert len(files) == 1

    def test_on_train_start_does_not_raise_on_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a callback with a config that has no paths section
        cfg = DotDict.from_dict({"model": {}, "training": {}})
        cb = RuntimeFingerprintCallback(cfg)
        trainer = MagicMock()
        trainer.logger = None
        # Must not raise even though resolve_path will fail
        cb.on_train_start(trainer, MagicMock())


# ---------------------------------------------------------------------------
# File hashing helper
# ---------------------------------------------------------------------------


class TestHashFile:
    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello world")
        b.write_text("hello world")
        assert _hash_file(a) == _hash_file(b)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello")
        b.write_text("world")
        assert _hash_file(a) != _hash_file(b)
