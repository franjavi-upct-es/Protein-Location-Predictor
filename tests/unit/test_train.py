# tests/unit/test_train.py
"""Tests for training runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.training.train import (
    _configure_torch_runtime,
    _resolve_mlflow_tracking_uri,
)
from src.utils.config import DotDict


class TestResolveMlflowTrackingUri:
    """Tests for MLflow tracking URI resolution."""

    def test_resolves_relative_tracking_path(self) -> None:
        cfg = DotDict.from_dict(
            {
                "project_root": "/tmp/protein-loc",
                "paths": {"mlflow_dir": "mlruns"},
                "training": {"experiment": {"tracking_uri": "mlruns"}},
            }
        )

        uri = _resolve_mlflow_tracking_uri(cfg)

        assert uri == "/tmp/protein-loc/mlruns"

    def test_preserves_database_uri(self) -> None:
        cfg = DotDict.from_dict(
            {
                "project_root": "/tmp/protein-loc",
                "paths": {"mlflow_dir": "mlruns"},
                "training": {
                    "experiment": {"tracking_uri": "sqlite:///mlflow.db"}
                },
            }
        )

        uri = _resolve_mlflow_tracking_uri(cfg)

        assert uri == "sqlite:///mlflow.db"


class TestConfigureTorchRuntime:
    """Tests for PyTorch runtime tuning."""

    def test_sets_matmul_precision_on_cuda(self, monkeypatch) -> None:
        calls: list[str] = []

        monkeypatch.setattr(
            "src.training.train.torch.set_float32_matmul_precision",
            lambda value: calls.append(value),
        )

        _configure_torch_runtime(SimpleNamespace(device="cuda"))

        assert calls == ["high"]

    def test_skips_matmul_precision_on_cpu(self, monkeypatch) -> None:
        calls: list[str] = []

        monkeypatch.setattr(
            "src.training.train.torch.set_float32_matmul_precision",
            lambda value: calls.append(value),
        )

        _configure_torch_runtime(SimpleNamespace(device="cpu"))

        assert calls == []


class TestTrainingHardwareOverrides:
    """Tests for hardware-driven training overrides."""

    def test_sequence_length_can_be_carried_in_training_cfg(self) -> None:
        cfg = DotDict.from_dict(
            {
                "training": {"max_sequence_length": None},
            }
        )

        cfg.training["max_sequence_length"] = 1024

        assert cfg.training.max_sequence_length == 1024
