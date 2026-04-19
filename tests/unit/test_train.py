# tests/unit/test_train.py
"""Tests for training runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.training.train import (
    _configure_torch_runtime,
    _resolve_mlflow_tracking_uri,
    train,
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
                "training": {"experiment": {"tracking_uri": "sqlite:///mlflow.db"}},
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


class TestCheckpointLoading:
    """Tests for trusted local checkpoint restore behavior."""

    def test_test_phase_uses_full_checkpoint_restore(self, monkeypatch) -> None:
        cfg = DotDict.from_dict(
            {
                "project": {"seed": 42},
                "training": {
                    "deterministic": True,
                    "batch_size": 2,
                    "precision": "32",
                    "gradient_checkpointing": False,
                    "max_sequence_length": 128,
                },
            }
        )

        test_calls: list[dict[str, object]] = []
        model = object()
        datamodule = object()

        class DummyTrainer:
            checkpoint_callback = SimpleNamespace(best_model_path="/tmp/best.ckpt")

            def fit(self, model, datamodule=None) -> None:  # type: ignore[no-untyped-def]
                return None

            def test(self, model, datamodule=None, ckpt_path=None, weights_only=None) -> None:  # type: ignore[no-untyped-def]
                test_calls.append(
                    {
                        "ckpt_path": ckpt_path,
                        "weights_only": weights_only,
                    }
                )

        monkeypatch.setattr("src.training.train.seed_everything", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "src.training.train.detect_hardware",
            lambda cfg: SimpleNamespace(
                batch_size=2,
                precision="32",
                gradient_checkpointing=False,
                max_sequence_length=128,
            ),
        )
        monkeypatch.setattr("src.training.train._configure_torch_runtime", lambda hw: None)
        monkeypatch.setattr(
            "src.training.train._discover_labels",
            lambda cfg: (["Cytoplasm"], torch.tensor([1.0])),
        )
        monkeypatch.setattr(
            "src.training.train.ProteinDataModule",
            lambda cfg, label_list: datamodule,
        )
        monkeypatch.setattr(
            "src.training.train.ProteinLocalizationModule",
            lambda cfg, label_list, class_frequencies: model,
        )
        monkeypatch.setattr("src.training.train._build_trainer", lambda cfg, hw: DummyTrainer())
        monkeypatch.setattr(
            "src.training.train._tune_and_save_thresholds",
            lambda cfg, model, dm: None,
        )

        train(cfg)

        assert test_calls == [
            {
                "ckpt_path": "/tmp/best.ckpt",
                "weights_only": False,
            }
        ]
