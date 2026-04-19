# tests/unit/test_lightning_module.py
"""Tests for Lightning module checkpoint metadata compatibility."""

from __future__ import annotations

import torch

from src.models.lightning_module import ProteinLocalizationModule
from src.utils.config import DotDict


class TestCheckpointHyperparameters:
    """Regression tests for PyTorch weights_only checkpoint loading."""

    def test_saved_hparams_use_builtin_containers(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "src.models.lightning_module.build_esm_lora_backbone",
            lambda cfg, enable_gradient_checkpointing: torch.nn.Identity(),
        )
        monkeypatch.setattr(
            "src.models.lightning_module.get_embedding_dim",
            lambda cfg: 8,
        )
        monkeypatch.setattr(
            "src.models.lightning_module.ClassifierHead.from_config",
            lambda cfg, input_dim, num_classes: torch.nn.Linear(input_dim, num_classes),
        )
        monkeypatch.setattr(
            "src.models.lightning_module.CombinedLoss.from_config",
            lambda cfg, label_list, class_frequencies: torch.nn.Identity(),
        )

        cfg = {
            "training": {"gradient_checkpointing": False},
            "model": {"pooling": "mean"},
            "features": {"biophysical": {"enabled": False, "properties": []}},
            "multi_task": {"enabled": False},
        }

        model = ProteinLocalizationModule(
            cfg=cfg,
            label_list=["Nucleus", "Cytoplasm"],
        )

        assert isinstance(model.cfg, DotDict)
        assert type(model.hparams["cfg"]) is dict
        assert type(model.hparams["cfg"]["model"]) is dict

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"hyper_parameters": dict(model.hparams)}, checkpoint_path)

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert loaded["hyper_parameters"]["cfg"]["model"]["pooling"] == "mean"
