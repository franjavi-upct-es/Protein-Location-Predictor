# tests/unit/test_predictor.py
"""Tests for predictor checkpoint loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models.lightning_module import ProteinLocalizationModule
from src.serving.predictor import Predictor
from src.utils.config import DotDict


class _DummyModel:
    def __init__(self) -> None:
        self.label_list = ["Nucleus"]

    def eval(self) -> _DummyModel:
        return self

    def to(self, device: str) -> _DummyModel:
        return self


class _DummyTokenizer:
    def __call__(
        self,
        sequence: str,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 1024,
    ) -> dict[str, torch.Tensor]:
        length = min(len(sequence), max_length)
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.label_list = ["Nucleus"]

    def eval(self) -> _RecordingModel:
        return self

    def to(self, device: str) -> _RecordingModel:
        return self

    def __call__(self, *, input_ids, attention_mask, external_features=None):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "input_ids_shape": tuple(input_ids.shape),
                "external_features": external_features.detach().cpu().numpy()
                if external_features is not None
                else None,
            }
        )
        return torch.tensor([[1.0]], dtype=torch.float32)


class TestPredictorCheckpointLoading:
    """Tests for restoring trained checkpoints in inference."""

    def test_from_checkpoint_uses_full_checkpoint_restore(self, monkeypatch, tmp_path) -> None:
        calls: list[dict[str, object]] = []

        def fake_load_from_checkpoint(cls, checkpoint_path, **kwargs):  # type: ignore[no-untyped-def]
            calls.append({"checkpoint_path": checkpoint_path, **kwargs})
            return _DummyModel()

        monkeypatch.setattr(
            ProteinLocalizationModule,
            "load_from_checkpoint",
            classmethod(fake_load_from_checkpoint),
        )
        monkeypatch.setattr(
            AutoTokenizer,
            "from_pretrained",
            lambda model_name: SimpleNamespace(model_name=model_name),
        )
        monkeypatch.setattr(
            "src.evaluation.threshold_tuning.load_thresholds",
            lambda path: None,
        )

        cfg = DotDict.from_dict(
            {
                "model": {
                    "backbone": {
                        "name": "facebook/esm2_t6_8M_UR50D",
                        "max_position_embeddings": 1024,
                    }
                },
                "inference": {"threshold": 0.5, "top_k": 3},
            }
        )

        predictor = Predictor.from_checkpoint(tmp_path / "model.ckpt", cfg, device="cpu")

        assert predictor.device == "cpu"
        assert calls[0]["strict"] is False
        assert calls[0]["weights_only"] is False


class TestPredictorInference:
    """Tests for batch inference behavior."""

    def test_predict_batch_passes_external_features_when_enabled(self, monkeypatch) -> None:
        feature_calls: list[list[str]] = []
        model = _RecordingModel()

        monkeypatch.setattr(
            "src.data.external_features.compute_all_external_features",
            lambda sequences, cfg: (
                feature_calls.append(list(sequences))
                or np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            ),
        )

        predictor = Predictor(
            model=model,
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus"],
            cfg=DotDict.from_dict(
                {
                    "features": {
                        "biophysical": {
                            "enabled": True,
                            "properties": ["a", "b"],
                        }
                    }
                }
            ),
            device="cpu",
        )

        outputs = predictor.predict_batch(["AAAAA", "CCCCC"])

        assert feature_calls == [["AAAAA", "CCCCC"]]
        assert outputs[0][0]["location"] == "Nucleus"
        assert model.calls[0]["external_features"].tolist() == [[1.0, 2.0]]
        assert model.calls[1]["external_features"].tolist() == [[3.0, 4.0]]
