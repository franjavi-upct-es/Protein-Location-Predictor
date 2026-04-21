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


class _FixedLogitsModel:
    def __init__(self, logits: list[float], label_list: list[str]) -> None:
        self.calls: list[dict[str, object]] = []
        self.label_list = label_list
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def eval(self) -> _FixedLogitsModel:
        return self

    def to(self, device: str) -> _FixedLogitsModel:
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
        return self._logits


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

    def test_from_checkpoint_loads_thresholds_and_temperatures(self, monkeypatch, tmp_path) -> None:
        def fake_load_from_checkpoint(cls, checkpoint_path, **kwargs):  # type: ignore[no-untyped-def]
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
            lambda path: {"Nucleus": 0.75},
        )
        monkeypatch.setattr(
            "src.evaluation.calibration.load_temperatures",
            lambda path: [1.5],
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

        assert predictor.per_class_thresholds == {"Nucleus": 0.75}
        assert predictor.temperatures == [1.5]


class TestPredictorInference:
    """Tests for batch inference behavior."""

    def test_predict_applies_temperature_scaling_and_external_features(self, monkeypatch) -> None:
        feature_calls: list[list[str]] = []
        temperature_calls: list[tuple[np.ndarray, list[float]]] = []
        model = _RecordingModel()

        monkeypatch.setattr(
            "src.data.external_features.compute_all_external_features",
            lambda sequences, cfg: (
                feature_calls.append(list(sequences)) or np.array([[1.0, 2.0]], dtype=np.float32)
            ),
        )
        monkeypatch.setattr(
            "src.evaluation.calibration.apply_temperatures",
            lambda logits, temperatures: (
                temperature_calls.append((logits.copy(), list(temperatures)))
                or np.array([[0.9]], dtype=np.float64)
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
        predictor.temperatures = [2.0]

        outputs = predictor.predict("AAAAA")

        assert feature_calls == [["AAAAA"]]
        assert model.calls[0]["external_features"].tolist() == [[1.0, 2.0]]
        assert temperature_calls[0][0].shape == (1, 1)
        assert temperature_calls[0][1] == [2.0]
        assert outputs == [{"location": "Nucleus", "confidence": 0.9}]

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

    def test_predict_uses_chunked_inference_for_long_sequences(self, monkeypatch) -> None:
        model = _RecordingModel()
        split_calls: list[tuple[str, int, int]] = []
        aggregate_calls: list[tuple[int, str]] = []

        monkeypatch.setattr(
            "src.serving.chunking.split_into_chunks",
            lambda sequence, window_size, overlap: (
                split_calls.append((sequence, window_size, overlap)) or ["AAAA", "CCCC"]
            ),
        )
        monkeypatch.setattr(
            "src.serving.chunking.aggregate_logits",
            lambda chunk_logits, strategy="mean": (
                aggregate_calls.append((len(chunk_logits), strategy))
                or np.array([2.0], dtype=np.float64)
            ),
        )

        predictor = Predictor(
            model=model,
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus"],
            device="cpu",
            max_length=4,
            chunk_long_sequences=True,
            chunk_overlap=1,
        )

        outputs = predictor.predict("AAAAACCCCC")

        assert split_calls == [("AAAAACCCCC", 4, 1)]
        assert aggregate_calls == [(2, "mean")]
        assert len(model.calls) == 2
        assert outputs == [{"location": "Nucleus", "confidence": 0.8808}]

    def test_predict_respects_explicit_zero_threshold_override(self) -> None:
        predictor = Predictor(
            model=_FixedLogitsModel(
                logits=[-1.0, -0.4],
                label_list=["Nucleus", "Cytoplasm"],
            ),
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus", "Cytoplasm"],
            device="cpu",
            threshold=0.5,
            top_k=2,
        )

        outputs = predictor.predict("AAAAA", threshold=0.0, top_k=2)

        assert outputs == [
            {"location": "Cytoplasm", "confidence": 0.4013},
            {"location": "Nucleus", "confidence": 0.2689},
        ]

    def test_predict_batch_respects_explicit_zero_threshold_override(self) -> None:
        predictor = Predictor(
            model=_FixedLogitsModel(
                logits=[-1.0, -0.4],
                label_list=["Nucleus", "Cytoplasm"],
            ),
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus", "Cytoplasm"],
            device="cpu",
            threshold=0.5,
            top_k=2,
        )

        outputs = predictor.predict_batch(["AAAAA"], threshold=0.0, top_k=2)

        assert outputs == [
            [
                {"location": "Cytoplasm", "confidence": 0.4013},
                {"location": "Nucleus", "confidence": 0.2689},
            ]
        ]

    def test_compute_external_features_returns_none_for_zero_width_result(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.data.external_features.compute_all_external_features",
            lambda sequences, cfg: np.empty((len(sequences), 0), dtype=np.float32),
        )

        predictor = Predictor(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus"],
            cfg=DotDict.from_dict(
                {
                    "features": {
                        "biophysical": {
                            "enabled": True,
                            "properties": ["a"],
                        }
                    }
                }
            ),
            device="cpu",
        )

        assert predictor._compute_external_features(["AAAAA"]) is None

    def test_build_results_uses_per_class_thresholds_and_fallback(self) -> None:
        predictor = Predictor(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus", "Cytoplasm"],
            device="cpu",
            threshold=0.5,
            top_k=2,
        )
        predictor.per_class_thresholds = {
            "Nucleus": 0.95,
            "Cytoplasm": 0.2,
        }

        filtered = predictor._build_results(np.array([0.8, 0.25]), threshold=None, top_k=None)
        fallback = predictor._build_results(np.array([0.1, 0.15]), threshold=0.9, top_k=1)

        assert filtered == [{"location": "Cytoplasm", "confidence": 0.25}]
        assert fallback == [{"location": "Cytoplasm", "confidence": 0.15}]

    def test_warmup_runs_dummy_prediction(self, monkeypatch) -> None:
        predictor = Predictor(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            label_list=["Nucleus"],
            device="cpu",
        )
        seen_sequences: list[str] = []

        monkeypatch.setattr(
            predictor,
            "predict",
            lambda sequence: seen_sequences.append(sequence) or [],
        )

        predictor.warmup()

        assert seen_sequences == ["M" * 50 + "ACDEFGHIKLMNPQRSTVWYXU"]
