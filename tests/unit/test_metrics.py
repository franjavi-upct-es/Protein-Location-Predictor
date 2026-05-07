# tests/unit/test_metrics.py
"""Tests for evaluation metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from src.evaluation.metrics import (
    collect_predictions,
    compute_metrics,
    format_classification_report,
    generate_report,
    plot_confusion_matrix,
    plot_per_class_f1,
)

LABEL_LIST = ["Cytoplasm", "Membrane", "Nucleus"]


class TestComputeMetrics:
    """Tests for multi-label metrics computation."""

    def test_perfect_predictions(self) -> None:
        targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
        predictions = targets.copy()

        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        assert metrics["overall"]["f1_macro"] == 1.0
        assert metrics["overall"]["exact_match_ratio"] == 1.0
        assert metrics["overall"]["hamming_loss"] == 0.0

    def test_all_wrong_predictions(self) -> None:
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[0, 1, 1], [1, 0, 1]])

        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        assert metrics["overall"]["f1_macro"] == 0.0
        assert metrics["overall"]["exact_match_ratio"] == 0.0

    def test_per_class_keys(self) -> None:
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])

        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        assert set(metrics["per_class"].keys()) == set(LABEL_LIST)
        for label in LABEL_LIST:
            m = metrics["per_class"][label]
            assert "precision" in m
            assert "recall" in m
            assert "f1" in m
            assert "support" in m

    def test_support_counts(self) -> None:
        targets = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        predictions = targets.copy()

        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        assert metrics["per_class"]["Cytoplasm"]["support"] == 2
        assert metrics["per_class"]["Membrane"]["support"] == 1
        assert metrics["per_class"]["Nucleus"]["support"] == 1

    def test_total_samples(self) -> None:
        targets = np.zeros((10, 3))
        predictions = np.zeros((10, 3))

        metrics = compute_metrics(predictions, targets, LABEL_LIST)
        assert metrics["overall"]["total_samples"] == 10

    def test_partial_match(self) -> None:
        """Multi-label: predict one correct out of two true labels."""
        targets = np.array([[1, 1, 0]])
        predictions = np.array([[1, 0, 0]])

        metrics = compute_metrics(predictions, targets, LABEL_LIST)
        assert metrics["overall"]["exact_match_ratio"] == 0.0  # Not exact
        assert metrics["per_class"]["Cytoplasm"]["f1"] == 1.0  # Class 0 perfect
        assert metrics["per_class"]["Membrane"]["recall"] == 0.0  # Class 1 missed


class TestFormatClassificationReport:
    """Tests for report formatting."""

    def test_contains_all_labels(self) -> None:
        targets = np.array([[1, 0, 0]])
        predictions = np.array([[1, 0, 0]])
        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        report = format_classification_report(metrics, LABEL_LIST)
        for label in LABEL_LIST:
            assert label in report

    def test_contains_metrics(self) -> None:
        targets = np.array([[1, 0, 0]])
        predictions = np.array([[1, 0, 0]])
        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        report = format_classification_report(metrics, LABEL_LIST)
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report
        assert "Hamming loss" in report


class TestCollectPredictions:
    def test_collect_predictions_with_external_features(self):
        class MockModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, external_features=None):
                # Returns high logits for class 0 if external_features present, low otherwise
                batch_size = input_ids.shape[0]
                logits = torch.full((batch_size, 3), -10.0)
                if external_features is not None:
                    logits[:, 0] = 10.0
                return logits

        model = MockModel()
        dataloader = [
            {
                "input_ids": torch.zeros((2, 5)),
                "attention_mask": torch.ones((2, 5)),
                "labels": torch.tensor([[1, 0, 0], [0, 1, 0]]),
                "external_features": torch.ones((2, 2)),
            }
        ]

        result = collect_predictions(model, dataloader, device="cpu", threshold=0.5)

        assert "probabilities" in result
        assert "predictions" in result
        assert "targets" in result

        np.testing.assert_allclose(result["probabilities"][:, 0], 1.0, atol=1e-3)
        assert np.all(result["predictions"][:, 0] == 1)

    def test_collect_predictions_no_external_features(self):
        class MockModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, external_features=None):
                batch_size = input_ids.shape[0]
                logits = torch.full((batch_size, 3), 10.0)
                logits[:, 1:] = -10.0
                return logits

        model = MockModel()
        model.eval = MagicMock()

        dataloader = [
            {
                "input_ids": torch.zeros((2, 5)),
                "attention_mask": torch.ones((2, 5)),
                "labels": torch.tensor([[1, 0, 0], [0, 1, 0]]),
            }
        ]

        result = collect_predictions(model, dataloader, device="cpu", threshold=0.5)

        assert np.all(result["predictions"][:, 0] == 1)
        assert np.all(result["predictions"][:, 1:] == 0)


class TestPlotting:
    def test_plot_per_class_f1(self, tmp_path):
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])
        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        output_file = tmp_path / "f1.png"
        plot_per_class_f1(metrics, LABEL_LIST, output_file)

        assert output_file.exists()

    def test_plot_confusion_matrix(self, tmp_path):
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])

        output_file = tmp_path / "cm.png"
        plot_confusion_matrix(predictions, targets, LABEL_LIST, output_file)

        assert output_file.exists()


class TestGenerateReport:
    def test_generate_report_success(self, tmp_path, monkeypatch):
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])
        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        generate_report(metrics, predictions, targets, LABEL_LIST, tmp_path)

        assert (tmp_path / "evaluation_report.txt").exists()
        assert (tmp_path / "per_class_metrics.csv").exists()
        assert (tmp_path / "figures" / "f1_scores_by_class.png").exists()
        assert (tmp_path / "figures" / "confusion_matrix.png").exists()

    def test_generate_report_import_error(self, tmp_path, monkeypatch):
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])
        metrics = compute_metrics(predictions, targets, LABEL_LIST)

        def mock_plot(*args, **kwargs):
            raise ImportError("Simulated missing matplotlib")

        monkeypatch.setattr("src.evaluation.metrics.plot_per_class_f1", mock_plot)

        generate_report(metrics, predictions, targets, LABEL_LIST, tmp_path)

        assert (tmp_path / "evaluation_report.txt").exists()
        # The exception is caught and logged, the function should return normally
