# tests/unit/test_metrics.py
"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_metrics,
    format_classification_report,
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
        assert (
            metrics["per_class"]["Cytoplasm"]["f1"] == 1.0
        )  # Class 0 perfect
        assert (
            metrics["per_class"]["Membrane"]["recall"] == 0.0
        )  # Class 1 missed


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
