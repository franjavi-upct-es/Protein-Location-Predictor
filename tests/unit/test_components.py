# tests/unit/test_components.py
"""
Tests for the small evaluation components:
threshold tuning, per-organism breakdown, and the comparison report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.per_organism import compute_per_organism_metrics
from src.evaluation.threshold_tuning import (
    load_thresholds,
    save_thresholds,
    tune_thresholds,
)

# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------


class TestTuneThresholds:
    """Tune thresholds on synthetic prob/target data."""

    def test_perfect_class_picks_low_threshold(self) -> None:
        """A class where positives have prob ~1 and negatives ~0
        should pick a threshold somewhere in the middle."""
        rng = np.random.default_rng(0)
        n = 200
        targets = np.zeros((n, 2), dtype=int)
        targets[: n // 2, 0] = 1
        probs = np.zeros((n, 2), dtype=np.float32)
        probs[: n // 2, 0] = rng.uniform(0.7, 1.0, size=n // 2)
        probs[n // 2 :, 0] = rng.uniform(0.0, 0.3, size=n // 2)

        result = tune_thresholds(probs, targets, ["A", "B"])
        assert 0.05 <= result["A"] <= 0.95
        # Class B has no positives -> fallback
        assert result["B"] == 0.5

    def test_thresholds_match_label_list(self) -> None:
        probs = np.random.rand(20, 3).astype(np.float32)
        targets = (np.random.rand(20, 3) > 0.5).astype(int)
        result = tune_thresholds(probs, targets, ["X", "Y", "Z"])
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            tune_thresholds(np.zeros((4, 3)), np.zeros((4, 2)), ["a", "b", "c"])

    def test_label_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="classes"):
            tune_thresholds(np.zeros((4, 3)), np.zeros((4, 3)), ["a", "b"])


class TestPersistThresholds:
    def test_round_trip(self, tmp_path: Path) -> None:
        thresholds = {"Nucleus": 0.42, "Cytoplasm": 0.6}
        path = tmp_path / "thresholds.json"
        save_thresholds(thresholds, path)
        loaded = load_thresholds(path)
        assert loaded == thresholds

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_thresholds(tmp_path / "nope.json") is None

    def test_load_invalid_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("not json")
        assert load_thresholds(path) is None


# ---------------------------------------------------------------------------
# Per-organism metrics
# ---------------------------------------------------------------------------


class TestPerOrganismMetrics:
    def test_basic_breakdown(self) -> None:
        # 6 samples, 3 classes, 2 organisms
        targets = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        preds = targets.copy()
        organisms = [9606, 9606, 559292, 559292, 559292, 9606]

        result = compute_per_organism_metrics(
            preds, targets, organisms, ["A", "B", "C"], min_samples=1
        )

        assert "Homo sapiens" in result
        assert "Saccharomyces cerevisiae" in result
        assert result["Homo sapiens"]["n_samples"] == 3
        assert result["Saccharomyces cerevisiae"]["n_samples"] == 3
        assert "_summary" in result
        assert result["_summary"]["n_organisms"] == 2

    def test_min_samples_skips_small_organisms(self) -> None:
        targets = np.array([[1, 0], [0, 1], [1, 0]])
        preds = targets.copy()
        organisms = [9606, 9606, 10090]
        result = compute_per_organism_metrics(preds, targets, organisms, ["A", "B"], min_samples=3)
        assert "Homo sapiens" not in result  # only 2 samples, below 3
        assert "Mus musculus" not in result

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_per_organism_metrics(
                np.zeros((3, 2)),
                np.zeros((3, 2)),
                [9606, 9606],  # only 2
                ["A", "B"],
            )


# ---------------------------------------------------------------------------
# Comparison report builder
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def test_runs_with_no_inputs(self, tmp_path: Path) -> None:
        from src.evaluation.comparison_report import build_report
        from src.utils.config import DotDict

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        cfg = DotDict.from_dict(
            {
                "paths": {"reports_dir": "reports"},
                "project_root": str(tmp_path),
            }
        )
        out = build_report(cfg)
        assert out.exists()
        text = out.read_text()
        assert "Sprint 6 — Comparison report" in text
        assert "not available" in text

    def test_picks_up_linear_probe(self, tmp_path: Path) -> None:
        from src.evaluation.comparison_report import build_report
        from src.utils.config import DotDict

        reports_dir = tmp_path / "reports"
        baselines_dir = reports_dir / "baselines"
        baselines_dir.mkdir(parents=True)

        (baselines_dir / "linear_probe.json").write_text(
            json.dumps(
                {
                    "model": "linear_probe",
                    "test": {
                        "f1_macro": 0.42,
                        "f1_micro": 0.5,
                        "precision_macro": 0.45,
                        "recall_macro": 0.4,
                        "exact_match_ratio": 0.3,
                        "hamming_loss": 0.2,
                    },
                }
            )
        )

        cfg = DotDict.from_dict(
            {
                "paths": {"reports_dir": "reports"},
                "project_root": str(tmp_path),
            }
        )
        out = build_report(cfg)
        text = out.read_text()
        assert "0.420" in text

    def test_describes_deeploc_demo_reference_honestly(self, tmp_path: Path) -> None:
        from src.evaluation.comparison_report import build_report
        from src.utils.config import DotDict

        reports_dir = tmp_path / "reports"
        benchmarks_dir = reports_dir / "benchmarks"
        benchmarks_dir.mkdir(parents=True)

        (benchmarks_dir / "deeploc.json").write_text(
            json.dumps(
                {
                    "reference_type": "deeploc_predictions",
                    "n_samples": 2,
                    "n_source_rows": 3,
                    "n_skipped_unmapped_labels": 1,
                    "overall": {
                        "f1_macro": 0.5,
                        "f1_micro": 0.5,
                        "precision_macro": 0.5,
                        "recall_macro": 0.5,
                        "exact_match_ratio": 0.5,
                        "hamming_loss": 0.1,
                    },
                }
            )
        )

        cfg = DotDict.from_dict(
            {
                "paths": {"reports_dir": "reports"},
                "project_root": str(tmp_path),
            }
        )
        out = build_report(cfg)
        text = out.read_text()
        assert "packaged DeepLoc 2.0 demo set" in text
        assert "not exist in this project's taxonomy" in text
