# tests/unit/test_comparison_report.py
"""Tests for the Sprint 6 comparison report builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import comparison_report
from src.utils.config import DotDict


class TestLoadJson:
    def test_load_json_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")

        assert comparison_report._load_json(path) is None

    def test_load_json_returns_none_for_non_object_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))

        assert comparison_report._load_json(path) is None


class TestRowFor:
    def test_row_for_uses_overall_when_requested_split_is_missing(self) -> None:
        label, values = comparison_report._row_for(
            "Trained model",
            {
                "overall": {
                    "f1_macro": 0.5,
                    "f1_micro": 0.6,
                    "precision_macro": 0.7,
                    "recall_macro": 0.8,
                    "exact_match_ratio": 0.9,
                    "hamming_loss": 0.1,
                }
            },
            "test",
        )

        assert label == "Trained model"
        assert values == ["0.500", "0.600", "0.700", "0.800", "0.900", "0.100"]


class TestBuildReport:
    def test_build_report_includes_per_class_sections_and_ground_truth_benchmark(
        self, tmp_path: Path
    ) -> None:
        reports_dir = tmp_path / "reports"
        baselines_dir = reports_dir / "baselines"
        benchmarks_dir = reports_dir / "benchmarks"
        baselines_dir.mkdir(parents=True)
        benchmarks_dir.mkdir(parents=True)

        (baselines_dir / "linear_probe.json").write_text(
            json.dumps(
                {
                    "test": {
                        "f1_macro": 0.41,
                        "f1_micro": 0.51,
                        "precision_macro": 0.61,
                        "recall_macro": 0.71,
                        "exact_match_ratio": 0.81,
                        "hamming_loss": 0.19,
                    },
                    "test_per_class": {
                        "Nucleus": {
                            "precision": 0.1,
                            "recall": 0.2,
                            "f1": 0.3,
                            "support": 4,
                        }
                    },
                }
            )
        )
        (baselines_dir / "xgboost_baseline.json").write_text(
            json.dumps(
                {
                    "test": {
                        "f1_macro": 0.31,
                        "f1_micro": 0.32,
                        "precision_macro": 0.33,
                        "recall_macro": 0.34,
                        "exact_match_ratio": 0.35,
                        "hamming_loss": 0.36,
                    },
                    "test_per_class": {
                        "Cytoplasm": {
                            "precision": 0.4,
                            "recall": 0.5,
                            "f1": 0.6,
                            "support": 7,
                        }
                    },
                }
            )
        )
        (reports_dir / "evaluation_report.json").write_text(
            json.dumps(
                {
                    "test": {
                        "f1_macro": 0.91,
                        "f1_micro": 0.92,
                        "precision_macro": 0.93,
                        "recall_macro": 0.94,
                        "exact_match_ratio": 0.95,
                        "hamming_loss": 0.05,
                    },
                    "per_class": {
                        "Membrane": {
                            "precision": 0.7,
                            "recall": 0.8,
                            "f1": 0.9,
                            "support": 10,
                        }
                    },
                }
            )
        )
        (benchmarks_dir / "deeploc.json").write_text(
            json.dumps(
                {
                    "reference_type": "ground_truth_labels",
                    "n_samples": 12,
                    "overall": {
                        "f1_macro": 0.55,
                        "f1_micro": 0.56,
                        "precision_macro": 0.57,
                        "recall_macro": 0.58,
                        "exact_match_ratio": 0.59,
                        "hamming_loss": 0.11,
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

        output_path = comparison_report.build_report(cfg)
        text = output_path.read_text()

        assert "Evaluated on **12 sequences** from the DeepLoc 2.0 test distribution." in text
        assert "## Trained model — per-class metrics" in text
        assert "## Linear probe — per-class (test)" in text
        assert "## XGBoost — per-class (test)" in text
        assert "DeepLoc benchmark" in text
        assert "found" in text


class TestMain:
    def test_main_passes_output_and_overrides(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda self: argparse.Namespace(
                output=str(tmp_path / "custom.md"),
                overrides=["project.seed=7"],
            ),
        )

        cfg = DotDict.from_dict({"project": {"log_level": "DEBUG"}})
        monkeypatch.setattr(
            comparison_report,
            "load_config",
            lambda mode, overrides: cfg,
        )

        seen_logging_levels: list[str] = []
        monkeypatch.setattr(
            comparison_report,
            "setup_logging",
            lambda level: seen_logging_levels.append(level),
        )

        seen_calls: list[tuple[DotDict, Path | None]] = []
        monkeypatch.setattr(
            comparison_report,
            "build_report",
            lambda cfg, output_path=None: (
                seen_calls.append((cfg, output_path))
                or Path(output_path or tmp_path / "default.md")
            ),
        )

        comparison_report.main()

        assert seen_logging_levels == ["DEBUG"]
        assert seen_calls == [(cfg, tmp_path / "custom.md")]
