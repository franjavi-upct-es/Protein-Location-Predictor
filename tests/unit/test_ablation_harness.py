# tests/unit/test_ablation_harness.py
"""Tests for the Sprint 7 ablation harness."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

from src.evaluation import ablation_harness as harness
from src.utils.config import DotDict


class TestBuildRuns:
    def test_build_runs_contains_expected_tags_and_epoch_override(self) -> None:
        runs = harness._build_runs(epochs=7)

        tags = [run.tag for run in runs]

        assert tags == [
            "baseline",
            "light_attention",
            "multi_task",
            "balanced_sampling",
            "length_bucketing",
            "all_on",
        ]
        assert all("training.max_epochs=7" in run.overrides for run in runs)
        assert all("project.seed=42" in run.overrides for run in runs)
        assert "model.pooling=light_attention" in runs[1].overrides
        assert "multi_task.enabled=true" in runs[2].overrides
        assert "training.use_balanced_sampling=true" in runs[3].overrides


class TestRunOne:
    def test_run_one_success_captures_metrics_from_report(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        cfg = DotDict.from_dict(
            {
                "project_root": str(tmp_path),
                "paths": {
                    "models_dir": "models",
                    "reports_dir": "reports",
                },
            }
        )

        monkeypatch.setattr(harness, "load_config", lambda **kwargs: cfg)

        def fake_train(train_cfg: DotDict) -> None:
            reports_dir = Path(train_cfg["project_root"]) / train_cfg["paths"]["reports_dir"]
            assert reports_dir == tmp_path / "ablations" / "baseline" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "evaluation_report.json").write_text(
                json.dumps({"overall": {"f1_macro": 0.812}})
            )

        fake_module = types.ModuleType("src.training.train")
        fake_module.train = fake_train  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.training.train", fake_module)

        run = harness.AblationRun(
            tag="baseline",
            description="Baseline run",
            overrides=["training.max_epochs=3"],
        )

        record = harness._run_one(run, ["training.batch_size=2"], tmp_path / "ablations")

        assert record["status"] == "ok"
        assert record["metrics"] == {"f1_macro": 0.812}
        assert record["overrides"] == ["training.max_epochs=3", "training.batch_size=2"]
        assert (tmp_path / "ablations" / "baseline" / "models").exists()

    def test_run_one_without_metrics_marks_ok_no_metrics(self, monkeypatch, tmp_path: Path) -> None:
        cfg = DotDict.from_dict(
            {
                "project_root": str(tmp_path),
                "paths": {
                    "models_dir": "models",
                    "reports_dir": "reports",
                },
            }
        )

        monkeypatch.setattr(harness, "load_config", lambda **kwargs: cfg)

        fake_module = types.ModuleType("src.training.train")
        fake_module.train = lambda train_cfg: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.training.train", fake_module)

        run = harness.AblationRun(tag="baseline", description="Baseline run")

        record = harness._run_one(run, [], tmp_path / "ablations")

        assert record["status"] == "ok_no_metrics"
        assert record["metrics"] == {}

    def test_run_one_failure_marks_failed(self, monkeypatch, tmp_path: Path) -> None:
        cfg = DotDict.from_dict(
            {
                "project_root": str(tmp_path),
                "paths": {
                    "models_dir": "models",
                    "reports_dir": "reports",
                },
            }
        )

        monkeypatch.setattr(harness, "load_config", lambda **kwargs: cfg)

        def fake_train(train_cfg: DotDict) -> None:
            raise RuntimeError("training exploded")

        fake_module = types.ModuleType("src.training.train")
        fake_module.train = fake_train  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.training.train", fake_module)

        run = harness.AblationRun(tag="baseline", description="Baseline run")

        record = harness._run_one(run, [], tmp_path / "ablations")

        assert record["status"] == "failed"
        assert "training exploded" in record["error"]


class TestBuildReport:
    def test_build_report_writes_markdown_table_and_errors(self, tmp_path: Path) -> None:
        output_path = tmp_path / "ablation.md"

        harness._build_report(
            [
                {
                    "tag": "baseline",
                    "description": "Baseline run",
                    "metrics": {"f1_macro": 0.5},
                    "elapsed_seconds": 1.2,
                    "status": "ok",
                    "overrides": ["training.max_epochs=3"],
                },
                {
                    "tag": "all_on",
                    "description": "Everything enabled",
                    "metrics": {},
                    "elapsed_seconds": 2.5,
                    "status": "failed",
                    "error": "boom",
                    "overrides": ["multi_task.enabled=true"],
                },
            ],
            output_path,
        )

        text = output_path.read_text()

        assert "# Sprint 7 — Ablation report" in text
        assert "| baseline | Baseline run | 0.500 |" in text
        assert "#### `all_on` — failed" in text
        assert "**Error:** boom" in text
        assert "multi_task.enabled=true" in text


class TestMain:
    def test_main_filters_runs_cleans_output_and_persists_records(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        reports_dir = tmp_path / "reports"
        output_dir = reports_dir / "ablations"
        output_dir.mkdir(parents=True)
        (output_dir / "stale.txt").write_text("old")

        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda self: argparse.Namespace(
                epochs=5,
                only="baseline",
                output_dir=None,
                clean=True,
            ),
        )

        cfg = DotDict.from_dict(
            {
                "project": {"log_level": "INFO"},
                "project_root": str(tmp_path),
            }
        )
        monkeypatch.setattr(harness, "load_config", lambda mode="training": cfg)
        monkeypatch.setattr(harness, "setup_logging", lambda level: None)
        monkeypatch.setattr(harness, "resolve_path", lambda cfg, key: reports_dir)
        monkeypatch.setattr(
            harness,
            "_build_runs",
            lambda epochs: [
                harness.AblationRun("baseline", "Baseline"),
                harness.AblationRun("all_on", "All on"),
            ],
        )

        seen_tags: list[str] = []

        def fake_run_one(
            run: harness.AblationRun, extra: list[str], out: Path
        ) -> dict[str, object]:
            seen_tags.append(run.tag)
            assert out == output_dir
            return {
                "tag": run.tag,
                "description": run.description,
                "status": "ok",
                "metrics": {"f1_macro": 0.7},
            }

        built_reports: list[tuple[list[dict[str, object]], Path]] = []

        monkeypatch.setattr(harness, "_run_one", fake_run_one)
        monkeypatch.setattr(
            harness,
            "_build_report",
            lambda records, path: built_reports.append((records, path)),
        )

        harness.main()

        assert seen_tags == ["baseline"]
        assert json.loads((output_dir / "ablation_records.json").read_text()) == [
            {
                "tag": "baseline",
                "description": "Baseline",
                "status": "ok",
                "metrics": {"f1_macro": 0.7},
            }
        ]
        assert built_reports == [
            (
                [
                    {
                        "tag": "baseline",
                        "description": "Baseline",
                        "status": "ok",
                        "metrics": {"f1_macro": 0.7},
                    }
                ],
                output_dir / "sprint-7-ablation.md",
            )
        ]
