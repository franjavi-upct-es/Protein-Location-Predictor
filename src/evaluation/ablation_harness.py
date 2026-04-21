# src/evaluation/ablation_harness.py
"""
Sprint 7 ablation harness.

Runs the trainer with several configurations that toggle one Sprint 7
component each, then summarises the results in a single Markdown table
written to ``reports/ablations/sprint-7-ablation.md``.

Each configuration is identified by a short tag and is materialized as
a list of ``--overrides`` strings. The script invokes
``src.training.train.train`` programmatically for each tag, captures
the test-set metrics from the resulting ``reports/evaluation_report.json``
(or directly from the trainer's test step), and writes the comparison
table at the end.

The harness is designed to fail soft: if a single configuration crashes
its row is marked "FAILED" in the report, but the rest of the runs
continue. This makes the harness useful for overnight runs where a
component may fail on the user's hardware (e.g. QLoRA on a CPU-only
host).

By default the harness uses a reduced training schedule
(``max_epochs=3``) so the full sweep finishes in a reasonable time.
Override with ``--epochs N`` for a more thorough comparison.

Usage:
    uv run python -m src.evaluation.ablation_harness
    uv run python -m src.evaluation.ablation_harness --epochs 10
    uv run python -m src.evaluation.ablation_harness --only light_attention,multi_task
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------


@dataclass
class AblationRun:
    """A single ablation configuration."""

    tag: str
    description: str
    overrides: list[str] = field(default_factory=list)


def _build_runs(epochs: int) -> list[AblationRun]:
    """The default ablation matrix for Sprint 7."""
    base_overrides = [
        f"training.max_epochs={epochs}",
        # Make every run reproducible at the same seed
        "project.seed=42",
    ]

    return [
        AblationRun(
            tag="baseline",
            description="Default v2.0 config (mean pooling, no aux, no balance)",
            overrides=base_overrides,
        ),
        AblationRun(
            tag="light_attention",
            description="Light attention pooling enabled",
            overrides=base_overrides + ["model.pooling=light_attention"],
        ),
        AblationRun(
            tag="multi_task",
            description="Auxiliary multi-task heads enabled",
            overrides=base_overrides + ["multi_task.enabled=true"],
        ),
        AblationRun(
            tag="balanced_sampling",
            description="Balanced multi-label sampler enabled",
            overrides=base_overrides + ["training.use_balanced_sampling=true"],
        ),
        AblationRun(
            tag="length_bucketing",
            description="Length bucketing sampler enabled",
            overrides=base_overrides + ["training.use_length_bucketing=true"],
        ),
        AblationRun(
            tag="all_on",
            description="All Sprint 7 components enabled together",
            overrides=base_overrides
            + [
                "model.pooling=light_attention",
                "multi_task.enabled=true",
                "training.use_balanced_sampling=true",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def _run_one(
    run: AblationRun,
    base_overrides_extra: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Execute a single ablation run, capturing its outputs."""
    from src.training.train import train

    logger.info("=" * 70)
    logger.info(f"Ablation run: {run.tag}")
    logger.info(f"Description:  {run.description}")
    logger.info(f"Overrides:    {run.overrides + base_overrides_extra}")
    logger.info("=" * 70)

    start = time.perf_counter()
    record: dict[str, Any] = {
        "tag": run.tag,
        "description": run.description,
        "overrides": run.overrides + base_overrides_extra,
    }

    try:
        cfg = load_config(
            mode="training",
            overrides=run.overrides + base_overrides_extra,
            validate=True,
        )

        # Redirect outputs to a per-run subdirectory so they don't
        # overwrite each other
        run_models_dir = output_dir / run.tag / "models"
        run_reports_dir = output_dir / run.tag / "reports"
        run_models_dir.mkdir(parents=True, exist_ok=True)
        run_reports_dir.mkdir(parents=True, exist_ok=True)

        # Patch the resolved cfg in place to point at the per-run dirs
        cfg["paths"]["models_dir"] = str(run_models_dir.relative_to(cfg["project_root"]))
        cfg["paths"]["reports_dir"] = str(run_reports_dir.relative_to(cfg["project_root"]))

        train(cfg)

        # Capture metrics from the per-run reports directory
        report_path = run_reports_dir / "evaluation_report.json"
        if report_path.exists():
            metrics = json.loads(report_path.read_text())
            record["metrics"] = metrics.get("overall") or metrics
            record["status"] = "ok"
        else:
            # Fall back to whatever Lightning logged
            record["metrics"] = {}
            record["status"] = "ok_no_metrics"

    except Exception as e:
        logger.error(f"Ablation run '{run.tag}' failed: {e}")
        record["status"] = "failed"
        record["error"] = str(e)

    record["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    return record


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_HEADLINE_METRICS = (
    "f1_macro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "exact_match_ratio",
)


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_report(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write the ablation table as Markdown."""
    lines: list[str] = []
    lines.append("# Sprint 7 — Ablation report")
    lines.append("")
    lines.append(
        "Each row is a single training run with one Sprint 7 component "
        "toggled on or off. The first row is the baseline."
    )
    lines.append("")

    header = "| Tag | Description | " + " | ".join(_HEADLINE_METRICS) + " | Time (s) | Status |"
    sep = "|---|---|" + "|".join(["---:"] * len(_HEADLINE_METRICS)) + "|---:|---|"
    lines.append(header)
    lines.append(sep)

    for r in records:
        metrics = r.get("metrics", {}) or {}
        row_values = [
            r["tag"],
            r["description"],
            *[_format_value(metrics.get(m)) for m in _HEADLINE_METRICS],
            str(r.get("elapsed_seconds", "—")),
            r.get("status", "?"),
        ]
        lines.append("| " + " | ".join(row_values) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Run details")
    lines.append("")
    for r in records:
        lines.append(f"#### `{r['tag']}` — {r.get('status', '?')}")
        lines.append("")
        lines.append("```")
        for ov in r.get("overrides", []):
            lines.append(ov)
        lines.append("```")
        if r.get("status") == "failed":
            lines.append(f"**Error:** {r.get('error', '?')}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Ablation report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sprint 7 ablation harness.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per ablation run (default: 3).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of tags to run (e.g. 'baseline,multi_task').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-run outputs. Defaults to reports/ablations/.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before starting.",
    )
    args = parser.parse_args()

    cfg = load_config(mode="training")
    setup_logging(level=cfg.project.log_level)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_path(cfg, "paths.reports_dir") / "ablations"
    )
    if args.clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _build_runs(args.epochs)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        runs = [r for r in runs if r.tag in wanted]
        logger.info(f"Filtered runs: {[r.tag for r in runs]}")

    records: list[dict[str, Any]] = []
    for run in runs:
        record = _run_one(run, [], output_dir)
        records.append(record)
        # Persist incrementally so partial results survive a crash
        (output_dir / "ablation_records.json").write_text(
            json.dumps(records, indent=2, default=str)
        )

    _build_report(records, output_dir / "sprint-7-ablation.md")


if __name__ == "__main__":
    main()
