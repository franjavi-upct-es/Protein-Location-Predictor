# src/evaluation/comparison_report.py
"""
Generate the Sprint 6 comparison report.

Reads every JSON metrics file produced by:
  - the linear probe baseline (`reports/baselines/linear_probe.json`)
  - the XGBoost baseline (`reports/baselines/xgboost_baseline.json`)
  - the DeepLoc benchmark (`reports/benchmarks/deeploc.json`)
  - the trained model evaluation, if produced separately
    (`reports/evaluation_report.json`)

…and emits a single Markdown report at
``reports/sprint-6-comparison.md`` that compares them side-by-side on
the headline metrics. This is the artifact that closes the sprint.

Missing files are reported as "not available" rather than skipped, so
the report makes it obvious what is still pending.

Usage::

    > uv run python -m src.evaluation.comparison_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


_HEADLINE_METRICS = (
    "f1_macro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "exact_match_ratio",
    "hamming_loss",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {path}: {e}")
        return None

    if not isinstance(payload, dict):
        logger.warning(f"Expected a JSON object in {path}, got {type(payload).__name__}")
        return None

    return cast(dict[str, Any], payload)


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _row_for(name: str, payload: dict[str, Any] | None, split: str) -> tuple[str, list[str]]:
    """Build one Markdown table row for a model on a given split."""
    if payload is None:
        return name, ["not available"] * len(_HEADLINE_METRICS)
    block = payload.get(split) or payload.get("overall") or {}
    return name, [_format_value(block.get(m)) for m in _HEADLINE_METRICS]


def build_report(cfg: DotDict, output_path: Path | None = None) -> Path:
    reports_dir = resolve_path(cfg, "paths.reports_dir")
    baselines_dir = reports_dir / "baselines"
    benchmarks_dir = reports_dir / "benchmarks"

    linear = _load_json(baselines_dir / "linear_probe.json")
    xgboost = _load_json(baselines_dir / "xgboost_baseline.json")
    deeploc = _load_json(benchmarks_dir / "deeploc.json")
    trained = _load_json(reports_dir / "evaluation_report.json")

    if output_path is None:
        output_path = reports_dir / "sprint-6-comparison.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Sprint 6 — Comparison report")
    lines.append("")
    lines.append(
        "Headline metrics across baselines, the trained model, and the "
        "external DeepLoc 2.0 benchmark."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Test-set comparison
    # ------------------------------------------------------------------
    lines.append("## Test-set metrics")
    lines.append("")
    header = "| Model | " + " | ".join(_HEADLINE_METRICS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(_HEADLINE_METRICS)) + "|"
    lines.append(header)
    lines.append(sep)

    for name, payload, split in (
        ("Linear probe (frozen ESM-2)", linear, "test"),
        ("XGBoost (frozen ESM-2, v1.0 replica)", xgboost, "test"),
        ("Trained model (this project)", trained, "test"),
    ):
        label, values = _row_for(name, payload, split)
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append("")

    # ------------------------------------------------------------------
    # External benchmark
    # ------------------------------------------------------------------
    lines.append("## External benchmark — DeepLoc 2.0")
    lines.append("")
    if deeploc is None:
        lines.append(
            "_DeepLoc benchmark not run yet. Place the test set under_ "
            "`benchmarks/deeploc/` _and run_ "
            "`uv run python -m src.baselines.deeploc_benchmark`."
        )
    else:
        n_samples = deeploc.get("n_samples", "?")
        reference_type = deeploc.get("reference_type", "ground_truth_labels")
        if reference_type == "deeploc_predictions":
            total = deeploc.get("n_source_rows", n_samples)
            lines.append(
                "Evaluated on "
                f"**{n_samples} of {total} sequences** from the packaged DeepLoc 2.0 "
                "demo set, using DeepLoc's own predictions as the reference labels."
            )
            skipped = deeploc.get("n_skipped_unmapped_labels", 0)
            if skipped:
                lines.append(
                    f"{skipped} sequences were dropped because their DeepLoc labels do "
                    "not exist in this project's taxonomy."
                )
        else:
            lines.append(
                f"Evaluated on **{n_samples} sequences** from the DeepLoc 2.0 test distribution."
            )
        lines.append("")
        lines.append(header)
        lines.append(sep)
        label, values = _row_for("Trained model on DeepLoc test set", deeploc, "overall")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Per-class breakdown of the trained model
    # ------------------------------------------------------------------
    if trained and "per_class" in trained:
        lines.append("## Trained model — per-class metrics")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, m in trained["per_class"].items():
            lines.append(
                f"| {cls} | "
                f"{_format_value(m.get('precision'))} | "
                f"{_format_value(m.get('recall'))} | "
                f"{_format_value(m.get('f1'))} | "
                f"{_format_value(m.get('support'))} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Baselines per-class
    # ------------------------------------------------------------------
    for source_name, source in (
        ("Linear probe", linear),
        ("XGBoost", xgboost),
    ):
        if source and "test_per_class" in source:
            lines.append(f"## {source_name} — per-class (test)")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|---|---:|---:|---:|---:|")
            for cls, m in source["test_per_class"].items():
                lines.append(
                    f"| {cls} | "
                    f"{_format_value(m.get('precision'))} | "
                    f"{_format_value(m.get('recall'))} | "
                    f"{_format_value(m.get('f1'))} | "
                    f"{_format_value(m.get('support'))} |"
                )
            lines.append("")

    # ------------------------------------------------------------------
    # Provenance footer
    # ------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("### Files consumed")
    lines.append("")
    for label, path in (
        ("Linear probe", baselines_dir / "linear_probe.json"),
        ("XGBoost", baselines_dir / "xgboost_baseline.json"),
        ("Trained model", reports_dir / "evaluation_report.json"),
        ("DeepLoc benchmark", benchmarks_dir / "deeploc.json"),
    ):
        status = "found" if path.exists() else "missing"
        lines.append(f"- {label}: `{path}` ({status})")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Comparison report written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Sprint 6 comparison report from JSON inputs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Markdown path. Defaults to reports/sprint-6-comparison.md.",
    )
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides.")
    args = parser.parse_args()

    cfg = load_config(mode="training", overrides=args.overrides)
    setup_logging(level=cfg.project.log_level)

    build_report(cfg, output_path=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
