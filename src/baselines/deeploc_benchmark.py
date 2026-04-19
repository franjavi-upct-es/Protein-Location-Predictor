# src/baselines/deeploc_benchmark.py
"""
DeepLoc 2.0 benchmark runner.

Evaluates a trained checkpoint against the DeepLoc 2.0 test set so the
project's numbers can be compared with a published external baseline.

DeepLoc 2.0 is published at https://services.healthtech.dtu.dk/services/DeepLoc-2.0/
The test set is distributed as a FASTA file plus a TSV with the
ground-truth labels. Because the exact distribution URL and license can
change without notice, this script does NOT auto-download the dataset.
Instead it expects the user to place the files manually under
``benchmarks/deeploc/`` and tells them what to put there.

Layout expected by this script:

    benchmarks/
        deeploc/
            test.fasta              # protein sequences
            test_labels.tsv         # accession <TAB> location[|location...]

The location names in the TSV are mapped to the project's internal
class names via a configurable label map (see ``DEFAULT_LABEL_MAP``
below). Anything not in the map is dropped from the evaluation.

Usage::

    > uv run python -m src.baselines.deeploc_benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# Map from DeepLoc 2.0 location names to the project's internal class
# names. Keys are case-insensitive at lookup time. Add new entries here
# if you change the project taxonomy.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "nucleus": "Nucleus",
    "cytoplasm": "Cytoplasm",
    "extracellular": "Secreted/Extracellular",
    "cell membrane": "Membrane",
    "plasma membrane": "Membrane",
    "mitochondrion": "Mitochondrion",
    "endoplasmic reticulum": "Endoplasmic Reticulum",
    "golgi apparatus": "Golgi Apparatus",
    "lysosome/vacuole": "Vacuole",
    "lysosome": "Vacuole",
    "vacuole": "Vacuole",
    "peroxisome": "Peroxisome",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_fasta(path: Path) -> dict[str, str]:
    """Tiny FASTA parser, returns {accession: sequence}."""
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_chunks: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_chunks)
                current_id = line[1:].split()[0]
                current_chunks = []
            else:
                current_chunks.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_chunks)
    return sequences


def _load_deeploc_test_set(benchmarks_dir: Path, label_map: dict[str, str]) -> pd.DataFrame:
    """
    Load the DeepLoc test set from disk and map labels to project classes.

    Returns a DataFrame with columns:
        accession, sequence, locations_str, raw_locations
    """
    fasta_path = benchmarks_dir / "test.fasta"
    labels_path = benchmarks_dir / "test_labels.tsv"

    if not fasta_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "DeepLoc 2.0 test set not found. Place the files manually:\n"
            f"  {fasta_path}\n"
            f"  {labels_path}\n"
            "See https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ "
            "for the dataset distribution."
        )

    sequences = _read_fasta(fasta_path)
    labels_df = pd.read_csv(labels_path, sep="\t", header=None)
    if labels_df.shape[1] < 2:
        raise ValueError(
            f"{labels_path} must have at least 2 tab-separated columns: "
            "accession and location[|location...]"
        )
    labels_df.columns = ["accession"] + [f"col{i}" for i in range(1, labels_df.shape[1])]
    # Use the second column as the labels column
    labels_df["raw_locations"] = labels_df.iloc[:, 1].astype(str)

    # Build the merged DataFrame
    rows = []
    skipped = 0
    lower_map = {k.lower(): v for k, v in label_map.items()}
    for _, row in labels_df.iterrows():
        acc = str(row["accession"])
        if acc not in sequences:
            skipped += 1
            continue
        raw = str(row["raw_locations"])
        mapped: list[str] = []
        for raw_loc in raw.split("|"):
            key = raw_loc.strip().lower()
            if key in lower_map:
                target = lower_map[key]
                if target not in mapped:
                    mapped.append(target)
        if not mapped:
            continue
        rows.append(
            {
                "accession": acc,
                "sequence": sequences[acc],
                "locations_str": "|".join(mapped),
                "raw_locations": raw,
            }
        )

    if skipped:
        logger.warning(
            f"DeepLoc: {skipped} accessions had labels but no sequence in the FASTA file — skipped"
        )
    if not rows:
        raise RuntimeError(
            "DeepLoc test set produced 0 rows after label mapping. "
            "Check that DEFAULT_LABEL_MAP covers your project classes."
        )

    df = pd.DataFrame(rows)
    logger.info(f"DeepLoc test set: {len(df)} rows after label mapping")
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _predict_with_checkpoint(
    df: pd.DataFrame, checkpoint_path: Path, cfg: DotDict
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the trained predictor on every row and return preds, targets, labels."""
    from src.serving.predictor import Predictor

    predictor = Predictor.from_checkpoint(checkpoint_path, cfg)
    label_list = predictor.label_list
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    n = len(df)
    n_classes = len(label_list)
    preds = np.zeros((n, n_classes), dtype=np.int64)
    targets = np.zeros((n, n_classes), dtype=np.int64)

    sequences = df["sequence"].astype(str).tolist()
    label_strings = df["locations_str"].astype(str).tolist()

    # Build target matrix
    for i, label_str in enumerate(label_strings):
        for loc in label_str.split("|"):
            if loc in label_to_idx:
                targets[i, label_to_idx[loc]] = 1

    # Predict
    logger.info(f"Predicting on {n} sequences...")
    batch_results = predictor.predict_batch(sequences)
    for i, results in enumerate(batch_results):
        for r in results:
            label = r["location"]
            if label in label_to_idx:
                preds[i, label_to_idx[label]] = 1

    return preds, targets, label_list


def run_deeploc_benchmark(
    cfg: DotDict,
    checkpoint_path: Path | None = None,
    benchmarks_dir: Path | None = None,
    output_dir: Path | None = None,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a trained checkpoint on the DeepLoc 2.0 test set.

    Returns:
        Dict with overall + per-class metrics on DeepLoc.
    """
    if benchmarks_dir is None:
        benchmarks_dir = Path(cfg.get("project_root", ".")) / "benchmarks" / "deeploc"

    if checkpoint_path is None:
        models_dir = resolve_path(cfg, "paths.models_dir")
        ckpt_dir = models_dir / "checkpoints"
        candidates = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}. Train a model first.")
        checkpoint_path = candidates[0]
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    df = _load_deeploc_test_set(benchmarks_dir, label_map or DEFAULT_LABEL_MAP)

    preds, targets, label_list = _predict_with_checkpoint(df, checkpoint_path, cfg)

    metrics = compute_metrics(preds, targets, label_list)

    summary = {
        "model": "trained_checkpoint",
        "checkpoint": str(checkpoint_path),
        "n_samples": int(len(df)),
        "label_list": label_list,
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
    }

    if output_dir is None:
        output_dir = resolve_path(cfg, "paths.reports_dir") / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "deeploc.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info(f"DeepLoc benchmark report written to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the DeepLoc 2.0 test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .ckpt file. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=str,
        default=None,
        help="Directory containing test.fasta and test_labels.tsv.",
    )
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides.")
    args = parser.parse_args()

    cfg = load_config(mode="training", overrides=args.overrides)
    setup_logging(level=cfg.project.log_level)

    run_deeploc_benchmark(
        cfg,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        benchmarks_dir=(Path(args.benchmarks_dir) if args.benchmarks_dir else None),
    )


if __name__ == "__main__":
    main()
