# src/baselines/deeploc_benchmark.py
"""
DeepLoc 2.0 benchmark runner.

Evaluates a trained checkpoint against a DeepLoc 2.0 reference set.

DeepLoc 2.0 is published at https://services.healthtech.dtu.dk/services/DeepLoc-2.0/
Two input layouts are supported:

1. A true benchmark layout with a FASTA file plus a TSV containing
   ground-truth labels.
2. The packaged DeepLoc 2.0 demo layout, where ``test.fasta`` is paired
   with ``outputs/results_test.csv``. In that mode the script compares
   this project's predictions against DeepLoc's own predictions instead
   of ground truth, which is useful as a compatibility check but is not
   a published benchmark.

Because the exact distribution URL and license can change without
notice, this script does NOT auto-download the dataset. Instead it
expects the user to place the files manually under ``benchmarks/deeploc/``
or point ``--benchmarks-dir`` at the unpacked DeepLoc package.

Supported layouts:

    benchmarks/
        deeploc/
            test.fasta                    # protein sequences
            test_labels.tsv               # accession <TAB> location[|location...]

    deeploc2_package/
        test.fasta
        outputs/
            results_test.csv             # DeepLoc's own predictions

DeepLoc location names are mapped to the project's internal class names
via a configurable label map (see ``DEFAULT_LABEL_MAP`` below). Any
labels with no overlap in this project's taxonomy are dropped.

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


def _location_tokens(raw_locations: str) -> list[str]:
    """Split a raw DeepLoc label string into normalized tokens."""
    return [token.strip() for token in raw_locations.split("|") if token.strip()]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _record_id_aliases(record_id: str) -> list[str]:
    """Return the canonical record id plus common aliases like UniProt accessions."""
    aliases = [record_id]
    parts = record_id.split("|")
    if len(parts) >= 3 and parts[1]:
        aliases.append(parts[1])
    return aliases


def _read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    """Tiny FASTA parser, returns {alias: (canonical_id, sequence)}."""
    sequence_lookup: dict[str, tuple[str, str]] = {}
    current_id: str | None = None
    current_chunks: list[str] = []

    def _store_record() -> None:
        if current_id is None:
            return
        sequence = "".join(current_chunks)
        for alias in _record_id_aliases(current_id):
            sequence_lookup.setdefault(alias, (current_id, sequence))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _store_record()
                current_id = line[1:].split()[0]
                current_chunks = []
            else:
                current_chunks.append(line)
    _store_record()
    return sequence_lookup


def _map_locations(raw_locations: str, label_map: dict[str, str]) -> list[str]:
    """Map DeepLoc label names to this project's internal taxonomy."""
    lower_map = {key.lower(): value for key, value in label_map.items()}
    mapped: list[str] = []
    for raw_loc in _location_tokens(raw_locations):
        target = lower_map.get(raw_loc.lower())
        if target is not None and target not in mapped:
            mapped.append(target)
    return mapped


def _build_reference_frame(
    rows: list[tuple[str, str]],
    sequence_lookup: dict[str, tuple[str, str]],
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the evaluation frame from raw DeepLoc ids plus label strings."""
    out_rows: list[dict[str, str]] = []
    skipped_missing_sequence = 0
    skipped_unmapped_labels = 0
    unmapped_tokens: set[str] = set()

    for protein_id, raw_locations in rows:
        lookup = sequence_lookup.get(str(protein_id))
        if lookup is None:
            skipped_missing_sequence += 1
            continue

        canonical_id, sequence = lookup
        mapped = _map_locations(str(raw_locations), label_map)
        if not mapped:
            skipped_unmapped_labels += 1
            unmapped_tokens.update(_location_tokens(str(raw_locations)))
            continue

        out_rows.append(
            {
                "accession": canonical_id,
                "sequence": sequence,
                "locations_str": "|".join(mapped),
                "raw_locations": str(raw_locations),
            }
        )

    stats = {
        "n_source_rows": len(rows),
        "n_skipped_missing_sequence": skipped_missing_sequence,
        "n_skipped_unmapped_labels": skipped_unmapped_labels,
        "unmapped_labels": sorted(unmapped_tokens),
    }
    return pd.DataFrame(out_rows), stats


def _read_reference_rows_from_tsv(labels_path: Path) -> list[tuple[str, str]]:
    """Read ground-truth DeepLoc labels from the benchmark TSV."""
    labels_df = pd.read_csv(labels_path, sep="\t", header=None)
    if labels_df.shape[1] < 2:
        raise ValueError(
            f"{labels_path} must have at least 2 tab-separated columns: "
            "accession and location[|location...]"
        )
    return [(str(row.iloc[0]), str(row.iloc[1])) for _, row in labels_df.iterrows()]


def _read_reference_rows_from_results_csv(results_path: Path) -> list[tuple[str, str]]:
    """Read reference labels from DeepLoc's packaged demo predictions."""
    results_df = pd.read_csv(results_path)
    required = {"Protein_ID", "Localizations"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"{results_path} is missing required columns: {sorted(missing)}")
    return list(
        zip(
            results_df["Protein_ID"].astype(str).tolist(),
            results_df["Localizations"].fillna("").astype(str).tolist(),
            strict=True,
        )
    )


def _supported_reference_paths(benchmarks_dir: Path) -> tuple[Path, Path, Path]:
    """Return the supported DeepLoc reference file locations."""
    return (
        benchmarks_dir / "test.fasta",
        benchmarks_dir / "test_labels.tsv",
        benchmarks_dir / "outputs" / "results_test.csv",
    )


def _load_deeploc_test_set(
    benchmarks_dir: Path, label_map: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a DeepLoc reference set from disk and map labels to project classes.

    Returns a DataFrame with columns:
        accession, sequence, locations_str, raw_locations
    """
    fasta_path, labels_path, results_path = _supported_reference_paths(benchmarks_dir)

    if not fasta_path.exists():
        raise FileNotFoundError(
            "DeepLoc FASTA file not found. Supported layouts are:\n"
            f"  {fasta_path} with {labels_path}\n"
            f"  {fasta_path} with {results_path}\n"
            "If you are using the packaged DeepLoc 2.0 demo, point "
            "--benchmarks-dir at the unpacked deeploc2_package directory."
        )

    sequence_lookup = _read_fasta(fasta_path)

    reference_rows: list[tuple[str, str]] | None = None
    reference_type: str | None = None
    source_path: Path | None = None

    if labels_path.exists():
        tsv_rows = _read_reference_rows_from_tsv(labels_path)
        if any(_map_locations(raw_locations, label_map) for _, raw_locations in tsv_rows):
            reference_rows = tsv_rows
            reference_type = "ground_truth_labels"
            source_path = labels_path
        elif results_path.exists():
            logger.warning(
                f"{labels_path} does not contain recognizable DeepLoc location labels; "
                f"falling back to {results_path}."
            )
        else:
            sample_values = ", ".join(repr(raw) for _, raw in tsv_rows[:3]) or "<empty>"
            raise RuntimeError(
                "DeepLoc test_labels.tsv did not contain recognizable location labels. "
                f"Sample values from the second column: {sample_values}. "
                "If you copied metadata from FASTA descriptions, replace it with true "
                "labels, or use the packaged DeepLoc demo layout with outputs/results_test.csv."
            )

    if reference_rows is None and results_path.exists():
        reference_rows = _read_reference_rows_from_results_csv(results_path)
        reference_type = "deeploc_predictions"
        source_path = results_path

    if reference_rows is None or reference_type is None or source_path is None:
        raise FileNotFoundError(
            "DeepLoc reference files not found. Supported layouts are:\n"
            f"  {fasta_path} with {labels_path}\n"
            f"  {fasta_path} with {results_path}\n"
            "The packaged DeepLoc demo ships with predictions only, not ground-truth labels."
        )

    df, stats = _build_reference_frame(reference_rows, sequence_lookup, label_map)

    if stats["n_skipped_missing_sequence"]:
        logger.warning(
            "DeepLoc: %s ids had labels but no matching sequence in the FASTA file — skipped",
            stats["n_skipped_missing_sequence"],
        )
    if stats["n_skipped_unmapped_labels"]:
        logger.warning(
            "DeepLoc: %s rows were dropped because their labels do not overlap "
            "with this project's taxonomy%s",
            stats["n_skipped_unmapped_labels"],
            (f" ({', '.join(stats['unmapped_labels'][:5])})" if stats["unmapped_labels"] else ""),
        )
    if df.empty:
        raise RuntimeError(
            "DeepLoc reference set produced 0 evaluable rows after label mapping. "
            "Check that DEFAULT_LABEL_MAP overlaps the labels in your chosen reference file."
        )

    metadata = {
        "reference_type": reference_type,
        "reference_source": str(source_path),
        **stats,
    }
    logger.info(
        "DeepLoc reference set: %s evaluable rows from %s (%s)",
        len(df),
        source_path,
        reference_type,
    )
    return df, metadata


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

    df, metadata = _load_deeploc_test_set(benchmarks_dir, label_map or DEFAULT_LABEL_MAP)

    preds, targets, label_list = _predict_with_checkpoint(df, checkpoint_path, cfg)

    metrics = compute_metrics(preds, targets, label_list)

    summary = {
        "model": "trained_checkpoint",
        "checkpoint": str(checkpoint_path),
        "n_samples": int(len(df)),
        "reference_type": metadata["reference_type"],
        "reference_source": metadata["reference_source"],
        "n_source_rows": metadata["n_source_rows"],
        "n_skipped_missing_sequence": metadata["n_skipped_missing_sequence"],
        "n_skipped_unmapped_labels": metadata["n_skipped_unmapped_labels"],
        "unmapped_labels": metadata["unmapped_labels"],
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
        help=(
            "Directory containing either test.fasta + test_labels.tsv or "
            "the packaged DeepLoc demo layout test.fasta + outputs/results_test.csv."
        ),
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
