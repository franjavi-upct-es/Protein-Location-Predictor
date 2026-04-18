# src/evaluation/per_organism.py
"""
Per-organism evaluation breakdown.

The v2.0 model trains on multiple organisms (yeast, human, mouse, ...)
and reports a single F1 across all of them. That single number can hide
a regression on yeast — the dominant organism in v1.0 — which is
exactly the kind of silent quality drop we want to catch.

This module slices an existing prediction set by organism and computes
``compute_metrics`` independently per slice. The output is a dict
suitable for serialization next to the global metrics.

Usage::

    from src.evaluation.per_organism import compute_per_organism_metrics
    breakdown = compute_per_organism_metrics(
        predictions, targets, organism_ids, label_list,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Common organism IDs we want pretty names for in the report
_ORGANISM_NAMES: dict[int, str] = {
    9606: "Homo sapiens",
    10090: "Mus musculus",
    559292: "Saccharomyces cerevisiae",
    10116: "Rattus norvegicus",
    7227: "Drosophila melanogaster",
    6239: "Caenorhabditis elegans",
    3702: "Arabidopsis thaliana",
    83333: "Escherichia coli",
}


def _organism_label(organism_id: Any) -> str:
    try:
        oid = int(organism_id)
    except (TypeError, ValueError):
        return str(organism_id)
    return _ORGANISM_NAMES.get(oid, f"organism_{oid}")


def compute_per_organism_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    organism_ids: list[Any] | np.ndarray,
    label_list: list[str],
    min_samples: int = 5,
) -> dict[str, Any]:
    """
    Slice predictions by organism and compute metrics per slice.

    Args:
        predictions: Binary prediction array of shape (N, C).
        targets: Binary target array of shape (N, C).
        organism_ids: Per-row organism identifier (typically the
            UniProt taxonomy ID, but any hashable works).
        label_list: Ordered list of class names.
        min_samples: Skip organisms with fewer than this many rows.

    Returns:
        Dict keyed by ``organism_label`` with one ``compute_metrics``
        result per organism plus a ``_summary`` key with the row counts.
    """
    if len(predictions) != len(organism_ids):
        raise ValueError(
            f"predictions and organism_ids length mismatch: "
            f"{len(predictions)} vs {len(organism_ids)}"
        )

    organism_array = np.asarray(organism_ids)
    unique_organisms, counts = np.unique(organism_array, return_counts=True)

    breakdown: dict[str, Any] = {}
    summary: dict[str, int] = {}

    for organism, count in zip(unique_organisms, counts, strict=True):
        if count < min_samples:
            logger.info(
                f"Per-organism: skipping {organism} (only {count} samples, "
                f"below min_samples={min_samples})"
            )
            continue

        mask = organism_array == organism
        org_preds = predictions[mask]
        org_targets = targets[mask]

        # Drop classes that are entirely absent in this slice — otherwise
        # F1 metrics divide by zero and look worse than they really are.
        # We keep the full label list in the output for consistency.
        try:
            metrics = compute_metrics(org_preds, org_targets, label_list)
        except Exception as e:
            logger.warning(f"Per-organism: failed to compute metrics for {organism}: {e}")
            continue

        label = _organism_label(organism)
        breakdown[label] = {
            "n_samples": int(count),
            "overall": metrics["overall"],
            "per_class": metrics["per_class"],
        }
        summary[label] = int(count)

    breakdown["_summary"] = {
        "total_samples": int(len(predictions)),
        "n_organisms": len(summary),
        "samples_per_organism": summary,
    }

    return breakdown
