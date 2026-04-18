# src/evaluation/threshold_tuning.py
"""
Per-class probability threshold tuning.

By default the predictor uses a single threshold of 0.5 for every
class, which is the wrong choice on an imbalanced multi-label problem.
For each class, the optimal threshold (in the F1 sense) depends on the
class frequency, the model's calibration, and the specific operating
point you care about.

This module sweeps thresholds in [0.05, 0.95] per class on a held-out
set of probability predictions and persists the best per-class
thresholds as JSON. The predictor reads that JSON at load time if it
exists and uses the per-class thresholds in place of the default.

Usage::

    from src.evaluation.threshold_tuning import tune_thresholds
    thresholds = tune_thresholds(probabilities, targets, label_list)
    save_thresholds(thresholds, "models/checkpoints/thresholds.json")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def tune_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    label_list: list[str],
    n_steps: int = 19,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    fallback: float = 0.5,
) -> dict[str, float]:
    """
    Find the F1-optimal threshold per class.

    Args:
        probabilities: Sigmoid output of shape (N, C).
        targets: Multi-hot target array of shape (N, C).
        label_list: Ordered list of class names of length C.
        n_steps: Number of threshold candidates to evaluate.
        min_threshold: Lowest threshold to try.
        max_threshold: Highest threshold to try.
        fallback: Threshold for classes with zero positives in targets.

    Returns:
        Dict ``{label: best_threshold}``.
    """
    from sklearn.metrics import f1_score

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets shape mismatch: {probabilities.shape} vs {targets.shape}"
        )
    if probabilities.shape[1] != len(label_list):
        raise ValueError(
            f"probabilities has {probabilities.shape[1]} classes but "
            f"label_list has {len(label_list)}"
        )

    candidates = np.linspace(min_threshold, max_threshold, n_steps)
    best_thresholds: dict[str, float] = {}

    for c_idx, label in enumerate(label_list):
        y_true = targets[:, c_idx].astype(int)
        if y_true.sum() == 0:
            logger.info(
                f"Threshold tuning: '{label}' has no positives in val "
                f"set — using fallback {fallback}"
            )
            best_thresholds[label] = float(fallback)
            continue

        y_probs = probabilities[:, c_idx]
        best_f1 = -1.0
        best_t = float(fallback)
        for t in candidates:
            y_pred = (y_probs >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thresholds[label] = best_t
        logger.info(f"Threshold tuning: '{label}' best={best_t:.2f} (F1={best_f1:.3f})")

    return best_thresholds


def save_thresholds(thresholds: dict[str, float], path: Path | str) -> None:
    """Persist a thresholds dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2, sort_keys=True))
    logger.info(f"Per-class thresholds saved to {path}")


def load_thresholds(path: Path | str) -> dict[str, float] | None:
    """Load a thresholds dict from JSON, returning None if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {str(k): float(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Could not parse thresholds at {path}: {e}")
        return None
