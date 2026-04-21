# src/evaluation/calibration.py
"""
Per-class temperature scaling for probability calibration.

Multi-label classifiers trained with focal loss tend to be poorly
calibrated: a predicted probability of 0.7 doesn't necessarily mean
the model is right 70% of the time. This is because focal loss
deliberately distorts the loss landscape to focus on hard examples,
which biases the resulting probabilities away from being honest
frequencies.

Temperature scaling is the standard fix:

    p_calibrated = sigmoid(logits / T)

For each class we learn a single scalar temperature T_c on a held-out
validation set by minimizing the binary cross-entropy. Temperatures
are stored as a list ``[T_0, T_1, ..., T_{C-1}]`` and persisted as
JSON next to the checkpoint.

The Predictor loads the temperatures at startup if they exist and
divides logits before applying sigmoid. The choice of threshold (also
loaded from disk) is then applied on top of calibrated probabilities.

Usage::

    from src.evaluation.calibration import (
        fit_temperatures,
        save_temperatures,
        apply_temperatures,
    )
    Ts = fit_temperatures(val_logits, val_targets)
    save_temperatures(Ts, "models/checkpoints/temperatures.json")
    calibrated_probs = apply_temperatures(test_logits, Ts)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _bce_loss(
    temperature: float,
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Binary cross-entropy on logits scaled by 1/temperature."""
    if temperature <= 0:
        return float("inf")
    scaled = logits / temperature
    # Numerically stable BCE
    log1p_exp = np.logaddexp(0.0, -np.abs(scaled))
    bce = np.where(scaled >= 0, log1p_exp, -scaled + log1p_exp)
    bce = bce + (1.0 - targets) * scaled
    return float(bce.mean())


def fit_temperatures(
    logits: np.ndarray,
    targets: np.ndarray,
    label_list: list[str] | None = None,
    grid: tuple[float, ...] = (
        0.5,
        0.7,
        0.85,
        1.0,
        1.15,
        1.3,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
    ),
    refine_steps: int = 30,
    fallback: float = 1.0,
) -> list[float]:
    """
    Fit one temperature per class to minimize BCE on the validation set.

    Uses a coarse grid search followed by a golden-section refinement
    around the best grid point. This avoids the dependency on scipy and
    is fast enough for the typical 10–20 class case.

    Args:
        logits: Raw logits of shape ``(N, C)``.
        targets: Multi-hot targets of shape ``(N, C)``.
        label_list: Optional class names for nicer logging.
        grid: Initial coarse temperature grid.
        refine_steps: Number of golden-section iterations after the grid.
        fallback: Temperature returned for classes with no positives.

    Returns:
        List of length ``C`` with the per-class temperatures.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets shape mismatch: {logits.shape} vs {targets.shape}")

    n_samples, n_classes = logits.shape
    temperatures: list[float] = []
    grid_arr = np.asarray(grid, dtype=np.float64)
    phi = (1 + 5**0.5) / 2  # golden ratio

    for c in range(n_classes):
        y = targets[:, c].astype(np.float64)
        if y.sum() == 0:
            temperatures.append(float(fallback))
            continue

        z = logits[:, c].astype(np.float64)

        # Coarse grid
        losses = np.array([_bce_loss(t, z, y) for t in grid_arr])
        best_idx = int(np.argmin(losses))
        best_t = float(grid_arr[best_idx])

        # Bracket around the best grid point
        lo = float(grid_arr[max(0, best_idx - 1)])
        hi = float(grid_arr[min(len(grid_arr) - 1, best_idx + 1)])
        if lo == hi:
            temperatures.append(best_t)
            continue

        # Golden-section search for refinement
        a, b = lo, hi
        for _ in range(refine_steps):
            d = (b - a) / phi
            x1, x2 = b - d, a + d
            if _bce_loss(x1, z, y) < _bce_loss(x2, z, y):
                b = x2
            else:
                a = x1
        best_t = float((a + b) / 2)

        temperatures.append(best_t)
        if label_list is not None:
            logger.info(f"Calibration: '{label_list[c]}' temperature = {best_t:.3f}")

    return temperatures


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_temperatures(logits: np.ndarray, temperatures: list[float]) -> np.ndarray:
    """
    Apply per-class temperature scaling and return sigmoid probabilities.

    Args:
        logits: Raw logits of shape ``(N, C)``.
        temperatures: List of length ``C``.

    Returns:
        Calibrated probability array of shape ``(N, C)``.
    """
    if logits.shape[1] != len(temperatures):
        raise ValueError(
            f"logits has {logits.shape[1]} classes but temperatures has {len(temperatures)}"
        )
    t = np.asarray(temperatures, dtype=np.float64)
    scaled = logits / t[np.newaxis, :]
    # Numerically stable sigmoid
    return 1.0 / (1.0 + np.exp(-scaled))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_temperatures(temperatures: list[float], path: Path | str) -> None:
    """Persist a list of per-class temperatures to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"temperatures": [float(t) for t in temperatures]}
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Calibration temperatures ({len(temperatures)}) saved to {path}")


def load_temperatures(path: Path | str) -> list[float] | None:
    """Load temperatures from JSON, returning None if missing or invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return [float(t) for t in data["temperatures"]]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Could not parse temperatures at {path}: {e}")
        return None
