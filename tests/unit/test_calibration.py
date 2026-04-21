# tests/unit/test_calibration.py
"""Tests for the temperature scaling calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.calibration import (
    apply_temperatures,
    fit_temperatures,
    load_temperatures,
    save_temperatures,
)

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestFitTemperatures:
    def test_returns_one_per_class(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((50, 4))
        targets = (rng.uniform(size=(50, 4)) > 0.5).astype(int)
        Ts = fit_temperatures(logits, targets)
        assert len(Ts) == 4
        assert all(t > 0 for t in Ts)

    def test_calibrated_logits_lower_bce(self) -> None:
        """The fitted temperature should give lower BCE than T=1."""
        rng = np.random.default_rng(0)
        n = 200
        # Construct overconfident logits: positives at +5, negatives at -5
        targets = (rng.uniform(size=(n, 1)) > 0.5).astype(int)
        logits = np.where(targets == 1, 5.0, -5.0)
        # Add noise so the perfect logits become slightly miscalibrated
        logits = logits + rng.normal(scale=2.0, size=logits.shape)

        Ts = fit_temperatures(logits, targets)

        from src.evaluation.calibration import _bce_loss

        loss_default = _bce_loss(1.0, logits[:, 0], targets[:, 0])
        loss_fitted = _bce_loss(Ts[0], logits[:, 0], targets[:, 0])
        assert loss_fitted <= loss_default + 1e-6

    def test_class_with_no_positives_uses_fallback(self) -> None:
        logits = np.random.randn(50, 3)
        targets = np.zeros((50, 3), dtype=int)
        targets[:25, 0] = 1  # only class 0 has positives
        Ts = fit_temperatures(logits, targets, fallback=1.0)
        assert Ts[1] == 1.0
        assert Ts[2] == 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            fit_temperatures(np.zeros((10, 3)), np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class TestApplyTemperatures:
    def test_temperature_one_equals_sigmoid(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((20, 3))
        probs = apply_temperatures(logits, [1.0, 1.0, 1.0])
        expected = 1.0 / (1.0 + np.exp(-logits))
        assert np.allclose(probs, expected)

    def test_higher_temperature_softens_predictions(self) -> None:
        logits = np.array([[5.0, -5.0]])
        cold = apply_temperatures(logits, [1.0, 1.0])
        warm = apply_temperatures(logits, [3.0, 3.0])
        # Warmer temperature pulls probabilities toward 0.5
        assert abs(warm[0, 0] - 0.5) < abs(cold[0, 0] - 0.5)
        assert abs(warm[0, 1] - 0.5) < abs(cold[0, 1] - 0.5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="classes"):
            apply_temperatures(np.zeros((4, 3)), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        Ts = [0.8, 1.2, 1.5]
        path = tmp_path / "temperatures.json"
        save_temperatures(Ts, path)
        loaded = load_temperatures(path)
        assert loaded == Ts

    def test_load_missing(self, tmp_path: Path) -> None:
        assert load_temperatures(tmp_path / "nope.json") is None

    def test_load_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("not json")
        assert load_temperatures(path) is None
