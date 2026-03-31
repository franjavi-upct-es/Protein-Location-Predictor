# tests/unit/test_reproducibility.py
"""Tests for reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np

from src.utils.reproducibility import seed_everything


class TestSeedEverything:
    """Tests for the seed_everything function."""

    def test_python_random_deterministic(self) -> None:
        seed_everything(42)
        a = [random.random() for _ in range(10)]
        seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_deterministic(self) -> None:
        seed_everything(42)
        a = np.random.rand(10).tolist()
        seed_everything(42)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_different_seeds_give_different_results(self) -> None:
        seed_everything(42)
        a = random.random()
        seed_everything(123)
        b = random.random()
        assert a != b
