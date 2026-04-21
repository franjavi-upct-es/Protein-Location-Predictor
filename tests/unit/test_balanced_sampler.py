# tests/unit/test_balanced_sampler.py
"""Tests for BalancedMultilabelBatchSampler."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.samplers import BalancedMultilabelBatchSampler


@pytest.fixture()
def imbalanced_labels() -> np.ndarray:
    """
    100 samples, 4 classes.
    Class 0: 50 positives (common)
    Class 1: 30 positives
    Class 2: 15 positives
    Class 3: 5 positives  (rare)
    """
    rng = np.random.default_rng(0)
    labels = np.zeros((100, 4), dtype=np.int8)
    labels[:50, 0] = 1
    labels[:30, 1] = 1
    labels[:15, 2] = 1
    labels[:5, 3] = 1
    # Shuffle so positives are scattered
    perm = rng.permutation(100)
    return labels[perm]


class TestConstruction:
    def test_basic(self, imbalanced_labels: np.ndarray) -> None:
        s = BalancedMultilabelBatchSampler(labels=imbalanced_labels, batch_size=8)
        assert len(s) == 100 // 8

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            BalancedMultilabelBatchSampler(labels=np.zeros((10, 2)), batch_size=0)

    def test_invalid_label_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            BalancedMultilabelBatchSampler(labels=np.zeros((10,)), batch_size=2)

    def test_rare_class_detection(self, imbalanced_labels: np.ndarray) -> None:
        s = BalancedMultilabelBatchSampler(
            labels=imbalanced_labels,
            batch_size=8,
            rare_class_threshold=0.5,
        )
        # Average is 100/4 = 25, threshold = 12.5
        # Classes with < 12.5 positives: class 2 (15? no, 15 > 12.5),
        # class 3 (5 < 12.5). Only class 3 should be rare.
        assert 3 in s.rare_classes
        assert 0 not in s.rare_classes
        assert 1 not in s.rare_classes


class TestRareClassInjection:
    def test_every_batch_contains_rare_positive(self, imbalanced_labels: np.ndarray) -> None:
        s = BalancedMultilabelBatchSampler(
            labels=imbalanced_labels,
            batch_size=8,
            rare_class_threshold=0.5,
            seed=0,
        )

        for batch in s:
            batch_labels = imbalanced_labels[batch]
            # Class 3 should have at least one positive in every batch
            assert batch_labels[:, 3].sum() >= 1, (
                f"Rare class missing in batch {batch}: labels={batch_labels[:, 3]}"
            )

    def test_no_duplicates_in_single_batch(self, imbalanced_labels: np.ndarray) -> None:
        s = BalancedMultilabelBatchSampler(labels=imbalanced_labels, batch_size=8, seed=0)
        for batch in s:
            assert len(set(batch)) == len(batch), f"Duplicate indices in batch: {batch}"


class TestEpochHandling:
    def test_set_epoch_changes_order(self, imbalanced_labels: np.ndarray) -> None:
        s = BalancedMultilabelBatchSampler(labels=imbalanced_labels, batch_size=8, seed=0)
        s.set_epoch(0)
        epoch0 = list(s)
        s.set_epoch(5)
        epoch5 = list(s)
        # The two epochs should produce different batches
        assert epoch0 != epoch5

    def test_same_seed_same_batches(self, imbalanced_labels: np.ndarray) -> None:
        s1 = BalancedMultilabelBatchSampler(labels=imbalanced_labels, batch_size=8, seed=42)
        s2 = BalancedMultilabelBatchSampler(labels=imbalanced_labels, batch_size=8, seed=42)
        s1.set_epoch(0)
        s2.set_epoch(0)
        assert list(s1) == list(s2)


class TestNoRareClasses:
    def test_balanced_dataset_still_works(self) -> None:
        """If no class is rare, the sampler should still produce
        valid batches (just random sampling)."""
        labels = np.ones((40, 3), dtype=np.int8)  # every sample has every class
        s = BalancedMultilabelBatchSampler(labels=labels, batch_size=4, seed=0)
        for batch in s:
            assert len(batch) == 4
            assert len(set(batch)) == 4
