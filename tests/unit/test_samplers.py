# tests/unit/test_samplers.py
"""Tests for LengthBucketBatchSampler."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.samplers import LengthBucketBatchSampler


@pytest.fixture()
def synthetic_lengths() -> list[int]:
    """100 sequences with lengths in [50, 1500] drawn from a power law."""
    rng = np.random.default_rng(0)
    return [int(50 + rng.exponential(300)) for _ in range(100)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, synthetic_lengths: list[int]) -> None:
        s = LengthBucketBatchSampler(synthetic_lengths, batch_size=8)
        assert len(s) == (100 + 7) // 8  # ceil(100/8)

    def test_drop_last(self, synthetic_lengths: list[int]) -> None:
        s = LengthBucketBatchSampler(
            synthetic_lengths, batch_size=8, drop_last=True
        )
        assert len(s) == 100 // 8

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            LengthBucketBatchSampler([1, 2, 3], batch_size=0)

    def test_empty_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            LengthBucketBatchSampler([], batch_size=8)

    def test_invalid_jitter_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            LengthBucketBatchSampler(
                [1, 2, 3], batch_size=2, jitter_fraction=2.0
            )


# ---------------------------------------------------------------------------
# Coverage and uniqueness
# ---------------------------------------------------------------------------


class TestCoverage:
    """Every sample must appear exactly once per epoch (unless drop_last)."""

    def test_all_indices_present(self, synthetic_lengths: list[int]) -> None:
        s = LengthBucketBatchSampler(
            synthetic_lengths, batch_size=8, shuffle=True, seed=42
        )
        all_indices = [idx for batch in s for idx in batch]
        assert sorted(all_indices) == list(range(100))

    def test_no_duplicates(self, synthetic_lengths: list[int]) -> None:
        s = LengthBucketBatchSampler(
            synthetic_lengths, batch_size=8, shuffle=True, seed=42
        )
        all_indices = [idx for batch in s for idx in batch]
        assert len(set(all_indices)) == len(all_indices)

    def test_drop_last_omits_remainder(
        self, synthetic_lengths: list[int]
    ) -> None:
        s = LengthBucketBatchSampler(
            synthetic_lengths, batch_size=8, drop_last=True, shuffle=False
        )
        all_indices = [idx for batch in s for idx in batch]
        # 100 / 8 = 12 full batches, 4 remainder dropped
        assert len(all_indices) == 96


# ---------------------------------------------------------------------------
# Padding efficiency
# ---------------------------------------------------------------------------


class TestPaddingEfficiency:
    """The whole point of bucketing: similar lengths in the same batch."""

    def test_bucketed_padding_lower_than_random(
        self, synthetic_lengths: list[int]
    ) -> None:
        lengths_arr = np.asarray(synthetic_lengths)
        batch_size = 8

        # Random batches: average waste is high
        rng = np.random.default_rng(0)
        random_order = rng.permutation(len(lengths_arr))
        random_waste = 0.0
        for start in range(0, len(lengths_arr), batch_size):
            chunk = lengths_arr[random_order[start : start + batch_size]]
            random_waste += int(chunk.max() * len(chunk) - chunk.sum())

        # Bucketed batches: should be much lower
        s = LengthBucketBatchSampler(
            synthetic_lengths,
            batch_size=batch_size,
            shuffle=True,
            seed=0,
        )
        bucketed_waste = 0.0
        for batch in s:
            chunk = lengths_arr[np.asarray(batch)]
            bucketed_waste += int(chunk.max() * len(chunk) - chunk.sum())

        # Expect at least 4x less padding waste
        assert bucketed_waste < random_waste / 4, (
            f"bucketed_waste={bucketed_waste}, random_waste={random_waste}"
        )


# ---------------------------------------------------------------------------
# Determinism and epoch handling
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_order(self, synthetic_lengths: list[int]) -> None:
        s1 = LengthBucketBatchSampler(synthetic_lengths, batch_size=8, seed=42)
        s2 = LengthBucketBatchSampler(synthetic_lengths, batch_size=8, seed=42)
        assert list(s1) == list(s2)

    def test_set_epoch_changes_order(
        self, synthetic_lengths: list[int]
    ) -> None:
        s = LengthBucketBatchSampler(synthetic_lengths, batch_size=8, seed=42)
        s.set_epoch(0)
        epoch0 = list(s)
        s.set_epoch(5)
        epoch5 = list(s)
        # Some batches will differ (jitter + reshuffling)
        assert epoch0 != epoch5

    def test_no_shuffle_is_fully_sorted(self) -> None:
        lengths = [100, 50, 200, 75, 300, 25]
        s = LengthBucketBatchSampler(lengths, batch_size=2, shuffle=False)
        batches = list(s)
        # With shuffle=False and stable sort, batches go in length order:
        # sorted indices = [5, 1, 3, 0, 2, 4] -> batches [[5,1],[3,0],[2,4]]
        assert batches == [[5, 1], [3, 0], [2, 4]]
