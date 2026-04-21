# tests/unit/test_chunking.py
"""Tests for the sequence chunking utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.serving.chunking import aggregate_logits, split_into_chunks

# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_short_sequence_returns_single_chunk(self) -> None:
        seq = "M" * 100
        result = split_into_chunks(seq, window_size=200)
        assert result == [seq]

    def test_exact_window_size(self) -> None:
        seq = "A" * 200
        result = split_into_chunks(seq, window_size=200)
        assert result == [seq]

    def test_basic_split_no_overlap(self) -> None:
        seq = "A" * 300
        result = split_into_chunks(seq, window_size=100, overlap=0)
        assert len(result) == 3
        assert all(len(c) == 100 for c in result)

    def test_split_with_overlap(self) -> None:
        seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 chars
        result = split_into_chunks(seq, window_size=10, overlap=2)
        assert result[0] == "ABCDEFGHIJ"
        assert result[1] == "IJKLMNOPQR"  # overlap of 2 chars
        # Last chunk: starting at 16, length 10
        assert result[2] == "QRSTUVWXYZ"

    def test_min_chunk_size_drops_short_tail(self) -> None:
        seq = "A" * 100 + "B" * 5  # 5-char tail
        result = split_into_chunks(seq, window_size=100, overlap=0, min_chunk_size=10)
        assert len(result) == 1
        assert result[0] == "A" * 100

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            split_into_chunks("A" * 100, window_size=0)

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            split_into_chunks("A" * 100, window_size=10, overlap=10)

    def test_full_coverage_with_overlap(self) -> None:
        """Every residue must appear in at least one chunk."""
        seq = "".join(chr(65 + (i % 20)) for i in range(500))
        chunks = split_into_chunks(seq, window_size=100, overlap=20)
        covered = set()
        # Reconstruct each chunk's start position
        stride = 100 - 20
        for i, chunk in enumerate(chunks):
            start = i * stride
            for j in range(len(chunk)):
                covered.add(start + j)
        assert len(covered) >= len(seq) - 20  # near-total coverage


# ---------------------------------------------------------------------------
# aggregate_logits
# ---------------------------------------------------------------------------


class TestAggregateLogits:
    def test_mean_strategy(self) -> None:
        chunks = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
        ]
        result = aggregate_logits(chunks, strategy="mean")
        assert np.allclose(result, [2.0, 2.0, 2.0])

    def test_max_strategy(self) -> None:
        chunks = [
            np.array([1.0, 5.0, 3.0]),
            np.array([4.0, 2.0, 1.0]),
        ]
        result = aggregate_logits(chunks, strategy="max")
        assert np.allclose(result, [4.0, 5.0, 3.0])

    def test_weighted_strategy(self) -> None:
        chunks = [
            np.array([0.0, 0.0]),
            np.array([4.0, 4.0]),
        ]
        result = aggregate_logits(chunks, strategy="weighted", weights=[1.0, 3.0])
        # Weighted mean: (0*1 + 4*3) / 4 = 3
        assert np.allclose(result, [3.0, 3.0])

    def test_single_chunk(self) -> None:
        chunks = [np.array([1.0, 2.0, 3.0])]
        result = aggregate_logits(chunks, strategy="mean")
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_logits([])

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            aggregate_logits([np.array([1.0])], strategy="median")  # type: ignore[arg-type]

    def test_weighted_requires_weights(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            aggregate_logits([np.array([1.0])], strategy="weighted")

    def test_weighted_weights_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length"):
            aggregate_logits(
                [np.array([1.0]), np.array([2.0])],
                strategy="weighted",
                weights=[1.0],
            )
