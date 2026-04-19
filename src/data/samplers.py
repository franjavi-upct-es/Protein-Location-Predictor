# src/data/samplers.py
"""
Length-aware batch samplers for protein sequence datasets.

Protein length distributions are heavy-tailed: a single 2000-residue
protein in a uniformly-shuffled batch wastes ~80% of compute on padding
for the other (typically much shorter) sequences. Grouping sequences of
similar length into the same batch eliminates this waste.

This module provides ``LengthBucketBatchSampler``, a drop-in
``BatchSampler`` for use with ``DataLoader(batch_sampler=...)``.

Usage::

    from torch.utils.data import DataLoader
    from src.data.samplers import LengthBucketBatchSampler

    sampler = LengthBucketBatchSampler(
        lengths=[len(s) for s in dataset.sequences],
        batch_size=8,
        shuffle=True,
        seed=42,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, ...)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    Yield batches of indices grouped by sequence length.

    The algorithm is intentionally simple and deterministic-given-seed:

    1. Pair every dataset index with its sequence length.
    2. Sort the pairs by length (with a small random jitter when
       ``shuffle=True`` so that consecutive epochs don't see identical
       batches even though their composition is similar).
    3. Slice the sorted list into chunks of ``batch_size``.
    4. Optionally drop the trailing partial chunk.
    5. Shuffle the order of the resulting chunks so that the model
       does not see batches in monotonically increasing length order
       (which biases gradient statistics).

    The padding waste in each batch is bounded by the length range
    inside the chunk, which is small for any reasonable
    ``batch_size`` relative to the dataset size.

    Args:

        lengths: Per-sample sequence lengths in dataset order. Must
            have the same length as the underlying dataset.
        batch_size: Number of samples per batch.
        shuffle: If True, jitter sorted positions and shuffle batch
            order across epochs. If False, the iteration order is
            fully deterministic given the input lengths.
        seed: Base RNG seed. The actual seed used per epoch is
            ``seed + epoch`` so that ``set_epoch(n)`` produces
            different orderings without losing reproducibility.
        drop_last: If True, discard the last batch when it is smaller
            than ``batch_size``.
        jitter_fraction: Magnitude of the random jitter applied to
            sort keys when ``shuffle=True``, as a fraction of the
            length range. Defaults to 0.05 (5%). Larger values trade
            padding efficiency for batch diversity.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        jitter_fraction: float = 0.05,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0.0 <= jitter_fraction <= 1.0:
            raise ValueError(f"jitter_fraction must be in [0, 1], got {jitter_fraction}")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.jitter_fraction = float(jitter_fraction)
        self._epoch = 0

        if len(self.lengths) == 0:
            raise ValueError("LengthBucketBatchSampler received empty lengths")

        n_full = len(self.lengths) // self.batch_size
        self._n_batches = (
            n_full if self.drop_last else n_full + (1 if len(self.lengths) % self.batch_size else 0)
        )

        logger.info(
            f"LengthBucketBatchSampler: {len(self.lengths)} samples, "
            f"batch_size={self.batch_size}, n_batches={self._n_batches}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}"
        )

    # ------------------------------------------------------------------
    # Epoch handling (compatible with DistributedSampler interface)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so reshuffling is deterministic."""
        self._epoch = int(epoch)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        n = len(self.lengths)

        # Build sort keys
        if self.shuffle and self.jitter_fraction > 0:
            length_range = float(self.lengths.max() - self.lengths.min())
            jitter_scale = max(1.0, length_range * self.jitter_fraction)
            jitter = rng.normal(0.0, jitter_scale, size=n)
            sort_keys = self.lengths.astype(np.float64) + jitter
        else:
            sort_keys = self.lengths.astype(np.float64)

        # Stable sort so equal-length samples preserve dataset order
        sorted_indices = np.argsort(sort_keys, kind="stable")

        # Chop into batches
        batches: list[list[int]] = []
        for start in range(0, n, self.batch_size):
            chunk = sorted_indices[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            batches.append([int(i) for i in chunk])

        # Shuffle batch order so the model does not see monotonically
        # increasing lengths over an epoch
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]

        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return self._n_batches
