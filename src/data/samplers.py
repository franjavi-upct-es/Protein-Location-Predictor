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
import torch
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
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0.0 <= jitter_fraction <= 1.0:
            raise ValueError(f"jitter_fraction must be in [0, 1], got {jitter_fraction}")

        # DDP awareness: auto-detect from torch.distributed when not given.
        if num_replicas is None or rank is None:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    if num_replicas is None:
                        num_replicas = dist.get_world_size()
                    if rank is None:
                        rank = dist.get_rank()
            except ImportError:
                pass
        self.num_replicas = int(num_replicas) if num_replicas is not None else 1
        self.rank = int(rank) if rank is not None else 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} out of range for num_replicas {self.num_replicas}")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last) or self.num_replicas > 1
        self.jitter_fraction = float(jitter_fraction)
        self._epoch = 0

        if len(self.lengths) == 0:
            raise ValueError("LengthBucketBatchSampler received empty lengths")

        global_bs = self.batch_size * self.num_replicas
        n_full = len(self.lengths) // global_bs
        self._n_batches = (
            n_full if self.drop_last else n_full + (1 if len(self.lengths) % global_bs else 0)
        )

        logger.info(
            f"LengthBucketBatchSampler: {len(self.lengths)} samples, "
            f"batch_size={self.batch_size}, n_batches={self._n_batches}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}, "
            f"num_replicas={self.num_replicas}, rank={self.rank}"
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

        # Stable sort so equal-length samples preserve dataset order.
        # rng is seeded identically across DDP ranks, so all ranks produce
        # the same sorted ordering and the same downstream permutation.
        sorted_indices = np.argsort(sort_keys, kind="stable")

        # Group consecutive sorted indices into "global batches" sized for
        # the whole world (batch_size × num_replicas). Each rank then takes
        # its contiguous slice of every global batch. This keeps per-step
        # sequence lengths balanced across ranks (DDP all-reduce waits on
        # the slowest rank, so length similarity at each step matters).
        global_bs = self.batch_size * self.num_replicas
        batches: list[list[int]] = []
        for start in range(0, n, global_bs):
            global_chunk = sorted_indices[start : start + global_bs]
            if len(global_chunk) < global_bs and (self.drop_last or self.num_replicas > 1):
                # Drop the trailing partial global batch under DDP to keep
                # rank counts identical (required by all_reduce).
                continue
            my_slice = global_chunk[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
            if len(my_slice) == 0:
                continue
            batches.append([int(i) for i in my_slice])

        # Shuffle batch order so the model does not see monotonically
        # increasing lengths over an epoch. Same rng → same permutation
        # on every rank, so step t aligns across ranks.
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]

        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return self._n_batches


class BalancedMultilabelBatchSampler(Sampler[list[int]]):
    """
    Yield batches that guarantee positive examples for rare classes.

    Each batch is filled in two phases:

    1. **Rare-class injection.** For every class that has fewer than
       ``min_positives_per_batch`` positives in the dataset, the sampler
       picks one positive sample of that class and adds it to the batch.
    2. **Random fill.** The rest of the batch is filled by sampling
       uniformly from the remaining samples (without replacement within
       the same batch, with replacement across batches).

    This is the standard recipe for imbalanced multi-label problems and
    is complementary to focal loss: focal loss says "pay more attention
    when this class appears", the sampler ensures it actually appears.

    Args:
        labels: Multi-hot label matrix of shape ``(N, C)``. Use 1 for
            positive, 0 for negative. Can be a numpy array or torch tensor.
        batch_size: Number of samples per batch.
        n_batches_per_epoch: How many batches to yield per epoch. If
            None, defaults to ``len(labels) // batch_size``.
        rare_class_threshold: A class is "rare" if its positive count
            is below ``rare_class_threshold * len(labels) / num_classes``.
            Default 0.5 means classes with less than half the average
            count are treated as rare.
        seed: Base RNG seed.
    """

    def __init__(
        self,
        labels: np.ndarray | Sequence[Sequence[int]] | torch.Tensor,
        batch_size: int,
        n_batches_per_epoch: int | None = None,
        rare_class_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, go {batch_size}")

        try:
            import torch as _torch

            if isinstance(labels, _torch.Tensor):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.asarray(labels)
        except ImportError:
            labels_np = np.asarray(labels)

        if labels_np.ndim != 2:
            raise ValueError(f"labels must be 2D (N, C), got shape {labels_np.shape}")

        self.labels = labels_np.astype(np.uint8)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = 0

        n_samples, n_classes = self.labels.shape
        per_class_counts = self.labels.sum(axis=0)
        avg = n_samples / max(1, n_classes)
        threshold = avg * rare_class_threshold

        # Identify rare classes and pre-compute their positive index lists
        self.rare_classes: list[int] = []
        self.positives_per_class: dict[int, np.ndarray] = {}
        for c in range(n_classes):
            positives = np.where(self.labels[:, c] == 1)[0]
            if len(positives) > 0 and per_class_counts[c] < threshold:
                self.rare_classes.append(c)
                self.positives_per_class[c] = positives

        self._n_batches = (
            int(n_batches_per_epoch)
            if n_batches_per_epoch is not None
            else max(1, n_samples // self.batch_size)
        )

        logger.info(
            f"BalancedMultilabelBatchSampler: {n_samples} samples, "
            f"{n_classes} classes, {len(self.rare_classes)} rare classes "
            f"(threshold={threshold:.1f}), batch_size={self.batch_size}, "
            f"n_batches={self._n_batches}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        n_samples = len(self.labels)
        all_indices = np.arange(n_samples)

        for _ in range(self._n_batches):
            batch: list[int] = []
            seen: set[int] = set()

            # Phase 1: rare-class injection
            for c in self.rare_classes:
                if len(batch) >= self.batch_size:
                    break
                positives = self.positives_per_class[c]
                # Pick a random positive that isn't already in the batch
                candidate = int(rng.choice(positives))
                if candidate not in seen:
                    batch.append(candidate)
                    seen.add(candidate)

            # Phase 2: random fill
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                pool = np.setdiff1d(all_indices, np.fromiter(seen, dtype=np.int64))
                if len(pool) >= remaining:
                    fill = rng.choice(pool, size=remaining, replace=False)
                else:
                    fill = rng.choice(pool, size=remaining, replace=True)
                batch.extend(int(i) for i in fill)

            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        return self._n_batches
