# src/serving/chunking.py
"""
Sliding-window inference for proteins longer than max_position_embeddings.

The ESM-2 backbone has a hard cap on sequence length (typically 1024
tokens). When a protein exceeds this cap, the dataset truncates it,
which means the predictor never sees the C-terminal half of long
proteins. For subcellular localization that is a real problem because
many targeting signals (e.g. PTS1, KKXX) live at the very end of the
sequence.

This module provides:

  - ``split_into_chunks``: cuts a long sequence into overlapping
    windows of length ``window_size`` with stride ``window_size - overlap``.
  - ``aggregate_logits``: combines per-chunk logits into a single
    sequence-level logit vector via mean pooling (default), max
    pooling, or a weighted average.

The default pooling is ``mean`` which mirrors the rest of the project.
``max`` is useful when even a single window with strong positive
evidence should suffice (e.g. detecting a signal peptide in a long
protein).

Usage:
    from src.serving.chunking import split_into_chunks, aggregate_logits
    chunks = split_into_chunks(seq, window_size=1024, overlap=128)
    per_chunk_logits = [predictor.raw_logits(c) for c in chunks]
    final_logits = aggregate_logits(per_chunk_logits, strategy="mean")
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_into_chunks(
    sequence: str,
    window_size: int,
    overlap: int = 0,
    min_chunk_size: int | None = None,
) -> list[str]:
    """
    Split a long sequence into (possibly overlapping) windows.

    Args:
        sequence: The full amino acid sequence.
        window_size: Maximum length of each chunk.
        overlap: Number of residues to overlap between consecutive
            chunks. Larger overlap means more redundant computation
            but smoother boundary transitions.
        min_chunk_size: Drop trailing chunks shorter than this. When
            omitted, defaults to ``max(1, window_size // 4)``, which is
            a reasonable default for most uses.

    Returns:
        List of sequence chunks. If ``len(sequence) <= window_size``,
        returns ``[sequence]``.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if overlap < 0 or overlap >= window_size:
        raise ValueError(f"overlap must be in [0, window_size), got {overlap}")
    if min_chunk_size is not None and min_chunk_size < 1:
        raise ValueError(f"min_chunk_size must be >= 1, got {min_chunk_size}")

    if len(sequence) <= window_size:
        return [sequence]

    effective_min_chunk_size = (
        max(1, window_size // 4) if min_chunk_size is None else min_chunk_size
    )
    stride = window_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(sequence):
        end = min(start + window_size, len(sequence))
        chunk = sequence[start:end]
        if len(chunk) >= effective_min_chunk_size:
            chunks.append(chunk)
        if end == len(sequence):
            break
        start += stride

    return chunks


def aggregate_logits(
    chunk_logits: list[np.ndarray],
    strategy: Literal["mean", "max", "weighted"] = "mean",
    weights: list[float] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Combine per-chunk logits into a single sequence-level logit vector.

    Args:
        chunk_logits: List of 1D logit arrays, one per chunk. All must
            have the same length (the number of classes).
        strategy: How to combine the chunks.
            - ``"mean"``: simple average. Conservative, default.
            - ``"max"``: per-class maximum across chunks. Use when a
              localization signal in any chunk should "win".
            - ``"weighted"``: weighted average (requires ``weights``).
        weights: Per-chunk weights for the ``"weighted"`` strategy. The
            list must have the same length as ``chunk_logits`` and sum
            to a positive number.

    Returns:
        1D logit array of shape ``(num_classes,)``.
    """
    if not chunk_logits:
        raise ValueError("chunk_logits is empty")

    stacked = cast(
        npt.NDArray[np.float64],
        np.stack([np.asarray(c, dtype=np.float64) for c in chunk_logits], axis=0),
    )

    if strategy == "mean":
        return cast(npt.NDArray[np.float64], stacked.mean(axis=0))
    elif strategy == "max":
        return cast(npt.NDArray[np.float64], stacked.max(axis=0))
    elif strategy == "weighted":
        if weights is None:
            raise ValueError("strategy='weighted' requires the weights argument")
        if len(weights) != len(chunk_logits):
            raise ValueError(
                f"weights length {len(weights)} != chunk_logits length {len(chunk_logits)}"
            )
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        w = w / total
        return cast(npt.NDArray[np.float64], (stacked * w[:, np.newaxis]).sum(axis=0))
    else:
        raise ValueError(f"Unknown aggregation strategy: '{strategy}'")
