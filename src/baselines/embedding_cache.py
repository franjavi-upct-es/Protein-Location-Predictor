# src/baselines/embedding_cache.py
"""
On-disk embedding cache for frozen-ESM-2 baselines.

Computing ESM-2 embeddings is expensive, but for *frozen* baselines
(linear probe, XGBoost) the embeddings depend only on:

    - the backbone checkpoint name
    - the pooling strategy
    - the maximum sequence length
    - the input sequences themselves

...and crucially do not change between training runs of the downstream
classifier. This module computes the embeddings once per (split,
backbone, pooling, max_length) combination and persists them as a
``.npz`` file alongside the per-row labels.

Usage::

    from src.baselines.embedding_cache import compute_or_load_embeddings
    X, y, accessions = compute_or_load_embeddings(
        cfg, split="train", backbone_name=cfg.model.backbone.name
    )
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from src.utils.config import DotDict, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def _cache_key(
    backbone_name: str,
    pooling: str,
    max_length: int,
    split_csv_path: Path,
) -> str:
    """Stable hash of every input that influences the embeddings."""
    payload = "|".join(
        [
            backbone_name,
            pooling,
            str(max_length),
            str(split_csv_path.resolve()),
            # Include the file + mtime so a re-download invalidates
            # the cache automatically
            str(split_csv_path.stat().st_size),
            str(int(split_csv_path.stat().st_mtime)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(cfg: DotDict, key: str, split: str) -> Path:
    project_root = Path(cfg.get("project_root", "."))
    return project_root / ".cache" / "embeddings" / f"{split}_{key}.npz"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _build_label_matrix(df: pd.DataFrame, label_list: list[str]) -> np.ndarray:
    """Build a multi-hot label matrix from the locations_str column."""
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    matrix = np.zeros((len(df), len(label_list)), dtype=np.float32)
    for row_idx, locations_str in enumerate(df["locations_str"].fillna("")):
        for loc in str(locations_str).split("|"):
            if loc in label_to_idx:
                matrix[row_idx, label_to_idx[loc]] = 1.0
    return matrix


@torch.no_grad()
def _compute_embeddings(
    sequences: list[str],
    backbone_name: str,
    pooling: str,
    max_length: int,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Run ESM-2 forward passes and pool to fixed-size embeddings."""
    from transformers import AutoTokenizer, EsmModel

    logger.info(
        f"Computing embeddings: {len(sequences)} sequences, "
        f"backbone={backbone_name}, pooling={pooling}"
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = cast(Any, EsmModel).from_pretrained(backbone_name).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        out = model(**encoded)
        hidden = out.last_hidden_state  # (B, L, D)
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        if pooling == "cls":
            pooled = hidden[:, 0, :]
        elif pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "mean_cls":
            cls = hidden[:, 0, :]
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.cat([cls, mean], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        if (start // batch_size) % 10 == 0:
            logger.info(f"  embedded {start + len(chunk)} / {len(sequences)}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_or_load_embeddings(
    cfg: DotDict,
    split: str,
    label_list: list[str],
    backbone_name: str | None = None,
    pooling: str | None = None,
    max_length: int | None = None,
    device: str | None = None,
    batch_size: int = 8,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return ``(X, y, accessions)`` for a given split, computing if needed.

    Args:
        cfg: Project configuration.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        label_list: Ordered list of class labels (used to build ``y``).
        backbone_name: Override the configured backbone.
        pooling: Override the configured pooling.
        max_length: Override the configured max length.
        device: ``"cuda"`` / ``"cpu"`` (auto-detected if None).
        batch_size: Forward-pass batch size.
        use_cache: If True, read from / write to ``.cache/embeddings``.

    Returns:
        ``X``: ``(n_samples, embedding_dim)`` float32 array.
        ``y``: ``(n_samples, n_classes)`` float32 multi-hot array.
        ``accessions``: list of UniProt accessions in row order.
    """
    backbone_name = backbone_name or str(cfg.model.backbone.name)
    pooling = pooling or str(cfg.model.get("pooling", "mean"))
    max_length = max_length or int(cfg.model.backbone.get("max_position_embeddings", 1024))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_dir = resolve_path(cfg, "paths.splits_dir")
    split_path = splits_dir / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    df = pd.read_csv(split_path)
    if "sequence" not in df.columns or "accession" not in df.columns:
        raise ValueError(f"Split CSV must have 'sequence' and 'accession' columns: {split_path}")

    accessions = df["accession"].astype(str).tolist()
    y = _build_label_matrix(df, label_list)

    key = _cache_key(backbone_name, pooling, max_length, split_path)
    cache_path = _cache_path(cfg, key, split)

    if use_cache and cache_path.exists():
        try:
            data = np.load(cache_path)
            cached_X = data["X"]
            if cached_X.shape[0] == len(df):
                logger.info(f"Embedding cache hit: {split} ({cache_path.name})")
                return cached_X, y, accessions
            logger.warning(f"Embedding cache shape mismatch — recomputing {split}")
        except (OSError, KeyError) as e:
            logger.warning(f"Embedding cache read failed ({e}) — recomputing {split}")

    X = _compute_embeddings(
        sequences=df["sequence"].astype(str).tolist(),
        backbone_name=backbone_name,
        pooling=pooling,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X)
        logger.info(f"Embeddings cached to {cache_path}")

    return X, y, accessions
