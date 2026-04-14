# src/data/splitting.py
"""
Homology-aware train/val/test splitting.

Clusters protein sequences by similarity using MMseqs2, then assigns
entire clusters to splits so that no homologous sequences appear in
both training and evaluation sets.

Falls back to stratified random splitting if MMseqs2 is not available.

Usage:
    from src.data.splitting import split_data
    splits = split_data(cfg, df)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MMseqs2 clustering
# ---------------------------------------------------------------------------


def _check_mmseqs2() -> bool:
    """Check if MMseqs2 is installed and available."""
    return shutil.which("mmseqs") is not None


def _run_mmseqs2_clustering(
    sequences: dict[str, str],
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
    work_dir: Path | None = None,
) -> dict[str, int]:
    """
    Cluster sequences using MMseqs2.

    Args:
        sequences: Mapping of accession -> amino acid sequence.
        identity_threshold: Minimum sequence identity for clustering (0-1).
        coverage_threshold: Minimum alignment coverage for clustering (0-1).
        work_dir: Working directory for temporary files. Uses tempdir if None.

    Returns:
        Mapping of accession -> cluster_id.
    """
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mmseqs_"))

    try:
        fasta_path = work_dir / "sequences.fasta"
        db_path = work_dir / "seqdb"
        cluster_path = work_dir / "clusters"
        tsv_path = work_dir / "clusters.tsv"
        tmp_dir = work_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        # Write FASTA
        with open(fasta_path, "w") as f:
            for acc, seq in sequences.items():
                f.write(f">{acc}\n{seq}\n")

        # Create MMseqs2 database
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            check=True,
            capture_output=True,
        )

        # Cluster
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                str(db_path),
                str(cluster_path),
                str(tmp_dir),
                "--min-seq-id",
                str(identity_threshold),
                "-c",
                str(coverage_threshold),
                "--cov-mode",
                "0",
                "-s",
                "7.5",
            ],
            check=True,
            capture_output=True,
        )

        # Extract cluster assignments to TSV
        subprocess.run(
            [
                "mmseqs",
                "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_path),
                str(tsv_path),
            ],
            check=True,
            capture_output=True,
        )

        # Parse TSV: columns are (representative, member)
        cluster_df = pd.read_csv(
            tsv_path, sep="\t", header=None, names=["representative", "member"]
        )

        # Map each representative to a numeric cluster ID
        reps = cluster_df["representative"].unique()
        rep_to_id = {rep: idx for idx, rep in enumerate(reps)}
        cluster_map = {
            row["member"]: rep_to_id[row["representative"]]
            for _, row in cluster_df.iterrows()
        }

        n_clusters = len(reps)
        logger.info(
            f"MMseqs2 clustering: {len(sequences)} sequences -> "
            f"{n_clusters} clusters at {identity_threshold:.0%} identity"
        )

        return {str(k): int(v) for k, v in cluster_map.items()}

    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cluster-level splitting
# ---------------------------------------------------------------------------


def _split_by_clusters(
    df: pd.DataFrame,
    cluster_map: dict[str, int],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at the cluster level to prevent homology leakage.

    Entire clusters are assigned to train, val, or test. No cluster
    spans multiple splits.

    Args:
        df: Processed protein DataFrame (must have 'accession' column).
        cluster_map: Mapping of accession -> cluster_id.
        test_size: Fraction of data for the test set.
        val_size: Fraction of data for the validation set.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.copy()
    df["cluster_id"] = df["accession"].map(cluster_map)  # pyright: ignore[reportArgumentType]

    # Drop any unmapped accessions (shouldn't happen, but defensive)
    unmapped = df["cluster_id"].isna().sum()
    if unmapped > 0:
        logger.warning(
            f"{unmapped} accessions not found in cluster map — dropping"
        )
        df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Get unique clusters and shuffl
    rng = np.random.RandomState(seed)
    clusters = df["cluster_id"].unique()
    rng.shuffle(clusters)  # pyright: ignore[reportArgumentType]

    n_total = len(clusters)
    n_test = max(1, int(n_total * test_size))
    n_val = max(1, int(n_total * val_size))

    test_clusters = set(clusters[:n_test])
    val_clusters = set(clusters[n_test : n_test + n_val])
    train_clusters = set(clusters[n_test + n_val :])

    train_df = df[df["cluster_id"].isin(list(train_clusters))].drop(
        columns=["cluster_id"]
    )
    val_df = df[df["cluster_id"].isin(list(val_clusters))].drop(
        columns=["cluster_id"]
    )
    test_df = df[df["cluster_id"].isin(list(test_clusters))].drop(
        columns=["cluster_id"]
    )

    logger.info(
        "Cluster-based split: "
        f"train={len(train_df)} ({len(train_clusters)} clusters), "
        f"val={len(val_df)} ({len(val_clusters)} clusters), "
        f"test={len(test_df)} ({len(test_clusters)} clusters)"
    )

    return train_df, val_df, test_df  # pyright: ignore[reportReturnType]


# ---------------------------------------------------------------------------
# Random stratified fallback
# ---------------------------------------------------------------------------


def _split_random_stratified(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback: stratified random split (no homology awareness).

    Uses the primary (first) location label for stratification in the
    multi-label case.
    """
    logger.warning(
        "Using random stratified split (no homology awareness). "
        "Evaluation metrics may be inflated due to sequence "
        "similarity leakage."
    )

    # Use first location as stratification key
    stratify_col = df["locations"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )

    # First split: train+val vs test
    train_val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_val_df, test_df = train_test_split(  # pyright: ignore[reportAssignmentType]
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col,
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    stratify_tv = train_val_df["locations"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_df, val_df = train_test_split(  # pyright: ignore[reportAssignmentType]
        train_val_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_tv,
    )

    logger.info(
        f"Random stratified split: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df  # pyright: ignore[reportReturnType]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_data(
    cfg: DotDict,
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split protein data into train/val/test sets.

    Uses homology-aware splitting via MMseqs2 if available, otherwise
    falls back to random stratified splitting.

    Args:
        cfg: Project configuration.
        df: Processed DataFrame (from process_data). Must have 'accession',
            'sequence', and 'locations' columns.

    Returns:
        Dict with keys 'train', 'val', 'test', each containing a DataFrame.
    """
    split_cfg = cfg.splitting
    method = split_cfg.get("method", "homology")
    test_size = split_cfg.get("test_size", 0.15)
    val_size = split_cfg.get("val_size", 0.15)
    seed = cfg.project.get("seed", 42)

    if method == "homology" and _check_mmseqs2():
        # Build sequence dict
        sequences = dict(zip(df["accession"], df["sequence"], strict=False))

        identity = split_cfg.get("sequence_identity_threshold", 0.3)
        coverage = split_cfg.get("coverage_threshold", 0.8)

        cluster_map = _run_mmseqs2_clustering(
            sequences,
            identity_threshold=identity,
            coverage_threshold=coverage,
        )

        train_df, val_df, test_df = _split_by_clusters(
            df,
            cluster_map,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

    elif method == "homology" and not _check_mmseqs2():
        if split_cfg.get("random_fallback", True):
            logger.warning(
                "MMseqs2 not found. Install it for homology-aware splitting: "
                "https://github.com/soedinglab/MMseqs2"
            )
            train_df, val_df, test_df = _split_random_stratified(
                df,
                test_size=test_size,
                val_size=val_size,
                seed=seed,
            )
        else:
            raise RuntimeError(
                "MMseqs2 is required for homology-aware "
                "splitting but was not found. Install "
                "MMseqs2 or set splitting.random_fallback"
                "=true in config."
            )
    else:
        # Explicitly requested random splitting
        train_df, val_df, test_df = _split_random_stratified(
            df,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

    # Save splits
    splits_dir = resolve_path(cfg, "paths.splits_dir")
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for name, split_df in splits.items():
        path = splits_dir / f"{name}.csv"
        save_df = split_df.drop(columns=["locations"], errors="ignore")
        save_df.to_csv(path, index=False)
        logger.info(f"Saved {name} split ({len(split_df)} samples) to {path}")

    return splits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for data splitting."""
    cfg = load_config()
    setup_logging(level=cfg.project.log_level)

    proc_path = (
        resolve_path(cfg, "paths.processed_dir") / "proteins_processed.csv"
    )
    df = pd.read_csv(proc_path)
    # Reconstruct the locations list from pipe-separated string
    df["locations"] = df["locations_str"].apply(
        lambda s: s.split("|") if isinstance(s, str) else []
    )

    split_data(cfg, df)


if __name__ == "__main__":
    main()
