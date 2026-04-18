# tests/unit/test_splitting.py
"""Tests for data splitting functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import (
    _check_mmseqs2,
    _split_by_clusters,
    _split_random_stratified,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with 100 proteins across 4 classes."""
    rng = np.random.RandomState(42)
    n = 100
    classes = ["Nucleus", "Cytoplasm", "Membrane", "Mitochondrion"]
    return pd.DataFrame(
        {
            "accession": [f"P{i:05d}" for i in range(n)],
            "sequence": ["M" + "A" * rng.randint(50, 200) for _ in range(n)],
            "locations": [[classes[rng.randint(0, len(classes))]] for _ in range(n)],
            "locations_str": None,  # Will be set below
        }
    )


@pytest.fixture()
def sample_df_with_str(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df["locations_str"] = sample_df["locations"].apply(lambda x: "|".join(x))
    return sample_df


# ---------------------------------------------------------------------------
# Cluster-based splitting
# ---------------------------------------------------------------------------


class TestSplitByClusters:
    """Tests for cluster-level splitting."""

    def test_no_accession_overlap(self, sample_df: pd.DataFrame) -> None:
        # Create simple cluster assignments: every 5 proteins share a cluster
        cluster_map = {row["accession"]: i // 5 for i, row in sample_df.iterrows()}
        train, val, test = _split_by_clusters(
            sample_df, cluster_map, test_size=0.2, val_size=0.2, seed=42
        )

        train_acc = set(train["accession"])
        val_acc = set(val["accession"])
        test_acc = set(test["accession"])

        assert len(train_acc & val_acc) == 0, "Train/val overlap"
        assert len(train_acc & test_acc) == 0, "Train/test overlap"
        assert len(val_acc & test_acc) == 0, "Val/test overlap"

    def test_all_data_accounted_for(self, sample_df: pd.DataFrame) -> None:
        cluster_map = {row["accession"]: i // 5 for i, row in sample_df.iterrows()}
        train, val, test = _split_by_clusters(
            sample_df, cluster_map, test_size=0.2, val_size=0.2, seed=42
        )

        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)

    def test_no_cluster_spans_splits(self, sample_df: pd.DataFrame) -> None:
        cluster_map = {row["accession"]: i // 5 for i, row in sample_df.iterrows()}
        train, val, test = _split_by_clusters(
            sample_df, cluster_map, test_size=0.2, val_size=0.2, seed=42
        )

        # Re-add cluster IDs to check
        for split_name, split_df in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:
            split_clusters = {cluster_map[acc] for acc in split_df["accession"]}
            for other_name, other_df in [
                ("train", train),
                ("val", val),
                ("test", test),
            ]:
                if split_name == other_name:
                    continue
                other_clusters = {cluster_map[acc] for acc in other_df["accession"]}
                shared = split_clusters & other_clusters
                assert len(shared) == 0, f"Cluster(s) {shared} span {split_name} and {other_name}"


# ---------------------------------------------------------------------------
# Random stratified splitting
# ---------------------------------------------------------------------------


class TestSplitRandomStratified:
    """Tests for random stratified splitting."""

    def test_no_accession_overlap(self, sample_df: pd.DataFrame) -> None:
        train, val, test = _split_random_stratified(sample_df, test_size=0.2, val_size=0.2, seed=42)

        train_acc = set(train["accession"])
        val_acc = set(val["accession"])
        test_acc = set(test["accession"])

        assert len(train_acc & val_acc) == 0
        assert len(train_acc & test_acc) == 0
        assert len(val_acc & test_acc) == 0

    def test_approximate_proportions(self, sample_df: pd.DataFrame) -> None:
        train, val, test = _split_random_stratified(sample_df, test_size=0.2, val_size=0.2, seed=42)

        total = len(sample_df)
        assert abs(len(test) / total - 0.2) < 0.1
        assert abs(len(val) / total - 0.2) < 0.1

    def test_reproducible(self, sample_df: pd.DataFrame) -> None:
        train1, val1, test1 = _split_random_stratified(sample_df, seed=42)
        train2, val2, test2 = _split_random_stratified(sample_df, seed=42)

        assert list(train1["accession"]) == list(train2["accession"])
        assert list(test1["accession"]) == list(test2["accession"])


# ---------------------------------------------------------------------------
# MMseqs2 availability
# ---------------------------------------------------------------------------


class TestMMseqs2Check:
    """Tests for MMseqs2 detection."""

    def test_returns_bool(self) -> None:
        result = _check_mmseqs2()
        assert isinstance(result, bool)
