# tests/unit/test_biophysical.py
"""Tests for biophysical feature computation."""

from __future__ import annotations

import numpy as np

from src.features.biophysical import (
    _clean_sequence,
    compute_biophysical_features,
    compute_single_sequence,
)


class TestCleanSequence:
    """Tests for sequence cleaning."""

    def test_standard_sequence_unchanged(self) -> None:
        assert _clean_sequence("MSKGEEL") == "MSKGEEL"

    def test_removes_nonstandard(self) -> None:
        assert _clean_sequence("MSKXUGEEL") == "MSKGEEL"

    def test_uppercases(self) -> None:
        assert _clean_sequence("mskgeel") == "MSKGEEL"

    def test_empty(self) -> None:
        assert _clean_sequence("") == ""


class TestComputeSingleSequence:
    """Tests for single-sequence feature computation."""

    def test_returns_all_properties(self) -> None:
        props = ["molecular_weight", "isoelectric_point", "gravy"]
        result = compute_single_sequence("MSKGEELFTGVVPILVELDGDVNGHK", props)
        assert set(result.keys()) == set(props)

    def test_molecular_weight_positive(self) -> None:
        result = compute_single_sequence(
            "MSKGEELFTGVVPILVELDGDVNGHK", ["molecular_weight"]
        )
        assert result["molecular_weight"] > 0

    def test_isoelectric_point_range(self) -> None:
        result = compute_single_sequence(
            "MSKGEELFTGVVPILVELDGDVNGHK", ["isoelectric_point"]
        )
        assert 0 < result["isoelectric_point"] < 14

    def test_short_sequence_returns_nan(self) -> None:
        result = compute_single_sequence("MSK", ["molecular_weight"])
        assert np.isnan(result["molecular_weight"])

    def test_unknown_property_returns_nan(self) -> None:
        result = compute_single_sequence(
            "MSKGEELFTGVVPILVELDGDVNGHK", ["nonexistent_prop"]
        )
        assert np.isnan(result["nonexistent_prop"])

    def test_sequence_length_property(self) -> None:
        seq = "MSKGEELFTGVVPILVELDGDVNGHK"
        result = compute_single_sequence(seq, ["sequence_length"])
        assert result["sequence_length"] == float(len(seq))


class TestComputeBiophysicalFeatures:
    """Tests for batch feature computation."""

    def test_output_shape(self) -> None:
        seqs = ["MSKGEELFTGVVPILVELDGDVNGHK", "MQIFVKTLTGKTITLEVEPSDTIENVK"]
        result = compute_biophysical_features(
            seqs, properties=["molecular_weight", "gravy"]
        )
        assert result.shape == (2, 2)

    def test_output_dtype(self) -> None:
        seqs = ["MSKGEELFTGVVPILVELDGDVNGHK"]
        result = compute_biophysical_features(
            seqs, properties=["molecular_weight"]
        )
        assert result.dtype == np.float32

    def test_standardized_output(self) -> None:
        """Output should be approximately standardized
        (zero mean, unit var)."""
        seqs = [
            "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKE",
            "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRL",
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYD",
        ]
        result = compute_biophysical_features(
            seqs, properties=["molecular_weight", "isoelectric_point", "gravy"]
        )
        # Mean should be close to 0 after standardization
        assert abs(result.mean()) < 0.5

    def test_empty_properties_returns_empty(self) -> None:
        seqs = ["MSKGEELFTGVVPILVELDGDVNGHK"]
        result = compute_biophysical_features(seqs, properties=[])
        assert result.shape == (1, 0)
