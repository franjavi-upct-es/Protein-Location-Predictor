# tests/unit/test_validation.py
"""Tests for data validation utilities."""

from __future__ import annotations

import pandas as pd

from src.data.validation import (
    ValidationReport,
    validate_processed_data,
    validate_raw_data,
    validate_splits,
)

# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for the ValidationReport container."""

    def test_starts_passing(self) -> None:
        r = ValidationReport()
        assert r.passed is True

    def test_error_sets_failed(self) -> None:
        r = ValidationReport()
        r.add_error("something broke")
        assert r.passed is False
        assert len(r.errors) == 1

    def test_warning_keeps_passing(self) -> None:
        r = ValidationReport()
        r.add_warning("something iffy")
        assert r.passed is True
        assert len(r.warnings) == 1

    def test_summary_format(self) -> None:
        r = ValidationReport()
        r.add_error("bad data")
        r.add_warning("weird value")
        r.stats["count"] = 42
        summary = r.summary()
        assert "FAILED" in summary
        assert "bad data" in summary
        assert "weird value" in summary
        assert "42" in summary


# ---------------------------------------------------------------------------
# Raw data validation
# ---------------------------------------------------------------------------


class TestValidateRawData:
    """Tests for raw data validation."""

    def test_valid_data_passes(self) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001", "P002"],
                "sequence": ["MSKGEEL" * 10, "MQIFVKT" * 10],
                "subcellular_location": [
                    "SUBCELLULAR LOCATION: Nucleus",
                    "SUBCELLULAR LOCATION: Cytoplasm",
                ],
            }
        )
        report = validate_raw_data(df)
        assert report.passed

    def test_missing_columns_fails(self) -> None:
        df = pd.DataFrame({"accession": ["P001"], "sequence": ["MSKGEEL"]})
        report = validate_raw_data(df)
        assert not report.passed

    def test_null_values_warned(self) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001", "P002"],
                "sequence": ["MSKGEEL" * 10, None],
                "subcellular_location": ["Nucleus", "Cytoplasm"],
            }
        )
        report = validate_raw_data(df)
        assert len(report.warnings) > 0

    def test_duplicates_warned(self) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001", "P001"],
                "sequence": ["MSKGEEL" * 10, "MSKGEEL" * 10],
                "subcellular_location": ["Nucleus", "Cytoplasm"],
            }
        )
        report = validate_raw_data(df)
        assert any("duplicate" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Processed data validation
# ---------------------------------------------------------------------------


class TestValidateProcessedData:
    """Tests for processed data validation."""

    def test_valid_processed_data(self) -> None:
        df = pd.DataFrame(
            {
                "accession": [f"P{i:03d}" for i in range(100)],
                "sequence": ["MSKGEEL" * 20] * 100,
                "locations": [["Nucleus"]] * 50 + [["Cytoplasm"]] * 50,
            }
        )
        report = validate_processed_data(df, min_samples_per_class=10)
        assert report.passed

    def test_empty_locations_fails(self) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001"],
                "sequence": ["MSKGEEL" * 20],
                "locations": [[]],
            }
        )
        report = validate_processed_data(df)
        assert not report.passed

    def test_multi_label_stats(self) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001", "P002", "P003"],
                "sequence": ["MSKGEEL" * 20] * 3,
                "locations": [
                    ["Nucleus", "Cytoplasm"],
                    ["Nucleus"],
                    ["Cytoplasm"],
                ],
            }
        )
        report = validate_processed_data(df, min_samples_per_class=1)
        assert report.stats["multi_label_fraction"] > 0


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------


class TestValidateSplits:
    """Tests for split validation."""

    def test_valid_splits_pass(self) -> None:
        splits = {
            "train": pd.DataFrame({"accession": ["P001", "P002"]}),
            "val": pd.DataFrame({"accession": ["P003"]}),
            "test": pd.DataFrame({"accession": ["P004"]}),
        }
        report = validate_splits(splits)
        assert report.passed

    def test_accession_leakage_fails(self) -> None:
        splits = {
            "train": pd.DataFrame({"accession": ["P001", "P002"]}),
            "val": pd.DataFrame(
                {"accession": ["P002", "P003"]}
            ),  # P002 leaked!
            "test": pd.DataFrame({"accession": ["P004"]}),
        }
        report = validate_splits(splits)
        assert not report.passed
        assert any("leakage" in e.lower() for e in report.errors)

    def test_empty_split_fails(self) -> None:
        splits = {
            "train": pd.DataFrame({"accession": ["P001"]}),
            "val": pd.DataFrame({"accession": []}),
            "test": pd.DataFrame({"accession": ["P002"]}),
        }
        report = validate_splits(splits)
        assert not report.passed
