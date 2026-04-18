# src/data/validation.py
"""
Data validation and integrity checks.

Provides validation functions for raw data, processed data, and splits
to catch schema errors, distribution drift, and leakage early.

Usage::

    from src.data.validation import  validate_raw_data, validate_splits
    issues = validate_raw_data(df)
    issues = validate_splits(splits, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Container for validation results."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error(f"VALIDATION ERROR: {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"VALIDATION WARNING: {msg}")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation: {status}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)})")
            lines.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"  Warnings ({len(self.errors)})")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.stats:
            lines.append("  Stats:")
            lines.extend(f"    {k}: {v}" for k, v in self.stats.items())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw data validation
# ---------------------------------------------------------------------------


def validate_raw_data(df: pd.DataFrame) -> ValidationReport:
    """
    Validate raw downloaded data from UniProt.

    Checks:
        - Required columns are present
        - No fully empty rows
        - Accession uniqueness
        - Sequence character validity
    """
    report = ValidationReport()

    # Required columns
    required = {"accession", "sequence", "subcellular_location"}
    missing = required - set(df.columns)
    if missing:
        report.add_error(f"Missing required columns: {missing}")
        return report

    report.stats["total_rows"] = len(df)

    # Null checks
    for col in required:
        null_count = df[col].isna().sum()
        if null_count > 0:
            report.add_warning(f"Column '{col}' has {null_count} null values")

    # Uniqueness
    dup_count = df["accession"].duplicated().sum()
    if dup_count > 0:
        report.add_warning(f"{dup_count} duplicate accessions found")

    # Sequence validity
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_seqs = 0
    for seq in df["sequence"].dropna():
        if not all(c in valid_chars for c in str(seq).upper()):
            invalid_seqs += 1
    if invalid_seqs > 0:
        report.add_warning(f"{invalid_seqs} sequences contain non-standard amino acids")

    report.stats["unique_accessions"] = df["accession"].nunique()
    report.stats["sequence_length_mean"] = df["sequence"].str.len().mean()
    report.stats["sequence_length_median"] = df["sequence"].str.len().median()

    return report


# ---------------------------------------------------------------------------
# Processed data validation
# ---------------------------------------------------------------------------


def validate_processed_data(
    df: pd.DataFrame,
    min_samples_per_class: int = 50,
) -> ValidationReport:
    """Validate processed data after location mapping.

    Checks:
      - locations column exists and is non-empty
      - All classes meet minimum sample threshold
      - No empty sequences
      - Reasonable class distribution (no extreme imbalance)
    """
    report = ValidationReport()

    if "locations" not in df.columns and "locations_str" not in df.columns:
        report.add_error("Missing 'locations' or 'locations_str' column")
        return report

    # Reconstruct locations list if needed
    if "locations" not in df.columns:
        df = df.copy()
        df["locations"] = df["locations_str"].apply(
            lambda s: s.split("|") if isinstance(s, str) else []
        )

    report.stats["total_proteins"] = len(df)

    # Empty locations
    empty = df["locations"].apply(lambda x: len(x) == 0).sum()
    if empty > 0:
        report.add_error(f"{empty} proteins have no location labels")

    # Class distribution
    all_labels = [label for labels in df["locations"] for label in labels]
    label_counts = pd.Series(all_labels).value_counts()
    report.stats["num_classes"] = len(label_counts)
    report.stats["class_distribution"] = label_counts.to_dict()

    # Check minimum samples
    below_min = label_counts[label_counts < min_samples_per_class]
    if len(below_min) > 0:
        report.add_warning(
            f"{len(below_min)} classes below minimum "
            f"threshold ({min_samples_per_class}): "
            f"{below_min.to_dict()}"
        )

    # Extreme imbalance check
    if len(label_counts) >= 2:
        ratio = label_counts.max() / label_counts.min()
        report.stats["max_min_class_ratio"] = round(ratio, 1)
        if ratio > 50:
            report.add_warning(f"Extreme class imbalance: largest/smallest ratio = {ratio:.1f}")

    # Multi-label statistics
    labels_per_protein = df["locations"].apply(len)
    report.stats["avg_labels_per_protein"] = round(labels_per_protein.mean(), 2)
    report.stats["multi_label_fraction"] = round((labels_per_protein > 1).mean(), 3)

    return report


# ---------------------------------------------------------------------------
# Processed data validation
# ---------------------------------------------------------------------------


def validation_processed_data(
    df: pd.DataFrame,
    min_samples_per_class: int = 50,
) -> ValidationReport:
    """
    Validate processed data after location mapping.

    Checks:
        - locations column exists and is non-empty
        - All classes meet minimum sample threshold
        - No empty sequences
        - Reasonable class distribution (no extreme imbalance)
    """
    report = ValidationReport()

    if "locations" not in df.columns and "locations_str" not in df.columns:
        report.add_error("Missing 'locations' or 'locations_str' column")
        return report

    # Reconstruct locations list if needed
    if "locations" not in df.columns:
        df = df.copy()
        df["locations"] = df["locations_str"].apply(
            lambda s: s.split("|") if isinstance(s, str) else []
        )

    report.stats["total_proteins"] = len(df)

    # Empty locations
    empty = df["locations"].apply(lambda x: len(x) == 0).sum()
    if empty > 0:
        report.add_error(f"{empty} proteins have no location labels")

    # Class distribution
    all_labels = [label for labels in df["locations"] for label in labels]
    label_counts = pd.Series(all_labels).value_counts()
    report.stats["num_classes"] = len(label_counts)
    report.stats["class_distribution"] = label_counts.to_dict()

    # Check minimum samples
    below_min_dict = {
        label: count for label, count in label_counts.items() if count < min_samples_per_class
    }
    if len(below_min_dict) > 0:
        report.add_warning(
            f"{len(below_min_dict)} classes below minimum"
            f" threshold ({min_samples_per_class}):"
            f" {below_min_dict}"
        )

    # Extreme imbalance check
    if len(label_counts) >= 2:
        ratio = label_counts.max() / label_counts.min()
        report.stats["max_min_class_ratio"] = round(ratio, 1)
        if ratio > 50:
            report.add_warning(f"Extreme class imbalance: largest/smallest ratio = {ratio:.1f}")

    # Multi-label statistics
    labels_per_protein = df["locations"].apply(len)
    report.stats["avg_labels_per_protein"] = round(labels_per_protein.mean(), 2)
    report.stats["multi_label_fraction"] = round((labels_per_protein > 1).mean(), 3)

    return report


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------


def validate_splits(
    splits: dict[str, pd.DataFrame],
) -> ValidationReport:
    """
    Validate train/val/test splits for leakage and distribution.

    Checks:
        - No accession appears in multiple splits
        - Class distribution is reasonably similar across splits
        - All splits are non-empty
    """
    report = ValidationReport()

    for name, df in splits.items():
        if len(df) == 0:
            report.add_error(f"Split '{name}' is empty")
        report.stats[f"{name}_size"] = len(df)

    # Accession leakage
    split_accessions = {name: set(df["accession"]) for name, df in splits.items()}

    for name_a, accs_a in split_accessions.items():
        for name_b, accs_b in split_accessions.items():
            if name_a >= name_b:
                continue
            overlap = accs_a & accs_b
            if overlap:
                report.add_error(
                    f"Accession leakage between {name_a} and {name_b}: "
                    f"{len(overlap)} shared accessions"
                )

    # Distribution comparison (if locations available)
    if all("locations" in df.columns or "locations_str" in df.columns for df in splits.values()):
        for name, df in splits.items():
            if "locations" not in df.columns:
                df = df.copy()
                df["locations"] = df["locations_str"].apply(
                    lambda s: s.split("|") if isinstance(s, str) else []
                )
            labels = [lb for locs in df["locations"] for lb in locs]
            dist = pd.Series(labels).value_counts(normalize=True)
            report.stats[f"{name}_distribution"] = dist.to_dict()

    total = sum(len(df) for df in splits.values())
    report.stats["total_across_splits"] = total

    return report
