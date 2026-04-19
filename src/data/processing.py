# src/data/processing.py
"""
Data cleaning and multi-label location annotation.

Parses UniProt subcellular location strings into structured multi-label
annotations, maps them to canonical location classes via configurable
hierarchical grouping, and filters by sequence validity and class frequency.

Usage::

    from src.data.processing import process_data
    df = process_data(cfg)
"""

from __future__ import annotations

import re

import pandas as pd

from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Location parsing
# ---------------------------------------------------------------------------


def parse_location_string(location_string: str | float) -> list[str]:
    """
    Parse a UniProt subcellular location string into a list of raw locations.

    UniProt encodes multiple locations in a single field, separated by periods
    and annotated with evidence codes in curly braces. This function extracts
    all distinct location mentions.

    Args:
        location_string: Raw UniProt 'Subcellular location [CC]' field.

    Returns:
        List of cleaned location strings, or empty list if unparseable.

    Examples::

        >>> parse_location_string(
        ...     "SUBCELLULAR LOCATION: Nucleus {ECO:0000269}. Cytoplasm."
        ... )
        ['Nucleus', 'Cytoplasm']
        >>> parse_location_string(
        ...     "SUBCELLULAR LOCATION: Cell membrane; "
        ...     "Multi-pass membrane protein."
        ... )
        ['Cell membrane', 'Multi-pass membrane protein']
    """
    if not isinstance(location_string, str):
        return []

    # Remove the common prefix
    text = location_string.replace("SUBCELLULAR LOCATION: ", "")

    # Split on periods (separate entries) and semicolons
    # but handle "Note:" sections by truncating there
    if "Note=" in text:
        text = text[: text.index("Note=")]

    # Split on periods first (major location entries)
    segments = text.split(".")

    locations = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Split on semicolons for sub-annotations within one entry
        parts = segment.split(";")
        for part in parts:
            # Remove evidence codes {ECO:...}
            cleaned = re.sub(r"\s*\{[^}]*\}", "", part).strip()
            # Remove TOPOLOGY annotations like "Multi-pass membrane protein"
            # These describe topology, not location
            if cleaned and not _is_topology_annotation(cleaned):
                locations.append(cleaned)

    return locations


def _is_topology_annotation(text: str) -> bool:
    """Check if text is a topology annotation, not a location."""
    topology_patterns = [
        "single-pass",
        "multi-pass",
        "peripheral",
        "lipid-anchor",
        "gpi-anchor",
        "type i membrane",
        "type ii membrane",
        "type iii membrane",
        "type iv membrane",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in topology_patterns)


# ---------------------------------------------------------------------------
# Hierarchical location mapping
# ---------------------------------------------------------------------------


def map_to_canonical(
    raw_locations: list[str],
    location_groups: dict[str, list[str]],
) -> list[str]:
    """
    Map raw location strings to canonical classes using hierarchical grouping.

    Args:
        raw_locations: List of raw location strings from
            parse_location_string().
        location_groups: Mapping from canonical name to list of
            keyword patterns (from config).

    Returns:
        Deduplicated list of canonical location classes. Empty locations are
        mapped to nothing (dropped).
    """
    canonical: set[str] = set()

    for raw in raw_locations:
        raw_lower = raw.lower()
        matched = False

        for group_name, keywords in location_groups.items():
            for keyword in keywords:
                if keyword.lower() in raw_lower:
                    canonical.add(group_name)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            logger.debug(f"Unmapped location: '{raw}'")

    return sorted(canonical)


# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------


def validate_sequence(
    sequence: str | float,
    valid_aa: str = "ACDEFGHIKLMNPQRSTVWY",
    min_length: int = 30,
    max_length: int = 2048,
) -> bool:
    """
    Check if a protein sequence is valid.

    Args:
        sequence: Amino acid sequence string.
        valid_aa: Allowed amino acid alphabet.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.

    Returns:
        True if the sequence passes all checks.
    """
    if not isinstance(sequence, str):
        return False
    if len(sequence) < min_length or len(sequence) > max_length:
        return False
    # Allow X (unknown) and U (selenocysteine) but not other non-standard chars
    allowed = set(valid_aa + "XU")
    return all(c in allowed for c in sequence.upper())


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def process_data(
    cfg: DotDict,
    input_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run the full data processing pipeline.

    Steps:
        1. Load raw data (or use provided DataFrame)
        2. Parse multi-label location annotations
        3. Map to canonical location classes
        4. Validate sequences
        5. Filter by minimum class frequency
        6. Save processed data

    Args:
        cfg: Project configuration.
        input_df: Optional pre-loaded DataFrame. If None, reads from disk.

    Returns:
        Processed DataFrame with columns:
          - accession, sequence, organism_id, organism_name
          - subcellular_location (original string)
          - locations (list of canonical location labels)
          - locations_str (pipe-separated string for CSV storage)
    """
    proc_cfg = cfg.processing

    # 1. Load data
    if input_df is not None:
        df: pd.DataFrame = input_df.copy()
    else:
        raw_path = resolve_path(cfg, "paths.raw_dir") / "uniprot_data.csv"
        logger.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)

    initial_count = len(df)
    logger.info(f"Starting processing with {initial_count} proteins")

    # 2. Drop rows missing essential columns
    df.dropna(subset=["sequence", "subcellular_location"], inplace=True)
    logger.info(f"After dropping missing values: {len(df)} proteins")

    # 3. Parse multi-label locations
    location_groups = proc_cfg.location_groups
    # Convert DotDict to plain dict for the mapping
    if hasattr(location_groups, "items"):
        location_groups = {k: list(v) for k, v in location_groups.items()}

    multi_label = proc_cfg.get("multi_label", True)

    df["parsed_locations"] = df["subcellular_location"].apply(parse_location_string)
    df["locations"] = df["parsed_locations"].apply(
        lambda locs: map_to_canonical(locs, location_groups)
    )

    # Drop proteins with no mapped locations
    df = df[df["locations"].apply(len) > 0].copy()  # pyright: ignore[reportAssignmentType]
    logger.info(f"After location mapping: {len(df)} proteins with valid locations")

    if not multi_label:
        # Single-label mode: keep only the first location
        df["locations"] = df["locations"].apply(lambda x: [x[0]] if x else [])

    # 4. Validate sequences
    valid_aa = proc_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")
    min_len = cfg.data.get("min_sequence_length", 30)
    max_len = cfg.data.get("max_sequence_length", 2048)

    valid_mask = df["sequence"].apply(
        lambda seq: validate_sequence(seq, valid_aa, min_len, max_len)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Filtered {invalid_count} proteins with invalid sequences")
    df = df[valid_mask].copy()  # pyright: ignore[reportAssignmentType]

    # 5. Filter by minimum class frequency
    min_samples = proc_cfg.get("min_samples_per_class", 50)

    # Count occurrences of each label across all proteins (multi-label aware)
    all_labels = [label for labels in df["locations"] for label in labels]
    label_counts = pd.Series(all_labels).value_counts()
    logger.info(f"Label distribution:\n{label_counts.to_string()}")

    valid_labels = set(label_counts[label_counts >= min_samples].index)  # pyright: ignore[reportAttributeAccessIssue]
    logger.info(
        f"Keeping {len(valid_labels)} classes with >= {min_samples} samples: {sorted(valid_labels)}"
    )

    # Filter: keep proteins that have at least one valid label
    df["locations"] = df["locations"].apply(
        lambda locs: [loc for loc in locs if loc in valid_labels]
    )
    df = df[df["locations"].apply(len) > 0].copy()  # pyright: ignore[reportAssignmentType]

    # 6. Create serializable column for CSV storage
    df["locations_str"] = df["locations"].apply(lambda x: "|".join(x))

    # Keep only needed columns
    keep_cols = [
        "accession",
        "sequence",
        "subcellular_location",
        "locations",
        "locations_str",
    ]
    if "organism_id" in df.columns:
        keep_cols.append("organism_id")
    if "organism_name" in df.columns:
        keep_cols.append("organism_name")
    df = df[keep_cols].reset_index(drop=True)  # pyright: ignore[reportAssignmentType]

    logger.info(f"Processing complete: {len(df)} proteins, {len(valid_labels)} location classes")

    # 7. Save processed data
    proc_dir = resolve_path(cfg, "paths.processed_dir")
    proc_dir.mkdir(parents=True, exist_ok=True)
    output_path = proc_dir / "proteins_processed.csv"

    # Save without the list column (not CSV-friendly)
    save_df = df.drop(columns=["locations"])
    save_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for data processing."""
    cfg = load_config()
    setup_logging(level=cfg.project.log_level)
    process_data(cfg)


if __name__ == "__main__":
    main()
