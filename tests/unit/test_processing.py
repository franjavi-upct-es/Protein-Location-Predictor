# tests/unit/test_processing.py
"""Tests for data processing and multi-label annotation."""

from __future__ import annotations

from src.data.processing import (
    _is_topology_annotation,
    map_to_canonical,
    parse_location_string,
    validate_sequence,
)

# ---------------------------------------------------------------------------
# Location string parsing
# ---------------------------------------------------------------------------


class TestParseLocationString:
    """Tests for UniProt location string parsing."""

    def test_single_location(self) -> None:
        result = parse_location_string("SUBCELLULAR LOCATION: Nucleus")
        assert "Nucleus" in result

    def test_multiple_locations_period_separated(self) -> None:
        result = parse_location_string(
            "SUBCELLULAR LOCATION: Nucleus. Cytoplasm."
        )
        assert "Nucleus" in result
        assert "Cytoplasm" in result

    def test_removes_evidence_codes(self) -> None:
        result = parse_location_string(
            "SUBCELLULAR LOCATION: Nucleus {ECO:0000269|PubMed:12345}"
        )
        assert len(result) >= 1
        assert "{" not in result[0]
        assert "ECO" not in result[0]

    def test_handles_semicolon_sub_annotations(self) -> None:
        result = parse_location_string(
            "SUBCELLULAR LOCATION: Cell membrane; Peripheral membrane protein"
        )
        # "Peripheral membrane protein" is a topology
        # annotation, should be filtered
        assert "Cell membrane" in result

    def test_filters_topology_annotations(self) -> None:
        result = parse_location_string(
            "SUBCELLULAR LOCATION: Membrane; "
            "Single-pass type I membrane protein"
        )
        assert "Membrane" in result
        # "Single-pass type I membrane protein" should be filtered
        assert not any("single-pass" in r.lower() for r in result)

    def test_truncates_at_note(self) -> None:
        result = parse_location_string(
            "SUBCELLULAR LOCATION: Cytoplasm. Note=Also found in nucleus."
        )
        assert "Cytoplasm" in result
        # The Note text should not be parsed as a location
        assert not any("also found" in r.lower() for r in result)

    def test_empty_string(self) -> None:
        assert parse_location_string("") == []

    def test_none_input(self) -> None:
        assert parse_location_string(None) == []

    def test_non_string_input(self) -> None:
        assert parse_location_string(42) == []


# ---------------------------------------------------------------------------
# Topology filtering
# ---------------------------------------------------------------------------


class TestIsTopologyAnnotation:
    """Tests for topology annotation detection."""

    def test_single_pass(self) -> None:
        assert _is_topology_annotation("Single-pass type I membrane protein")

    def test_multi_pass(self) -> None:
        assert _is_topology_annotation("Multi-pass membrane protein")

    def test_peripheral(self) -> None:
        assert _is_topology_annotation("Peripheral membrane protein")

    def test_not_topology(self) -> None:
        assert not _is_topology_annotation("Nucleus")
        assert not _is_topology_annotation("Cytoplasm")
        assert not _is_topology_annotation("Endoplasmic reticulum membrane")


# ---------------------------------------------------------------------------
# Canonical mapping
# ---------------------------------------------------------------------------


SAMPLE_GROUPS = {
    "Nucleus": ["nucleus", "nucleolus"],
    "Cytoplasm": ["cytoplasm", "cytosol"],
    "Membrane": ["membrane", "cell membrane"],
    "Mitochondrion": ["mitochondrion", "mitochondria"],
}


class TestMapToCanonical:
    """Tests for hierarchical location mapping."""

    def test_single_match(self) -> None:
        result = map_to_canonical(["Nucleus"], SAMPLE_GROUPS)
        assert result == ["Nucleus"]

    def test_multiple_matches(self) -> None:
        result = map_to_canonical(["Nucleus", "Cytoplasm"], SAMPLE_GROUPS)
        assert "Nucleus" in result
        assert "Cytoplasm" in result

    def test_keyword_matching(self) -> None:
        result = map_to_canonical(["Nucleolus"], SAMPLE_GROUPS)
        assert result == ["Nucleus"]

    def test_case_insensitive(self) -> None:
        result = map_to_canonical(["MITOCHONDRION"], SAMPLE_GROUPS)
        assert result == ["Mitochondrion"]

    def test_unmatched_returns_empty(self) -> None:
        result = map_to_canonical(["Unknown organelle"], SAMPLE_GROUPS)
        assert result == []

    def test_deduplication(self) -> None:
        result = map_to_canonical(["Cytoplasm", "Cytosol"], SAMPLE_GROUPS)
        assert result == ["Cytoplasm"]

    def test_empty_input(self) -> None:
        assert map_to_canonical([], SAMPLE_GROUPS) == []


# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------


class TestValidateSequence:
    """Tests for protein sequence validation."""

    def test_valid_sequence(self) -> None:
        assert validate_sequence("MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG")

    def test_too_short(self) -> None:
        assert not validate_sequence("MSK", min_length=30)

    def test_too_long(self) -> None:
        assert not validate_sequence("M" * 3000, max_length=2048)

    def test_invalid_characters(self) -> None:
        assert not validate_sequence("MSKGEELFTG123VVPILVELDGDVNGHKFSVR")

    def test_allows_x_unknown(self) -> None:
        assert validate_sequence("M" * 30 + "X" + "A" * 10)

    def test_allows_u_selenocysteine(self) -> None:
        assert validate_sequence("M" * 30 + "U" + "A" * 10)

    def test_none_input(self) -> None:
        assert not validate_sequence(None)

    def test_empty_string(self) -> None:
        assert not validate_sequence("")

    def test_exact_min_length(self) -> None:
        assert validate_sequence("M" * 30, min_length=30)

    def test_exact_max_length(self) -> None:
        assert validate_sequence("M" * 2048, max_length=2048)
