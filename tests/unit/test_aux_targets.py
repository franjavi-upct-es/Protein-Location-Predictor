# tests/unit/test_aux_targets.py
"""Tests for the auxiliary target extractors."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.aux_targets import (
    AUX_TARGET_NAMES,
    aux_targets_to_tensor,
    compute_aux_targets,
)


class TestComputeAuxTargets:
    def test_returns_all_names(self) -> None:
        result = compute_aux_targets("MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG")
        assert set(result.keys()) == set(AUX_TARGET_NAMES)

    def test_values_in_unit_interval(self) -> None:
        result = compute_aux_targets("MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG")
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_empty_sequence_returns_zeros(self) -> None:
        result = compute_aux_targets("")
        for v in result.values():
            assert v == 0.0

    def test_short_sequence_returns_zeros(self) -> None:
        # Below the minimum length the heuristics need
        result = compute_aux_targets("MSK")
        for v in result.values():
            assert v == 0.0


class TestSignalPeptideHeuristic:
    """The heuristic should correlate with hydrophobic N-termini."""

    def test_hydrophobic_n_terminus_returns_one(self) -> None:
        # All hydrophobic residues at the start (Leu = 3.8)
        seq = "M" + "L" * 30 + "S" * 50
        result = compute_aux_targets(seq)
        assert result["has_signal_peptide"] == 1.0

    def test_hydrophilic_n_terminus_returns_zero(self) -> None:
        # All charged residues at the start (Lys = -3.9)
        seq = "M" + "K" * 30 + "S" * 50
        result = compute_aux_targets(seq)
        assert result["has_signal_peptide"] == 0.0


class TestTransmembraneHeuristic:
    def test_internal_hydrophobic_window_returns_one(self) -> None:
        # 50 polar residues + 25 hydrophobic + 50 polar
        seq = "S" * 50 + "I" * 25 + "S" * 50
        result = compute_aux_targets(seq)
        assert result["has_transmembrane"] == 1.0

    def test_no_hydrophobic_window_returns_zero(self) -> None:
        seq = "M" + "S" * 100  # all serine — never above threshold
        result = compute_aux_targets(seq)
        assert result["has_transmembrane"] == 0.0


class TestAuxTargetsToTensor:
    def test_shape(self) -> None:
        sequences = [
            "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDK",
        ]
        out = aux_targets_to_tensor(sequences)
        assert out.shape == (2, len(AUX_TARGET_NAMES))
        assert out.dtype == np.float32

    def test_empty_list(self) -> None:
        out = aux_targets_to_tensor([])
        assert out.shape == (0, len(AUX_TARGET_NAMES))

    def test_values_match_per_sequence_call(self) -> None:
        sequences = ["M" + "L" * 30 + "S" * 50, "M" + "K" * 30 + "S" * 50]
        out = aux_targets_to_tensor(sequences)

        for i, seq in enumerate(sequences):
            ref = compute_aux_targets(seq)
            for j, name in enumerate(AUX_TARGET_NAMES):
                assert out[i, j] == pytest.approx(ref[name])
