# src/data/aux_targets.py
"""
Auxiliary targets for multi-task learning.

The main task is multi-label subcellular localization. As regularization
and inductive bias, we can also train the model to predict simpler
biological properties of the same sequence:

  - ``has_signal_peptide`` — binary, the protein starts with a secretion
    signal peptide
  - ``has_transmembrane`` — binary, the protein contains at least one
    transmembrane helix

These targets are derived from cheap heuristics over the raw sequence
so the project does not need SignalP / TMHMM to be installed for the
multi-task path to work. The heuristics are intentionally rough — they
are correlated with the real labels well enough to act as auxiliary
training signal but should not be used for actual predictions of those
properties.

If you want to use real SignalP / TMHMM annotations as auxiliary
targets, replace ``DEFAULT_AUX_EXTRACTORS`` with your own dict mapping
target name to a callable. The callable receives the raw sequence and
must return a float in ``[0, 1]``.

Usage::

    from src.data.aux_targets import compute_aux_targets, AUX_TARGET_NAMES
    aux = compute_aux_targets("MSKGEEL...")
    # -> {"has_signal_peptide": 1.0, "has_transmembrane": 0.0}
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _heuristic_signal_peptide(sequence: str) -> float:
    """
    Cheap heuristic for "has signal peptide".

    Real signal peptides:
      - Are 15–30 residues long, at the N-terminus
      - Have 1–5 positively charged residues at the very start (n-region)
      - Followed by a hydrophobic stretch (h-region) of ~7–15 residues
      - Followed by a more polar c-region with a cleavage site

    We approximate by checking the hydrophobicity of residues 1–25:
    if the average hydrophobicity (Kyte-Doolittle) is high, return 1.

    The heuristic is not great in absolute terms (~75% precision), but
    it is fast, deterministic, and correlated enough with the real
    label to be useful as auxiliary training signal.
    """
    if not sequence or len(sequence) < 25:
        return 0.0

    # Kyte-Doolittle hydrophobicity scale (positive = hydrophobic)
    kd = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }

    n_terminal = sequence[1:25].upper()  # skip the start methionine
    scores = [kd.get(aa, 0.0) for aa in n_terminal]
    if not scores:
        return 0.0

    avg_hydrophobicity = float(np.mean(scores))
    # Empirical threshold: signal peptides usually score above 1.0
    return 1.0 if avg_hydrophobicity > 1.0 else 0.0


def _heuristic_transmembrane(sequence: str) -> float:
    """
    Cheap heuristic for "has transmembrane helix".

    A transmembrane helix is typically a stretch of ~20 consecutive
    hydrophobic residues. We slide a window of 19 residues over the
    sequence and return 1 if any window has high mean hydrophobicity.
    """
    if not sequence or len(sequence) < 19:
        return 0.0

    kd = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }

    upper = sequence.upper()
    window = 19
    threshold = 1.6  # empirically separates TM from non-TM

    scores = np.array([kd.get(aa, 0.0) for aa in upper], dtype=np.float32)
    if len(scores) < window:
        return 0.0

    # Rolling mean via cumulative sum (O(L), no Python loops)
    csum = np.cumsum(np.insert(scores, 0, 0.0))
    rolling = (csum[window:] - csum[:-window]) / window

    return 1.0 if rolling.max() > threshold else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Order matters: this is also the order of the auxiliary head outputs
AUX_TARGET_NAMES: tuple[str, ...] = (
    "has_signal_peptide",
    "has_transmembrane",
)


# Default extractor map. Override at runtime by passing a custom dict.
DEFAULT_AUX_EXTRACTORS: dict[str, Callable[[str], float]] = {
    "has_signal_peptide": _heuristic_signal_peptide,
    "has_transmembrane": _heuristic_transmembrane,
}


def compute_aux_targets(
    sequence: str,
    extractors: dict[str, Callable[[str], float]] | None = None,
) -> dict[str, float]:
    """
    Compute every configured auxiliary target for a single sequence.

    Args:
        sequence: Raw amino acid sequence.
        extractors: Override the default extractor map.

    Returns:
        Dict ``{target_name: value_in_[0,1]}``.
    """
    extractors = extractors or DEFAULT_AUX_EXTRACTORS
    return {name: float(fn(sequence)) for name, fn in extractors.items()}


def aux_targets_to_tensor(
    sequences: list[str],
    extractors: dict[str, Callable[[str], float]] | None = None,
) -> np.ndarray:
    """
    Compute auxiliary targets for a batch of sequences.

    Returns:
        Float32 array of shape ``(n_sequences, len(AUX_TARGET_NAMES))``.
    """
    extractors = extractors or DEFAULT_AUX_EXTRACTORS
    n = len(sequences)
    out = np.zeros((n, len(AUX_TARGET_NAMES)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, name in enumerate(AUX_TARGET_NAMES):
            if name in extractors:
                out[i, j] = float(extractors[name](seq))
    return out
