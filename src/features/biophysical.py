# src/features/biophysical.py
"""
Biophysical property computation from amino acid sequences.

Computes sequence-level physicochemical properties that are directly
relevant to subcellular localization (e.g., hydrophobic proteins tend
toward membranes, highly charged proteins toward the nucleus).

Uses BioPython's ProteinAnalysis for the heavy lifting.

Usage::

    from src.features.biophysical import compute_biophysical_features
    features = compute_biophysical_features(sequences, cfg)
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Amino acids that ProteinAnalysis can handle
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_sequence(sequence: str) -> str:
    """Remove non-standard amino acids for BioPython compatibility."""
    return "".join(c for c in sequence.upper() if c in _STANDARD_AA)


def compute_single_sequence(
    sequence: str, properties: list[str]
) -> dict[str, float]:
    """
    Compute biophysical properties for a single protein sequence.

    Args:
        sequence: Amino acid sequence string.
        properties: List of property names to compute.

    Returns:
        Dict mapping property names to computed float values.
        Returns NaN for any property that fails to compute.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    cleaned = _clean_sequence(sequence)
    if len(cleaned) < 5:
        return {prop: float("nan") for prop in properties}

    try:
        analysis = ProteinAnalysis(cleaned)
    except Exception:
        return {prop: float("nan") for prop in properties}

    result: dict[str, float] = {}

    for prop in properties:
        try:
            if prop == "molecular_weight":
                result[prop] = analysis.molecular_weight()
            elif prop == "isoelectric_point":
                result[prop] = analysis.isoelectric_point()
            elif prop == "gravy":
                result[prop] = analysis.gravy()
            elif prop == "aromaticity":
                result[prop] = analysis.aromaticity()
            elif prop == "instability_index":
                result[prop] = analysis.instability_index()
            elif prop == "charge_at_ph7":
                result[prop] = analysis.charge_at_pH(7.0)
            elif prop == "helix_fraction":
                helix, turn, sheet = analysis.secondary_structure_fraction()
                result[prop] = helix
            elif prop == "turn_fraction":
                helix, turn, sheet = analysis.secondary_structure_fraction()
                result[prop] = turn
            elif prop == "sheet_fraction":
                helix, turn, sheet = analysis.secondary_structure_fraction()
                result[prop] = sheet
            elif prop == "sequence_length":
                result[prop] = float(len(cleaned))
            else:
                logger.warning(f"Unknown biophysical property: '{prop}'")
                result[prop] = float("nan")
        except Exception as e:
            logger.debug(f"Failed to compute {prop}: {e}")
            result[prop] = float("nan")

    return result


def compute_biophysical_features(
    sequences: list[str] | pd.Series,
    cfg: DotDict | None = None,
    properties: list[str] | None = None,
) -> npt.NDArray[np.float32]:
    """
    Compute biophysical features for a batch of sequences.

    Args:
        sequences: List or Series of amino acid sequence strings.
        cfg: Project configuration. If provided, reads the property list
             from features.biophysical.properties
        properties: Explicit list of properties to compute. Overrides config.

    Returns:
        NumPy array of shape (n_sequences, n_properties) with computed
        features values. NaN values are replaced with column means.
    """
    if properties is None:
        if cfg is not None:
            feat_cfg = cfg.get("features", {}).get("biophysical", {})
            properties = list(
                feat_cfg.get(
                    "properties",
                    [
                        "molecular_weight",
                        "isoelectric_point",
                        "gravy",
                        "aromaticity",
                        "instability_index",
                    ],
                )
            )
        else:
            properties = [
                "molecular_weight",
                "isoelectric_point",
                "gravy",
                "aromaticity",
                "instability_index",
            ]

    if not properties:
        return np.empty((len(sequences), 0), dtype=np.float32)

    logger.info(
        f"Computing {len(properties)} biophysical "
        f"features for {len(sequences)} sequences"
    )

    rows = []
    for i, seq in enumerate(sequences):
        rows.append(compute_single_sequence(str(seq), properties))
        if (i + 1) % 1000 == 0:
            logger.debug(
                f"  Computed features for {i + 1}/{len(sequences)} sequences"
            )

    df = pd.DataFrame(rows, columns=pd.Index(properties))

    # Replace NaN with column means (robust to occasional failures)
    for col in df.columns:
        col_mean = df[col].mean()
        if np.isnan(col_mean):
            col_mean = 0.0
        df[col] = df[col].fillna(col_mean)

    # Standardize (zero mean, unit variance) for model compatibility
    result = cast(npt.NDArray[np.float32], df.to_numpy(dtype=np.float32))
    means = result.mean(axis=0, keepdims=True)
    stds = result.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    result = (result - means) / stds

    logger.info(f"Biophysical features computed: shape {result.shape}")
    return cast(npt.NDArray[np.float32], result)
