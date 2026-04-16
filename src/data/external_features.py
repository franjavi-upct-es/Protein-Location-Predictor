# src/data/external_features.py
"""
External biological feature extraction orchestrator.

Combines biophysical properties, signal peptide predictions, and
transmembrane topology predictions into a single feature matrix
that can be concatenated with ESM-2 embeddings in the classifier head.

Gracefully degrades: any unavailable external tool produces zero
columns instead of crashing the pipeline.

Usage::

    from src.data.external_features import compute_all_external_features
    features = compute_all_external_features(sequences, cfg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_all_external_features(
    sequences: list[str] | pd.Series,
    cfg: DotDict,
) -> np.ndarray:
    """
    Compute all enabled external features and concatenate them.

    Features are concatenated in a fixed order:
      1. Biophysical properties (molecular weight, pI, GRAVY, etc.)
      2. Signal peptide predictions (has_sp, sp_probability)
      3. Transmembrane predictions (n_helices, has_tm, fraction_in_membrane)

    Disabled or unavailable features contribute zero-width columns,
    so the total feature dimension is deterministic given the config.

    Args:
        sequences: List or Series of amino acid sequences.
        cfg: Project configuration.

    Returns:
        NumPy array of shape (n_sequences, total_feature_dim).
        Returns an empty array of shape (n_sequences, 0) if all
        features are disabled.
    """
    features_cfg = cfg.get("features", {})
    parts: list[np.ndarray] = []

    # 1. Biophysical properties
    bio_cfg = features_cfg.get("biophysical", {})
    if bio_cfg.get("enabled", False):
        from src.features.biophysical import compute_biophysical_features

        bio_features = compute_biophysical_features(sequences, cfg)
        parts.append(bio_features)
        logger.info(
            f"Biophysical features: {bio_features.shape[1]} dimensions"
        )

    # 2. Signal peptide predictions
    sp_cfg = features_cfg.get("signal_peptide", {})
    if sp_cfg.get("enabled", False):
        from src.features.signal_peptide import predict_signal_peptides

        sp_features = predict_signal_peptides(sequences, cfg)
        parts.append(sp_features)
        logger.info(
            f"Signal peptide features: {sp_features.shape[1]} dimensions"
        )

    # 3. Transmembrane predictions
    tm_cfg = features_cfg.get("transmembrane", {})
    if tm_cfg.get("enabled", False):
        from src.features.transmembrane import predict_transmembrane

        tm_features = predict_transmembrane(sequences, cfg)
        parts.append(tm_features)
        logger.info(
            f"Transmembrane features: {tm_features.shape[1]} dimensions"
        )

    # Concatenate all parts
    if not parts:
        logger.info("No external features enabled")
        return np.empty((len(sequences), 0), dtype=np.float32)

    combined = np.concatenate(parts, axis=1)
    logger.info(
        f"Total external features: {combined.shape[1]} dimensions "
        f"for {combined.shape[0]} sequences"
    )

    return combined


def get_external_feature_dim(cfg: DotDict) -> int:
    """
    Calculate the total external feature dimension from config.

    This is useful for building the classifier head with the correct
    input dimension without needing to compute the actual features.

    Args:
        cfg: Project configuration.

    Returns:
        Total number of external feature dimensions.
    """
    features_cfg = cfg.get("features", {})
    dim = 0

    bio_cfg = features_cfg.get("biophysical", {})
    if bio_cfg.get("enabled", False):
        dim += len(bio_cfg.get("properties", []))

    sp_cfg = features_cfg.get("signal_peptide", {})
    if sp_cfg.get("enabled", False):
        dim += 2  # has_signal_peptide, sp_probability

    tm_cfg = features_cfg.get("transmembrane", {})
    if tm_cfg.get("enabled", False):
        dim += 3  # n_helices, has_tm, fraction_in_membrane

    return dim
