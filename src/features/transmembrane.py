# src/features/transmembrane.py
"""
Transmembrane topology prediction wrapper.

Wraps TMHMM 2.0 or DeepTMHMM as optional external tools. Predicts the
number of transmembrane helices per protein (strong indicator of membrane
localization). Gracefully degrades to zero vectors when not available.

Usage:
    from src.features.transmembrane import predict_transmembrane
    features = predict_transmembrane(sequences, cfg)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_tmhmm_available(binary_path: str | None = None) -> bool:
    """Check if TMHMM or DeepTMHMM is installed."""
    for cmd in [binary_path, "tmhmm", "deeptmhmm"]:
        if cmd and shutil.which(cmd) is not None:
            return True
    return False


def predict_transmembrane(
    sequences: list[str] | pd.Series,
    cfg: DotDict | None = None,
    binary_path: str | None = None,
) -> np.ndarray:
    """
    Predict transmembrane topology for a batch of sequences.

    Returns a feature array with three columns per sequence:
      - num_tm_helices (integer count)
      - has_tm_helix (0 or 1)
      - fraction_in_membrane (0.0 to 1.0, fraction of residues in TM segments)

    If TMHMM is not installed, returns zeros with a warning.

    Args:
        sequences: List or Series of amino acid sequences.
        cfg: Project configuration (optional).
        binary_path: Path to the TMHMM binary. Overrides config.

    Returns:
        NumPy array of shape (n_sequences, 3).
    """
    if cfg is not None and binary_path is None:
        tm_cfg = cfg.get("features", {}).get("transmembrane", {})
        binary_path = tm_cfg.get("binary_path")
        if not tm_cfg.get("enabled", False):
            logger.debug("Transmembrane features disabled in config")
            return np.zeros((len(sequences), 3), dtype=np.float32)

    if not is_tmhmm_available(binary_path):
        logger.warning(
            "TMHMM/DeepTMHMM not found. Transmembrane features will be zero. "
            "Install from https://services.healthtech.dtu.dk/services/TMHMM-2.0/"
        )
        return np.zeros((len(sequences), 3), dtype=np.float32)

    cmd = (
        binary_path
        or shutil.which("deeptmhmm")
        or shutil.which("tmhmm")
        or "tmhmm"
    )
    n = len(sequences)

    logger.info(f"Running TMHMM on {n} sequences...")

    try:
        with tempfile.TemporaryDirectory(prefix="tmhmm_") as tmpdir:
            fasta_path = Path(tmpdir) / "input.fasta"
            with open(fasta_path, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")

            result = subprocess.run(
                [cmd, str(fasta_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"TMHMM failed: {result.stderr}")
                return np.zeros((n, 3), dtype=np.float32)

            features = np.zeros((n, 3), dtype=np.float32)

            # Parse TMHMM short output format
            # Example: seq_0 len=350 ExpAA=45.5 ...
            for line in result.stdout.splitlines():
                if not line.strip() or line.startswith("#"):
                    continue

                # Extract sequence index
                id_match = re.match(r"seq_(\d+)", line)
                if not id_match:
                    continue
                idx = int(id_match.group(1))

                # Extract number of predicted helices
                hel_match = re.search(r"PredHel=(\d+)", line)
                if hel_match:
                    n_helices = int(hel_match.group(1))
                    features[idx, 0] = float(n_helices)
                    features[idx, 1] = 1.0 if n_helices > 0 else 0.0

                # Extract expected AAs in TM
                exp_match = re.search(r"ExpAA=([\d.]+)", line)
                len_match = re.search(r"len=(\d+)", line)
                if exp_match and len_match:
                    exp_aa = float(exp_match.group(1))
                    seq_len = int(len_match.group(1))
                    features[idx, 2] = exp_aa / max(seq_len, 1)

            tm_count = int((features[:, 1] > 0).sum())
            logger.info(
                f"TMHMM complete: {tm_count} sequences "
                "with transmembrane helices"
            )
            return features

    except Exception as e:
        logger.error(f"TMHMM execution failed: {e}")
        return np.zeros((n, 3), dtype=np.float32)
