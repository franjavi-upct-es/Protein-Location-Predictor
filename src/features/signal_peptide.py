# src/features/signal_peptide.py
"""
Signal peptide prediction wrapper.

Wraps SignalP 6.0 as an optional external tool. When installed, predicts
whether each protein has a signal peptide (strong indicator of secretory
pathway / extracellular localization). Gracefully degrades to zero vectors
when SignalP is not available.

Usage:
    from src.features.signal_peptide import predict_signal_peptides
    features = predict_signal_peptides(sequences, cfg)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_signalp_available(binary_path: str | None = None) -> bool:
    """Check if SignalP 6.0 is installed and accessible."""
    cmd = binary_path or "signalp6"
    return shutil.which(cmd) is not None


def predict_signal_peptides(
    sequences: list[str] | pd.Series,
    cfg: DotDict | None = None,
    binary_path: str | None = None,
) -> np.ndarray:
    """
    Predict signal peptides for a batch of sequences.

    Returns a feature array with two columns per sequence:
      - has_signal_peptide (0 or 1)
      - signal_peptide_probability (0.0 to 1.0)

    If SignalP is not installed, returns zeros with a warning.

    Args:
        sequences: List or Series of amino acid sequences.
        cfg: Project configuration (optional).
        binary_path: Path to the signalp6 binary. Overrides config.

    Returns:
        NumPy array of shape (n_sequences, 2).
    """
    if cfg is not None and binary_path is None:
        sp_cfg = cfg.get("features", {}).get("signal_peptide", {})
        binary_path = sp_cfg.get("binary_path")
        if not sp_cfg.get("enabled", False):
            logger.debug("Signal peptide features disabled in config")
            return np.zeros((len(sequences), 2), dtype=np.float32)

    if not is_signalp_available(binary_path):
        logger.warning(
            "SignalP 6.0 not found. Signal peptide features will be zero. "
            "Install SignalP from https://services.healthtech.dtu.dk/services/SignalP-6.0/"
        )
        return np.zeros((len(sequences), 2), dtype=np.float32)

    cmd = binary_path or "signalp6"
    n = len(sequences)

    logger.info(f"Running SignalP 6.0 on {n} sequences...")

    try:
        with tempfile.TemporaryDirectory(prefix="signalp_") as tmpdir:
            # Write input FASTA
            fasta_path = Path(tmpdir) / "input.fasta"
            with open(fasta_path, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Run SignalP
            result = subprocess.run(
                [
                    cmd,
                    "--fasta",
                    str(fasta_path),
                    "--output_dir",
                    str(output_dir),
                    "--format",
                    "short",
                    "--organism",
                    "eukarya",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"SignalP failed: {result.stderr}")
                return np.zeros((n, 2), dtype=np.float32)

            # Parse output
            prediction_file = output_dir / "prediction_results.txt"
            if not prediction_file.exists():
                # Try alternative output naming
                for p in output_dir.iterdir():
                    if p.suffix == ".txt" and "prediction" in p.name.lower():
                        prediction_file = p
                        break

            if not prediction_file.exists():
                logger.error("SignalP output file not found")
                return np.zeros((n, 2), dtype=np.float32)

            features = np.zeros((n, 2), dtype=np.float32)
            with open(prediction_file) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    seq_id = parts[0]
                    idx = int(seq_id.replace("seq_", ""))
                    prediction = parts[1]  # SP or OTHER
                    probability = float(parts[2]) if len(parts) > 2 else 0.0

                    features[idx, 0] = 1.0 if prediction != "OTHER" else 0.0
                    features[idx, 1] = probability

            logger.info(
                f"SignalP complete: {int(features[:, 0].sum())} sequences "
                f"with signal peptides"
            )
            return features

    except Exception as e:
        logger.error(f"SignalP execution failed: {e}")
        return np.zeros((n, 2), dtype=np.float32)
