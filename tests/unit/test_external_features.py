# tests/unit/test_external_features.py
"""Tests for external feature wrappers
(SignalP, TMHMM) graceful degradation."""

from __future__ import annotations

import numpy as np

from src.features.signal_peptide import (
    is_signalp_available,
    predict_signal_peptides,
)
from src.features.transmembrane import (
    is_tmhmm_available,
    predict_transmembrane,
)

SAMPLE_SEQUENCES = [
    "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKE",
    "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRL",
]


class TestSignalPeptide:
    """Tests for SignalP wrapper."""

    def test_availability_returns_bool(self) -> None:
        result = is_signalp_available()
        assert isinstance(result, bool)

    def test_graceful_degradation(self) -> None:
        """When SignalP is not installed, should return zeros."""
        result = predict_signal_peptides(SAMPLE_SEQUENCES)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        # If SignalP is not installed, all values should be zero
        if not is_signalp_available():
            assert np.all(result == 0)

    def test_disabled_in_config(self) -> None:
        """When disabled in config, should return zeros immediately."""
        from src.utils.config import DotDict

        cfg = DotDict.from_dict({"features": {"signal_peptide": {"enabled": False}}})
        result = predict_signal_peptides(SAMPLE_SEQUENCES, cfg=cfg)
        assert result.shape == (3, 2)
        assert np.all(result == 0)


class TestTransmembrane:
    """Tests for TMHMM wrapper."""

    def test_availability_returns_bool(self) -> None:
        result = is_tmhmm_available()
        assert isinstance(result, bool)

    def test_graceful_degradation(self) -> None:
        """When TMHMM is not installed, should return zeros."""
        result = predict_transmembrane(SAMPLE_SEQUENCES)
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
        if not is_tmhmm_available():
            assert np.all(result == 0)

    def test_disabled_in_config(self) -> None:
        """When disabled in config, should return zeros immediately."""
        from src.utils.config import DotDict

        cfg = DotDict.from_dict({"features": {"transmembrane": {"enabled": False}}})
        result = predict_transmembrane(SAMPLE_SEQUENCES, cfg=cfg)
        assert result.shape == (3, 3)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestComputeAllExternalFeatures:
    """Tests for the feature orchestrator."""

    def test_all_disabled_returns_empty(self) -> None:
        from src.data.external_features import compute_all_external_features
        from src.utils.config import DotDict

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {"enabled": False},
                    "signal_peptide": {"enabled": False},
                    "transmembrane": {"enabled": False},
                }
            }
        )
        result = compute_all_external_features(SAMPLE_SEQUENCES, cfg)
        assert result.shape == (3, 0)

    def test_biophysical_only(self) -> None:
        from src.data.external_features import compute_all_external_features
        from src.utils.config import DotDict

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {
                        "enabled": True,
                        "properties": ["molecular_weight", "gravy"],
                    },
                    "signal_peptide": {"enabled": False},
                    "transmembrane": {"enabled": False},
                }
            }
        )
        result = compute_all_external_features(SAMPLE_SEQUENCES, cfg)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32

    def test_no_features_config(self) -> None:
        from src.data.external_features import compute_all_external_features
        from src.utils.config import DotDict

        cfg = DotDict.from_dict({})
        result = compute_all_external_features(SAMPLE_SEQUENCES, cfg)
        assert result.shape == (3, 0)

    def test_all_enabled_concatenates_features_in_fixed_order(self, monkeypatch) -> None:
        from src.data.external_features import compute_all_external_features
        from src.utils.config import DotDict

        monkeypatch.setattr(
            "src.features.biophysical.compute_biophysical_features",
            lambda sequences, cfg: np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        )
        monkeypatch.setattr(
            "src.features.signal_peptide.predict_signal_peptides",
            lambda sequences, cfg: np.array(
                [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
                dtype=np.float32,
            ),
        )
        monkeypatch.setattr(
            "src.features.transmembrane.predict_transmembrane",
            lambda sequences, cfg: np.array(
                [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0], [26.0, 27.0, 28.0]],
                dtype=np.float32,
            ),
        )

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {
                        "enabled": True,
                        "properties": ["molecular_weight"],
                    },
                    "signal_peptide": {"enabled": True},
                    "transmembrane": {"enabled": True},
                }
            }
        )

        result = compute_all_external_features(SAMPLE_SEQUENCES, cfg)

        assert result.shape == (3, 6)
        assert result.dtype == np.float32
        assert np.allclose(result[0], [1.0, 10.0, 11.0, 20.0, 21.0, 22.0])


class TestGetExternalFeatureDim:
    """Tests for feature dimension calculation."""

    def test_all_disabled(self) -> None:
        from src.data.external_features import get_external_feature_dim
        from src.utils.config import DotDict

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {"enabled": False},
                    "signal_peptide": {"enabled": False},
                    "transmembrane": {"enabled": False},
                }
            }
        )
        assert get_external_feature_dim(cfg) == 0

    def test_all_enabled(self) -> None:
        from src.data.external_features import get_external_feature_dim
        from src.utils.config import DotDict

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {
                        "enabled": True,
                        "properties": [
                            "molecular_weight",
                            "gravy",
                            "isoelectric_point",
                        ],
                    },
                    "signal_peptide": {"enabled": True},
                    "transmembrane": {"enabled": True},
                }
            }
        )
        # 3 biophysical + 2 signal peptide + 3 transmembrane = 8
        assert get_external_feature_dim(cfg) == 8

    def test_partial_enabled(self) -> None:
        from src.data.external_features import get_external_feature_dim
        from src.utils.config import DotDict

        cfg = DotDict.from_dict(
            {
                "features": {
                    "biophysical": {"enabled": False},
                    "signal_peptide": {"enabled": True},
                    "transmembrane": {"enabled": False},
                }
            }
        )
        assert get_external_feature_dim(cfg) == 2
