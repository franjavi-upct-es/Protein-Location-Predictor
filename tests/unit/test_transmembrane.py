# tests/unit/test_transmembrane.py
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.features.transmembrane import predict_transmembrane

SAMPLE_SEQUENCES = ["MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKE"]


def test_predict_transmembrane_success(monkeypatch, tmp_path):
    monkeypatch.setattr("src.features.transmembrane.is_tmhmm_available", lambda x: True)

    class MockTmpDir:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("src.features.transmembrane.tempfile.TemporaryDirectory", MockTmpDir)

    mock_run = MagicMock()
    mock_run.returncode = 0
    # Simulate TMHMM short output
    mock_run.stdout = """# comment
seq_0	len=33	ExpAA=15.0	First60=0.0	PredHel=1	Topology=o15-33i
seq_1	len=34	ExpAA=0.0	First60=0.0	PredHel=0	Topology=o
"""
    monkeypatch.setattr(
        "src.features.transmembrane.subprocess.run", lambda *args, **kwargs: mock_run
    )

    result = predict_transmembrane(SAMPLE_SEQUENCES, binary_path="tmhmm")

    assert result.shape == (2, 3)
    # seq_0: 1 helix, has_tm=1, fraction_in_membrane = 15.0 / 33 = 0.4545...
    assert result[0, 0] == 1.0
    assert result[0, 1] == 1.0
    np.testing.assert_allclose(result[0, 2], 15.0 / 33)

    # seq_1: 0 helix, has_tm=0, fraction = 0.0 / 34 = 0.0
    assert result[1, 0] == 0.0
    assert result[1, 1] == 0.0
    assert result[1, 2] == 0.0


def test_predict_transmembrane_subprocess_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("src.features.transmembrane.is_tmhmm_available", lambda x: True)

    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stderr = "Error"
    monkeypatch.setattr(
        "src.features.transmembrane.subprocess.run", lambda *args, **kwargs: mock_run
    )

    result = predict_transmembrane(SAMPLE_SEQUENCES, binary_path="tmhmm")

    assert result.shape == (2, 3)
    assert np.all(result == 0.0)


def test_predict_transmembrane_exception(monkeypatch):
    monkeypatch.setattr("src.features.transmembrane.is_tmhmm_available", lambda x: True)

    def mock_run(*args, **kwargs):
        raise ValueError("Simulated Exception")

    monkeypatch.setattr("src.features.transmembrane.subprocess.run", mock_run)

    result = predict_transmembrane(SAMPLE_SEQUENCES, binary_path="tmhmm")

    assert result.shape == (2, 3)
    assert np.all(result == 0.0)
