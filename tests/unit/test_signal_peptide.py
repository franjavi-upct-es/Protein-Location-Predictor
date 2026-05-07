# tests/unit/test_signal_peptide.py
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.features.signal_peptide import predict_signal_peptides

SAMPLE_SEQUENCES = ["MSKGEELFTGVVPILVELDGDVNGHKFSVRGEG", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKE"]


def test_predict_signal_peptides_success(monkeypatch, tmp_path):
    # Mock is_signalp_available
    monkeypatch.setattr("src.features.signal_peptide.is_signalp_available", lambda x: True)

    # Mock tempfile.TemporaryDirectory to return our tmp_path
    class MockTmpDir:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("src.features.signal_peptide.tempfile.TemporaryDirectory", MockTmpDir)

    # Mock subprocess.run to create the output file
    def mock_run_func(*args, **kwargs):
        output_dir = tmp_path / "output"
        # Since the function calls output_dir.mkdir(), it already exists here
        prediction_file = output_dir / "prediction_results.txt"
        with open(prediction_file, "w") as f:
            f.write("# comment\n")
            f.write("seq_0\tSP\t0.95\n")
            f.write("seq_1\tOTHER\t0.05\n")
        mock = MagicMock()
        mock.returncode = 0
        return mock

    monkeypatch.setattr("src.features.signal_peptide.subprocess.run", mock_run_func)

    result = predict_signal_peptides(SAMPLE_SEQUENCES, binary_path="signalp6")

    assert result.shape == (2, 2)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.95
    assert result[1, 0] == 0.0
    assert result[1, 1] == 0.05


def test_predict_signal_peptides_subprocess_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("src.features.signal_peptide.is_signalp_available", lambda x: True)

    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stderr = "Error occurred"
    monkeypatch.setattr(
        "src.features.signal_peptide.subprocess.run", lambda *args, **kwargs: mock_run
    )

    result = predict_signal_peptides(SAMPLE_SEQUENCES, binary_path="signalp6")

    assert result.shape == (2, 2)
    assert np.all(result == 0.0)


def test_predict_signal_peptides_no_output_file(monkeypatch, tmp_path):
    monkeypatch.setattr("src.features.signal_peptide.is_signalp_available", lambda x: True)

    class MockTmpDir:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("src.features.signal_peptide.tempfile.TemporaryDirectory", MockTmpDir)

    # Create the output directory but NO prediction_results.txt
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    mock_run = MagicMock()
    mock_run.returncode = 0
    monkeypatch.setattr(
        "src.features.signal_peptide.subprocess.run", lambda *args, **kwargs: mock_run
    )

    result = predict_signal_peptides(SAMPLE_SEQUENCES, binary_path="signalp6")

    assert result.shape == (2, 2)
    assert np.all(result == 0.0)


def test_predict_signal_peptides_exception(monkeypatch):
    monkeypatch.setattr("src.features.signal_peptide.is_signalp_available", lambda x: True)

    def mock_run(*args, **kwargs):
        raise ValueError("Simulated Exception")

    monkeypatch.setattr("src.features.signal_peptide.subprocess.run", mock_run)

    result = predict_signal_peptides(SAMPLE_SEQUENCES, binary_path="signalp6")

    assert result.shape == (2, 2)
    assert np.all(result == 0.0)
