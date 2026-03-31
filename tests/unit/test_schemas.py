# tests/unit/test_schemas.py
"""Tests for API request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    LocationPrediction,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for input validation."""

    def test_valid_sequence(self) -> None:
        req = PredictionRequest(sequence="MSKGEELFTGVVPILVELDG")
        assert req.sequence == "MSKGEELFTGVVPILVELDG"

    def test_lowercased_gets_uppercased(self) -> None:
        req = PredictionRequest(sequence="mskgeelftgvvpilveldg")
        assert req.sequence == "MSKGEELFTGVVPILVELDG"

    def test_whitespace_stripped(self) -> None:
        req = PredictionRequest(sequence="  MSKGEELFTGVVPILVELDG  ")
        assert req.sequence == "MSKGEELFTGVVPILVELDG"

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(sequence="MSK")

    def test_invalid_chars_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(sequence="MSKGEELFTG123VVPILVELDG")

    def test_allows_x_and_u(self) -> None:
        req = PredictionRequest(sequence="MSKGEELFTGXUPILVELDG")
        assert "X" in req.sequence
        assert "U" in req.sequence

    def test_numbers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(sequence="M1S2K3GEELFTGVVPILVELDG")


class TestPredictionResponse:
    """Tests for response schema."""

    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            sequence_length=100,
            predictions=[
                LocationPrediction(location="Nucleus", confidence=0.92),
                LocationPrediction(location="Cytoplasm", confidence=0.55),
            ],
            threshold=0.5,
        )
        assert len(resp.predictions) == 2
        assert resp.predictions[0].location == "Nucleus"

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LocationPrediction(location="Nucleus", confidence=1.5)
        with pytest.raises(ValidationError):
            LocationPrediction(location="Nucleus", confidence=-0.1)


class TestBatchPredictionRequest:
    """Tests for batch request validation."""

    def test_valid_batch(self) -> None:
        req = BatchPredictionRequest(
            sequences=[
                PredictionRequest(sequence="MSKGEELFTGVVPILVELDG"),
                PredictionRequest(sequence="MQIFVKTLTGKTITLEVEPSD"),
            ]
        )
        assert len(req.sequences) == 2

    def test_empty_batch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(sequences=[])


class TestHealthResponse:
    """Tests for health check response."""

    def test_healthy(self) -> None:
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="esm2",
            device="cuda",
            num_classes=9,
            label_list=["Nucleus", "Cytoplasm"],
        )
        assert resp.model_loaded is True

    def test_degraded(self) -> None:
        resp = HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name="none",
            device="none",
            num_classes=0,
            label_list=[],
        )
        assert resp.model_loaded is False
