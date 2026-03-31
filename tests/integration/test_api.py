# tests/integration/test_api.py
"""Integration tests for the FastAPI serving application.

Tests the API endpoints without a loaded model (degraded mode).
Full model integration tests require a trained checkpoint.
"""

from __future__ import annotations

import pytest

from src.serving.app import app


@pytest.fixture()
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_degraded_without_model(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        # Without a trained model, should report degraded
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_without_model_returns_503(self, client) -> None:
        response = client.post(
            "/predict",
            json={"sequence": "MSKGEELFTGVVPILVELDG"},
        )
        assert response.status_code == 503

    def test_invalid_sequence_returns_422(self, client) -> None:
        response = client.post(
            "/predict",
            json={"sequence": "12345"},
        )
        assert response.status_code == 422

    def test_too_short_sequence_returns_422(self, client) -> None:
        response = client.post(
            "/predict",
            json={"sequence": "MSK"},
        )
        assert response.status_code == 422


class TestBatchEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_without_model_returns_503(self, client) -> None:
        response = client.post(
            "/predict/batch",
            json={
                "sequences": [
                    {"sequence": "MSKGEELFTGVVPILVELDG"},
                    {"sequence": "MQIFVKTLTGKTITLEVEPSD"},
                ]
            },
        )
        assert response.status_code == 503

    def test_empty_batch_returns_422(self, client) -> None:
        response = client.post(
            "/predict/batch",
            json={"sequences": []},
        )
        assert response.status_code == 422


class TestLabelsEndpoint:
    """Tests for the /labels endpoint."""

    def test_labels_without_model_returns_503(self, client) -> None:
        response = client.get("/labels")
        assert response.status_code == 503
