# src/serving/schemas.py
"""
Pydantic models for API request and response validation.

Defines the data contracts for the prediction API endpoints.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Single protein prediction request."""

    sequence: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Amino acid sequence (standard single-letter codes)",
        examples=["MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATIGKLTLKFICTTGKLPVP"],
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWYXU]+$", v):
            raise ValueError(
                "Sequence must contain only standard amino acid characters "
                "(ACDEFGHIKLMNPQRSTVWYXU)"
            )
        return v


class BatchPredictionRequest(BaseModel):
    """Batch protein prediction request."""

    sequences: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of protein sequences to predict",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class LocationPrediction(BaseModel):
    """A single predicted subcellular location with confidence."""

    location: str = Field(
        ..., description="Predicted subcellular location name"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence (sigmoid probability)",
    )


class PredictionResponse(BaseModel):
    """Response for a single protein prediction."""

    predictions: list[LocationPrediction] = Field(
        ..., description="List of predicted locations"
    )
    sequence_length: int = Field(
        ..., description="Length of the input sequence"
    )
    threshold: float = Field(..., description="Confidence threshold used")


class BatchPredictionResponse(BaseModel):
    """Response for batch protein prediction."""

    results: list[PredictionResponse]
    total_sequences: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    device: str
    num_classes: int
    label_list: list[str]


class LabelListResponse(BaseModel):
    """Available location labels."""

    labels: list[str]
    num_labels: int
