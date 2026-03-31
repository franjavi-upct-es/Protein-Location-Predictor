# src/serving/app.py
"""
FastAPI REST service for protein localization prediction.

Endpoints:
  POST /predict         — Single sequence prediction
  POST /predict/batch   — Batch prediction (up to 100 sequences)
  GET  /health          — Health check and model info
  GET  /labels          — List of supported location classes

Usage:
    uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
    uv run python -m src.serving.app
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LabelListResponse,
    LocationPrediction,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_predictor = None
_config: DotDict | None = None


def _find_best_checkpoint(cfg: DotDict) -> Path | None:
    """Find the best available model checkpoint."""
    # Check explicit config path first
    inf_cfg = cfg.get("inference", {})
    explicit = inf_cfg.get("checkpoint_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    # Search in models/checkpoints for the latest 'last.ckpt' or best
    models_dir = resolve_path(cfg, "paths.models_dir")
    ckpt_dir = models_dir / "checkpoints"

    if ckpt_dir.exists():
        # Prefer 'last.ckpt'
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last

        # Otherwise find any .ckpt file (sorted by modification time)
        ckpts = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            return ckpts[0]

    return None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Load the model on startup, clean up on shutdown."""
    global _predictor, _config

    _config = load_config(mode="inference")
    setup_logging(level=_config.project.log_level)

    checkpoint = _find_best_checkpoint(_config)

    if checkpoint is not None:
        try:
            from src.serving.predictor import Predictor

            logger.info(f"Loading model from {checkpoint}")
            _predictor = Predictor.from_checkpoint(checkpoint, _config)

            serving_cfg = _config.get("serving", {})
            if serving_cfg.get("model_warmup", True):
                _predictor.warmup()

            logger.info("Model loaded and ready for inference")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning(
                "API will start without a model — predictions will fail"
            )
    else:
        logger.warning(
            "No model checkpoint found. API starting in degraded mode. "
            "Train a model first with: uv run python -m src.training.train"
        )

    yield

    # Cleanup
    _predictor = None
    logger.info("API shutdown complete")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Protein Subcellular Localization Predictor",
    description=(
        "Predicts the subcellular localization of proteins "
        "from their amino acid sequences using ESM-2 "
        "with LoRA fine-tuning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _require_model() -> Any:
    """Check that the model is loaded, raise 503 otherwise."""
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train a model first and restart the server."
            ),
        )
    return _predictor


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — returns model info and device."""
    if _predictor is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=_config.model.backbone.name if _config else "unknown",
            device=_predictor.device,
            num_classes=len(_predictor.label_list),
            label_list=_predictor.label_list,
        )
    return HealthResponse(
        status="degraded",
        model_loaded=False,
        model_name="none",
        device="none",
        num_classes=0,
        label_list=[],
    )


@app.get("/labels", response_model=LabelListResponse)
async def list_labels() -> LabelListResponse:
    """List all supported subcellular location classes."""
    predictor = _require_model()
    return LabelListResponse(
        labels=predictor.label_list,
        num_labels=len(predictor.label_list),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """Predict subcellular location(s) for a single protein sequence."""
    predictor = _require_model()

    start = time.perf_counter()
    results = predictor.predict(request.sequence)
    elapsed = time.perf_counter() - start

    logger.debug(
        f"Prediction for {len(request.sequence)}aa sequence in {elapsed:.3f}s"
    )

    return PredictionResponse(
        sequence_length=len(request.sequence),
        predictions=[
            LocationPrediction(
                location=r["location"], confidence=r["confidence"]
            )
            for r in results
        ],
        threshold=predictor.threshold,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict subcellular locations for a batch of protein sequences."""
    predictor = _require_model()

    serving_cfg = _config.get("serving", {}) if _config else {}
    max_len = serving_cfg.get("max_sequence_length", 2048)

    sequences = []
    for item in request.sequences:
        if len(item.sequence) > max_len:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Sequence exceeds maximum length of {max_len} residues"
                ),
            )
        sequences.append(item.sequence)

    start = time.perf_counter()
    batch_results = predictor.predict_batch(sequences)
    elapsed = time.perf_counter() - start

    logger.info(
        f"Batch prediction for {len(sequences)} sequences in {elapsed:.3f}s"
    )

    responses = []
    for seq, results in zip(sequences, batch_results, strict=True):
        responses.append(
            PredictionResponse(
                sequence_length=len(seq),
                predictions=[
                    LocationPrediction(
                        location=r["location"], confidence=r["confidence"]
                    )
                    for r in results
                ],
                threshold=predictor.threshold,
            )
        )

    return BatchPredictionResponse(
        results=responses,
        total_sequences=len(sequences),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the API server."""
    import uvicorn

    cfg = load_config(mode="inference")
    setup_logging(level=cfg.project.log_level)

    serving_cfg = cfg.get("serving", {})
    host = serving_cfg.get("host", "0.0.0.0")  # noqa: S104  # nosec B104
    port = serving_cfg.get("port", 8000)
    reload = serving_cfg.get("reload", False)

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
