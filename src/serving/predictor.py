# src/serving/predictor.py
"""
Inference-time predictor class.

Loads a trained checkpoint and provides single/batch prediction with
configurable thresholds and top-k ranking.

Usage::

    from src.serving.predictor import Predictor
    predictor = Predictor.from_checkpoint("models/checkpoints/best.ckpt", cfg)
    result = predictor.predict("MSKGEEL...")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Inference predictor for protein subcellular localization.

    Wraps a trained Lightning module for efficient inference with
    automatic device management and precision handling.

    Args:
        model: Trained Lightning module.
        tokenizer: ESM-2 tokenizer.
        label_list: Ordered list of location class names.
        device: Device for inference.
        threshold: Sigmoid threshold for positive predictions.
        top_k: Maximum number of locations to return per sequence.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        label_list: list[str],
        cfg: DotDict | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        top_k: int = 3,
        max_length: int = 1024,
        chunk_long_sequences: bool = False,
        chunk_overlap: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.cfg = cfg
        self.device = device
        self.threshold = threshold
        self.top_k = top_k
        self.max_length = max_length
        self.chunk_long_sequences = chunk_long_sequences
        self.chunk_overlap = chunk_overlap

        # Per-class thresholds (None means use the global threshold)
        self.per_class_thresholds: dict[str, float] | None = None
        # Per-class temperature calibration
        self.temperatures: list[float] | None = None

        self.model.eval()
        self.model.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfg: DotDict,
        device: str | None = None,
    ) -> Predictor:
        """
        Load a predictor from a training checkpoint.

        Args:
            checkpoint_path: Path to the .ckpt file.
            cfg: Project configuration.
            device: Inference device. Auto-detected if None.

        Returns:
            Configured Predictor instance.
        """
        from transformers import AutoTokenizer

        from src.models.lightning_module import ProteinLocalizationModule

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load model
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading model from {checkpoint_path} onto {device}")

        model = ProteinLocalizationModule.load_from_checkpoint(
            str(checkpoint_path),
            cfg=cfg,
            map_location=device,
            strict=False,
            weights_only=False,
        )

        # Load tokenizer
        model_name = cfg.model.backbone.name
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Read inference config
        inf_cfg = cfg.get("inference", {})
        threshold = inf_cfg.get("threshold", 0.5)
        top_k = inf_cfg.get("top_k", 3)
        max_length = cfg.model.backbone.get("max_position_embeddings", 1024)

        label_list = model.label_list

        # Try to load per-class thresholds from the same directory
        from src.evaluation.threshold_tuning import load_thresholds

        thresholds_path = Path(checkpoint_path).parent / "thresholds.json"
        per_class_thresholds = load_thresholds(thresholds_path)

        from src.evaluation.calibration import load_temperatures

        temperatures_path = Path(checkpoint_path).parent / "temperatures.json"
        temperatures = load_temperatures(temperatures_path)

        predictor = cls(
            model=model,
            tokenizer=tokenizer,
            label_list=label_list,
            cfg=cfg,
            device=device,
            threshold=threshold,
            top_k=top_k,
            max_length=max_length,
        )
        if per_class_thresholds is not None:
            predictor.per_class_thresholds = per_class_thresholds
            logger.info(f"Loaded {len(per_class_thresholds)} per-class thresholds")

        if temperatures is not None:
            predictor.temperatures = temperatures
            logger.info(f"Loaded {len(temperatures)} per-class temperatures")

        return predictor

    @torch.no_grad()
    def predict(
        self,
        sequence: str,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Predict subcellular locations for a single sequence.

        Args:
            sequence: Amino acid sequence string.
            threshold: Override default threshold.
            top_k: Override default top_k.

        Returns:
            List of dicts with 'location' and 'confidence' keys,
            sorted by confidence descending. Only locations above
            threshold are included, up to top_k.
        """
        # Sliding-window inference for long sequences
        if self.chunk_long_sequences and len(sequence) > self.max_length:
            return cast(
                list[dict[str, Any]],
                self._predict_long(sequence, threshold, top_k),
            )

        external_features = self._compute_external_features([sequence])
        ext = external_features[:1] if external_features is not None else None

        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            external_features=ext,
        )

        if self.temperatures is not None:
            from src.evaluation.calibration import apply_temperatures

            logits_np = logits.float().cpu().numpy()
            probabilities = apply_temperatures(logits_np, self.temperatures)[0]
        else:
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        return self._build_results(probabilities, threshold, top_k)

    def _compute_external_features(
        self,
        sequences: list[str],
    ) -> torch.Tensor | None:
        """Compute optional external features for a batch of sequences."""
        if self.cfg is None:
            return None

        from src.data.external_features import get_external_feature_dim

        if get_external_feature_dim(self.cfg) == 0:
            return None

        from src.data.external_features import compute_all_external_features

        features = compute_all_external_features(sequences, self.cfg)
        if features.shape[1] == 0:
            return None

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: list[str],
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Predict subcellular locations for a batch of sequences.

        Args.
            sequences: List of amino acid sequences.
            threshold: Override default threshold.
            top_k: Override default top_k.

        Returns:
            List of prediction lists (one per sequence).
        """
        threshold = self.threshold if threshold is None else threshold
        top_k = self.top_k if top_k is None else top_k

        if not sequences:
            return []

        external_features = self._compute_external_features(sequences)
        outputs: list[list[dict[str, Any]]] = []

        # Keep tokenization one sequence at a time for memory safety, but compute
        # external features once across the full input list so feature scaling
        # matches the same batch context.
        for i, sequence in enumerate(sequences):
            encoding = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            ext = external_features[i : i + 1] if external_features is not None else None

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                external_features=ext,
            )
            if self.temperatures is not None:
                from src.evaluation.calibration import apply_temperatures

                logits_np = logits.float().cpu().numpy()
                probabilities = apply_temperatures(logits_np, self.temperatures)[0]
            else:
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            outputs.append(self._build_results(probabilities, threshold, top_k))

        return outputs

    @torch.no_grad()
    def _predict_long(
        self,
        sequence: str,
        threshold: float | None,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Predict on a long sequence by chunking + aggregating logits."""
        import numpy as np

        from src.serving.chunking import (
            aggregate_logits,
            split_into_chunks,
        )

        chunks = split_into_chunks(
            sequence,
            window_size=self.max_length,
            overlap=self.chunk_overlap,
        )
        logger.info(f"Long sequence ({len(sequence)} aa) → {len(chunks)} chunks")

        chunk_logits: list[np.ndarray] = []
        for chunk in chunks:
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            chunk_logits.append(logits.float().cpu().numpy()[0])

        agg = aggregate_logits(chunk_logits, strategy="mean")

        # Apply temperature scaling if available, then threshold
        if self.temperatures is not None:
            from src.evaluation.calibration import apply_temperatures

            probabilities = apply_temperatures(agg[np.newaxis, :], self.temperatures)[0]
        else:
            probabilities = 1.0 / (1.0 + np.exp(-agg))

        return self._build_results(probabilities, threshold, top_k)

    def _build_results(
        self,
        probabilities: np.ndarray,
        threshold: float | None,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Apply thresholds and top-k to a probability vector."""
        threshold = self.threshold if threshold is None else threshold
        top_k = self.top_k if top_k is None else top_k

        results: list[dict[str, str | float]] = []
        for label, prob in zip(self.label_list, probabilities, strict=True):
            class_threshold = (
                self.per_class_thresholds.get(label, threshold)
                if self.per_class_thresholds is not None
                else threshold
            )
            if prob >= class_threshold:
                results.append(
                    {
                        "location": label,
                        "confidence": round(float(prob), 4),
                    }
                )
        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = results[:top_k]

        if not results:
            best_idx = int(probabilities.argmax())
            results = [
                {
                    "location": self.label_list[best_idx],
                    "confidence": round(float(probabilities[best_idx]), 4),
                }
            ]
        return results

    def warmup(self) -> None:
        """Run a dummy prediction to warm ip the model (precompile, cache)."""
        dummy = "M" * 50 + "ACDEFGHIKLMNPQRSTVWYXU"
        _ = self.predict(dummy)
        logger.info("Model warmup complete")
