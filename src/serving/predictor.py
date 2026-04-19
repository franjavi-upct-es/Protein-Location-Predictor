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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.cfg = cfg
        self.device = device
        self.threshold = threshold
        self.top_k = top_k
        self.max_length = max_length

        # Per-class thresholds (None means use the global threshold)
        self.per_class_thresholds: dict[str, float] | None = None

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
        batch_predictions = self.predict_batch([sequence], threshold, top_k)
        return cast(list[dict[str, Any]], batch_predictions[0])

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

    def _format_predictions(
        self,
        probabilities: Any,
        threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Convert class probabilities into the public prediction format."""
        results: list[dict[str, Any]] = []
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
        threshold = threshold or self.threshold
        top_k = top_k or self.top_k

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
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            outputs.append(self._format_predictions(probabilities, threshold, top_k))

        return outputs

    def warmup(self) -> None:
        """Run a dummy prediction to warm ip the model (precompile, cache)."""
        dummy = "M" * 50 + "ACDEFGHIKLMNPQRSTVWYXU"
        _ = self.predict(dummy)
        logger.info("Model warmup complete")
