# src/serving/predictor.py
"""
Inference-time predictor class.

Loads a trained checkpoint and provides single/batch prediction with
configurable thresholds and top-k ranking.

Usage:
    from src.serving.predictor import Predictor
    predictor = Predictor.from_checkpoint("models/checkpoints/best.ckpt", cfg)
    result = predictor.predict("MSKGEEL...")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        device: str = "cpu",
        threshold: float = 0.5,
        top_k: int = 3,
        max_length: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device
        self.threshold = threshold
        self.top_k = top_k
        self.max_length = max_length

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
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
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

        return cls(
            model=model,
            tokenizer=tokenizer,
            label_list=label_list,
            device=device,
            threshold=threshold,
            top_k=top_k,
            max_length=max_length,
        )

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
        threshold = threshold or self.threshold
        top_k = top_k or self.top_k

        # Tokenize
        enconding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            trunctaion=True,
            max_length=self.max_length,
        )

        input_ids = enconding["input_ids"].to(self.device)
        attention_mask = enconding["attention_mask"].to(self.device)

        # Forward pass
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Build results
        results: list[dict[str, str | float]] = []
        for label, prob in zip(self.label_list, probabilities, strict=True):
            if prob >= threshold:
                results.append(
                    {
                        "location": label,
                        "confidence": round(float(prob), 4),
                    }
                )

        # Sort by confidence and limit to top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = results[:top_k]

        # If nothing passed the threshold, return the top prediction anyway
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
        # For simplicity and memory safety, process one at a time
        # (batch tokenization with dynamic padding could be added later)
        return [self.predict(seq, threshold, top_k) for seq in sequences]

    def warmup(self) -> None:
        """Run a dummy prediction to warm ip the model (precompile, cache)."""
        dummy = "M" * 50 + "ACDEFGHIKLMNPQRSTVWYXU"
        _ = self.predict(dummy)
        logger.info("Model warmup complete")
