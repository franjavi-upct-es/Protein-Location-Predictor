# src/models/classifier_head.py
"""
Multi-label classification head.

An MLP that takes ESM-2 pooled representations (optionally concatenated
with external feature vectors) and ouputs per-class logits for sigmoid
multi-label prediction.

Usage::

    from src.models.classifier_head import ClassifierHead
    head = ClassifierHead(input_dim=640, num_classes=9, cfg=cfg)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ClassifierHead(nn.Module):
    """
    Multi-label classification MLP.

    Architecture: Input -> [Linear -> GELU -> Dropout] x N -> Linear -> logtis

    No sigmoid is applied here — it's handled by the loss function
    (BCEWithLogitsLoss / FocalLoss) for numerical stability.

    Args:
        input_dim: Dimension of the input representation (ESM-2 embedding +
                   optional extenral features).
        num_classes: Number of output location classes.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability between layers.
        activation: Activation function name ("gelu" or "relu").
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize final layer with small weights for stable early training
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)

        logger.info(
            f"ClassifierHead: {input_dim} -> {hidden_dims} -> {num_classes} "
            f"(dropout={dropout}, activation={activation})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, input_dim)

        Returns:
            Logits tensor of shape (B, num_classes).
            Apply sigmoid for probabilities.
        """
        return self.network(x)

    @classmethod
    def from_config(
        cls, cfg: DotDict, input_dim: int, num_classes: int
    ) -> ClassifierHead:
        """
        Create a ClassifierHead from configuration.

        Args:
            cfg: Project configuration.
            input_dim: Dimension of input embeddings.
            num_classes: Number of location classes.

        Returns:
            Configured ClassifierHead instance.
        """
        head_cfg = cfg.model.classifier
        hidden_dims = list(head_cfg.get("hidden_dims", [512, 256]))
        dropout = head_cfg.get("dropout", 0.3)
        activation = head_cfg.get("activation", "gelu")

        return cls(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )
