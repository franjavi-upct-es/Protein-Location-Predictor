# src/models/pooling.py
"""
Trainable pooling modules for sequence-level protein representations.

The stock ``extract_sequence_representation`` function in
``src/models/esm_lora.py`` provides three parameter-free pooling
strategies (``mean``, ``cls``, ``mean_cls``). This module adds a
trainable alternative — *light attention* — which learns to weight
tokens by importance before averaging.

Light attention is the pooling strategy used by DeepLoc 2.0 and consistently
outperforms mean pooling on subcellular localization benchmarks. It
costs ~5k parameters per backbone (a single linear projection over the
hidden size) and adds negligible compute.

The architecture follows the original DeepLoc 2.0 formulation:

    1. Apply a 1D convolution (kernel size 1, equivalent to a per-token
       Linear) to project token embeddings to a single attention score.
    2. Mask out padding positions with -inf.
    3. Softmax over sequence length to get a per-token weight.
    4. Multiply weights by the original embeddings and sum.

The output has the same dimensionality as the input embeddings.

Usage::

    from src.models.pooling import LightAttentionPooler
    pooler = LightAttentionPooler(hidden_dim=640)
    pooled = pooler(hidden_states, attention_mask)  # (B, hidden_dim)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightAttentionPooler(nn.Module):
    """
    Light attention pooling over a sequence of token embeddings.

    Args:
        hidden_dim: Dimensionality of the input token embeddings.
        dropout: Dropout applied to the attention weights before
            multiplying by the embeddings. Helps regularize the
            attention layer.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        # Per-token linear projection to a scalar attention score
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        # Light dropout on attention weights
        self.dropout = nn.Dropout(dropout)

        # Initialize with a small variance so the early-training
        # softmax distribution is close to uniform (i.e. close to
        # mean pooling), giving the optimizer a sensible starting point.
        nn.init.normal_(self.attention.weight, std=0.02)

        logger.info(f"LightAttentionPooler initialized: hidden_dim={hidden_dim}, dropout={dropout}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings into a single sequence-level vector.

        Args:
            hidden_states: Token embeddings of shape (B, L, D).
            attention_mask: 1 for real tokens, 0 for padding (B, L).

        Returns:
            Tensor of shape (B, D).
        """
        # (B, L, 1) attention scores
        scores = self.attention(hidden_states)
        scores = scores.squeeze(-1)  # (B, L)

        # Mask padding with a very negative number so softmax assigns
        # ~0 weight to those positions
        mask = attention_mask.to(dtype=scores.dtype)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Defensive: if a row is all-padding (shouldn't happen) avoid NaN
        all_padding = mask.sum(dim=1) == 0
        if all_padding.any():
            scores[all_padding] = 0.0

        weights = torch.softmax(scores, dim=1)  # (B, L)
        weights = self.dropout(weights)
        weights = weights.unsqueeze(-1)  # (B, L, 1)

        pooled = (hidden_states * weights).sum(dim=1)  # (B, D)
        return cast(torch.Tensor, pooled)
