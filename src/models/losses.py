# src/models/losses.py
"""
Loss functions for multi-label protein localization.

Provides:
  - FocalLoss: Handles class imbalance by down-weighting easy examples.
  - HierarchicalLoss: Penalizes biologically distant misclassifications.
  - CombinedLoss: Weighted sum of Focal and Hierarchical terms.

All losses operate on raw logits (pre-sigmoid) for numerical stability.

Usage:
    from src.models.losses import CombinedLoss
    criterion = CombinedLoss.from_config(cfg, label_list, class_frequencies)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss (multi-label, per-class)
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification.

    Extends binary cross-entropy with a modulating factor (1 - p_t)^gamma
    that reduces the loss contribution from asy (well-classified) examples,
    focusing training on hard cases and minority classes.

    Operates on raw logits for numerical stability.

    Args:
        gamma: Focusing parameter. gamma=0 recovers standard BCE.
               Higher values focus more on hard examples. Default: 2.0.
        alpha: Per-class weight tensor of shape (num_classes,). If None,
               all classes are weighted equally. Can be auto-computed
               from class frequencies.
        reduction: "mean" or "sum" or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output of shape (B, C), pre-sigmoid.
            targets: Multi-hot target tensor of shape (B, C).

        Returns:
            Scalar loss value.
        """
        # Numerically stable sigmoid + BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, C)

        # Probability of the true class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce  # (B, C)

        # Per-class weighting
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(
                0
            )  # broadcast (1, C) over (B, C)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def compute_alpha_from_frequencies(
        class_frequencies: torch.Tensor,
        smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute per-class weights inversely proportional to frequency.

        Args:
            class_frequencies: Tensor of shape (C,) with the count or fraction
                               of each class in the training set.
            smoothing: Laplace smoothing to prevent extreme weights.

        Returns:
            Normalized weight tensor of shape (C,).
        """
        freqs = class_frequencies.float() + smoothing
        weights = 1.0 / freqs
        weights = weights / weights.sum() * len(weights)  # normalize to mean=1
        return weights


# ---------------------------------------------------------------------------
# Hierarchical Loss
# ---------------------------------------------------------------------------

# Biological distance matrix between compartments.
# Values represent how "far apart" two compartments are in the cell.
# 0 = same, 1 = adjacent/related, 2 = distant.
# This encodes the intuition that confusing Nucleus with Cytoplasm
# (both intracellular, adjacent) is less wrong than confusing
# Nucleus with Secreted/Extracellular (intracellular vs extracellular).

_DEFAULT_HIERARCHY = {
    ("Nucleus", "Cytoplasm"): 1,
    ("Nucleus", "Membrane"): 2,
    ("Nucleus", "Mitochondrion"): 2,
    ("Nucleus", "Endoplasmic Reticulum"): 2,
    ("Nucleus", "Golgi Apparatus"): 2,
    ("Nucleus", "Secreted/Extracellular"): 3,
    ("Nucleus", "Vacuole"): 2,
    ("Nucleus", "Peroxisome"): 2,
    ("Cytoplasm", "Membrane"): 1,
    ("Cytoplasm", "Mitochondrion"): 1,
    ("Cytoplasm", "Endoplasmic Reticulum"): 1,
    ("Cytoplasm", "Golgi Apparatus"): 2,
    ("Cytoplasm", "Secreted/Extracellular"): 2,
    ("Cytoplasm", "Vacuole"): 2,
    ("Cytoplasm", "Peroxisome"): 1,
    ("Membrane", "Endoplasmic Reticulum"): 1,
    ("Membrane", "Golgi Apparatus"): 1,
    ("Membrane", "Secreted/Extracellular"): 1,
    ("Membrane", "Mitochondrion"): 2,
    ("Membrane", "Vacuole"): 1,
    ("Membrane", "Peroxisome"): 2,
    ("Mitochondrion", "Endoplasmic Reticulum"): 2,
    ("Mitochondrion", "Golgi Apparatus"): 2,
    ("Mitochondrion", "Secreted/Extracellular"): 3,
    ("Mitochondrion", "Vacuole"): 2,
    ("Mitochondrion", "Peroxisome"): 1,
    ("Endoplasmic Reticulum", "Golgi Apparatus"): 1,
    ("Endoplasmic Reticulum", "Secreted/Extracellular"): 1,
    ("Endoplasmic Reticulum", "Vacuole"): 2,
    ("Endoplasmic Reticulum", "Peroxisome"): 2,
    ("Golgi Apparatus", "Secreted/Extracellular"): 1,
    ("Golgi Apparatus", "Vacuole"): 1,
    ("Golgi Apparatus", "Peroxisome"): 2,
    ("Secreted/Extracellular", "Vacuole"): 2,
    ("Secreted/Extracellular", "Peroxisome"): 3,
    ("Vacuole", "Peroxisome"): 2,
}


class HierarchicalLoss(nn.Module):
    """
    Hierarchical penalty for biologically distant misclassifications.

    Adds a penalty proportional to the biological distance between
    predicted and true compartments. This encourages the model to make
    "nearby" mistakes rather than completely wrong ones.

    For multi-label: computes pairwise penalties between predicted
    positive classes and ground truth negative classes, weighted by
    the biological distance.

    Args:
        distance_matrix: Tensor of shape (C, C) enconding pairwise distances.
        label_list: Ordered list of location class names.
    """

    def __init__(
        self,
        distance_matrix: torch.Tensor,
        label_list: list[str],
    ) -> None:
        super().__init__()
        self.register_buffer("distance_matrix", distance_matrix.float())
        self.label_list = label_list

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hierarchical penalty.

        Args:
            logits: Raw model output of shape (B, C).
            targets: Multi-hot target tensor of shape (B, C).

        Returns:
            Scalar hierarchical penalty.
        """
        probs = torch.sigmoid(logits)  # (B, C)

        # False positive penalty: predicted positive but target is negative
        # Weight by distance from the nearest true positive class
        false_pos = probs * (
            1 - targets
        )  # (B, C) — high where wrongly confident

        # For each false positive, compute its average
        # distance to all true classes
        # targets: (B, C), distance_matrix: (C, C)
        # dist_to_true: (B, C) — average distance
        # from each class to the true set
        true_count = targets.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        dist_to_true = (
            torch.matmul(targets, self.distance_matrix) / true_count
        )  # (B, C)

        # Penalty = false positive probability * distance to nearest true class
        penalty = (false_pos * dist_to_true).mean()

        return penalty

    @classmethod
    def build_distance_matrix(
        cls,
        label_list: list[str],
        hierarchy: dict[tuple[str, str], int] | None = None,
        default_distance: float = 2.0,
    ) -> torch.Tensor:
        """
        Build a pairwise distance matrix from the hierarchy definition.

        Args:
            label_list: Ordered list of location class names.
            hierarchy: Dict mapping (class_a, class_b) -> distance.
                       If None, uses the built-in biological hierarchy.
            default_distance: Distance for pairs not in the hierarchy.

        Returns:
            Tensor of shape (C, C) with pairwise distances.
        """
        if hierarchy is None:
            hierarchy = _DEFAULT_HIERARCHY

        n = len(label_list)
        matrix = torch.full((n, n), default_distance)

        label_to_idx = {label: i for i, label in enumerate(label_list)}

        for (a, b), dist in hierarchy.items():
            if a in label_to_idx and b in label_to_idx:
                i, j = label_to_idx[a], label_to_idx[b]
                matrix[i, j] = dist
                matrix[j, i] = dist

        # Diagonal is zero (no distance to self)
        matrix.fill_diagonal_(0.0)

        # Normalize to [0, 1] range
        max_dist = matrix.max()
        if max_dist > 0:
            matrix = matrix / max_dist

        return matrix


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Hierarchical Loss.

    Args:
        focal_loss: FocalLoss instance.
        hierarchical_loss: HierarchicalLoss instance (or None to disable).
        hierarchical_weight: Weight of the hierarchical term relative to focal.
    """

    def __init__(
        self,
        focal_loss: FocalLoss,
        hierarchical_loss: HierarchicalLoss | None = None,
        hierarchical_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.focal_loss = focal_loss
        self.hierarchical_loss = hierarchical_loss
        self.hierarchical_weight = hierarchical_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns a dict with individual loss components and the total,
        useful for logging.
        """
        focal = self.focal_loss(logits, targets)
        result = {"focal": focal, "total": focal}

        if self.hierarchical_loss is not None and self.hierarchical_weight > 0:
            hier = self.hierarchical_loss(logits, targets)
            total = focal + self.hierarchical_weight * hier
            result["hierarchical"] = hier
            result["total"] = total

        return result

    @classmethod
    def from_config(
        cls,
        cfg: DotDict,
        label_list: list[str],
        class_frequencies: torch.Tensor | None = None,
    ) -> CombinedLoss:
        """
        Build CombinedLoss from configuration.

        Args:
            cfg: Project configuration.
            label_list: Ordered list of location class names.
            class_frequencies: Optional tensor of per-class sample counts.
                               Uses for auto-computing focal alpha weights.

        Returns:
            Configured CombinedLoss instance.
        """
        loss_cfg = cfg.loss

        # Focal loss
        gamma = loss_cfg.focal.get("gamma", 2.0)
        alpha = None
        if (
            class_frequencies is not None
            and loss_cfg.focal.get("alpha") is None
        ):
            alpha = FocalLoss.compute_alpha_from_frequencies(class_frequencies)
            logger.info(
                f"Auto-computed focal alpha from class frequencies: {alpha}"
            )
        focal = FocalLoss(gamma=gamma, alpha=alpha)

        # Hierarchical loss
        hier_loss = None
        hier_cfg = loss_cfg.get("hierarchical", {})
        if hier_cfg.get("enabled", False):
            dist_matrix = HierarchicalLoss.build_distance_matrix(label_list)
            hier_loss = HierarchicalLoss(dist_matrix, label_list)
            logger.info("Hierarchical loss enabled")

        hier_weight = hier_cfg.get("weight", 0.1)

        return cls(focal, hier_loss, hier_weight)
