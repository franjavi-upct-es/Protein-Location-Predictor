# tests/unit/test_losses.py
"""Tests for loss functions."""

from __future__ import annotations

import torch

from src.models.losses import (
    CombinedLoss,
    FocalLoss,
    HierarchicalLoss,
)

# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    """Tests for the FocalLoss implementation."""

    def test_output_is_scalar(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        targets = torch.zeros(4, 3)
        targets[0, 0] = 1.0
        targets[1, 1] = 1.0

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar

    def test_loss_is_non_negative(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8, 5)).float()

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_gamma_zero_recovers_bce(self) -> None:
        """With gamma=0 and no alpha, focal loss should equal BCE."""
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        focal = FocalLoss(gamma=0.0)(logits, targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="mean"
        )

        assert torch.allclose(focal, bce, atol=1e-5)

    def test_higher_gamma_reduces_easy_loss(self) -> None:
        """Higher gamma should reduce loss for well-classified examples."""
        # Create a confidently correct prediction
        logits = torch.tensor([[5.0, -5.0]])  # strong prediction for class 0
        targets = torch.tensor([[1.0, 0.0]])  # class 0 is correct

        loss_low_gamma = FocalLoss(gamma=0.0)(logits, targets)
        loss_high_gamma = FocalLoss(gamma=2.0)(logits, targets)

        assert loss_high_gamma < loss_low_gamma

    def test_alpha_weighting(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, alpha=torch.tensor([2.0, 1.0, 0.5]))
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_compute_alpha_from_frequencies(self) -> None:
        freqs = torch.tensor([100.0, 50.0, 10.0])
        alpha = FocalLoss.compute_alpha_from_frequencies(freqs)

        # Rare classes should get higher weight
        assert alpha[2] > alpha[1] > alpha[0]
        # Mean should be approximately 1 (normalized)
        assert abs(alpha.mean().item() - 1.0) < 0.1

    def test_reduction_none(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        loss = loss_fn(logits, targets)
        assert loss.shape == (4, 3)


# ---------------------------------------------------------------------------
# HierarchicalLoss
# ---------------------------------------------------------------------------


LABEL_LIST = ["Nucleus", "Cytoplasm", "Membrane", "Mitochondrion"]


class TestHierarchicalLoss:
    """Tests for the HierarchicalLoss implementation."""

    def test_build_distance_matrix_shape(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        assert matrix.shape == (4, 4)

    def test_diagonal_is_zero(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        for i in range(4):
            assert matrix[i, i].item() == 0.0

    def test_matrix_is_symmetric(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        assert torch.allclose(matrix, matrix.T)

    def test_output_is_scalar(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        loss_fn = HierarchicalLoss(matrix, LABEL_LIST)

        logits = torch.randn(4, 4)
        targets = torch.zeros(4, 4)
        targets[0, 0] = 1.0
        targets[1, 1] = 1.0

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        loss_fn = HierarchicalLoss(matrix, LABEL_LIST)

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()
        targets[:, 0] = 1  # Ensure at least one positive

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_perfect_prediction_low_penalty(self) -> None:
        matrix = HierarchicalLoss.build_distance_matrix(LABEL_LIST)
        loss_fn = HierarchicalLoss(matrix, LABEL_LIST)

        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # Perfect logits: high for true, very low for others
        good_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        bad_logits = torch.tensor([[-10.0, 10.0, 10.0, 10.0]])

        good_loss = loss_fn(good_logits, targets)
        bad_loss = loss_fn(bad_logits, targets)

        assert good_loss < bad_loss


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    """Tests for the CombinedLoss wrapper."""

    def test_returns_dict_with_total(self) -> None:
        focal = FocalLoss(gamma=2.0)
        combined = CombinedLoss(focal, hierarchical_loss=None)

        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        result = combined(logits, targets)
        assert "total" in result
        assert "focal" in result

    def test_with_hierarchical(self) -> None:
        labels = LABEL_LIST
        focal = FocalLoss(gamma=2.0)
        matrix = HierarchicalLoss.build_distance_matrix(labels)
        hier = HierarchicalLoss(matrix, labels)
        combined = CombinedLoss(focal, hier, hierarchical_weight=0.1)

        logits = torch.randn(4, 4)
        targets = torch.zeros(4, 4)
        targets[:, 0] = 1.0

        result = combined(logits, targets)
        assert "total" in result
        assert "focal" in result
        assert "hierarchical" in result
        # Total should be focal + weight * hierarchical
        expected = result["focal"] + 0.1 * result["hierarchical"]
        assert torch.allclose(result["total"], expected, atol=1e-5)

    def test_without_hierarchical(self) -> None:
        focal = FocalLoss(gamma=2.0)
        combined = CombinedLoss(focal, hierarchical_loss=None)

        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        result = combined(logits, targets)
        assert torch.allclose(result["total"], result["focal"])
