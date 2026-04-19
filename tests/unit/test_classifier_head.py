# tests/unit/test_classifier_head.py
"""Tests for the classifier head module."""

from __future__ import annotations

import torch

from src.models.classifier_head import ClassifierHead


class TestClassifierHead:
    """Tests for the ClassifierHead MLP."""

    def test_output_shape(self) -> None:
        head = ClassifierHead(input_dim=128, num_classes=5)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 5)

    def test_custom_hidden_dims(self) -> None:
        head = ClassifierHead(input_dim=256, num_classes=3, hidden_dims=[128, 64])
        x = torch.randn(2, 256)
        out = head(x)
        assert out.shape == (2, 3)

    def test_single_hidden_layer(self) -> None:
        head = ClassifierHead(input_dim=64, num_classes=2, hidden_dims=[32])
        x = torch.randn(3, 64)
        out = head(x)
        assert out.shape == (3, 2)

    def test_no_activation_at_output(self) -> None:
        """Output should be raw logits (can be negative)."""
        head = ClassifierHead(input_dim=64, num_classes=3, hidden_dims=[32])
        x = torch.randn(100, 64)
        out = head(x)
        # With random inputs, some logits should be negative
        assert out.min().item() < 0

    def test_gradient_flow(self) -> None:
        head = ClassifierHead(input_dim=64, num_classes=3, hidden_dims=[32])
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 64)

    def test_different_activations(self) -> None:
        for act in ["gelu", "relu"]:
            head = ClassifierHead(input_dim=32, num_classes=2, activation=act)
            x = torch.randn(2, 32)
            out = head(x)
            assert out.shape == (2, 2)

    def test_dropout_effect_in_train_mode(self) -> None:
        """In training mode with dropout, outputs
        should vary between forward passes."""
        head = ClassifierHead(input_dim=64, num_classes=3, dropout=0.5)
        head.train()
        x = torch.randn(4, 64)
        out1 = head(x)
        out2 = head(x)
        # Very unlikely to be exactly equal with 50% dropout
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, outputs should be deterministic."""
        head = ClassifierHead(input_dim=64, num_classes=3, dropout=0.5)
        head.eval()
        x = torch.randn(4, 64)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)
