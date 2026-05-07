# tests/unit/test_callbacks.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytorch_lightning as pl
import torch

from src.training.callbacks import GradientNormCallback, VRAMMonitorCallback


class TestVRAMMonitorCallback:
    def test_init(self):
        cb = VRAMMonitorCallback(log_every_n_steps=10)
        assert cb.log_every_n_steps == 10

    def test_on_train_batch_end_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cb = VRAMMonitorCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        pl_module = MagicMock(spec=pl.LightningModule)

        cb.on_train_batch_end(trainer, pl_module, None, None, 0)

        pl_module.log.assert_not_called()

    def test_on_train_batch_end_wrong_step(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        cb = VRAMMonitorCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 5
        pl_module = MagicMock(spec=pl.LightningModule)

        cb.on_train_batch_end(trainer, pl_module, None, None, 0)

        pl_module.log.assert_not_called()

    def test_on_train_batch_end_logs_vram(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 2 * 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 1024**3)

        cb = VRAMMonitorCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 20
        pl_module = MagicMock(spec=pl.LightningModule)

        cb.on_train_batch_end(trainer, pl_module, None, None, 0)

        assert pl_module.log.call_count == 3
        pl_module.log.assert_any_call("gpu/allocated_gb", 1.0, on_step=True, on_epoch=False)
        pl_module.log.assert_any_call("gpu/reserved_gb", 2.0, on_step=True, on_epoch=False)
        pl_module.log.assert_any_call("gpu/peak_gb", 3.0, on_step=True, on_epoch=False)


class TestGradientNormCallback:
    def test_init(self):
        cb = GradientNormCallback(log_every_n_steps=5)
        assert cb.log_every_n_steps == 5

    def test_on_before_optimizer_step_wrong_step(self):
        cb = GradientNormCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 5
        pl_module = MagicMock(spec=pl.LightningModule)
        optimizer = MagicMock()

        cb.on_before_optimizer_step(trainer, pl_module, optimizer)

        pl_module.log.assert_not_called()

    def test_on_before_optimizer_step_logs_norms(self):
        cb = GradientNormCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 20
        pl_module = MagicMock(spec=pl.LightningModule)
        optimizer = MagicMock()

        # Mock parameters
        param1 = MagicMock()
        param1.grad.data.norm.return_value.item.return_value = 2.0

        param2 = MagicMock()
        param2.grad.data.norm.return_value.item.return_value = 4.0

        param3 = MagicMock()
        param3.grad.data.norm.return_value.item.return_value = 1.0

        param_no_grad = MagicMock()
        param_no_grad.grad = None

        pl_module.named_parameters.return_value = [
            ("backbone.layer1", param1),
            ("backbone.layer2", param2),
            ("classifier.head", param3),
            ("other", param_no_grad),
        ]

        cb.on_before_optimizer_step(trainer, pl_module, optimizer)

        assert pl_module.log.call_count == 2
        # (2.0 + 4.0) / 2 = 3.0
        pl_module.log.assert_any_call("grad/backbone_norm", 3.0, on_step=True, on_epoch=False)
        # 1.0 / 1 = 1.0
        pl_module.log.assert_any_call("grad/head_norm", 1.0, on_step=True, on_epoch=False)

    def test_on_before_optimizer_step_empty_grads(self):
        cb = GradientNormCallback(log_every_n_steps=10)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 20
        pl_module = MagicMock(spec=pl.LightningModule)
        optimizer = MagicMock()

        pl_module.named_parameters.return_value = []

        cb.on_before_optimizer_step(trainer, pl_module, optimizer)

        pl_module.log.assert_not_called()
