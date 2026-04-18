# src/training/callbacks.py
"""
Custom PyTorch Lightning callbacks.

Provides monitoring and logging callbacks for training.
    - VRAMMonitorCallback: Logs GPU memory usage per step
    - GradientNormCallback: Monitors gradient norms for stability
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VRAMMonitorCallback(pl.Callback):
    """
    Log GPU VRAM usage at configurable intervals.

    Logs current allocated, reserved, and peak memory to the trainer's
    logger (e.g., MLflow, TensorBoard).

    Args:
        log_every_n_steps: How often to log memory stats (in training steps).
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not torch.cuda.is_available():
            return

        if trainer.global_step % self.log_every_n_steps != 0:
            return

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        pl_module.log("gpu/allocated_gb", allocated, on_step=True, on_epoch=False)
        pl_module.log("gpu/reserved_gb", reserved, on_step=True, on_epoch=False)
        pl_module.log("gpu/peak_gb", peak, on_step=True, on_epoch=False)


class GradientNormCallback(pl.Callback):
    """Log gradient norms per parameter group
    for training stability monitoring.

    Useful for detecting gradient explosions or vanishing gradients,
    especially when fine-tuning large pre-trained models with LoRA.

    Args:
        log_every_n_steps: How often to log gradient norms.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Compute gradient norms for backbone vs head
        backbone_grads = []
        head_grads = []

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if "backbone" in name:
                    backbone_grads.append(grad_norm)
                elif "classifier" in name:
                    head_grads.append(grad_norm)

        if backbone_grads:
            avg = sum(backbone_grads) / len(backbone_grads)
            pl_module.log("grad/backbone_norm", avg, on_step=True, on_epoch=False)

        if head_grads:
            avg = sum(head_grads) / len(head_grads)
            pl_module.log("grad/head_norm", avg, on_step=True, on_epoch=False)
