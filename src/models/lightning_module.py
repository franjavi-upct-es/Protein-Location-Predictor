# src/models/lightning_module.py
"""
PyTorch Lightning module for end-to-end training.

Wraps the ESM-2 + LoRA backbone and classifier head into a single
trainable module with differential learning rates, cosine warmup
scheduling, and comprehensive metric logging.

Usage::

    from src.models.lightning_module import ProteinLocalizationModule
    model = ProteinLocalizationModule(cfg, label_list)
"""

from __future__ import annotations

from typing import TypedDict, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from src.models.classifier_head import ClassifierHead
from src.models.esm_lora import (
    build_esm_lora_backbone,
    extract_sequence_representation,
    get_embedding_dim,
)
from src.models.losses import CombinedLoss
from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StepResult(TypedDict):
    """Typed structure returned by the shared train/val/test step."""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    preds: torch.Tensor
    targets: torch.Tensor


class ProteinLocalizationModule(pl.LightningModule):
    """
    Lightning module for multi-label protein localization.

    Components:
      - ESM-2 backbone with LoRA adapters (low LR)
      - Classification MLP head (higher LR)
      - CombinedLoss (Focal + Hierarchical)
      - Per-class and macro F1/Precision/Recall metrics

    Args:
        cfg: Project configuration.
        label_list: Ordered list of location class names.
        class_frequencies: Optional per-class sample counts for loss weighting.
    """

    def __init__(
        self,
        cfg: DotDict,
        label_list: list[str],
        class_frequencies: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_frequencies"])

        self.cfg = cfg
        self.label_list = label_list
        self.num_classes = len(label_list)

        training_cfg = cfg.get("training", {})
        self.enable_gc = training_cfg.get("gradient_checkpointing", True)
        self.pooling = cfg.model.get("pooling", "mean")

        # Build backbone
        self.backbone = build_esm_lora_backbone(
            cfg,
            enable_gradient_checkpointing=self.enable_gc,
        )

        # Build classifier head
        emb_dim = get_embedding_dim(cfg)
        if self.pooling == "mean_cls":
            emb_dim *= 2

        # Account for optional external features
        ext_dim = 0
        feat_cfg = cfg.get("features", {})
        if feat_cfg.get("biophysical", {}).get("enabled", False):
            ext_dim += len(feat_cfg.biophysical.get("properties", []))
        input_dim = emb_dim + ext_dim

        self.classifier = ClassifierHead.from_config(
            cfg, input_dim, self.num_classes
        )

        # Build loss
        self.criterion = CombinedLoss.from_config(
            cfg, label_list, class_frequencies
        )

        # Metrics
        self.train_f1 = MultilabelF1Score(
            num_labels=self.num_classes, average="macro"
        )
        self.val_f1 = MultilabelF1Score(
            num_labels=self.num_classes, average="macro"
        )
        self.val_precision = MultilabelPrecision(
            num_labels=self.num_classes, average="macro"
        )
        self.val_recall = MultilabelRecall(
            num_labels=self.num_classes, average="macro"
        )
        self.test_f1 = MultilabelF1Score(
            num_labels=self.num_classes, average="macro"
        )

        # Per-class F1 for detailed logging
        self.val_f1_per_class = MultilabelF1Score(
            num_labels=self.num_classes, average="none"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        external_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through backbone + classifier.

        Args:
            input_ids: Tokenized sequences (B, L).
            attention_mask: Attention mask (B, L).
            external_features: Optional biophysical features (B, F).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # ESM-2 forward
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Pool to sequence-level representation
        embeddings = extract_sequence_representation(
            outputs,
            attention_mask,
            pooling=self.pooling,
        )

        # Concatenate external features if provided
        if external_features is not None:
            embeddings = torch.cat([embeddings, external_features], dim=-1)

        # Classify
        return cast(torch.Tensor, self.classifier(embeddings))

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> StepResult:
        """Shared logic for train/val/test steps."""
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            external_features=batch.get("external_features"),
        )
        targets = batch["labels"]

        losses = self.criterion(logits, targets)
        preds = (torch.sigmoid(logits) > 0.5).int()

        return {
            "loss": losses["total"],
            "losses": losses,
            "preds": preds,
            "targets": targets,
        }

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        result = self._shared_step(batch, "train")

        self.log(
            "train/loss",
            result["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/focal",
            result["losses"]["focal"],
            on_step=False,
            on_epoch=True,
        )
        if "hierarchical" in result["losses"]:
            self.log(
                "train/hier",
                result["losses"]["hierarchical"],
                on_step=False,
                on_epoch=True,
            )

        self.train_f1(result["preds"], result["targets"].int())
        self.log(
            "train/f1_macro",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return result["loss"]

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        result = self._shared_step(batch, "val")

        self.log(
            "val/loss",
            result["loss"],
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        targets_int = result["targets"].int()
        self.val_f1(result["preds"], targets_int)
        self.val_precision(result["preds"], targets_int)
        self.val_recall(result["preds"], targets_int)
        self.val_f1_per_class(result["preds"], targets_int)

        self.log(
            "val/f1_macro",
            self.val_f1,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/precision", self.val_precision, on_epoch=True, sync_dist=True
        )
        self.log("val/recall", self.val_recall, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Log per-class F1 scores at the end of each validation epoch."""
        per_class = self.val_f1_per_class.compute()
        for i, label in enumerate(self.label_list):
            self.log(f"val/f1_{label}", per_class[i], sync_dist=True)
        self.val_f1_per_class.reset()

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        result = self._shared_step(batch, "test")
        self.log("test/loss", result["loss"], on_epoch=True, sync_dist=True)
        self.test_f1(result["preds"], result["targets"].int())
        self.log("test/f1_macro", self.test_f1, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer with differential learning
        rates and cosine warmup."""
        training_cfg = self.cfg.training
        opt_cfg = training_cfg.optimizer

        # Separate parameters into backbone (LoRA) and head groups
        backbone_params = [
            p for n, p in self.backbone.named_parameters() if p.requires_grad
        ]
        head_params = list(self.classifier.parameters())

        param_groups = [
            {"params": backbone_params, "lr": opt_cfg.lr},
            {
                "params": head_params,
                "lr": opt_cfg.get("head_lr", opt_cfg.lr * 5),
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Cosine warmup scheduler
        sched_cfg = training_cfg.get("scheduler", {})
        warmup_frac = sched_cfg.get("warmup_steps_fraction", 0.1)
        min_lr = sched_cfg.get("min_lr", 1e-6)

        # Estimate total steps
        total_steps = int(self.trainer.estimated_stepping_batches)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[opt_cfg.lr, opt_cfg.get("head_lr", opt_cfg.lr * 5)],
            total_steps=total_steps,
            pct_start=warmup_frac,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=opt_cfg.lr / min_lr if min_lr > 0 else 1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
