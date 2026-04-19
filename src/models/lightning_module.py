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

from typing import Any, TypedDict

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
from src.utils.config import DotDict, to_builtin
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
        cfg: DotDict | dict[str, Any],
        label_list: list[str],
        class_frequencies: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(cfg, DotDict):
            cfg = DotDict.from_dict(cfg)

        # Save only builtin containers so future checkpoints remain compatible
        # with PyTorch's weights_only checkpoint loader.
        self.save_hyperparameters(
            {
                "cfg": to_builtin(cfg),
                "label_list": list(label_list),
            }
        )

        self.cfg = cfg
        self.label_list = label_list
        self.num_classes = len(label_list)

        training_cfg = cfg.get("training", {})
        self.enable_gc = training_cfg.get("gradient_checkpointing", True)
        self.pooling = cfg.model.get("pooling", "mean")

        # Optional backbone
        self.attention_pooler: torch.nn.Module | None = None
        if self.pooling == "light_attention":
            from src.models.pooling import LightAttentionPooler

            self.attention_pooler = LightAttentionPooler(
                hidden_dim=get_embedding_dim(cfg),
                dropout=float(cfg.model.get("pooling_dropout", 0.1)),
            )

        # Build backbone
        self.backbone = build_esm_lora_backbone(
            cfg,
            enable_gradient_checkpointing=self.enable_gc,
        )

        # Build classifier head
        emb_dim = get_embedding_dim(cfg)
        if self.pooling == "mean_cls":
            emb_dim *= 2
        # light_attention preserves the original hidden dimension

        # Account for optional external features
        ext_dim = 0
        feat_cfg = cfg.get("features", {})
        if feat_cfg.get("biophysical", {}).get("enabled", False):
            ext_dim += len(feat_cfg.biophysical.get("properties", []))
        input_dim = emb_dim + ext_dim

        self.classifier = ClassifierHead.from_config(cfg, input_dim, self.num_classes)

        # Optional auxiliary multi-task head
        multi_task_cfg = cfg.get("multi_task", {}) or {}
        self.use_multi_task = bool(multi_task_cfg.get("enabled", False))
        self.aux_loss_weight = float(multi_task_cfg.get("loss_weight", 0.2))
        self.aux_head: torch.nn.Module | None = None
        if self.use_multi_task:
            from src.data.aux_targets import AUX_TARGET_NAMES

            self.num_aux = len(AUX_TARGET_NAMES)
            self.aux_head = torch.nn.Linear(emb_dim, self.num_aux)
            logger.info(
                f"Multi-task enabled: {self.num_aux} auxiliary targets, "
                f"loss_weight={self.aux_loss_weight}"
            )

        # Build loss
        self.criterion = CombinedLoss.from_config(cfg, label_list, class_frequencies)

        # Metrics
        self.train_f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.val_precision = MultilabelPrecision(num_labels=self.num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_labels=self.num_classes, average="macro")
        self.test_f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")

        # Per-class F1 for detailed logging
        self.val_f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average="none")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        external_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through backbone + classifier.

        Returns the main task logits as a tensor for backward compatibility.
        Use ``forward_with_aux`` if you also need the auxiliary logits.
        """
        return self.forward_with_aux(input_ids, attention_mask, external_features)["logits"]

    def forward_with_aux(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        external_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning both main and (optional) auxiliary logits.

        Returns a dict with:
            logits: (B, num_classes) main task
            aux_logits: (B, num_aux) auxiliary task, only if enabled
        """
        # ESM-2 forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Pool to sequence-level representation
        if self.pooling == "light_attention":
            assert self.attention_pooler is not None
            pooled = self.attention_pooler(outputs.last_hidden_state, attention_mask)
        else:
            pooled = extract_sequence_representation(
                outputs,
                attention_mask,
                pooling=self.pooling,
            )

        # Auxiliary head reads the pooled embedding *before* external features
        # are concatenated, so it depends only on the backbone signal
        result: dict[str, torch.Tensor] = {}
        if self.aux_head is not None:
            result["aux_logits"] = self.aux_head(pooled)

        # Concatenate external features for the main classifier
        if external_features is not None:
            embeddings = torch.cat([pooled, external_features], dim=-1)
        else:
            embeddings = pooled

        result["logits"] = self.classifier(embeddings)
        return result

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> StepResult:
        """Shared logic for train/val/test steps."""
        forward_out = self.forward_with_aux(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            external_features=batch.get("external_features"),
        )
        logits = forward_out["logits"]
        targets = batch["labels"]

        losses = self.criterion(logits, targets)
        total_loss = losses["total"]

        # Auxiliary loss (binary cross-entropy on the aux head)
        if "aux_logits" in forward_out and "aux_labels" in batch and self.aux_loss_weight > 0:
            aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                forward_out["aux_logits"], batch["aux_labels"]
            )
            losses["aux"] = aux_loss
            total_loss = total_loss + self.aux_loss_weight * aux_loss

        preds = (torch.sigmoid(logits) > 0.5).int()

        return {
            "loss": total_loss,
            "losses": losses,
            "preds": preds,
            "targets": targets,
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
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
        self.log("val/precision", self.val_precision, on_epoch=True, sync_dist=True)
        self.log("val/recall", self.val_recall, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Log per-class F1 scores at the end of each validation epoch."""
        per_class = self.val_f1_per_class.compute()
        for i, label in enumerate(self.label_list):
            self.log(f"val/f1_{label}", per_class[i], sync_dist=True)
        self.val_f1_per_class.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
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
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
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
