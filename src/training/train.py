# src/training/train.py
"""
Main training entry point.

Orchestrates the full training pipeline:
  1. Load config, set up logging and reproducibility
  2. Detect hardware and resolve training parameters
  3. Load data and determine label set
  4. Build model (ESM-2 + LoRA + classifier head)
  5. Configure Lightning Trainer with callbacks and MLflow
  6. Train with early stopping and checkpointing

Usage:
    uv run python -m src.training.train
    uv run python -m src.training.train \
        --overrides model.lora.rank=16 training.max_epochs=5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.data.datasets import ProteinDataModule
from src.models.lightning_module import ProteinLocalizationModule
from src.training.callbacks import GradientNormCallback, VRAMMonitorCallback
from src.utils.config import DotDict, load_config, resolve_path
from src.utils.hardware import detect_hardware
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import seed_everything

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Label and class frequency discovery
# ---------------------------------------------------------------------------


def _discover_labels(cfg: DotDict) -> tuple[list[str], torch.Tensor]:
    """Discover the ordered label list and class
    frequencies from the training split.

    Returns:
        Tuple of (label_list, class_frequencies) where class_frequencies is a
        tensor of per-class sample counts in the training set.
    """
    splits_dir = resolve_path(cfg, "paths.splits_dir")
    train_path = splits_dir / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found at {train_path}. "
            "Run the data pipeline first (make download && make process)."
        )

    df = pd.read_csv(train_path)
    if "locations_str" not in df.columns:
        raise ValueError("Training split missing 'locations_str' column.")

    # Collect all labels across the training set
    all_labels = []
    for loc_str in df["locations_str"].dropna():
        all_labels.extend(loc_str.split("|"))

    label_counts = pd.Series(all_labels).value_counts()
    label_list = sorted(label_counts.index.tolist())

    frequencies = torch.tensor(
        [label_counts.get(label, 0) for label in label_list],
        dtype=torch.float32,
    )

    logger.info(f"Discovered {len(label_list)} classes from training data:")
    for label, count in zip(label_list, frequencies.tolist(), strict=True):
        logger.info(f"  {label}: {int(count)} samples")

    return label_list, frequencies


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------


def _resolve_mlflow_tracking_uri(cfg: DotDict) -> str:
    """Resolve the MLflow tracking URI from config."""
    exp_cfg = cfg.training.get("experiment", {})
    tracking_uri = exp_cfg.get("tracking_uri")

    if tracking_uri:
        if "://" in tracking_uri:
            return str(tracking_uri)
        return str(Path(cfg.project_root) / tracking_uri)

    return str(resolve_path(cfg, "paths.mlflow_dir"))


def _configure_torch_runtime(hw_profile: object) -> None:
    """Tune PyTorch runtime settings for the detected hardware."""
    if getattr(hw_profile, "device", None) == "cuda" and hasattr(
        torch, "set_float32_matmul_precision"
    ):
        torch.set_float32_matmul_precision("high")
        logger.info(
            "Set float32 matmul precision to high for CUDA Tensor Cores."
        )


def _build_callbacks(cfg: DotDict) -> list[pl.Callback]:
    """Build the list of Lightning callbacks from config."""
    training_cfg = cfg.training
    exp_cfg = training_cfg.get("experiment", {})

    callbacks: list[pl.Callback] = []

    # Model checkpointing (save best by val F1)
    models_dir = resolve_path(cfg, "paths.models_dir")
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(models_dir / "checkpoints"),
            filename="epoch={epoch}-val_f1={val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=exp_cfg.get("save_top_k", 3),
            auto_insert_metric_name=False,
            save_last=True,
        )
    )

    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor="val/f1_macro",
            mode="max",
            patience=training_cfg.get("patience", 5),
            min_delta=0.001,
            verbose=True,
        )
    )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # VRAM monitoring (only on CUDA)
    callbacks.append(
        VRAMMonitorCallback(
            log_every_n_steps=exp_cfg.get("log_every_n_steps", 10),
        )
    )

    # Gradient norm monitoring
    callbacks.append(
        GradientNormCallback(
            log_every_n_steps=exp_cfg.get("log_every_n_steps", 10),
        )
    )

    return callbacks


def _build_trainer(cfg: DotDict, hw_profile: object) -> pl.Trainer:
    """Build the Lightning Trainer with hardware-aware configuration."""
    training_cfg = cfg.training
    exp_cfg = training_cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "protein-loc-v2")
    tracking_uri = _resolve_mlflow_tracking_uri(cfg)

    callbacks = _build_callbacks(cfg)

    # Logger (MLflow)
    try:
        from pytorch_lightning.loggers import MLFlowLogger

        mlflow_logger = MLFlowLogger(
            experiment_name=exp_name,
            tracking_uri=tracking_uri,
            log_model=False,  # We handle checkpoints separately
        )
        pl_logger = mlflow_logger
        logger.info(f"MLflow tracking enabled: {exp_name} ({tracking_uri})")
    except (ImportError, Exception) as e:
        logger.warning(f"MLflow not available, using CSV logger: {e}")
        from pytorch_lightning.loggers import CSVLogger

        pl_logger = CSVLogger(
            save_dir=str(resolve_path(cfg, "paths.reports_dir")),
            name="training_logs",
        )

    # Resolve precision from hardware profile
    precision = getattr(
        hw_profile, "precision", training_cfg.get("precision", "32")
    )

    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 30),
        precision=precision,
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=exp_cfg.get("log_every_n_steps", 10),
        deterministic=training_cfg.get("deterministic", True),
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
    )

    return trainer


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(cfg: DotDict) -> None:
    """Run the full training pipeline.

    Args:
        cfg: Merged project configuration.
    """
    # 1. Reproducibility
    seed = cfg.project.get("seed", 42)
    deterministic = cfg.training.get("deterministic", True)
    seed_everything(seed, deterministic=deterministic)

    # 2. Hardware detection
    hw = detect_hardware(cfg)
    _configure_torch_runtime(hw)

    # Apply hardware-recommended settings if not manually overridden
    training_cfg = cfg.training
    if training_cfg.get("batch_size") is None:
        training_cfg["batch_size"] = hw.batch_size
    if training_cfg.get("precision") is None:
        training_cfg["precision"] = hw.precision
    if training_cfg.get("gradient_checkpointing") is None:
        training_cfg["gradient_checkpointing"] = hw.gradient_checkpointing
    if training_cfg.get("max_sequence_length") is None:
        training_cfg["max_sequence_length"] = hw.max_sequence_length

    logger.info(
        f"Training config: batch_size={training_cfg.batch_size}, "
        f"precision={training_cfg.precision}, "
        f"grad_checkpoint={training_cfg.gradient_checkpointing}, "
        f"max_seq_len={training_cfg.max_sequence_length}"
    )

    # 3. Discover labels
    label_list, class_frequencies = _discover_labels(cfg)

    # 4. Build DataModule
    dm = ProteinDataModule(cfg, label_list=label_list)

    # 5. Build model
    model = ProteinLocalizationModule(
        cfg=cfg,
        label_list=label_list,
        class_frequencies=class_frequencies,
    )

    # 6. Build trainer
    trainer = _build_trainer(cfg, hw)

    # 7. Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=dm)

    # 8. Test with best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        logger.info(f"Testing with best checkpoint: {best_path}")
        trainer.test(model, datamodule=dm, ckpt_path=best_path)
    else:
        logger.warning(
            "No best checkpoint found — testing with last model state"
        )
        trainer.test(model, datamodule=dm)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the training CLI command."""
    parser = argparse.ArgumentParser(
        description="Train the protein localization model."
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format.",
    )
    args = parser.parse_args()

    cfg = load_config(mode="training", overrides=args.overrides)
    setup_logging(level=cfg.project.log_level)

    train(cfg)


if __name__ == "__main__":
    main()
