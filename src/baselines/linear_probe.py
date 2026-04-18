# src/baselines/linear_probe.py
"""
Linear-probe baseline on frozen ESM-2 embeddings.

This is the simplest possible "lower bound" baseline: take the frozen
backbone, mean-pool the token embeddings into one vector per protein,
and train a one-vs-rest logistic regression on top.

If your fine-tuned ESM-2 + LoRA model does not beat this, the
fine-tuning is not adding value and something is wrong with the
training pipeline (loss, learning rate, data, etc).

Usage (programmatic)::

    from src.baselines.linear_probe import run_linear_probe
    metrics = run_linear_probe(cfg, label_list)

Usage (CLI)::

    > uv run python -m src.baselines.linear_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.embedding_cache import compute_or_load_embeddings
from src.evaluation.metrics import compute_metrics
from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _discover_label_list(cfg: DotDict) -> list[str]:
    """Read the train split and discover the ordered label list."""
    splits_dir = resolve_path(cfg, "paths.splits_dir")
    train_df = pd.read_csv(splits_dir / "train.csv")
    all_labels: list[str] = []
    for s in train_df["locations_str"].dropna():
        all_labels.extend(str(s).split("|"))
    return sorted(set(all_labels))


def run_linear_probe(
    cfg: DotDict,
    label_list: list[str] | None = None,
    output_dir: Path | None = None,
    max_iter: int = 1000,
    regularization_strength: float = 1.0,
) -> dict:
    """
    Train a linear probe on frozen ESM-2 embeddings and evaluate it.

    Args:
        cfg: Project configuration.
        label_list: Ordered class labels. Auto-discovered if None.
        output_dir: Where to write the JSON metrics report.
        max_iter: Max iterations for the logistic regression solver.
        regularization_strength: Inverse regularization strength.

    Returns:
        Dict with train/val/test metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    if label_list is None:
        label_list = _discover_label_list(cfg)

    logger.info(f"Linear probe on {len(label_list)} classes: {label_list}")

    X_train, y_train, _ = compute_or_load_embeddings(cfg, "train", label_list)
    X_val, y_val, _ = compute_or_load_embeddings(cfg, "val", label_list)
    X_test, y_test, _ = compute_or_load_embeddings(cfg, "test", label_list)

    logger.info(f"Embeddings ready: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    clf = MultiOutputClassifier(
        LogisticRegression(
            max_iter=max_iter,
            C=regularization_strength,
            solver="lbfgs",
            class_weight="balanced",
        ),
        n_jobs=1,
    )

    logger.info("Fitting linear probe...")
    clf.fit(X_train, y_train)

    def _predict(features: np.ndarray) -> np.ndarray:
        # MultiOutputClassifier returns a list of (n_samples, 2) arrays;
        # we want a (n_samples, n_classes) binary matrix.
        preds = clf.predict(features)
        return np.asarray(preds, dtype=np.int64)

    train_preds = _predict(X_train)
    val_preds = _predict(X_val)
    test_preds = _predict(X_test)

    train_metrics = compute_metrics(train_preds, y_train.astype(int), label_list)
    val_metrics = compute_metrics(val_preds, y_val.astype(int), label_list)
    test_metrics = compute_metrics(test_preds, y_test.astype(int), label_list)

    summary = {
        "model": "linear_probe",
        "backbone": str(cfg.model.backbone.name),
        "pooling": str(cfg.model.get("pooling", "mean")),
        "n_classes": len(label_list),
        "label_list": label_list,
        "train": train_metrics["overall"],
        "val": val_metrics["overall"],
        "test": test_metrics["overall"],
        "test_per_class": test_metrics["per_class"],
    }

    logger.info(
        f"Linear probe results: "
        f"train F1={train_metrics['overall']['f1_macro']:.3f}, "
        f"val F1={val_metrics['overall']['f1_macro']:.3f}, "
        f"test F1={test_metrics['overall']['f1_macro']:.3f}"
    )

    if output_dir is None:
        output_dir = resolve_path(cfg, "paths.reports_dir") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "linear_probe.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info(f"Linear probe report written to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a frozen-ESM-2 linear probe.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations for the logistic regression.",
    )
    parser.add_argument(
        "--C",
        type=float,
        dest="regularization_strength",
        default=1.0,
        help="Inverse regularization strength for the logistic regression.",
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

    run_linear_probe(
        cfg,
        max_iter=args.max_iter,
        regularization_strength=args.regularization_strength,
    )


if __name__ == "__main__":
    main()
