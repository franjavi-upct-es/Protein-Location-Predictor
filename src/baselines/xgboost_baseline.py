# src/baselines/xgboost_baseline.py
"""
Frozen ESM-2 + XGBoost baseline.

Replicates the v1.0 architecture (frozen ESM-2 embeddings → gradient
boosting classifier) over the v2.0 multi-species, homology-aware splits.

The point of this baseline is *not* to beat the linear probe — it
sometimes does, sometimes does not — but to provide an apples-to-apples
comparison with v1.0 on the new data: any difference between this
baseline and the v1.0 published numbers is purely a consequence of the
new data and splitting strategy, not of the model architecture.

Multi-label classification is handled by training one XGBoost classifier
per class (one-vs-rest), which is what the original v1.0 code did.

XGBoost is an optional dependency installed via::

    > uv sync --group baselines

Usage::

    > uv run python -m src.baselines.xgboost_baseline
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
    splits_dir = resolve_path(cfg, "paths.splits_dir")
    train_df = pd.read_csv(splits_dir / "train.csv")
    all_labels: list[str] = []
    for s in train_df["locations_str"].dropna():
        all_labels.extend(str(s).split("|"))
    return sorted(set(all_labels))


def run_xgboost_baseline(
    cfg: DotDict,
    label_list: list[str] | None = None,
    output_dir: Path | None = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict:
    """Train one-vs-rest XGBoost classifiers and evaluate them."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "XGBoost is not installed. Install the baselines group:\n    uv sync --group baselines"
        ) from e

    if label_list is None:
        label_list = _discover_label_list(cfg)

    logger.info(f"XGBoost baseline on {len(label_list)} classes: {label_list}")

    X_train, y_train, _ = compute_or_load_embeddings(cfg, "train", label_list)
    X_val, y_val, _ = compute_or_load_embeddings(cfg, "val", label_list)
    X_test, y_test, _ = compute_or_load_embeddings(cfg, "test", label_list)

    logger.info(f"Embeddings ready: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    classifiers = []
    n_classes = len(label_list)
    for c_idx, label in enumerate(label_list):
        n_pos = int(y_train[:, c_idx].sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = max(1.0, n_neg / max(1, n_pos))

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        logger.info(
            f"  fitting class {c_idx + 1}/{n_classes} ({label}, pos={n_pos}, neg={n_neg})..."
        )
        clf.fit(X_train, y_train[:, c_idx].astype(int))
        classifiers.append(clf)

    def _predict(features: np.ndarray) -> np.ndarray:
        out = np.zeros((features.shape[0], n_classes), dtype=np.int64)
        for c_idx, clf in enumerate(classifiers):
            out[:, c_idx] = clf.predict(features).astype(np.int64)
        return out

    train_preds = _predict(X_train)
    val_preds = _predict(X_val)
    test_preds = _predict(X_test)

    train_metrics = compute_metrics(train_preds, y_train.astype(int), label_list)
    val_metrics = compute_metrics(val_preds, y_val.astype(int), label_list)
    test_metrics = compute_metrics(test_preds, y_test.astype(int), label_list)

    summary = {
        "model": "xgboost_baseline",
        "backbone": str(cfg.model.backbone.name),
        "pooling": str(cfg.model.get("pooling", "mean")),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_classes": n_classes,
        "label_list": label_list,
        "train": train_metrics["overall"],
        "val": val_metrics["overall"],
        "test": test_metrics["overall"],
        "test_per_class": test_metrics["per_class"],
    }

    logger.info(
        f"XGBoost baseline results: "
        f"train F1={train_metrics['overall']['f1_macro']:.3f}, "
        f"val F1={val_metrics['overall']['f1_macro']:.3f}, "
        f"test F1={test_metrics['overall']['f1_macro']:.3f}"
    )

    if output_dir is None:
        output_dir = resolve_path(cfg, "paths.reports_dir") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "xgboost_baseline.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info(f"XGBoost baseline report written to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the frozen ESM-2 + XGBoost baseline."
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides.")
    args = parser.parse_args()

    cfg = load_config(mode="training", overrides=args.overrides)
    setup_logging(level=cfg.project.log_level)

    run_xgboost_baseline(
        cfg,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
