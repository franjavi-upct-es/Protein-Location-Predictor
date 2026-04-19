# src/evaluation/metrics.py
"""
Comprehensive evaluation metrics for multi-label protein localization.

Generates per-class metrics, confusion-style analysis, calibration
assessment, and publication-ready visualizations.

Usage::

    from src.evaluation.metrics import evaluate_model, generate_report
    results = evaluate_model(model, dataloader, label_list)
    generate_report(results, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(
    model: Any,
    dataloader: Any,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Run inference on a dataloader and collect predictions.

    Args:
        model: Trained Lightning module or nn.Module with
            forward(input_ids, attention_mask).
        dataloader: PyTorch DataLoader yielding dicts with
            input_ids, attention_mask, labels.
        device: Device to run inference on.
        threshold: Sigmoid threshold for positive prediction.

    Returns:
        Dict with keys:
            - probabilities: (N, C) array of sigmoid probabilities
            - predictions: (N, C) binary array of threshold predictions
            - targets: (N, C) binary array of ground truth labels
    """
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)

    all_probs = []
    all_targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"]

        external_features = batch.get("external_features")
        if external_features is not None:
            external_features = external_features.to(device)

        if hasattr(model, "forward"):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                external_features=external_features,
            )
        else:
            logits = model(input_ids, attention_mask)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "targets": targets,
    }


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_list: list[str],
) -> dict[str, Any]:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        predictions: Binary prediction array (N, C).
        targets: Binary target array (N, C).
        label_list: Ordered list of class names.

    Returns:
        Dict with overall and per-class metrics.
    """
    from sklearn.metrics import (
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    # Overall metrics
    overall = {
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0.0),
        "f1_micro": f1_score(targets, predictions, average="micro", zero_division=0.0),
        "f1_weighted": f1_score(targets, predictions, average="weighted", zero_division=0.0),
        "precision_macro": precision_score(
            targets, predictions, average="macro", zero_division=0.0
        ),
        "recall_macro": recall_score(targets, predictions, average="macro", zero_division=0.0),
        "hamming_loss": hamming_loss(targets, predictions),
        "exact_match_ratio": np.mean(np.all(predictions == targets, axis=1)),
        "total_samples": len(targets),
    }

    # Per-class metrics
    per_class = {}
    for i, label in enumerate(label_list):
        t = targets[:, i]
        p = predictions[:, i]
        support = int(t.sum())

        per_class[label] = {
            "precision": precision_score(t, p, zero_division=0.0),
            "recall": recall_score(t, p, zero_division=0.0),
            "f1": f1_score(t, p, zero_division=0.0),
            "support": support,
            "predicted_positive": int(p.sum()),
        }

    return {"overall": overall, "per_class": per_class}


def format_classification_report(
    metrics: dict[str, Any],
    label_list: list[str],
) -> str:
    """
    Format metrics as a human-readable classification report.

    Args:
        metrics: Output from compute_metrics().
        label_list: Ordered list of class names.

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append(f"{'Class':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 70)

    for label in label_list:
        m = metrics["per_class"][label]
        lines.append(
            f"{label:<28} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>10d}"
        )

    lines.append("-" * 70)
    o = metrics["overall"]
    lines.append(
        f"{'Macro avg':<28} "
        f"{o['precision_macro']:>10.3f} "
        f"{o['recall_macro']:>10.3f} "
        f"{o['f1_macro']:>10.3f} "
        f"{o['total_samples']:>10d}"
    )
    lines.append(f"\nExact match ratio: {o['exact_match_ratio']:.3f}")
    lines.append(f"Hamming loss: {o['hamming_loss']:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    metrics: dict[str, Any],
    label_list: list[str],
    output_path: Path | str,
) -> None:
    """Generate a bar chart of per-class F1 scores.

    Args:
        metrics: Output from compute_metrics().
        label_list: Ordered list of class names.
        output_path: Path to save the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1_scores = [metrics["per_class"][label]["f1"] for label in label_list]
    supports = [metrics["per_class"][label]["support"] for label in label_list]

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(label_list, f1_scores, color="#4C72B0", edgecolor="white")

    # Add support counts as text
    for bar, support in zip(bars, supports, strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={support}",
            va="center",
            fontsize=9,
            color="#666",
        )

    ax.set_xlabel("F1 Score")
    ax.set_title("F1 Score by Location Class")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class F1 chart saved to {output_path}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_list: list[str],
    output_path: Path | str,
) -> None:
    """Generate a multi-label confusion-style co-occurrence heatmap.

    For multi-label classification, this shows the co-occurrence matrix:
    how often each predicted class appears alongside each true class.

    Args:
        predictions: Binary prediction array (N, C).
        targets: Binary target array (N, C).
        label_list: Ordered list of class names.
        output_path: Path to save the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(label_list)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(n_classes):
        for j in range(n_classes):
            # Count samples where true class is i and predicted class is j
            matrix[i, j] = int(((targets[:, i] == 1) & (predictions[:, j] == 1)).sum())

    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(label_list, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_list, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Prediction Co-occurrence Matrix")

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def generate_report(
    metrics: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    label_list: list[str],
    output_dir: Path | str,
) -> None:
    """Generate a full evaluation report with metrics and visualizations.

    Args:
        metrics: Output from compute_metrics().
        predictions: Binary prediction array.
        targets: Binary target array.
        label_list: Ordered list of class names.
        output_dir: Directory to save report files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Text report
    report = format_classification_report(metrics, label_list)
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    print(report)

    # Metrics CSV
    per_class_df = pd.DataFrame(metrics["per_class"]).T
    per_class_df.to_csv(output_dir / "per_class_metrics.csv")

    # Visualizations
    try:
        plot_per_class_f1(metrics, label_list, figures_dir / "f1_scores_by_class.png")
        plot_confusion_matrix(
            predictions,
            targets,
            label_list,
            figures_dir / "confusion_matrix.png",
        )
    except ImportError:
        logger.warning("matplotlib not available — skipping visualizations")
