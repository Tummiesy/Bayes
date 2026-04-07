"""Evaluation metrics and artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, float]:
    """Compute common multiclass metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_classification_report(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
) -> str:
    """Return sklearn classification report as a string."""
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=4,
    )


def save_confusion_matrix_plot(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
    output_path: Path,
    title: str,
) -> None:
    """Save confusion matrix plot for the provided predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    num_classes = len(labels)
    fig_width = max(8, min(24, 0.45 * num_classes))
    fig_height = max(6, min(20, 0.45 * num_classes))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        include_values=num_classes <= 30,
        xticks_rotation=90 if num_classes > 10 else 45,
    )
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_macro_f1_barplot(dataset_to_macro_f1: Dict[str, float], output_path: Path) -> None:
    """Save a simple bar chart for best test macro-F1 by dataset."""
    datasets = list(dataset_to_macro_f1.keys())
    values = [dataset_to_macro_f1[name] for name in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(datasets, values)
    ax.set_title("Best Test Macro-F1 by Dataset")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, min(1.0, max(values) + 0.1 if values else 1.0))

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
