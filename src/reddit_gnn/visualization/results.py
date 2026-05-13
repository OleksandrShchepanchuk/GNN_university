"""Training-curve, confusion-matrix, and ROC/PR plots."""

from __future__ import annotations

from typing import Any


def plot_training_curves(history: dict[str, list[float]], ax: Any | None = None) -> Any:
    """Loss and primary metric over epochs."""
    raise NotImplementedError("visualization.results.plot_training_curves is not implemented yet")


def plot_confusion_matrix(cm: Any, class_names: list[str], ax: Any | None = None) -> Any:
    """Annotated 2x2 confusion matrix."""
    raise NotImplementedError("visualization.results.plot_confusion_matrix is not implemented yet")


def plot_roc_pr(y_true: Any, y_score: Any, ax: Any | None = None) -> Any:
    """Side-by-side ROC and Precision-Recall curves."""
    raise NotImplementedError("visualization.results.plot_roc_pr is not implemented yet")


__all__ = ["plot_confusion_matrix", "plot_roc_pr", "plot_training_curves"]
