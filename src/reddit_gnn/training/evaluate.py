"""Metric computation: accuracy, balanced accuracy, F1 (macro + per-class),
ROC-AUC, PR-AUC, confusion matrix.

All metrics expect ``y_true ∈ {0,1}`` (i.e. POST_LABEL already remapped).
"""

from __future__ import annotations

from typing import Any


def compute_metrics(y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, Any]:
    """Return the full metric dict; ``y_score`` enables ROC/PR-AUC."""
    raise NotImplementedError("training.evaluate.compute_metrics is not implemented yet")


def confusion_matrix(y_true: Any, y_pred: Any) -> Any:
    """Return a 2x2 confusion matrix with rows = true, cols = predicted."""
    raise NotImplementedError("training.evaluate.confusion_matrix is not implemented yet")


def evaluate_model(model: Any, loader: Any, device: str) -> dict[str, Any]:
    """Run inference, then ``compute_metrics``. Returns metrics + predictions."""
    raise NotImplementedError("training.evaluate.evaluate_model is not implemented yet")


__all__ = ["compute_metrics", "confusion_matrix", "evaluate_model"]
