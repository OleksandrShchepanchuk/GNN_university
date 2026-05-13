"""Classification metrics geared at the class-imbalanced edge sign task.

The headline numbers are computed for the **negative class** (``label == 0``,
which is the minority "negative sentiment" class with ~10% prevalence in
SNAP Reddit). PR-AUC for that class is the primary leaderboard metric;
F1-macro is the secondary report.

All functions accept ``y_score`` as the probability of class 1 (the output of
``torch.sigmoid(logits)``). Internally we flip both labels and scores when a
metric is defined with respect to class 0 (PR-AUC, precision, recall).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)


def _safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    """Return ``fn(*args, **kwargs)`` or ``default`` if the metric is undefined."""
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the full metric panel for binary edge sign classification.

    Returns a dict with:
        * ``pr_auc``              — PR-AUC for the NEGATIVE class (label 0).
        * ``pr_auc_positive``     — PR-AUC for the POSITIVE class (label 1), reported as a check.
        * ``roc_auc``             — ROC-AUC (symmetric in class).
        * ``f1_macro``            — macro-averaged F1.
        * ``f1_negative_class``   — F1 with ``pos_label=0``.
        * ``balanced_accuracy``   — mean of TPR and TNR.
        * ``mcc``                 — Matthews correlation coefficient.
        * ``precision_negative``  — precision w.r.t. the negative class.
        * ``recall_negative``     — recall w.r.t. the negative class.
        * ``confusion_matrix``    — ``np.ndarray`` of shape ``(2, 2)``,
          rows = true label, cols = predicted (``labels=[0, 1]``).
        * ``threshold``           — the decision threshold used.
        * ``n_positive`` / ``n_negative`` — class counts in ``y_true``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")

    y_pred = (y_score >= threshold).astype(int)

    # Negative-class PR-AUC: flip labels and scores so class 0 becomes the
    # "positive class" for sklearn's convention.
    pr_auc_negative = _safe_metric(average_precision_score, 1 - y_true, 1.0 - y_score)
    pr_auc_positive = _safe_metric(average_precision_score, y_true, y_score)
    roc_auc = _safe_metric(roc_auc_score, y_true, y_score)

    f1_macro = _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0)
    f1_negative_class = _safe_metric(f1_score, y_true, y_pred, pos_label=0, zero_division=0)
    bacc = _safe_metric(balanced_accuracy_score, y_true, y_pred)
    mcc = _safe_metric(matthews_corrcoef, y_true, y_pred)
    precision_neg = _safe_metric(precision_score, y_true, y_pred, pos_label=0, zero_division=0)
    recall_neg = _safe_metric(recall_score, y_true, y_pred, pos_label=0, zero_division=0)

    try:
        cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    except ValueError:
        cm = np.zeros((2, 2), dtype=np.int64)

    return {
        "pr_auc": pr_auc_negative,
        "pr_auc_positive": pr_auc_positive,
        "roc_auc": roc_auc,
        "f1_macro": f1_macro,
        "f1_negative_class": f1_negative_class,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "precision_negative": precision_neg,
        "recall_negative": recall_neg,
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }


__all__ = ["classification_metrics"]
