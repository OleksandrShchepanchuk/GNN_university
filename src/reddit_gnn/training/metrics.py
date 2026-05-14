"""Classification metrics geared at the class-imbalanced edge sign task.

The headline numbers are computed for the **negative class** (``label == 0``,
which is the minority "negative sentiment" class with ~10% prevalence in
SNAP Reddit). PR-AUC for that class is the primary leaderboard metric;
F1-macro is the secondary report.

All functions accept ``y_score`` as the probability of class 1 (the output of
``torch.sigmoid(logits)``). Internally we flip both labels and scores when a
metric is defined with respect to class 0 (PR-AUC, precision, recall).

Lift framing
============
The headline metric is **PR-AUC neg lift** — the ratio of the model's PR-AUC
on the rare class to the class-prior baseline PR-AUC. A model with
PR-AUC-neg = 0.50 on a 10%-prevalence task achieves a 5x lift over chance;
quoting the raw 0.50 without context understates the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
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


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Top-K precision on the **negative** class.

    Rank examples by ``1 - y_score`` (highest = most likely negative),
    take the top-K, and report what fraction are actually negative. This
    measures how well the model surfaces hostility candidates for review.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size == 0 or k <= 0:
        return float("nan")
    k = min(k, y_true.size)
    neg_score = 1.0 - y_score
    order = np.argsort(-neg_score, kind="stable")
    top_k = order[:k]
    return float((y_true[top_k] == 0).mean())


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the full metric panel for binary edge sign classification.

    Returns a dict with:
        * ``pr_auc``              — PR-AUC for the NEGATIVE class (label 0). Headline.
        * ``pr_auc_positive``     — PR-AUC for the POSITIVE class (label 1), reported as a check.
        * ``pr_auc_lift``         — ``pr_auc / class_prior_negative``. >1 means
          the model beats the prior. The headline framing.
        * ``class_prior_negative``— ``(y_true == 0).mean()``. Random baseline
          for ``pr_auc``.
        * ``roc_auc``             — ROC-AUC (symmetric in class).
        * ``accuracy``            — overall accuracy at ``threshold``.
        * ``f1_macro``            — macro-averaged F1.
        * ``f1_negative_class``   — F1 with ``pos_label=0``.
        * ``f1_positive_class``   — F1 with ``pos_label=1``.
        * ``balanced_accuracy``   — mean of TPR and TNR.
        * ``mcc``                 — Matthews correlation coefficient.
        * ``precision_negative`` / ``recall_negative`` — for class 0.
        * ``precision_positive`` / ``recall_positive`` — for class 1.
        * ``precision_at_50`` / ``precision_at_100`` / ``precision_at_500`` —
          top-K precision for surfacing negative-class candidates.
        * ``confusion_matrix``    — ``(2, 2)`` nested list (``[[TN, FP], [FN, TP]]``),
          rows = true label, cols = predicted (``labels=[0, 1]``).
        * ``threshold``           — the decision threshold used.
        * ``n_positive`` / ``n_negative`` — class counts in ``y_true``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")

    y_pred = (y_score >= threshold).astype(int)

    n_total = int(y_true.size)
    n_negative = int((y_true == 0).sum())
    n_positive = int((y_true == 1).sum())
    class_prior_negative = (n_negative / n_total) if n_total > 0 else float("nan")

    # Negative-class PR-AUC: flip labels and scores so class 0 becomes the
    # "positive class" for sklearn's convention.
    pr_auc_negative = _safe_metric(average_precision_score, 1 - y_true, 1.0 - y_score)
    pr_auc_positive = _safe_metric(average_precision_score, y_true, y_score)
    roc_auc = _safe_metric(roc_auc_score, y_true, y_score)

    pr_auc_lift = (
        float(pr_auc_negative / class_prior_negative)
        if (
            np.isfinite(pr_auc_negative)
            and np.isfinite(class_prior_negative)
            and class_prior_negative > 0
        )
        else float("nan")
    )

    accuracy = _safe_metric(accuracy_score, y_true, y_pred)
    f1_macro = _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0)
    f1_negative_class = _safe_metric(f1_score, y_true, y_pred, pos_label=0, zero_division=0)
    f1_positive_class = _safe_metric(f1_score, y_true, y_pred, pos_label=1, zero_division=0)
    bacc = _safe_metric(balanced_accuracy_score, y_true, y_pred)
    mcc = _safe_metric(matthews_corrcoef, y_true, y_pred)
    precision_neg = _safe_metric(precision_score, y_true, y_pred, pos_label=0, zero_division=0)
    recall_neg = _safe_metric(recall_score, y_true, y_pred, pos_label=0, zero_division=0)
    precision_pos = _safe_metric(precision_score, y_true, y_pred, pos_label=1, zero_division=0)
    recall_pos = _safe_metric(recall_score, y_true, y_pred, pos_label=1, zero_division=0)

    try:
        cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_list = cm.tolist()
    except ValueError:
        cm_list = [[0, 0], [0, 0]]

    return {
        "pr_auc": pr_auc_negative,
        "pr_auc_positive": pr_auc_positive,
        "pr_auc_lift": pr_auc_lift,
        "class_prior_negative": float(class_prior_negative),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_negative_class": f1_negative_class,
        "f1_positive_class": f1_positive_class,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "precision_negative": precision_neg,
        "recall_negative": recall_neg,
        "precision_positive": precision_pos,
        "recall_positive": recall_pos,
        "precision_at_50": precision_at_k(y_true, y_score, 50),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "precision_at_500": precision_at_k(y_true, y_score, 500),
        "confusion_matrix": cm_list,
        "threshold": float(threshold),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


__all__ = ["classification_metrics", "precision_at_k"]
