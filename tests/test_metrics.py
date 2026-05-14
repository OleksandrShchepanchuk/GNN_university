"""Tests for ``reddit_gnn.training.metrics``.

The metric panel is the contract that the training loop, evaluator, and
report exporters all depend on. We exercise the three corner-case prediction
regimes — perfect, random, and all-positive — plus the single-class edge
cases that sklearn raises on (which our wrapper coerces to ``NaN``).
"""

from __future__ import annotations

import numpy as np
import pytest

from reddit_gnn.training.metrics import classification_metrics, precision_at_k


def _balanced_labels(n: int = 200, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).astype(np.int64)


def _cm_arr(cm) -> np.ndarray:
    """Cast the confusion-matrix nested list back to a numpy array for assertions."""
    return np.asarray(cm)


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------


def test_perfect_predictions_yield_top_scores():
    y_true = _balanced_labels()
    y_score = y_true.astype(float)  # 1.0 for label=1, 0.0 for label=0
    m = classification_metrics(y_true, y_score)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["pr_auc_positive"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["f1_negative_class"] == pytest.approx(1.0)
    assert m["f1_positive_class"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    assert m["precision_negative"] == pytest.approx(1.0)
    assert m["recall_negative"] == pytest.approx(1.0)
    assert m["precision_positive"] == pytest.approx(1.0)
    assert m["recall_positive"] == pytest.approx(1.0)
    cm = _cm_arr(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm[0, 1] == 0
    assert cm[1, 0] == 0


# ---------------------------------------------------------------------------
# Random predictions
# ---------------------------------------------------------------------------


def test_random_predictions_have_near_chance_aucs():
    rng = np.random.default_rng(42)
    n = 2000
    y_true = rng.integers(0, 2, size=n).astype(np.int64)
    y_score = rng.uniform(size=n)  # independent of y_true
    m = classification_metrics(y_true, y_score)
    assert 0.45 <= m["roc_auc"] <= 0.55
    prev_neg = (y_true == 0).mean()
    assert abs(m["pr_auc"] - prev_neg) < 0.1
    assert abs(m["mcc"]) < 0.1


# ---------------------------------------------------------------------------
# All-positive predictions
# ---------------------------------------------------------------------------


def test_all_positive_predictions_floor_negative_recall():
    y_true = _balanced_labels()
    y_score = np.full_like(y_true, fill_value=0.99, dtype=float)
    m = classification_metrics(y_true, y_score)

    assert m["recall_negative"] == pytest.approx(0.0)
    assert m["precision_negative"] == pytest.approx(0.0)
    assert m["f1_negative_class"] == pytest.approx(0.0)
    cm = _cm_arr(m["confusion_matrix"])
    assert (cm[:, 0] == 0).all()


# ---------------------------------------------------------------------------
# Threshold knob
# ---------------------------------------------------------------------------


def test_threshold_controls_predicted_class():
    y_true = np.array([0, 1])
    y_score = np.array([0.4, 0.6])

    m_default = classification_metrics(y_true, y_score, threshold=0.5)
    cm = _cm_arr(m_default["confusion_matrix"])
    assert (cm[:, 0] == [1, 0]).all()

    m_low = classification_metrics(y_true, y_score, threshold=0.3)
    cm = _cm_arr(m_low["confusion_matrix"])
    assert cm[0, 1] == 1
    assert cm[1, 1] == 1

    m_high = classification_metrics(y_true, y_score, threshold=0.7)
    cm = _cm_arr(m_high["confusion_matrix"])
    assert cm[0, 0] == 1
    assert cm[1, 0] == 1


# ---------------------------------------------------------------------------
# Single-class edge case
# ---------------------------------------------------------------------------


def test_single_class_input_does_not_raise():
    y_true = np.zeros(10, dtype=np.int64)
    y_score = np.linspace(0.0, 1.0, 10)
    m = classification_metrics(y_true, y_score)
    assert np.isnan(m["roc_auc"]) or np.isnan(m["pr_auc"])


# ---------------------------------------------------------------------------
# Shape mismatch
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1, 1]), np.array([0.1, 0.2]))


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_counts_are_reported():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    m = classification_metrics(y_true, y_score)
    assert m["n_positive"] == 3
    assert m["n_negative"] == 2


# ---------------------------------------------------------------------------
# NEW: PR-AUC lift / class_prior_negative
# ---------------------------------------------------------------------------


def test_pr_auc_lift_equals_pr_auc_over_class_prior():
    """Lift = PR-AUC neg / class_prior_negative — the headline framing."""
    # 90% positive / 10% negative, perfect predictions → pr_auc=1, prior=0.1, lift=10.
    y_true = np.array([0] * 10 + [1] * 90)
    y_score = np.array([0.0] * 10 + [1.0] * 90)
    m = classification_metrics(y_true, y_score)
    assert m["class_prior_negative"] == pytest.approx(0.10)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["pr_auc_lift"] == pytest.approx(10.0)

    # Random scores → pr_auc ≈ class_prior_negative → lift ≈ 1.
    rng = np.random.default_rng(0)
    n = 5000
    y_true_imb = (rng.uniform(size=n) > 0.9).astype(int)  # ~10% are 1
    # Flip so class 0 is rare: target ~10% class 0.
    y_true_imb = 1 - y_true_imb
    y_score_imb = rng.uniform(size=n)
    m2 = classification_metrics(y_true_imb, y_score_imb)
    # Random scores: lift should be near 1, almost certainly in [0.5, 2.0].
    assert 0.5 <= m2["pr_auc_lift"] <= 2.0


# ---------------------------------------------------------------------------
# NEW: precision_at_k surfaces top-K negative-class candidates
# ---------------------------------------------------------------------------


def test_precision_at_k_perfect_and_random():
    """For a perfect score, top-K negatives are all true negatives → 1.0;
    for random scores, top-K precision is near the class prior."""
    # 100 examples, 10 negatives, all negatives have the smallest scores.
    y_true = np.array([0] * 10 + [1] * 90)
    y_score = np.linspace(0.0, 1.0, 100)
    assert precision_at_k(y_true, y_score, 10) == pytest.approx(1.0)
    # Top-50 contains all 10 true negatives, so precision = 10/50 = 0.2.
    assert precision_at_k(y_true, y_score, 50) == pytest.approx(0.20)

    # Random scores, ~10% negative prevalence — top-K precision ≈ prior.
    rng = np.random.default_rng(1)
    n = 5000
    y_true_r = (rng.uniform(size=n) > 0.9).astype(int)
    y_true_r = 1 - y_true_r
    y_score_r = rng.uniform(size=n)
    p_at_100 = precision_at_k(y_true_r, y_score_r, 100)
    assert 0.03 <= p_at_100 <= 0.20  # well within random tolerance of 0.10

    # Edge cases.
    assert np.isnan(precision_at_k(np.array([]), np.array([]), 10))
    assert np.isnan(precision_at_k(y_true_r, y_score_r, 0))


# ---------------------------------------------------------------------------
# NEW: accuracy, positive-class precision/recall, confusion_matrix as list
# ---------------------------------------------------------------------------


def test_positive_class_metrics_and_confusion_matrix_shape():
    """The metric panel reports symmetric class-1 metrics and a nested-list CM."""
    # 7 examples: 4 negative, 3 positive. Predict [0,0,0,1,  1,1,0]
    # → for class 1: TP=2 (last-but-one and middle of the 1-block), FP=1 (the 4th).
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.2, 0.6, 0.7, 0.8, 0.4])
    m = classification_metrics(y_true, y_score)

    # Confusion matrix is a nested list (json-serialisable).
    cm = m["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm)
    arr = _cm_arr(cm)
    tn, fp = arr[0]
    fn, tp = arr[1]
    assert tn == 3 and fp == 1  # one positive prediction (idx 3) wrongly flagged
    assert fn == 1 and tp == 2  # idx 6 was missed; idx 4, 5 caught

    # Symmetric class-1 metrics check.
    assert m["precision_positive"] == pytest.approx(tp / (tp + fp))
    assert m["recall_positive"] == pytest.approx(tp / (tp + fn))
    assert m["f1_positive_class"] == pytest.approx(
        2
        * m["precision_positive"]
        * m["recall_positive"]
        / (m["precision_positive"] + m["recall_positive"])
    )

    # Accuracy: 5 of 7 correct (idx 3 and 6 wrong).
    assert m["accuracy"] == pytest.approx(5 / 7)
