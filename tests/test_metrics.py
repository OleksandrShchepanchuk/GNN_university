"""Tests for ``reddit_gnn.training.metrics.classification_metrics``.

The metric panel is the contract that the training loop, evaluator, and
report exporters all depend on. We exercise the three corner-case prediction
regimes — perfect, random, and all-positive — plus the single-class edge
cases that sklearn raises on (which our wrapper coerces to ``NaN``).
"""

from __future__ import annotations

import numpy as np
import pytest

from reddit_gnn.training.metrics import classification_metrics


def _balanced_labels(n: int = 200, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).astype(np.int64)


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
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["f1_negative_class"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    assert m["precision_negative"] == pytest.approx(1.0)
    assert m["recall_negative"] == pytest.approx(1.0)
    # Confusion matrix is fully diagonal.
    cm = m["confusion_matrix"]
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
    # AUCs should be near 0.5 for random scores.
    assert 0.45 <= m["roc_auc"] <= 0.55
    # PR-AUC for either class is close to the class prevalence under random scores.
    prev_neg = (y_true == 0).mean()
    assert abs(m["pr_auc"] - prev_neg) < 0.1
    # MCC near zero for random predictions.
    assert abs(m["mcc"]) < 0.1


# ---------------------------------------------------------------------------
# All-positive predictions
# ---------------------------------------------------------------------------


def test_all_positive_predictions_floor_negative_recall():
    y_true = _balanced_labels()
    # Score above threshold for every example -> always predict label 1.
    y_score = np.full_like(y_true, fill_value=0.99, dtype=float)
    m = classification_metrics(y_true, y_score)

    # Predicting the majority always means negative recall = 0.
    assert m["recall_negative"] == pytest.approx(0.0)
    # And negative precision = 0 by sklearn's zero_division convention here
    # (no positive predictions of class 0).
    assert m["precision_negative"] == pytest.approx(0.0)
    # F1 for the negative class falls to 0.
    assert m["f1_negative_class"] == pytest.approx(0.0)
    # Confusion matrix: every prediction lands in column 1.
    cm = m["confusion_matrix"]
    assert (cm[:, 0] == 0).all()


# ---------------------------------------------------------------------------
# Threshold knob
# ---------------------------------------------------------------------------


def test_threshold_controls_predicted_class():
    # Two examples, scores 0.4 and 0.6. With default threshold 0.5: predictions = [0, 1].
    # With threshold 0.3: predictions = [1, 1]; with threshold 0.7: predictions = [0, 0].
    y_true = np.array([0, 1])
    y_score = np.array([0.4, 0.6])

    m_default = classification_metrics(y_true, y_score, threshold=0.5)
    assert (m_default["confusion_matrix"][:, 0] == [1, 0]).all()

    m_low = classification_metrics(y_true, y_score, threshold=0.3)
    # Both predicted 1: false positive in row 0, true positive in row 1.
    assert m_low["confusion_matrix"][0, 1] == 1
    assert m_low["confusion_matrix"][1, 1] == 1

    m_high = classification_metrics(y_true, y_score, threshold=0.7)
    # Both predicted 0.
    assert m_high["confusion_matrix"][0, 0] == 1
    assert m_high["confusion_matrix"][1, 0] == 1


# ---------------------------------------------------------------------------
# Single-class edge case
# ---------------------------------------------------------------------------


def test_single_class_input_does_not_raise():
    y_true = np.zeros(10, dtype=np.int64)
    y_score = np.linspace(0.0, 1.0, 10)
    # roc_auc / pr_auc are undefined on a single-class y_true; we coerce to NaN.
    m = classification_metrics(y_true, y_score)
    assert np.isnan(m["roc_auc"]) or np.isnan(m["pr_auc"])
    # The function must not raise.


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
