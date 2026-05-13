"""Tests for ``reddit_gnn.training.error_analysis``.

Each helper consumes plain arrays + a thin ``df_test`` table and returns a
``DataFrame`` (or dict thereof). We assert structure + a handful of computed
values on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reddit_gnn.training.error_analysis import (
    confusion_examples,
    errors_by_degree_bin,
    errors_by_subreddit,
    errors_by_time_bin,
    model_agreement,
)


def _synthetic_test_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2014-06-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "source_id": rng.integers(0, 10, size=n).astype(np.int64),
            "target_id": rng.integers(0, 10, size=n).astype(np.int64),
            "source_subreddit_norm": ["sr" + str(i) for i in rng.integers(0, 10, size=n)],
            "target_subreddit_norm": ["sr" + str(i) for i in rng.integers(0, 10, size=n)],
            "timestamp": ts,
            "POST_ID": [f"p{i:04d}" for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# errors_by_degree_bin
# ---------------------------------------------------------------------------


def test_errors_by_degree_bin_returns_correct_columns_and_sums():
    n = 200
    rng = np.random.default_rng(1)
    df = _synthetic_test_frame(n=n)
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_pred = rng.integers(0, 2, size=n).astype(int)
    degree = rng.integers(1, 100, size=n).astype(int)

    out = errors_by_degree_bin(df, y_true, y_pred, degree, n_bins=10)
    assert isinstance(out, pd.DataFrame)
    for col in (
        "bin",
        "n",
        "n_errors",
        "error_rate",
        "fp_rate",
        "fn_rate",
        "degree_min",
        "degree_max",
    ):
        assert col in out.columns
    # Total counts conserve across bins.
    assert out["n"].sum() == n
    # Total error count equals the number of disagreements.
    assert int(out["n_errors"].sum()) == int((y_true != y_pred).sum())
    # Per-bin error_rate ∈ [0, 1].
    assert ((out["error_rate"] >= 0) & (out["error_rate"] <= 1)).all()


def test_errors_by_degree_bin_perfect_predictions_have_zero_error():
    n = 100
    rng = np.random.default_rng(2)
    df = _synthetic_test_frame(n=n)
    y = rng.integers(0, 2, size=n).astype(int)
    degree = rng.integers(0, 50, size=n).astype(int)
    out = errors_by_degree_bin(df, y, y, degree, n_bins=5)
    assert (out["error_rate"] == 0).all()
    assert (out["fp_rate"] == 0).all()
    assert (out["fn_rate"] == 0).all()


def test_errors_by_degree_bin_rejects_length_mismatch():
    df = _synthetic_test_frame(n=10)
    with pytest.raises(ValueError):
        errors_by_degree_bin(df, np.zeros(10), np.zeros(5), np.zeros(10))


# ---------------------------------------------------------------------------
# errors_by_time_bin
# ---------------------------------------------------------------------------


def test_errors_by_time_bin_columns_and_monotonic_bins():
    n = 120
    rng = np.random.default_rng(3)
    df = _synthetic_test_frame(n=n)
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_pred = rng.integers(0, 2, size=n).astype(int)
    out = errors_by_time_bin(df, y_true, y_pred, n_bins=6)
    for col in ("bin", "n", "n_errors", "error_rate", "fp_rate", "fn_rate", "time_min", "time_max"):
        assert col in out.columns
    # Bin boundaries are temporally monotonic.
    tmins = pd.to_datetime(out["time_min"])
    assert tmins.is_monotonic_increasing
    # Totals conserve.
    assert out["n"].sum() == n


def test_errors_by_time_bin_accepts_capital_timestamp_column():
    df = _synthetic_test_frame(n=20).rename(columns={"timestamp": "TIMESTAMP"})
    y_true = np.zeros(20, dtype=int)
    y_pred = np.zeros(20, dtype=int)
    out = errors_by_time_bin(df, y_true, y_pred, n_bins=3)
    assert len(out) >= 1


def test_errors_by_time_bin_missing_column_raises():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(KeyError):
        errors_by_time_bin(df, np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# errors_by_subreddit
# ---------------------------------------------------------------------------


def test_errors_by_subreddit_returns_two_top_k_tables():
    n = 200
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "source_subreddit_norm": rng.choice(
                ["a", "b", "c", "d", "e"], size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05]
            ),
            "target_subreddit_norm": rng.choice(["x", "y", "z"], size=n),
            "timestamp": pd.date_range("2014-01-01", periods=n, freq="h"),
        }
    )
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_pred = rng.integers(0, 2, size=n).astype(int)
    out = errors_by_subreddit(df, y_true, y_pred, top_k=3, min_count=5)
    assert set(out.keys()) == {"top_fp_subreddits", "top_fn_subreddits"}
    for key in out:
        sub = out[key]
        assert len(sub) <= 3
        for col in ("source_subreddit_norm", "n", "fp", "fn", "fp_rate", "fn_rate", "error_rate"):
            assert col in sub.columns
        assert (sub["n"] >= 5).all()


def test_errors_by_subreddit_falls_back_to_source_id():
    n = 30
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "source_id": rng.integers(0, 5, size=n).astype(np.int64),
            "target_id": rng.integers(0, 5, size=n).astype(np.int64),
            "timestamp": pd.date_range("2014-01-01", periods=n, freq="h"),
        }
    )
    y_true = np.zeros(n, dtype=int)
    y_pred = np.ones(n, dtype=int)  # all FP
    out = errors_by_subreddit(df, y_true, y_pred, top_k=2, min_count=1)
    fp_top = out["top_fp_subreddits"]
    assert "source_id" in fp_top.columns
    # Every row has FP rate of exactly 1.
    assert (fp_top["fp_rate"] == 1.0).all()


# ---------------------------------------------------------------------------
# model_agreement
# ---------------------------------------------------------------------------


def test_model_agreement_diagonal_is_one():
    preds = {
        "a": np.array([0, 1, 0, 1, 1]),
        "b": np.array([0, 1, 1, 1, 0]),
        "c": np.array([0, 0, 0, 1, 1]),
    }
    out = model_agreement(preds)
    assert list(out["agreement"].columns) == ["a", "b", "c"]
    assert (np.diag(out["agreement"].to_numpy()) == 1.0).all()
    assert (np.diag(out["kappa"].to_numpy()) == 1.0).all()
    # Symmetric.
    np.testing.assert_allclose(
        out["agreement"].to_numpy(), out["agreement"].to_numpy().T, atol=1e-9
    )


def test_model_agreement_perfect_models_have_kappa_one():
    preds = {"a": np.array([0, 1, 1, 0]), "b": np.array([0, 1, 1, 0])}
    out = model_agreement(preds)
    assert out["agreement"].iloc[0, 1] == 1.0
    assert out["kappa"].iloc[0, 1] == 1.0


def test_model_agreement_rejects_unequal_lengths():
    preds = {"a": np.array([0, 1]), "b": np.array([0, 1, 0])}
    with pytest.raises(ValueError):
        model_agreement(preds)


def test_model_agreement_empty_input_returns_empty_frames():
    out = model_agreement({})
    assert out["agreement"].empty
    assert out["kappa"].empty


# ---------------------------------------------------------------------------
# confusion_examples
# ---------------------------------------------------------------------------


def test_confusion_examples_returns_one_frame_per_cell():
    n = 80
    rng = np.random.default_rng(6)
    df = _synthetic_test_frame(n=n)
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_pred = rng.integers(0, 2, size=n).astype(int)
    out = confusion_examples(df, y_true, y_pred, n=5)
    assert set(out.keys()) == {"true_0_pred_0", "true_0_pred_1", "true_1_pred_0", "true_1_pred_1"}
    for cell, sub in out.items():
        assert len(sub) <= 5
        if len(sub) > 0:
            # All rows in the cell match the true/pred combo.
            true_label = int(cell.split("_")[1])
            pred_label = int(cell.split("_")[-1])
            assert (sub["y_true"] == true_label).all()
            assert (sub["y_pred"] == pred_label).all()
        # Identifying columns are preserved.
        for col in ("source_id", "target_id", "timestamp", "POST_ID"):
            assert col in sub.columns


def test_confusion_examples_is_deterministic_under_seed():
    n = 80
    rng = np.random.default_rng(7)
    df = _synthetic_test_frame(n=n)
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_pred = rng.integers(0, 2, size=n).astype(int)
    a = confusion_examples(df, y_true, y_pred, n=5, seed=1)
    b = confusion_examples(df, y_true, y_pred, n=5, seed=1)
    for cell in a:
        pd.testing.assert_frame_equal(a[cell], b[cell])


def test_confusion_examples_rejects_length_mismatch():
    df = _synthetic_test_frame(n=10)
    with pytest.raises(ValueError):
        confusion_examples(df, np.zeros(5), np.zeros(10))
