"""Post-hoc error analysis on a trained model's test predictions.

Every function here is **pure**: it consumes ``y_true`` / ``y_pred`` arrays
plus the ``df_test`` table (whatever columns are needed for slicing) and
returns a ``DataFrame``. Plotting is left to
:mod:`reddit_gnn.visualization.results`; the script entry point in
``scripts/export_report_assets.py`` chains both.

The headline question for this project's imbalanced setting (~90/10) is
**where does the model under-predict the rare class?**, so the helpers
report per-bucket false-positive and false-negative rates rather than just
overall error rates.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _as_int_array(arr) -> np.ndarray:
    return np.asarray(arr).astype(int).ravel()


# ---------------------------------------------------------------------------
# Degree-bin error rates
# ---------------------------------------------------------------------------


def errors_by_degree_bin(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    degree_array,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin ``degree_array`` into deciles (or fewer when ties collapse bins) and
    report the per-bin error / FP / FN rates.

    ``df_test`` is currently unused — the signature keeps it for forward
    compatibility (later versions may want to surface POST_IDs per bin).
    """
    del df_test  # kept for API stability
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    deg = np.asarray(degree_array).astype(float).ravel()
    if not (len(y_true) == len(y_pred) == len(deg)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)} deg={len(deg)}"
        )

    df = pd.DataFrame(
        {
            "degree": deg,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["fp"] = ((df["y_true"] == 0) & (df["y_pred"] == 1)).astype(int)
    df["fn"] = ((df["y_true"] == 1) & (df["y_pred"] == 0)).astype(int)

    try:
        df["bin"] = pd.qcut(df["degree"], q=n_bins, duplicates="drop", labels=False)
    except ValueError:
        # All values are equal — drop into a single bin.
        df["bin"] = 0

    agg = (
        df.groupby("bin", dropna=True)
        .agg(
            n=("error", "size"),
            n_errors=("error", "sum"),
            error_rate=("error", "mean"),
            fp_rate=("fp", "mean"),
            fn_rate=("fn", "mean"),
            degree_min=("degree", "min"),
            degree_max=("degree", "max"),
            degree_median=("degree", "median"),
        )
        .reset_index()
        .sort_values("bin")
    )
    agg["bin"] = agg["bin"].astype(int)
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-bin error rates
# ---------------------------------------------------------------------------


def errors_by_time_bin(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    *,
    n_bins: int = 10,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Equal-width temporal bins over the test period.

    Picks the timestamp column automatically: ``timestamp`` (predictions CSV
    convention) or ``TIMESTAMP`` (processed parquet convention).
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if time_col is None:
        if "timestamp" in df_test.columns:
            time_col = "timestamp"
        elif "TIMESTAMP" in df_test.columns:
            time_col = "TIMESTAMP"
        else:
            raise KeyError(
                "errors_by_time_bin: df_test must have a 'timestamp' or 'TIMESTAMP' column"
            )
    times = pd.to_datetime(df_test[time_col]).reset_index(drop=True)
    if not (len(y_true) == len(y_pred) == len(times)):
        raise ValueError("length mismatch between y_true, y_pred, and df_test[time_col]")

    df = pd.DataFrame(
        {
            "time": times,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["fp"] = ((df["y_true"] == 0) & (df["y_pred"] == 1)).astype(int)
    df["fn"] = ((df["y_true"] == 1) & (df["y_pred"] == 0)).astype(int)
    df["bin"] = pd.cut(df["time"], bins=n_bins, labels=False, include_lowest=True)

    agg = (
        df.groupby("bin", dropna=False)
        .agg(
            n=("error", "size"),
            n_errors=("error", "sum"),
            error_rate=("error", "mean"),
            fp_rate=("fp", "mean"),
            fn_rate=("fn", "mean"),
            time_min=("time", "min"),
            time_max=("time", "max"),
        )
        .reset_index()
        .sort_values("bin")
    )
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-subreddit error rates
# ---------------------------------------------------------------------------


def errors_by_subreddit(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    *,
    top_k: int = 20,
    min_count: int = 5,
) -> dict[str, pd.DataFrame]:
    """Top-``k`` subreddits by FP and FN rate (sources of misclassified edges).

    ``min_count`` filters out subreddits with too few edges to support a stable
    rate estimate. The function reports rates from the source perspective
    (``source_subreddit_norm`` or ``source_id``).
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) != len(df_test):
        raise ValueError("length mismatch between y_true, y_pred, and df_test")

    col = "source_subreddit_norm" if "source_subreddit_norm" in df_test.columns else "source_id"
    df = df_test[[col]].copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["fp"] = ((df["y_true"] == 0) & (df["y_pred"] == 1)).astype(int)
    df["fn"] = ((df["y_true"] == 1) & (df["y_pred"] == 0)).astype(int)
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)

    agg = (
        df.groupby(col)
        .agg(
            n=("y_true", "size"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            error_rate=("error", "mean"),
            fp_rate=("fp", "mean"),
            fn_rate=("fn", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["n"] >= min_count]

    top_fp = agg.sort_values(["fp_rate", "n"], ascending=[False, False]).head(top_k)
    top_fn = agg.sort_values(["fn_rate", "n"], ascending=[False, False]).head(top_k)
    return {
        "top_fp_subreddits": top_fp.reset_index(drop=True),
        "top_fn_subreddits": top_fn.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------


def model_agreement(predictions_dict: Mapping[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Pairwise raw-agreement and Cohen's-kappa tables across models.

    All prediction vectors must have the same length (a single test split with
    a fixed row order). Returns ``{'agreement': DataFrame, 'kappa': DataFrame}``
    with model names as both the index and columns.
    """
    models = list(predictions_dict.keys())
    if not models:
        return {
            "agreement": pd.DataFrame(),
            "kappa": pd.DataFrame(),
        }

    arrays = {m: _as_int_array(p) for m, p in predictions_dict.items()}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"all prediction vectors must have the same length; got {lengths}")

    n = len(models)
    agreement = np.zeros((n, n), dtype=float)
    kappa = np.zeros((n, n), dtype=float)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            a = arrays[m1]
            b = arrays[m2]
            agreement[i, j] = float((a == b).mean())
            if i == j:
                kappa[i, j] = 1.0
            else:
                try:
                    kappa[i, j] = float(cohen_kappa_score(a, b))
                except ValueError:
                    kappa[i, j] = float("nan")

    return {
        "agreement": pd.DataFrame(agreement, index=models, columns=models),
        "kappa": pd.DataFrame(kappa, index=models, columns=models),
    }


# ---------------------------------------------------------------------------
# Confusion-matrix cell samples
# ---------------------------------------------------------------------------


def confusion_examples(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    *,
    n: int = 10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Sample ``n`` rows from each cell of the 2x2 confusion matrix.

    Returned keys: ``true_0_pred_0`` (TN), ``true_0_pred_1`` (FP),
    ``true_1_pred_0`` (FN), ``true_1_pred_1`` (TP). Each DataFrame preserves
    whatever identifying columns (``POST_ID``, ``timestamp``, subreddit names,
    edge ids, etc.) live on ``df_test``, so it is suitable for qualitative
    inspection in a report.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) != len(df_test):
        raise ValueError("length mismatch between y_true, y_pred, and df_test")

    base = df_test.copy().reset_index(drop=True)
    base["y_true"] = y_true
    base["y_pred"] = y_pred

    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for true_label in (0, 1):
        for pred_label in (0, 1):
            mask = (base["y_true"] == true_label) & (base["y_pred"] == pred_label)
            cell = base.loc[mask]
            if len(cell) == 0:
                out[f"true_{true_label}_pred_{pred_label}"] = cell.head(0)
                continue
            k = min(n, len(cell))
            picks = rng.choice(len(cell), size=k, replace=False)
            out[f"true_{true_label}_pred_{pred_label}"] = cell.iloc[picks].reset_index(drop=True)
    return out


__all__ = [
    "confusion_examples",
    "errors_by_degree_bin",
    "errors_by_subreddit",
    "errors_by_time_bin",
    "model_agreement",
]
