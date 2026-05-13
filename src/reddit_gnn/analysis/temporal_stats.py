"""Temporal statistics for the EDA + split motivation.

We always work directly with the ``TIMESTAMP`` column of the processed edge
frame (``datetime64[ns]``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def edges_over_time(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Edge counts per period, broken down by label.

    Returns a DataFrame indexed by period start (the freq alias is passed
    directly to ``pd.Grouper``). Columns: ``edge_count``, ``positive_count``,
    ``negative_count``.
    """
    if df.empty:
        return pd.DataFrame(columns=["edge_count", "positive_count", "negative_count"])
    grouped = (
        df.assign(
            _pos=(df["label_binary"] == 1).astype(np.int64),
            _neg=(df["label_binary"] == 0).astype(np.int64),
        )
        .groupby(pd.Grouper(key="TIMESTAMP", freq=freq))[["_pos", "_neg"]]
        .sum()
    )
    grouped["edge_count"] = grouped["_pos"] + grouped["_neg"]
    grouped = grouped.rename(columns={"_pos": "positive_count", "_neg": "negative_count"})
    return grouped[["edge_count", "positive_count", "negative_count"]]


def negative_ratio_over_time(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Per-period negative-label share. Returns columns ``negative_count``,
    ``edge_count``, ``negative_ratio`` (NaN for empty periods)."""
    counts = edges_over_time(df, freq=freq)
    if counts.empty:
        out = counts.copy()
        out["negative_ratio"] = pd.Series(dtype=float)
        return out[["negative_count", "edge_count", "negative_ratio"]]
    counts = counts.copy()
    counts["negative_ratio"] = np.where(
        counts["edge_count"] > 0,
        counts["negative_count"] / counts["edge_count"],
        np.nan,
    )
    return counts[["negative_count", "edge_count", "negative_ratio"]]


def summarize_temporal_split(
    df: pd.DataFrame,
    train_idx: np.ndarray | pd.Index | list[int],
    val_idx: np.ndarray | pd.Index | list[int],
    test_idx: np.ndarray | pd.Index | list[int],
) -> pd.DataFrame:
    """Per-split row count, time range, and label balance."""
    rows = []
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sub = df.iloc[list(idx)] if len(idx) else df.iloc[0:0]
        rows.append(
            {
                "split": name,
                "num_edges": len(sub),
                "time_min": sub["TIMESTAMP"].min() if len(sub) else pd.NaT,
                "time_max": sub["TIMESTAMP"].max() if len(sub) else pd.NaT,
                "positive_ratio": float(sub["label_binary"].mean()) if len(sub) else float("nan"),
                "negative_ratio": float(1 - sub["label_binary"].mean())
                if len(sub)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "edges_over_time",
    "negative_ratio_over_time",
    "summarize_temporal_split",
]
