"""Distribution plots: label balance, degree distributions, top-k subreddits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from reddit_gnn.visualization import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    PRIMARY_COLOR,
    _maybe_save,
)


def plot_label_distribution(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of the two-class edge label distribution."""
    counts = df["label_binary"].value_counts().sort_index()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    total = max(n_neg + n_pos, 1)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    labels = ["negative\n(label = 0)", "neutral / positive\n(label = 1)"]
    bars = ax.bar(
        labels,
        [n_neg, n_pos],
        color=[NEGATIVE_COLOR, POSITIVE_COLOR],
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
    )
    for bar, count in zip(bars, (n_neg, n_pos), strict=True):
        pct = count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{pct:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#222",
        )
    ax.set_ylabel("edge count")
    ax.set_title(
        f"Edge label distribution  —  {total:,} edges, {n_neg / total:.1%} negative (rare-class)",
        loc="left",
    )
    ax.set_ylim(0, max(n_neg, n_pos) * 1.22)
    # Suppress the default tick labels on Y in favour of the per-bar annotations.
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_degree_distribution(
    degree: np.ndarray,
    *,
    degree_type: str = "total",
    log_scale: bool = True,
    bins: int = 60,
    save_path: str | Path | None = None,
) -> Figure:
    """Histogram of a degree array (in / out / total)."""
    deg = np.asarray(degree, dtype=np.int64)
    if deg.size == 0:
        deg = np.array([0])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if log_scale:
        nonzero = deg[deg > 0]
        if nonzero.size == 0:
            nonzero = np.array([1])
        edges = np.logspace(0, np.log10(nonzero.max() + 1), bins)
        ax.hist(
            deg + 1, bins=edges, color=PRIMARY_COLOR, edgecolor="white", linewidth=0.4, alpha=0.92
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{degree_type} degree + 1 (log scale)")
        ax.set_ylabel("number of nodes (log scale)")
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
    else:
        ax.hist(deg, bins=bins, color=PRIMARY_COLOR, edgecolor="white", linewidth=0.4, alpha=0.92)
        ax.set_xlabel(f"{degree_type} degree")
        ax.set_ylabel("number of nodes")

    median_deg = float(np.median(deg))
    max_deg = int(deg.max())
    ax.set_title(
        f"{degree_type.capitalize()}-degree distribution  —  "
        f"median = {median_deg:.0f}, max = {max_deg:,}, n = {deg.size:,}",
        loc="left",
    )
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_top_subreddits_by_degree(
    df: pd.DataFrame,
    *,
    top_k: int = 20,
    mode: str = "total",
    save_path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart of top-``k`` subreddits by ``in`` / ``out`` / ``total`` degree."""
    if mode not in {"in", "out", "total"}:
        raise ValueError(f"mode must be 'in', 'out', or 'total'; got {mode!r}")

    in_deg = df.groupby("target_subreddit_norm").size()
    out_deg = df.groupby("source_subreddit_norm").size()
    if mode == "in":
        deg = in_deg
    elif mode == "out":
        deg = out_deg
    else:
        deg = in_deg.add(out_deg, fill_value=0).astype(int)
    top = deg.sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, 0.36 * len(top) + 1.5))
    values = top.values[::-1]
    labels = top.index[::-1]
    bars = ax.barh(
        labels, values, color=PRIMARY_COLOR, edgecolor="white", linewidth=0.6, alpha=0.92
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {int(value):,}",
            va="center",
            fontsize=9,
            color="#333",
        )
    ax.set_xlabel("edge count")
    mode_label = {"in": "incoming", "out": "outgoing", "total": "total"}[mode]
    ax.set_title(f"Top-{top_k} subreddits by {mode_label}-edge count", loc="left")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


__all__ = [
    "plot_degree_distribution",
    "plot_label_distribution",
    "plot_top_subreddits_by_degree",
]
