"""Temporal plots: edge volume, sentiment drift, train/val/test split timeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from reddit_gnn.analysis.temporal_stats import edges_over_time, negative_ratio_over_time
from reddit_gnn.visualization import NEGATIVE_COLOR, POSITIVE_COLOR, _maybe_save


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _add_split_cutoffs(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Annotate the chronological 70/15/15 split boundaries as vertical lines.

    Reads timestamps directly from `df` and computes the 70th and 85th time
    percentiles — same logic as `chronological_edge_split`. Skipped silently
    if `df` lacks a TIMESTAMP column or is empty.
    """
    if "TIMESTAMP" not in df.columns or df.empty:
        return
    ts = pd.to_datetime(df["TIMESTAMP"]).sort_values()
    n = len(ts)
    if n < 10:
        return
    train_cutoff = ts.iloc[int(n * 0.70)]
    val_cutoff = ts.iloc[int(n * 0.85)]
    for cutoff, label in [(train_cutoff, "train / val"), (val_cutoff, "val / test")]:
        ax.axvline(cutoff, color="#555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            cutoff,
            ax.get_ylim()[1] * 0.96,
            f"  {label}\n  {cutoff:%Y-%m}",
            color="#555",
            fontsize=8,
            va="top",
            ha="left",
        )


def plot_edges_over_time(
    df: pd.DataFrame,
    *,
    freq: str = "ME",
    save_path: str | Path | None = None,
) -> Figure:
    """Stacked area of negative vs positive edge counts per period."""
    series = edges_over_time(df, freq=freq)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    if not series.empty:
        x = series.index.to_numpy()
        ax.stackplot(
            x,
            series["negative_count"].to_numpy(),
            series["positive_count"].to_numpy(),
            labels=["negative (label = 0)", "neutral / positive (label = 1)"],
            colors=[NEGATIVE_COLOR, POSITIVE_COLOR],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.legend(loc="upper left")
        _format_date_axis(ax)
        _add_split_cutoffs(ax, df)
    ax.set_title(
        f"Edges per month over the {len(series)} monthly periods  —  stacked by label",
        loc="left",
    )
    ax.set_ylabel("edges per month")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_negative_ratio_over_time(
    df: pd.DataFrame,
    *,
    freq: str = "ME",
    save_path: str | Path | None = None,
) -> Figure:
    """Line plot of per-period negative-label share with the overall mean as a baseline."""
    series = negative_ratio_over_time(df, freq=freq)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    if not series.empty:
        ax.plot(
            series.index.to_numpy(),
            series["negative_ratio"].to_numpy(),
            color=NEGATIVE_COLOR,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
        )
        overall = (df["label_binary"] == 0).mean() if len(df) else 0
        ax.axhline(
            overall,
            color="#444",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"overall mean = {overall:.3f}",
        )
        ax.legend(loc="upper left")
        _format_date_axis(ax)
        ymax = max(0.05, series["negative_ratio"].max() * 1.25)
        ax.set_ylim(0, ymax)
        _add_split_cutoffs(ax, df)
    ax.set_title(
        "Negative-label share per month  —  temporal non-stationarity motivates the chronological split",
        loc="left",
    )
    ax.set_ylabel("share of edges with label = 0")
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_split_timeline(
    df: pd.DataFrame,
    *,
    split_col: str = "split",
    save_path: str | Path | None = None,
) -> Figure:
    """Render a horizontal timeline showing train/val/test boundaries.

    ``split_col`` must be a column in ``df`` whose values are in
    ``{"train", "val", "test"}``. The function tolerates other split names too,
    using a stable color cycle.
    """
    if split_col not in df.columns:
        raise KeyError(
            f"plot_split_timeline: column {split_col!r} not found in df. "
            f"Use reddit_gnn.data.splits.* to add it."
        )
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    palette = {
        "train": "#4c72b0",
        "val": "#dd8452",
        "test": "#55a467",
    }
    splits = df[split_col].astype(str).tolist()
    times = pd.to_datetime(df["TIMESTAMP"]).to_numpy()
    color_cycle = ["#4c72b0", "#dd8452", "#55a467", "#937860", "#8172b3"]
    seen: dict[str, str] = {}
    for name in pd.Series(splits).unique():
        seen[str(name)] = palette.get(str(name), color_cycle[len(seen) % len(color_cycle)])

    for name, color in seen.items():
        mask = np.asarray([s == name for s in splits])
        ax.scatter(times[mask], np.zeros(mask.sum()), s=4, color=color, label=name)

    ax.set_yticks([])
    ax.legend(loc="upper center", ncol=len(seen), frameon=False, bbox_to_anchor=(0.5, 1.25))
    ax.set_title("Temporal split membership over time")
    _format_date_axis(ax)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


__all__ = [
    "plot_edges_over_time",
    "plot_negative_ratio_over_time",
    "plot_split_timeline",
]
