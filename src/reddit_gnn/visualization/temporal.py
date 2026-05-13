"""Temporal plots: edge volume over time, sentiment drift, split boundaries."""

from __future__ import annotations

from typing import Any


def plot_edges_over_time(edge_df: Any, freq: str = "M", ax: Any | None = None) -> Any:
    """Time-series of edge counts."""
    raise NotImplementedError("visualization.temporal.plot_edges_over_time is not implemented yet")


def plot_sign_drift(edge_df: Any, freq: str = "M", ax: Any | None = None) -> Any:
    """Per-period stacked area of negative vs neutral/positive share."""
    raise NotImplementedError("visualization.temporal.plot_sign_drift is not implemented yet")


def plot_split_boundaries(edge_df: Any, masks: dict[str, Any], ax: Any | None = None) -> Any:
    """Overlay train/val/test boundary lines on the temporal axis."""
    raise NotImplementedError("visualization.temporal.plot_split_boundaries is not implemented yet")


__all__ = ["plot_edges_over_time", "plot_sign_drift", "plot_split_boundaries"]
