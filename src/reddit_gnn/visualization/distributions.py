"""Distribution plots: degree, edge label balance, edge feature histograms.

Functions take a matplotlib ``Axes`` so callers can compose multi-panel
figures. When ``ax is None`` a new figure is created.
"""

from __future__ import annotations

from typing import Any


def plot_degree_distribution(graph: Any, ax: Any | None = None) -> Any:
    """Log-log degree histogram."""
    raise NotImplementedError(
        "visualization.distributions.plot_degree_distribution is not implemented yet"
    )


def plot_label_balance(edge_df: Any, ax: Any | None = None) -> Any:
    """Bar plot of edge label counts (0 = negative, 1 = neutral/positive)."""
    raise NotImplementedError(
        "visualization.distributions.plot_label_balance is not implemented yet"
    )


def plot_edge_feature_histograms(edge_df: Any, cols: list[str], ax: Any | None = None) -> Any:
    """Grid of histograms for selected edge features."""
    raise NotImplementedError(
        "visualization.distributions.plot_edge_feature_histograms is not implemented yet"
    )


__all__ = [
    "plot_degree_distribution",
    "plot_edge_feature_histograms",
    "plot_label_balance",
]
