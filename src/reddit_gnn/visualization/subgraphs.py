"""Ego-graph and small-subgraph visualizations for qualitative error analysis."""

from __future__ import annotations

from typing import Any


def plot_ego_subgraph(graph: Any, node: str, hops: int = 1, ax: Any | None = None) -> Any:
    """Draw the ``hops``-hop ego subgraph around ``node`` with signed edge colors."""
    raise NotImplementedError("visualization.subgraphs.plot_ego_subgraph is not implemented yet")


def plot_misclassified_subgraph(
    graph: Any, edges: list[tuple[int, int]], ax: Any | None = None
) -> Any:
    """Render a subgraph induced by a list of (possibly misclassified) edges."""
    raise NotImplementedError(
        "visualization.subgraphs.plot_misclassified_subgraph is not implemented yet"
    )


__all__ = ["plot_ego_subgraph", "plot_misclassified_subgraph"]
