"""Basic graph descriptive statistics: |V|, |E|, density, degree moments,
reciprocity, weakly/strongly connected components, clustering, PageRank.

Implementations should accept either a NetworkX ``DiGraph`` or a PyG ``Data``
object — pick whichever is more natural inside each function.
"""

from __future__ import annotations

from typing import Any


def basic_counts(graph: Any) -> dict[str, int]:
    """Return ``{num_nodes, num_edges, num_self_loops, num_unique_pairs}``."""
    raise NotImplementedError("analysis.graph_stats.basic_counts is not implemented yet")


def degree_distribution(graph: Any) -> dict[str, Any]:
    """Return mean/median/max in/out/total degree plus full arrays for plotting."""
    raise NotImplementedError("analysis.graph_stats.degree_distribution is not implemented yet")


def connected_components(graph: Any) -> dict[str, Any]:
    """Sizes of weakly and strongly connected components."""
    raise NotImplementedError("analysis.graph_stats.connected_components is not implemented yet")


def reciprocity(graph: Any) -> float:
    """Fraction of edges ``(u,v)`` for which ``(v,u)`` also exists."""
    raise NotImplementedError("analysis.graph_stats.reciprocity is not implemented yet")


def top_pagerank(graph: Any, k: int = 20) -> list[tuple[str, float]]:
    """Return the top-``k`` subreddits by PageRank."""
    raise NotImplementedError("analysis.graph_stats.top_pagerank is not implemented yet")


__all__ = [
    "basic_counts",
    "connected_components",
    "degree_distribution",
    "reciprocity",
    "top_pagerank",
]
