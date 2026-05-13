"""Statistics specific to the **signed** structure of the graph.

* Class balance over edges (always reported on the trained labels 0/1).
* Triangle counting: how many balanced vs unbalanced triads exist?
  Balance theory predicts homophily of negative ties under transitivity.
* Per-node positive/negative out-link ratio — useful for error analysis.
"""

from __future__ import annotations

from typing import Any


def class_balance(edge_df: Any) -> dict[int, int]:
    """Return ``{0: n_negative, 1: n_neutral_positive}``."""
    raise NotImplementedError("analysis.signed_stats.class_balance is not implemented yet")


def signed_triangles(graph: Any) -> dict[str, int]:
    """Counts of balanced (+,+,+ or +,-,-) and unbalanced triangles."""
    raise NotImplementedError("analysis.signed_stats.signed_triangles is not implemented yet")


def per_node_sign_ratio(edge_df: Any) -> Any:
    """Per-source-subreddit fraction of negative outgoing edges."""
    raise NotImplementedError("analysis.signed_stats.per_node_sign_ratio is not implemented yet")


__all__ = ["class_balance", "per_node_sign_ratio", "signed_triangles"]
