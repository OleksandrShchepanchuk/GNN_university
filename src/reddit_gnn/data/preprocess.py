"""Preprocess raw SNAP DataFrames into a clean edge table.

Responsibilities (to be implemented):
    * concatenate body + title rows;
    * normalize subreddit names (lowercase, strip ``/r/`` if present);
    * parse ``TIMESTAMP`` to ``datetime64[ns]``;
    * remap ``POST_LABEL``: ``-1 -> 0`` (negative), ``+1 -> 1`` (neutral_or_positive);
    * expand ``POST_PROPERTIES`` into 86 numeric columns;
    * optionally drop rows whose endpoints have no LIWC embedding;
    * optionally aggregate repeated ``(src, dst)`` pairs by majority label.

The output is a tidy DataFrame plus a ``subreddit -> node_id`` mapping. Both
get persisted to ``data/interim/`` as parquet for fast reload.
"""

from __future__ import annotations

from typing import Any


def remap_post_label(label: int) -> int:
    """``-1 -> 0``, ``+1 -> 1``. Any other value is an error.

    Never collapse to a "non-edge" class; observed hyperlinks are all real edges.
    """
    if label == -1:
        return 0
    if label == 1:
        return 1
    raise ValueError(f"Unexpected POST_LABEL {label!r}; expected -1 or +1")


def build_edge_table(raw_dfs: dict[str, Any], cfg: dict[str, Any]) -> Any:
    """Return the cleaned edge DataFrame ready for feature engineering."""
    raise NotImplementedError("data.preprocess.build_edge_table is not implemented yet")


def build_node_index(edge_df: Any, embeddings_df: Any | None) -> Any:
    """Return a ``subreddit -> int`` mapping with stable, contiguous ids."""
    raise NotImplementedError("data.preprocess.build_node_index is not implemented yet")


__all__ = ["remap_post_label", "build_edge_table", "build_node_index"]
