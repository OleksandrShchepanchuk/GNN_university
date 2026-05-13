"""Node and edge feature construction.

Node features:
    * 300-D LIWC embeddings from ``web-redditEmbeddings-subreddits.csv`` for
      subreddits that have them; zero-vector + ``has_embedding`` flag otherwise.
    * Optional structural features (in/out degree, total volume) computed
      strictly on the **train** subgraph to avoid leakage.

Edge features (per directed hyperlink event):
    * 86-D ``POST_PROPERTIES`` vector parsed from the SNAP TSV;
    * log(post age) relative to the dataset start;
    * src_outdeg / dst_indeg computed on train edges only;
    * number of common neighbors in the train graph;
    * optional Hadamard / abs-diff of endpoint embeddings.

All scalers (``StandardScaler``) are fit on training rows only.
"""

from __future__ import annotations

from typing import Any


def build_node_features(node_index: Any, embeddings_df: Any, cfg: dict[str, Any]) -> Any:
    """Return a ``(num_nodes, d)`` tensor of node features."""
    raise NotImplementedError("data.features.build_node_features is not implemented yet")


def build_edge_features(edge_df: Any, train_mask: Any, cfg: dict[str, Any]) -> Any:
    """Return a ``(num_edges, m)`` tensor of edge features (no leakage)."""
    raise NotImplementedError("data.features.build_edge_features is not implemented yet")


def parse_post_properties(s: str) -> list[float]:
    """Parse the comma-separated ``POST_PROPERTIES`` field into 86 floats."""
    raise NotImplementedError("data.features.parse_post_properties is not implemented yet")


__all__ = ["build_node_features", "build_edge_features", "parse_post_properties"]
