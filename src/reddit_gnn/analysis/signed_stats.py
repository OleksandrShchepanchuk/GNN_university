"""Statistics specific to the signed structure of the graph.

* :func:`compute_label_stats` — overall class balance.
* :func:`negative_ratio_by_source` / :func:`negative_ratio_by_target` —
  per-subreddit negativity, top-k for the EDA tables.
* :func:`signed_triad_counts` — balanced vs unbalanced triads on a *sampled*
  subgraph (uses NetworkX; the full graph is too dense for triangle
  enumeration in pure Python).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def compute_label_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Class balance: counts and ratios of negative vs neutral/positive edges."""
    if df.empty:
        return {
            "num_positive": 0,
            "num_negative": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
        }
    counts = df["label_binary"].value_counts()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    total = n_neg + n_pos
    return {
        "num_positive": n_pos,
        "num_negative": n_neg,
        "positive_ratio": n_pos / total if total else 0.0,
        "negative_ratio": n_neg / total if total else 0.0,
    }


def _neg_ratio_by_group(df: pd.DataFrame, by: str, top_k: int) -> pd.DataFrame:
    g = df.groupby(by)["label_binary"].agg(["count", "mean"])
    g = g.rename(columns={"count": "edge_count", "mean": "positive_ratio"})
    g["negative_ratio"] = 1.0 - g["positive_ratio"]
    g = g.sort_values(["negative_ratio", "edge_count"], ascending=[False, False])
    g = g.head(top_k).reset_index()
    return g[[by, "edge_count", "negative_ratio", "positive_ratio"]]


def negative_ratio_by_source(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Top-``k`` source subreddits by fraction of *outgoing* negative edges."""
    return _neg_ratio_by_group(df, by="source_subreddit_norm", top_k=top_k)


def negative_ratio_by_target(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Top-``k`` target subreddits by fraction of *incoming* negative edges."""
    return _neg_ratio_by_group(df, by="target_subreddit_norm", top_k=top_k)


def _aggregate_signed_undirected(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse directed edges to undirected, sign = sum of ±1 labels."""
    signs = (2 * df["label_binary"].astype(np.int8) - 1).to_numpy()
    u = np.minimum(df["source_id"].to_numpy(), df["target_id"].to_numpy())
    v = np.maximum(df["source_id"].to_numpy(), df["target_id"].to_numpy())
    agg = pd.DataFrame({"u": u, "v": v, "sign_sum": signs})
    agg = agg[agg["u"] != agg["v"]]
    agg = agg.groupby(["u", "v"], as_index=False)["sign_sum"].sum()
    agg["sign"] = np.where(agg["sign_sum"] >= 0, 1, -1).astype(np.int8)
    return agg


def signed_triad_counts(
    df: pd.DataFrame,
    sample_cap: int = 50_000,
    seed: int = 42,
) -> dict[str, int | bool]:
    """Count balanced vs unbalanced triangles on a sampled, signed, undirected graph.

    Sampling is on the *edge frame* (uniform random rows) — we then collapse
    duplicates and direction to an undirected graph whose sign is the
    majority of contributing ±1 labels. Balance theory: a triangle is
    *balanced* iff the product of its three edge signs is positive.

    Returns counts together with ``sampled`` flag and the population size used.
    """
    import networkx as nx

    if df.empty:
        return {
            "balanced": 0,
            "unbalanced": 0,
            "total_triangles": 0,
            "sampled": False,
            "sample_size": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
        }

    sampled = len(df) > sample_cap
    work = df.sample(n=sample_cap, random_state=seed) if sampled else df

    edges = _aggregate_signed_undirected(work)

    G: nx.Graph = nx.Graph()
    for u, v, sign in edges[["u", "v", "sign"]].itertuples(index=False, name=None):
        G.add_edge(int(u), int(v), sign=int(sign))

    balanced = 0
    unbalanced = 0
    # Enumerate triangles via the canonical u < v < w pattern. NetworkX's
    # built-in triangle iterator is for undirected counts; we need per-triangle
    # signs, so we do a small manual loop.
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    for u, u_nbrs in adj.items():
        u_nbrs_higher = {n for n in u_nbrs if n > u}
        for v in u_nbrs_higher:
            common = u_nbrs_higher & adj[v]
            for w in common:
                if w <= v:
                    continue
                s_uv = G[u][v]["sign"]
                s_vw = G[v][w]["sign"]
                s_uw = G[u][w]["sign"]
                if s_uv * s_vw * s_uw > 0:
                    balanced += 1
                else:
                    unbalanced += 1

    return {
        "balanced": balanced,
        "unbalanced": unbalanced,
        "total_triangles": balanced + unbalanced,
        "sampled": bool(sampled),
        "sample_size": len(work),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
    }


__all__ = [
    "compute_label_stats",
    "negative_ratio_by_source",
    "negative_ratio_by_target",
    "signed_triad_counts",
]
