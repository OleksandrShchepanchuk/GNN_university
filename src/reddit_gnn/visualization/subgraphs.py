"""Signed subgraph visualizations using NetworkX (sampled — not full-graph)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from reddit_gnn.visualization import NEGATIVE_COLOR, POSITIVE_COLOR, _maybe_save


def _build_signed_digraph(edges: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from an edge DataFrame, preserving per-edge label."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for u, v, label in edges[
        ["source_subreddit_norm", "target_subreddit_norm", "label_binary"]
    ].itertuples(index=False, name=None):
        G.add_edge(str(u), str(v), label=int(label))
    return G


def _draw_signed_digraph(
    G: nx.MultiDiGraph,
    *,
    title: str,
    seed: int = 42,
    node_size: int = 220,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    if G.number_of_nodes() == 0:
        ax.set_title(title + " (empty)")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, seed=seed, k=1.5 / max(G.number_of_nodes(), 1) ** 0.4)

    neg_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d["label"] == 0]
    pos_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d["label"] == 1]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color="#cccccc", edgecolors="black"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in neg_edges],
        ax=ax,
        edge_color=NEGATIVE_COLOR,
        arrows=True,
        arrowsize=10,
        width=1.0,
        alpha=0.85,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in pos_edges],
        ax=ax,
        edge_color=POSITIVE_COLOR,
        arrows=True,
        arrowsize=10,
        width=1.0,
        alpha=0.85,
        connectionstyle="arc3,rad=0.08",
    )

    # Label only the top-degree nodes to keep the figure readable.
    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[: min(25, len(deg))]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes}, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis("off")

    # Custom legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=2, label="negative (label=0)"),
        Line2D([0], [0], color=POSITIVE_COLOR, lw=2, label="neutral/positive (label=1)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()
    return fig


def plot_sampled_signed_subgraph(
    df: pd.DataFrame,
    *,
    max_edges: int = 300,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> Figure:
    """Random uniform sample of edges drawn as a directed signed graph."""
    n = min(max_edges, len(df))
    if n == 0:
        return _draw_signed_digraph(nx.MultiDiGraph(), title="Sampled signed subgraph", seed=seed)
    sampled = df.sample(n=n, random_state=seed)
    G = _build_signed_digraph(sampled)
    fig = _draw_signed_digraph(
        G,
        title=f"Sampled signed subgraph ({n} edges)",
        seed=seed,
    )
    return _maybe_save(fig, save_path)


def plot_ego_signed_subgraph(
    df: pd.DataFrame,
    subreddit: str,
    *,
    radius: int = 1,
    max_edges: int = 300,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> Figure:
    """Ego subgraph (BFS up to ``radius`` hops) around ``subreddit``.

    The subreddit name is matched against the lowercased / stripped values
    in ``source_subreddit_norm`` / ``target_subreddit_norm``.
    """
    target = subreddit.strip().lower()

    # Frontier expansion: at each hop, add all nodes incident to the current set.
    frontier: set[str] = {target}
    nodes: set[str] = {target}
    for _ in range(max(radius, 1)):
        incident = df[
            df["source_subreddit_norm"].isin(frontier) | df["target_subreddit_norm"].isin(frontier)
        ]
        new_nodes = set(incident["source_subreddit_norm"]).union(incident["target_subreddit_norm"])
        new_nodes -= nodes
        if not new_nodes:
            break
        nodes |= new_nodes
        frontier = new_nodes

    induced = df[df["source_subreddit_norm"].isin(nodes) & df["target_subreddit_norm"].isin(nodes)]
    n = min(max_edges, len(induced))
    if n == 0:
        rng = np.random.default_rng(seed)  # unused, but kept for reproducibility
        del rng
        return _draw_signed_digraph(
            nx.MultiDiGraph(),
            title=f"Ego signed subgraph: {subreddit} (no edges in {radius}-hop ego)",
            seed=seed,
        )
    if len(induced) > n:
        induced = induced.sample(n=n, random_state=seed)
    G = _build_signed_digraph(induced)
    fig = _draw_signed_digraph(
        G,
        title=f"Ego signed subgraph: {subreddit} (radius={radius}, {n} edges)",
        seed=seed,
    )
    return _maybe_save(fig, save_path)


__all__ = ["plot_ego_signed_subgraph", "plot_sampled_signed_subgraph"]
