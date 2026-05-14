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
) -> Figure:
    """Render a signed multidigraph with:

    * **Node size** proportional to degree (popular subreddits look big).
    * **Node color** by the share of outgoing edges with `label = 0` —
      continuous red → grey → green gradient via the `RdYlGn_r` colormap.
      A node that sends mostly negative edges turns red; mostly positive
      turns green; mixed lands in the middle.
    * **Edge color** by sign (negative red, positive green) with mild
      transparency so dense regions read as texture instead of a solid block.
    * **Halo'd labels** on the top-degree nodes only; smaller hub names get
      a white outline so they survive against dense edge tangles.
    """
    fig, ax = plt.subplots(figsize=(10.5, 9))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    if G.number_of_nodes() == 0:
        ax.set_title(title + "  —  (empty subgraph)", loc="left", fontsize=12)
        ax.axis("off")
        return fig

    # Layout: kamada_kawai is more visually balanced for ≤ 200 nodes; fall back
    # to spring for larger graphs where KK becomes very slow.
    n_nodes = G.number_of_nodes()
    if n_nodes <= 200:
        try:
            pos = nx.kamada_kawai_layout(nx.DiGraph(G))
        except Exception:
            pos = nx.spring_layout(G, seed=seed, k=1.6 / max(n_nodes, 1) ** 0.42, iterations=80)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.6 / max(n_nodes, 1) ** 0.42, iterations=80)

    # Per-node "negative-share" of outgoing edges → continuous color.
    out_total: dict[str, int] = {n: 0 for n in G.nodes()}
    out_neg: dict[str, int] = {n: 0 for n in G.nodes()}
    for u, _v, d in G.edges(data=True):
        out_total[u] += 1
        if d["label"] == 0:
            out_neg[u] += 1
    neg_share = np.array(
        [out_neg[n] / out_total[n] if out_total[n] > 0 else 0.5 for n in G.nodes()],
        dtype=float,
    )

    # Per-node total degree → size. Clamp so isolated nodes are still visible
    # and hubs don't dominate the canvas.
    deg = dict(G.degree())
    deg_arr = np.array([deg[n] for n in G.nodes()], dtype=float)
    max_deg = max(deg_arr.max(), 1.0)
    node_sizes = 80 + 480 * (deg_arr / max_deg) ** 0.6

    # Edge buckets — negatives drawn second so they sit on top of positives.
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["label"] == 1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["label"] == 0]

    # Draw positive edges first (background)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=pos_edges,
        ax=ax,
        edge_color=POSITIVE_COLOR,
        arrows=True,
        arrowsize=8,
        arrowstyle="-|>",
        width=0.9,
        alpha=0.45,
        connectionstyle="arc3,rad=0.10",
        min_target_margin=8,
    )
    # Negative edges on top, slightly thicker + more opaque (rare-class focus).
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=neg_edges,
        ax=ax,
        edge_color=NEGATIVE_COLOR,
        arrows=True,
        arrowsize=11,
        arrowstyle="-|>",
        width=1.5,
        alpha=0.85,
        connectionstyle="arc3,rad=0.13",
        min_target_margin=8,
    )

    # Nodes
    node_collection = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=neg_share,
        cmap=plt.cm.RdYlGn_r,
        vmin=0.0,
        vmax=1.0,
        edgecolors="#333",
        linewidths=0.7,
    )

    # Labels: only top-K hubs, with a white halo so they're readable on red.
    k_labels = min(15, n_nodes)
    top_nodes = sorted(deg, key=deg.get, reverse=True)[:k_labels]
    text_objs = nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in top_nodes},
        ax=ax,
        font_size=8.5,
        font_weight="semibold",
    )
    import matplotlib.patheffects as path_effects

    for t in text_objs.values():
        t.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])

    n_neg, n_pos = len(neg_edges), len(pos_edges)
    n_total = max(n_neg + n_pos, 1)
    ax.set_title(
        f"{title}  —  {n_nodes} nodes, {n_total} edges ({n_neg / n_total:.1%} negative)",
        loc="left",
        fontsize=12,
    )
    ax.axis("off")

    # Color bar for the node-color scale
    cbar = fig.colorbar(node_collection, ax=ax, shrink=0.5, pad=0.02, fraction=0.04)
    cbar.set_label("share of outgoing edges that are negative", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Edge legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=2.4, label=f"negative edges ({n_neg})"),
        Line2D(
            [0], [0], color=POSITIVE_COLOR, lw=1.4, alpha=0.6, label=f"positive edges ({n_pos})"
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)
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
