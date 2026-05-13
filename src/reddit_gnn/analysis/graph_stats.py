"""Whole-graph descriptive statistics.

All routines accept the processed edge DataFrame produced by
:mod:`reddit_gnn.data.preprocess` (columns include ``source_id``,
``target_id``, ``label_binary``, ``TIMESTAMP``, ``source_subreddit_norm``,
``target_subreddit_norm``).

For anything that would otherwise iterate edges across the full ~858k-edge
graph we use ``scipy.sparse`` directly — NetworkX is reserved for sampled
subgraphs in :mod:`reddit_gnn.analysis.signed_stats` and the visualization
helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def _num_nodes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(max(df["source_id"].max(), df["target_id"].max())) + 1


def compute_basic_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Top-level counts: nodes, edges, density, degree summary, duplicates."""
    if df.empty:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "self_loop_count": 0,
            "duplicate_edge_count": 0,
        }

    n_nodes = _num_nodes(df)
    n_edges = len(df)
    src = df["source_id"].to_numpy()
    dst = df["target_id"].to_numpy()

    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)

    self_loops = int((src == dst).sum())
    unique_pair_count = df[["source_id", "target_id"]].drop_duplicates().shape[0]
    duplicate_edge_count = int(n_edges - unique_pair_count)

    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": float(density),
        "avg_in_degree": float(in_deg.mean()),
        "avg_out_degree": float(out_deg.mean()),
        "max_in_degree": int(in_deg.max()),
        "max_out_degree": int(out_deg.max()),
        "self_loop_count": self_loops,
        "duplicate_edge_count": duplicate_edge_count,
    }


def compute_degree_stats(df: pd.DataFrame) -> dict[str, np.ndarray | dict[str, float]]:
    """Per-node degree arrays plus a small summary dict for plotting."""
    n_nodes = _num_nodes(df)
    src = df["source_id"].to_numpy() if not df.empty else np.array([], dtype=np.int64)
    dst = df["target_id"].to_numpy() if not df.empty else np.array([], dtype=np.int64)
    out_deg = np.bincount(src, minlength=n_nodes).astype(np.int64)
    in_deg = np.bincount(dst, minlength=n_nodes).astype(np.int64)
    total_deg = in_deg + out_deg

    def _summary(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": total_deg,
        "summary": {
            "in": _summary(in_deg),
            "out": _summary(out_deg),
            "total": _summary(total_deg),
        },
    }


def _build_sparse_adjacency(df: pd.DataFrame, *, binary: bool = True) -> sp.csr_matrix:
    n_nodes = _num_nodes(df)
    src = df["source_id"].to_numpy()
    dst = df["target_id"].to_numpy()
    data = np.ones(len(df), dtype=np.int32)
    mat = sp.coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    if binary:
        mat.data = np.minimum(mat.data, 1)
        mat.eliminate_zeros()
    return mat


def compute_reciprocity_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Reciprocity = |{(u,v): (v,u) also exists}| / |unique directed edges|.

    Implemented via a sparse binary adjacency: ``A.multiply(A.T)`` has a non-zero
    entry at exactly the reciprocated pairs. We never materialize the dense
    ``A`` matrix, so this is O(nnz) memory.
    """
    if df.empty:
        return {
            "reciprocity": 0.0,
            "reciprocated_edge_count": 0,
            "unique_directed_edges": 0,
            "total_rows": 0,
        }
    A = _build_sparse_adjacency(df, binary=True)
    A_T = A.T.tocsr()
    reciprocal = A.multiply(A_T)
    n_recip = int(reciprocal.nnz)
    n_unique = int(A.nnz)
    reciprocity = n_recip / n_unique if n_unique > 0 else 0.0
    return {
        "reciprocity": float(reciprocity),
        "reciprocated_edge_count": n_recip,
        "unique_directed_edges": n_unique,
        "total_rows": len(df),
    }


def compute_component_stats(
    df: pd.DataFrame,
    sample_cap: int | None = None,
    seed: int = 42,
) -> dict[str, int | bool]:
    """Weakly + strongly connected component statistics via ``scipy.sparse.csgraph``.

    If ``sample_cap`` is set and the edge frame is larger, a uniform random
    subset of edges is used; nodes are re-indexed so isolated holdouts don't
    inflate the component count.
    """
    if df.empty:
        return {
            "wcc_count": 0,
            "largest_wcc_size": 0,
            "scc_count": 0,
            "largest_scc_size": 0,
            "sampled": False,
            "sample_size": 0,
        }

    sampled = sample_cap is not None and len(df) > sample_cap
    work = df.sample(n=sample_cap, random_state=seed).reset_index(drop=True) if sampled else df

    # Compact the node index to the set of nodes touched by the (possibly
    # sampled) edges; otherwise isolated SNAP-wide nodes look like extra
    # components in the sampled graph.
    src_arr = work["source_id"].to_numpy()
    dst_arr = work["target_id"].to_numpy()
    present = pd.unique(np.concatenate([src_arr, dst_arr]))
    remap = {int(old): new for new, old in enumerate(present)}
    src_local = np.fromiter((remap[int(s)] for s in src_arr), dtype=np.int64, count=len(src_arr))
    dst_local = np.fromiter((remap[int(d)] for d in dst_arr), dtype=np.int64, count=len(dst_arr))
    n_local = len(present)

    mat = sp.coo_matrix(
        (np.ones(len(work), dtype=np.int32), (src_local, dst_local)),
        shape=(n_local, n_local),
    ).tocsr()

    n_wcc, wcc_labels = connected_components(mat, directed=False, return_labels=True)
    wcc_sizes = np.bincount(wcc_labels)
    n_scc, scc_labels = connected_components(
        mat, directed=True, connection="strong", return_labels=True
    )
    scc_sizes = np.bincount(scc_labels)

    return {
        "wcc_count": int(n_wcc),
        "largest_wcc_size": int(wcc_sizes.max()),
        "scc_count": int(n_scc),
        "largest_scc_size": int(scc_sizes.max()),
        "sampled": bool(sampled),
        "sample_size": len(work),
    }


__all__ = [
    "compute_basic_stats",
    "compute_component_stats",
    "compute_degree_stats",
    "compute_reciprocity_stats",
]
