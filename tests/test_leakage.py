"""Hard guards against label / temporal leakage in the split pipeline.

The invariant being defended:

    * Inside every split, the encoder's message-passing edges and the
      supervision edges are **disjoint** as ``(src, target, time)`` triples.
      A leak here would let the model read the label it is supposed to
      predict via the aggregated edge.
    * For ``val`` and ``test`` splits, every supervision timestamp is
      ``>= max(train_mp_edge_time)`` — no "future" edge sneaks into the
      "past" message-passing graph.
    * Supervision edges in ``train`` / ``val`` / ``test`` are pairwise
      disjoint as ``(src, target, time)`` triples.

We test on a deterministic 500-edge synthetic frame AND, if a processed
parquet is present, on the real SNAP-derived dataset (this catches issues
that only surface at scale or with the real timestamp distribution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reddit_gnn.data.splits import (
    assert_no_leakage as _package_assert_no_leakage,
)
from reddit_gnn.data.splits import (
    build_message_passing_split,
    chronological_edge_split,
)
from reddit_gnn.paths import PATHS

POST_PROPERTIES_DIM = 86


# ---------------------------------------------------------------------------
# Synthetic data + shared assertion helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 500, *, num_nodes: int = 20, seed: int = 0) -> pd.DataFrame:
    """Deterministic 500-edge frame sorted by TIMESTAMP, mixed labels."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2014-01-01", periods=n, freq="h")
    src = rng.integers(0, num_nodes, size=n)
    tgt = rng.integers(0, num_nodes, size=n)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % num_nodes
    return pd.DataFrame(
        {
            "source_id": src.astype(np.int64),
            "target_id": tgt.astype(np.int64),
            "source_subreddit_norm": [f"sr{s}" for s in src],
            "target_subreddit_norm": [f"sr{t}" for t in tgt],
            "is_title": rng.integers(0, 2, size=n).astype(np.int8),
            "label_binary": rng.integers(0, 2, size=n).astype(np.int8),
            "TIMESTAMP": ts,
            **{
                f"p{i}": rng.standard_normal(n).astype(np.float32)
                for i in range(POST_PROPERTIES_DIM)
            },
        }
    )


def _triples(edge_index, edge_time) -> set[tuple[int, int, int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    times = edge_time.tolist()
    return set(zip(src, dst, times, strict=True))


def _assert_no_leakage(df: pd.DataFrame, *, disjoint_train_ratio: float = 0.2) -> None:
    """Thin wrapper around the package function — keeps the test stable across refactors."""
    _package_assert_no_leakage(df, disjoint_train_ratio=disjoint_train_ratio, seed=42)


# ---------------------------------------------------------------------------
# Synthetic checks
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """Modules import cleanly even without the real dataset present."""
    from reddit_gnn.data import features, pyg_dataset, splits  # noqa: F401
    from reddit_gnn.training import loaders  # noqa: F401


def test_no_leakage_on_synthetic_500_edges() -> None:
    df = _make_synthetic_df(n=500)
    _assert_no_leakage(df)


def test_train_mp_and_sup_are_disjoint_under_varying_ratios() -> None:
    """The disjoint_train_ratio knob doesn't break the within-train invariant."""
    df = _make_synthetic_df(n=500)
    sp = chronological_edge_split(df)
    for ratio in (0.0, 0.1, 0.3, 0.5, 0.9):
        splits = build_message_passing_split(df, sp, disjoint_train_ratio=ratio, seed=7)
        mp = _triples(splits["train"]["mp_edge_index"], splits["train"]["mp_edge_time"])
        sup = _triples(splits["train"]["sup_edge_index"], splits["train"]["sup_edge_time"])
        assert mp.isdisjoint(sup), f"train mp/sup overlap at ratio={ratio}"
        expected_sup = int(np.floor(len(sp.train_idx) * ratio))
        assert len(sup) == expected_sup
        assert len(mp) + len(sup) == len(sp.train_idx)


# ---------------------------------------------------------------------------
# Real dataset check — skipped automatically when no parquet is present
# ---------------------------------------------------------------------------


_REAL_PARQUET = PATHS.data_processed / "edges.parquet"


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="Processed parquet not present; run `make data` to enable this check.",
)
def test_no_leakage_on_real_processed_dataset() -> None:
    """The same leakage suite, on the real SNAP-derived edge frame."""
    df = pd.read_parquet(_REAL_PARQUET)
    # The frame is supposed to already be sorted by TIMESTAMP after
    # `clean_edges`, but the check is cheap — re-sort defensively for
    # robustness against externally supplied parquets.
    df = df.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)
    _assert_no_leakage(df)
