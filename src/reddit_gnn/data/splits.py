"""Train / val / test edge splits with disjoint message-passing and supervision sets.

The most error-prone correctness invariant of the entire project lives here:
**no edge that we supervise on may also be present in the encoder's
message-passing graph for that split.** Violating this is a silent leak —
the model gets to see the label it is supposed to predict and metrics look
suspiciously good.

The default split is :func:`chronological_edge_split` — a strict row-position
cut on a frame already sorted by ``TIMESTAMP``. This is the realistic regime:
val and test edges always have timestamps no earlier than every train edge,
so val/test supervision can never "leak into the past" of the message-passing
graph.

For ablations we also expose :func:`stratified_random_edge_split` — a
class-balanced random split. It is **not** the default and is documented as
such; use it only to measure how much the temporal split hurts.

:func:`build_message_passing_split` then takes the :class:`SplitResult` and
produces, for each of ``train`` / ``val`` / ``test``, a dict of tensors with:

* ``mp_edge_index`` / ``mp_edge_attr`` / ``mp_edge_time``   — the encoder's
  message-passing graph for that split (no labels seen here, by construction).
* ``sup_edge_index`` / ``sup_edge_label`` / ``sup_edge_time`` — the edges the
  loss is computed on for that split.
* ``mp_idx`` / ``sup_idx`` — row indices into the original frame; the
  downstream PyG builder uses these to pick the right rows out of any
  per-edge feature tensor (engineered features include scaled temporal cols
  that we never want to recompute).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from reddit_gnn.data.features import parse_post_properties
from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SplitResult:
    """Output of :func:`chronological_edge_split`.

    Indices are arrays of row positions into the (timestamp-sorted) edge frame
    that produced them. Cutoff timestamps record the inclusive upper bound of
    each preceding split — useful for figures and sanity checks.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_cutoff_ts: pd.Timestamp
    val_cutoff_ts: pd.Timestamp


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


def _assert_sorted_by_timestamp(df: pd.DataFrame) -> None:
    if "TIMESTAMP" not in df.columns:
        raise KeyError("df must contain a 'TIMESTAMP' column")
    ts_int = df["TIMESTAMP"].astype("int64").to_numpy()
    if ts_int.size and (np.diff(ts_int) < 0).any():
        raise ValueError(
            "df must be sorted by TIMESTAMP ascending before splitting. "
            "Run reddit_gnn.data.preprocess.clean_edges first."
        )


def chronological_edge_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Cut the edge frame at row-position quantiles of TIMESTAMP.

    The frame **must** already be sorted ascending by ``TIMESTAMP``. We split
    by row position (not value) so that ties at split boundaries deterministically
    fall into the earlier split — the temporal monotonicity guarantee then
    holds up to equality at the boundary timestamp.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(
            f"train + val + test ratios must sum to 1; got {total!r}"
        )
    for name, r in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if r < 0:
            raise ValueError(f"{name}_ratio must be non-negative; got {r}")

    _assert_sorted_by_timestamp(df)

    n = len(df)
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    # Test soaks up the remainder so the three slices exactly partition [0, n).
    n_train_val = n_train + n_val
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train_val, dtype=np.int64)
    test_idx = np.arange(n_train_val, n, dtype=np.int64)

    ts = df["TIMESTAMP"]
    train_cutoff = ts.iloc[n_train - 1] if n_train > 0 else pd.NaT
    val_cutoff = ts.iloc[n_train_val - 1] if n_val > 0 else train_cutoff

    log.info(
        "chronological_edge_split: %d -> train=%d val=%d test=%d "
        "(train_cutoff=%s, val_cutoff=%s)",
        n,
        len(train_idx),
        len(val_idx),
        len(test_idx),
        train_cutoff,
        val_cutoff,
    )

    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_cutoff_ts=pd.Timestamp(train_cutoff),
        val_cutoff_ts=pd.Timestamp(val_cutoff),
    )


# ---------------------------------------------------------------------------
# Stratified random (ablation only)
# ---------------------------------------------------------------------------


def stratified_random_edge_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Class-stratified random split. **Ablation only** — not the default.

    Use to quantify the gap between random and chronological evaluation.
    Returns a SplitResult whose ``*_cutoff_ts`` fields are the global min/max
    timestamps of the train / train+val halves (they have no chronological
    meaning under this split).
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"train + val + test ratios must sum to 1; got {total!r}")
    if "label_binary" not in df.columns:
        raise KeyError("stratified_random_edge_split requires a 'label_binary' column")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label, idx in df.groupby("label_binary").groups.items():
        del label
        idx_arr = np.asarray(idx, dtype=np.int64)
        rng.shuffle(idx_arr)
        n = len(idx_arr)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        train_parts.append(idx_arr[:n_train])
        val_parts.append(idx_arr[n_train : n_train + n_val])
        test_parts.append(idx_arr[n_train + n_val :])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)

    ts = df["TIMESTAMP"]
    train_cutoff = ts.iloc[train_idx].max() if len(train_idx) else pd.NaT
    val_cutoff = ts.iloc[np.concatenate([train_idx, val_idx])].max() if len(val_idx) else train_cutoff

    log.warning(
        "stratified_random_edge_split is for ABLATIONS only. The default "
        "evaluation protocol uses chronological_edge_split."
    )
    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_cutoff_ts=pd.Timestamp(train_cutoff),
        val_cutoff_ts=pd.Timestamp(val_cutoff),
    )


# ---------------------------------------------------------------------------
# Message-passing / supervision partition
# ---------------------------------------------------------------------------


def _ts_int64(df: pd.DataFrame, idx: np.ndarray) -> torch.Tensor:
    """TIMESTAMP[idx] as a torch int64 tensor of nanoseconds since epoch."""
    arr = df["TIMESTAMP"].iloc[idx].astype("int64").to_numpy().copy()
    return torch.from_numpy(arr).to(torch.int64)


def _edge_index(df: pd.DataFrame, idx: np.ndarray) -> torch.Tensor:
    src = df["source_id"].iloc[idx].to_numpy().astype(np.int64, copy=True)
    dst = df["target_id"].iloc[idx].to_numpy().astype(np.int64, copy=True)
    return torch.from_numpy(np.stack([src, dst], axis=0))


def _build_one_split(
    df: pd.DataFrame,
    edge_attr_all: np.ndarray,
    mp_idx: np.ndarray,
    sup_idx: np.ndarray,
) -> dict[str, torch.Tensor]:
    mp_edge_index = _edge_index(df, mp_idx)
    mp_edge_time = _ts_int64(df, mp_idx)
    mp_edge_attr = torch.from_numpy(
        np.ascontiguousarray(edge_attr_all[mp_idx]).copy()
    ).to(torch.float32)

    sup_edge_index = _edge_index(df, sup_idx)
    sup_edge_time = _ts_int64(df, sup_idx)
    sup_edge_label = torch.from_numpy(
        df["label_binary"].iloc[sup_idx].to_numpy().astype(np.int64, copy=True)
    ).to(torch.int64)

    return {
        "mp_edge_index": mp_edge_index,
        "mp_edge_attr": mp_edge_attr,
        "mp_edge_time": mp_edge_time,
        "sup_edge_index": sup_edge_index,
        "sup_edge_label": sup_edge_label,
        "sup_edge_time": sup_edge_time,
        "mp_idx": torch.from_numpy(np.asarray(mp_idx, dtype=np.int64).copy()),
        "sup_idx": torch.from_numpy(np.asarray(sup_idx, dtype=np.int64).copy()),
    }


def build_message_passing_split(
    df: pd.DataFrame,
    split: SplitResult,
    disjoint_train_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, dict[str, torch.Tensor]]:
    """Split each fold into disjoint message-passing and supervision tensors.

    Returns ``{"train": {...}, "val": {...}, "test": {...}}``. Inside each
    inner dict the keys match the user spec
    (``mp_edge_index / mp_edge_attr / mp_edge_time / sup_edge_index /
    sup_edge_label / sup_edge_time``); we additionally surface ``mp_idx`` and
    ``sup_idx`` as row indices into the original ``df``, which the PyG builder
    uses to pull the right rows out of an engineered edge feature tensor.

    Rules:
        * **TRAIN** — uniformly partition ``train_idx`` into a supervision
          subset of size ``disjoint_train_ratio * len(train_idx)`` and a
          message-passing subset (the remainder). The two sets are disjoint
          by construction.
        * **VAL** — ``mp = all train edges`` (both train_mp and train_sup
          halves), ``sup = val edges``. Validation lets the encoder see the
          complete train graph but never any val edges.
        * **TEST** — ``mp = train + val edges``, ``sup = test edges``.

    Raw POST_PROPERTIES (``p0..p85``) are used as the message-passing
    ``edge_attr`` (SNAP normalizes them; no scaling required). Engineered /
    scaled edge features stay in the supervision-side ``edge_label_attr``,
    which the PyG builder attaches downstream.
    """
    if not 0.0 <= disjoint_train_ratio < 1.0:
        raise ValueError(
            f"disjoint_train_ratio must be in [0, 1); got {disjoint_train_ratio!r}"
        )

    edge_attr_all = parse_post_properties(df)

    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    val_idx = np.asarray(split.val_idx, dtype=np.int64)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)

    # ----- Train: partition into disjoint mp / sup halves
    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    n_train_sup = int(np.floor(n_train * disjoint_train_ratio))
    permuted = rng.permutation(n_train)
    train_sup_local = np.sort(permuted[:n_train_sup])
    train_mp_local = np.sort(permuted[n_train_sup:])
    train_mp_idx = train_idx[train_mp_local]
    train_sup_idx = train_idx[train_sup_local]

    # ----- Val: mp = ALL train edges (mp + sup halves), sup = val
    val_mp_idx = train_idx
    val_sup_idx = val_idx

    # ----- Test: mp = train + val edges (chronologically sorted), sup = test
    test_mp_idx = np.concatenate([train_idx, val_idx])
    test_mp_idx.sort()
    test_sup_idx = test_idx

    out = {
        "train": _build_one_split(df, edge_attr_all, train_mp_idx, train_sup_idx),
        "val": _build_one_split(df, edge_attr_all, val_mp_idx, val_sup_idx),
        "test": _build_one_split(df, edge_attr_all, test_mp_idx, test_sup_idx),
    }

    log.info(
        "build_message_passing_split: train mp=%d sup=%d | val mp=%d sup=%d | "
        "test mp=%d sup=%d (disjoint_train_ratio=%.3f, seed=%d)",
        len(train_mp_idx),
        len(train_sup_idx),
        len(val_mp_idx),
        len(val_sup_idx),
        len(test_mp_idx),
        len(test_sup_idx),
        disjoint_train_ratio,
        seed,
    )
    return out


__all__ = [
    "SplitResult",
    "build_message_passing_split",
    "chronological_edge_split",
    "stratified_random_edge_split",
]
