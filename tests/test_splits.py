"""Tests for ``reddit_gnn.data.splits``.

The split primitives are correctness-critical; the heavy "no-leakage"
end-to-end checks live in ``tests/test_leakage.py`` and exercise the joint
behavior of ``chronological_edge_split`` + ``build_message_passing_split``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reddit_gnn.data.splits import (
    SplitResult,
    build_message_passing_split,
    chronological_edge_split,
    stratified_random_edge_split,
)

POST_PROPERTIES_DIM = 86


def _synthetic_df(n: int = 500, *, num_nodes: int = 20, seed: int = 0) -> pd.DataFrame:
    """Sorted-by-TIMESTAMP synthetic edge frame with mixed labels."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2014-01-01", periods=n, freq="h")
    src = rng.integers(0, num_nodes, size=n)
    tgt = rng.integers(0, num_nodes, size=n)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % num_nodes
    labels = rng.integers(0, 2, size=n).astype(np.int8)
    props = {f"p{i}": rng.standard_normal(n).astype(np.float32) for i in range(POST_PROPERTIES_DIM)}
    return pd.DataFrame(
        {
            "source_id": src.astype(np.int64),
            "target_id": tgt.astype(np.int64),
            "source_subreddit_norm": [f"sr{s}" for s in src],
            "target_subreddit_norm": [f"sr{t}" for t in tgt],
            "is_title": rng.integers(0, 2, size=n).astype(np.int8),
            "label_binary": labels,
            "TIMESTAMP": ts,
            **props,
        }
    )


# ---------------------------------------------------------------------------
# Chronological split — ratio / disjoint / boundary tests
# ---------------------------------------------------------------------------


def test_chronological_split_ratios_sum_to_one() -> None:
    """train + val + test indices exactly partition the frame."""
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df, 0.70, 0.15, 0.15)

    total = len(sp.train_idx) + len(sp.val_idx) + len(sp.test_idx)
    assert total == len(df)
    # Floor-based sizes for 500 with 0.70/0.15/0.15 -> 350 / 75 / 75
    assert len(sp.train_idx) == 350
    assert len(sp.val_idx) == 75
    assert len(sp.test_idx) == 75


def test_chronological_split_ratios_must_sum_to_one() -> None:
    df = _synthetic_df(n=100)
    with pytest.raises(ValueError):
        chronological_edge_split(df, 0.7, 0.2, 0.2)
    with pytest.raises(ValueError):
        chronological_edge_split(df, 0.7, -0.05, 0.35)


def test_chronological_split_indices_are_disjoint() -> None:
    """No row index belongs to more than one split."""
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)
    train, val, test = (
        set(sp.train_idx.tolist()),
        set(sp.val_idx.tolist()),
        set(sp.test_idx.tolist()),
    )
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(range(len(df)))


def test_chronological_split_time_monotonicity_at_boundaries() -> None:
    """Timestamps respect train < val < test at the boundary rows."""
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)

    train_max = df["TIMESTAMP"].iloc[sp.train_idx].max()
    val_min = df["TIMESTAMP"].iloc[sp.val_idx].min()
    val_max = df["TIMESTAMP"].iloc[sp.val_idx].max()
    test_min = df["TIMESTAMP"].iloc[sp.test_idx].min()

    assert train_max <= val_min
    assert val_max <= test_min
    assert sp.train_cutoff_ts == train_max
    assert sp.val_cutoff_ts == val_max


def test_chronological_split_requires_sorted_df() -> None:
    df = _synthetic_df(n=100)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError):
        chronological_edge_split(shuffled)


# ---------------------------------------------------------------------------
# Message-passing partition: surface-level sanity
# ---------------------------------------------------------------------------


def test_build_mp_split_train_mp_and_sup_are_disjoint() -> None:
    """The train half is the one place we partition into mp / sup ourselves."""
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)
    out = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)

    train = out["train"]
    mp_idx = set(int(i) for i in train["mp_idx"].tolist())
    sup_idx = set(int(i) for i in train["sup_idx"].tolist())
    assert mp_idx & sup_idx == set()
    # Exact sizes: floor(350 * 0.2) = 70 sup, 280 mp.
    assert len(sup_idx) == 70
    assert len(mp_idx) == 280


def test_build_mp_split_val_mp_is_all_train() -> None:
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)
    out = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)
    val_mp_idx = set(int(i) for i in out["val"]["mp_idx"].tolist())
    assert val_mp_idx == set(sp.train_idx.tolist())


def test_build_mp_split_test_mp_is_train_plus_val() -> None:
    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)
    out = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)
    test_mp_idx = set(int(i) for i in out["test"]["mp_idx"].tolist())
    assert test_mp_idx == set(sp.train_idx.tolist()) | set(sp.val_idx.tolist())


def test_build_mp_split_tensor_dtypes() -> None:
    """Edge index / time / label int64, edge_attr float32."""
    import torch

    df = _synthetic_df(n=200)
    sp = chronological_edge_split(df)
    out = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)
    for split_name, t in out.items():
        assert t["mp_edge_index"].dtype == torch.int64, split_name
        assert t["mp_edge_time"].dtype == torch.int64, split_name
        assert t["mp_edge_attr"].dtype == torch.float32, split_name
        assert t["sup_edge_index"].dtype == torch.int64, split_name
        assert t["sup_edge_label"].dtype == torch.int64, split_name
        assert t["sup_edge_time"].dtype == torch.int64, split_name


# ---------------------------------------------------------------------------
# Stratified random split (ablation only)
# ---------------------------------------------------------------------------


def test_stratified_random_split_preserves_class_balance() -> None:
    df = _synthetic_df(n=500)
    sp = stratified_random_edge_split(df)
    assert isinstance(sp, SplitResult)
    # Class shares per split should be within 5pp of the global share.
    global_pos = (df["label_binary"] == 1).mean()
    for idx in (sp.train_idx, sp.val_idx, sp.test_idx):
        share = (df["label_binary"].iloc[idx] == 1).mean()
        assert abs(share - global_pos) < 0.05


def test_partition_seed_is_independent_of_training_seed() -> None:
    """The MP/sup partition must depend only on ``partition_seed`` — not on the
    model's ``training.seed``. Otherwise a multi-seed retrain ends up training
    on structurally different graphs and conflates init-noise with split-noise.

    Concretely: the public partition entrypoint is
    :func:`reddit_gnn.data.splits.build_message_passing_split`, whose ``seed``
    argument is what ``scripts/run_experiment.py`` now wires to
    ``cfg["data"]["partition_seed"]``. The training-side seed (model init,
    dropout) is the global seed set by :func:`reddit_gnn.seed.set_global_seed`.
    Calling the partition with a fixed ``seed`` must always emit the same
    ``mp_idx`` / ``sup_idx`` tensors, regardless of what the global RNG state
    looks like at call time.
    """
    import torch

    from reddit_gnn.seed import set_global_seed

    df = _synthetic_df(n=500)
    sp = chronological_edge_split(df)

    # Same partition_seed under two wildly different training seeds (set as
    # the *global* RNG state to mimic what the script does).
    set_global_seed(0)
    splits_a = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=42)
    set_global_seed(2026)
    splits_b = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=42)

    for split_name in ("train", "val", "test"):
        for key in ("mp_idx", "sup_idx", "mp_edge_index", "sup_edge_index"):
            a = splits_a[split_name][key]
            b = splits_b[split_name][key]
            assert torch.equal(a, b), (
                f"{split_name}.{key} differs even though partition_seed is fixed: "
                f"the partition leaked the global RNG state"
            )

    # Cross-check: changing the partition_seed *does* produce a different
    # train partition (otherwise the test would pass trivially).
    splits_c = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=1234)
    assert not torch.equal(splits_a["train"]["sup_idx"], splits_c["train"]["sup_idx"]), (
        "different partition_seed values should produce different train_sup_idx"
    )
