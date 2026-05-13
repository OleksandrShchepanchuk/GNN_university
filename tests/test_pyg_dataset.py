"""Tests for ``reddit_gnn.data.pyg_dataset``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from reddit_gnn.data.pyg_dataset import (
    build_pyg_data_per_split,
    load_pyg_data,
    save_pyg_data,
)
from reddit_gnn.data.splits import (
    build_message_passing_split,
    chronological_edge_split,
)

POST_PROPERTIES_DIM = 86


def _make_df(n: int = 200, *, num_nodes: int = 10, seed: int = 0) -> pd.DataFrame:
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


def test_imports() -> None:
    from reddit_gnn.data import pyg_dataset  # noqa: F401


def test_build_pyg_data_per_split_returns_three_data_objects() -> None:
    df = _make_df(n=200, num_nodes=10)
    sp = chronological_edge_split(df)
    splits = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)

    node_features = torch.randn(10, 16, dtype=torch.float32)
    edge_features = torch.randn(len(df), 93, dtype=torch.float32)

    out = build_pyg_data_per_split(df, node_features, edge_features, splits)
    assert set(out.keys()) == {"train", "val", "test"}

    for name, data in out.items():
        # Shape checks
        assert data.x.shape == (10, 16), name
        assert data.edge_index.shape[0] == 2, name
        assert data.edge_attr.shape[1] == POST_PROPERTIES_DIM, name
        assert data.edge_label_attr.shape[1] == 93, name
        assert data.edge_label.shape[0] == data.edge_label_index.shape[1], name

        # Dtype checks
        assert data.edge_index.dtype == torch.int64, name
        assert data.edge_label_index.dtype == torch.int64, name
        assert data.edge_label.dtype == torch.int64, name
        assert data.edge_attr.dtype == torch.float32, name
        assert data.edge_label_attr.dtype == torch.float32, name
        assert data.x.dtype == torch.float32, name

        # Value checks
        assert set(data.edge_label.unique().tolist()) <= {0, 1}, name
        assert int(data.edge_index.max()) < data.num_nodes, name
        assert int(data.edge_label_index.max()) < data.num_nodes, name


def test_build_pyg_data_per_split_validates_oob_indices() -> None:
    df = _make_df(n=50, num_nodes=10)
    sp = chronological_edge_split(df)
    splits = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)
    node_features = torch.randn(5, 16, dtype=torch.float32)  # too small on purpose
    edge_features = torch.randn(len(df), 93, dtype=torch.float32)
    with pytest.raises(ValueError):
        build_pyg_data_per_split(df, node_features, edge_features, splits)


def test_save_load_round_trip(tmp_path) -> None:
    df = _make_df(n=80, num_nodes=8)
    sp = chronological_edge_split(df)
    splits = build_message_passing_split(df, sp, disjoint_train_ratio=0.2, seed=0)
    node_features = torch.randn(8, 16, dtype=torch.float32)
    edge_features = torch.randn(len(df), 93, dtype=torch.float32)
    out = build_pyg_data_per_split(df, node_features, edge_features, splits)

    path = tmp_path / "train.pt"
    save_pyg_data(out["train"], path)
    reloaded = load_pyg_data(path)
    assert torch.allclose(reloaded.x, out["train"].x)
    assert torch.equal(reloaded.edge_index, out["train"].edge_index)
    assert torch.equal(reloaded.edge_label, out["train"].edge_label)
