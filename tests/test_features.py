"""Tests for ``reddit_gnn.data.features``.

The most important invariant guarded here is *no leakage*: scalers and
temporal bounds inside :class:`FeatureBuilder` must be fit on training rows
only. We assert this by computing the expected statistics manually on the
training frame and comparing them to ``scaler.mean_`` / ``scaler.scale_``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reddit_gnn.data.features import (
    POST_PROPERTIES_DIM,
    SNAP_EMBED_DIM,
    STRUCTURAL_COLUMNS,
    FeatureBuilder,
    build_edge_features,
    create_aggregated_edge_property_node_features,
    create_structural_node_features,
    load_snap_subreddit_embeddings,
    parse_post_properties,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_processed_df(
    src_ids: list[int],
    tgt_ids: list[int],
    labels: list[int],
    timestamps: list[str],
    *,
    is_title: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a processed-parquet-shaped df with random POST_PROPERTIES."""
    n = len(src_ids)
    rng = np.random.default_rng(seed)
    props = {f"p{i}": rng.standard_normal(n).astype(np.float32) for i in range(POST_PROPERTIES_DIM)}
    if is_title is None:
        is_title = [i % 2 for i in range(n)]
    names = [f"sr{i}" for i in range(max(max(src_ids), max(tgt_ids)) + 1)]
    return pd.DataFrame(
        {
            "source_id": np.asarray(src_ids, dtype=np.int64),
            "target_id": np.asarray(tgt_ids, dtype=np.int64),
            "source_subreddit_norm": [names[i] for i in src_ids],
            "target_subreddit_norm": [names[i] for i in tgt_ids],
            "label_binary": np.asarray(labels, dtype=np.int8),
            "is_title": np.asarray(is_title, dtype=np.int8),
            "TIMESTAMP": pd.to_datetime(timestamps),
            **props,
        }
    )


def _write_snap_embeddings(
    path: Path, names: list[str], *, dim: int = SNAP_EMBED_DIM, seed: int = 0
) -> np.ndarray:
    """Write a SNAP-style CSV (no header) and return the embedding matrix."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((len(names), dim)).astype(np.float32)
    lines = [
        ",".join([name, *(f"{v:.6f}" for v in row)])
        for name, row in zip(names, matrix, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return matrix


# ---------------------------------------------------------------------------
# parse_post_properties shape
# ---------------------------------------------------------------------------


def test_parse_post_properties_shape() -> None:
    df = _make_processed_df([0, 1], [1, 0], [1, 0], ["2014-01-01", "2014-02-01"])
    arr = parse_post_properties(df)
    assert arr.shape == (2, POST_PROPERTIES_DIM)
    assert arr.dtype == np.float32


def test_parse_post_properties_missing_columns_raise() -> None:
    df = pd.DataFrame({"p0": [0.1], "p1": [0.2]})  # only 2 of 86
    with pytest.raises(KeyError):
        parse_post_properties(df)


# ---------------------------------------------------------------------------
# Structural feature contract
# ---------------------------------------------------------------------------


def test_structural_features_have_no_nan_or_inf() -> None:
    df = _make_processed_df(
        src_ids=[0, 1, 2, 0, 1],
        tgt_ids=[1, 2, 0, 2, 0],
        labels=[1, 0, 1, 0, 1],
        timestamps=["2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01", "2014-05-01"],
    )
    struct = create_structural_node_features(df, num_nodes=4)  # node 3 is isolated
    assert list(struct.columns) == list(STRUCTURAL_COLUMNS)
    arr = struct.to_numpy()
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
    # Isolated node has zero in every column.
    assert (struct.iloc[3] == 0).all()


# ---------------------------------------------------------------------------
# Leakage guard: FeatureBuilder.fit must see train rows only
# ---------------------------------------------------------------------------


def test_featurebuilder_fit_uses_only_train_rows() -> None:
    train_df = _make_processed_df(
        src_ids=[0, 1, 2, 0, 1],
        tgt_ids=[1, 2, 0, 2, 0],
        labels=[1, 0, 1, 0, 1],
        timestamps=["2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01", "2014-05-01"],
        seed=1,
    )
    # Val rows have a deliberately *different* structural distribution: a
    # superhub node 4 with many incoming edges. If fit() leaked, scaler.mean_
    # would shift toward the val statistics.
    val_df = _make_processed_df(
        src_ids=[0, 1, 2, 3] * 5,
        tgt_ids=[4] * 20,
        labels=[1, 0, 1, 0] * 5,
        timestamps=["2014-06-01"] * 20,
        seed=2,
    )

    fb = FeatureBuilder(use_snap_embeddings=False)
    fb.fit(train_df, num_nodes=5)

    expected = create_structural_node_features(train_df, num_nodes=5)
    expected_arr = expected[list(STRUCTURAL_COLUMNS)].to_numpy(dtype=np.float64)
    expected_mean = expected_arr.mean(axis=0)
    expected_var = expected_arr.var(axis=0, ddof=0)
    expected_scale = np.sqrt(np.where(expected_var > 0, expected_var, 1.0))

    np.testing.assert_allclose(fb.structural_scaler.mean_, expected_mean, atol=1e-7)
    np.testing.assert_allclose(fb.structural_scaler.scale_, expected_scale, atol=1e-7)

    # And — crucial — applying the *fit-on-train* scaler to (train + val)
    # rows yields different summary stats than re-fitting on the union would.
    union = pd.concat([train_df, val_df], ignore_index=True)
    union_struct = create_structural_node_features(union, num_nodes=5)
    union_mean = union_struct[list(STRUCTURAL_COLUMNS)].to_numpy().mean(axis=0)
    # Sanity: union mean differs from train mean -> our scaler is locked to train.
    assert not np.allclose(union_mean, expected_mean)

    # The temporal bounds must also be the train bounds, not the union bounds.
    assert fb.time_bounds[0] == pd.to_datetime("2014-01-01")
    assert fb.time_bounds[1] == pd.to_datetime("2014-05-01")
    assert fb.year_bounds == (2014, 2014)


# ---------------------------------------------------------------------------
# Output finite-ness
# ---------------------------------------------------------------------------


def test_featurebuilder_outputs_are_finite(tmp_path: Path) -> None:
    df = _make_processed_df(
        src_ids=[0, 1, 2, 0, 1, 2, 3, 4],
        tgt_ids=[1, 2, 0, 2, 3, 4, 0, 1],
        labels=[1, 0, 1, 1, 0, 1, 0, 1],
        timestamps=[
            "2014-01-01",
            "2014-02-01",
            "2014-03-01",
            "2014-04-01",
            "2014-05-01",
            "2014-06-01",
            "2014-07-01",
            "2014-08-01",
        ],
        seed=3,
    )

    snap_path = tmp_path / "emb.csv"
    _write_snap_embeddings(snap_path, ["sr0", "sr1", "sr2"])  # nodes 3, 4 missing

    fb = FeatureBuilder()
    fb.fit(df, num_nodes=5, embeddings_path=snap_path)

    node_feats = fb.transform_node_features(df, num_nodes=5)
    edge_feats = fb.transform_edge_features(df)

    assert isinstance(node_feats, torch.Tensor)
    assert isinstance(edge_feats, torch.Tensor)
    assert node_feats.dtype == torch.float32
    assert edge_feats.dtype == torch.float32

    expected_node_dim = SNAP_EMBED_DIM + 1 + len(STRUCTURAL_COLUMNS) + 2 * POST_PROPERTIES_DIM
    assert node_feats.shape == (5, expected_node_dim)
    assert edge_feats.shape == (len(df), POST_PROPERTIES_DIM + 7)

    assert torch.isfinite(node_feats).all().item()
    assert torch.isfinite(edge_feats).all().item()


# ---------------------------------------------------------------------------
# SNAP embedding fallback for unknown subreddits
# ---------------------------------------------------------------------------


def test_missing_subreddits_get_zeros_and_unknown_flag(tmp_path: Path) -> None:
    snap_path = tmp_path / "emb.csv"
    # The SNAP file only knows sr0 and sr1; sr2 is "unknown".
    known_matrix = _write_snap_embeddings(snap_path, ["sr0", "sr1"], seed=7)

    node_to_id = {"sr0": 0, "sr1": 1, "sr2": 2}
    X_emb, has = load_snap_subreddit_embeddings(node_to_id, snap_path)

    assert X_emb.shape == (3, SNAP_EMBED_DIM)
    assert has.shape == (3, 1)

    # The two known rows match the SNAP matrix (lookup is by lowercased name).
    np.testing.assert_allclose(X_emb[0], known_matrix[0], atol=1e-5)
    np.testing.assert_allclose(X_emb[1], known_matrix[1], atol=1e-5)
    assert has[0, 0] == 1.0
    assert has[1, 0] == 1.0

    # The unknown row is zeros and flagged.
    assert np.all(X_emb[2] == 0.0)
    assert has[2, 0] == 0.0

    # Now run it through FeatureBuilder and verify the unknown_flag column
    # (block index = SNAP_EMBED_DIM) is 1 for node 2 and 0 for nodes 0/1.
    df = _make_processed_df(
        src_ids=[0, 1, 2, 0, 1],
        tgt_ids=[1, 2, 0, 2, 0],
        labels=[1, 0, 1, 0, 1],
        timestamps=["2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01", "2014-05-01"],
    )
    fb = FeatureBuilder()
    fb.fit(df, num_nodes=3, embeddings_path=snap_path)
    node_feats = fb.transform_node_features(df, num_nodes=3).numpy()

    unknown_col = node_feats[:, SNAP_EMBED_DIM]
    assert unknown_col[0] == 0.0
    assert unknown_col[1] == 0.0
    assert unknown_col[2] == 1.0

    # And the SNAP embedding block for the unknown node is exactly zero.
    np.testing.assert_array_equal(node_feats[2, :SNAP_EMBED_DIM], np.zeros(SNAP_EMBED_DIM))


# ---------------------------------------------------------------------------
# Aggregated edge property contract
# ---------------------------------------------------------------------------


def test_aggregated_edge_property_per_node_means() -> None:
    df = _make_processed_df(
        src_ids=[0, 0, 1],
        tgt_ids=[1, 1, 0],
        labels=[1, 0, 1],
        timestamps=["2014-01-01", "2014-02-01", "2014-03-01"],
    )
    edge_attr = parse_post_properties(df)
    agg = create_aggregated_edge_property_node_features(df, edge_attr, num_nodes=2)

    # Node 0 outgoing: rows 0 and 1 -> mean of edge_attr[[0, 1]] along axis=0
    expected_out_0 = edge_attr[[0, 1]].mean(axis=0)
    np.testing.assert_allclose(agg[0, POST_PROPERTIES_DIM:], expected_out_0, atol=1e-6)

    # Node 1 incoming: rows 0 and 1
    expected_in_1 = edge_attr[[0, 1]].mean(axis=0)
    np.testing.assert_allclose(agg[1, :POST_PROPERTIES_DIM], expected_in_1, atol=1e-6)


# ---------------------------------------------------------------------------
# build_edge_features standalone
# ---------------------------------------------------------------------------


def test_build_edge_features_shape_and_dtype() -> None:
    df = _make_processed_df(
        src_ids=[0, 1],
        tgt_ids=[1, 0],
        labels=[1, 0],
        timestamps=["2014-01-01", "2014-06-01"],
    )
    out = build_edge_features(df)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, POST_PROPERTIES_DIM + 7)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_featurebuilder_save_load_round_trip(tmp_path: Path) -> None:
    df = _make_processed_df(
        src_ids=[0, 1, 2, 0],
        tgt_ids=[1, 2, 0, 2],
        labels=[1, 0, 1, 0],
        timestamps=["2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01"],
    )
    fb = FeatureBuilder(use_snap_embeddings=False)
    fb.fit(df, num_nodes=3)

    out_path = tmp_path / "featurebuilder.pkl"
    fb.save(out_path)
    assert out_path.exists()

    fb2 = FeatureBuilder.load(out_path)
    np.testing.assert_allclose(fb2.structural_scaler.mean_, fb.structural_scaler.mean_)
    np.testing.assert_allclose(fb2.structural_scaler.scale_, fb.structural_scaler.scale_)
    assert fb2.time_bounds == fb.time_bounds
    assert fb2.year_bounds == fb.year_bounds

    feats_a = fb.transform_node_features(df, num_nodes=3).numpy()
    feats_b = fb2.transform_node_features(df, num_nodes=3).numpy()
    np.testing.assert_allclose(feats_a, feats_b, atol=1e-7)
