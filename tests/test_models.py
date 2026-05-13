"""Tests for ``reddit_gnn.models``.

We exercise every encoder + the decoder + the full ``EdgeClassifier`` on a
tiny synthetic graph (10 nodes, 30 edges, 8 supervision edges). The signed
encoder additionally has a graceful-fallback test for an empty negative
half-graph — explicitly required by the project spec.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from reddit_gnn.models.baselines import (
    LogisticRegressionBaseline,
    LogisticRegressionWithNodeFeats,
    MajorityClassifier,
    MLPEdgeBaseline,
)
from reddit_gnn.models.decoders import EdgeMLPDecoder
from reddit_gnn.models.edge_classifier import EdgeClassifier, parameter_count
from reddit_gnn.models.encoders import (
    GATEncoder,
    GCNEncoder,
    SAGEEncoder,
    SignedGCNEncoder,
)

# Module-wide synthetic graph parameters.
NUM_NODES = 10
NUM_EDGES = 30
NUM_SUP = 8
IN_CHANNELS = 16
HIDDEN = 32
OUT_CHANNELS = 24
EDGE_FEATURE_DIM = 5


def _synthetic_graph(seed: int = 0):
    """Reproducible synthetic graph for encoder/decoder tests."""
    torch.manual_seed(seed)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    edge_label_index = torch.randint(0, NUM_NODES, (2, NUM_SUP))
    edge_attr_for_label = torch.randn(NUM_SUP, EDGE_FEATURE_DIM)
    return x, edge_index, edge_label_index, edge_attr_for_label


# ---------------------------------------------------------------------------
# Encoder forward shapes
# ---------------------------------------------------------------------------


def test_gcn_encoder_forward_shape():
    x, edge_index, _, _ = _synthetic_graph()
    enc = GCNEncoder(
        IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, dropout=0.1, use_batchnorm=True
    )
    enc.eval()
    z = enc(x, edge_index)
    assert z.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(z).all()


def test_sage_encoder_forward_shape_with_each_aggregator():
    x, edge_index, _, _ = _synthetic_graph()
    # LSTM aggregation requires edge_index sorted by destination; sort up-front
    # so the same edge set works for every aggregator.
    order = edge_index[1].argsort()
    sorted_ei = edge_index[:, order]
    for aggr in ("mean", "max", "sum", "lstm"):
        enc = SAGEEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, dropout=0.0, aggr=aggr)
        enc.eval()
        z = enc(x, sorted_ei)
        assert z.shape == (NUM_NODES, OUT_CHANNELS), aggr
        assert torch.isfinite(z).all(), aggr


def test_sage_encoder_rejects_bad_aggr():
    with pytest.raises(ValueError):
        SAGEEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, aggr="bogus")


def test_gat_encoder_forward_shape_with_per_layer_concat():
    x, edge_index, _, _ = _synthetic_graph()
    # Mixed concat-per-layer: last layer averages heads instead of concatenating.
    enc = GATEncoder(
        IN_CHANNELS,
        HIDDEN,
        OUT_CHANNELS,
        num_layers=2,
        heads=4,
        concat=[True, False],
        dropout=0.0,
        attn_dropout=0.0,
    )
    enc.eval()
    z = enc(x, edge_index)
    assert z.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(z).all()


def test_signed_encoder_forward_shape():
    x, edge_index, _, _ = _synthetic_graph()
    pos = edge_index[:, : 2 * NUM_EDGES // 3]
    neg = edge_index[:, 2 * NUM_EDGES // 3 :]
    enc = SignedGCNEncoder(IN_CHANNELS, hidden_channels=HIDDEN, num_layers=2)
    enc.eval()
    z = enc(x, pos, neg)
    # PyG's SignedGCN: conv1 emits `2 * (hidden // 2) = hidden_channels` and
    # subsequent layers preserve that width, so the final embedding width is
    # exactly hidden_channels (not 2 * hidden_channels — a common misread).
    assert z.shape == (NUM_NODES, HIDDEN)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# Decoder + EdgeClassifier end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder_factory",
    [
        lambda: GCNEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, dropout=0.0),
        lambda: SAGEEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, dropout=0.0),
        lambda: GATEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, heads=4, dropout=0.0),
    ],
    ids=["gcn", "sage", "gat"],
)
def test_edge_classifier_produces_S_logits(encoder_factory):
    x, edge_index, edge_label_index, edge_attr = _synthetic_graph()
    encoder = encoder_factory()
    decoder = EdgeMLPDecoder(
        node_dim=OUT_CHANNELS, edge_feature_dim=EDGE_FEATURE_DIM, hidden=32, dropout=0.0
    )
    model = EdgeClassifier(encoder, decoder).eval()
    logits = model(x, edge_index, edge_label_index, edge_attr)
    assert logits.shape == (NUM_SUP,)
    assert torch.isfinite(logits).all()
    assert parameter_count(model) > 0


def test_edge_classifier_supports_no_edge_attr():
    x, edge_index, edge_label_index, _ = _synthetic_graph()
    encoder = GCNEncoder(IN_CHANNELS, HIDDEN, OUT_CHANNELS, num_layers=2, dropout=0.0)
    decoder = EdgeMLPDecoder(node_dim=OUT_CHANNELS, edge_feature_dim=0, hidden=16, dropout=0.0)
    model = EdgeClassifier(encoder, decoder).eval()
    logits = model(x, edge_index, edge_label_index, edge_attr_for_label=None)
    assert logits.shape == (NUM_SUP,)


def test_edge_classifier_signed_forward_produces_S_logits():
    x, edge_index, edge_label_index, edge_attr = _synthetic_graph()
    pos = edge_index[:, :20]
    neg = edge_index[:, 20:]
    encoder = SignedGCNEncoder(IN_CHANNELS, hidden_channels=HIDDEN, num_layers=2)
    decoder = EdgeMLPDecoder(
        node_dim=HIDDEN, edge_feature_dim=EDGE_FEATURE_DIM, hidden=32, dropout=0.0
    )
    model = EdgeClassifier(encoder, decoder).eval()
    logits = model.forward_signed(x, pos, neg, edge_label_index, edge_attr)
    assert logits.shape == (NUM_SUP,)
    assert torch.isfinite(logits).all()


def test_signed_encoder_warns_on_empty_neg_and_does_not_crash(caplog):
    """Spec: forward must not crash when all train edges happen to be positive."""
    x, edge_index, edge_label_index, edge_attr = _synthetic_graph()
    pos = edge_index
    empty_neg = torch.zeros((2, 0), dtype=torch.long)
    encoder = SignedGCNEncoder(IN_CHANNELS, hidden_channels=HIDDEN, num_layers=2)
    decoder = EdgeMLPDecoder(
        node_dim=HIDDEN, edge_feature_dim=EDGE_FEATURE_DIM, hidden=32, dropout=0.0
    )
    model = EdgeClassifier(encoder, decoder).eval()
    with caplog.at_level(logging.WARNING, logger="reddit_gnn.models.encoders"):
        logits = model.forward_signed(x, pos, empty_neg, edge_label_index, edge_attr)
    assert logits.shape == (NUM_SUP,)
    assert torch.isfinite(logits).all()
    assert any("negative" in r.getMessage().lower() for r in caplog.records), (
        "Expected a 'no negative edges' warning"
    )


def test_signed_encoder_warns_on_empty_pos_and_does_not_crash(caplog):
    x, edge_index, edge_label_index, edge_attr = _synthetic_graph()
    empty_pos = torch.zeros((2, 0), dtype=torch.long)
    neg = edge_index
    encoder = SignedGCNEncoder(IN_CHANNELS, hidden_channels=HIDDEN, num_layers=2)
    decoder = EdgeMLPDecoder(
        node_dim=HIDDEN, edge_feature_dim=EDGE_FEATURE_DIM, hidden=32, dropout=0.0
    )
    model = EdgeClassifier(encoder, decoder).eval()
    with caplog.at_level(logging.WARNING, logger="reddit_gnn.models.encoders"):
        logits = model.forward_signed(x, empty_pos, neg, edge_label_index, edge_attr)
    assert logits.shape == (NUM_SUP,)
    assert any("positive" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Decoder contract: edge_attr_for_label is required when edge_feature_dim > 0
# ---------------------------------------------------------------------------


def test_edge_mlp_decoder_rejects_missing_edge_attr_when_required():
    decoder = EdgeMLPDecoder(node_dim=OUT_CHANNELS, edge_feature_dim=EDGE_FEATURE_DIM, hidden=16)
    z = torch.randn(NUM_SUP, OUT_CHANNELS)
    with pytest.raises(ValueError):
        decoder(z, z, None)


def test_edge_mlp_decoder_rejects_wrong_edge_attr_dim():
    decoder = EdgeMLPDecoder(node_dim=OUT_CHANNELS, edge_feature_dim=EDGE_FEATURE_DIM, hidden=16)
    z = torch.randn(NUM_SUP, OUT_CHANNELS)
    bad_attr = torch.randn(NUM_SUP, EDGE_FEATURE_DIM + 1)
    with pytest.raises(ValueError):
        decoder(z, z, bad_attr)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_majority_classifier_predicts_train_majority():
    y = np.array([1, 1, 0, 1, 1, 0, 1])  # majority is 1
    mc = MajorityClassifier().fit(y)
    assert mc.majority_ == 1
    preds = mc.predict(np.zeros((5, 3)))
    assert preds.shape == (5,)
    assert (preds == 1).all()
    proba = mc.predict_proba(np.zeros((5, 3)))
    assert proba.shape == (5, 2)
    # All rows identical (the empirical class shares).
    np.testing.assert_allclose(proba[0], proba[-1])


def test_logreg_baseline_on_edge_features():
    rng = np.random.default_rng(0)
    n, d = 60, 7
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    lr = LogisticRegressionBaseline().fit(X, y)
    preds = lr.predict(X)
    assert preds.shape == (n,)
    # The signal is on X[:,0] > 0; the baseline should clearly beat random.
    assert (preds == y).mean() > 0.7


def test_logreg_with_node_feats_uses_concat():
    rng = np.random.default_rng(1)
    n_nodes, n_edges, d_edge, d_node = 8, 30, 5, 4
    x = rng.standard_normal((n_nodes, d_node)).astype(np.float32)
    edge_index = rng.integers(0, n_nodes, size=(2, n_edges))
    edge_feats = rng.standard_normal((n_edges, d_edge)).astype(np.float32)
    y = rng.integers(0, 2, size=n_edges)
    lrn = LogisticRegressionWithNodeFeats().fit(edge_feats, x, edge_index, y)
    preds = lrn.predict(edge_feats, x, edge_index)
    assert preds.shape == (n_edges,)
    # The underlying sklearn model has coef_ over d_edge + 4*d_node columns.
    assert lrn.model.coef_.shape[1] == d_edge + 4 * d_node


def test_mlp_edge_baseline_forward_shape():
    rng = np.random.default_rng(2)
    n_nodes, n_edges, d_edge, d_node = 8, 16, 5, 4
    x = torch.from_numpy(rng.standard_normal((n_nodes, d_node)).astype(np.float32))
    edge_feats = torch.from_numpy(rng.standard_normal((n_edges, d_edge)).astype(np.float32))
    edge_label_index = torch.from_numpy(
        rng.integers(0, n_nodes, size=(2, n_edges)).astype(np.int64)
    )
    model = MLPEdgeBaseline(
        edge_feature_dim=d_edge, node_feature_dim=d_node, hidden=16, dropout=0.0
    )
    # The MLP baseline shares the EdgeClassifier call signature so the training
    # loop can dispatch uniformly; the ``edge_index`` argument is ignored.
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
    logits = model(x, dummy_edge_index, edge_label_index, edge_feats)
    assert logits.shape == (n_edges,)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Public symbol surface
# ---------------------------------------------------------------------------


def test_imports() -> None:
    from reddit_gnn.models import baselines, decoders, edge_classifier, encoders  # noqa: F401


def test_public_classes_exist() -> None:
    classes = (
        MajorityClassifier,
        LogisticRegressionBaseline,
        LogisticRegressionWithNodeFeats,
        MLPEdgeBaseline,
        EdgeMLPDecoder,
        EdgeClassifier,
        GCNEncoder,
        SAGEEncoder,
        GATEncoder,
        SignedGCNEncoder,
    )
    for cls in classes:
        assert isinstance(cls, type)
