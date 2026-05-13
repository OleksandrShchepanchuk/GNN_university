"""Smoke tests for model module imports."""

from __future__ import annotations


def test_imports() -> None:
    from reddit_gnn.models import baselines, decoders, edge_classifier, encoders  # noqa: F401


def test_public_classes_exist() -> None:
    from reddit_gnn.models.baselines import LogRegBaseline, MLPBaseline
    from reddit_gnn.models.decoders import MLPEdgeDecoder
    from reddit_gnn.models.edge_classifier import EdgeClassifier
    from reddit_gnn.models.encoders import GATEncoder, GCNEncoder, SAGEEncoder, SignedGCNEncoder

    for cls in (
        LogRegBaseline,
        MLPBaseline,
        MLPEdgeDecoder,
        EdgeClassifier,
        GATEncoder,
        GCNEncoder,
        SAGEEncoder,
        SignedGCNEncoder,
    ):
        assert isinstance(cls, type)
