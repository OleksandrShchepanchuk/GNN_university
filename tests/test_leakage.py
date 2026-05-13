"""Tests guarding against the two leakage modes this project is designed to avoid:

1. **Label leakage via message passing** — when scoring val/test edges, the
   encoder's ``edge_index`` must be restricted to TRAIN edges. No val/test
   edge labels (or even val/test edges themselves) may participate in
   aggregation that produces the embeddings used to score them.

2. **Feature leakage** — engineered edge features (degrees, common neighbors,
   etc.) and feature scalers must be computed on the TRAIN edge set only, then
   applied to val/test. Computing them on the full graph contaminates val/test.

These tests assert the *contract*; concrete checks attach once the data
pipeline lands.
"""

from __future__ import annotations

import pytest


def test_imports() -> None:
    from reddit_gnn.data import features, pyg_dataset, splits  # noqa: F401
    from reddit_gnn.training import loaders  # noqa: F401


@pytest.mark.xfail(reason="data pipeline not implemented yet", strict=False)
def test_message_passing_uses_train_edges_only() -> None:
    """The encoder must never see val/test edges during forward()."""
    raise AssertionError("Wire this once pyg_dataset + loaders are implemented")


@pytest.mark.xfail(reason="feature pipeline not implemented yet", strict=False)
def test_engineered_features_fit_on_train_only() -> None:
    """StandardScaler / degree counts must be fit on train rows only."""
    raise AssertionError("Wire this once features.build_edge_features is implemented")
