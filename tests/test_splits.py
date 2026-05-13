"""Smoke tests for edge split helpers."""

from __future__ import annotations

import pytest


def test_imports() -> None:
    from reddit_gnn.data import splits  # noqa: F401


@pytest.mark.xfail(reason="split routines not implemented yet", strict=False)
def test_temporal_split_fractions_sum_to_one() -> None:
    from reddit_gnn.data.splits import temporal_edge_split

    masks = temporal_edge_split(edge_df=None, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    assert masks is not None
