"""Smoke tests for metric helpers."""

from __future__ import annotations

import pytest


def test_imports() -> None:
    from reddit_gnn.training import evaluate, losses  # noqa: F401


@pytest.mark.xfail(reason="metric functions not implemented yet", strict=False)
def test_compute_metrics_returns_dict() -> None:
    from reddit_gnn.training.evaluate import compute_metrics

    out = compute_metrics([0, 1], [0, 1])
    assert isinstance(out, dict)
