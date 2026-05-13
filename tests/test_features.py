"""Smoke tests for the features module."""

from __future__ import annotations

import pytest


def test_imports() -> None:
    from reddit_gnn.data import features  # noqa: F401


@pytest.mark.xfail(reason="feature builders not implemented yet", strict=False)
def test_parse_post_properties_round_trip() -> None:
    from reddit_gnn.data.features import parse_post_properties

    vec = parse_post_properties(",".join(["0.5"] * 86))
    assert len(vec) == 86
