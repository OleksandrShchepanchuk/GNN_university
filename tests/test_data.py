"""Smoke tests for the data subpackage. Real assertions arrive with the pipeline."""

from __future__ import annotations

import pytest


def test_imports() -> None:
    """Every data module must import cleanly even without raw files present."""
    from reddit_gnn.data import (  # noqa: F401
        download,
        features,
        load,
        preprocess,
        pyg_dataset,
        splits,
    )


def test_label_remap_constants_present() -> None:
    """The label-remap function must be a stable public symbol."""
    from reddit_gnn.data.preprocess import remap_post_label

    # Contract: -1 -> 0 (negative), +1 -> 1 (neutral/positive).
    assert remap_post_label(-1) == 0
    assert remap_post_label(1) == 1
    with pytest.raises(ValueError):
        remap_post_label(0)
