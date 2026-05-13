"""Smoke tests for the PyG dataset assembler."""

from __future__ import annotations

import pytest


def test_imports() -> None:
    from reddit_gnn.data import pyg_dataset  # noqa: F401


@pytest.mark.xfail(reason="pyg_dataset.build_pyg_data not implemented yet", strict=False)
def test_build_pyg_data_returns_data_object() -> None:
    from reddit_gnn.data.pyg_dataset import build_pyg_data

    build_pyg_data(processed_dir="data/processed", cfg={})
