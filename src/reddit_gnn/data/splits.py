"""Train / val / test splits over **edges only**.

We always split edges, never nodes — edge sign classification needs every edge
to be a real, observed hyperlink, and the message-passing graph used by the
encoders is built from **train edges only** so val/test labels never leak
through aggregation.

Strategies:

* ``temporal`` — sort edges by ``TIMESTAMP`` and cut at frac quantiles
  (the realistic setting; new sentiment must be predicted from past structure);
* ``random_stratified`` — shuffle-split that preserves class balance per fold
  (useful for ablations / sanity checks).

Both return boolean masks of length ``num_edges``.
"""

from __future__ import annotations

from typing import Any


def temporal_edge_split(
    edge_df: Any,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[Any, Any, Any]:
    """Return ``(train_mask, val_mask, test_mask)`` ordered by TIMESTAMP."""
    raise NotImplementedError("data.splits.temporal_edge_split is not implemented yet")


def stratified_edge_split(
    edge_df: Any,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Class-stratified random split over edges."""
    raise NotImplementedError("data.splits.stratified_edge_split is not implemented yet")


__all__ = ["temporal_edge_split", "stratified_edge_split"]
