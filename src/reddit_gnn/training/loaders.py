"""DataLoader builders.

Two regimes:
    * **Full-batch** — the entire graph fits in memory; one forward per epoch.
    * **Neighbor sampling** — :class:`torch_geometric.loader.NeighborLoader`
      for SAGE / GAT on larger settings; configured via ``sage.yaml``.

In both regimes the message-passing edges are restricted to ``edge_index[:, train_mask]``;
val/test rows are scored but never contribute aggregation weights.
"""

from __future__ import annotations

from typing import Any


def build_full_batch_loader(data: Any, split: str) -> Any:
    """Wrap ``data`` for a full-batch pass over the chosen split."""
    raise NotImplementedError("training.loaders.build_full_batch_loader is not implemented yet")


def build_neighbor_loader(data: Any, split: str, cfg: dict[str, Any]) -> Any:
    """Build a :class:`NeighborLoader` for mini-batched training."""
    raise NotImplementedError("training.loaders.build_neighbor_loader is not implemented yet")


__all__ = ["build_full_batch_loader", "build_neighbor_loader"]
