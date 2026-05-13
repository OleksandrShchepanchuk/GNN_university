"""PyTorch Geometric :class:`~torch_geometric.data.Data` assembly.

Produces a single in-memory ``Data`` object with:

* ``x``           — ``[num_nodes, d_node]`` node features (LIWC + optional structural).
* ``edge_index``  — ``[2, num_edges]`` directed indices for every observed hyperlink.
* ``edge_attr``   — ``[num_edges, d_edge]`` engineered edge features.
* ``y``           — ``[num_edges]`` edge labels in ``{0, 1}`` (remapped POST_LABEL).
* ``train_mask`` / ``val_mask`` / ``test_mask`` — ``[num_edges]`` boolean masks.
* ``edge_time``   — ``[num_edges]`` timestamps (UNIX seconds) for temporal analysis.

The encoders' message-passing graph during training uses only the train edges
(``edge_index[:, train_mask]``); val/test edges are *predicted* over but never
contribute to aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_pyg_data(processed_dir: str | Path, cfg: dict[str, Any]) -> Any:
    """Assemble and return the in-memory ``Data`` object."""
    raise NotImplementedError("data.pyg_dataset.build_pyg_data is not implemented yet")


def save_pyg_data(data: Any, path: str | Path) -> None:
    """Persist a ``Data`` object via :func:`torch.save`."""
    raise NotImplementedError("data.pyg_dataset.save_pyg_data is not implemented yet")


def load_pyg_data(path: str | Path) -> Any:
    """Load a previously saved ``Data`` object."""
    raise NotImplementedError("data.pyg_dataset.load_pyg_data is not implemented yet")


__all__ = ["build_pyg_data", "save_pyg_data", "load_pyg_data"]
