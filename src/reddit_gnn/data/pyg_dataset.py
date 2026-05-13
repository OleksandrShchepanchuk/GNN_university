"""PyTorch Geometric :class:`~torch_geometric.data.Data` assembly per split.

We produce **one Data object per split** (train / val / test). Each Data
carries:

* ``x``                — node features (shared across splits).
* ``edge_index``       — the encoder's message-passing graph for this split.
* ``edge_attr``        — raw POST_PROPERTIES of those message-passing edges.
* ``edge_time``        — timestamps (int64 ns since epoch) of MP edges.
* ``edge_label``       — supervision labels in ``{0, 1}``.
* ``edge_label_index`` — endpoints of supervision edges (``[2, S]``).
* ``edge_label_attr``  — engineered (scaled) edge features for supervision edges.
* ``edge_label_time``  — supervision timestamps.

The split dictionary fed in is what :func:`build_message_passing_split`
returns; the only role of :func:`build_pyg_data_per_split` on top is to
attach the node feature tensor, look up engineered edge features for
supervision rows, and validate dtype / range / finiteness invariants.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def _validate_data(data: Data, *, num_nodes: int) -> None:
    """Hard runtime checks. Raises ValueError on any violation."""
    if data.edge_index.dtype != torch.int64:
        raise ValueError(f"edge_index must be int64; got {data.edge_index.dtype}")
    if data.edge_label_index.dtype != torch.int64:
        raise ValueError(
            f"edge_label_index must be int64; got {data.edge_label_index.dtype}"
        )
    if data.edge_label.dtype != torch.int64:
        raise ValueError(f"edge_label must be int64; got {data.edge_label.dtype}")
    if data.edge_attr.dtype != torch.float32:
        raise ValueError(f"edge_attr must be float32; got {data.edge_attr.dtype}")
    if data.edge_label_attr.dtype != torch.float32:
        raise ValueError(
            f"edge_label_attr must be float32; got {data.edge_label_attr.dtype}"
        )
    if data.x.dtype != torch.float32:
        raise ValueError(f"x must be float32; got {data.x.dtype}")

    if data.edge_index.numel() and int(data.edge_index.max()) >= num_nodes:
        raise ValueError(
            f"edge_index.max()={int(data.edge_index.max())} >= num_nodes={num_nodes}"
        )
    if data.edge_label_index.numel() and int(data.edge_label_index.max()) >= num_nodes:
        raise ValueError(
            f"edge_label_index.max()={int(data.edge_label_index.max())} >= num_nodes={num_nodes}"
        )

    labels = set(int(v) for v in data.edge_label.unique().tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"edge_label must be in {{0, 1}}; saw {labels}")

    for name, t in (
        ("x", data.x),
        ("edge_attr", data.edge_attr),
        ("edge_label_attr", data.edge_label_attr),
    ):
        if t.numel() and not torch.isfinite(t).all().item():
            raise ValueError(f"{name} contains NaN or inf values")


def build_pyg_data_per_split(
    df,
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    split_outputs: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Data]:
    """Assemble a :class:`torch_geometric.data.Data` per split.

    ``edge_features`` is a per-edge tensor (one row per row of ``df``) of the
    engineered + scaled edge features. We index into it with the ``sup_idx``
    tensors stored alongside each split to populate ``edge_label_attr``.
    """
    del df  # kept in the signature for future use; we currently rely only on
            # the precomputed tensors in split_outputs + edge_features.

    if not isinstance(node_features, torch.Tensor):
        raise TypeError("node_features must be a torch.Tensor")
    if node_features.dtype != torch.float32:
        node_features = node_features.to(torch.float32)
    if not isinstance(edge_features, torch.Tensor):
        raise TypeError("edge_features must be a torch.Tensor")
    if edge_features.dtype != torch.float32:
        edge_features = edge_features.to(torch.float32)

    num_nodes = int(node_features.shape[0])
    out: dict[str, Data] = {}
    for split_name, tensors in split_outputs.items():
        try:
            sup_idx = tensors["sup_idx"]
        except KeyError as exc:
            raise KeyError(
                "split_outputs must include 'sup_idx' "
                "(produced by build_message_passing_split)"
            ) from exc

        edge_label_attr = edge_features.index_select(0, sup_idx.to(torch.int64))

        data = Data(
            x=node_features,
            edge_index=tensors["mp_edge_index"].to(torch.int64),
            edge_attr=tensors["mp_edge_attr"].to(torch.float32),
            edge_time=tensors["mp_edge_time"].to(torch.int64),
            edge_label=tensors["sup_edge_label"].to(torch.int64),
            edge_label_index=tensors["sup_edge_index"].to(torch.int64),
            edge_label_attr=edge_label_attr.to(torch.float32),
            edge_label_time=tensors["sup_edge_time"].to(torch.int64),
        )
        data.num_nodes = num_nodes
        _validate_data(data, num_nodes=num_nodes)
        out[split_name] = data

    log.info(
        "build_pyg_data_per_split: built %d Data objects (num_nodes=%d, edge_feature_dim=%d)",
        len(out),
        num_nodes,
        int(edge_features.shape[1]) if edge_features.ndim > 1 else 0,
    )
    return out


def save_pyg_data(data: Data, path: str | Path) -> Path:
    """Persist a :class:`Data` object via :func:`torch.save`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def load_pyg_data(path: str | Path) -> Data:
    """Load a previously saved :class:`Data` object."""
    obj = torch.load(path, weights_only=False)
    if not isinstance(obj, Data):
        raise TypeError(f"{path} did not contain a torch_geometric.data.Data; got {type(obj)}")
    return obj


__all__ = ["build_pyg_data_per_split", "load_pyg_data", "save_pyg_data"]
