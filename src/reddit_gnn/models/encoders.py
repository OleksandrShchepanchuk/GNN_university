"""GNN encoders: GCN, GraphSAGE, GAT, SignedGCN.

Each encoder maps ``(x, edge_index)`` to ``(num_nodes, out_channels)`` node
embeddings. The ``edge_index`` passed at training time is built from **train
edges only** so that val/test labels never flow through message passing.
"""

from __future__ import annotations

from typing import Any


class GCNEncoder:
    """Stack of ``GCNConv`` layers with ReLU + dropout."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float) -> None:
        raise NotImplementedError("GCNEncoder.__init__ is not implemented yet")


class SAGEEncoder:
    """Stack of ``SAGEConv`` layers (mean aggregator by default)."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float) -> None:
        raise NotImplementedError("SAGEEncoder.__init__ is not implemented yet")


class GATEncoder:
    """Stack of ``GATv2Conv`` layers with multi-head attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
        attn_dropout: float = 0.2,
    ) -> None:
        raise NotImplementedError("GATEncoder.__init__ is not implemented yet")


class SignedGCNEncoder:
    """Thin wrapper around :class:`torch_geometric.nn.SignedGCN`.

    The signed encoder partitions training edges into positive/negative sets
    according to the **trained labels** (0 -> negative, 1 -> positive) and
    learns separate aggregations along each. Critically, only train labels
    feed the encoder — val/test labels are inputs to the loss, never to
    message passing.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, lamb: float = 5.0) -> None:
        raise NotImplementedError("SignedGCNEncoder.__init__ is not implemented yet")


__all__ = ["GCNEncoder", "SAGEEncoder", "GATEncoder", "SignedGCNEncoder"]
