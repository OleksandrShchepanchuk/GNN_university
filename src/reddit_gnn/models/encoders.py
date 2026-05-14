"""GNN encoders.

All encoders share the same forward signature ``forward(x, edge_index)`` and
produce ``(num_nodes, out_channels)``. The signed encoder additionally exposes
``forward(x, pos_edge_index, neg_edge_index)`` matching PyG's
:class:`torch_geometric.nn.SignedGCN` convention.

Implementation notes:
    * Each stack has an explicit "input" layer (``in_channels -> hidden``),
      ``num_layers - 1`` hidden layers, and a final linear projection to
      ``out_channels``. Residual connections fire only when the input and
      output shapes of a layer match (so we never silently drop information
      across a dimension change).
    * Optional BatchNorm is applied *before* the activation, dropout *after*.
    * The Signed encoder gracefully handles a sub-graph that has no negative
      edges (warn + supply a dummy self-loop) — this is required for the
      first epoch on small batches and for the "all-positive" failure mode
      explicitly mentioned in the project spec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, SignedGCN

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Small helper for the residual + norm + activation + dropout pattern
# ---------------------------------------------------------------------------


def _apply_block(
    h: torch.Tensor,
    h_in: torch.Tensor,
    norm: nn.Module | None,
    dropout: float,
    training: bool,
    activation: str = "relu",
) -> torch.Tensor:
    if norm is not None:
        h = norm(h)
    if activation == "elu":
        h = F.elu(h)
    else:
        h = F.relu(h)
    h = F.dropout(h, p=dropout, training=training)
    if h.shape == h_in.shape:
        h = h + h_in
    return h


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------


class GCNEncoder(nn.Module):
    """Stack of :class:`GCNConv` layers + final linear projection."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if use_batchnorm else None

        self.convs.append(GCNConv(in_channels, hidden_channels))
        if self.norms is not None:
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if self.norms is not None:
                self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            norm = self.norms[i] if self.norms is not None else None
            h = _apply_block(h, h_in, norm, self.dropout, self.training)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------------


class SAGEEncoder(nn.Module):
    """Stack of :class:`SAGEConv` layers (configurable aggregator)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = False,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if aggr not in {"mean", "max", "sum", "lstm"}:
            raise ValueError(f"aggr must be one of mean/max/sum/lstm; got {aggr!r}")
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if use_batchnorm else None

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        if self.norms is not None:
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            if self.norms is not None:
                self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            norm = self.norms[i] if self.norms is not None else None
            h = _apply_block(h, h_in, norm, self.dropout, self.training)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# GAT (v2)
# ---------------------------------------------------------------------------


class GATEncoder(nn.Module):
    """Stack of :class:`GATv2Conv` layers, multi-head with optional concat-per-layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        concat: bool | list[bool] = True,
        dropout: float = 0.5,
        attn_dropout: float = 0.2,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        self.dropout = dropout
        self.heads = heads

        # Per-layer concat flag — broadcast a bool to all layers.
        if isinstance(concat, bool):
            concat_flags = [concat] * num_layers
        else:
            if len(concat) != num_layers:
                raise ValueError(
                    f"len(concat) must equal num_layers ({num_layers}); got {len(concat)}"
                )
            concat_flags = list(concat)

        # When concat=True, the layer emits ``heads * per_head`` channels;
        # we size ``per_head`` so the post-concat width stays at hidden_channels.
        per_head = max(hidden_channels // heads, 1)
        # Output width of a layer given its concat flag.
        def layer_out(flag: bool) -> int:
            return per_head * heads if flag else per_head

        self.convs = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if use_batchnorm else None

        cur_in = in_channels
        for i in range(num_layers):
            flag = concat_flags[i]
            self.convs.append(
                GATv2Conv(
                    cur_in,
                    per_head,
                    heads=heads,
                    concat=flag,
                    dropout=attn_dropout,
                )
            )
            cur_in = layer_out(flag)
            if self.norms is not None:
                self.norms.append(nn.BatchNorm1d(cur_in))

        self.out_proj = nn.Linear(cur_in, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            norm = self.norms[i] if self.norms is not None else None
            h = _apply_block(h, h_in, norm, self.dropout, self.training, activation="elu")
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# GATv1 (original) — for the static-vs-dynamic-attention comparison vs GATv2.
# ---------------------------------------------------------------------------


class GATv1Encoder(nn.Module):
    """Same architecture as :class:`GATEncoder` but with the original
    :class:`torch_geometric.nn.GATConv` instead of :class:`GATv2Conv`.

    The Brody, Alon & Yahav (2022) paper argues GATv1's attention is *static*
    (the ranking of keys is the same regardless of query); GATv2 makes it
    *dynamic*. This class lets us run the same config under both attentions
    to test the claim on this dataset.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        concat: bool | list[bool] = True,
        dropout: float = 0.5,
        attn_dropout: float = 0.2,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        self.dropout = dropout
        self.heads = heads

        if isinstance(concat, bool):
            concat_flags = [concat] * num_layers
        else:
            if len(concat) != num_layers:
                raise ValueError(
                    f"len(concat) must equal num_layers ({num_layers}); got {len(concat)}"
                )
            concat_flags = list(concat)

        per_head = max(hidden_channels // heads, 1)

        def layer_out(flag: bool) -> int:
            return per_head * heads if flag else per_head

        self.convs = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if use_batchnorm else None

        cur_in = in_channels
        for i in range(num_layers):
            flag = concat_flags[i]
            self.convs.append(
                GATConv(
                    cur_in,
                    per_head,
                    heads=heads,
                    concat=flag,
                    dropout=attn_dropout,
                )
            )
            cur_in = layer_out(flag)
            if self.norms is not None:
                self.norms.append(nn.BatchNorm1d(cur_in))

        self.out_proj = nn.Linear(cur_in, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            norm = self.norms[i] if self.norms is not None else None
            h = _apply_block(h, h_in, norm, self.dropout, self.training, activation="elu")
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# SignedGCN
# ---------------------------------------------------------------------------


class SignedGCNEncoder(nn.Module):
    """Thin wrapper around :class:`torch_geometric.nn.SignedGCN`.

    The encoder takes ``(x, pos_edge_index, neg_edge_index)`` and returns
    ``(num_nodes, hidden_channels)``. Empty positive/negative sub-graphs
    don't crash the forward pass: we log a warning and inject a single
    dummy self-loop on node 0, which has a negligible impact on aggregation
    relative to a realistically populated counterpart.

    Note: ``out_channels`` of this encoder equals ``hidden_channels`` (there
    is no separate linear projection — the SignedGCN implementation already
    returns ``hidden_channels``-dimensional embeddings).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        lamb: float = 5.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.lamb = float(lamb)
        # PyG's SignedGCN doesn't ingest external ``x`` directly — its
        # ``conv1`` is sized for the spectral feature dim it computes
        # internally. We expose a learnable input projection so callers can
        # pass our own ``x`` and have it shaped correctly for the model.
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.signed = SignedGCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            lamb=lamb,
        )

    @staticmethod
    def _dummy_edge(device: torch.device) -> torch.Tensor:
        return torch.zeros((2, 1), dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if pos_edge_index.numel() == 0 or pos_edge_index.shape[1] == 0:
            log.warning(
                "SignedGCNEncoder: no POSITIVE edges in the message-passing graph; "
                "injecting a dummy self-loop. Train metrics may be unreliable."
            )
            pos_edge_index = self._dummy_edge(x.device)
        if neg_edge_index.numel() == 0 or neg_edge_index.shape[1] == 0:
            log.warning(
                "SignedGCNEncoder: no NEGATIVE edges in the message-passing graph; "
                "injecting a dummy self-loop. The encoder will degenerate to a "
                "single-sign aggregation for this forward pass."
            )
            neg_edge_index = self._dummy_edge(x.device)

        x_proj = self.input_proj(x)
        return self.signed(x_proj, pos_edge_index, neg_edge_index)


__all__ = ["GATEncoder", "GATv1Encoder", "GCNEncoder", "SAGEEncoder", "SignedGCNEncoder"]
