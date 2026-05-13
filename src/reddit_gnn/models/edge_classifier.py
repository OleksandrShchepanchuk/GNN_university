"""End-to-end edge sign classifier: ``encoder -> decoder -> logits[S]``.

For regular (single-graph) encoders call :meth:`EdgeClassifier.forward`. For
the signed encoder, which expects positive / negative edge sets separately,
call :meth:`EdgeClassifier.forward_signed`. The caller is responsible for
selecting the right method; using the wrong one will raise a clear ``TypeError``
from the underlying encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class EdgeClassifier(nn.Module):
    """Compose an encoder + decoder into a single trainable module."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _decode(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_attr_for_label: torch.Tensor | None,
    ) -> torch.Tensor:
        src, tgt = edge_label_index[0], edge_label_index[1]
        z_src = z.index_select(0, src)
        z_tgt = z.index_select(0, tgt)
        return self.decoder(z_src, z_tgt, edge_attr_for_label)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_attr_for_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self._decode(z, edge_label_index, edge_attr_for_label)

    def forward_signed(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_attr_for_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encoder(x, pos_edge_index, neg_edge_index)
        return self._decode(z, edge_label_index, edge_attr_for_label)


def parameter_count(model: nn.Module) -> int:
    """Total number of *trainable* parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_torch_model(
    cfg: dict,
    node_feature_dim: int,
    edge_feature_dim: int,
) -> nn.Module:
    """Construct a torch model (GNN or MLP baseline) from a merged YAML config.

    The sklearn baselines (``baseline_logreg``) are handled by the training
    script directly; this factory only covers the ``nn.Module`` paths.
    """
    from reddit_gnn.models.baselines import MLPEdgeBaseline
    from reddit_gnn.models.decoders import EdgeMLPDecoder
    from reddit_gnn.models.encoders import (
        GATEncoder,
        GCNEncoder,
        SAGEEncoder,
        SignedGCNEncoder,
    )

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type")
    if model_type is None:
        raise KeyError("cfg['model']['type'] is required")

    if model_type == "baseline_mlp":
        return MLPEdgeBaseline(
            edge_feature_dim=edge_feature_dim,
            node_feature_dim=node_feature_dim,
            hidden=int(model_cfg.get("hidden", 256)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    enc_cfg = dict(model_cfg.get("encoder", {}))
    dec_cfg = dict(model_cfg.get("decoder", {}))
    enc_cfg.pop("in_channels", None)  # always derived from data

    if model_type == "gcn":
        encoder = GCNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=int(enc_cfg.get("hidden_channels", 128)),
            out_channels=int(enc_cfg.get("out_channels", 64)),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            dropout=float(enc_cfg.get("dropout", 0.5)),
            use_batchnorm=bool(enc_cfg.get("use_batchnorm", False)),
        )
        node_dim_for_decoder = int(enc_cfg.get("out_channels", 64))
    elif model_type == "sage":
        encoder = SAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=int(enc_cfg.get("hidden_channels", 128)),
            out_channels=int(enc_cfg.get("out_channels", 64)),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            dropout=float(enc_cfg.get("dropout", 0.5)),
            use_batchnorm=bool(enc_cfg.get("use_batchnorm", False)),
            aggr=str(enc_cfg.get("aggr", "mean")),
        )
        node_dim_for_decoder = int(enc_cfg.get("out_channels", 64))
    elif model_type == "gat":
        encoder = GATEncoder(
            in_channels=node_feature_dim,
            hidden_channels=int(enc_cfg.get("hidden_channels", 128)),
            out_channels=int(enc_cfg.get("out_channels", 64)),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            heads=int(enc_cfg.get("heads", 4)),
            concat=enc_cfg.get("concat", True),
            dropout=float(enc_cfg.get("dropout", 0.5)),
            attn_dropout=float(enc_cfg.get("attn_dropout", 0.2)),
            use_batchnorm=bool(enc_cfg.get("use_batchnorm", False)),
        )
        node_dim_for_decoder = int(enc_cfg.get("out_channels", 64))
    elif model_type == "signed_gcn":
        encoder = SignedGCNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=int(enc_cfg.get("hidden_channels", 64)),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            lamb=float(enc_cfg.get("lamb", 5.0)),
        )
        # PyG's SignedGCN returns ``hidden_channels`` (a common misread is 2×).
        node_dim_for_decoder = int(enc_cfg.get("hidden_channels", 64))
    else:
        raise ValueError(
            f"Unknown model type {model_type!r}; expected one of "
            "{baseline_mlp, gcn, sage, gat, signed_gcn}"
        )

    decoder = EdgeMLPDecoder(
        node_dim=node_dim_for_decoder,
        edge_feature_dim=edge_feature_dim,
        hidden=int(dec_cfg.get("hidden", 64)),
        dropout=float(dec_cfg.get("dropout", 0.3)),
    )
    return EdgeClassifier(encoder, decoder)


__all__ = ["EdgeClassifier", "build_torch_model", "parameter_count"]
