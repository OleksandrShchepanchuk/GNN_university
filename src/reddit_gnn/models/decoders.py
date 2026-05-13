"""Edge decoders.

Given encoder embeddings for the two endpoints of a supervision edge and
(optionally) per-edge attributes, the decoder emits a single logit per edge.
"""

from __future__ import annotations

import torch
from torch import nn


class EdgeMLPDecoder(nn.Module):
    """MLP decoder over the canonical edge representation.

    Edge representation:
        ``[z_src || z_tgt || |z_src - z_tgt| || z_src * z_tgt || edge_attr_for_label]``

    Input dimension is fixed at construction:
        ``4 * node_dim + edge_feature_dim``

    Returns ``[S]`` logits (one per supervision edge); the caller applies
    ``BCEWithLogitsLoss`` / sigmoid downstream.
    """

    def __init__(
        self,
        node_dim: int,
        edge_feature_dim: int,
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        in_dim = 4 * self.node_dim + self.edge_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        edge_attr_for_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [z_src, z_tgt, torch.abs(z_src - z_tgt), z_src * z_tgt]
        if self.edge_feature_dim > 0:
            if edge_attr_for_label is None:
                raise ValueError(
                    "EdgeMLPDecoder was built with edge_feature_dim>0 but "
                    "edge_attr_for_label is None at forward time"
                )
            if edge_attr_for_label.shape[-1] != self.edge_feature_dim:
                raise ValueError(
                    f"edge_attr_for_label has {edge_attr_for_label.shape[-1]} cols; "
                    f"expected {self.edge_feature_dim}"
                )
            parts.append(edge_attr_for_label)
        edge_repr = torch.cat(parts, dim=-1)
        return self.mlp(edge_repr).squeeze(-1)


__all__ = ["EdgeMLPDecoder"]
