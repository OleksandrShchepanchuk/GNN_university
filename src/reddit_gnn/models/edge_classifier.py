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


__all__ = ["EdgeClassifier", "parameter_count"]
