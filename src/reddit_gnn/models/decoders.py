"""Edge decoders that map ``(z_src, z_dst, edge_attr) -> logits``.

Combiners:
    * ``concat`` — ``[z_src || z_dst]``;
    * ``hadamard`` — ``z_src * z_dst``;
    * ``abs_diff`` — ``|z_src - z_dst|``;
    * ``concat_plus_edgefeat`` — ``[z_src || z_dst || edge_attr]``.

The chosen combiner feeds a small MLP that outputs ``num_classes`` logits.
"""

from __future__ import annotations

from typing import Any


class MLPEdgeDecoder:
    """Edge-level MLP decoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        combiner: str = "concat",
    ) -> None:
        raise NotImplementedError("MLPEdgeDecoder.__init__ is not implemented yet")

    def forward(self, z_src: Any, z_dst: Any, edge_attr: Any | None = None) -> Any:
        raise NotImplementedError("MLPEdgeDecoder.forward is not implemented yet")


def combine_endpoint_embeddings(
    z_src: Any, z_dst: Any, edge_attr: Any | None, mode: str
) -> Any:
    """Apply one of {concat, hadamard, abs_diff, concat_plus_edgefeat}."""
    raise NotImplementedError("decoders.combine_endpoint_embeddings is not implemented yet")


__all__ = ["MLPEdgeDecoder", "combine_endpoint_embeddings"]
