"""End-to-end edge sign classifier: ``encoder -> decoder -> logits``.

The ``forward`` method takes:
    x          — node features ``[N, d_node]``
    edge_index — message-passing graph (TRAIN edges only at train time)
    pred_edges — the ``[2, K]`` edges whose labels we want to predict; these
                 may be the same as ``edge_index`` (train rows) or different
                 (val/test rows scored without exposing their labels).
    edge_attr  — optional ``[K, d_edge]`` engineered edge features for ``pred_edges``.

Returning ``[K, num_classes]`` logits.
"""

from __future__ import annotations

from typing import Any


class EdgeClassifier:
    """Compose an encoder + decoder into a single trainable module."""

    def __init__(self, encoder: Any, decoder: Any) -> None:
        raise NotImplementedError("EdgeClassifier.__init__ is not implemented yet")

    def forward(
        self,
        x: Any,
        edge_index: Any,
        pred_edges: Any,
        edge_attr: Any | None = None,
    ) -> Any:
        raise NotImplementedError("EdgeClassifier.forward is not implemented yet")


def build_model(cfg: dict[str, Any]) -> EdgeClassifier:
    """Factory: read a merged YAML config and return an :class:`EdgeClassifier`."""
    raise NotImplementedError("edge_classifier.build_model is not implemented yet")


__all__ = ["EdgeClassifier", "build_model"]
