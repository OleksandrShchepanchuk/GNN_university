"""Non-graph baselines.

* :class:`LogRegBaseline` — sklearn ``LogisticRegression`` on
  ``[src_embed || dst_embed || engineered_edge_features]`` per edge.
* :class:`MLPBaseline` — a small PyTorch MLP on the same concatenated input.

Both ignore the message-passing structure entirely; they exist so we can quote
the lift that GNN-based aggregation provides over pure feature concatenation.
"""

from __future__ import annotations

from typing import Any


class LogRegBaseline:
    """Sklearn logistic regression edge classifier (no graph structure)."""

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError("LogRegBaseline.__init__ is not implemented yet")

    def fit(self, X: Any, y: Any) -> "LogRegBaseline":
        raise NotImplementedError("LogRegBaseline.fit is not implemented yet")

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("LogRegBaseline.predict_proba is not implemented yet")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("LogRegBaseline.predict is not implemented yet")


class MLPBaseline:
    """PyTorch MLP edge classifier (no graph structure)."""

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        raise NotImplementedError("MLPBaseline.__init__ is not implemented yet")


__all__ = ["LogRegBaseline", "MLPBaseline"]
