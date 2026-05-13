"""Loss functions for edge sign classification.

* :func:`weighted_cross_entropy` — standard CE with per-class weights to
  counteract the heavy skew toward neutral/positive edges.
* :func:`focal_loss` — optional alternative for severe imbalance.
"""

from __future__ import annotations

from typing import Any


def weighted_cross_entropy(logits: Any, y: Any, class_weight: Any | None = None) -> Any:
    """Cross-entropy with optional per-class weight tensor."""
    raise NotImplementedError("training.losses.weighted_cross_entropy is not implemented yet")


def focal_loss(logits: Any, y: Any, gamma: float = 2.0, alpha: float | None = None) -> Any:
    """Focal loss for class-imbalanced edge sign classification."""
    raise NotImplementedError("training.losses.focal_loss is not implemented yet")


def compute_class_weight(y: Any, scheme: str = "balanced") -> Any:
    """Compute per-class weights from training labels."""
    raise NotImplementedError("training.losses.compute_class_weight is not implemented yet")


__all__ = ["compute_class_weight", "focal_loss", "weighted_cross_entropy"]
