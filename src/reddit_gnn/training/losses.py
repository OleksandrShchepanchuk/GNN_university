"""Losses for edge sign classification.

The dataset is heavily class-imbalanced (~90% positive / 10% negative), so
the default loss is :func:`weighted_bce_with_logits` with ``pos_weight``
computed once on the training supervision set. The focal loss variant is
exposed for ablation experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def compute_pos_weight(labels) -> float:
    """``pos_weight = #neg / #pos`` for use with BCEWithLogitsLoss.

    Note: ``BCEWithLogitsLoss(pos_weight=w)`` re-scales the positive-class
    loss term by ``w``. Setting ``w = #neg / #pos`` makes the expected
    contribution from each class equal, which is the standard inverse-
    frequency reweighting.
    """
    arr = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    arr = arr.astype(np.int64)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """``BCEWithLogitsLoss`` with a per-class positive weight (scalar)."""
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), pos_weight=pw)


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Binary focal loss with logits.

    With ``alpha < 0.5`` the rare class (``y == 0``) gets *more* weight, which
    is the intended behaviour in our skewed dataset (class 1 is majority).
    """
    target_f = target.to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target_f > 0.5, p, 1.0 - p)
    alpha_t = torch.where(
        target_f > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1.0 - alpha)
    )
    loss = alpha_t * (1.0 - pt).pow(gamma) * ce
    return loss.mean()


__all__ = [
    "compute_pos_weight",
    "focal_loss_with_logits",
    "weighted_bce_with_logits",
]
