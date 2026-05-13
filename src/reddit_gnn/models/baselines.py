"""Non-graph baselines for edge sign classification.

These ignore the message-passing structure entirely; they exist so we can
quote the lift that a GNN's aggregation gives over flat feature concatenation.

* :class:`MajorityClassifier` — always predicts the train-majority label.
* :class:`LogisticRegressionBaseline` — sklearn ``LogisticRegression`` on
  raw edge features only.
* :class:`LogisticRegressionWithNodeFeats` — same, but the per-edge feature
  vector is concatenated with the four endpoint-derived blocks
  ``[x_src, x_tgt, |x_src - x_tgt|, x_src * x_tgt]``.
* :class:`MLPEdgeBaseline` — a small PyTorch MLP on the same concatenation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Concatenation helper shared by node-aware baselines and the MLP
# ---------------------------------------------------------------------------


def _concat_node_blocks_numpy(
    edge_feats: np.ndarray,
    x: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """``[edge_feats, x_src, x_tgt, |x_src - x_tgt|, x_src * x_tgt]`` for numpy inputs."""
    src = edge_index[0]
    tgt = edge_index[1]
    x_src = x[src]
    x_tgt = x[tgt]
    return np.concatenate(
        [edge_feats, x_src, x_tgt, np.abs(x_src - x_tgt), x_src * x_tgt],
        axis=1,
    )


def _concat_node_blocks_torch(
    edge_feats: torch.Tensor,
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
) -> torch.Tensor:
    """Torch variant: caller has already gathered ``x_src`` / ``x_tgt``."""
    return torch.cat(
        [edge_feats, x_src, x_tgt, torch.abs(x_src - x_tgt), x_src * x_tgt],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Majority-class baseline
# ---------------------------------------------------------------------------


class MajorityClassifier:
    """Predicts the most frequent training label for every input.

    Useful only as a sanity floor — any real model must beat this.
    """

    def __init__(self) -> None:
        self.majority_: int | None = None
        self.classes_: np.ndarray | None = None
        self.proba_: np.ndarray | None = None  # empirical class shares

    def fit(self, y: np.ndarray | torch.Tensor) -> "MajorityClassifier":
        y_arr = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        counts = Counter(int(v) for v in y_arr.tolist())
        self.majority_ = counts.most_common(1)[0][0]
        self.classes_ = np.array(sorted(counts.keys()), dtype=np.int64)
        total = sum(counts.values())
        self.proba_ = np.array([counts[c] / total for c in self.classes_], dtype=np.float64)
        return self

    def _length(self, X) -> int:
        if hasattr(X, "shape"):
            return int(X.shape[0])
        return int(len(X))

    def predict(self, X) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("MajorityClassifier was not fit yet")
        return np.full(self._length(X), self.majority_, dtype=np.int64)

    def predict_proba(self, X) -> np.ndarray:
        if self.majority_ is None or self.classes_ is None or self.proba_ is None:
            raise RuntimeError("MajorityClassifier was not fit yet")
        out = np.tile(self.proba_, (self._length(X), 1))
        return out


# ---------------------------------------------------------------------------
# Logistic regression on edge features only
# ---------------------------------------------------------------------------


class LogisticRegressionBaseline:
    """``sklearn.linear_model.LogisticRegression`` on per-edge features only.

    Kwargs (``C``, ``solver``, ``max_iter``, ``class_weight``, ...) are passed
    straight through to sklearn.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_iter", 1000)
        kwargs.setdefault("class_weight", "balanced")
        self.model = LogisticRegression(**kwargs)

    def fit(self, edge_feats: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(edge_feats, y)
        return self

    def predict(self, edge_feats: np.ndarray) -> np.ndarray:
        return self.model.predict(edge_feats)

    def predict_proba(self, edge_feats: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(edge_feats)


# ---------------------------------------------------------------------------
# Logistic regression with node-feature concatenation
# ---------------------------------------------------------------------------


class LogisticRegressionWithNodeFeats:
    """LogReg on ``[edge_feats || x_src || x_tgt || |x_src - x_tgt| || x_src * x_tgt]``."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_iter", 1000)
        kwargs.setdefault("class_weight", "balanced")
        self.model = LogisticRegression(**kwargs)

    def _features(
        self, edge_feats: np.ndarray, x: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        return _concat_node_blocks_numpy(edge_feats, x, edge_index)

    def fit(
        self,
        edge_feats: np.ndarray,
        x: np.ndarray,
        edge_index: np.ndarray,
        y: np.ndarray,
    ) -> "LogisticRegressionWithNodeFeats":
        self.model.fit(self._features(edge_feats, x, edge_index), y)
        return self

    def predict(self, edge_feats: np.ndarray, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self.model.predict(self._features(edge_feats, x, edge_index))

    def predict_proba(
        self, edge_feats: np.ndarray, x: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        return self.model.predict_proba(self._features(edge_feats, x, edge_index))


# ---------------------------------------------------------------------------
# MLP baseline (torch.nn.Module)
# ---------------------------------------------------------------------------


class MLPEdgeBaseline(nn.Module):
    """Two-hidden-layer MLP on the same ``[edge || node-block]`` concatenation.

    Input dim is fixed at construction: ``edge_feature_dim + 4 * node_feature_dim``.
    Returns a single logit per supervision edge (binary classification).
    """

    def __init__(
        self,
        edge_feature_dim: int,
        node_feature_dim: int,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.edge_feature_dim = int(edge_feature_dim)
        self.node_feature_dim = int(node_feature_dim)
        in_dim = self.edge_feature_dim + 4 * self.node_feature_dim
        self.net = nn.Sequential(
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
        edge_feats: torch.Tensor,
        x: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src, tgt = edge_label_index[0], edge_label_index[1]
        x_src = x.index_select(0, src)
        x_tgt = x.index_select(0, tgt)
        feats = _concat_node_blocks_torch(edge_feats, x_src, x_tgt)
        return self.net(feats).squeeze(-1)


__all__ = [
    "LogisticRegressionBaseline",
    "LogisticRegressionWithNodeFeats",
    "MLPEdgeBaseline",
    "MajorityClassifier",
]
