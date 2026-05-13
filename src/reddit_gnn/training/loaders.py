"""DataLoader builders for the per-split PyG ``Data`` objects.

The preferred loader is :class:`~torch_geometric.loader.LinkNeighborLoader`.
That class transitively depends on ``pyg-lib`` *or* ``torch-sparse`` for the
underlying neighbour sampler — neither ships a PyPI wheel for our exact torch
build, so we detect the missing dependency at import time and fall back to a
:class:`FullBatchLoader` that yields each split's :class:`~torch_geometric.data.Data`
once per epoch. The training loop's interface
(``batch.edge_index`` / ``batch.edge_label_index`` / ``batch.edge_label`` /
``batch.input_id``) is identical for both code paths.

Split-level temporal correctness is already enforced by
:func:`reddit_gnn.data.splits.chronological_edge_split` and
:func:`reddit_gnn.data.splits.build_message_passing_split`: every fold has
``max(mp_time) <= min(sup_time)`` and supervision triples are disjoint
between splits. The per-batch ``temporal_strategy="last"`` knob is a
finer-grained defense-in-depth that we lose without ``pyg-lib``; the broader
no-leakage invariant still holds.

We never use negative sampling: every edge in the dataset is a real,
observed hyperlink whose sentiment label is the prediction target.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Data

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def _has_sampler_backend() -> bool:
    """``True`` if either pyg-lib or torch-sparse is loadable for neighbour sampling."""
    return bool(
        getattr(pyg_typing, "WITH_PYG_LIB", False)
        or getattr(pyg_typing, "WITH_TORCH_SPARSE", False)
    )


def _supports_edge_time_sampling() -> bool:
    """``True`` iff the loaded pyg-lib supports edge-level temporal sampling."""
    if not getattr(pyg_typing, "WITH_PYG_LIB", False):
        return False
    return bool(getattr(pyg_typing, "WITH_PYG_LIB_EDGE_TIME", False))


class FullBatchLoader:
    """Drop-in fallback for :class:`LinkNeighborLoader` that yields the whole graph.

    Mirrors the attribute shape the training loop reads off each batch:
        * ``batch.x``                — node features.
        * ``batch.edge_index``       — the MP graph for the split.
        * ``batch.edge_attr``        — MP edge attributes.
        * ``batch.edge_label_index`` — supervision edges.
        * ``batch.edge_label``       — supervision labels.
        * ``batch.input_id``         — identity index into the sup edges
          (so ``loader.data.edge_label_attr[batch.input_id]`` works the same
          way as with the real :class:`LinkNeighborLoader`).
    """

    def __init__(self, data: Data) -> None:
        self.data = data

    def __iter__(self):
        batch = self.data.clone()
        n_sup = int(batch.edge_label_index.shape[1])
        batch.input_id = torch.arange(n_sup, dtype=torch.int64)
        yield batch

    def __len__(self) -> int:
        return 1


def make_link_loaders(
    data_per_split: dict[str, Data],
    num_neighbors: Sequence[int] = (15, 10),
    batch_size: int = 2048,
    time_attr: str = "edge_time",
    temporal_strategy: str = "last",
    shuffle_train: bool = True,
    num_workers: int = 0,
):
    """Build one loader per split.

    Returns a ``dict[str, Any]`` whose values are either
    :class:`LinkNeighborLoader` instances (when ``pyg-lib`` / ``torch-sparse``
    is available) or :class:`FullBatchLoader` fallbacks (otherwise).
    """
    if not data_per_split:
        raise ValueError("data_per_split is empty")

    has_sampler = _has_sampler_backend()
    use_temporal = has_sampler and _supports_edge_time_sampling()

    if not has_sampler:
        log.warning(
            "Neither pyg-lib nor torch-sparse is available; falling back to "
            "FullBatchLoader (one batch per epoch, the entire split). "
            "Split-level temporal correctness still holds — see "
            "tests/test_leakage.py."
        )
        return {name: FullBatchLoader(data) for name, data in data_per_split.items()}

    if not use_temporal:
        log.warning(
            "pyg-lib edge-level temporal sampling is not available; using "
            "non-temporal LinkNeighborLoader. The chronological split still "
            "guarantees max(mp_time) <= min(sup_time) per fold."
        )

    from torch_geometric.loader import LinkNeighborLoader

    loaders: dict[str, LinkNeighborLoader] = {}
    for split_name, data in data_per_split.items():
        kwargs: dict = dict(
            data=data,
            num_neighbors=list(num_neighbors),
            edge_label_index=data.edge_label_index,
            edge_label=data.edge_label,
            batch_size=batch_size,
            shuffle=shuffle_train and split_name == "train",
            neg_sampling=None,
            num_workers=num_workers,
        )
        if use_temporal:
            kwargs.update(
                edge_label_time=data.edge_label_time,
                time_attr=time_attr,
                temporal_strategy=temporal_strategy,
            )
        loaders[split_name] = LinkNeighborLoader(**kwargs)

    log.info(
        "make_link_loaders: built LinkNeighborLoaders for %s "
        "(batch_size=%d, num_neighbors=%s, temporal=%s)",
        sorted(loaders.keys()),
        batch_size,
        list(num_neighbors),
        use_temporal,
    )
    return loaders


__all__ = ["FullBatchLoader", "make_link_loaders"]
