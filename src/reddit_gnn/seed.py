"""Deterministic seeding utilities.

:func:`set_global_seed` seeds Python's :mod:`random`, NumPy, PyTorch
(CPU + CUDA), enables deterministic cuDNN, and delegates to
``torch_geometric.seed_everything`` for PyG-internal RNGs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch/CUDA/PyG with the given integer.

    Imports of torch and PyG are deferred so this module remains importable
    even before the heavy ML deps are installed (useful in CI smoke checks).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism — accept the perf hit for reproducible thesis results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover — torch is a hard dep
        pass

    try:
        from torch_geometric import seed_everything as pyg_seed_everything

        pyg_seed_everything(seed)
    except ImportError:  # pragma: no cover — PyG is a hard dep
        pass


__all__ = ["set_global_seed"]
