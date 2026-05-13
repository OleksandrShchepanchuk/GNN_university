"""Optuna sweep utilities.

``configs/sweep.yaml`` declares a search space over keys in a base config.
:func:`run_study` instantiates the study, runs ``n_trials`` objectives, and
returns the best trial together with the merged config that produced it.
"""

from __future__ import annotations

from typing import Any


def build_search_space(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized search-space dict consumed by :func:`objective`."""
    raise NotImplementedError("training.tune.build_search_space is not implemented yet")


def objective(trial: Any, base_cfg: dict[str, Any], search_space: dict[str, Any]) -> float:
    """Single Optuna trial. Samples a config and returns the primary metric."""
    raise NotImplementedError("training.tune.objective is not implemented yet")


def run_study(cfg: dict[str, Any]) -> Any:
    """Run the full study described by ``configs/sweep.yaml``."""
    raise NotImplementedError("training.tune.run_study is not implemented yet")


__all__ = ["build_search_space", "objective", "run_study"]
