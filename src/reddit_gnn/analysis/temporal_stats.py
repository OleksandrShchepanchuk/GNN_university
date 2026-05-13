"""Temporal statistics: edge counts per month, sentiment drift, burstiness.

These feed the temporal plots in the EDA notebook and motivate the temporal
train/val/test split.
"""

from __future__ import annotations

from typing import Any


def edges_per_period(edge_df: Any, freq: str = "M") -> Any:
    """Return a Series indexed by period (e.g. month) with edge counts."""
    raise NotImplementedError("analysis.temporal_stats.edges_per_period is not implemented yet")


def sign_share_over_time(edge_df: Any, freq: str = "M") -> Any:
    """Return per-period share of negative vs neutral/positive edges."""
    raise NotImplementedError("analysis.temporal_stats.sign_share_over_time is not implemented yet")


def inter_event_time(edge_df: Any) -> Any:
    """Distribution of time between consecutive hyperlink events."""
    raise NotImplementedError("analysis.temporal_stats.inter_event_time is not implemented yet")


__all__ = ["edges_per_period", "inter_event_time", "sign_share_over_time"]
