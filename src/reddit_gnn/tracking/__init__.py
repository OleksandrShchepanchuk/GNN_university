"""Public tracking API.

The project never imports ``mlflow.*`` directly — all callers route through
this module. That makes it possible to disable tracking (or to swap MLflow
for a different backend) without touching training code.
"""

from __future__ import annotations

from reddit_gnn.tracking.mlflow_backend import (
    ExperimentContext,
    active_run_id,
    end_run,
    init_tracking,
    log_artifact,
    log_figure,
    log_metrics,
    log_model_state,
    log_params,
    log_text,
    set_tags,
    start_run,
)

__all__ = [
    "ExperimentContext",
    "active_run_id",
    "end_run",
    "init_tracking",
    "log_artifact",
    "log_figure",
    "log_metrics",
    "log_model_state",
    "log_params",
    "log_text",
    "set_tags",
    "start_run",
]
