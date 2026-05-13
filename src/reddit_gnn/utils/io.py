"""File I/O helpers for YAML configs, parquet tables, JSON, and metrics.

The two helpers :func:`save_metrics_json` and :func:`save_predictions_csv`
both write to local disk first (the local file remains the source of truth)
and then mirror the artifact to MLflow when a run is active. MLflow being
disabled is a no-op — the local write still happens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(obj: dict[str, Any], path: str | Path) -> Path:
    """Serialize ``obj`` to YAML at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet table into a pandas DataFrame."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a DataFrame to parquet at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: str | Path) -> Path:
    """Persist a JSON-serialisable object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
    return path


def save_metrics_json(
    d: dict[str, Any],
    path: str | Path,
    *,
    artifact_path: str | None = "metrics",
) -> Path:
    """Write a metrics dict to JSON locally **and** mirror it to MLflow.

    The local file is the source of truth — MLflow merely gets a copy when a
    run is active. When tracking is disabled the mirror call is a no-op.
    """
    out = save_json(d, path)
    # Lazy import to avoid pulling MLflow into modules that just want IO.
    from reddit_gnn.tracking import log_artifact

    log_artifact(out, artifact_path=artifact_path)
    return out


def save_predictions_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    artifact_path: str | None = "predictions",
) -> Path:
    """Write a predictions DataFrame to CSV locally **and** mirror to MLflow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    from reddit_gnn.tracking import log_artifact

    log_artifact(path, artifact_path=artifact_path)
    return path


__all__ = [
    "load_json",
    "load_parquet",
    "load_yaml",
    "save_json",
    "save_metrics_json",
    "save_parquet",
    "save_predictions_csv",
    "save_yaml",
]
