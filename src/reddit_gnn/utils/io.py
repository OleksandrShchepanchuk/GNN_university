"""File I/O helpers for YAML configs, parquet tables, and checkpoint metadata.

All functions raise :class:`NotImplementedError` until the data pipeline is
wired up. They are declared here so callers can import stable names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict."""
    raise NotImplementedError("io.load_yaml is not implemented yet")


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    """Serialize ``obj`` to YAML at ``path``."""
    raise NotImplementedError("io.save_yaml is not implemented yet")


def load_parquet(path: str | Path) -> Any:
    """Load a parquet table (returns a pandas DataFrame)."""
    raise NotImplementedError("io.load_parquet is not implemented yet")


def save_parquet(df: Any, path: str | Path) -> None:
    """Persist a pandas DataFrame to parquet at ``path``."""
    raise NotImplementedError("io.save_parquet is not implemented yet")


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Persist a JSON-serialisable object."""
    raise NotImplementedError("io.save_json is not implemented yet")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    raise NotImplementedError("io.load_json is not implemented yet")


__all__ = [
    "load_json",
    "load_parquet",
    "load_yaml",
    "save_json",
    "save_parquet",
    "save_yaml",
]
