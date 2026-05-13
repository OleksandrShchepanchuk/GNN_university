"""MLflow tracking backend.

A thin, dependency-injected wrapper around the parts of MLflow that this
project actually uses. The rest of the codebase imports from
:mod:`reddit_gnn.tracking` (which re-exports the names here) so MLflow can
be disabled — or swapped for a different backend — without touching any
training or scripting code.

Two invariants the helpers maintain:

1. **Tracking failure never crashes training.** Every public helper catches
   ``BaseException`` from MLflow internals and logs a warning via the
   project's :func:`reddit_gnn.utils.logging.get_logger`; the caller never
   sees an exception.
2. **No-op when disabled.** When :func:`init_tracking` is called with
   ``enabled=False`` (or never called at all), every helper returns without
   side effects. This is what makes the tests robust against the global
   state MLflow maintains on its singleton client.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)

# Module-level singletons. These are intentionally simple — pytest fixtures
# reset them between tests via direct assignment.
_ENABLED: bool = False
_INITIALIZED: bool = False
_TRACKING_URI: str | None = None
_EXPERIMENT_NAME: str = "reddit_signed"


def _safe_mlflow():
    """Lazy import of mlflow. Returns ``None`` if mlflow is unavailable."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def init_tracking(
    tracking_uri: str | None = None,
    experiment_name: str = "reddit_signed",
    enabled: bool = True,
) -> None:
    """Configure the MLflow client.

    When ``tracking_uri`` is ``None`` we default to ``file:<project_root>/mlruns``
    so a fresh checkout writes runs into the repo's working tree. Set ``enabled``
    to ``False`` to make every other helper here a no-op (handy for tests and
    for the ``--no-tracking`` CLI override).
    """
    global _ENABLED, _INITIALIZED, _TRACKING_URI, _EXPERIMENT_NAME
    _INITIALIZED = True
    _ENABLED = bool(enabled)
    _EXPERIMENT_NAME = experiment_name

    if not _ENABLED:
        log.info("MLflow tracking is DISABLED")
        return

    mlflow = _safe_mlflow()
    if mlflow is None:
        log.warning("mlflow is not installed; tracking silently disabled")
        _ENABLED = False
        return

    if tracking_uri is None:
        from reddit_gnn.config import Paths

        tracking_uri = "file:" + str(Paths().project_root / "mlruns")
    _TRACKING_URI = tracking_uri

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("MLflow init failed (%s); disabling tracking", exc)
        _ENABLED = False
        return

    try:
        sys_metrics = getattr(mlflow, "system_metrics", None)
        if sys_metrics is not None and hasattr(sys_metrics, "enable_system_metrics_logging"):
            sys_metrics.enable_system_metrics_logging()
    except Exception as exc:
        log.debug("could not enable system_metrics logging: %s", exc)

    log.info(
        "MLflow tracking initialised: uri=%s experiment=%s",
        tracking_uri,
        experiment_name,
    )


def _has_active_run() -> bool:
    if not _ENABLED:
        return False
    mlflow = _safe_mlflow()
    if mlflow is None:
        return False
    try:
        return mlflow.active_run() is not None
    except Exception:
        return False


def active_run_id() -> str | None:
    """Return the currently active mlflow run's id, or ``None``."""
    if not _ENABLED:
        return None
    mlflow = _safe_mlflow()
    if mlflow is None:
        return None
    try:
        run = mlflow.active_run()
        return run.info.run_id if run is not None else None
    except Exception:
        return None


def start_run(
    run_name: str | None = None,
    nested: bool = False,
    tags: Mapping[str, str] | None = None,
):
    """Start an mlflow run. Returns ``None`` when tracking is disabled."""
    if not _ENABLED:
        return None
    mlflow = _safe_mlflow()
    if mlflow is None:
        return None
    try:
        return mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=dict(tags) if tags else None,
        )
    except Exception as exc:
        log.warning("MLflow start_run failed: %s", exc)
        return None


def end_run(status: str | None = None) -> None:
    """End the active mlflow run, if any."""
    if not _ENABLED:
        return
    mlflow = _safe_mlflow()
    if mlflow is None:
        return
    try:
        if status is None:
            mlflow.end_run()
        else:
            mlflow.end_run(status=status)
    except Exception as exc:
        log.warning("MLflow end_run failed: %s", exc)


def log_params(d: Mapping[str, Any]) -> None:
    """Log a dict of params. Non-(int|float|bool|str) values are stringified."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    coerced: dict[str, Any] = {}
    for k, v in d.items():
        coerced[str(k)] = v if isinstance(v, (int, float, bool, str)) else str(v)
    try:
        mlflow.log_params(coerced)
    except Exception as exc:
        log.warning("MLflow log_params failed: %s", exc)


def log_metrics(d: Mapping[str, float], step: int | None = None) -> None:
    """Log numeric metrics. Non-finite values are dropped with a warning."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    cleaned: dict[str, float] = {}
    dropped: list[str] = []
    for k, v in d.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            dropped.append(str(k))
            continue
        if not np.isfinite(fv):
            dropped.append(str(k))
            continue
        cleaned[str(k)] = fv
    if dropped:
        log.warning("Dropping non-finite metric(s): %s", dropped)
    if not cleaned:
        return
    try:
        mlflow.log_metrics(cleaned, step=step)
    except Exception as exc:
        log.warning("MLflow log_metrics failed: %s", exc)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """Upload an existing local file to the active run's artifact store."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        log.warning("MLflow log_artifact failed (%s): %s", path, exc)


def log_figure(fig, filename: str, artifact_path: str = "figures") -> None:
    """Save a matplotlib ``Figure`` to a temp file, log it, then close it."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    try:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight", dpi=150)
            mlflow.log_artifact(str(out), artifact_path=artifact_path)
        plt.close(fig)
    except Exception as exc:
        log.warning("MLflow log_figure failed (%s): %s", filename, exc)


def log_text(text: str, filename: str) -> None:
    """Log a short text artifact (config dumps, summaries, etc.)."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    try:
        mlflow.log_text(text, filename)
    except Exception as exc:
        log.warning("MLflow log_text failed (%s): %s", filename, exc)


def log_model_state(state_dict_path: Path, artifact_path: str = "model") -> None:
    """Upload a saved PyTorch state-dict file. (We avoid mlflow autolog.)"""
    if not _has_active_run():
        return
    log_artifact(state_dict_path, artifact_path=artifact_path)


def set_tags(d: Mapping[str, str]) -> None:
    """Set or update mlflow tags on the active run."""
    if not _has_active_run():
        return
    mlflow = _safe_mlflow()
    try:
        mlflow.set_tags({str(k): str(v) for k, v in d.items()})
    except Exception as exc:
        log.warning("MLflow set_tags failed: %s", exc)


class ExperimentContext:
    """Context manager that wraps ``mlflow.start_run`` / ``mlflow.end_run``.

    The same ``log_*`` helpers exist as instance methods so callers can chain
    them off the context (``with ExperimentContext('run') as ctx: ctx.log_params(...)``).
    Exceptions raised inside the ``with`` block are recorded as tags on the
    run (``exception_type`` / ``exception_message``) before the run ends.
    """

    def __init__(
        self,
        run_name: str,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.run_name = run_name
        self.nested = nested
        self.tags = dict(tags) if tags else None
        self._run_id: str | None = None
        self._run = None

    def __enter__(self) -> ExperimentContext:
        self._run = start_run(run_name=self.run_name, nested=self.nested, tags=self.tags)
        # Capture the run id at enter time so it remains correct after a nested
        # child run becomes the "active" one.
        self._run_id = active_run_id()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            with contextlib.suppress(Exception):
                set_tags(
                    {
                        "exception_type": exc_type.__name__,
                        "exception_message": str(exc_val)[:512],
                    }
                )
            end_run(status="FAILED")
        else:
            end_run()
        return False  # never swallow exceptions

    @property
    def run_id(self) -> str | None:
        return self._run_id

    # ---- chained helpers --------------------------------------------------
    def log_params(self, d: Mapping[str, Any]) -> None:
        log_params(d)

    def log_metrics(self, d: Mapping[str, float], step: int | None = None) -> None:
        log_metrics(d, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        log_artifact(path, artifact_path=artifact_path)

    def log_figure(self, fig, filename: str, artifact_path: str = "figures") -> None:
        log_figure(fig, filename, artifact_path=artifact_path)

    def log_text(self, text: str, filename: str) -> None:
        log_text(text, filename)

    def log_model_state(self, state_dict_path: Path, artifact_path: str = "model") -> None:
        log_model_state(state_dict_path, artifact_path=artifact_path)

    def set_tags(self, d: Mapping[str, str]) -> None:
        set_tags(d)


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
