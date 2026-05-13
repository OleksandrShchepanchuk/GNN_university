"""Rich-configured logging helpers.

:func:`get_logger` returns a stdlib :class:`logging.Logger` whose root handler
is a :class:`rich.logging.RichHandler`. The handler is installed only once
per process so repeated calls don't duplicate log lines.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_DEFAULT_LEVEL = os.environ.get("REDDIT_GNN_LOG_LEVEL", "INFO").upper()
_CONFIGURED: bool = False


def _configure_root(level: str = _DEFAULT_LEVEL) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_time=True,
        show_path=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a rich-configured logger for the given dotted ``name``."""
    _configure_root(level or _DEFAULT_LEVEL)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
