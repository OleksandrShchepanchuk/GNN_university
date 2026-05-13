"""Matplotlib plotting helpers used by the notebooks and report assets.

Every public ``plot_*`` function returns a ``matplotlib.figure.Figure`` and
accepts an optional ``save_path`` argument. When ``save_path`` is provided the
figure is written there (parent directories are created on demand) and the
returned figure remains usable for inline display in notebooks.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

POSITIVE_COLOR = "#2ca02c"  # green
NEGATIVE_COLOR = "#d62728"  # red


def _maybe_save(fig: Figure, save_path: str | Path | None, dpi: int = 150) -> Figure:
    """Persist ``fig`` to ``save_path`` if it is not ``None``; return ``fig``."""
    if save_path is None:
        return fig
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return fig


__all__ = ["NEGATIVE_COLOR", "POSITIVE_COLOR", "_maybe_save"]
