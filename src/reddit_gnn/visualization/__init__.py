"""Matplotlib plotting helpers used by the notebooks and report assets.

Every public ``plot_*`` function returns a ``matplotlib.figure.Figure`` and
accepts an optional ``save_path`` argument. When ``save_path`` is provided the
figure is written there (parent directories are created on demand) and the
returned figure remains usable for inline display in notebooks.

Call :func:`setup_plotting_style` once at the top of a notebook or script to
apply the project's house style (consistent palette, fonts, grid). The style
is opt-in so that script consumers who want raw matplotlib defaults are
unaffected.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Headline class colors — used consistently across distribution, temporal,
# subgraph, and result plots so the reader can recognise the negative-class
# color even at thumbnail size.
POSITIVE_COLOR = (
    "#3a8a3a"  # muted green (was "#2ca02c", which clashed with the categorical palette)
)
NEGATIVE_COLOR = "#c0392b"  # muted red

# Categorical palette for cross-model overlays etc. Picked for high contrast +
# print-friendliness; first colour matches PRIMARY_COLOR.
PRIMARY_COLOR = "#3b6ea8"
PALETTE = [
    "#3b6ea8",  # blue   (primary)
    "#d97a3e",  # orange
    "#3a8a3a",  # green  (also POSITIVE_COLOR)
    "#c0392b",  # red    (also NEGATIVE_COLOR)
    "#7d6cb0",  # purple
    "#7f6a4f",  # brown
]


def setup_plotting_style() -> None:
    """Apply the project house style to matplotlib's global ``rcParams``.

    Idempotent. Call once at the top of a notebook. Affects font sizes, grid
    style, axis spines, default figure DPI, and the categorical color cycle.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.titlepad": 10,
            "axes.labelsize": 10.0,
            "axes.labelcolor": "#222",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#bbb",
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.6,
            "xtick.color": "#444",
            "ytick.color": "#444",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.size": 10,
            "font.family": "sans-serif",
            "image.cmap": "viridis",
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def _maybe_save(fig: Figure, save_path: str | Path | None, dpi: int = 150) -> Figure:
    """Persist ``fig`` to ``save_path`` if it is not ``None``; return ``fig``."""
    if save_path is None:
        return fig
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return fig


__all__ = [
    "NEGATIVE_COLOR",
    "PALETTE",
    "POSITIVE_COLOR",
    "PRIMARY_COLOR",
    "_maybe_save",
    "setup_plotting_style",
]
