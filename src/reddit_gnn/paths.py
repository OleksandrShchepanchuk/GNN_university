"""Project path constants.

A thin wrapper around :class:`reddit_gnn.config.Paths` so that other modules
can do ``from reddit_gnn.paths import PATHS`` without re-importing the config
module each time. All paths are derived from the project root; nothing is
hardcoded to an absolute filesystem location.
"""

from __future__ import annotations

from reddit_gnn.config import Paths

PATHS: Paths = Paths()

__all__ = ["PATHS"]
