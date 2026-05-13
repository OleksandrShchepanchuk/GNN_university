"""reddit_gnn — edge sign classification on the SNAP Reddit Hyperlink Network.

The task is classification of the sentiment label attached to *observed*
subreddit-to-subreddit hyperlinks (`POST_LABEL`), remapped as
`-1 -> 0` (negative) and `+1 -> 1` (neutral/positive). No negative sampling
is ever performed; label `0` is a real class, not a "non-edge".

This file is a thin package marker. Public surface is exposed by submodules.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Oleksandr Shchepanchuk"

__all__ = ["__author__", "__version__"]
