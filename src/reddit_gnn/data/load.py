"""Parse the SNAP TSV/CSV files into pandas DataFrames.

The two hyperlink TSVs share the schema::

    SOURCE_SUBREDDIT \\t TARGET_SUBREDDIT \\t POST_ID \\t TIMESTAMP \\t POST_LABEL \\t POST_PROPERTIES

``POST_PROPERTIES`` is a comma-separated 86-D vector. Loaders may keep it as a
string column for downstream parsing or expand to columns ``prop_0`` … ``prop_85``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_hyperlinks_tsv(path: str | Path) -> Any:
    """Read one hyperlink TSV and return a pandas DataFrame."""
    raise NotImplementedError("data.load.load_hyperlinks_tsv is not implemented yet")


def load_subreddit_embeddings(path: str | Path) -> Any:
    """Read ``web-redditEmbeddings-subreddits.csv`` and return a DataFrame
    indexed by subreddit name with 300 numeric columns."""
    raise NotImplementedError("data.load.load_subreddit_embeddings is not implemented yet")


def load_all_raw(raw_dir: str | Path) -> dict[str, Any]:
    """Load all three SNAP files and return a dict of DataFrames."""
    raise NotImplementedError("data.load.load_all_raw is not implemented yet")


__all__ = ["load_hyperlinks_tsv", "load_subreddit_embeddings", "load_all_raw"]
