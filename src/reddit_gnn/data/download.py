"""Download the three SNAP Reddit Hyperlink files into ``data/raw/``.

Authoritative URLs:
    https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
    https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv
    https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv

Behavior to be implemented:
    * skip downloads when the file already exists and its size matches the
      expected ``Content-Length`` (mirror the SNAP page);
    * stream with ``tqdm`` progress bars;
    * verify TSVs parse to a non-empty header line.
"""

from __future__ import annotations

from pathlib import Path

REDDIT_HYPERLINK_URLS: dict[str, str] = {
    "soc-redditHyperlinks-body.tsv": "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    "soc-redditHyperlinks-title.tsv": "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
    "web-redditEmbeddings-subreddits.csv": "https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv",
}


def download_all(raw_dir: str | Path, force: bool = False) -> list[Path]:
    """Download every SNAP file into ``raw_dir``. Returns the list of paths."""
    raise NotImplementedError("data.download.download_all is not implemented yet")


def download_one(url: str, dest: str | Path, force: bool = False) -> Path:
    """Stream a single URL to ``dest`` with progress."""
    raise NotImplementedError("data.download.download_one is not implemented yet")


__all__ = ["REDDIT_HYPERLINK_URLS", "download_all", "download_one"]
