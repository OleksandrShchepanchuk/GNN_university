"""Idempotent download of the three SNAP Reddit Hyperlink files.

Authoritative URLs:
    https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
    https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv
    http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv

The embeddings file is the one URL SNAP currently serves over plain http; the
two TSVs are https. The constants below mirror those URLs verbatim.

:func:`ensure_raw_files` is idempotent: if the local file's size matches the
remote ``Content-Length`` (HEAD), it skips the download. Otherwise it streams
with a tqdm progress bar and verifies the result is non-empty.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)

BODY_FILENAME = "soc-redditHyperlinks-body.tsv"
TITLE_FILENAME = "soc-redditHyperlinks-title.tsv"
EMBEDDINGS_FILENAME = "web-redditEmbeddings-subreddits.csv"

REDDIT_HYPERLINK_URLS: dict[str, str] = {
    BODY_FILENAME: "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    TITLE_FILENAME: "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
    EMBEDDINGS_FILENAME: "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv",
}

_USER_AGENT = "reddit-gnn/0.1 (+https://github.com/oleksandr-shchepanchuk/reddit-hyperlink-gnn)"
_CHUNK_BYTES = 65_536
_HEAD_TIMEOUT_S = 30


def _remote_size(url: str) -> int | None:
    """Return the remote ``Content-Length`` via HEAD, or ``None`` on failure."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HEAD_TIMEOUT_S) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.debug("HEAD %s failed (%s); will skip size check", url, exc)
        return None


def _download_url(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a tqdm progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0) or None
        with (
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as pbar,
            open(tmp, "wb") as out,
        ):
            while True:
                buf = resp.read(_CHUNK_BYTES)
                if not buf:
                    break
                out.write(buf)
                pbar.update(len(buf))
    shutil.move(tmp, dest)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {dest}")


def _should_skip(dest: Path, remote_size: int | None) -> bool:
    if not dest.exists():
        return False
    local_size = dest.stat().st_size
    if local_size == 0:
        return False
    if remote_size is None:
        log.info("Reusing existing %s (remote size unknown)", dest.name)
        return True
    if local_size == remote_size:
        log.info("Reusing existing %s (size matches remote)", dest.name)
        return True
    log.info(
        "Re-downloading %s (local=%d, remote=%d)", dest.name, local_size, remote_size
    )
    return False


def ensure_raw_files(raw_dir: str | Path) -> dict[str, Path]:
    """Download (idempotently) all three SNAP files into ``raw_dir``.

    Returns a mapping ``filename -> Path`` for the three files. If a file is
    already present and its size matches the remote ``Content-Length`` it is
    reused without re-downloading. When the HEAD request fails (server quirks,
    no network), any existing non-empty local file is also reused.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for fname, url in REDDIT_HYPERLINK_URLS.items():
        dest = raw / fname
        remote = _remote_size(url)
        if _should_skip(dest, remote):
            out[fname] = dest
            continue
        log.info("Downloading %s -> %s", url, dest)
        _download_url(url, dest)
        out[fname] = dest
    return out


__all__ = [
    "BODY_FILENAME",
    "EMBEDDINGS_FILENAME",
    "REDDIT_HYPERLINK_URLS",
    "TITLE_FILENAME",
    "ensure_raw_files",
]
