"""Preprocess the raw SNAP edge frame into a clean, indexed edge table.

Pipeline:
    1. :func:`clean_edges` — drop self-loops + duplicates, remap the SNAP
       label (``-1 -> 0`` negative, ``+1 -> 1`` neutral/positive) into
       ``label_binary`` (int8), tag ``is_title``, sort by timestamp.
    2. :func:`build_node_mapping` — assign contiguous integer ids to every
       unique subreddit found in the union of source + target columns;
       attach ``source_id`` / ``target_id`` int64 columns to the frame.
    3. :func:`save_processed_dataset` — persist parquet + JSON sidecars to
       ``data/processed/``.
    4. :func:`preprocess_dataset` — the full pipeline (optional debug
       sub-sampling), returning the path to the written parquet.

Edge sign classification reminder: we never sample non-edges and never treat
``label_binary == 0`` as a non-edge. Every row here is a real, observed
hyperlink whose sentiment label is the prediction target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reddit_gnn.data.load import load_reddit_dataset
from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)

# Columns that uniquely identify a hyperlink event for deduplication.
_DEDUPE_KEY = ("source_subreddit_norm", "target_subreddit_norm", "TIMESTAMP", "POST_ID")


def remap_post_label(label: int) -> int:
    """Pure scalar remap: ``-1 -> 0`` (negative), ``+1 -> 1`` (neutral/positive).

    Raises ``ValueError`` for any other value. We never collapse to a
    "non-edge" class — observed hyperlinks are all real edges.
    """
    if label == -1:
        return 0
    if label == 1:
        return 1
    raise ValueError(f"Unexpected POST_LABEL {label!r}; expected -1 or +1")


def clean_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Drop self-loops + duplicates, remap labels, tag ``is_title``, sort by time.

    Input is expected to be the frame produced by
    :func:`reddit_gnn.data.load.load_reddit_dataset`, i.e. with normalized
    subreddit names, parsed ``TIMESTAMP`` (``datetime64[ns]``), raw
    ``POST_LABEL`` in ``{-1, +1}``, and a ``source_file`` tag.

    Output:
        * ``label_binary`` (int8) — remapped POST_LABEL.
        * ``is_title`` (int8) — 1 iff the row came from the *title* TSV.
        * ``POST_LABEL`` removed (binary version supersedes it).
        * No self-loops.
        * No duplicate ``(src, dst, TIMESTAMP, POST_ID)`` rows.
        * Sorted by ``TIMESTAMP`` ascending; index reset.
    """
    if df.empty:
        return df.copy()

    required = {
        "source_subreddit_norm",
        "target_subreddit_norm",
        "TIMESTAMP",
        "POST_LABEL",
        "POST_ID",
        "source_file",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clean_edges: input frame is missing columns: {sorted(missing)}")

    out = df.copy()
    n_in = len(out)

    self_loop_mask = out["source_subreddit_norm"] == out["target_subreddit_norm"]
    if self_loop_mask.any():
        n_loops = int(self_loop_mask.sum())
        log.info("clean_edges: dropping %d self-loop edge(s)", n_loops)
        out = out.loc[~self_loop_mask]

    # Label remap: -1 -> 0, +1 -> 1; anything else is dropped (we never invent
    # a class). Pandas Int64 is nullable, so coerce-then-map handles NaN cleanly.
    raw_label = pd.to_numeric(out["POST_LABEL"], errors="coerce")
    label_map = {-1: 0, 1: 1}
    label_binary = raw_label.map(label_map)
    unknown_mask = label_binary.isna()
    if unknown_mask.any():
        n_unknown = int(unknown_mask.sum())
        log.warning(
            "clean_edges: dropping %d row(s) with POST_LABEL not in {-1, +1}", n_unknown
        )
        out = out.loc[~unknown_mask]
        label_binary = label_binary.loc[~unknown_mask]
    out = out.assign(label_binary=label_binary.astype("int8")).drop(columns=["POST_LABEL"])

    before_dedup = len(out)
    out = out.drop_duplicates(subset=list(_DEDUPE_KEY), keep="first")
    n_dupes = before_dedup - len(out)
    if n_dupes:
        log.info("clean_edges: dropped %d duplicate row(s)", n_dupes)

    out["is_title"] = (out["source_file"] == "title").astype("int8")

    out = out.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)

    log.info("clean_edges: %d -> %d row(s) after cleaning", n_in, len(out))
    return out


def build_node_mapping(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """Assign contiguous int ids to every unique subreddit name.

    Returns ``(df_with_ids, node_to_id, id_to_node)``:
        * ``df_with_ids`` has new int64 columns ``source_id`` / ``target_id``;
        * ``node_to_id`` maps a normalized subreddit name to its integer id;
        * ``id_to_node`` is the inverse mapping.

    Ids are assigned in sorted order of names so that the mapping is
    deterministic across runs.
    """
    if df.empty:
        return df.copy(), {}, {}

    required = {"source_subreddit_norm", "target_subreddit_norm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_node_mapping: input frame is missing columns: {sorted(missing)}"
        )

    names = pd.concat(
        [df["source_subreddit_norm"], df["target_subreddit_norm"]],
        ignore_index=True,
    ).dropna().unique()
    names = sorted(str(n) for n in names if str(n) != "")

    node_to_id: dict[str, int] = {name: i for i, name in enumerate(names)}
    id_to_node: dict[int, str] = {i: name for name, i in node_to_id.items()}

    out = df.copy()
    out["source_id"] = out["source_subreddit_norm"].map(node_to_id).astype("int64")
    out["target_id"] = out["target_subreddit_norm"].map(node_to_id).astype("int64")

    return out, node_to_id, id_to_node


def _label_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label_binary"].value_counts().to_dict()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _timestamp_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty:
        return {"min": None, "max": None}
    ts_min = df["TIMESTAMP"].min()
    ts_max = df["TIMESTAMP"].max()
    return {
        "min": ts_min.isoformat() if pd.notna(ts_min) else None,
        "max": ts_max.isoformat() if pd.notna(ts_max) else None,
    }


def save_processed_dataset(
    df: pd.DataFrame,
    node_to_id: dict[str, int],
    id_to_node: dict[int, str],
    processed_dir: str | Path,
    *,
    network_type: str = "both",
) -> dict[str, Path]:
    """Write the processed dataset to disk and return the artifact paths.

    Artifacts in ``processed_dir/``:
        * ``edges.parquet`` — the cleaned, indexed edge frame.
        * ``node_to_id.json`` — ``{subreddit_name: int}`` mapping.
        * ``id_to_node.json`` — ``{stringified_int: subreddit_name}`` mapping.
          JSON only allows string keys, so the caller must cast keys back to
          ``int`` after loading.
        * ``metadata.json`` — summary stats for sanity checks.
    """
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    edges_path = processed / "edges.parquet"
    node_to_id_path = processed / "node_to_id.json"
    id_to_node_path = processed / "id_to_node.json"
    metadata_path = processed / "metadata.json"

    df.to_parquet(edges_path, index=False)

    with node_to_id_path.open("w", encoding="utf-8") as f:
        json.dump(node_to_id, f, ensure_ascii=False, indent=2, sort_keys=True)

    serializable_id_to_node = {str(int(k)): v for k, v in id_to_node.items()}
    with id_to_node_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_id_to_node, f, ensure_ascii=False, indent=2)

    metadata = {
        "num_nodes": len(node_to_id),
        "num_edges": int(len(df)),
        "label_distribution": _label_distribution(df) if not df.empty else {},
        "timestamp_range": _timestamp_range(df),
        "network_type": network_type,
        "is_title_distribution": (
            {str(int(k)): int(v) for k, v in df["is_title"].value_counts().to_dict().items()}
            if not df.empty
            else {}
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)

    log.info(
        "Saved processed dataset to %s (%d edges, %d nodes)",
        processed,
        metadata["num_edges"],
        metadata["num_nodes"],
    )
    return {
        "edges": edges_path,
        "node_to_id": node_to_id_path,
        "id_to_node": id_to_node_path,
        "metadata": metadata_path,
    }


def preprocess_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    network_type: str = "both",
    *,
    df: pd.DataFrame | None = None,
    sample_edges: int | None = None,
    seed: int = 42,
) -> Path:
    """Full preprocessing pipeline. Returns the path to ``edges.parquet``.

    Pass a pre-loaded ``df`` to avoid re-parsing the SNAP TSVs (the scripts/
    CLI loads once, prints a preview, then hands the frame here).

    ``sample_edges`` is for **debugging only**; if set, a uniform random subset
    is taken from the cleaned frame *before* the node mapping is built. A
    prominent warning is logged whenever this is active — the resulting parquet
    must not be used for any reported metric.
    """
    if df is None:
        df = load_reddit_dataset(raw_dir, network_type=network_type)

    cleaned = clean_edges(df)

    if sample_edges is not None and sample_edges > 0 and sample_edges < len(cleaned):
        log.warning(
            "DEBUG ONLY: sampling %d of %d cleaned edges (seed=%d). "
            "DO NOT USE the resulting dataset for the final report.",
            sample_edges,
            len(cleaned),
            seed,
        )
        cleaned = (
            cleaned.sample(n=sample_edges, random_state=seed)
            .sort_values("TIMESTAMP", kind="mergesort")
            .reset_index(drop=True)
        )

    mapped, node_to_id, id_to_node = build_node_mapping(cleaned)
    artifacts = save_processed_dataset(
        mapped,
        node_to_id,
        id_to_node,
        processed_dir,
        network_type=network_type,
    )
    return artifacts["edges"]


__all__ = [
    "build_node_mapping",
    "clean_edges",
    "preprocess_dataset",
    "remap_post_label",
    "save_processed_dataset",
]
