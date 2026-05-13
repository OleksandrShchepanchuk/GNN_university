"""Node and edge feature engineering with **train-only** scaler fitting.

The single most important invariant in this module is that every statistic
which gets fit (sklearn ``StandardScaler``, temporal min/max for normalization,
node feature aggregation) is computed from training rows only. Validation and
test rows may flow through :meth:`FeatureBuilder.transform_node_features` /
:meth:`FeatureBuilder.transform_edge_features`, but never through ``fit``.

The standalone functions exposed here are pure transformations — they don't
fit anything on their own. The :class:`FeatureBuilder` is the only object
that owns fitted state.

Output dimensionality (with all toggles on):
    node features  = 300 (SNAP) + 1 (unknown flag) + 11 (structural, scaled)
                     + 172 (aggregated edge attr) = 484.
    edge features  = 86 (raw POST_PROPERTIES) + 1 (is_title, scaled)
                     + 6 (temporal, scaled) = 93.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)

POST_PROPERTIES_DIM = 86
SNAP_EMBED_DIM = 300
STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_degree",
    "positive_in_degree",
    "negative_in_degree",
    "positive_out_degree",
    "negative_out_degree",
    "negative_in_ratio",
    "negative_out_ratio",
    "log_in_degree",
    "log_out_degree",
)
EDGE_TEMPORAL_COLUMNS: tuple[str, ...] = (
    "is_title",
    "year_norm",
    "month_sin",
    "month_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "days_since_min_norm",
)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def parse_post_properties(df: pd.DataFrame) -> np.ndarray:
    """Return ``df[p0..p85]`` as a ``(num_edges, 86)`` float32 array.

    Raises ``KeyError`` if any of the expected columns is missing.
    """
    cols = [f"p{i}" for i in range(POST_PROPERTIES_DIM)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"parse_post_properties: missing {len(missing)} POST_PROPERTIES "
            f"columns (first few: {missing[:5]})"
        )
    return df[cols].to_numpy(dtype=np.float32)


def create_structural_node_features(
    df: pd.DataFrame, num_nodes: int
) -> pd.DataFrame:
    """Compute per-node structural features from the provided edge frame.

    The caller controls which edges feed in — pass a train-only frame at fit
    time, train+val for validation-time inference, etc. This function never
    fits anything; it just aggregates.

    Columns (in :data:`STRUCTURAL_COLUMNS` order):
        in_degree, out_degree, total_degree, positive_in_degree,
        negative_in_degree, positive_out_degree, negative_out_degree,
        negative_in_ratio (safe div, 0 when denom = 0), negative_out_ratio,
        log_in_degree (log1p), log_out_degree.
    """
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")

    if df.empty:
        zeros = np.zeros(num_nodes, dtype=np.float32)
        return pd.DataFrame({c: zeros for c in STRUCTURAL_COLUMNS})

    src = df["source_id"].to_numpy()
    dst = df["target_id"].to_numpy()
    labels = df["label_binary"].astype(np.int8).to_numpy()
    pos_mask = labels == 1
    neg_mask = labels == 0

    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    pos_out = np.bincount(src[pos_mask], minlength=num_nodes).astype(np.float32)
    neg_out = np.bincount(src[neg_mask], minlength=num_nodes).astype(np.float32)
    pos_in = np.bincount(dst[pos_mask], minlength=num_nodes).astype(np.float32)
    neg_in = np.bincount(dst[neg_mask], minlength=num_nodes).astype(np.float32)

    in_total = pos_in + neg_in
    out_total = pos_out + neg_out
    neg_in_ratio = np.where(in_total > 0, neg_in / np.maximum(in_total, 1), 0.0).astype(
        np.float32
    )
    neg_out_ratio = np.where(out_total > 0, neg_out / np.maximum(out_total, 1), 0.0).astype(
        np.float32
    )

    return pd.DataFrame(
        {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": in_deg + out_deg,
            "positive_in_degree": pos_in,
            "negative_in_degree": neg_in,
            "positive_out_degree": pos_out,
            "negative_out_degree": neg_out,
            "negative_in_ratio": neg_in_ratio,
            "negative_out_ratio": neg_out_ratio,
            "log_in_degree": np.log1p(in_deg).astype(np.float32),
            "log_out_degree": np.log1p(out_deg).astype(np.float32),
        }
    )


def create_aggregated_edge_property_node_features(
    df: pd.DataFrame,
    edge_attr: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Average POST_PROPERTIES over incoming and outgoing edges per node.

    Returns a ``(num_nodes, 2*86)`` array: ``[mean_incoming || mean_outgoing]``.
    Nodes with no incoming (resp. outgoing) edges get zeros for the
    corresponding 86-D block.
    """
    if edge_attr.shape[1] != POST_PROPERTIES_DIM:
        raise ValueError(
            f"edge_attr must have {POST_PROPERTIES_DIM} columns; got {edge_attr.shape[1]}"
        )
    if len(df) != edge_attr.shape[0]:
        raise ValueError(
            f"len(df)={len(df)} != edge_attr rows={edge_attr.shape[0]}"
        )

    sum_in = np.zeros((num_nodes, POST_PROPERTIES_DIM), dtype=np.float32)
    sum_out = np.zeros((num_nodes, POST_PROPERTIES_DIM), dtype=np.float32)
    count_in = np.zeros(num_nodes, dtype=np.int64)
    count_out = np.zeros(num_nodes, dtype=np.int64)

    if not df.empty:
        attr = edge_attr.astype(np.float32, copy=False)
        src = df["source_id"].to_numpy()
        dst = df["target_id"].to_numpy()
        np.add.at(sum_in, dst, attr)
        np.add.at(sum_out, src, attr)
        np.add.at(count_in, dst, 1)
        np.add.at(count_out, src, 1)

    denom_in = np.maximum(count_in, 1)[:, None].astype(np.float32)
    denom_out = np.maximum(count_out, 1)[:, None].astype(np.float32)
    mean_in = np.where(count_in[:, None] > 0, sum_in / denom_in, 0.0).astype(np.float32)
    mean_out = np.where(count_out[:, None] > 0, sum_out / denom_out, 0.0).astype(np.float32)

    return np.concatenate([mean_in, mean_out], axis=1)


def load_snap_subreddit_embeddings(
    node_to_id: dict[str, int],
    embeddings_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up the SNAP LIWC embedding for every node id.

    Returns ``(X_emb, has_embedding_indicator)``:
        * ``X_emb`` of shape ``(num_nodes, 300)`` — SNAP vector when available,
          all-zero otherwise.
        * ``has_embedding_indicator`` of shape ``(num_nodes, 1)`` — ``1.0``
          when the subreddit has a SNAP embedding, ``0.0`` otherwise. The
          FeatureBuilder converts this into the explicit ``unknown_flag``
          column (``1 - has``) before handing it to the model so the network
          can learn a separate behavior for unmatched nodes.
    """
    from reddit_gnn.data.load import parse_subreddit_embeddings

    name_to_idx, emb_matrix = parse_subreddit_embeddings(embeddings_path)
    if emb_matrix.shape[1] != SNAP_EMBED_DIM:
        raise ValueError(
            f"SNAP embeddings file must have {SNAP_EMBED_DIM} dims; got {emb_matrix.shape[1]}"
        )

    if not node_to_id:
        return (
            np.zeros((0, SNAP_EMBED_DIM), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        )

    num_nodes = max(node_to_id.values()) + 1
    X_emb = np.zeros((num_nodes, SNAP_EMBED_DIM), dtype=np.float32)
    has = np.zeros((num_nodes, 1), dtype=np.float32)
    for name, node_id in node_to_id.items():
        norm = str(name).strip().lower()
        if norm in name_to_idx:
            X_emb[node_id] = emb_matrix[name_to_idx[norm]]
            has[node_id, 0] = 1.0
    return X_emb, has


# ---------------------------------------------------------------------------
# Edge temporal helpers (shared between build_edge_features + FeatureBuilder)
# ---------------------------------------------------------------------------


def _compute_temporal_columns(
    df: pd.DataFrame,
    *,
    time_min: pd.Timestamp,
    time_max: pd.Timestamp,
    year_min: int,
    year_max: int,
) -> np.ndarray:
    """Compute the 7-column ``EDGE_TEMPORAL_COLUMNS`` matrix using fixed bounds.

    The bounds are kwargs so the FeatureBuilder can pass *train-only* min/max
    to keep val/test transforms leakage-free.
    """
    ts = pd.to_datetime(df["TIMESTAMP"])
    is_title = df["is_title"].astype(np.float32).to_numpy()

    if year_max == year_min:
        year_norm = np.zeros(len(df), dtype=np.float32)
    else:
        year_norm = ((ts.dt.year - year_min) / (year_max - year_min)).to_numpy(dtype=np.float32)
        year_norm = np.clip(year_norm, 0.0, 1.0)

    month = ts.dt.month.to_numpy(dtype=np.float32)
    dow = ts.dt.dayofweek.to_numpy(dtype=np.float32)
    month_sin = np.sin(2 * np.pi * month / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * month / 12).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    span_days = max((time_max - time_min) / np.timedelta64(1, "D"), 1.0)
    days_since = ((ts - time_min) / np.timedelta64(1, "D")).to_numpy(dtype=np.float32)
    days_since_norm = np.clip(days_since / span_days, 0.0, 1.0)

    out = np.column_stack(
        [
            is_title,
            year_norm,
            month_sin,
            month_cos,
            dow_sin,
            dow_cos,
            days_since_norm,
        ]
    ).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_edge_features(
    df: pd.DataFrame,
    edge_attr_train_fit_scaler: StandardScaler | None = None,
    *,
    time_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    year_bounds: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Standalone edge-feature builder. Returns ``[N, 93]`` float32 tensor.

    * Raw POST_PROPERTIES (86 columns) are pass-through — SNAP normalizes them
      already and the user spec explicitly says they are scale-free.
    * ``is_title`` + 6 temporal columns are stacked next. If
      ``edge_attr_train_fit_scaler`` is provided (the train-fitted scaler),
      these 7 columns are standardized before concatenation.
    * ``time_bounds`` / ``year_bounds`` are optional kwargs; when ``None``
      the function falls back to ``df``'s own min/max. The FeatureBuilder
      always passes the training bounds at transform time, so val/test
      transforms never leak.
    """
    props = parse_post_properties(df)

    ts = pd.to_datetime(df["TIMESTAMP"])
    t_min, t_max = time_bounds if time_bounds is not None else (ts.min(), ts.max())
    y_min, y_max = (
        year_bounds
        if year_bounds is not None
        else (int(ts.dt.year.min()), int(ts.dt.year.max()))
    )

    temporal = _compute_temporal_columns(
        df, time_min=t_min, time_max=t_max, year_min=y_min, year_max=y_max,
    )
    if edge_attr_train_fit_scaler is not None:
        temporal = edge_attr_train_fit_scaler.transform(temporal).astype(np.float32)

    combined = np.concatenate([props, temporal], axis=1).astype(np.float32)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(combined)


# ---------------------------------------------------------------------------
# FeatureBuilder (the one place fitted state lives)
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Fit-on-train, transform-on-anything feature builder.

    Toggles select which feature blocks are computed. Anything that requires a
    fitted statistic is stored on the instance and reused at transform time;
    nothing is ever re-fit on val/test rows.
    """

    def __init__(
        self,
        use_structural: bool = True,
        use_aggregated_edge_attr: bool = True,
        use_snap_embeddings: bool = True,
    ) -> None:
        self.use_structural = use_structural
        self.use_aggregated_edge_attr = use_aggregated_edge_attr
        self.use_snap_embeddings = use_snap_embeddings

        # Filled by fit()
        self.structural_scaler: StandardScaler | None = None
        self.edge_temporal_scaler: StandardScaler | None = None
        self.time_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.year_bounds: tuple[int, int] | None = None
        self.snap_embeddings: np.ndarray | None = None
        self.has_embedding_indicator: np.ndarray | None = None
        self.num_nodes: int = 0
        self._is_fit: bool = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df_train: pd.DataFrame,
        num_nodes: int,
        embeddings_path: str | Path | None = None,
    ) -> "FeatureBuilder":
        """Fit scalers on training rows. Never sees val/test."""
        self.num_nodes = int(num_nodes)

        # (a) Structural scaler — train-only edges feed the per-node degrees.
        if self.use_structural:
            struct = create_structural_node_features(df_train, num_nodes)
            scaler = StandardScaler()
            scaler.fit(struct[list(STRUCTURAL_COLUMNS)].to_numpy(dtype=np.float64))
            self.structural_scaler = scaler

        # (b) Edge temporal scaler — bounds come from train timestamps only.
        ts = pd.to_datetime(df_train["TIMESTAMP"])
        self.time_bounds = (ts.min(), ts.max())
        self.year_bounds = (int(ts.dt.year.min()), int(ts.dt.year.max()))
        temporal_train = _compute_temporal_columns(
            df_train,
            time_min=self.time_bounds[0],
            time_max=self.time_bounds[1],
            year_min=self.year_bounds[0],
            year_max=self.year_bounds[1],
        )
        edge_scaler = StandardScaler()
        edge_scaler.fit(temporal_train.astype(np.float64))
        self.edge_temporal_scaler = edge_scaler

        # (c) SNAP embeddings — looked up once and cached. We need node_to_id;
        # since the processed parquet's source_id / target_id are contiguous
        # over normalized names, the caller passes embeddings_path and we
        # reconstruct node_to_id from df_train's columns. Callers that already
        # have a node_to_id can call load_snap_subreddit_embeddings directly
        # and then attach the result through `set_snap_embeddings`.
        if self.use_snap_embeddings and embeddings_path is not None:
            node_to_id = _node_to_id_from_df(df_train, num_nodes)
            X_emb, has = load_snap_subreddit_embeddings(node_to_id, embeddings_path)
            self.snap_embeddings = X_emb
            self.has_embedding_indicator = has

        self._is_fit = True
        return self

    def set_snap_embeddings(
        self,
        X_emb: np.ndarray,
        has_embedding_indicator: np.ndarray,
    ) -> "FeatureBuilder":
        """Attach pre-computed SNAP embeddings (for callers with their own loader)."""
        if X_emb.shape[1] != SNAP_EMBED_DIM:
            raise ValueError(f"X_emb must have {SNAP_EMBED_DIM} columns")
        if has_embedding_indicator.shape != (X_emb.shape[0], 1):
            raise ValueError("has_embedding_indicator must be (num_nodes, 1)")
        self.snap_embeddings = X_emb.astype(np.float32)
        self.has_embedding_indicator = has_embedding_indicator.astype(np.float32)
        return self

    # ------------------------------------------------------------- transform
    def _ensure_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("FeatureBuilder has not been fit yet; call fit() first")

    def transform_node_features(
        self,
        df_subset: pd.DataFrame,
        num_nodes: int,
    ) -> torch.Tensor:
        """Concatenate the enabled per-node feature blocks into one tensor.

        The order is fixed: ``[snap_emb (300), unknown_flag (1),
        structural (11, scaled), aggregated_edge_attr (172)]``.
        Disabled blocks are simply dropped from the concatenation.
        """
        self._ensure_fit()
        blocks: list[np.ndarray] = []

        if self.use_snap_embeddings:
            X_emb = self.snap_embeddings
            has = self.has_embedding_indicator
            if X_emb is None or has is None:
                X_emb = np.zeros((num_nodes, SNAP_EMBED_DIM), dtype=np.float32)
                has = np.zeros((num_nodes, 1), dtype=np.float32)
            X_emb = _resize_node_matrix(X_emb, num_nodes)
            has = _resize_node_matrix(has, num_nodes)
            unknown_flag = (1.0 - has).astype(np.float32)
            blocks.extend([X_emb, unknown_flag])

        if self.use_structural:
            if self.structural_scaler is None:
                raise RuntimeError("structural_scaler missing; ensure use_structural was on at fit")
            struct = create_structural_node_features(df_subset, num_nodes)
            scaled = self.structural_scaler.transform(
                struct[list(STRUCTURAL_COLUMNS)].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            blocks.append(scaled)

        if self.use_aggregated_edge_attr:
            edge_attr = parse_post_properties(df_subset)
            agg = create_aggregated_edge_property_node_features(
                df_subset, edge_attr, num_nodes
            )
            blocks.append(agg)

        if not blocks:
            raise ValueError("FeatureBuilder has no enabled feature blocks; nothing to transform")

        out = np.concatenate(blocks, axis=1).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(out)

    def transform_edge_features(self, df_edges: pd.DataFrame) -> torch.Tensor:
        """Leakage-free edge feature transform using the fitted scaler + bounds."""
        self._ensure_fit()
        if self.edge_temporal_scaler is None or self.time_bounds is None or self.year_bounds is None:
            raise RuntimeError("edge_temporal_scaler / bounds missing; call fit() first")
        return build_edge_features(
            df_edges,
            edge_attr_train_fit_scaler=self.edge_temporal_scaler,
            time_bounds=self.time_bounds,
            year_bounds=self.year_bounds,
        )

    # --------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBuilder":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a FeatureBuilder; got {type(obj)}")
        return obj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_to_id_from_df(df: pd.DataFrame, num_nodes: int) -> dict[str, int]:
    """Reconstruct ``{name: id}`` from a frame that has ``*_subreddit_norm``
    and ``source_id`` / ``target_id`` columns.

    Falls back to ``str(int_id)`` for any node id that doesn't appear in the
    frame, so the SNAP lookup still aligns by index (those nodes will simply
    get the unknown flag).
    """
    out: dict[str, int] = {}
    if {"source_subreddit_norm", "source_id"}.issubset(df.columns):
        out.update(
            df[["source_subreddit_norm", "source_id"]]
            .drop_duplicates(subset=["source_subreddit_norm"])
            .set_index("source_subreddit_norm")["source_id"]
            .astype(int)
            .to_dict()
        )
    if {"target_subreddit_norm", "target_id"}.issubset(df.columns):
        out.update(
            df[["target_subreddit_norm", "target_id"]]
            .drop_duplicates(subset=["target_subreddit_norm"])
            .set_index("target_subreddit_norm")["target_id"]
            .astype(int)
            .to_dict()
        )
    return out


def _resize_node_matrix(mat: np.ndarray, num_nodes: int) -> np.ndarray:
    """Pad or truncate a ``(?, d)`` matrix to exactly ``(num_nodes, d)`` rows."""
    if mat.shape[0] == num_nodes:
        return mat
    if mat.shape[0] > num_nodes:
        return mat[:num_nodes]
    pad = np.zeros((num_nodes - mat.shape[0], mat.shape[1]), dtype=mat.dtype)
    return np.concatenate([mat, pad], axis=0)


__all__ = [
    "EDGE_TEMPORAL_COLUMNS",
    "POST_PROPERTIES_DIM",
    "SNAP_EMBED_DIM",
    "STRUCTURAL_COLUMNS",
    "FeatureBuilder",
    "build_edge_features",
    "create_aggregated_edge_property_node_features",
    "create_structural_node_features",
    "load_snap_subreddit_embeddings",
    "parse_post_properties",
]
