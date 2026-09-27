"""
Candidate Generation Orchestrator & Integration Layer
======================================================

Public Entry Point
------------------
generate_candidates(s1_df, s2_df, s3_df, config=None) -> pd.DataFrame

Orchestrates all 7 candidate generation / blocking routes:
    1. Route 1: exact_name
    2. Route 2: tfidf_name
    3. Route 3: rare_token_name
    4. Route 4: tfidf_address
    5. Route 5: numeric_address
    6. Route 6: rare_token_address
    7. Route 7: reverse_retrieval

Canonical Output Schema
-----------------------
[pair_key, s1_id, candidate_id, candidate_source, route, rank, score]

Invariants & Contracts
----------------------
1. pair_key = s1_id + "::" + candidate_id
2. Deduplicated: unique per (s1_id, candidate_id)
3. candidate_source in {"S2", "S3"}
4. Deterministic ordering: grouped by s1_id, sorted by score desc, candidate_id asc
5. Ranks are 1-based integers computed on the final union set
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .exact_name import retrieve_exact_name
from .tfidf_name import (
    DEFAULT_CANDIDATE_BLOCK as TFIDF_NAME_CANDIDATE_BLOCK,
    DEFAULT_QUERY_BLOCK as TFIDF_QUERY_BLOCK,
    retrieve_tfidf_name,
)
from .rare_token import retrieve_rare_token_name, retrieve_rare_token_address
from .tfidf_address import retrieve_tfidf_address
from .numeric import retrieve_numeric
from .reverse import (
    DEFAULT_DEDUP_BUCKETS,
    DEFAULT_INDEX_BLOCK as REVERSE_INDEX_BLOCK,
    retrieve_reverse,
)
from .spill import (
    CANONICAL_CANDIDATE_COLUMNS,
    DEFAULT_RECORD_BUFFER,
    EventSpillWriter,
    RecordBuffer,
    SpillManifest,
    deduplicate_frame,
    iter_canonical_candidates_merged,
    iter_spilled_buckets,
    resolve_spill_dir,
    write_candidate_dataset,
    write_events_stream,
)


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------
DEFAULT_BLOCKING_CONFIG: Dict[str, Any] = {
    # Route 1: Exact Name
    "exact_name_max_block": 500,
    # Route 2: TF-IDF Name
    "tfidf_name_top_k": 20,
    "tfidf_name_chunk_size": 5000,
    "tfidf_name_candidate_chunk": TFIDF_NAME_CANDIDATE_BLOCK,
    # Route 3: Rare-Token Name
    "rare_name_top_k": 20,
    "rare_name_max_block": 500,
    # Route 4: TF-IDF Address
    "tfidf_addr_top_k": 20,
    "tfidf_addr_chunk_size": 5000,
    "tfidf_addr_candidate_chunk": TFIDF_NAME_CANDIDATE_BLOCK,
    # Shared 2D query block for both TF-IDF routes (see tfidf_name.DEFAULT_QUERY_BLOCK)
    "tfidf_query_block": TFIDF_QUERY_BLOCK,
    # Route 5: Numeric Address
    "numeric_top_k": 20,
    "numeric_max_block": 500,
    # Route 6: Rare-Token Address
    "rare_addr_top_k": 20,
    "rare_addr_max_block": 500,
    # Route 7: Reverse Retrieval
    "reverse_top_k": 5,
    "reverse_chunk_size": 5000,
    "reverse_index_block": REVERSE_INDEX_BLOCK,
    # Bounded-memory spill (L4). Set "spill_dir" to enable; when None the previous
    # in-memory concat+dedup path is used, which is only safe for sample-sized inputs.
    "spill_dir": None,
    "dedup_buckets": DEFAULT_DEDUP_BUCKETS,
    # Streaming controls. "record_buffer" is the hard cap on buffered event records between
    # flushes (owner-fixed at 50_000). "spill_output_path"/"spill_events_path" make the
    # finalization streaming; without them the in-memory aggregation path is used, which is
    # only safe for sample-sized inputs.
    "record_buffer": DEFAULT_RECORD_BUFFER,
    "spill_output_path": None,
    "spill_events_path": None,
    "reset_spill_dir": False,
    # Enabled routes
    "enabled_routes": [
        "exact_name",
        "tfidf_name",
        "rare_token_name",
        "tfidf_address",
        "numeric_address",
        "rare_token_address",
        "reverse_retrieval",
    ],
}


# ---------------------------------------------------------------------------
# Core Orchestration Function
# ---------------------------------------------------------------------------
def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    addr_col: str = "business_address",
    return_events: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate and union candidates from all 7 blocking routes.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, name_col, addr_col].
    config              : Optional dict overriding blocking parameters.
    id_col              : Entity ID column name (default 'entity_id').
    name_col            : Business name column name (default 'business_name').
    addr_col            : Business address column name (default 'business_address').
    return_events       : If True, returns (candidates_df, retrieval_events_df).

    Returns
    -------
    candidates_df : Deduplicated canonical candidate DataFrame with columns:
                    pair_key, s1_id, candidate_id, candidate_source,
                    route, rank, score
    """
    cfg = {**DEFAULT_BLOCKING_CONFIG, **(config or {})}
    enabled = set(cfg.get("enabled_routes", DEFAULT_BLOCKING_CONFIG["enabled_routes"]))

    # L4: when a spill directory is configured, every route's output is streamed into
    # hash-bucketed Parquet shards and the aggregation runs one bucket at a time. Without
    # it, the historical in-memory path is preserved for sample-sized inputs and tests.
    spill_root = resolve_spill_dir(cfg.get("spill_dir"), reset=bool(cfg.get("reset_spill_dir", False)))
    n_buckets = int(cfg["dedup_buckets"])
    writer: Optional[EventSpillWriter] = None
    if spill_root is not None:
        writer = EventSpillWriter(spill_root, n_buckets=n_buckets)

    def _collect(frame: pd.DataFrame, sink: Optional[object] = None) -> None:
        """Route result -> spill, or -> the in-memory accumulator."""
        if frame is None or len(frame) == 0:
            return
        if writer is not None:
            writer.write_frame(frame)
        else:
            route_dfs.append(frame)

    route_dfs: List[pd.DataFrame] = []

    # -----------------------------------------------------------------------
    # Bounded-memory path: a single RecordBuffer is handed to every route, so
    # each one streams its records straight into the spill instead of building
    # an unbounded Python list first. This is the change that removes the
    # ~11.9 GB (Route 1) / ~24 GB (Routes 2-6) accumulations.
    # -----------------------------------------------------------------------
    stream_buf: Optional[RecordBuffer] = None
    if writer is not None:
        stream_buf = RecordBuffer(
            writer, max_records=int(cfg.get("record_buffer", DEFAULT_RECORD_BUFFER))
        )

    # 1. Route 1: Exact Normalized-Name Blocking
    if "exact_name" in enabled:
        df1 = retrieve_exact_name(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            name_col=name_col,
            max_block_size=cfg["exact_name_max_block"],
            sink=stream_buf,
        )
        _collect(df1)
        del df1

    # 2. Route 2: Name Char N-Gram TF-IDF Retrieval (2D blocked; L2/L3)
    if "tfidf_name" in enabled:
        df2 = retrieve_tfidf_name(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            name_col=name_col,
            top_k=cfg["tfidf_name_top_k"],
            chunk_size=cfg["tfidf_name_chunk_size"],
            query_block=cfg["tfidf_query_block"],
            candidate_block=cfg["tfidf_name_candidate_chunk"],
            sink=stream_buf,
        )
        _collect(df2)
        del df2

    # 3. Route 3: Rare-Token Name Retrieval
    if "rare_token_name" in enabled:
        df3 = retrieve_rare_token_name(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            name_col=name_col,
            top_k=cfg["rare_name_top_k"],
            max_block_size=cfg["rare_name_max_block"],
            sink=stream_buf,
        )
        _collect(df3)
        del df3

    # 4. Route 4: Address Char N-Gram TF-IDF Retrieval (2D blocked; L2/L3)
    if "tfidf_address" in enabled:
        df4 = retrieve_tfidf_address(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            addr_col=addr_col,
            top_k=cfg["tfidf_addr_top_k"],
            chunk_size=cfg["tfidf_addr_chunk_size"],
            query_block=cfg["tfidf_query_block"],
            candidate_block=cfg["tfidf_addr_candidate_chunk"],
            sink=stream_buf,
        )
        _collect(df4)
        del df4

    # 5. Route 5: Numeric / Address Token Retrieval
    if "numeric_address" in enabled:
        df5 = retrieve_numeric(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            addr_col=addr_col,
            top_k=cfg["numeric_top_k"],
            max_block_size=cfg["numeric_max_block"],
            sink=stream_buf,
        )
        _collect(df5)
        del df5

    # 6. Route 6: Rare-Token Address Retrieval
    if "rare_token_address" in enabled:
        df6 = retrieve_rare_token_address(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            addr_col=addr_col,
            top_k=cfg["rare_addr_top_k"],
            max_block_size=cfg["rare_addr_max_block"],
            sink=stream_buf,
        )
        _collect(df6)
        del df6

    # 7. Route 7: Reverse Retrieval (2D blocked + bucketed link table)
    if "reverse_retrieval" in enabled:
        df7 = retrieve_reverse(
            s1_df,
            s2_df,
            s3_df,
            id_col=id_col,
            name_col=name_col,
            top_k_reverse=cfg["reverse_top_k"],
            chunk_size=cfg["reverse_chunk_size"],
            query_block=cfg["tfidf_query_block"],
            index_block=cfg["reverse_index_block"],
            sink=stream_buf,
            spill_dir=None if spill_root is None else str(spill_root),
            dedup_buckets=n_buckets,
        )
        _collect(df7)
        del df7

    if stream_buf is not None:
        stream_buf.close()

    if writer is not None:
        writer.close()
        n_events = writer.rows_written

        # Stream both artifacts to disk bucket-by-bucket. Neither the events table nor the
        # candidate set is ever fully resident: the previous implementation read every
        # bucket back and pd.concat-ed them, which reproduced the entire memory spike the
        # spill exists to prevent.
        events_path = cfg.get("spill_events_path")
        candidates_path = cfg.get("spill_output_path")
        if events_path:
            write_events_stream(
                (frame for _b, frame in iter_spilled_buckets(spill_root, n_buckets=n_buckets)),
                events_path,
            )
        if candidates_path:
            write_candidate_dataset(
                iter_canonical_candidates_merged(spill_root, n_buckets=n_buckets),
                candidates_path,
                columns=tuple(CANONICAL_CANDIDATE_COLUMNS),
            )
            return _empty_candidate_df()

        # No output paths requested: fall back to the in-memory aggregation, which is only
        # safe for sample-sized inputs (this is the path tests and the 1K fixtures use).
        candidates_df = _finalize_spilled(spill_root, n_buckets, None)
        retrieval_events_df = (
            _collect_spilled_events(spill_root, n_buckets) if return_events else _empty_candidate_df()
        )
        if return_events:
            return candidates_df, retrieval_events_df
        return candidates_df

    if not route_dfs:
        empty_df = _empty_candidate_df()
        return (empty_df, empty_df) if return_events else empty_df

    # Combine all long-form retrieval events
    retrieval_events_df = pd.concat(route_dfs, ignore_index=True)

    # Deduplicate into canonical candidate DataFrame
    candidates_df = deduplicate_candidates(retrieval_events_df)

    if return_events:
        return candidates_df, retrieval_events_df
    return candidates_df


# ---------------------------------------------------------------------------
# Candidate Deduplication & Ranking
# ---------------------------------------------------------------------------
def _iter_deduplicated_buckets(
    spill_root: Path, n_buckets: int
) -> Any:
    """Yield deduplicated candidates one hash bucket at a time."""
    for _bucket_id, frame in iter_spilled_buckets(spill_root, n_buckets=n_buckets):
        yield deduplicate_frame(frame)


def _finalize_spilled(
    spill_root: Path,
    n_buckets: int,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run the external deduplication pass and return the candidate set.

    If ``out_path`` is given the result is streamed to that Parquet file as it is produced
    and an empty DataFrame is returned, which is the memory-safe mode for a full run. If
    it is None the buckets are concatenated in memory, which is convenient for tests and
    sample-sized inputs.
    """
    if out_path:
        write_candidate_dataset(_iter_deduplicated_buckets(spill_root, n_buckets), out_path)
        return _empty_candidate_df()
    frames = [
        deduplicate_frame(frame)
        for _bucket_id, frame in iter_spilled_buckets(spill_root, n_buckets=n_buckets)
    ]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return _empty_candidate_df()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        by=["s1_id", "rank"], ascending=[True, True]
    ).reset_index(drop=True)


def _collect_spilled_events(spill_root: Path, n_buckets: int) -> pd.DataFrame:
    """Reassemble the retrieval-events table from its shards, one bucket at a time."""
    frames = [
        frame for _bucket_id, frame in iter_spilled_buckets(spill_root, n_buckets=n_buckets)
    ]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return _empty_candidate_df()
    return pd.concat(frames, ignore_index=True)


def deduplicate_candidates(events_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate long-form retrieval events into unique candidate pairs.

    Aggregates multi-route occurrences:
    - pair_key = s1_id + "::" + candidate_id (unique identity)
    - score = max(score) across routes
    - route = comma-separated sorted list of retrieving routes
    - rank = 1-based integer rank per s1_id (sorted by score desc, candidate_id asc)
    """
    if len(events_df) == 0:
        return _empty_candidate_df()

    # Aggregate by (pair_key, s1_id, candidate_id, candidate_source)
    # Using groupby + agg for efficiency
    agg_df = (
        events_df.groupby(["pair_key", "s1_id", "candidate_id", "candidate_source"], as_index=False)
        .agg(
            score=("score", "max"),
            route=("route", lambda routes: ",".join(sorted(set(routes)))),
        )
    )

    # Sort deterministically: primary s1_id asc, secondary score desc, tertiary candidate_id asc
    agg_df = agg_df.sort_values(
        by=["s1_id", "score", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    # Assign 1-based rank within each s1_id
    agg_df["rank"] = agg_df.groupby("s1_id").cumcount() + 1

    # Order columns strictly to schema
    cols = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]
    return agg_df[cols]


# ---------------------------------------------------------------------------
# Ground-Truth Recall & Diagnostics Evaluation
# ---------------------------------------------------------------------------
def evaluate_candidate_recall(
    candidates_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    s1_ids_subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate pair-level and entity-level recall against ground truth.

    Parameters
    ----------
    candidates_df : Deduplicated candidate DataFrame with [s1_id, candidate_id, candidate_source].
    gt_df         : Ground truth DataFrame with [source1_entity_id, matched_entity_ids].
    s1_ids_subset : Optional list of S1 entity IDs to restrict evaluation to.

    Returns
    -------
    Dictionary of comprehensive recall & coverage metrics.
    """
    # 1. Parse ground-truth mapping: s1_id -> set(matched_cids)
    gt_map: Dict[str, Set[str]] = {}
    gt_s2_map: Dict[str, Set[str]] = {}
    gt_s3_map: Dict[str, Set[str]] = {}

    gt_s1_ids = gt_df["source1_entity_id"].astype(str).tolist()
    gt_matched_raw = gt_df["matched_entity_ids"].fillna("").astype(str).tolist() if "matched_entity_ids" in gt_df.columns else [""] * len(gt_df)
    subset_set = set(s1_ids_subset) if s1_ids_subset is not None else None

    for s1_id, raw_matches in zip(gt_s1_ids, gt_matched_raw):
        if subset_set is not None and s1_id not in subset_set:
            continue
        if raw_matches and raw_matches.lower() not in ("nan", "none", ""):
            matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
        else:
            matches = set()

        gt_map[s1_id] = matches
        gt_s2_map[s1_id] = {m for m in matches if m.startswith("S2-")}
        gt_s3_map[s1_id] = {m for m in matches if m.startswith("S3-")}

    # 2. Parse candidate mapping: s1_id -> set(cand_ids)
    cand_map: Dict[str, Set[str]] = defaultdict(set)
    c_s1_ids = candidates_df["s1_id"].astype(str).tolist()
    c_cand_ids = candidates_df["candidate_id"].astype(str).tolist()
    for s1_id, cand_id in zip(c_s1_ids, c_cand_ids):
        cand_map[s1_id].add(cand_id)

    # 3. Calculate recall metrics
    total_gt_pairs = sum(len(matches) for matches in gt_map.values())
    total_gt_s2 = sum(len(matches) for matches in gt_s2_map.values())
    total_gt_s3 = sum(len(matches) for matches in gt_s3_map.values())

    recovered_pairs = 0
    recovered_s2 = 0
    recovered_s3 = 0

    entities_with_gt = 0
    entities_any_match = 0
    entities_all_matches = 0
    singletons_count = 0

    for s1_id, true_matches in gt_map.items():
        if not true_matches:
            singletons_count += 1
            continue

        entities_with_gt += 1
        retrieved = cand_map.get(s1_id, set())

        shared = true_matches.intersection(retrieved)
        recovered_pairs += len(shared)

        recovered_s2 += len(gt_s2_map[s1_id].intersection(retrieved))
        recovered_s3 += len(gt_s3_map[s1_id].intersection(retrieved))

        if len(shared) > 0:
            entities_any_match += 1
        if len(shared) == len(true_matches):
            entities_all_matches += 1

    pair_recall = recovered_pairs / max(total_gt_pairs, 1)
    s2_recall = recovered_s2 / max(total_gt_s2, 1) if total_gt_s2 > 0 else 0.0
    s3_recall = recovered_s3 / max(total_gt_s3, 1) if total_gt_s3 > 0 else 0.0
    any_match_recall = entities_any_match / max(entities_with_gt, 1)
    all_match_recall = entities_all_matches / max(entities_with_gt, 1)

    # Candidate statistics
    cands_per_s1 = [len(cand_map.get(s1_id, set())) for s1_id in gt_map.keys()]
    if not cands_per_s1 and len(candidates_df) > 0:
        cands_per_s1 = candidates_df.groupby("s1_id").size().tolist()

    return {
        "total_evaluated_s1": len(gt_map),
        "entities_with_matches": entities_with_gt,
        "singletons": singletons_count,
        "total_gt_pairs": total_gt_pairs,
        "total_gt_s2_pairs": total_gt_s2,
        "total_gt_s3_pairs": total_gt_s3,
        "recovered_pairs": recovered_pairs,
        "pair_recall": round(pair_recall, 4),
        "s2_recall": round(s2_recall, 4),
        "s3_recall": round(s3_recall, 4),
        "any_match_recall": round(any_match_recall, 4),
        "all_match_recall": round(all_match_recall, 4),
        "total_candidates": len(candidates_df),
        "mean_candidates_per_s1": round(float(np.mean(cands_per_s1)), 2) if cands_per_s1 else 0.0,
        "median_candidates_per_s1": round(float(np.median(cands_per_s1)), 2) if cands_per_s1 else 0.0,
        "max_candidates_per_s1": int(np.max(cands_per_s1)) if cands_per_s1 else 0,
        "zero_candidate_s1_count": sum(1 for c in cands_per_s1 if c == 0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _candidate_columns() -> List[str]:
    return [
        "pair_key",
        "s1_id",
        "candidate_id",
        "candidate_source",
        "route",
        "rank",
        "score",
    ]


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_columns())
