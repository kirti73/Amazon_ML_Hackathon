"""
P4 Adapters for P1 and P2 Integration.
Handles schema reconciliation, column standardizations, and Parquet contract alignment.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def adapt_raw_for_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapts a raw official TSV DataFrame (which uses 'business_name' and 'business_address')
    into a DataFrame compatible with P2's preprocess_records_df() (which expects 'name'/'raw_name'
    and 'address'/'raw_address').

    Preserves original columns and adds standard aliases.
    """
    adapted = df.copy()
    if "business_name" in adapted.columns and "name" not in adapted.columns:
        adapted["name"] = adapted["business_name"].fillna("").astype(str)
    if "business_address" in adapted.columns and "address" not in adapted.columns:
        adapted["address"] = adapted["business_address"].fillna("").astype(str)
    if "country" in adapted.columns and "raw_country" not in adapted.columns:
        adapted["raw_country"] = adapted["country"].fillna("").astype(str)
    return adapted


def reconcile_candidates_schema(
    candidates_df: pd.DataFrame,
    retrieval_events_df: pd.DataFrame,
    strict_schema_md: bool = True,
) -> pd.DataFrame:
    """
    Reconciles P1's deduplicated candidates DataFrame with docs/schemas.md Section 6:
    docs/schemas.md requires:
        ['pair_key', 's1_id', 'candidate_id', 'candidate_source', 'n_routes', 'best_rank', 'best_score']

    P1's raw candidates_df provides:
        ['pair_key', 's1_id', 'candidate_id', 'candidate_source', 'route', 'rank', 'score']
    where P1's 'rank' is the global union rank per S1, not the minimum route rank.

    Canonical metadata is derived strictly from retrieval_events_df (no silent fallbacks):
        - n_routes: number of distinct routes per pair_key
        - best_rank: minimum retrieval-event rank per pair_key across routes
        - best_score: maximum retrieval-event score per pair_key across routes

    This metadata is attached to the exact P1 candidate set, strictly asserting:
        - pair_key, s1_id, candidate_id, candidate_source
        - all entity IDs as strings
        - exact candidate pair set (no pairs added or removed)
        - exact row ordering and index preservation
    """
    if len(candidates_df) == 0:
        cols = [
            "pair_key", "s1_id", "candidate_id", "candidate_source",
            "n_routes", "best_rank", "best_score"
        ] if strict_schema_md else list(candidates_df.columns)
        return pd.DataFrame(columns=cols)

    if retrieval_events_df is None or len(retrieval_events_df) == 0:
        raise ValueError(
            "retrieval_events_df is required to derive canonical candidate metadata (n_routes, best_rank, best_score). "
            "Silent fallback to union rank is disabled in production."
        )

    res = candidates_df.copy()
    res["pair_key"] = res["pair_key"].astype(str)
    res["s1_id"] = res["s1_id"].astype(str)
    res["candidate_id"] = res["candidate_id"].astype(str)
    res["candidate_source"] = res["candidate_source"].astype(str)

    events = retrieval_events_df.copy()
    events["pair_key"] = events["pair_key"].astype(str)

    # Validate that every candidate pair exists in retrieval events
    missing_keys = set(res["pair_key"]) - set(events["pair_key"])
    if missing_keys:
        raise ValueError(
            f"Missing retrieval events for {len(missing_keys)} candidate pairs. "
            f"Every candidate pair must originate from at least one retrieval event. Example missing: {list(missing_keys)[:3]}"
        )

    events["rank_num"] = pd.to_numeric(events["rank"], errors="raise")
    events["score_num"] = pd.to_numeric(events["score"], errors="raise")
    events["route_str"] = events["route"].astype(str)

    agg = events.groupby("pair_key").agg(
        n_routes=("route_str", "nunique"),
        best_rank=("rank_num", "min"),
        best_score=("score_num", "max"),
    )

    # Attach canonical metadata via pair_key map without altering row order or index
    res["n_routes"] = res["pair_key"].map(agg["n_routes"]).astype(int)
    res["best_rank"] = res["pair_key"].map(agg["best_rank"]).astype(int)
    res["best_score"] = res["pair_key"].map(agg["best_score"]).astype(float)

    # Strengthened invariant assertions
    assert len(res) == len(candidates_df), "Row count must remain identical"
    assert (res["pair_key"].values == candidates_df["pair_key"].values).all(), "Row ordering or pair_key values altered"
    assert set(res["pair_key"]) == set(candidates_df["pair_key"]), "Set of candidate pair_keys altered"
    assert (res["s1_id"].values == candidates_df["s1_id"].values).all(), "s1_id values altered"
    assert (res["candidate_id"].values == candidates_df["candidate_id"].values).all(), "candidate_id values altered"
    assert (res["candidate_source"].values == candidates_df["candidate_source"].values).all(), "candidate_source values altered"
    assert (res["n_routes"] >= 1).all(), "n_routes must be >= 1 for all candidates"
    assert (res["best_rank"] >= 1).all(), "best_rank must be >= 1 for all candidates"
    assert not res["best_score"].isna().any(), "best_score must not contain NaN"

    if strict_schema_md:
        cols = [
            "pair_key",
            "s1_id",
            "candidate_id",
            "candidate_source",
            "n_routes",
            "best_rank",
            "best_score",
        ]
        return res[cols]
    return res
