"""
Retrieval metadata feature extraction and pivoting for candidate pairs.
Pivots long-form retrieval_events.parquet into candidate-pair level features.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Set
import numpy as np
import pandas as pd


RETRIEVAL_FEATURE_COLS = [
    "n_routes",
    "retrieval_best_rank",
    "retrieval_best_score",
    "retrieved_by_exact_name",
    "retrieved_by_tfidf_name",
    "retrieved_by_tfidf_address",
    "retrieved_by_numeric",
    "retrieved_by_reverse",
]


def pivot_retrieval_features(
    retrieval_events_df: Optional[pd.DataFrame],
    pair_keys: Optional[List[str] | pd.Series] = None,
) -> pd.DataFrame:
    """
    Pivots long-form retrieval events (pair_key, route, rank, score) into wide feature columns.
    If retrieval_events_df is None or empty, returns default/fallback feature DataFrame.
    """
    if pair_keys is not None:
        p_keys = list(pair_keys)
    elif retrieval_events_df is not None and "pair_key" in retrieval_events_df.columns:
        p_keys = list(retrieval_events_df["pair_key"].unique())
    else:
        p_keys = []

    n_pairs = len(p_keys)

    if n_pairs == 0:
        cols = ["pair_key"] + RETRIEVAL_FEATURE_COLS
        return pd.DataFrame(columns=cols)

    # If no retrieval events provided, populate default values
    if retrieval_events_df is None or len(retrieval_events_df) == 0:
        return pd.DataFrame({
            "pair_key": p_keys,
            "n_routes": [1.0] * n_pairs,
            "retrieval_best_rank": [np.nan] * n_pairs,
            "retrieval_best_score": [np.nan] * n_pairs,
            "retrieved_by_exact_name": [0.0] * n_pairs,
            "retrieved_by_tfidf_name": [0.0] * n_pairs,
            "retrieved_by_tfidf_address": [0.0] * n_pairs,
            "retrieved_by_numeric": [0.0] * n_pairs,
            "retrieved_by_reverse": [0.0] * n_pairs,
        })

    # Group by pair_key
    grouped = retrieval_events_df.groupby("pair_key")

    res_data: Dict[str, List[float]] = {
        "n_routes": [],
        "retrieval_best_rank": [],
        "retrieval_best_score": [],
        "retrieved_by_exact_name": [],
        "retrieved_by_tfidf_name": [],
        "retrieved_by_tfidf_address": [],
        "retrieved_by_numeric": [],
        "retrieved_by_reverse": [],
    }

    # Pre-aggregate route presence, min rank, max score
    route_agg: Dict[str, Dict[str, Any]] = {}
    for pair_key, group in grouped:
        routes = set(group["route"].dropna().astype(str)) if "route" in group.columns else set()
        
        ranks = pd.to_numeric(group["rank"], errors="coerce").dropna() if "rank" in group.columns else pd.Series(dtype=float)
        best_rank = float(ranks.min()) if len(ranks) > 0 else np.nan

        scores = pd.to_numeric(group["score"], errors="coerce").dropna() if "score" in group.columns else pd.Series(dtype=float)
        best_score = float(scores.max()) if len(scores) > 0 else np.nan

        route_agg[pair_key] = {
            "n_routes": float(len(routes)) if routes else 1.0,
            "best_rank": best_rank,
            "best_score": best_score,
            "exact_name": 1.0 if ("exact_name" in routes or any("exact" in r for r in routes)) else 0.0,
            "tfidf_name": 1.0 if ("tfidf_name" in routes or any("name" in r and "tfidf" in r for r in routes)) else 0.0,
            "tfidf_address": 1.0 if ("tfidf_address" in routes or any("address" in r and "tfidf" in r for r in routes)) else 0.0,
            "numeric": 1.0 if ("numeric" in routes or any("numeric" in r for r in routes)) else 0.0,
            "reverse": 1.0 if ("reverse" in routes or any("reverse" in r for r in routes)) else 0.0,
        }

    # Match order of input pair_keys
    for pk in p_keys:
        if pk in route_agg:
            info = route_agg[pk]
            res_data["n_routes"].append(info["n_routes"])
            res_data["retrieval_best_rank"].append(info["best_rank"])
            res_data["retrieval_best_score"].append(info["best_score"])
            res_data["retrieved_by_exact_name"].append(info["exact_name"])
            res_data["retrieved_by_tfidf_name"].append(info["tfidf_name"])
            res_data["retrieved_by_tfidf_address"].append(info["tfidf_address"])
            res_data["retrieved_by_numeric"].append(info["numeric"])
            res_data["retrieved_by_reverse"].append(info["reverse"])
        else:
            res_data["n_routes"].append(1.0)
            res_data["retrieval_best_rank"].append(np.nan)
            res_data["retrieval_best_score"].append(np.nan)
            res_data["retrieved_by_exact_name"].append(0.0)
            res_data["retrieved_by_tfidf_name"].append(0.0)
            res_data["retrieved_by_tfidf_address"].append(0.0)
            res_data["retrieved_by_numeric"].append(0.0)
            res_data["retrieved_by_reverse"].append(0.0)

    res_df = pd.DataFrame(res_data)
    res_df.insert(0, "pair_key", p_keys)
    return res_df
