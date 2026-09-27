"""
Retrieval metadata feature extraction and pivoting for candidate pairs.
Pivots long-form retrieval_events.parquet into candidate-pair level features.
"""

from __future__ import annotations

from typing import Any, Optional, Dict, List, Set
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


def _pivot_retrieval_features_reference(
    retrieval_events_df: Optional[pd.DataFrame],
    pair_keys: Optional[List[str] | pd.Series] = None,
) -> pd.DataFrame:
    """Original row-by-row implementation, retained verbatim as an equivalence oracle.

    Kept because the vectorized :func:`pivot_retrieval_features` below replaced it for
    scale, and the only acceptable evidence for that swap is that the two agree exactly.
    ``tests/test_features_retrieval_pivot.py`` asserts that on real retrieval events.
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


#: Defaults for a pair with no retrieval events, matching the reference implementation.
_PIVOT_DEFAULTS: Dict[str, float] = {
    "n_routes": 1.0,
    "retrieval_best_rank": np.nan,
    "retrieval_best_score": np.nan,
    "retrieved_by_exact_name": 0.0,
    "retrieved_by_tfidf_name": 0.0,
    "retrieved_by_tfidf_address": 0.0,
    "retrieved_by_numeric": 0.0,
    "retrieved_by_reverse": 0.0,
}

#: (output column, route-name substrings that must ALL appear) for the binary flags.
#:
#: The reference computed e.g.
#:     "tfidf_name" in routes or any("name" in r and "tfidf" in r for r in routes)
#: The first disjunct is redundant: a route equal to "tfidf_name" necessarily contains
#: both "name" and "tfidf", so it is already covered by the second. Every flag therefore
#: reduces to "some event in the group has a route containing these substrings", which is
#: a per-row boolean followed by a groupby max.
_PIVOT_FLAGS = [
    ("retrieved_by_exact_name", ("exact",)),
    ("retrieved_by_tfidf_name", ("name", "tfidf")),
    ("retrieved_by_tfidf_address", ("address", "tfidf")),
    ("retrieved_by_numeric", ("numeric",)),
    ("retrieved_by_reverse", ("reverse",)),
]


def pivot_retrieval_features(
    retrieval_events_df: Optional[pd.DataFrame],
    pair_keys: Optional[List[str] | pd.Series] = None,
) -> pd.DataFrame:
    """
    Pivots long-form retrieval events (pair_key, route, rank, score) into wide feature columns.
    If retrieval_events_df is None or empty, returns default/fallback feature DataFrame.

    Semantics are identical to :func:`_pivot_retrieval_features_reference`; only the
    implementation differs. This exists because the reference walked a Python
    ``groupby`` and accumulated a dict-of-dicts with one entry per pair, which measured
    1,440 pairs/s and ~500 bytes per pair -- about 12 GB and 4.8 h at the 25M candidate
    pairs this stage actually has to cover. Sharding the caller does not help, because the
    dict is O(total pairs) no matter how the work is split.

    The aggregations are all vectorizable without approximation:
      * ``n_routes``      -> ``nunique()`` over non-null routes, with 0 remapped to 1.0
                             (the reference fell back to 1.0 for a group with no usable route);
      * ``best_rank``     -> ``min()`` of the numeric-coerced, null-dropped ranks;
      * ``best_score``    -> ``max()`` of the numeric-coerced, null-dropped scores;
      * each binary flag  -> ``max()`` of a per-row boolean, which is exactly
                             ``any(predicate(route) for route in that group's routes)``.
    """
    if pair_keys is not None:
        p_keys = list(pair_keys)
    elif retrieval_events_df is not None and "pair_key" in retrieval_events_df.columns:
        p_keys = list(retrieval_events_df["pair_key"].unique())
    else:
        p_keys = []

    if len(p_keys) == 0:
        return pd.DataFrame(columns=["pair_key"] + RETRIEVAL_FEATURE_COLS)

    if retrieval_events_df is None or len(retrieval_events_df) == 0:
        out = pd.DataFrame({"pair_key": p_keys})
        for col in RETRIEVAL_FEATURE_COLS:
            out[col] = _PIVOT_DEFAULTS[col]
        return out

    events = retrieval_events_df
    grouped = events.groupby("pair_key", sort=False)

    agg: Dict[str, pd.Series] = {}

    if "route" in events.columns:
        # nunique() drops nulls, matching `set(group["route"].dropna().astype(str))`.
        n_routes = grouped["route"].nunique().astype("float64")
        # A group whose routes are all null yields 0 distinct routes; the reference
        # substituted 1.0 in that case.
        agg["n_routes"] = n_routes.mask(n_routes == 0, 1.0)

        # Empty string (not NaN) so a null route yields False rather than propagating.
        route_text = events["route"].astype("string").fillna("")
        flag_frame = {}
        for col, needles in _PIVOT_FLAGS:
            hit = pd.Series(True, index=events.index, dtype=bool)
            for needle in needles:
                hit = hit & route_text.str.contains(needle, regex=False, na=False)
            flag_frame[col] = hit
        flag_agg = pd.DataFrame(flag_frame, index=events.index).groupby(
            events["pair_key"], sort=False
        ).max()
        for col, _ in _PIVOT_FLAGS:
            agg[col] = flag_agg[col].astype("float64")
    else:
        agg["n_routes"] = pd.Series(1.0, index=grouped.size().index)
        for col, _ in _PIVOT_FLAGS:
            agg[col] = pd.Series(0.0, index=grouped.size().index)

    if "rank" in events.columns:
        ranks = pd.to_numeric(events["rank"], errors="coerce")
        agg["retrieval_best_rank"] = ranks.groupby(events["pair_key"], sort=False).min().astype(
            "float64"
        )
    if "score" in events.columns:
        scores = pd.to_numeric(events["score"], errors="coerce")
        agg["retrieval_best_score"] = scores.groupby(events["pair_key"], sort=False).max().astype(
            "float64"
        )

    out = pd.DataFrame({"pair_key": p_keys})
    for col in RETRIEVAL_FEATURE_COLS:
        if col in agg:
            out[col] = agg[col].reindex(out["pair_key"]).to_numpy()
        else:
            out[col] = _PIVOT_DEFAULTS[col]
        if col in ("n_routes",) or col.startswith("retrieved_by_"):
            out[col] = out[col].fillna(_PIVOT_DEFAULTS[col]).astype("float64")
        else:
            out[col] = out[col].astype("float64")

    return out
