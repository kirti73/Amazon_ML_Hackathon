"""
Batch feature pipeline builder for Amazon ML Challenge 2026: Business Entity Resolution.
Constructs canonical features.parquet from candidate pairs and normalized records.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, List, Dict, Set
import numpy as np
import pandas as pd

from .name_features import compute_name_features_batch
from .address_features import compute_address_features_batch
from .cross_features import compute_cross_features_batch
from .retrieval_features import pivot_retrieval_features, RETRIEVAL_FEATURE_COLS


FEATURE_COLUMNS = [
    # Name features
    "name_exact_cons",
    "name_exact_aggr",
    "name_char_cos",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_containment_s1_in_cand",
    "name_containment_cand_in_s1",
    "name_len_diff",
    "name_len_ratio",
    "name_token_count_diff",
    "name_first_token_match",
    # Address features
    "addr_exact_cons",
    "addr_char_cos",
    "addr_token_sort_ratio",
    "addr_containment_s1_in_cand",
    "addr_containment_cand_in_s1",
    "addr_len_diff",
    "addr_num_jaccard",
    "addr_num_conflict",
    "addr_first_num_match",
    # Country & Missingness & Cross
    "country_eq",
    "name_missing_cand",
    "addr_missing_s1",
    "addr_missing_cand",
    "addr_missing_either",
    "min_name_addr_sim",
    "max_name_addr_sim",
    "both_strong_match",
    # Retrieval metadata
    "n_routes",
    "retrieval_best_rank",
    "retrieval_best_score",
    "retrieved_by_exact_name",
    "retrieved_by_tfidf_name",
    "retrieved_by_tfidf_address",
    "retrieved_by_numeric",
    "retrieved_by_reverse",
]


def build_features(
    candidates_df: pd.DataFrame,
    records_df: pd.DataFrame,
    retrieval_events_df: Optional[pd.DataFrame] = None,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """
    Constructs the canonical features.parquet DataFrame for given candidate pairs.

    Parameters:
    -----------
    candidates_df: pd.DataFrame with ['s1_id', 'candidate_id'] and optional ['pair_key']
    records_df: pd.DataFrame with ['entity_id', 'raw_name', 'raw_address', 'raw_country',
                                  'norm_name_cons', 'norm_name_aggr', 'norm_address_cons', 'country_norm']
    retrieval_events_df: Optional pd.DataFrame with ['pair_key', 'route', 'rank', 'score']
    chunk_size: Processing batch size for memory efficiency on multi-million row datasets

    Returns:
    --------
    pd.DataFrame with ['pair_key', 's1_id', 'candidate_id'] + all feature columns in float32.
    """
    if len(candidates_df) == 0:
        cols = ["pair_key", "s1_id", "candidate_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    # Ensure string types on IDs
    cand_df = candidates_df.copy()
    cand_df["s1_id"] = cand_df["s1_id"].astype(str)
    cand_df["candidate_id"] = cand_df["candidate_id"].astype(str)
    if "pair_key" not in cand_df.columns:
        cand_df["pair_key"] = cand_df["s1_id"] + "::" + cand_df["candidate_id"]
    else:
        cand_df["pair_key"] = cand_df["pair_key"].astype(str)

    # Index records table by entity_id for high-speed join / lookup
    rec_indexed = records_df.set_index("entity_id")

    # Pre-pivot retrieval features if available
    retrieval_feats_df = None
    if retrieval_events_df is not None and len(retrieval_events_df) > 0:
        retrieval_feats_df = pivot_retrieval_features(
            retrieval_events_df=retrieval_events_df,
            pair_keys=cand_df["pair_key"],
        ).set_index("pair_key")

    chunks: List[pd.DataFrame] = []
    n_total = len(cand_df)

    for start_idx in range(0, n_total, chunk_size):
        end_idx = min(start_idx + chunk_size, n_total)
        chunk_cands = cand_df.iloc[start_idx:end_idx].copy()

        # Join S1 records
        s1_joined = chunk_cands[["s1_id"]].join(rec_indexed, on="s1_id", how="left")
        cand_joined = chunk_cands[["candidate_id"]].join(rec_indexed, on="candidate_id", how="left")

        # Extract normalized strings
        s1_name_cons = s1_joined["norm_name_cons"].fillna("").to_numpy(dtype=object)
        cand_name_cons = cand_joined["norm_name_cons"].fillna("").to_numpy(dtype=object)

        s1_name_aggr = s1_joined["norm_name_aggr"].fillna("").to_numpy(dtype=object)
        cand_name_aggr = cand_joined["norm_name_aggr"].fillna("").to_numpy(dtype=object)

        s1_addr_cons = s1_joined["norm_address_cons"].fillna("").to_numpy(dtype=object)
        cand_addr_cons = cand_joined["norm_address_cons"].fillna("").to_numpy(dtype=object)

        s1_name_raw = s1_joined["raw_name"].fillna("").to_numpy(dtype=object)
        cand_name_raw = cand_joined["raw_name"].fillna("").to_numpy(dtype=object)

        s1_addr_raw = s1_joined["raw_address"].fillna("").to_numpy(dtype=object)
        cand_addr_raw = cand_joined["raw_address"].fillna("").to_numpy(dtype=object)

        s1_ctry_norm = s1_joined["country_norm"].fillna("").to_numpy(dtype=object)
        cand_ctry_norm = cand_joined["country_norm"].fillna("").to_numpy(dtype=object)

        # 1. Compute Name Features
        name_df = compute_name_features_batch(
            s1_names_cons=s1_name_cons,
            cand_names_cons=cand_name_cons,
            s1_names_aggr=s1_name_aggr,
            cand_names_aggr=cand_name_aggr,
        )

        # 2. Compute Address Features
        addr_df = compute_address_features_batch(
            s1_addrs_cons=s1_addr_cons,
            cand_addrs_cons=cand_addr_cons,
        )

        # 3. Compute Country & Missingness & Cross Features
        cross_df = compute_cross_features_batch(
            s1_names_raw=s1_name_raw,
            cand_names_raw=cand_name_raw,
            s1_addrs_raw=s1_addr_raw,
            cand_addrs_raw=cand_addr_raw,
            s1_countries_norm=s1_ctry_norm,
            cand_countries_norm=cand_ctry_norm,
            name_char_cos_arr=name_df["name_char_cos"].to_numpy(),
            addr_char_cos_arr=addr_df["addr_char_cos"].to_numpy(),
        )

        # 4. Join Retrieval Features
        chunk_pkeys = chunk_cands["pair_key"].tolist()
        if retrieval_feats_df is not None:
            ret_chunk = retrieval_feats_df.reindex(chunk_pkeys).reset_index(drop=True)
        else:
            # Fallback retrieval features
            ret_chunk = pd.DataFrame({
                "n_routes": [1.0] * len(chunk_cands),
                "retrieval_best_rank": [np.nan] * len(chunk_cands),
                "retrieval_best_score": [np.nan] * len(chunk_cands),
                "retrieved_by_exact_name": [0.0] * len(chunk_cands),
                "retrieved_by_tfidf_name": [0.0] * len(chunk_cands),
                "retrieved_by_tfidf_address": [0.0] * len(chunk_cands),
                "retrieved_by_numeric": [0.0] * len(chunk_cands),
                "retrieved_by_reverse": [0.0] * len(chunk_cands),
            })

        # Assemble chunk
        chunk_out = pd.DataFrame({
            "pair_key": chunk_cands["pair_key"].values,
            "s1_id": chunk_cands["s1_id"].values,
            "candidate_id": chunk_cands["candidate_id"].values,
        })

        for df_part in (name_df, addr_df, cross_df, ret_chunk):
            for col in df_part.columns:
                if col not in chunk_out.columns:
                    chunk_out[col] = df_part[col].astype(np.float32).values

        chunks.append(chunk_out)

    del rec_indexed
    gc.collect()

    features_df = pd.concat(chunks, ignore_index=True)
    return features_df
