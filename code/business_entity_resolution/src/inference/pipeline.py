"""
End-to-end P1 (Candidate Generation) and P2 (Preprocessing & Feature Engineering) pipeline.
Orchestrated by Person 4 (Integration & Packaging).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ..blocking import generate_candidates
from ..preprocessing import preprocess_records_df
from ..features import build_features
from ..utils.adapters import adapt_raw_for_preprocessing, reconcile_candidates_schema


def run_p1_p2_pipeline(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    artifacts_dir: Optional[Union[str, Path]] = None,
    blocking_config: Optional[Dict[str, Any]] = None,
    chunk_size: int = 50000,
) -> Dict[str, pd.DataFrame]:
    """
    Executes the end-to-end candidate generation, record preprocessing, and feature engineering flow.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Raw entity DataFrames loaded via loaders.py (with entity_id, business_name, business_address, country).
    artifacts_dir : Optional[Union[str, Path]]
        Directory to save canonical Parquet artifacts. If None, artifacts are not written to disk.
    blocking_config : Optional[Dict[str, Any]]
        Optional overrides for P1 blocking routes and parameters.
    chunk_size : int
        Chunk size for P2 vectorized feature building (default 50,000).

    Returns
    -------
    Dict[str, pd.DataFrame] containing:
        - 'records': Preprocessed records DataFrame (records.parquet schema)
        - 'retrieval_events': Long-form retrieval events DataFrame (retrieval_events.parquet schema)
        - 'candidates': Canonical deduplicated candidates DataFrame (candidates.parquet schema)
        - 'features': Pairwise feature matrix (features.parquet schema)
    """
    # 1. P1: Candidate Generation with retrieval events enabled
    candidates_raw_df, retrieval_events_df = generate_candidates(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        config=blocking_config,
        return_events=True,
    )

    # 2. Reconcile candidates schema with docs/schemas.md Section 6
    candidates_df = reconcile_candidates_schema(
        candidates_df=candidates_raw_df,
        retrieval_events_df=retrieval_events_df,
        strict_schema_md=True,
    )

    # 3. P2: Preprocessing of all records (S1, S2, S3)
    s1_adapted = adapt_raw_for_preprocessing(s1_df)
    s2_adapted = adapt_raw_for_preprocessing(s2_df)
    s3_adapted = adapt_raw_for_preprocessing(s3_df)

    s1_pre = preprocess_records_df(s1_adapted)
    s2_pre = preprocess_records_df(s2_adapted)
    s3_pre = preprocess_records_df(s3_adapted)

    records_df = pd.concat([s1_pre, s2_pre, s3_pre], ignore_index=True).drop_duplicates(
        subset=["entity_id"]
    ).reset_index(drop=True)

    # 4. P2: Pairwise Feature Engineering
    features_df = build_features(
        candidates_df=candidates_raw_df,
        records_df=records_df,
        retrieval_events_df=retrieval_events_df,
        chunk_size=chunk_size,
    )

    # 5. Persist canonical artifacts if requested
    if artifacts_dir is not None:
        art_path = Path(artifacts_dir)
        art_path.mkdir(parents=True, exist_ok=True)

        records_df.to_parquet(art_path / "records.parquet", index=False)
        retrieval_events_df.to_parquet(art_path / "retrieval_events.parquet", index=False)
        candidates_df.to_parquet(art_path / "candidates.parquet", index=False)
        features_df.to_parquet(art_path / "features.parquet", index=False)

    return {
        "records": records_df,
        "retrieval_events": retrieval_events_df,
        "candidates": candidates_df,
        "features": features_df,
    }
