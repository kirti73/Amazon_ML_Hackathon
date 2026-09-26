"""
Integration test for Person 1 (Candidate Generation) and Person 2 (Preprocessing & Features).
Validates schema compliance against docs/schemas.md, ID string dtypes, pair_key invariant,
and Parquet persistence.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

from src.inference.pipeline import run_p1_p2_pipeline
from src.utils.adapters import adapt_raw_for_preprocessing, reconcile_candidates_schema


@pytest.fixture
def sample_data():
    """
    Creates realistic multilingual test records across S1, S2, and S3
    including exact matches, acronyms, subset names, Devanagari records,
    and missing address cases.
    """
    s1_records = [
        {
            "entity_id": "S1-001",
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
            "country": "India",
        },
        {
            "entity_id": "S1-002",
            "business_name": "Holloway Peak Seafood",
            "business_address": "105 ELM ST, MORGANTON, NC",
            "country": "US",
        },
        {
            "entity_id": "S1-003",
            "business_name": "Kalyani Welfare Society",
            "business_address": "70 Beacon Court, Springfield, IL",
            "country": "US",
        },
        {
            "entity_id": "S1-004",
            "business_name": "Delta Telecommunication Inc",
            "business_address": "914 Pierpont Ave, Cleveland, OH",
            "country": "US",
        },
    ]

    s2_records = [
        {
            "entity_id": "S2-001",
            "business_name": "राम मार्केटिंग Pvt Ltd",
            "business_address": "KH NO. 570/13, New Delhi, Delhi",
            "country": "India",
        },
        {
            "entity_id": "S2-002",
            "business_name": "-- Holloway Peak Inc Seafood",
            "business_address": "105 ELM ST, MORGANTON, NC",
            "country": "US",
        },
        {
            "entity_id": "S2-003",
            "business_name": "Kalyani",
            "business_address": "",  # Missing candidate address
            "country": "US",
        },
    ]

    s3_records = [
        {
            "entity_id": "S3-001",
            "business_name": "Delta Telecomm",
            "business_address": "914 Pierpont Ave, Cleveland, OH",
            "country": "US",
        },
        {
            "entity_id": "S3-002",
            "business_name": "Unrelated Business Corp",
            "business_address": "400 Oak Street, Austin, TX",
            "country": "US",
        },
    ]

    s1_df = pd.DataFrame(s1_records)
    s2_df = pd.DataFrame(s2_records)
    s3_df = pd.DataFrame(s3_records)

    return s1_df, s2_df, s3_df


def test_p1_p2_end_to_end_pipeline(sample_data):
    s1_df, s2_df, s3_df = sample_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        art_dir = Path(tmp_dir) / "artifacts"

        results = run_p1_p2_pipeline(
            s1_df=s1_df,
            s2_df=s2_df,
            s3_df=s3_df,
            artifacts_dir=art_dir,
            chunk_size=1000,
        )

        records_df = results["records"]
        candidates_df = results["candidates"]
        events_df = results["retrieval_events"]
        features_df = results["features"]

        # 1. Validate records.parquet
        assert len(records_df) == len(s1_df) + len(s2_df) + len(s3_df)
        expected_rec_cols = [
            "entity_id", "source", "raw_name", "raw_address", "raw_country",
            "norm_name_cons", "norm_name_aggr", "norm_address_cons", "country_norm"
        ]
        assert list(records_df.columns) == expected_rec_cols
        # Check that business_name was not dropped
        assert (records_df["raw_name"] != "").all()
        # Check Devanagari preservation
        hindi_rec = records_df[records_df["entity_id"] == "S1-001"].iloc[0]
        assert "राम मार्केटिंग" in hindi_rec["norm_name_cons"]

        # 2. Validate retrieval_events.parquet
        expected_events_cols = [
            "pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"
        ]
        assert list(events_df.columns) == expected_events_cols
        assert (events_df["pair_key"] == events_df["s1_id"] + "::" + events_df["candidate_id"]).all()
        assert events_df["candidate_source"].isin(["S2", "S3"]).all()

        # 3. Validate candidates.parquet schema (Section 6)
        expected_cand_cols = [
            "pair_key", "s1_id", "candidate_id", "candidate_source", "n_routes", "best_rank", "best_score"
        ]
        assert list(candidates_df.columns) == expected_cand_cols
        assert len(candidates_df) == candidates_df["pair_key"].nunique(), "Candidates must be unique per pair_key"
        assert set(candidates_df["pair_key"]) == set(events_df["pair_key"]), "Pair keys before and after must match exactly"
        assert (candidates_df["n_routes"] >= 1).all()
        assert (candidates_df["best_rank"] >= 1).all()

        # Validate that best_rank and best_score match retrieval_events aggregates
        for pk in candidates_df["pair_key"].sample(min(len(candidates_df), 5), random_state=42):
            ev_subset = events_df[events_df["pair_key"] == pk]
            cand_row = candidates_df[candidates_df["pair_key"] == pk].iloc[0]
            assert cand_row["n_routes"] == ev_subset["route"].nunique()
            assert cand_row["best_rank"] == int(ev_subset["rank"].min())
            assert np.isclose(cand_row["best_score"], float(ev_subset["score"].max()))

        # 4. Validate features.parquet schema (Section 7)
        assert len(features_df) == len(candidates_df)
        assert list(features_df.columns[:3]) == ["pair_key", "s1_id", "candidate_id"]
        assert len(features_df.columns) == 39  # 3 keys + 36 features
        assert (features_df["pair_key"] == candidates_df["pair_key"]).all()

        # Verify float32 feature dtypes
        for col in features_df.columns[3:]:
            assert features_df[col].dtype == np.float32, f"Feature column {col} is not float32"

        # Verify missing address handling (S1-003 :: S2-003)
        kalyani_pair = features_df[features_df["pair_key"] == "S1-003::S2-003"]
        if len(kalyani_pair) > 0:
            row = kalyani_pair.iloc[0]
            assert row["addr_missing_cand"] == 1.0
            assert np.isnan(row["addr_char_cos"])

        # 5. Verify Parquet files persisted on disk
        assert (art_dir / "records.parquet").exists()
        assert (art_dir / "retrieval_events.parquet").exists()
        assert (art_dir / "candidates.parquet").exists()
        assert (art_dir / "features.parquet").exists()

        # Reload from Parquet and ensure byte/schema integrity
        reloaded_rec = pd.read_parquet(art_dir / "records.parquet")
        reloaded_cand = pd.read_parquet(art_dir / "candidates.parquet")
        reloaded_events = pd.read_parquet(art_dir / "retrieval_events.parquet")
        reloaded_feat = pd.read_parquet(art_dir / "features.parquet")

        assert len(reloaded_rec) == len(records_df)
        assert len(reloaded_cand) == len(candidates_df)
        assert len(reloaded_events) == len(events_df)
        assert len(reloaded_feat) == len(features_df)
        assert list(reloaded_feat.columns) == list(features_df.columns)


def test_reconcile_candidates_schema_multi_route_regression():
    """
    Regression test:
    Validates that reconcile_candidates_schema() derives:
    - best_rank as MIN(retrieval_events['rank']) across routes, NOT the union rank in candidates_df
    - best_score as MAX(retrieval_events['score']) across routes
    - n_routes as distinct route count
    - strictly preserves set(candidate_pairs_before.pair_key) == set(candidate_pairs_after.pair_key)
    - keeps all entity IDs as strings
    """
    # Synthetic candidate table mimicking P1's deduplicate_candidates output
    # Pair 1: S1-100::S2-200 has a P1 union rank of 5
    # Pair 2: S1-100::S3-300 has a P1 union rank of 2
    candidate_pairs_before = pd.DataFrame([
        {
            "pair_key": "S1-100::S2-200",
            "s1_id": "S1-100",
            "candidate_id": "S2-200",
            "candidate_source": "S2",
            "route": "exact_name,tfidf_name,numeric_address",
            "rank": 5,          # Global union rank per S1 in P1 output
            "score": 0.96,       # P1 max score
        },
        {
            "pair_key": "S1-100::S3-300",
            "s1_id": "S1-100",
            "candidate_id": "S3-300",
            "candidate_source": "S3",
            "route": "tfidf_address",
            "rank": 2,          # Global union rank per S1 in P1 output
            "score": 0.72,
        },
    ])

    # Retrieval events with distinct ranks and scores across routes
    # For Pair 1:
    #   - exact_name: rank 3, score 0.80
    #   - tfidf_name: rank 8, score 0.96
    #   - numeric_address: rank 1, score 0.60
    # Expected: best_rank = min(3, 8, 1) = 1, best_score = max(0.80, 0.96, 0.60) = 0.96, n_routes = 3
    retrieval_events_df = pd.DataFrame([
        {"pair_key": "S1-100::S2-200", "s1_id": "S1-100", "candidate_id": "S2-200", "candidate_source": "S2", "route": "exact_name", "rank": 3, "score": 0.80},
        {"pair_key": "S1-100::S2-200", "s1_id": "S1-100", "candidate_id": "S2-200", "candidate_source": "S2", "route": "tfidf_name", "rank": 8, "score": 0.96},
        {"pair_key": "S1-100::S2-200", "s1_id": "S1-100", "candidate_id": "S2-200", "candidate_source": "S2", "route": "numeric_address", "rank": 1, "score": 0.60},
        {"pair_key": "S1-100::S3-300", "s1_id": "S1-100", "candidate_id": "S3-300", "candidate_source": "S3", "route": "tfidf_address", "rank": 6, "score": 0.72},
    ])

    candidate_pairs_after = reconcile_candidates_schema(
        candidates_df=candidate_pairs_before,
        retrieval_events_df=retrieval_events_df,
        strict_schema_md=True,
    )

    # 1. Exact pair_key preservation assertion requested by user
    assert set(candidate_pairs_before["pair_key"]) == set(candidate_pairs_after["pair_key"]), (
        "Reconciled candidate pairs must have identical pair_key set"
    )
    assert len(candidate_pairs_after) == len(candidate_pairs_before), "Row count must remain identical"

    # 2. Invariant: all IDs must be strings
    for col in ["pair_key", "s1_id", "candidate_id", "candidate_source"]:
        assert (candidate_pairs_after[col].apply(lambda x: isinstance(x, str))).all(), f"{col} must be string"
    assert candidate_pairs_after["candidate_source"].isin(["S2", "S3"]).all()

    # 3. Canonical columns per docs/schemas.md Section 6
    assert list(candidate_pairs_after.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "n_routes", "best_rank", "best_score"
    ]

    # 4. Multi-route pair validation
    p1_row = candidate_pairs_after[candidate_pairs_after["pair_key"] == "S1-100::S2-200"].iloc[0]
    assert p1_row["n_routes"] == 3, f"Expected n_routes=3, got {p1_row['n_routes']}"
    assert p1_row["best_rank"] == 1, f"Expected min rank=1, got {p1_row['best_rank']}"
    assert p1_row["best_rank"] != 5, "best_rank must NOT equal the union rank (5)!"
    assert np.isclose(p1_row["best_score"], 0.96), f"Expected max score=0.96, got {p1_row['best_score']}"

    # 5. Single-route pair validation
    p2_row = candidate_pairs_after[candidate_pairs_after["pair_key"] == "S1-100::S3-300"].iloc[0]
    assert p2_row["n_routes"] == 1
    assert p2_row["best_rank"] == 6  # min retrieval-event rank is 6, union rank was 2
    assert np.isclose(p2_row["best_score"], 0.72)


def test_reconcile_candidates_schema_no_silent_fallback():
    """
    Validates that reconcile_candidates_schema():
    - Raises ValueError when retrieval_events_df is None or empty (no silent fallback).
    - Raises ValueError when candidate pairs are missing from retrieval_events_df.
    """
    candidates = pd.DataFrame([
        {
            "pair_key": "S1-1::S2-1",
            "s1_id": "S1-1",
            "candidate_id": "S2-1",
            "candidate_source": "S2",
            "route": "exact_name",
            "rank": 1,
            "score": 1.0,
        }
    ])

    with pytest.raises(ValueError, match="retrieval_events_df is required"):
        reconcile_candidates_schema(candidates, retrieval_events_df=None)

    with pytest.raises(ValueError, match="retrieval_events_df is required"):
        reconcile_candidates_schema(candidates, retrieval_events_df=pd.DataFrame())

    # Incomplete retrieval events (missing S1-1::S2-1)
    incomplete_events = pd.DataFrame([
        {
            "pair_key": "S1-2::S2-2",
            "s1_id": "S1-2",
            "candidate_id": "S2-2",
            "candidate_source": "S2",
            "route": "exact_name",
            "rank": 1,
            "score": 1.0,
        }
    ])
    with pytest.raises(ValueError, match="Missing retrieval events"):
        reconcile_candidates_schema(candidates, retrieval_events_df=incomplete_events)

