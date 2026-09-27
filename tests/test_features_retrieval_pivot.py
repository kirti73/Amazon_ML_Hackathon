"""Equivalence tests for the vectorized retrieval pivot.

``pivot_retrieval_features`` was rewritten from a per-pair Python loop into vectorized
groupby aggregations so that P2 can cover the full candidate set (measured 1,440 pairs/s
and ~500 bytes per pair -> 12 GB and 4.8 h at 25M pairs).

A rewrite like that is only acceptable with proof, so
``_pivot_retrieval_features_reference`` -- the original implementation, kept verbatim --
is the oracle here. These tests assert bit-identical output (NaN placement included) on
real-shaped events and on the edge cases the rewrite could plausibly break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.retrieval_features import (
    RETRIEVAL_FEATURE_COLS,
    _pivot_retrieval_features_reference,
    pivot_retrieval_features,
)

REAL_ROUTES = [
    "exact_name",
    "tfidf_name",
    "rare_token_name",
    "tfidf_address",
    "numeric_address",
    "rare_token_address",
    "reverse_retrieval",
]


def _events(n_pairs: int = 400, seed: int = 3) -> pd.DataFrame:
    """Event frame shaped like the real retrieval_events.parquet, multi-route per pair."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        pair = f"S1-{i:05d}::S2-{i:05d}"
        for route in rng.choice(REAL_ROUTES, size=rng.integers(1, 4), replace=False):
            rows.append(
                dict(
                    pair_key=pair,
                    s1_id=f"S1-{i:05d}",
                    candidate_id=f"S2-{i:05d}",
                    candidate_source="S2",
                    route=str(route),
                    rank=int(rng.integers(1, 21)),
                    score=round(float(rng.random()), 6),
                )
            )
    return pd.DataFrame(rows)


def _assert_identical(events, pair_keys) -> None:
    ref = _pivot_retrieval_features_reference(events, pair_keys)
    got = pivot_retrieval_features(events, pair_keys)
    assert list(got.columns) == list(ref.columns) == ["pair_key"] + RETRIEVAL_FEATURE_COLS
    assert len(got) == len(ref)
    for col in ref.columns:
        if col == "pair_key":
            assert got[col].tolist() == ref[col].tolist()
            continue
        # Bit-identical, with NaN treated as equal to NaN.
        np.testing.assert_array_equal(
            got[col].to_numpy(dtype="float64"),
            ref[col].to_numpy(dtype="float64"),
        )


def test_matches_reference_on_realistic_multiroute_events():
    events = _events()
    _assert_identical(events, list(events["pair_key"].unique()))


def test_matches_reference_with_duplicated_and_reordered_pair_keys():
    events = _events(120)
    keys = list(events["pair_key"].unique())
    keys = list(np.random.default_rng(11).permutation(keys)) + keys[:40]
    _assert_identical(events, keys)


def test_matches_reference_for_pair_keys_absent_from_events():
    events = _events(60)
    keys = list(events["pair_key"].unique())[:20] + ["S1-99999::S2-99999"] * 3
    _assert_identical(events, keys)


def test_matches_reference_with_null_routes_ranks_and_scores():
    """The rewrite maps null routes to "" and coerces non-numeric ranks/scores to NaN;
    every one of those has to land exactly where the reference put it."""
    events = pd.DataFrame(
        [
            # group with an all-null route -> n_routes must fall back to 1.0
            dict(pair_key="p2", s1_id="b", candidate_id="2", candidate_source="S2",
                 route=None, rank=2, score=0.4),
            dict(pair_key="p2", s1_id="b", candidate_id="2", candidate_source="S2",
                 route=None, rank=None, score=None),
            # group whose rank/score are unparseable -> both must be NaN
            dict(pair_key="p3", s1_id="c", candidate_id="3", candidate_source="S3",
                 route="tfidf_address", rank="bad", score="x"),
            # ordinary multi-route group
            dict(pair_key="p1", s1_id="a", candidate_id="1", candidate_source="S2",
                 route="exact_name", rank=1, score=0.5),
            dict(pair_key="p1", s1_id="a", candidate_id="1", candidate_source="S2",
                 route="numeric_address", rank=3, score=0.2),
            # all-null rank but a valid route
            dict(pair_key="p4", s1_id="d", candidate_id="4", candidate_source="S3",
                 route="reverse_retrieval", rank=None, score=None),
        ]
    )
    _assert_identical(events, ["p1", "p2", "p3", "p4", "p_absent"])


def test_matches_reference_when_rank_or_score_column_absent():
    events = _events(40)
    _assert_identical(events.drop(columns=["rank"]), list(events["pair_key"].unique()))
    _assert_identical(events.drop(columns=["score"]), list(events["pair_key"].unique()))
    _assert_identical(events.drop(columns=["rank", "score"]), list(events["pair_key"].unique()))


def test_matches_reference_when_route_column_absent():
    events = _events(40)
    _assert_identical(events.drop(columns=["route"]), list(events["pair_key"].unique()))


def test_matches_reference_on_empty_and_none_inputs():
    events = _events(10)
    keys = list(events["pair_key"].unique())
    _assert_identical(events.iloc[0:0], keys)   # empty events
    _assert_identical(None, keys)                # no events at all
    _assert_identical(events, [])                # no pair keys requested


def test_route_substring_flags_match_reference_semantics():
    """A route name carrying several substrings must light up every matching flag."""
    events = pd.DataFrame(
        [dict(pair_key="p", s1_id="a", candidate_id="1", candidate_source="S2",
              route="tfidf_name_reverse_numeric", rank=1, score=0.5)]
    )
    got = pivot_retrieval_features(events, ["p"]).iloc[0]
    assert got["retrieved_by_tfidf_name"] == 1.0
    assert got["retrieved_by_numeric"] == 1.0
    assert got["retrieved_by_reverse"] == 1.0
    assert got["retrieved_by_exact_name"] == 0.0
    assert got["retrieved_by_tfidf_address"] == 0.0
    _assert_identical(events, ["p"])


def test_absent_pairs_get_documented_defaults():
    got = pivot_retrieval_features(_events(5), ["nope::nope"])
    row = got.iloc[0]
    assert row["n_routes"] == 1.0
    assert np.isnan(row["retrieval_best_rank"])
    assert np.isnan(row["retrieval_best_score"])
    for col in RETRIEVAL_FEATURE_COLS:
        if col.startswith("retrieved_by_"):
            assert row[col] == 0.0


def test_vectorized_pivot_is_materially_faster():
    """The reason the rewrite exists. Generous bound so this is not flaky on slow CI."""
    import time

    events = _events(2000, seed=5)
    keys = list(events["pair_key"].unique())
    t0 = time.perf_counter()
    pivot_retrieval_features(events, keys)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    _pivot_retrieval_features_reference(events, keys)
    slow = time.perf_counter() - t0
    assert fast * 5 < slow, f"vectorized={fast:.3f}s reference={slow:.3f}s"
