"""Shared synthetic fixtures for the Person 3 test suite.

Every fixture here is fully synthetic and deterministic. They exist to validate
mechanics (shapes, dtypes, contracts, leakage) only. No score produced against
these fixtures is a competition-valid result.

The candidate/feature schemas mirror the canonical contracts recorded in
``docs/schemas.md``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

N_S1 = 50

S1_POOL_IDS = [f"S1-{i:04d}" for i in range(N_S1)]
S2_POOL_IDS = [f"S2-{i:04d}" for i in range(120)]
S3_POOL_IDS = [f"S3-{i:04d}" for i in range(120)]

# Deterministic country mix for the S1 pool.
COUNTRY_CYCLE = ("US", "India", "France")


def _country_for(entity_id: str) -> str:
    """Deterministic country for any synthetic entity id (S1/S2/S3 alike)."""
    return COUNTRY_CYCLE[int(entity_id.split("-")[1]) % len(COUNTRY_CYCLE)]


def _true_match_count(index: int) -> int:
    """Deterministic spread of ground-truth match counts across all buckets.

    Bucket 0 (zero matches), bucket 1, bucket 2-3 and bucket 4+ are all populated
    so fold stratification has something to balance.
    """
    pattern = (0, 1, 0, 2, 1, 0, 3, 5, 0, 1, 2, 0, 4, 1, 0, 2)
    return pattern[index % len(pattern)]


def _build_s1_pool() -> pd.DataFrame:
    rows = []
    for s1_id in S1_POOL_IDS:
        rows.append(
            {
                "entity_id": s1_id,
                "business_name": f"Synthetic Business {s1_id}",
                "business_address": f"{int(s1_id.split('-')[1]) + 1} Synthetic Street",
                "country": _country_for(s1_id),
            }
        )
    return pd.DataFrame(rows)


def _build_ground_truth() -> "dict[str, set[str]]":
    gt = {}
    pool2 = itertools.cycle(S2_POOL_IDS)
    pool3 = itertools.cycle(S3_POOL_IDS)
    for index, s1_id in enumerate(S1_POOL_IDS):
        n_true = _true_match_count(index)
        matches = set()
        for j in range(n_true):
            if j % 2 == 0:
                matches.add(next(pool2))
            else:
                matches.add(next(pool3))
        gt[s1_id] = matches
    return gt


def _build_candidates(ground_truth: "dict[str, set[str]") -> pd.DataFrame:
    """Every true match is present in the candidate set, plus hard negatives.

    Zero-match S1 entities receive only negative candidates, which is what the
    decision layer must abstain on.
    """
    rows = []
    candidate_pool = S2_POOL_IDS + S3_POOL_IDS
    for index, s1_id in enumerate(S1_POOL_IDS):
        true_matches = ground_truth[s1_id]
        n_negatives = 1 if not true_matches else 2 + (index % 3)
        negatives = []
        cursor = (index * 7) % len(candidate_pool)
        while len(negatives) < n_negatives:
            candidate_id = candidate_pool[cursor % len(candidate_pool)]
            cursor += 1
            if candidate_id in true_matches or candidate_id in negatives:
                continue
            negatives.append(candidate_id)

        ranked = list(true_matches) + negatives
        for rank, candidate_id in enumerate(ranked, start=1):
            is_true = candidate_id in true_matches
            source = "S2" if candidate_id.startswith("S2-") else "S3"
            rows.append(
                {
                    "pair_key": f"{s1_id}::{candidate_id}",
                    "s1_id": s1_id,
                    "candidate_id": candidate_id,
                    "candidate_source": source,
                    "n_routes": 1 if is_true else (2 if index % 2 else 1),
                    "best_rank": rank,
                    "best_score": round(0.95 - 0.07 * rank, 6) if is_true else round(0.45 - 0.03 * rank, 6),
                }
            )
    return pd.DataFrame(rows)


# Feature columns grouped by the Person 2 contract in the master plan Section 6.
RETRIEVAL_RANK_COLUMNS = [
    "exact_name_rank",
    "numeric_rank",
    "rare_token_rank",
    "tfidf_name_rank",
    "tfidf_addr_rank",
]

RETRIEVAL_SCORE_COLUMNS = [
    "exact_name_score",
    "numeric_score",
    "rare_token_score",
    "tfidf_name_score",
    "tfidf_addr_score",
]

NAME_SIMILARITY_COLUMNS = [
    "name_char_cos",
    "name_token_set",
    "name_token_sort",
    "name_contain_idf",
]

ADDR_SIMILARITY_COLUMNS = [
    "addr_char_cos",
    "addr_contain_idf",
    "addr_numeric_jaccard",
]

MONOTONIC_SIMILARITY_COLUMNS = (
    NAME_SIMILARITY_COLUMNS + ADDR_SIMILARITY_COLUMNS + ["best_score"]
)

FEATURE_COLUMNS = (
    [
        "pair_key",
        "s1_id",
        "candidate_id",
        "candidate_source",
        "n_routes",
        "best_rank",
        "best_score",
    ]
    + RETRIEVAL_RANK_COLUMNS
    + RETRIEVAL_SCORE_COLUMNS
    + NAME_SIMILARITY_COLUMNS
    + ["name_exact_conservative", "name_exact_aggressive", "name_len_diff", "name_token_count_diff"]
    + ADDR_SIMILARITY_COLUMNS
    + ["addr_numeric_conflict", "addr_len_diff"]
    + ["country_eq"]
    + [
        "s1_name_missing",
        "cand_name_missing",
        "s1_addr_missing",
        "cand_addr_missing",
    ]
)


def _build_features(
    candidates_df: pd.DataFrame, ground_truth: "dict[str, set[str]]", rng: np.random.RandomState
) -> pd.DataFrame:
    n = len(candidates_df)
    out = candidates_df.copy()

    for column in RETRIEVAL_RANK_COLUMNS:
        out[column] = rng.randint(1, 20, size=n).astype(np.float32)
    for column in RETRIEVAL_SCORE_COLUMNS:
        out[column] = rng.uniform(0.1, 0.95, size=n).astype(np.float32)
    for column in NAME_SIMILARITY_COLUMNS:
        out[column] = rng.uniform(0.05, 0.99, size=n).astype(np.float32)
    for column in ADDR_SIMILARITY_COLUMNS:
        out[column] = rng.uniform(0.05, 0.99, size=n).astype(np.float32)

    is_true = np.array(
        [c in ground_truth[s] for s, c in zip(out["s1_id"], out["candidate_id"])],
        dtype=bool,
    )
    # True matches get systematically higher similarity so a model can learn.
    boost = np.where(is_true, 0.15, -0.1)
    for column in NAME_SIMILARITY_COLUMNS + ADDR_SIMILARITY_COLUMNS:
        out[column] = np.clip(out[column].to_numpy() + boost, 0.0, 1.0).astype(np.float32)
    out["best_score"] = np.clip(out["best_score"].to_numpy() + boost, 0.0, 1.0).astype(np.float32)

    out["name_exact_conservative"] = is_true & (rng.uniform(size=n) < 0.5)
    out["name_exact_aggressive"] = is_true & (rng.uniform(size=n) < 0.7)
    out["name_len_diff"] = rng.randint(0, 6, size=n).astype(np.float32)
    out["name_token_count_diff"] = rng.randint(0, 3, size=n).astype(np.float32)
    out["addr_numeric_conflict"] = ~is_true & (rng.uniform(size=n) < 0.3)
    out["addr_len_diff"] = rng.randint(0, 25, size=n).astype(np.float32)

    same_country = np.array(
        [_country_for(s) == _country_for(c) for s, c in zip(out["s1_id"], out["candidate_id"])],
        dtype=bool,
    )
    country_eq = np.where(same_country, 1.0, 0.0).astype(np.float32)
    country_eq[rng.uniform(size=n) < 0.05] = np.nan
    out["country_eq"] = country_eq

    out["s1_name_missing"] = False
    out["cand_name_missing"] = rng.uniform(size=n) < 0.07
    out["s1_addr_missing"] = rng.uniform(size=n) < 0.05
    out["cand_addr_missing"] = rng.uniform(size=n) < 0.12

    # Deliberate NaNs so NaN handling is exercised by the model tests.
    for column in ("addr_char_cos", "addr_contain_idf", "addr_numeric_jaccard"):
        values = out[column].to_numpy().copy()
        values[rng.uniform(size=n) < 0.08] = np.nan
        out[column] = values.astype(np.float32)

    return out[FEATURE_COLUMNS]


@pytest.fixture
def synthetic_s1_pool() -> pd.DataFrame:
    """50 S1 entities with a deterministic country distribution."""
    return _build_s1_pool()


@pytest.fixture
def synthetic_ground_truth() -> "dict[str, set[str]]":
    """Mapping from S1 id to the set of true candidate ids."""
    return _build_ground_truth()


@pytest.fixture
def synthetic_candidates_df(
    synthetic_ground_truth: "dict[str, set[str]]",
) -> pd.DataFrame:
    """Candidate pairs with retrieval metadata; contains every true match."""
    return _build_candidates(synthetic_ground_truth)


@pytest.fixture
def synthetic_features_df(
    synthetic_candidates_df: pd.DataFrame,
    synthetic_ground_truth: "dict[str, set[str]]",
) -> pd.DataFrame:
    """Plausible feature matrix matching the canonical feature contract."""
    rng = np.random.RandomState(42)
    return _build_features(synthetic_candidates_df, synthetic_ground_truth, rng)


@pytest.fixture
def synthetic_folds_df(
    synthetic_candidates_df: pd.DataFrame,
    synthetic_s1_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Frozen-style 5-fold assignment over the synthetic S1 pool."""
    from src.validation.splits import assign_folds_from_ground_truth

    fold_assignments = assign_folds_from_ground_truth(
        s1_ids=list(synthetic_s1_pool["entity_id"]),
        ground_truth_by_s1=_build_ground_truth(),
        countries=dict(zip(synthetic_s1_pool["entity_id"], synthetic_s1_pool["country"])),
        n_folds=5,
        random_state=42,
    )
    folds = pd.DataFrame(
        {
            "s1_id": list(fold_assignments.keys()),
            "fold_id": np.array(list(fold_assignments.values()), dtype=np.int8),
        }
    )
    return synthetic_candidates_df[["pair_key", "s1_id"]].merge(folds, on="s1_id", how="left")
