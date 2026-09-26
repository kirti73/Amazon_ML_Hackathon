"""Unit tests for the binary label / hard-negative builder.

All tests use the synthetic fixtures from ``tests/conftest.py``. No real dataset is
loaded and no score here is competition-valid.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.labels import (
    build_train_labels,
    build_true_positive_pairs,
    candidate_recall,
    make_pair_key,
)


def _wide_gt(ground_truth):
    """Ground truth in the real dataset layout: comma-separated matches per S1 row."""
    rows = []
    for s1_id in sorted(ground_truth):
        rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(sorted(ground_truth[s1_id])),
            }
        )
    return pd.DataFrame(rows)


def test_positive_matches_are_labeled_one(synthetic_candidates_df, synthetic_ground_truth):
    labels = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)
    merged = synthetic_candidates_df.merge(labels, on="pair_key", how="left")

    positives = merged[merged["y"] == 1]
    assert len(positives) > 0, "fixture must contain true matches"

    for _, row in positives.iterrows():
        assert row["candidate_id"] in synthetic_ground_truth[row["s1_id"]], (
            f"pair {row['pair_key']} labeled 1 but is not in ground truth"
        )

    # And the converse: every ground-truth pair present in candidates is labeled 1.
    for s1_id, matches in synthetic_ground_truth.items():
        for candidate_id in matches:
            key = make_pair_key(s1_id, candidate_id)
            in_candidates = (synthetic_candidates_df["pair_key"] == key).any()
            if in_candidates:
                assert labels.loc[labels["pair_key"] == key, "y"].iloc[0] == 1


def test_unmatched_candidates_are_labeled_zero(synthetic_candidates_df, synthetic_ground_truth):
    labels = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)
    merged = synthetic_candidates_df.merge(labels, on="pair_key", how="left")

    negatives = merged[merged["y"] == 0]
    assert len(negatives) > 0, "fixture must contain hard negatives"

    for _, row in negatives.iterrows():
        assert row["candidate_id"] not in synthetic_ground_truth[row["s1_id"]], (
            f"pair {row['pair_key']} labeled 0 but is a true match"
        )


def test_zero_hallucinated_rows(synthetic_candidates_df, synthetic_ground_truth):
    """Output row count must equal candidate row count, order preserved, no dupes."""
    labels = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)

    assert len(labels) == len(synthetic_candidates_df), (
        f"expected {len(synthetic_candidates_df)} label rows, got {len(labels)}"
    )
    assert list(labels.columns) == ["pair_key", "y"]
    assert not labels["pair_key"].duplicated().any(), "label rows must be 1:1 with candidates"
    assert labels["y"].notna().all()
    assert set(labels["y"].unique()) <= {0, 1}


def test_labels_dtypes_are_int8_and_string(synthetic_candidates_df, synthetic_ground_truth):
    labels = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)

    assert labels["y"].dtype == np.int8, f"y dtype is {labels['y'].dtype}, expected int8"
    assert all(isinstance(v, str) for v in labels["pair_key"])
    assert all(isinstance(v, str) for v in synthetic_candidates_df["s1_id"])
    assert all(isinstance(v, str) for v in synthetic_candidates_df["candidate_id"])


def test_wide_ground_truth_format_matches_long_format(
    synthetic_candidates_df, synthetic_ground_truth
):
    """The real comma-separated TSV layout must label identically to the mapping form."""
    from_mapping = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)
    from_wide = build_train_labels(synthetic_candidates_df, _wide_gt(synthetic_ground_truth))

    pd.testing.assert_frame_equal(from_mapping, from_wide)


def test_zero_match_entities_contribute_no_positives(synthetic_candidates_df, synthetic_ground_truth):
    labels = build_train_labels(synthetic_candidates_df, synthetic_ground_truth)
    merged = synthetic_candidates_df.merge(labels, on="pair_key", how="left")

    zero_match_s1 = [s1 for s1, matches in synthetic_ground_truth.items() if not matches]
    assert zero_match_s1, "fixture must contain zero-match entities"

    for s1_id in zero_match_s1:
        rows = merged[merged["s1_id"] == s1_id]
        assert len(rows) > 0
        assert (rows["y"] == 0).all(), (
            f"zero-match entity {s1_id} must have no positive candidates"
        )


def test_empty_ground_truth_labels_everything_zero(synthetic_candidates_df):
    """An empty ground truth still has to carry the correct schema.

    A frame with no rows and the right columns means "no true pairs exist", which is a
    legitimate all-negative label set. A frame with the wrong columns is malformed and
    must raise rather than silently yield all-zero labels.
    """
    empty_wide = pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    labels = build_train_labels(synthetic_candidates_df, empty_wide)
    assert len(labels) == len(synthetic_candidates_df)
    assert (labels["y"] == 0).all()
    assert labels["y"].dtype == np.int8


def test_all_singleton_ground_truth_labels_everything_zero(
    synthetic_candidates_df, synthetic_s1_pool
):
    """Every S1 entity a true singleton: no positives anywhere."""
    singleton_gt = pd.DataFrame(
        {
            "source1_entity_id": list(synthetic_s1_pool["entity_id"]),
            "matched_entity_ids": ["" for _ in range(len(synthetic_s1_pool))],
        }
    )
    labels = build_train_labels(synthetic_candidates_df, singleton_gt)
    assert len(labels) == len(synthetic_candidates_df)
    assert (labels["y"] == 0).all()


def test_candidate_recall_is_full_on_synthetic_fixture(
    synthetic_candidates_df, synthetic_ground_truth
):
    covered, total, recall = candidate_recall(synthetic_candidates_df, synthetic_ground_truth)
    assert total == sum(len(v) for v in synthetic_ground_truth.values())
    assert covered == total, "synthetic candidates must contain every true pair"
    assert recall == 1.0


def test_build_true_positive_pairs_is_key_based():
    pairs = build_true_positive_pairs({"S1-1": {"S2-2", "S3-3"}})
    assert pairs == {"S1-1::S2-2", "S1-1::S3-3"}


def test_malformed_inputs_raise(synthetic_candidates_df, synthetic_ground_truth):
    with pytest.raises(ValueError):
        build_train_labels(synthetic_candidates_df.drop(columns=["pair_key"]), synthetic_ground_truth)

    with pytest.raises(TypeError):
        build_train_labels("not a dataframe", synthetic_ground_truth)

    with pytest.raises(ValueError):
        build_train_labels(synthetic_candidates_df, pd.DataFrame({"wrong": ["a"]}))
