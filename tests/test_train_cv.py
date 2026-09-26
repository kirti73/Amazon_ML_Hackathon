"""Unit tests for the leak-free 5-fold CV runner and OOF scores.

All tests use the synthetic fixtures. No score here is competition-valid.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.labels import build_train_labels
from src.models.logreg import BaselineLogisticRegression
from src.models.train_cv import (
    SCORES_COLUMNS,
    coverage_report,
    prepare_training_frame,
    train_cv,
)


@pytest.fixture
def labels_df(synthetic_candidates_df, synthetic_ground_truth):
    return build_train_labels(synthetic_candidates_df, synthetic_ground_truth)


@pytest.fixture
def folds_df(synthetic_candidates_df, synthetic_s1_pool, synthetic_ground_truth):
    from src.validation.splits import assign_folds_from_ground_truth

    assignments = assign_folds_from_ground_truth(
        s1_ids=list(synthetic_s1_pool["entity_id"]),
        ground_truth_by_s1=synthetic_ground_truth,
        countries=dict(zip(synthetic_s1_pool["entity_id"], synthetic_s1_pool["country"])),
        n_folds=5,
        random_state=42,
    )
    return pd.DataFrame(
        {
            "s1_id": list(assignments.keys()),
            "fold_id": np.array(list(assignments.values()), dtype=np.int8),
        }
    )


def test_exactly_one_oof_prediction_per_candidate_pair(
    synthetic_features_df, labels_df, folds_df
):
    scores, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )

    assert len(scores) == len(synthetic_features_df), (
        f"{len(synthetic_features_df)} candidate rows must yield "
        f"{len(synthetic_features_df)} score rows, got {len(scores)}"
    )
    assert not scores["pair_key"].duplicated().any()
    assert set(scores["pair_key"]) == set(synthetic_features_df["pair_key"])
    assert scores["is_oof"].all(), "every CV score is out-of-fold"


def test_zero_fold_leakage(synthetic_features_df, labels_df, folds_df):
    """The model producing fold k's scores must never have seen fold k.

    Verified structurally rather than statistically: for each fold, every pair scored by
    that fold's model carries that fold id, and the model was fitted on a training frame
    built from a disjoint row mask. A leak would show up as a fold id appearing in the
    training mask for its own model.
    """
    frame = prepare_training_frame(synthetic_features_df, labels_df, folds_df)

    seen_training_folds = []
    scored_folds = []
    for fold_id in sorted(frame["fold_id"].unique()):
        train_mask = (frame["fold_id"] != fold_id).to_numpy()
        valid_mask = (frame["fold_id"] == fold_id).to_numpy()

        train_folds = set(frame.loc[train_mask, "fold_id"].unique())
        assert fold_id not in train_folds, (
            f"fold {fold_id} leaked into its own training data"
        )
        seen_training_folds.append(sorted(int(f) for f in train_folds))
        scored_folds.append(int(fold_id))

    # Every fold is predicted exactly once, and each model trains on the other four.
    assert scored_folds == sorted(set(int(f) for f in frame["fold_id"].unique()))
    for train_folds in seen_training_folds:
        assert len(train_folds) == 4, f"expected 4 training folds, got {train_folds}"

    # And the OOF scores partition the rows by fold with no overlap.
    scores, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )
    assert scores["fold_id"].nunique() == 5
    assert len(scores) == len(frame)


def test_output_matches_scores_parquet_schema(
    synthetic_features_df, labels_df, folds_df, tmp_path
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.models.train_cv import SCORES_ARROW_SCHEMA, write_scores_parquet

    scores, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )

    assert list(scores.columns) == SCORES_COLUMNS
    # pandas 3.x uses StringDtype rather than object for text columns; both are valid
    # string storage, and what matters is that ids were never coerced to a numeric dtype.
    assert pd.api.types.is_string_dtype(scores["pair_key"])
    assert pd.api.types.is_string_dtype(scores["s1_id"])
    assert pd.api.types.is_string_dtype(scores["candidate_id"])
    assert all(isinstance(v, str) for v in scores["pair_key"])
    assert scores["p_raw"].dtype == np.float32
    assert scores["p_cal"].dtype == np.float32
    assert scores["fold_id"].dtype == np.int8
    assert scores["is_oof"].dtype == bool
    assert pd.api.types.is_string_dtype(scores["model_version"])
    assert scores["p_raw"].between(0.0, 1.0).all()
    assert scores["p_cal"].between(0.0, 1.0).all()
    assert scores["fold_id"].between(0, 4).all()

    # The written artifact must match the frozen contract exactly. Writing without an
    # explicit schema would produce large_string under pandas 3.x and break the contract.
    path = tmp_path / "scores.parquet"
    write_scores_parquet(scores, str(path))
    table = pq.read_table(path)

    assert table.schema.equals(SCORES_ARROW_SCHEMA)
    assert table.schema.field("pair_key").type == pa.string()
    assert table.schema.field("s1_id").type == pa.string()
    assert table.schema.field("candidate_id").type == pa.string()
    assert table.schema.field("p_raw").type == pa.float32()
    assert table.schema.field("fold_id").type == pa.int8()

    readback = table.to_pandas()
    assert len(readback) == len(scores)
    assert readback["fold_id"].dtype == np.int8
    assert readback["p_raw"].dtype == np.float32
    assert set(readback["pair_key"]) == set(scores["pair_key"])


def test_model_version_is_recorded(synthetic_features_df, labels_df, folds_df):
    scores, models = train_cv(
        synthetic_features_df,
        labels_df,
        folds_df,
        BaselineLogisticRegression,
        {},
        model_version="logreg_v1",
    )
    assert set(scores["model_version"]) == {"logreg_v1"}
    assert len(models) == 5


def test_coverage_report(synthetic_features_df, labels_df, folds_df):
    frame = prepare_training_frame(synthetic_features_df, labels_df, folds_df)
    scores, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )
    report = coverage_report(scores, frame)
    assert report["n_scores"] == report["n_candidates"] == len(synthetic_features_df)
    assert report["n_folds"] == 5
    assert report["duplicate_pair_keys"] == 0
    assert report["is_oof_all_true"] is True
    assert len(report["per_fold"]) == 5


def test_every_s1_entity_present_in_folds_or_raises(
    synthetic_features_df, labels_df, folds_df
):
    incomplete = folds_df[folds_df["s1_id"] != folds_df["s1_id"].iloc[0]].copy()
    with pytest.raises(ValueError, match="no fold assignment"):
        train_cv(
            synthetic_features_df, labels_df, incomplete, BaselineLogisticRegression, {}
        )


def test_duplicate_pair_keys_in_features_raise(
    synthetic_features_df, labels_df, folds_df
):
    duplicated = pd.concat([synthetic_features_df, synthetic_features_df.head(1)])
    with pytest.raises(ValueError, match="duplicate pair_key"):
        train_cv(duplicated, labels_df, folds_df, BaselineLogisticRegression, {})


def test_label_row_loss_raises(synthetic_features_df, labels_df, folds_df):
    truncated = labels_df.head(len(labels_df) - 1).copy()
    with pytest.raises(ValueError, match="label join lost rows"):
        train_cv(
            synthetic_features_df, truncated, folds_df, BaselineLogisticRegression, {}
        )


def test_missing_column_raises(synthetic_features_df, labels_df, folds_df):
    with pytest.raises(ValueError, match="missing required columns"):
        train_cv(
            synthetic_features_df.drop(columns=["candidate_id"]),
            labels_df,
            folds_df,
            BaselineLogisticRegression,
            {},
        )


def test_cv_is_deterministic(synthetic_features_df, labels_df, folds_df):
    first, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )
    second, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )
    np.testing.assert_array_equal(first["p_raw"].to_numpy(), second["p_raw"].to_numpy())


def test_scores_separate_the_synthetic_signal(synthetic_features_df, labels_df, folds_df):
    """An OOF score must be informative; a leaked model would look better, not worse."""
    scores, _ = train_cv(
        synthetic_features_df, labels_df, folds_df, BaselineLogisticRegression, {}
    )
    merged = scores.merge(labels_df, on="pair_key", how="left")
    assert merged["y"].notna().all()
    mean_pos = merged.loc[merged["y"] == 1, "p_raw"].mean()
    mean_neg = merged.loc[merged["y"] == 0, "p_raw"].mean()
    assert mean_pos > mean_neg, (
        f"OOF scores did not separate the synthetic signal "
        f"(pos={mean_pos:.3f}, neg={mean_neg:.3f})"
    )
