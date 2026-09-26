"""Unit tests for the threshold tuning and prediction decision layer.

All tests use the synthetic fixtures. No score here is competition-valid.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.decision.threshold import (
    apply_decision_rules,
    build_decision_config,
    evaluate_thresholds,
    load_decision_config,
    report_decision,
    save_decision_config,
    tune_global_threshold,
    tune_two_thresholds,
)


@pytest.fixture
def oof_scores(synthetic_candidates_df, synthetic_ground_truth):
    """A minimal, explicit OOF score frame: true pairs score high, negatives low."""
    rows = []
    for _, row in synthetic_candidates_df.iterrows():
        is_true = row["candidate_id"] in synthetic_ground_truth[row["s1_id"]]
        rows.append(
            {
                "pair_key": row["pair_key"],
                "s1_id": row["s1_id"],
                "candidate_id": row["candidate_id"],
                "p_raw": 0.9 if is_true else 0.2,
                "p_cal": 0.9 if is_true else 0.2,
                "fold_id": np.int8(0),
                "is_oof": True,
                "model_version": "test_v1",
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- decision rules
def test_empty_prediction_when_max_score_below_t1(oof_scores):
    """An entity whose best candidate cannot clear T1 must predict nothing."""
    predictions = apply_decision_rules(oof_scores, t1=0.95, t2=0.95)

    assert all(matched == set() for matched in predictions.values()), (
        "no candidate reaches 0.95, so every entity must abstain"
    )
    assert len(predictions) == oof_scores["s1_id"].nunique()


def test_multi_match_continuation_uses_t2():
    """One strong candidate is accepted, further candidates must clear T2."""
    scores = pd.DataFrame(
        {
            "pair_key": [
                "S1-1::S2-1",
                "S1-1::S2-2",
                "S1-1::S2-3",
                "S1-2::S2-9",
            ],
            "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
            "candidate_id": ["S2-1", "S2-2", "S2-3", "S2-9"],
            "p_cal": [0.90, 0.60, 0.30, 0.80],
            "is_oof": True,
        }
    )

    # T1 = 0.5 admits the 0.60 and 0.80 candidates; T2 = 0.5 keeps both.
    predictions = apply_decision_rules(scores, t1=0.5, t2=0.5)
    assert predictions["S1-1"] == {"S2-1", "S2-2"}, "0.60 clears T2=0.5 and is kept"
    assert predictions["S1-2"] == {"S2-9"}
    assert "S2-3" not in predictions["S1-1"], "0.30 is below T2 and must be dropped"

    # Raising T2 above the second candidate excludes it: one match protected.
    predictions = apply_decision_rules(scores, t1=0.5, t2=0.85)
    assert predictions["S1-1"] == {"S2-1"}, "only the 0.90 candidate clears T2=0.85"
    assert predictions["S1-2"] == set(), "0.80 is eligible for entry but not for T2=0.85"


def test_entry_and_continuation_gates_are_distinct():
    """T2 > T1 lets an entity accept a first strong match but reject weak extras."""
    scores = pd.DataFrame(
        {
            "pair_key": ["S1-1::S2-1", "S1-1::S2-2"],
            "s1_id": ["S1-1", "S1-1"],
            "candidate_id": ["S2-1", "S2-2"],
            "p_cal": [0.70, 0.40],
            "is_oof": True,
        }
    )
    predictions = apply_decision_rules(scores, t1=0.40, t2=0.60)
    assert predictions["S1-1"] == {"S2-1"}


def test_deterministic_and_order_independent(oof_scores):
    forward = apply_decision_rules(oof_scores, t1=0.5, t2=0.5)
    shuffled = apply_decision_rules(
        oof_scores.sample(frac=1.0, random_state=7).reset_index(drop=True), t1=0.5, t2=0.5
    )
    assert forward == shuffled, "prediction must not depend on input row order"

    again = apply_decision_rules(oof_scores, t1=0.5, t2=0.5)
    assert forward == again, "repeated calls must be identical"


def test_duplicate_candidate_ids_never_appear():
    """A repeated candidate row must not produce a duplicate prediction id."""
    scores = pd.DataFrame(
        {
            "pair_key": ["S1-1::S2-1", "S1-1::S2-1", "S1-1::S2-2"],
            "s1_id": ["S1-1", "S1-1", "S1-1"],
            "candidate_id": ["S2-1", "S2-1", "S2-2"],
            "p_cal": [0.90, 0.90, 0.80],
            "is_oof": True,
        }
    )
    predictions = apply_decision_rules(scores, t1=0.5, t2=0.5)
    assert predictions["S1-1"] == {"S2-1", "S2-2"}
    for matched in predictions.values():
        assert isinstance(matched, set)
        assert len(matched) == len(set(matched))


def test_duplicate_scores_tie_break_deterministically():
    """Equal scores must break ties by candidate id, not by row order."""
    base = {
        "s1_id": ["S1-1", "S1-1"],
        "candidate_id": ["S2-2", "S2-1"],
        "p_cal": [0.7, 0.7],
        "is_oof": True,
    }
    forward = pd.DataFrame({**base, "pair_key": ["S1-1::S2-2", "S1-1::S2-1"]})
    reversed_order = forward.iloc[::-1].reset_index(drop=True)
    assert apply_decision_rules(forward, 0.5, 0.5) == apply_decision_rules(
        reversed_order, 0.5, 0.5
    )


# ------------------------------------------------------------------------- tuning
def test_global_threshold_separates_clean_signal(oof_scores, synthetic_ground_truth):
    threshold = tune_global_threshold(oof_scores, synthetic_ground_truth, n_grid=41)
    assert 0.2 < threshold <= 0.9, f"unexpected threshold {threshold}"

    score = evaluate_thresholds(oof_scores, synthetic_ground_truth, threshold, threshold)
    assert score > 0.9, f"a cleanly separable fixture should score high, got {score}"


def test_two_threshold_tuning_never_returns_t2_below_t1(oof_scores, synthetic_ground_truth):
    t1, t2 = tune_two_thresholds(oof_scores, synthetic_ground_truth, n_grid=21)
    assert t2 >= t1, f"T2 ({t2}) must not be below T1 ({t1})"
    assert 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0


def test_two_threshold_is_at_least_as_good_as_global(
    oof_scores, synthetic_ground_truth
):
    """The two-threshold search space contains the global one, so it cannot do worse."""
    t1, t2 = tune_two_thresholds(oof_scores, synthetic_ground_truth, n_grid=21)
    two_score = evaluate_thresholds(oof_scores, synthetic_ground_truth, t1, t2)

    t_global = tune_global_threshold(oof_scores, synthetic_ground_truth, n_grid=21)
    global_score = evaluate_thresholds(oof_scores, synthetic_ground_truth, t_global, t_global)

    assert two_score >= global_score - 1e-12, (
        f"two-threshold {two_score:.6f} worse than global {global_score:.6f}"
    )


def test_tuning_is_deterministic(oof_scores, synthetic_ground_truth):
    first = tune_two_thresholds(oof_scores, synthetic_ground_truth, n_grid=15)
    second = tune_two_thresholds(oof_scores, synthetic_ground_truth, n_grid=15)
    assert first == second


def test_fast_path_matches_reference_implementation(oof_scores):
    """The vectorised tuning path must agree exactly with the readable row-by-row rules.

    ``_PreparedScores`` derives a closed form from the fact that candidates are sorted by
    descending score. That optimisation is only safe if it is provably identical to
    :func:`apply_decision_rules`, including the ``T2 < T1`` ordering the default search
    excludes, so it is checked across a spread of thresholds and both orderings.
    """
    from src.decision.threshold import _PreparedScores

    prepared = _PreparedScores(oof_scores, "p_cal")
    for t1 in (0.0, 0.2, 0.5, 0.9, 1.5):
        for t2 in (0.0, 0.2, 0.5, 0.9, 1.5):
            reference = apply_decision_rules(oof_scores, t1=t1, t2=t2)
            fast = prepared.predictions(t1, t2)
            assert fast == reference, (
                f"fast path disagrees with reference at t1={t1}, t2={t2}"
            )


def test_fast_path_handles_shuffled_input(oof_scores):
    """Sorting happens once internally, so input row order must not matter."""
    from src.decision.threshold import _PreparedScores

    reference = _PreparedScores(oof_scores, "p_cal").predictions(0.5, 0.5)
    shuffled = oof_scores.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert _PreparedScores(shuffled, "p_cal").predictions(0.5, 0.5) == reference


def test_tuning_rejects_non_oof_scores(oof_scores, synthetic_ground_truth):
    """Tuning on in-fold scores would be self-deception, so it must be refused."""
    contaminated = oof_scores.copy()
    contaminated.loc[0, "is_oof"] = False
    with pytest.raises(ValueError, match="out-of-fold"):
        tune_global_threshold(contaminated, synthetic_ground_truth, n_grid=5)
    with pytest.raises(ValueError, match="out-of-fold"):
        tune_two_thresholds(contaminated, synthetic_ground_truth, n_grid=5)


def test_tuning_rejects_missing_columns(oof_scores, synthetic_ground_truth):
    with pytest.raises(ValueError, match="missing required columns"):
        tune_global_threshold(oof_scores.drop(columns=["p_cal"]), synthetic_ground_truth)


# ------------------------------------------------------------------ metric behavior
def test_abstaining_everywhere_scores_the_singleton_floor(
    synthetic_candidates_df, synthetic_ground_truth
):
    """Predicting nothing must score 1.0 on zero-match entities, not 0.0."""
    high = apply_decision_rules(
        pd.DataFrame(
            {
                "pair_key": synthetic_candidates_df["pair_key"],
                "s1_id": synthetic_candidates_df["s1_id"],
                "candidate_id": synthetic_candidates_df["candidate_id"],
                "p_cal": 0.99,
                "is_oof": True,
            }
        ),
        t1=1.01,
        t2=1.01,
    )
    score = evaluate_thresholds(
        pd.DataFrame(
            {
                "pair_key": synthetic_candidates_df["pair_key"],
                "s1_id": synthetic_candidates_df["s1_id"],
                "candidate_id": synthetic_candidates_df["candidate_id"],
                "p_cal": 0.99,
                "is_oof": True,
            }
        ),
        synthetic_ground_truth,
        1.01,
        1.01,
    )
    expected = sum(1 for m in synthetic_ground_truth.values() if not m) / len(
        synthetic_ground_truth
    )
    assert score == pytest.approx(expected, abs=1e-9), (
        f"predict-nothing score {score:.4f} should equal the zero-match share {expected:.4f}"
    )
    assert all(matched == set() for matched in high.values())


def test_false_positive_on_singleton_is_catastrophic(
    synthetic_candidates_df, synthetic_ground_truth
):
    """Emitting anything for a true singleton zeroes that entity, so abstention matters."""
    singleton = next(s1 for s1, matches in synthetic_ground_truth.items() if not matches)
    scores = pd.DataFrame(
        {
            "pair_key": [f"{singleton}::S2-9999"],
            "s1_id": [singleton],
            "candidate_id": ["S2-9999"],
            "p_cal": [0.5],
            "is_oof": True,
        }
    )
    violating = evaluate_thresholds(scores, synthetic_ground_truth, 0.4, 0.4)
    abstaining = evaluate_thresholds(scores, synthetic_ground_truth, 0.6, 0.6)
    assert violating < abstaining, (
        f"abstaining ({abstaining:.4f}) must beat a false match ({violating:.4f})"
    )


# ------------------------------------------------------------------- config & report
def test_decision_config_roundtrip(tmp_path):
    config = build_decision_config(
        t_global=0.4, t1=0.35, t2=0.55, model_version="logreg_v1", n_oof_rows=179
    )
    assert config["tuned_on"] == "oof"
    assert config["n_oof_rows"] == 179
    assert config["enable_reverse_consistency"] is False

    path = save_decision_config(config, tmp_path / "decision_config.json")
    assert path.is_file()
    loaded = load_decision_config(path)
    assert loaded == config

    # And it is plain, reviewable JSON.
    with path.open() as handle:
        raw = json.load(handle)
    assert set(raw) >= {"t_global", "t1", "t2", "model_version", "n_oof_rows", "tuned_on"}


def test_report_decision_summarises_abstentions(oof_scores, synthetic_ground_truth):
    report = report_decision(oof_scores, synthetic_ground_truth, t1=0.5, t2=0.5)

    assert report["n_entities"] == len(synthetic_ground_truth)
    assert report["n_singleton_entities"] == sum(
        1 for m in synthetic_ground_truth.values() if not m
    )
    assert 0.0 <= report["macro_f05"] <= 1.0
    assert report["n_predictions"] == sum(
        1 for row in oof_scores.itertuples() if row.p_cal >= 0.5
    )
    assert report["n_singleton_abstained"] <= report["n_singleton_entities"]


def test_reverse_consistency_is_a_documented_noop_without_the_column(
    oof_scores, synthetic_ground_truth
):
    """With no reverse-retrieval column the flag must not silently change behaviour."""
    with_flag = apply_decision_rules(oof_scores, 0.5, 0.5, enable_reverse_consistency=True)
    without_flag = apply_decision_rules(oof_scores, 0.5, 0.5, enable_reverse_consistency=False)
    assert with_flag == without_flag


def test_reverse_consistency_filters_when_column_present(oof_scores):
    scores = oof_scores.head(20).copy()
    scores["retrieved_reverse"] = [False] * len(scores)
    predictions = apply_decision_rules(
        scores, t1=0.0, t2=0.0, enable_reverse_consistency=True
    )
    assert all(matched == set() for matched in predictions.values())


def test_entity_with_no_rows_yields_no_prediction(oof_scores):
    predictions = apply_decision_rules(oof_scores, t1=0.5, t2=0.5)
    assert "S1-NOT-PRESENT" not in predictions
