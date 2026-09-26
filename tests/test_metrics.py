"""Unit tests for the macro-F0.5 metric implementation.

All tests use small hand-computed examples. No real dataset is used.
"""

import pytest

from src.validation.metrics import (
    compute_per_entity_f05,
    macro_f05,
    build_ground_truth_from_tsv,
    build_predictions_from_dict,
)


def test_perfect_prediction_non_empty_entity():
    """Perfect prediction for a non-empty entity -> score = 1.0."""
    ground_truth = {"S1-001": {"S2-100", "S2-200"}}
    predictions = {"S1-001": {"S2-100", "S2-200"}}

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 1.0


def test_zero_match_entity_empty_prediction():
    """Zero-match entity + empty prediction -> score = 1.0."""
    ground_truth = {"S1-001": set()}  # empty set means zero matches
    predictions = {"S1-001": set()}

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 1.0


def test_zero_match_entity_false_prediction():
    """Zero-match entity + false prediction -> score = 0.0."""
    ground_truth = {"S1-001": set()}
    predictions = {"S1-001": {"S2-999"}}  # false match

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 0.0


def test_non_empty_gt_empty_prediction():
    """Non-empty ground truth + empty prediction -> score = 0.0."""
    ground_truth = {"S1-001": {"S2-100"}}
    predictions = {"S1-001": set()}

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 0.0


def test_non_empty_gt_completely_wrong_prediction():
    """Non-empty ground truth + completely wrong prediction -> score = 0.0."""
    ground_truth = {"S1-001": {"S2-100"}}
    predictions = {"S1-001": {"S2-999"}}  # different entity, 0 overlap

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 0.0


def test_partial_overlap_correct_f05():
    """Partial overlap -> correct F0.5 calculation."""
    # S1 has 2 true matches; prediction has 1 true + 1 false
    ground_truth = {"S1-001": {"S2-100", "S2-200"}}
    predictions = {"S1-001": {"S2-100", "S2-300"}}  # 1 TP, 1 FP

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    # TP=1, |P_i|=2, |T_i|=2
    # precision = 1/2 = 0.5
    # recall = 1/2 = 0.5
    # F0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    assert abs(scores["S1-001"] - 0.5) < 1e-10


def test_duplicate_predicted_ids_do_not_inflate():
    """Duplicate predicted IDs do not inflate TP (sets handle this)."""
    ground_truth = {"S1-001": {"S2-100"}}
    predictions = {"S1-001": {"S2-100", "S2-100"}}  # duplicate, but set dedupes

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 1.0  # TP=1, precision=1/1=1, recall=1/1=1, F0.5=1.0


def test_multiple_s1_entities_correct_macro_average():
    """Multiple S1 entities -> correct macro average."""
    ground_truth = {
        "S1-001": {"S2-100"},      # 1 true match
        "S1-002": {"S2-200", "S2-300"},  # 2 true matches
    }
    predictions = {
        "S1-001": {"S2-100"},      # perfect: F0.5 = 1.0
        "S1-002": {"S2-200"},      # 1 TP, 1 FN, 0 FP: precision=1, recall=0.5
    }

    scores = compute_per_entity_f05(ground_truth, predictions)
    # S1-001: TP=1, P=1, T=1 => precision=1, recall=1 => F0.5 = (1.25*1*1)/(0.25*1+1) = 1.25/1.25 = 1.0
    # S1-002: TP=1, P=1, T=2 => precision=1, recall=0.5 => F0.5 = (1.25*1*0.5)/(0.25*1+0.5) = 0.625/0.75 = 5/6
    macro = macro_f05(ground_truth, predictions)
    expected = (1.0 + 5 / 6) / 2  # (1.0 + 0.8333...)/2
    assert abs(macro - expected) < 1e-10


def test_prediction_order_does_not_matter():
    """Prediction order does not matter (sets are unordered)."""
    ground_truth = {"S1-001": {"S2-100", "S2-200"}}
    predictions = {"S1-001": {"S2-200", "S2-100"}}  # reversed order

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert scores["S1-001"] == 1.0


def test_string_ids_remain_strings():
    """IDs are treated as strings even if numeric-looking."""
    ground_truth = {"S1-001": {"S2-100"}, "S1-002": {"S2-200"}}
    predictions = {"S1-001": {"S2-100"}, "S1-002": {"S2-200"}}

    # Verify keys are strings
    assert all(isinstance(k, str) for k in ground_truth.keys())
    assert all(isinstance(k, str) for k in predictions.keys())

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert "S1-002" in scores


def test_entity_missing_from_predictions_treated_as_empty():
    """Entity missing from predictions -> treated as empty prediction -> score depends on GT."""
    ground_truth = {"S1-001": {"S2-100"}, "S1-002": set()}  # S1-002 is zero-match

    # S1-001 IS in predictions -> perfect
    # S1-002 is NOT in predictions -> treated as empty prediction
    predictions = {"S1-001": {"S2-100"}}  # S1-002 key missing

    scores = compute_per_entity_f05(ground_truth, predictions)
    assert "S1-001" in scores
    assert "S1-002" in scores
    # S1-002: zero-match + empty prediction -> score = 1.0
    assert scores["S1-002"] == 1.0