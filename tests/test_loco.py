"""Unit tests for the LOCO (leave-one-country-out) robustness diagnostic.

The synthetic pool cycles US / India / France, so the *mechanics* of all three country
directions are testable here even though the real training data has only US and India.
That separation is deliberate: the real-data constraint (France is test-only) and the
code path (any country pair can be sliced) are verified independently.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.logreg import BaselineLogisticRegression
from src.models.train_cv import prepare_training_frame
from src.validation.loco import (
    DEFAULT_COUNTRY_PAIRS,
    UNTUNED_THRESHOLD,
    format_loco_report,
    prepare_loco_frame,
    run_loco_direction,
    run_loco_evaluation,
)


class RecordingModel:
    """The real baseline, wrapped so tests can see exactly which rows it was trained on.

    Real probabilities matter â€” a fake constant-score model would make every threshold
    comparison degenerate â€” but the recorded fit frame is what proves the held-out
    country never leaked into training.
    """

    instances = []

    def __init__(self, **kwargs):
        self._inner = BaselineLogisticRegression(**kwargs)
        self.kwargs = kwargs
        self.fit_X = None
        self.fit_y = None
        RecordingModel.instances.append(self)

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = np.asarray(y).copy()
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._inner.predict_proba(X)


@pytest.fixture
def recording_model():
    RecordingModel.instances = []
    yield RecordingModel
    RecordingModel.instances = []


@pytest.fixture
def synthetic_labels_df(synthetic_candidates_df, synthetic_ground_truth):
    """Binary labels built by the real labeler, so LOCO is tested on the real contract."""
    from src.models.labels import build_train_labels

    return build_train_labels(synthetic_candidates_df, synthetic_ground_truth)


@pytest.fixture
def countries_by_s1(synthetic_s1_pool):
    return dict(zip(synthetic_s1_pool["entity_id"], synthetic_s1_pool["country"]))


@pytest.fixture
def countries_in_frame(countries_by_s1, synthetic_features_df):
    """Country mapping restricted to entities that actually have candidate rows."""
    wanted = set(synthetic_features_df["s1_id"])
    return {s1: c for s1, c in countries_by_s1.items() if s1 in wanted}


# ------------------------------------------------------------------- frame assembly
def test_prepare_loco_frame_attaches_country(
    synthetic_features_df, synthetic_labels_df, countries_in_frame
):
    frame = prepare_loco_frame(
        synthetic_features_df, synthetic_labels_df, countries_in_frame
    )
    assert "country" in frame.columns
    assert "y" in frame.columns and "fold_id" in frame.columns
    assert not frame["country"].isna().any()
    assert set(frame["country"]) == set(countries_in_frame.values())


def test_prepare_loco_frame_preserves_every_candidate_row(
    synthetic_features_df, synthetic_labels_df, countries_in_frame
):
    frame = prepare_loco_frame(
        synthetic_features_df, synthetic_labels_df, countries_in_frame
    )
    assert len(frame) == len(synthetic_features_df), "LOCO must not drop candidate rows"
    assert not frame["pair_key"].duplicated().any()


def test_prepare_loco_frame_rejects_non_mapping(
    synthetic_features_df, synthetic_labels_df, synthetic_s1_pool
):
    with pytest.raises(TypeError, match="must be a Mapping"):
        prepare_loco_frame(
            synthetic_features_df, synthetic_labels_df, synthetic_s1_pool
        )


def test_prepare_loco_frame_reports_entities_without_a_country(
    synthetic_features_df, synthetic_labels_df, countries_in_frame
):
    partial = dict(countries_in_frame)
    partial.pop(next(iter(partial)))
    with pytest.raises(ValueError, match="no country"):
        prepare_loco_frame(synthetic_features_df, synthetic_labels_df, partial)


def test_prepare_loco_frame_still_validates_labels(
    synthetic_features_df, synthetic_labels_df, countries_in_frame
):
    """Coverage checks are inherited, not bypassed, by taking the country route."""
    truncated = synthetic_labels_df.head(10)
    with pytest.raises(ValueError, match="missing required columns|label join"):
        prepare_loco_frame(synthetic_features_df, truncated, countries_in_frame)


# ---------------------------------------------------------------------- direction runs
def test_both_default_directions_complete(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )

    assert len(report["directions"]) == 2
    assert not any(d.get("skipped") for d in report["directions"]), (
        f"a default direction was skipped: {report['directions']}"
    )
    assert [d["test_country"] for d in report["directions"]] == ["India", "US"]
    assert report["summary"]["n_directions_completed"] == 2
    assert report["summary"]["n_directions_skipped"] == 0


def test_default_country_pairs_are_us_and_india_only():
    """France is test-only in the real data, so it must not be a default direction."""
    assert DEFAULT_COUNTRY_PAIRS == (("US", "India"), ("India", "US"))


def test_model_never_trains_on_the_held_out_country(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    """The core anti-leak guarantee of LOCO, asserted directly on the fit frame."""
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )
    assert len(recording_model.instances) == 2, "one fit per direction is expected"

    for model, direction in zip(recording_model.instances, report["directions"]):
        seen_countries = {
            countries_in_frame[s1] for s1 in model.fit_X["s1_id"]
        }
        assert seen_countries == {direction["train_country"]}, (
            f"model for {direction['train_country']} -> {direction['test_country']} "
            f"was fitted on {seen_countries}"
        )
        assert direction["test_country"] not in seen_countries, (
            f"held-out country {direction['test_country']} leaked into training"
        )
        assert "country" not in model.fit_X.columns, (
            "the country column must not be offered to the model as a feature"
        )


def test_evaluation_counts_cover_only_the_held_out_country(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )
    total_rows = len(synthetic_features_df)
    all_countries = set(countries_in_frame.values())
    for direction in report["directions"]:
        expected_test_rows = sum(
            1 for s1 in synthetic_features_df["s1_id"] if countries_in_frame[s1] == direction["test_country"]
        )
        expected_train_rows = sum(
            1 for s1 in synthetic_features_df["s1_id"] if countries_in_frame[s1] == direction["train_country"]
        )
        assert direction["n_test_rows"] == expected_test_rows
        assert direction["n_train_rows"] == expected_train_rows
        assert direction["n_train_rows"] + direction["n_test_rows"] <= total_rows, (
            "train and held-out slices cannot overlap"
        )
        # The synthetic pool carries a third country, which is in neither slice. LOCO
        # trains on exactly one country, so leftover countries are expected here and
        # absent in the real US/India data.
        untouched = all_countries - {direction["train_country"], direction["test_country"]}
        if untouched:
            assert direction["n_train_rows"] + direction["n_test_rows"] < total_rows, (
                f"{untouched} is present but was expected to be excluded from both slices"
            )


def test_thresholds_are_ordered_and_in_range(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )
    for direction in report["directions"]:
        assert 0.0 <= direction["t1"] <= 1.0
        assert 0.0 <= direction["t2"] <= 1.0
        assert direction["t2"] >= direction["t1"]


def test_untuned_score_is_independent_of_the_tuned_thresholds(
    synthetic_features_df,
    synthetic_labels_df,
    synthetic_ground_truth,
    countries_in_frame,
    recording_model,
    monkeypatch,
):
    """The untuned column must not inherit the tuned column's optimism.

    Forcing the fixed threshold above the maximum score makes the whole held-out slice
    abstain, so the untuned score becomes exactly the share of zero-match *entities* â€”
    computable straight from the labels. If the untuned path secretly reused the tuned
    thresholds, this would not move.
    """
    from src.validation import loco

    baseline = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )
    tuned_before = [d["macro_f05_tuned_optimistic"] for d in baseline["directions"]]
    untuned_before = [d["macro_f05_untuned_0.5"] for d in baseline["directions"]]

    monkeypatch.setattr(loco, "UNTUNED_THRESHOLD", 1.5)
    forced = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )

    evaluated_entities = {
        s1 for s1 in synthetic_features_df["s1_id"].astype(str)
    }
    for direction, tuned_score, untuned_score in zip(
        forced["directions"], tuned_before, untuned_before
    ):
        held_out = [
            s1
            for s1 in evaluated_entities
            if countries_in_frame[s1] == direction["test_country"]
        ]
        # The floor is entity-level, not row-level: a zero-match entity scores 1.0 when
        # abstained, and any entity with a true match scores 0.0.
        expected_floor = sum(
            1 for s1 in held_out if not synthetic_ground_truth[s1]
        ) / len(held_out)

        assert direction["macro_f05_untuned_0.5"] == pytest.approx(expected_floor), (
            f"predict-nothing should score the zero-match entity share for "
            f"{direction['test_country']}, got {direction['macro_f05_untuned_0.5']}, "
            f"expected {expected_floor}"
        )
        assert direction["n_predictions_untuned"] == 0, (
            "the forced untuned threshold is above every score, so it must abstain"
        )
        assert direction["n_singleton_violated"] == 0

        # And the tuned column is untouched by the patch, so the two are separate paths.
        assert direction["macro_f05_tuned_optimistic"] == pytest.approx(tuned_score)
        assert untuned_score >= 0.0

    assert UNTUNED_THRESHOLD == 0.5, "the module default must stay at 0.5"


def test_single_class_training_slice_is_rejected(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    """A country with only positives cannot be trained on; the error must be explicit."""
    frame = prepare_loco_frame(
        synthetic_features_df, synthetic_labels_df, countries_in_frame
    )
    us = frame[(frame["country"] == "US") & (frame["y"] == 1)]
    assert len(us) > 0, "fixture must have positive US rows to collapse"
    # Keep India intact so the held-out slice still exists and the failure is the
    # single-class training slice rather than a missing country.
    india = frame[frame["country"] == "India"]
    collapsed = pd.concat([us, india], ignore_index=True)

    with pytest.raises(ValueError, match="single class"):
        run_loco_direction(
            collapsed,
            train_country="US",
            test_country="India",
            model_cls=recording_model,
        )


# ------------------------------------------------------------------------ skipping
def test_absent_country_is_reported_not_raised(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        country_pairs=[("US", "Atlantis")],
        n_grid=11,
    )
    direction = report["directions"][0]
    assert direction["skipped"] is True
    assert "Atlantis" in direction["reason"]
    assert report["summary"]["n_directions_completed"] == 0
    assert "no LOCO direction had usable data" in report["summary"]["reason"]


def test_summary_aggregates_across_directions(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        n_grid=11,
    )
    summary = report["summary"]
    scores = [d["macro_f05_untuned_0.5"] for d in report["directions"]]
    assert summary["macro_f05_untuned_mean"] == pytest.approx(float(np.mean(scores)))
    assert summary["macro_f05_untuned_min"] == pytest.approx(min(scores))
    assert summary["macro_f05_untuned_max"] == pytest.approx(max(scores))
    worst = min(report["directions"], key=lambda d: d["macro_f05_untuned_0.5"])
    assert summary["worst_direction_untuned"] == worst["test_country"]


def test_france_direction_runs_when_data_supports_it(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    """No hardcoding: any country present in the frame can be a direction.

    Real data cannot do this, but the synthetic pool can, and proving the code slices by
    the data rather than by a constant is worth the test.
    """
    assert "France" in set(countries_in_frame.values())
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        country_pairs=[("US", "France")],
        n_grid=11,
    )
    assert report["directions"][0]["test_country"] == "France"
    assert not report["directions"][0].get("skipped")


# ----------------------------------------------------------------------- formatting
def test_report_formats_completed_and_skipped_directions(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        country_pairs=[("US", "India"), ("US", "Atlantis")],
        n_grid=11,
    )
    text = format_loco_report(report)

    assert "LOCO cross-country robustness" in text
    assert "India" in text and "Atlantis" in text
    assert "SKIPPED" in text
    assert "optimistic" in text, "the tuned column must carry its warning"
    assert "weakest direction" in text


def test_report_formats_when_nothing_completed(
    synthetic_features_df,
    synthetic_labels_df,
    countries_in_frame,
    recording_model,
):
    report = run_loco_evaluation(
        synthetic_features_df,
        synthetic_labels_df,
        countries_in_frame,
        model_cls=recording_model,
        country_pairs=[("US", "Atlantis")],
        n_grid=11,
    )
    text = format_loco_report(report)
    assert "no direction completed" in text


def test_loco_module_cannot_freeze_a_decision_config():
    """LOCO must stay a diagnostic and never gain the power to ship thresholds.

    Its thresholds are fitted on the held-out country's own scores, so freezing them
    would publish an optimistic number as if it were validated. Enforced structurally by
    checking the module never imports the config-writing helpers.
    """
    import inspect

    from src.validation import loco

    source = inspect.getsource(loco)
    for forbidden in ("save_decision_config", "load_decision_config", "write_scores_parquet"):
        assert forbidden not in source, (
            f"loco.py references {forbidden}; a diagnostic must not write frozen "
            "artifacts"
        )
    assert not hasattr(loco, "save_decision_config")
