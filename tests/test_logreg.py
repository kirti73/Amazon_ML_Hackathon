"""Unit tests for the logistic regression baseline.

All tests use the synthetic fixtures. No score here is competition-valid.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.labels import build_train_labels
from src.models.logreg import BaselineLogisticRegression, select_feature_columns


@pytest.fixture
def labelled_features(synthetic_features_df, synthetic_ground_truth):
    """Feature matrix joined to binary labels."""
    features = synthetic_features_df
    labels = build_train_labels(synthetic_features_df, synthetic_ground_truth)
    merged = features.merge(labels, on="pair_key", how="left")
    assert merged["y"].notna().all()
    return merged


def test_handles_missing_values_without_crashing(labelled_features):
    df = labelled_features
    assert df.isna().any().any(), "fixture must contain NaNs to exercise imputation"

    # Corrupt every numeric feature with heavy additional NaN coverage.
    X = df.drop(columns=["y"]).copy()
    numeric = list(X.select_dtypes(include=[np.number]).columns)
    X[numeric] = X[numeric].astype(float)
    mask = np.random.RandomState(0).uniform(size=(len(X), len(numeric))) < 0.5
    X[numeric] = X[numeric].mask(mask)
    X.loc[0, numeric] = np.nan
    X.loc[1, numeric] = np.nan

    model = BaselineLogisticRegression(random_state=42)
    model.fit(X, df["y"])
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all(), "non-finite probabilities from NaN input"
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_non_finite_input_is_rejected_loudly(labelled_features):
    """Infinity is a data bug, not a missing value, and must not be silently imputed.

    ``SimpleImputer`` raises on ``inf``/overflow inputs. That behaviour is intentional and
    is asserted here so it is not "fixed" later by quietly coercing infinities to NaN,
    which would hide an upstream feature-construction bug.
    """
    df = labelled_features
    X = df.drop(columns=["y"]).copy()
    numeric = list(X.select_dtypes(include=[np.number]).columns)
    X[numeric] = X[numeric].astype(float)
    X.loc[0, numeric] = np.inf

    model = BaselineLogisticRegression(random_state=42)
    with pytest.raises(ValueError):
        model.fit(X, df["y"])


def test_all_nan_feature_column_is_imputed(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"]).copy()
    X["all_nan"] = np.nan

    model = BaselineLogisticRegression(random_state=42)
    model.fit(X, df["y"])
    proba = model.predict_proba(X)

    assert np.isfinite(proba).all()


def test_probabilities_are_calibrated_in_unit_interval(labelled_features):
    df = labelled_features
    model = BaselineLogisticRegression(random_state=42)
    model.fit(df.drop(columns=["y"]), df["y"])
    proba = model.predict_proba(df.drop(columns=["y"]))

    assert proba.shape == (len(df), 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_deterministic_across_runs(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    y = df["y"]

    first = BaselineLogisticRegression(random_state=42).fit(X, y).predict_proba(X)
    second = BaselineLogisticRegression(random_state=42).fit(X, y).predict_proba(X)
    third = BaselineLogisticRegression(random_state=42).fit(X, y).predict_proba(X)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(second, third)


def test_different_random_state_does_not_break_contract(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    y = df["y"]
    model = BaselineLogisticRegression(random_state=7).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_identifier_columns_are_never_used_as_features():
    df = pd.DataFrame(
        {
            "pair_key": ["S1-1::S2-1", "S1-1::S2-2", "S1-2::S2-1"],
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_id": ["S2-1", "S2-2", "S2-1"],
            "candidate_source": ["S2", "S2", "S2"],
            "country": ["US", "US", "India"],
            "name_char_cos": [0.9, 0.1, 0.5],
            "n_routes": [1, 2, 1],
        }
    )
    features = select_feature_columns(df)
    assert features == ["name_char_cos", "n_routes"]
    for excluded in ("pair_key", "s1_id", "candidate_id", "candidate_source", "country"):
        assert excluded not in features


def test_feature_importance_is_sorted_and_labelled(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    model = BaselineLogisticRegression(random_state=42).fit(X, df["y"])
    importance = model.get_feature_importance()

    assert list(importance.columns) == ["feature", "coefficient", "abs_coefficient"]
    assert len(importance) == len(select_feature_columns(X))
    assert importance["abs_coefficient"].is_monotonic_decreasing
    assert set(importance["feature"]) == set(select_feature_columns(X))
    # Coefficients are only comparable because the scaler ran first.
    assert importance["coefficient"].abs().max() > 0.0


def test_unfitted_model_raises(labelled_features):
    model = BaselineLogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict_proba(labelled_features.drop(columns=["y"]))
    with pytest.raises(RuntimeError):
        model.predict(labelled_features.drop(columns=["y"]))
    with pytest.raises(RuntimeError):
        model.get_feature_importance()


def test_column_order_does_not_change_predictions(labelled_features):
    """A shuffled column order must not silently change the model semantics."""
    df = labelled_features
    X = df.drop(columns=["y"])
    y = df["y"]

    model = BaselineLogisticRegression(random_state=42).fit(X, y)
    baseline = model.predict_proba(X)

    shuffled = X[list(X.columns)[::-1]]
    reordered = model.predict_proba(shuffled)
    np.testing.assert_allclose(baseline, reordered, rtol=0, atol=1e-12)


def test_missing_column_at_predict_time_raises(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    model = BaselineLogisticRegression(random_state=42).fit(X, df["y"])
    with pytest.raises(ValueError):
        model.predict_proba(X.drop(columns=["name_char_cos"]))


def test_numpy_input_is_accepted(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    numeric = X.select_dtypes(include=[np.number])
    model = BaselineLogisticRegression(random_state=42).fit(numeric, df["y"])
    proba = model.predict_proba(numeric.to_numpy())
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_class_weight_balanced_learns_both_classes(labelled_features):
    """The fixture is imbalanced; a balanced model must not collapse to one class."""
    df = labelled_features
    X = df.drop(columns=["y"])
    model = BaselineLogisticRegression(random_state=42).fit(X, df["y"])
    predictions = model.predict(X)
    assert set(np.unique(predictions)) == {0, 1}


def test_model_separates_separable_signal(labelled_features):
    """Sanity check that the synthetic signal is learnable, not noise."""
    df = labelled_features
    X = df.drop(columns=["y"])
    model = BaselineLogisticRegression(random_state=42).fit(X, df["y"])
    p1 = model.predict_proba(X)[:, 1]
    y = df["y"].to_numpy()

    mean_pos = p1[y == 1].mean()
    mean_neg = p1[y == 0].mean()
    assert mean_pos > mean_neg, (
        f"model did not learn the synthetic signal (pos={mean_pos:.3f}, neg={mean_neg:.3f})"
    )
