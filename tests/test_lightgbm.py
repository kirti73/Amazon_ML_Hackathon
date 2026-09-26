"""Unit tests for the LightGBM wrapper and its monotonic constraints.

All tests use the synthetic fixtures. No score here is competition-valid.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.labels import build_train_labels
from src.models.lightgbm_model import (
    MONOTONIC_SIMILARITY_COLUMNS,
    LightGBMClassifierWrapper,
    _import_lightgbm,
    build_monotonic_constraints,
    constrained_feature_names,
    select_feature_columns,
)

lightgbm = _import_lightgbm()


@pytest.fixture
def labelled_features(synthetic_features_df, synthetic_ground_truth):
    features = synthetic_features_df
    labels = build_train_labels(synthetic_features_df, synthetic_ground_truth)
    return features.merge(labels, on="pair_key", how="left")


@pytest.fixture
def fitted_model(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    model = LightGBMClassifierWrapper(
        n_estimators=60,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        early_stopping_rounds=None,
    )
    return model.fit(X, df["y"])


# --------------------------------------------------------------------- constraints
def test_monotonic_constraints_built_for_similarity_columns():
    features = list(MONOTONIC_SIMILARITY_COLUMNS) + ["n_routes", "addr_numeric_conflict"]
    constraints = build_monotonic_constraints(features)
    assert constraints == [1] * len(MONOTONIC_SIMILARITY_COLUMNS) + [0, 0]


def test_unknown_constrained_columns_are_ignored():
    constraints = build_monotonic_constraints(["a", "b"], constrained_columns=["a", "ghost"])
    assert constraints == [1, 0]
    assert constrained_feature_names(["a", "b"], constrained_columns=["a", "ghost"]) == ["a"]


def test_similarity_columns_present_in_fixture_are_constrained(labelled_features):
    """The real fixture must actually exercise the constraints, not skip them."""
    X = labelled_features.drop(columns=["y"])
    features = select_feature_columns(X)
    constrained = set(constrained_feature_names(features))
    expected = set(MONOTONIC_SIMILARITY_COLUMNS) & set(features)
    assert expected, "fixture must contain the constrained similarity columns"
    assert constrained == expected


def test_model_reports_applied_constraints(fitted_model):
    report = fitted_model.get_monotonic_constraints()
    assert list(report.columns) == ["feature", "monotone_constraint"]
    assert (report["monotone_constraint"] == 1).any()
    assert report.loc[report["feature"].isin(MONOTONIC_SIMILARITY_COLUMNS), "monotone_constraint"].eq(1).all()


def test_constraints_can_be_disabled(labelled_features):
    df = labelled_features
    model = LightGBMClassifierWrapper(
        monotone_constraints=[], early_stopping_rounds=None, n_estimators=30
    )
    model.fit(df.drop(columns=["y"]), df["y"])
    assert model.params.get("monotone_constraints") is None
    with pytest.raises(RuntimeError):
        model.get_monotonic_constraints()


# --------------------------------------------------------------------- monotonicity
def _sweep_feature(model, X, feature, reference_indices, n_points=9):
    """Sweep ``feature`` across its observed range from several reference rows.

    Returns a ``(n_reference_rows, n_points)`` array of probability curves. The probe
    frame is built from the model's fitted feature columns as float, because LightGBM
    rejects object-dtype columns under pandas 3.x.

    The curves are returned stacked rather than concatenated so that callers can take
    ``np.diff(..., axis=1)`` per curve. Flattening first would make ``np.diff`` compare
    the last point of one reference row (at the grid maximum) against the first point of
    the next (at the grid minimum), inventing a large fake decrease at the seam.
    """
    features = list(model.feature_columns_)
    values = X[feature].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lo, hi = float(finite.min()), float(finite.max())
    grid = np.linspace(lo, hi, n_points)

    curves = []
    for i in reference_indices:
        base = X.iloc[i][features].astype(float).to_numpy()
        probe = pd.DataFrame(np.tile(base, (n_points, 1)), columns=features)
        probe[feature] = grid
        curves.append(model.predict_proba(probe)[:, 1])
    return np.vstack(curves)


def test_monotonicity_sweep_per_constrained_feature(fitted_model, labelled_features):
    """The core guarantee: raising a constrained feature must not lower P(match).

    Each constrained similarity is swept across its observed range one at a time while
    all other features are held at reference rows. Every resulting probability curve must
    be non-decreasing. A free GBDT reliably fails this on the same data.
    """
    X = labelled_features.drop(columns=["y"])
    constrained = constrained_feature_names(list(X.columns))
    assert constrained, "fixture must contain constrained columns"

    rng = np.random.RandomState(0)
    references = rng.choice(len(X), size=min(8, len(X)), replace=False)
    violations = []

    for feature in constrained:
        values = X[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 2 or finite.max() <= finite.min():
            continue
        deltas = np.diff(_sweep_feature(fitted_model, X, feature, references), axis=1).ravel()
        if np.any(deltas < -1e-9):
            violations.append((feature, float(deltas.min())))

    assert not violations, f"monotonicity violated (feature, worst drop): {violations}"


def test_unconstrained_model_violates_monotonicity_on_same_data(labelled_features, fitted_model):
    """Control test proving the constraint is doing real work.

    The same data and hyperparameters without constraints are allowed to violate
    monotonicity. If the unconstrained model also happened to be monotone, the
    constrained test above would be vacuous.
    """
    X = labelled_features.drop(columns=["y"])
    free = LightGBMClassifierWrapper(
        monotone_constraints=[],
        early_stopping_rounds=None,
        n_estimators=fitted_model.params["n_estimators"],
        learning_rate=fitted_model.params["learning_rate"],
        num_leaves=fitted_model.params["num_leaves"],
        min_child_samples=5,
    )
    free.fit(X, labelled_features["y"])

    feature = "name_char_cos"
    rng = np.random.RandomState(1)
    references = rng.choice(len(X), size=20, replace=False)

    free_drop = float(np.diff(_sweep_feature(free, X, feature, references), axis=1).min())
    constrained_drop = float(
        np.diff(_sweep_feature(fitted_model, X, feature, references), axis=1).min()
    )

    assert constrained_drop >= -1e-9, "constrained model must never decrease"
    assert free_drop < constrained_drop, (
        "unconstrained model was also monotone, so the monotonicity test is vacuous "
        f"(unconstrained worst drop {free_drop:.3e}, constrained worst drop {constrained_drop:.3e})"
    )


# --------------------------------------------------------------------- nan handling
def test_native_nan_support_for_missing_address_and_country(fitted_model, labelled_features):
    X = labelled_features.drop(columns=["y"])
    assert X.isna().any().any(), "fixture must contain NaNs"

    X_nan = X.copy()
    for column in ("addr_char_cos", "addr_contain_idf", "addr_numeric_jaccard", "country_eq"):
        if column in X_nan.columns:
            X_nan[column] = np.nan

    proba = fitted_model.predict_proba(X_nan)
    assert proba.shape == (len(X_nan), 2)
    assert np.isfinite(proba).all(), "NaN input produced non-finite probabilities"
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_all_nan_feature_column_is_handled(fitted_model, labelled_features):
    X = labelled_features.drop(columns=["y"]).copy()
    X["entirely_missing"] = np.nan
    proba = fitted_model.predict_proba(X)
    assert np.isfinite(proba).all()


# --------------------------------------------------------------------- early stopping
def test_early_stopping_records_best_iteration(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    y = df["y"]
    split = int(len(X) * 0.8)
    model = LightGBMClassifierWrapper(n_estimators=300, early_stopping_rounds=20)
    model.fit(X.iloc[:split], y.iloc[:split], eval_set=(X.iloc[split:], y.iloc[split:]))
    assert model.best_iteration_ is not None
    assert model.best_iteration_ > 0
    assert model.predict_proba(X.iloc[split:]).shape == (len(X) - split, 2)


def test_early_stopping_with_average_precision(labelled_features):
    df = labelled_features
    X = df.drop(columns=["y"])
    y = df["y"]
    split = int(len(X) * 0.8)
    model = LightGBMClassifierWrapper(n_estimators=300, early_stopping_rounds=20)
    model.fit(
        X.iloc[:split],
        y.iloc[:split],
        eval_set=(X.iloc[split:], y.iloc[split:]),
        eval_metric="average_precision",
    )
    assert np.isfinite(model.predict_proba(X.iloc[split:])).all()


# --------------------------------------------------------------------- serialization
def test_serialization_roundtrip(fitted_model, labelled_features, tmp_path):
    X = labelled_features.drop(columns=["y"])
    before = fitted_model.predict_proba(X)

    path = tmp_path / "lgbm.joblib"
    fitted_model.save(str(path))
    assert path.is_file()

    restored = LightGBMClassifierWrapper.load(str(path))
    after = restored.predict_proba(X)

    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
    assert restored.feature_columns_ == fitted_model.feature_columns_
    pd.testing.assert_frame_equal(
        restored.get_monotonic_constraints(), fitted_model.get_monotonic_constraints()
    )


# --------------------------------------------------------------------- contract
def test_probabilities_in_unit_interval_and_rows_sum_to_one(fitted_model, labelled_features):
    X = labelled_features.drop(columns=["y"])
    proba = fitted_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_unfitted_model_raises(labelled_features):
    model = LightGBMClassifierWrapper()
    with pytest.raises(RuntimeError):
        model.predict_proba(labelled_features.drop(columns=["y"]))
    with pytest.raises(RuntimeError):
        model.save("/tmp/should_not_exist.joblib")


def test_missing_column_at_predict_time_raises(fitted_model, labelled_features):
    X = labelled_features.drop(columns=["y"])
    with pytest.raises(ValueError):
        fitted_model.predict_proba(X.drop(columns=["name_char_cos"]))


def test_model_learns_synthetic_signal(fitted_model, labelled_features):
    X = labelled_features.drop(columns=["y"])
    p1 = fitted_model.predict_proba(X)[:, 1]
    y = labelled_features["y"].to_numpy()
    assert p1[y == 1].mean() > p1[y == 0].mean()


def test_lightgbm_version_is_available():
    assert hasattr(lightgbm, "__version__")
