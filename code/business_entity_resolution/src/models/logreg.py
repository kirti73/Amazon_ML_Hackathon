"""Fast, leak-free logistic regression baseline classifier.

Purpose
-------
This is the reference model, not the final model. It exists to answer three questions
quickly and honestly:

1. Is the feature set informative at all?
2. What does macro-F0.5 look like after threshold tuning, before any GBDT complexity?
3. Which features carry signal, cheaply, via coefficients on a comparable scale?

Design decisions
----------------
* ``SimpleImputer(strategy="median")`` then ``StandardScaler``. The split is in that order
  because a scaler must not see imputed values it will never see at inference time, and a
  median is only meaningful on the raw scale. The pipeline guarantees that
  ``fit_transform`` on training rows and ``transform`` on any other rows use the *same*
  fitted imputations and scaling, which is what keeps CV folds leak-free.
* ``class_weight="balanced"``. The candidate set is heavily imbalanced: the real training
  data has 123,247 zero-match S1 entities out of 2,206,821 and the vast majority of
  retrieved pairs are hard negatives. Without reweighting the baseline collapses to
  predicting "not a match" everywhere, which scores ~0.0558 macro-F0.5.
* ``max_iter=1000`` with a fixed ``random_state`` so repeated runs are bit-identical.

Documented edge-case behaviour (verified by tests, not assumed):

* ``NaN`` is imputed with the training median, per fold, so imputation cannot leak.
* A column that is *entirely* ``NaN`` in the training fold cannot have a median.
  ``SimpleImputer`` fills it with ``0.0`` and emits a ``UserWarning``. This is the
  blank-address / missing-country case from the master plan, so the warning is left
  visible rather than suppressed â€” it is a real signal about the data, not noise.
* ``inf`` and overflow values are **rejected** with a ``ValueError``. Infinity is a
  feature-construction bug, not a missing value, and silently coercing it to ``NaN``
  would hide the bug. This is asserted by a test so it is not "fixed" later.

The model only ever sees numeric feature columns. Identifier columns and raw text are
excluded by :func:`select_feature_columns` so that an id can never be memorized as a
shortcut.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Columns that identify a row rather than describe a match. They must never be fed to the
# model, otherwise a unique id becomes a perfect (and completely useless) predictor.
NON_FEATURE_COLUMNS = frozenset(
    {
        "pair_key",
        "s1_id",
        "candidate_id",
        "candidate_source",
        "s2_id",
        "s3_id",
        "country",
        "business_name",
        "business_address",
        "s1_name",
        "s1_address",
        "cand_name",
        "cand_address",
        "fold_id",
        "is_oof",
        "model_version",
        "p_raw",
        "p_cal",
    }
)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the numeric feature columns of ``df``, in input order.

    Identifier/metadata columns are dropped. Non-numeric leftovers are dropped rather than
    one-hot encoded, because the Person 2 contract already encodes categorical information
    as explicit flags such as ``country_eq``.
    """
    feature_columns: List[str] = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError(
            "no numeric feature columns found; available columns: "
            f"{list(df.columns)}"
        )
    return feature_columns


class BaselineLogisticRegression:
    """Impute -> scale -> logistic regression, with a clean sklearn interface.

    Parameters
    ----------
    max_iter
        Maximum solver iterations.
    random_state
        Seed for the solver, making runs reproducible.
    class_weight
        Passed through to ``LogisticRegression``; defaults to ``"balanced"``.
    C:
        Inverse regularization strength.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str = "balanced",
        C: float = 1.0,
    ) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = C
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    # ------------------------------------------------------------------ internals
    def _coerce_X(self, X: Any) -> pd.DataFrame:
        """Align ``X`` to the fitted feature columns.

        A fitted column order is enforced rather than assumed, so a caller passing columns
        in a different order gets the same model semantics instead of silently scrambled
        coefficients.
        """
        if self.feature_columns_ is None:
            raise RuntimeError("model is not fitted; call fit() first")

        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns_ if c not in X.columns]
            if missing:
                raise ValueError(f"missing feature columns at predict time: {missing}")
            return X.loc[:, self.feature_columns_]

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(self.feature_columns_):
            raise ValueError(
                f"expected {len(self.feature_columns_)} features, got {array.shape[1]}"
            )
        return pd.DataFrame(array, columns=self.feature_columns_)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        C=self.C,
                    ),
                ),
            ]
        )

    # ---------------------------------------------------------------------- api
    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "BaselineLogisticRegression":
        """Fit the imputer, scaler and classifier on the given rows."""
        if isinstance(X, pd.DataFrame):
            self.feature_columns_ = select_feature_columns(X)
        else:
            array = np.asarray(X, dtype=float)
            if array.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            self.feature_columns_ = [f"f{i}" for i in range(array.shape[1])]
            X = pd.DataFrame(array, columns=self.feature_columns_)

        y_series = pd.Series(np.asarray(y).ravel())
        if y_series.isna().any():
            raise ValueError("y contains missing values")
        self.classes_ = np.sort(y_series.unique())

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X.loc[:, self.feature_columns_], y_series)
        self.n_features_in_ = len(self.feature_columns_)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities with shape ``(n_rows, 2)``, column order ``[0, 1]``."""
        if self.pipeline is None:
            raise RuntimeError("model is not fitted; call fit() first")
        X_aligned = self._coerce_X(X)
        proba = self.pipeline.predict_proba(X_aligned)
        proba = np.asarray(proba, dtype=float)
        if proba.shape[1] != 2:
            raise RuntimeError(
                f"expected 2 probability columns, got {proba.shape[1]}"
            )
        return proba

    def predict(self, X: Any) -> np.ndarray:
        """Return hard class predictions."""
        if self.pipeline is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self.pipeline.predict(self._coerce_X(X))

    def get_feature_importance(self) -> pd.DataFrame:
        """Return standardized coefficients as a feature-importance table.

        Coefficients are comparable to each other because the scaler standardized every
        feature first. The magnitude is the feature's effect on log-odds; the sign is its
        direction. Sorted by absolute magnitude, strongest first.
        """
        if self.pipeline is None or self.feature_columns_ is None:
            raise RuntimeError("model is not fitted; call fit() first")

        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        coefficients = np.asarray(classifier.coef_, dtype=float).ravel()

        table = pd.DataFrame(
            {
                "feature": self.feature_columns_,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False, ignore_index=True)
        return table

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist the fitted pipeline with joblib."""
        import joblib

        if self.pipeline is None:
            raise RuntimeError("model is not fitted; nothing to save")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaselineLogisticRegression":
        """Load a model persisted by :meth:`save`."""
        import joblib

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model
