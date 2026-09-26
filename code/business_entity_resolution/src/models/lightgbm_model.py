"""LightGBM classifier with monotonic constraints on similarity features.

Why monotonic constraints
-------------------------
The similarity features are, by construction, ordered: a candidate pair with a higher
name cosine cannot be a *worse* name match than one with a lower name cosine. A free
GBDT can and does violate this on sparse data, producing the familiar nonsense where
raising a similarity score lowers the predicted match probability, purely because a
correlated feature took a different tree path. Constraining the constrained columns to a
``+1`` (non-decreasing) relationship with the log-odds:

* encodes a domain fact rather than a statistical accident,
* regularizes the model on small or sparse data,
* and makes the model auditable â€” a reviewer can verify a prediction moves in a
  defensible direction when a similarity improves.

Only genuine similarity columns are constrained. Conflict and difference features such as
``addr_numeric_conflict``, ``name_len_diff`` or ``n_routes`` are deliberately left free,
because "more routes" or "bigger length difference" is not monotonically related to
matching, and forcing a direction on them would inject a false prior.

NaN handling
------------
LightGBM handles missing values natively and routes them down a default direction per
split, so no imputation is performed here. This matters for real entity-resolution data:
the master plan's Example 71 has a true match with a blank address, and a blank must be
learned from other evidence rather than silently imputed to a median similarity.

OpenMP runtime
--------------
LightGBM's manylinux wheel links against ``libgomp.so.1``. On hosts without a system GNU
OpenMP runtime the plain import raises ``OSError``. :func:`_import_lightgbm` first tries
the normal import so a system runtime is always preferred, and only on failure preloads
the committed fallback in ``.vendor/libgomp`` (see
``scripts/fetch_opensmp_runtime.sh``) and retries. No environment variable is required.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "code").exists():
            return parent
    return current.parents[4]

VENDORED_OPENMP_DIR = _find_repo_root() / ".vendor" / "libgomp"
OPENMP_LIBRARY_NAMES = ("libgomp.so.1", "libgomp.so")

# Similarity columns where a higher value always means a better match. These are the
# columns the master plan's Section 6 defines as similarity, plus the retrieval score.
MONOTONIC_SIMILARITY_COLUMNS = (
    "name_char_cos",
    "name_token_set",
    "name_token_sort",
    "name_contain_idf",
    "addr_char_cos",
    "addr_contain_idf",
    "addr_numeric_jaccard",
    "best_score",
)

# Never feed these to the model: identifiers, metadata and post-hoc score columns.
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
        "fold_id",
        "is_oof",
        "model_version",
        "p_raw",
        "p_cal",
        "y",
    }
)

DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def _preload_vendored_opensmp() -> Optional[str]:
    """Load a vendored GNU OpenMP runtime into the global symbol namespace.

    Returns the library path on success, ``None`` if no vendored copy exists or it
    cannot be loaded.
    """
    for name in OPENMP_LIBRARY_NAMES:
        candidate = VENDORED_OPENMP_DIR / name
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return str(candidate)
    return None


def _import_lightgbm():
    """Import lightgbm, falling back to the vendored OpenMP runtime.

    Raises a clear, actionable error if neither route works.
    """
    try:
        import lightgbm

        return lightgbm
    except OSError as first_error:
        # A failed import can leave a half-initialised module behind, which would make
        # the retry a no-op. Purge it first.
        for name in [n for n in sys.modules if n == "lightgbm" or n.startswith("lightgbm.")]:
            del sys.modules[name]

        preloaded = _preload_vendored_opensmp()
        if preloaded is None:
            raise ImportError(
                "lightgbm could not be imported and no vendored GNU OpenMP runtime was "
                f"found under {VENDORED_OPENMP_DIR}. Original error: {first_error}. "
                "Install a system OpenMP runtime (apt-get install libgomp1) or run "
                "bash scripts/fetch_opensmp_runtime.sh"
            ) from first_error

        import lightgbm

        return lightgbm


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the numeric feature columns of ``df``, in input order."""
    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError(f"no numeric feature columns found; got {list(df.columns)}")
    return feature_columns


def build_monotonic_constraints(
    feature_columns: Sequence[str],
    constrained_columns: Optional[Sequence[str]] = None,
) -> List[int]:
    """Build the LightGBM ``monotone_constraints`` vector for ``feature_columns``.

    Returns a ``+1`` entry for each constrained similarity column and ``0`` for every
    other column. Unknown constrained column names are ignored, so the helper stays
    usable if Person 2 renames or drops a feature.
    """
    if constrained_columns is None:
        constrained_columns = MONOTONIC_SIMILARITY_COLUMNS
    constrained = set(constrained_columns)
    return [1 if column in constrained else 0 for column in feature_columns]


def constrained_feature_names(
    feature_columns: Sequence[str],
    constrained_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return which of ``feature_columns`` are actually monotonic-constrained."""
    if constrained_columns is None:
        constrained_columns = MONOTONIC_SIMILARITY_COLUMNS
    constrained = set(constrained_columns)
    return [column for column in feature_columns if column in constrained]


class LightGBMClassifierWrapper:
    """Binary LightGBM classifier with optional early stopping and monotonic constraints.

    Parameters
    ----------
    **params
        Overrides merged over :data:`DEFAULT_PARAMS`.
    random_state
        Seed, also applied to ``params["random_state"]`` unless overridden.
    monotone_constraints
        Columns to constrain to ``+1``. Defaults to
        :data:`MONOTONIC_SIMILARITY_COLUMNS`. Pass ``[]`` to disable constraints
        entirely, which is only useful for control experiments.
    early_stopping_rounds
        When given, and a validation set is passed to :meth:`fit`, use it.
    """

    def __init__(
        self,
        monotone_constraints: Optional[Sequence[str]] = None,
        early_stopping_rounds: Optional[int] = 100,
        **params: Any,
    ) -> None:
        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        self.params.update(params)
        # `None` means "use the standard constrained similarity columns"; an empty
        # sequence is the explicit opt-out used by the control test.
        self.monotone_constraints_: List[str] = (
            list(MONOTONIC_SIMILARITY_COLUMNS)
            if monotone_constraints is None
            else list(monotone_constraints)
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.feature_columns_: Optional[List[str]] = None
        self.best_iteration_: Optional[int] = None
        self.fitted_ = False
        self.constraints_enabled_ = bool(self.monotone_constraints_)

    # ------------------------------------------------------------------ internals
    def _coerce_X(self, X: Any) -> pd.DataFrame:
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

    # ---------------------------------------------------------------------- api
    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[int],
        eval_set: Optional["tuple[pd.DataFrame, Sequence[int]]"] = None,
        eval_metric: Optional[str] = None,
    ) -> "LightGBMClassifierWrapper":
        """Fit the booster.

        Parameters
        ----------
        X, y
            Training features and binary labels.
        eval_set
            Optional ``(X_valid, y_valid)`` for early stopping. When omitted, or when
            ``early_stopping_rounds`` is ``None``, the model simply trains for
            ``n_estimators`` rounds.
        eval_metric
            Metric for early stopping, e.g. ``"average_precision"`` or
            ``"binary_logloss"``. Defaults to the configured ``metric``.
        """
        lgb = _import_lightgbm()

        if isinstance(X, pd.DataFrame):
            self.feature_columns_ = select_feature_columns(X)
        else:
            array = np.asarray(X, dtype=float)
            if array.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            self.feature_columns_ = [f"f{i}" for i in range(array.shape[1])]
            X = pd.DataFrame(array, columns=self.feature_columns_)

        y_array = np.asarray(y).ravel()

        params = dict(self.params)
        if self.constraints_enabled_:
            constraints = build_monotonic_constraints(
                self.feature_columns_, self.monotone_constraints_
            )
            params["monotone_constraints"] = constraints
            params["monotone_constraints_method"] = "advanced"
        else:
            params.pop("monotone_constraints", None)
            params.pop("monotone_constraints_method", None)

        self.model_ = lgb.LGBMClassifier(**params)

        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None and self.early_stopping_rounds:
            X_valid, y_valid = eval_set
            X_valid = self._coerce_X(X_valid)
            y_valid = np.asarray(y_valid).ravel()
            fit_kwargs["eval_metric"] = eval_metric or params.get("metric", "binary_logloss")
            # LightGBM 4.7 renamed the evaluation arguments; `eval_X`/`eval_y` are the
            # supported spellings and `eval_set` now emits a deprecation warning. A single
            # validation set is passed unwrapped, since `eval_X` rejects a list of frames.
            fit_kwargs["eval_X"] = X_valid
            fit_kwargs["eval_y"] = y_valid
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            ]

        self.model_.fit(X.loc[:, self.feature_columns_], y_array, **fit_kwargs)
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        self.fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities with shape ``(n_rows, 2)``, column order ``[0, 1]``."""
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("model is not fitted; call fit() first")
        proba = np.asarray(
            self.model_.predict_proba(self._coerce_X(X)), dtype=float
        )
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise RuntimeError(f"expected 2 probability columns, got {proba.shape}")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        """Return hard class predictions."""
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self.model_.predict(self._coerce_X(X))

    def get_monotonic_constraints(self) -> pd.DataFrame:
        """Report the constraint actually applied to each feature column."""
        if self.feature_columns_ is None:
            raise RuntimeError("model is not fitted; call fit() first")
        if not self.constraints_enabled_:
            raise RuntimeError("monotonic constraints are disabled for this model")
        constraints = build_monotonic_constraints(
            self.feature_columns_, self.monotone_constraints_
        )
        return pd.DataFrame(
            {
                "feature": self.feature_columns_,
                "monotone_constraint": np.array(constraints, dtype=np.int8),
            }
        )

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist the fitted wrapper (including the booster) with joblib."""
        import joblib

        if not self.fitted_:
            raise RuntimeError("model is not fitted; nothing to save")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMClassifierWrapper":
        """Load a model persisted by :meth:`save`.

        The OpenMP runtime is preloaded before the booster is unpickled, because
        unpickling a LightGBM booster reloads the native library.
        """
        import joblib

        _import_lightgbm()
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model
