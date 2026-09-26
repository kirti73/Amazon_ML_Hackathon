"""Leak-free 5-fold cross-validation runner producing out-of-fold scores.

The only reason this module exists is to make leakage *structurally impossible* rather
than merely unlikely:

* A fold is defined at the **S1 entity** level, and every candidate pair inherits its
  entity's fold. A model trained for fold ``k`` therefore cannot see any pair belonging to
  an entity in fold ``k``, because those rows are physically absent from its training
  frame. No feature engineering, no join and no downstream step can reintroduce them.
* Rows are partitioned by an explicit inner join on the frozen ``fold_id``, and any pair
  whose fold is missing or out of range raises rather than being silently dropped. A
  silently dropped row would quietly remove training data and inflate CV.

Why OOF scores matter here
--------------------------
Every threshold in Phase 6 is tuned on OOF predictions. Using in-fold predictions would
let the decision layer see scores from rows the model memorised, which produces a
threshold that looks good on paper and collapses on the test set. The ``is_oof`` column
exists so a downstream consumer can *refuse* to tune on anything else.

Calibration
-----------
``p_raw`` is the model probability and ``p_cal`` is the calibrated probability. No
calibrator is fitted yet, so ``p_cal`` is currently equal to ``p_raw`` rather than
silently pretending to be calibrated; fitting an isotonic/sigmoid calibrator on OOF
predictions is the natural follow-up, and it must be done inside this module so it is
fitted on OOF data only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

SCORES_COLUMNS = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "p_raw",
    "p_cal",
    "fold_id",
    "is_oof",
    "model_version",
]

# The scores.parquet contract is enforced at the write boundary with an explicit Arrow
# schema. This matters because pandas 3.x defaults text columns to `StringDtype`, which
# pyarrow maps to `large_string` (64-bit offsets). Left alone, the artifact would silently
# drift away from the `string` type declared in docs/schemas.md, and Person 4's loader and
# validator would be written against a schema nobody actually produces.
SCORES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("pair_key", pa.string()),
        pa.field("s1_id", pa.string()),
        pa.field("candidate_id", pa.string()),
        pa.field("p_raw", pa.float32()),
        pa.field("p_cal", pa.float32()),
        pa.field("fold_id", pa.int8()),
        pa.field("is_oof", pa.bool_()),
        pa.field("model_version", pa.string()),
    ]
)

REQUIRED_FEATURES_COLUMNS = ("pair_key", "s1_id", "candidate_id")
REQUIRED_LABELS_COLUMNS = ("pair_key", "y")
REQUIRED_FOLDS_COLUMNS = ("s1_id", "fold_id")

DEFAULT_N_FOLDS = 5


def prepare_training_frame(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    folds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join features, labels and folds into one training frame, validating as it goes.

    The returned frame has exactly one row per candidate pair, carrying ``y`` and
    ``fold_id``. Validation failures raise, because a fold coverage gap silently shrinks
    the training set and inflates every score computed from it.
    """
    for name, frame, required in (
        ("features_df", features_df, REQUIRED_FEATURES_COLUMNS),
        ("labels_df", labels_df, REQUIRED_LABELS_COLUMNS),
        ("folds_df", folds_df, REQUIRED_FOLDS_COLUMNS),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if features_df["pair_key"].duplicated().any():
        raise ValueError("features_df contains duplicate pair_key values")

    frame = features_df.merge(
        labels_df[["pair_key", "y"]], on="pair_key", how="inner", validate="one_to_one"
    )
    if len(frame) != len(features_df):
        raise ValueError(
            f"label join lost rows: {len(features_df)} features -> {len(frame)} joined. "
            "Every candidate pair must have exactly one label."
        )

    frame = frame.merge(
        folds_df[["s1_id", "fold_id"]], on="s1_id", how="left", validate="many_to_one"
    )
    if frame["fold_id"].isna().any():
        missing_entities = frame.loc[frame["fold_id"].isna(), "s1_id"].nunique()
        raise ValueError(
            f"{missing_entities} S1 entities in the candidate set have no fold "
            "assignment; regenerate artifacts/folds.parquet over the full S1 pool"
        )

    frame["fold_id"] = frame["fold_id"].astype("int8")
    return frame


def train_cv(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    model_cls: Any,
    model_params: Optional[Dict[str, Any]] = None,
    n_folds: int = DEFAULT_N_FOLDS,
    model_version: str = "model_v1",
    expected_folds: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """Run leak-free k-fold cross-validation and return OOF scores.

    Parameters
    ----------
    features_df
        Candidate features, one row per pair, containing at least
        ``pair_key``, ``s1_id``, ``candidate_id``.
    labels_df
        Binary labels with ``pair_key`` and ``y``.
    folds_df
        Entity-level fold assignment with ``s1_id`` and ``fold_id``. Every S1 entity in
        ``features_df`` must be present.
    model_cls
        Model class exposing ``fit(X, y)`` and ``predict_proba(X)``; both
        ``BaselineLogisticRegression`` and ``LightGBMClassifierWrapper`` qualify.
    model_params
        Keyword arguments passed to ``model_cls``. ``random_state`` is injected when the
        model accepts it, so a model that supports seeding is always seeded.
    n_folds
        Number of folds to iterate over.
    model_version
        Identifier written into the ``model_version`` output column.
    expected_folds
        Fold ids to iterate over. Defaults to ``range(n_folds)``. Pass this to run CV over
        a specific subset such as the LOCO country slices.

    Returns
    -------
    tuple
        ``(scores_df, models)`` where ``scores_df`` follows the ``scores.parquet``
        contract and ``models[k]`` is the model fitted without fold ``k``.
    """
    model_params = dict(model_params or {})
    frame = prepare_training_frame(features_df, labels_df, folds_df)

    observed = sorted(int(f) for f in frame["fold_id"].unique())
    if expected_folds is None:
        expected_folds = list(range(n_folds))
    expected_folds = [int(f) for f in expected_folds]

    missing = [f for f in expected_folds if f not in observed]
    if missing:
        raise ValueError(
            f"fold(s) {missing} are absent from folds_df, which contains folds {observed}"
        )

    folds_column = frame["fold_id"].to_numpy()
    fold_array = folds_column.astype("int8")

    score_rows: List[pd.DataFrame] = []
    models: List[Any] = []

    for fold_id in expected_folds:
        train_mask = fold_array != fold_id
        valid_mask = fold_array == fold_id

        if not train_mask.any():
            raise ValueError(f"fold {fold_id} has no training rows outside itself")
        if not valid_mask.any():
            raise ValueError(f"fold {fold_id} has no rows to predict")

        X_train = frame.loc[train_mask]
        y_train = X_train["y"].to_numpy()
        X_valid = frame.loc[valid_mask]

        params = dict(model_params)
        if "random_state" not in params and "random_state" in _init_params(model_cls):
            params["random_state"] = 42

        model = model_cls(**params)
        model.fit(X_train.drop(columns=["y", "fold_id"]), y_train)
        models.append(model)

        p_raw = np.asarray(model.predict_proba(X_valid.drop(columns=["y", "fold_id"])), dtype=float)
        if p_raw.shape[0] != len(X_valid):
            raise ValueError(
                f"fold {fold_id}: model returned {p_raw.shape[0]} predictions "
                f"for {len(X_valid)} rows"
            )

        block = pd.DataFrame(
            {
                "pair_key": X_valid["pair_key"].to_numpy(),
                "s1_id": X_valid["s1_id"].to_numpy(),
                "candidate_id": X_valid["candidate_id"].to_numpy(),
                "p_raw": p_raw[:, 1].astype("float32"),
                # No calibrator is fitted yet; p_cal is p_raw rather than a pretence.
                "p_cal": p_raw[:, 1].astype("float32"),
                "fold_id": np.full(len(X_valid), fold_id, dtype="int8"),
                "is_oof": np.ones(len(X_valid), dtype=bool),
                "model_version": model_version,
            }
        )
        score_rows.append(block)

    scores = pd.concat(score_rows, ignore_index=True)
    scores = scores[SCORES_COLUMNS]

    if scores["pair_key"].duplicated().any():
        raise ValueError("OOF scores contain duplicate pair_key values")
    if len(scores) != len(frame):
        raise ValueError(
            f"OOF score rows ({len(scores)}) do not match candidate rows ({len(frame)})"
        )
    return scores, models


def write_scores_parquet(scores_df: pd.DataFrame, output_path: str) -> None:
    """Write ``scores.parquet`` with the exact contract schema.

    The schema is passed explicitly rather than inferred, so the string columns are
    written as Arrow ``string`` and never as ``large_string`` (see
    :data:`SCORES_ARROW_SCHEMA`).
    """
    import pyarrow.parquet as pq

    missing = [column for column in SCORES_COLUMNS if column not in scores_df.columns]
    if missing:
        raise ValueError(f"scores frame is missing contract columns: {missing}")

    table = pa.Table.from_pandas(
        scores_df.loc[:, SCORES_COLUMNS], schema=SCORES_ARROW_SCHEMA, preserve_index=False
    )
    pq.write_table(table, output_path)


def _init_params(model_cls: Any) -> Sequence[str]:
    """Return the constructor parameter names of ``model_cls``, best effort."""
    import inspect

    try:
        return list(inspect.signature(model_cls).parameters)
    except (TypeError, ValueError):
        return []


def coverage_report(scores_df: pd.DataFrame, frame: pd.DataFrame) -> Dict[str, Any]:
    """Summarise OOF coverage for logging: rows, folds and per-fold entity counts."""
    per_fold = (
        scores_df.groupby("fold_id")
        .agg(
            rows=("pair_key", "size"),
            s1_entities=("s1_id", "nunique"),
            mean_p=("p_raw", "mean"),
        )
        .reset_index()
    )
    return {
        "n_scores": int(len(scores_df)),
        "n_candidates": int(len(frame)),
        "n_s1_entities": int(frame["s1_id"].nunique()),
        "n_folds": int(scores_df["fold_id"].nunique()),
        "is_oof_all_true": bool(scores_df["is_oof"].all()),
        "duplicate_pair_keys": int(scores_df["pair_key"].duplicated().sum()),
        "per_fold": per_fold.to_dict(orient="records"),
    }
