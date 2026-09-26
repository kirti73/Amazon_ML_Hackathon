"""Leave-one-country-out (LOCO) robustness diagnostic.

Purpose
-------
The competition's test set contains a country the training data does not: `train_source1`
covers only `US` and `India`, while the test set adds `France`. A model tuned only on
in-country folds can look excellent under 5-fold CV and still transfer poorly, because
CV folds drawn from the training pool never ask "does this work when the country itself
changes?".

LOCO answers that question directly: train on one country, evaluate on the other, in both
directions. It is a **diagnostic, not a model selection tool** â€” the thresholds reported
here are tuned on the held-out country's own scores, which makes the resulting macro-F0.5
optimistic by construction and unfit for comparison against the honest OOF number. The
function therefore reports both the optimistic tuned score and the pessimistic
untuned-at-0.5 score, and never writes a `decision_config.json`.

What a healthy result looks like
--------------------------------
A small gap between the LOCO score and the in-country OOF score means the model learned
name/address similarity rather than country-specific priors. A large gap, particularly
combined with a high false-positive rate on held-out singletons, means the model has
learnt `US`-shaped data and should not be trusted on `France` without extra evidence.

Constraints
-----------
* `France` cannot appear in a training-side diagnostic because it is absent from the
  training data. Only `US` and `India` directions are possible.
* A direction with no rows on one side is reported as skipped, with the reason, rather
  than raising. A missing country slice is a data fact to report, not a crash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.decision.threshold import (
    apply_decision_rules,
    evaluate_thresholds,
    tune_two_thresholds,
)

DEFAULT_COUNTRY_PAIRS = (("US", "India"), ("India", "US"))
UNTUNED_THRESHOLD = 0.5


class _MissingCountrySlice(ValueError):
    """Raised internally when a requested country direction has no usable rows."""


def prepare_loco_frame(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    countries_by_s1: Mapping[str, str],
) -> pd.DataFrame:
    """Join features and labels and attach the training country of each S1 entity."""
    from src.models.train_cv import prepare_training_frame

    if not isinstance(countries_by_s1, Mapping):
        raise TypeError(
            f"countries_by_s1 must be a Mapping, got {type(countries_by_s1).__name__}"
        )

    # Country coverage is checked *before* the frame is assembled, because the assembled
    # frame's own fold-coverage check would otherwise fire first and blame the fold
    # artifact for what is really a missing country.
    country_by_s1 = {str(s1): str(c) for s1, c in countries_by_s1.items()}
    missing = sorted(set(features_df["s1_id"].astype(str)) - set(country_by_s1))
    if missing:
        raise ValueError(
            f"{len(missing)} S1 entities in the candidate set have no country in "
            f"countries_by_s1, e.g. {missing[:5]}; LOCO cannot slice them"
        )

    # LOCO slices by country rather than by fold, so folds are supplied as a single
    # constant column; prepare_training_frame is reused purely for its validation.
    countries = pd.DataFrame(
        {
            "s1_id": [str(s1) for s1 in country_by_s1],
            "fold_id": np.zeros(len(country_by_s1), dtype="int8"),
        }
    )
    frame = prepare_training_frame(features_df, labels_df, countries)

    frame = frame.merge(
        pd.DataFrame(
            {"s1_id": list(country_by_s1), "country": list(country_by_s1.values())}
        ),
        on="s1_id",
        how="left",
        validate="many_to_one",
    )
    if frame["country"].isna().any():
        unreachable = frame.loc[frame["country"].isna(), "s1_id"].nunique()
        raise ValueError(
            f"{unreachable} S1 entities lost their country during the join; this is a bug"
        )
    return frame


def _slice(
    frame: pd.DataFrame,
    countries: Sequence[str],
    keep: str,
) -> pd.DataFrame:
    if keep not in set(countries):
        raise _MissingCountrySlice(
            f"country {keep!r} is not present in the training data; available: "
            f"{sorted(set(countries))}"
        )
    return frame[frame["country"] == keep]


def run_loco_direction(
    frame: pd.DataFrame,
    train_country: str,
    test_country: str,
    model_cls: Any,
    model_params: Optional[Dict[str, Any]] = None,
    n_grid: int = 21,
    score_column: str = "p_cal",
) -> Dict[str, Any]:
    """Train on ``train_country`` and evaluate on ``test_country``."""
    model_params = dict(model_params or {})

    train = _slice(frame, frame["country"].unique(), train_country)
    test = _slice(frame, frame["country"].unique(), test_country)

    if train.empty:
        raise _MissingCountrySlice(f"no training rows for country {train_country!r}")
    if test.empty:
        raise _MissingCountrySlice(f"no evaluation rows for country {test_country!r}")

    if "y" not in test.columns:
        raise ValueError("evaluation slice is missing labels")

    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"training slice {train_country!r} contains a single class; LOCO needs both"
        )

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"y", "fold_id", "country"}
    ]

    params = dict(model_params)
    if "random_state" not in params and "random_state" in _init_params(model_cls):
        params["random_state"] = 42

    model = model_cls(**params)
    model.fit(train[feature_columns], y_train)
    p_test = np.asarray(model.predict_proba(test[feature_columns]), dtype=float)[:, 1]

    test_scores = pd.DataFrame(
        {
            "pair_key": test["pair_key"].to_numpy(),
            "s1_id": test["s1_id"].to_numpy(),
            "candidate_id": test["candidate_id"].to_numpy(),
            score_column: p_test.astype("float32"),
            "is_oof": True,
        }
    )

    truth: Dict[str, set] = {}
    for s1_id, block in test.groupby("s1_id", sort=True):
        truth[str(s1_id)] = {
            str(c) for c, y in zip(block["candidate_id"], block["y"]) if y == 1
        }

    # Optimistic: thresholds tuned on the held-out country's own scores. Reported for
    # diagnosis only, never frozen.
    t1, t2 = tune_two_thresholds(test_scores, truth, n_grid=n_grid, score_column=score_column)
    tuned = evaluate_thresholds(
        test_scores, truth, t1, t2, score_column=score_column
    )

    # Pessimistic and comparable: a fixed, untuned threshold.
    untuned = evaluate_thresholds(
        test_scores, truth, UNTUNED_THRESHOLD, UNTUNED_THRESHOLD, score_column=score_column
    )

    predictions = apply_decision_rules(
        test_scores, t1=t1, t2=t2, score_column=score_column
    )
    untuned_predictions = apply_decision_rules(
        test_scores,
        t1=UNTUNED_THRESHOLD,
        t2=UNTUNED_THRESHOLD,
        score_column=score_column,
    )
    singleton_truth = [s1 for s1, matches in truth.items() if not matches]

    return {
        "train_country": train_country,
        "test_country": test_country,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_entities": int(train["s1_id"].nunique()),
        "n_test_entities": int(test["s1_id"].nunique()),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "t1": t1,
        "t2": t2,
        "macro_f05_tuned_optimistic": tuned,
        "macro_f05_untuned_0.5": untuned,
        # Named per-column on purpose: the tuned and untuned decisions emit different
        # match sets, and an unlabelled count beside a two-column report invites the
        # reader to attribute it to the wrong one.
        "n_predictions_tuned": int(sum(len(m) for m in predictions.values())),
        "n_predictions_untuned": int(sum(len(m) for m in untuned_predictions.values())),
        "n_singleton_entities": len(singleton_truth),
        "n_singleton_violated": int(
            sum(1 for s1 in singleton_truth if predictions.get(s1))
        ),
    }


def run_loco_evaluation(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    countries_by_s1: Mapping[str, str],
    model_cls: Any,
    model_params: Optional[Dict[str, Any]] = None,
    country_pairs: Optional[Sequence[Sequence[str]]] = None,
    n_grid: int = 21,
    score_column: str = "p_cal",
) -> Dict[str, Any]:
    """Run both LOCO directions and return a structured robustness report.

    Parameters
    ----------
    features_df, labels_df
        Candidate features and binary labels, as consumed by the CV runner.
    countries_by_s1
        Mapping from S1 entity id to its training country.
    model_cls, model_params
        Model to evaluate, as in :func:`src.models.train_cv.train_cv`.
    country_pairs
        ``(train, test)`` direction pairs. Defaults to US -> India and India -> US.
    n_grid
        Threshold grid size for the diagnostic tuning.

    Returns
    -------
    dict
        ``{"directions": [...], "summary": {...}}``. Each direction is either a result
        dict or a ``{"skipped": True, "reason": ...}`` entry.
    """
    pairs = [tuple(pair) for pair in (country_pairs or DEFAULT_COUNTRY_PAIRS)]
    frame = prepare_loco_frame(features_df, labels_df, countries_by_s1)

    results: List[Dict[str, Any]] = []
    for train_country, test_country in pairs:
        try:
            results.append(
                run_loco_direction(
                    frame,
                    train_country=train_country,
                    test_country=test_country,
                    model_cls=model_cls,
                    model_params=model_params,
                    n_grid=n_grid,
                    score_column=score_column,
                )
            )
        except _MissingCountrySlice as error:
            results.append(
                {
                    "train_country": train_country,
                    "test_country": test_country,
                    "skipped": True,
                    "reason": str(error),
                }
            )

    completed = [r for r in results if not r.get("skipped")]
    if completed:
        untuned_scores = [r["macro_f05_untuned_0.5"] for r in completed]
        summary = {
            "n_directions_completed": len(completed),
            "n_directions_skipped": len(results) - len(completed),
            "macro_f05_untuned_mean": float(np.mean(untuned_scores)),
            "macro_f05_untuned_min": float(np.min(untuned_scores)),
            "macro_f05_untuned_max": float(np.max(untuned_scores)),
            "worst_direction_untuned": min(
                completed, key=lambda r: r["macro_f05_untuned_0.5"]
            )["test_country"],
        }
    else:
        summary = {
            "n_directions_completed": 0,
            "n_directions_skipped": len(results),
            "reason": "no LOCO direction had usable data",
        }

    return {"directions": results, "summary": summary}


def _init_params(model_cls: Any) -> Sequence[str]:
    import inspect

    try:
        return list(inspect.signature(model_cls).parameters)
    except (TypeError, ValueError):
        return []


def format_loco_report(report: Dict[str, Any]) -> str:
    """Render a LOCO report as a readable block for the log and console."""
    lines = ["", "LOCO cross-country robustness", "=" * 78]
    header = (
        f"{'train':>8} {'test':>8} {'train rows':>11} {'test rows':>10} "
        f"{'t1':>7} {'t2':>7} {'untuned F0.5':>13} {'tuned*':>9} {'singleton FP':>13}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for result in report["directions"]:
        if result.get("skipped"):
            lines.append(
                f"{result['train_country']:>8} {result['test_country']:>8} "
                f"{'SKIPPED':>11} - reason: {result['reason']}"
            )
            continue
        lines.append(
            f"{result['train_country']:>8} {result['test_country']:>8} "
            f"{result['n_train_rows']:>11,} {result['n_test_rows']:>10,} "
            f"{result['t1']:>7.4f} {result['t2']:>7.4f} "
            f"{result['macro_f05_untuned_0.5']:>13.4f} "
            f"{result['macro_f05_tuned_optimistic']:>9.4f} "
            f"{result['n_singleton_violated']:>13,}"
        )

    lines.append("-" * 78)
    lines.append("* tuned column is optimistic: thresholds fitted on the held-out country.")
    summary = report["summary"]
    if summary.get("n_directions_completed"):
        lines.append(
            f"untuned macro-F0.5 across directions: mean {summary['macro_f05_untuned_mean']:.4f}, "
            f"min {summary['macro_f05_untuned_min']:.4f}, max {summary['macro_f05_untuned_max']:.4f}"
        )
        lines.append(f"weakest direction: train -> {summary['worst_direction_untuned']}")
    else:
        lines.append(f"no direction completed: {summary.get('reason')}")
    lines.append("")
    return "\n".join(lines)
