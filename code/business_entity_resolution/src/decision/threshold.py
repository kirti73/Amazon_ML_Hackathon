"""Threshold tuning and prediction rules, evaluated strictly on OOF scores.

The asymmetry this layer exists to exploit
------------------------------------------
F0.5 weights precision four times as heavily as recall (beta^2 = 0.25, so
F_beta = 1.25 * P * R / (0.25 * P + R)). Because the competition metric is macro-averaged
over **every** S1 entity, including the 123,247 zero-match entities in the training pool,
a single false match costs a full 1.0 of headroom on that entity. On a non-singleton with
a single true match, adding one wrong candidate drops that entity's score from 1.0 to
0.67, while dropping the one true match drops it to 0.0. Both errors are expensive, but
they are not symmetric, and the direction depends on the local ground-truth size.

That is why the layer is built around abstention:

* The predict-nothing baseline already scores **0.0558** macro-F0.5, because every
  zero-match entity scores 1.0. Emitting nothing is never a zero score.
* On a non-singleton, emitting nothing scores 0.0. So the model has to clear a real bar
  before it is allowed to speak at all.

Two thresholds
--------------
``T1`` is the entry threshold: a candidate's score must reach it to be selected. ``T2``
is the continuation threshold, applied to each subsequent candidate. Setting ``T2 >= T1``
lets the layer accept one strong candidate and then demand more evidence for the rest,
which protects an entity that has exactly one true match from absorbing a borderline
second candidate. ``T1 = T2`` reduces to plain global thresholding, so a single-threshold
baseline remains available for comparison.

Every threshold here is tuned on OOF predictions only. Tuning on in-fold scores would let
the search see scores from rows the model memorised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.validation.metrics import macro_f05

DEFAULT_THRESHOLD_GRID_SIZE = 201
DECISION_CONFIG_FILENAME = "decision_config.json"

REQUIRED_SCORE_COLUMNS = ("pair_key", "s1_id", "candidate_id", "p_cal")


def _validate_oof(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Reject anything that is not a clean out-of-fold score frame."""
    if not isinstance(scores_df, pd.DataFrame):
        raise TypeError(
            f"scores_df must be a pandas DataFrame, got {type(scores_df).__name__}"
        )
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in scores_df.columns]
    if missing:
        raise ValueError(f"scores_df is missing required columns: {missing}")
    if scores_df[list(REQUIRED_SCORE_COLUMNS)].isna().any().any():
        raise ValueError("scores_df contains missing values in the required columns")
    if "is_oof" in scores_df.columns and not scores_df["is_oof"].all():
        raise ValueError(
            "thresholds must be tuned on out-of-fold scores only; "
            f"{int((~scores_df['is_oof']).sum())} rows have is_oof=False"
        )
    return scores_df


def _normalize_ground_truth(
    ground_truth_by_s1: Mapping[str, Iterable[str]]
) -> Dict[str, Set[str]]:
    """Coerce every truth set to a set of strings."""
    return {
        str(s1_id): {str(candidate_id) for candidate_id in matches}
        for s1_id, matches in ground_truth_by_s1.items()
    }


def build_threshold_grid(
    scores: Sequence[float],
    n_grid: int = DEFAULT_THRESHOLD_GRID_SIZE,
) -> np.ndarray:
    """Build a candidate threshold grid covering the observed score range.

    Includes a threshold above the maximum score, which represents predicting nothing for
    every entity. That option is a real candidate here â€” it is the 0.0558 floor â€” and a
    grid that excluded it could never choose to stay silent.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.5])
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([0.5])
    if hi <= lo:
        return np.array([lo])
    grid = np.linspace(lo, hi, int(n_grid))
    return np.concatenate([grid, [hi + 1e-6]])


def apply_decision_rules(
    scores_df: pd.DataFrame,
    t1: float,
    t2: float,
    enable_reverse_consistency: bool = False,
    score_column: str = "p_cal",
) -> Dict[str, Set[str]]:
    """Turn candidate scores into a prediction per S1 entity.

    Rules, applied in order within each entity:

    1. A candidate is eligible when its score is at least ``t1``. If nothing is eligible,
       the entity predicts the empty set and scores 1.0 if it truly is a singleton.
    2. Among eligible candidates, the highest score is taken first. It is kept only if it
       reaches ``t2``; ``t2`` is the bar for *continuing* to accept further candidates,
       so with ``t1 == t2`` this is identical to plain global thresholding.
    3. Every subsequent candidate must reach ``t2`` to be appended.
    4. Predictions are de-duplicated and order-independent, so a repeated candidate can
       never inflate a score.

    Parameters
    ----------
    enable_reverse_consistency
        When ``True``, a candidate is also required to have been *retrieved in reverse*
        (candidate -> s1 direction). With no reverse-retrieval column present this is a
        no-op rather than a silent behaviour change, and the decision config records the
        requested value.

    Returns
    -------
    dict
        Mapping from ``s1_id`` to the set of predicted candidate ids. Entities with no
        eligible candidate are present with an empty set, so the caller can distinguish
        "abstained" from "not evaluated".
    """
    if score_column not in scores_df.columns:
        raise ValueError(f"scores_df is missing score column {score_column!r}")
    if not scores_df["s1_id"].notna().all():
        raise ValueError("scores_df contains missing s1_id values")

    predictions: Dict[str, Set[str]] = {}

    for s1_id, block in scores_df.groupby("s1_id", sort=False):
        if enable_reverse_consistency and "retrieved_reverse" in block.columns:
            block = block[block["retrieved_reverse"].astype(bool)]

        # Sort by score descending, then by candidate id so ties break deterministically
        # instead of depending on input row order.
        ordered = block.sort_values(
            by=[score_column, "candidate_id"], ascending=[False, True], kind="mergesort"
        )
        scores = ordered[score_column].to_numpy(dtype=float)
        candidates = ordered["candidate_id"].to_numpy()

        selected: Set[str] = set()
        for rank, (score, candidate_id) in enumerate(zip(scores, candidates)):
            if rank == 0:
                # Entry gate.
                if score < t1:
                    break
                if score < t2:
                    # Eligible for entry but not strong enough to continue: abstain.
                    break
            else:
                if score < t2:
                    break
            selected.add(str(candidate_id))

        predictions[str(s1_id)] = selected

    return predictions


class _PreparedScores:
    """Pre-grouped, pre-sorted scores for fast repeated threshold evaluation.

    Tuning evaluates a grid of up to ``n_grid ** 2`` ``(T1, T2)`` pairs, and re-running a
    ``groupby`` + ``sort_values`` for every one of them dominates the runtime â€” on real
    candidate volumes that is the difference between minutes and hours.

    The data is sorted once by ``(s1_id, score descending, candidate_id)`` so each entity
    occupies one contiguous block, ordered strongest candidate first.

    The selection rule simplifies to a closed form, because candidates are sorted by
    descending score:

    * the first (strongest) candidate is accepted only if it clears ``max(T1, T2)``;
    * every later candidate is accepted if it clears ``T2``, which â€” being sorted
      descending â€” is a prefix of the block.

    So an entity contributes all candidates with ``score >= T2`` when its best score
    reaches ``max(T1, T2)``, and nothing otherwise. That is exactly what the readable
    row-by-row :func:`apply_decision_rules` computes, and
    ``test_fast_path_matches_reference_implementation`` asserts the two agree.
    """

    __slots__ = ("_s1", "_candidate", "_score", "_starts", "n_entities")

    def __init__(self, scores_df: pd.DataFrame, score_column: str) -> None:
        ordered = scores_df.sort_values(
            by=["s1_id", score_column, "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        self._s1 = ordered["s1_id"].to_numpy()
        self._candidate = ordered["candidate_id"].to_numpy()
        self._score = ordered[score_column].to_numpy(dtype=float)

        s1 = self._s1
        if s1.size == 0:
            self._starts = np.zeros(0, dtype=np.int64)
        else:
            change = np.empty(s1.shape[0], dtype=bool)
            change[0] = True
            change[1:] = s1[1:] != s1[:-1]
            self._starts = np.flatnonzero(change).astype(np.int64)
        self.n_entities = int(self._starts.size)

    @property
    def best_scores(self) -> np.ndarray:
        """Best (maximum) score per entity, in sorted entity order."""
        if self.n_entities == 0:
            return np.zeros(0, dtype=float)
        return self._score[self._starts]

    def _ends(self) -> np.ndarray:
        starts = self._starts
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = self._score.size
        return ends

    def predictions(self, t1: float, t2: float) -> Dict[str, Set[str]]:
        """Apply the decision rules for one ``(T1, T2)`` pair."""
        predictions: Dict[str, Set[str]] = {}
        if self.n_entities == 0:
            return predictions

        active = self.best_scores >= max(float(t1), float(t2))
        ends = self._ends()
        starts = self._starts
        cutoff = float(t2)

        for entity_index in range(self.n_entities):
            start = int(starts[entity_index])
            end = int(ends[entity_index])
            s1_id = str(self._s1[start])
            if not active[entity_index]:
                predictions[s1_id] = set()
                continue
            block = self._score[start:end]
            # block is descending, so -block is ascending: count how many clear the cutoff.
            keep = int(np.searchsorted(-block, -cutoff, side="right"))
            predictions[s1_id] = {str(c) for c in self._candidate[start : start + keep]}
        return predictions


def evaluate_thresholds(
    scores_df: pd.DataFrame,
    ground_truth_by_s1: Mapping[str, Iterable[str]],
    t1: float,
    t2: float,
    score_column: str = "p_cal",
    prepared: Optional["_PreparedScores"] = None,
) -> float:
    """Macro-F0.5 of a ``(t1, t2)`` decision, over every ground-truth entity."""
    prepared = prepared or _PreparedScores(_validate_oof(scores_df), score_column)
    predictions = prepared.predictions(t1, t2)
    return macro_f05(_normalize_ground_truth(ground_truth_by_s1), predictions)


def tune_global_threshold(
    oof_scores_df: pd.DataFrame,
    ground_truth_by_s1: Mapping[str, Iterable[str]],
    n_grid: int = DEFAULT_THRESHOLD_GRID_SIZE,
    score_column: str = "p_cal",
) -> float:
    """Grid-search the single global threshold maximizing macro-F0.5 on OOF scores.

    Equivalent to ``tune_two_thresholds`` restricted to ``t1 == t2``, and kept as a named
    entry point because the single-threshold result is the baseline the two-threshold layer
    has to beat.
    """
    scores_df = _validate_oof(oof_scores_df)
    grid = build_threshold_grid(scores_df[score_column].to_numpy(dtype=float), n_grid=n_grid)
    prepared = _PreparedScores(scores_df, score_column)

    best_threshold = float(grid[0])
    best_score = -np.inf
    for threshold in grid:
        score = evaluate_thresholds(
            scores_df, ground_truth_by_s1, float(threshold), float(threshold),
            score_column=score_column, prepared=prepared,
        )
        # Ties resolve to the lowest threshold, which is the more conservative choice.
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def tune_two_thresholds(
    oof_scores_df: pd.DataFrame,
    ground_truth_by_s1: Mapping[str, Iterable[str]],
    n_grid: int = DEFAULT_THRESHOLD_GRID_SIZE,
    score_column: str = "p_cal",
    require_t2_ge_t1: bool = True,
) -> Tuple[float, float]:
    """Grid-search the ``(T1, T2)`` pair maximizing macro-F0.5 on OOF scores.

    ``T1 <= T2`` is enforced by default, so the entry bar is never stricter than the
    continuation bar. That ordering is what makes the "accept one strong candidate, demand
    more for the rest" behaviour expressible; relaxing it is available for experiments.

    Returns
    -------
    tuple
        ``(t1, t2)``, rounded to 6 decimals for a stable, readable frozen config.
    """
    scores_df = _validate_oof(oof_scores_df)
    grid = build_threshold_grid(scores_df[score_column].to_numpy(dtype=float), n_grid=n_grid)
    prepared = _PreparedScores(scores_df, score_column)

    best_t1 = float(grid[0])
    best_t2 = float(grid[0])
    best_score = -np.inf

    for t1 in grid:
        for t2 in grid:
            if require_t2_ge_t1 and t2 < t1:
                continue
            score = evaluate_thresholds(
                scores_df, ground_truth_by_s1, t1, t2, score_column, prepared=prepared
            )
            if score > best_score:
                best_score = score
                best_t1 = float(t1)
                best_t2 = float(t2)

    return round(best_t1, 6), round(best_t2, 6)


class _EntityTally:
    """Per-entity arrays that turn threshold search into array arithmetic.

    The reference path materialises a ``dict`` of ``set``s at every one of the
    ``n_grid ** 2`` grid points, so it pays a full Python pass over the candidates per
    grid point. On a real candidate set that is hours rather than minutes, and it is the
    same shape of problem as the retrieval pivot in ``src.features.retrieval_features``.

    Candidates are pre-sorted by descending score, so the ones clearing a threshold form
    a *prefix* of each entity's block. That means two prefix sums are all a ``t2`` needs:

    * ``keep`` -- rows with ``score >= t2``, so ``keep_i`` is a prefix length;
    * ``tp`` -- rows that are true matches, so the true positives inside that prefix are a
      prefix sum too.

    ``t1`` never changes *which* candidates are emitted, only whether the entity abstains
    at all: it contributes nothing when ``best_score < max(t1, t2)``. So one vectorised
    pass per ``t2`` serves every ``t1`` on that row of the grid.
    """

    __slots__ = (
        "_score", "_starts", "_ends", "_best", "_first", "_in_truth",
        "_k_true", "_n_truth", "_n_singletons", "_n_scored_singletons",
    )

    def __init__(self, scores_df: pd.DataFrame, ground_truth_by_s1, score_column: str) -> None:
        truth = _normalize_ground_truth(ground_truth_by_s1)
        self._n_truth = len(truth)
        self._n_singletons = sum(1 for m in truth.values() if not m)

        ordered = scores_df.sort_values(
            by=["s1_id", score_column, "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        s1 = ordered["s1_id"].to_numpy()
        candidate = ordered["candidate_id"].to_numpy()
        self._score = ordered[score_column].to_numpy(dtype=float)

        if s1.size:
            change = np.empty(s1.shape[0], dtype=bool)
            change[0] = True
            change[1:] = s1[1:] != s1[:-1]
            self._starts = np.flatnonzero(change).astype(np.int64)
        else:
            self._starts = np.zeros(0, dtype=np.int64)
        self._ends = np.empty_like(self._starts)
        if self._starts.size:
            self._ends[:-1] = self._starts[1:]
            self._ends[-1] = self._score.size

        # A candidate counts once, no matter how many rows it occupies. Rows are sorted by
        # descending score, so a candidate's *first* row carries its maximum score and a
        # candidate is in the selected set exactly when that first row clears the
        # threshold. Counting only first occurrences therefore reproduces the reference's
        # set semantics; counting rows would let a duplicate inflate n_pred and push
        # F0.5 above 1.0.
        #
        # ``duplicated`` rather than an adjacency test: two rows for the same candidate can
        # carry different scores, so after sorting by score they are *not* necessarily
        # adjacent (``A(0.9), B(0.8), A(0.7)``). ``duplicated`` keys on the group identity
        # and keeps whichever row it saw first, which is the highest-scoring one.
        if s1.size:
            keys = pd.DataFrame({"_s1": s1, "_cand": candidate})
            first = ~keys.duplicated(subset=["_s1", "_cand"]).to_numpy()
        else:
            first = np.zeros(0, dtype=bool)
        self._first = first

        # Per-row truth flag via a hash join rather than a Python loop over candidates.
        in_truth = np.zeros(s1.shape[0], dtype=bool)
        if s1.size and truth:
            pairs_s1: List[str] = []
            pairs_c: List[str] = []
            for s1_id, matches in truth.items():
                if matches:
                    pairs_s1.extend([s1_id] * len(matches))
                    pairs_c.extend(matches)
            if pairs_s1:
                joined = pd.DataFrame({"s1_id": pairs_s1, "candidate_id": pairs_c})
                joined["_tp"] = 1
                marked = pd.DataFrame({"s1_id": s1, "candidate_id": candidate})
                marked = marked.merge(joined, on=["s1_id", "candidate_id"], how="left")
                in_truth = marked["_tp"].fillna(0).to_numpy().astype(bool)
        self._in_truth = in_truth
        self._best = self._score[self._starts] if self._starts.size else np.zeros(0, float)
        # ``get`` with an empty default: a truth entity with no candidate rows is not in
        # ``_starts`` and is accounted for by the unscored-entity tally below, so it must
        # not be indexed here.
        self._k_true = np.array(
            [len(truth.get(str(s1[s]), ())) for s in self._starts], dtype=np.float64
        ) if self._starts.size else np.zeros(0, dtype=np.float64)
        self._n_scored_singletons = int((self._k_true == 0).sum())

    @property
    def best_scores(self) -> np.ndarray:
        return self._best

    def keep_and_tp(self, t2: float) -> Tuple[np.ndarray, np.ndarray]:
        """``(keep, tp)`` per scored entity for one ``t2`` value."""
        if self._starts.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty
        # Both tallies must be prefix sums of the *same* masked array: the true positives
        # that survive are the ones inside the kept prefix, not every truth row in the
        # entity block. A block-wide truth count would credit matches the threshold
        # excluded, and can drive F0.5 above 1.0.
        mask = (self._score >= float(t2)) & self._first
        zero = np.zeros(1, dtype=np.int64)
        keep_cum = np.concatenate((zero, np.cumsum(mask)))
        keep = keep_cum[self._ends] - keep_cum[self._starts]
        tp_cum = np.concatenate((zero, np.cumsum(mask & self._in_truth)))
        tp = tp_cum[self._ends] - tp_cum[self._starts]
        return keep.astype(np.float64), tp.astype(np.float64)

    def macro_f05_for(self, t1: float, t2: float, keep: np.ndarray, tp: np.ndarray) -> float:
        """Macro-F0.5 for one ``(T1, T2)`` pair, from the precomputed ``t2`` tallies."""
        if self._n_truth == 0:
            return 0.0
        if self._starts.size == 0:
            return float(self._n_singletons) / self._n_truth

        active = self._best >= max(float(t1), float(t2))
        n_pred = np.where(active, keep, 0.0)
        tp = np.where(active, tp, 0.0)
        k_true = self._k_true

        # F0.5 written in counts: 1.25 * tp / (0.25 * n_true + n_pred). This is the
        # precision/recall form with n_pred and n_true divided through, and it is exact.
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(n_pred > 0, 1.25 * tp / (0.25 * k_true + n_pred), 0.0)
        # A singleton that emitted nothing is correct; one that emitted anything is not.
        f = np.where(k_true == 0, np.where(n_pred == 0, 1.0, 0.0), f)

        total = float(f.sum())
        # Truth entities with no candidate rows at all are n_pred=0, so singletons score
        # 1.0 and everything else 0.0 -- they still count in the mean.
        total += self._n_singletons - self._n_scored_singletons
        return total / self._n_truth


def tune_two_thresholds_fast(
    oof_scores_df: pd.DataFrame,
    ground_truth_by_s1: Mapping[str, Iterable[str]],
    n_grid: int = DEFAULT_THRESHOLD_GRID_SIZE,
    score_column: str = "p_cal",
    require_t2_ge_t1: bool = True,
) -> Tuple[float, float]:
    """Vectorised equivalent of :func:`tune_two_thresholds`.

    Identical grid, identical tie-breaking (lowest ``T1`` then lowest ``T2`` wins) and
    identical ``macro_f05`` values; only the per-grid-point cost changes, from a Python
    pass over every candidate to array arithmetic over the score column. The reference
    implementation is retained and ``tests/test_threshold_fast.py`` asserts the two agree
    exactly on randomised inputs, including degenerate ones.
    """
    scores_df = _validate_oof(oof_scores_df)
    grid = build_threshold_grid(scores_df[score_column].to_numpy(dtype=float), n_grid=n_grid)
    tally = _EntityTally(scores_df, ground_truth_by_s1, score_column)

    best_t1 = float(grid[0])
    best_t2 = float(grid[0])
    best_score = -np.inf

    for t2 in grid:
        keep, tp = tally.keep_and_tp(float(t2))
        for t1 in grid:
            if require_t2_ge_t1 and t2 < t1:
                continue
            score = tally.macro_f05_for(float(t1), float(t2), keep, tp)
            if score > best_score:
                best_score = score
                best_t1 = float(t1)
                best_t2 = float(t2)

    return round(best_t1, 6), round(best_t2, 6)


def save_decision_config(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Path:
    """Write the frozen decision parameters to ``artifacts/decision_config.json``."""
    if output_path is None:
        output_path = Path("artifacts") / DECISION_CONFIG_FILENAME
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_decision_config(
    input_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a frozen decision config written by :func:`save_decision_config`."""
    if input_path is None:
        input_path = Path("artifacts") / DECISION_CONFIG_FILENAME
    with Path(input_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_decision_config(
    t_global: float,
    t1: float,
    t2: float,
    model_version: str = "model_v1",
    enable_reverse_consistency: bool = False,
    n_oof_rows: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the decision config document, recording that it was tuned on OOF only."""
    config: Dict[str, Any] = {
        "t_global": round(float(t_global), 6),
        "t1": round(float(t1), 6),
        "t2": round(float(t2), 6),
        "model_version": model_version,
        "enable_reverse_consistency": bool(enable_reverse_consistency),
        "n_oof_rows": int(n_oof_rows),
        "tuned_on": "oof",
    }
    if extra:
        config.update(extra)
    return config


def report_decision(
    scores_df: pd.DataFrame,
    ground_truth_by_s1: Mapping[str, Iterable[str]],
    t1: float,
    t2: float,
    score_column: str = "p_cal",
) -> Dict[str, Any]:
    """Summarise a decision for logging: abstentions, matches emitted and macro-F0.5."""
    truth = _normalize_ground_truth(ground_truth_by_s1)
    predictions = apply_decision_rules(scores_df, t1=t1, t2=t2, score_column=score_column)

    abstained = [s1 for s1, matched in predictions.items() if not matched]
    singleton_truth = [s1 for s1, matches in truth.items() if not matches]
    singleton_abstained = [s1 for s1 in singleton_truth if not predictions.get(s1)]

    emitted = sum(len(matched) for matched in predictions.values())
    true_positives = sum(
        len(predictions.get(s1, set()) & matches) for s1, matches in truth.items()
    )

    return {
        "t1": float(t1),
        "t2": float(t2),
        "macro_f05": macro_f05(truth, predictions),
        "n_entities": len(truth),
        "n_predictions": emitted,
        "n_true_positives": true_positives,
        "n_abstained": len(abstained),
        "abstention_rate": len(abstained) / len(predictions) if predictions else 0.0,
        "n_singleton_entities": len(singleton_truth),
        "n_singleton_abstained": len(singleton_abstained),
        "n_singleton_violated": len(singleton_truth) - len(singleton_abstained),
    }
