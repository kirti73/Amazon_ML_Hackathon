"""The vectorised threshold tuner must be indistinguishable from the reference.

``tune_two_thresholds_fast`` exists purely for speed: it shares no control flow with
``tune_two_thresholds``. That makes agreement a real property to test rather than a
tautology, so these tests compare the two on randomised inputs -- including the
degenerate shapes that a fast path is most likely to get wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.decision.threshold import (
    _EntityTally,
    apply_decision_rules,
    tune_global_threshold,
    tune_two_thresholds,
    tune_two_thresholds_fast,
)
from src.validation.metrics import macro_f05


def _make_case(rng, n_entities, max_cands, pos_rate, singleton_rate, tie_scores=False):
    """Build a (scores frame, ground truth) pair with a realistic shape."""
    s1_ids, candidates, scores, truth = [], [], [], {}
    for i in range(n_entities):
        s1 = f"S1-{i}"
        k = int(rng.integers(0, max_cands + 1))
        if rng.random() < singleton_rate:
            truth[s1] = set()
        else:
            truth[s1] = {f"S2-{i}-{j}" for j in range(int(rng.integers(1, 3)))}
        for j in range(k):
            s1_ids.append(s1)
            candidates.append(f"S2-{i}-{j}")
            if tie_scores:
                # Coarse score values force many exact ties at the threshold.
                scores.append(float(rng.integers(0, 4)) / 4.0)
            else:
                scores.append(float(rng.random()))
    if not s1_ids:
        return pd.DataFrame({"pair_key": [], "s1_id": [], "candidate_id": [], "p_cal": []}), truth
    scores = np.array(scores)
    frame = pd.DataFrame(
        {
            "pair_key": [f"{a}::{b}" for a, b in zip(s1_ids, candidates)],
            "s1_id": s1_ids,
            "candidate_id": candidates,
            "p_cal": scores.astype("float32"),
            "p_raw": scores.astype("float32"),
            "is_oof": True,
        }
    )
    # Some entities are given no candidates at all, so truth keys are missing from the
    # score frame -- the "no prediction" branch both implementations must agree on.
    truth_pairs = {(s, c) for s, m in truth.items() for c in m}
    for (s, c) in truth_pairs:
        if rng.random() > pos_rate:
            continue
        if (s, c) in set(zip(frame["s1_id"], frame["candidate_id"])):
            continue
        frame.loc[len(frame)] = [
            f"{s}::{c}", s, c, float(rng.random()), float(rng.random()), True,
        ]
    return frame, truth


def test_fast_matches_reference_randomised():
    rng = np.random.default_rng(0)
    for trial in range(12):
        frame, truth = _make_case(
            rng,
            n_entities=int(rng.integers(2, 40)),
            max_cands=int(rng.integers(1, 6)),
            pos_rate=float(rng.uniform(0.2, 1.0)),
            singleton_rate=float(rng.uniform(0.0, 0.5)),
            tie_scores=bool(trial % 2),
        )
        n_grid = int(rng.integers(3, 12))
        ref = tune_two_thresholds(frame, truth, n_grid=n_grid)
        fast = tune_two_thresholds_fast(frame, truth, n_grid=n_grid)
        assert ref == fast, f"trial {trial}: reference {ref} != fast {fast}"


def test_fast_matches_reference_allow_t2_lt_t1():
    rng = np.random.default_rng(7)
    frame, truth = _make_case(rng, 20, 5, 0.8, 0.3)
    ref = tune_two_thresholds(frame, truth, n_grid=9, require_t2_ge_t1=False)
    fast = tune_two_thresholds_fast(frame, truth, n_grid=9, require_t2_ge_t1=False)
    assert ref == fast


def test_tally_macro_f05_matches_reference_decision_rules():
    """The per-``t2`` tallies must reproduce ``apply_decision_rules`` exactly."""
    rng = np.random.default_rng(11)
    frame, truth = _make_case(rng, 60, 4, 0.9, 0.35)
    tally = _EntityTally(frame, truth, "p_cal")
    for t2 in (0.0, 0.15, 0.3, 0.5, 0.75, 0.999):
        for t1 in (0.0, 0.2, 0.4, 0.6, 0.9):
            keep, tp = tally.keep_and_tp(t2)
            fast = tally.macro_f05_for(t1, t2, keep, tp)
            ref = macro_f05(
                {k: set(v) for k, v in truth.items()},
                apply_decision_rules(frame, t1=t1, t2=t2),
            )
            assert fast == pytest.approx(ref, abs=1e-12), f"t1={t1} t2={t2}"


def test_entities_absent_from_scores_still_count():
    """A truth entity with no candidates scores 1.0 if singleton, else 0.0, and counts."""
    frame = pd.DataFrame(
        {
            "pair_key": ["S1-0::S2-0-0"],
            "s1_id": ["S1-0"],
            "candidate_id": ["S2-0-0"],
            "p_cal": np.array([0.9], dtype="float32"),
            "is_oof": [True],
        }
    )
    truth = {"S1-0": {"S2-0-0"}, "S1-missing": set(), "S1-miss2": {"S2-x"}}
    ref = tune_two_thresholds(frame, truth, n_grid=5)
    fast = tune_two_thresholds_fast(frame, truth, n_grid=5)
    assert ref == fast
    # 1 correct + 1 correct abstain + 1 missed = 2/3 at best.
    assert fast[0] <= 0.9 + 1e-6


def test_all_singletons_prefers_silence():
    frame = pd.DataFrame(
        {
            "pair_key": ["S1-0::S2-0-0", "S1-1::S2-1-0"],
            "s1_id": ["S1-0", "S1-1"],
            "candidate_id": ["S2-0-0", "S2-1-0"],
            "p_cal": np.array([0.99, 0.98], dtype="float32"),
            "is_oof": [True, True],
        }
    )
    truth = {"S1-0": set(), "S1-1": set()}
    ref = tune_global_threshold(frame, truth, n_grid=11)
    t1, t2 = tune_two_thresholds_fast(frame, truth, n_grid=11)
    # The two-threshold optimum with t1 == t2 must reproduce the single-threshold answer.
    assert ref == pytest.approx(max(t1, t2), abs=1e-6)


def test_empty_scores_frame():
    empty = pd.DataFrame(
        {"pair_key": [], "s1_id": [], "candidate_id": [], "p_cal": np.array([], dtype="float32")}
    )
    truth = {"S1-a": set(), "S1-b": set()}
    ref = tune_two_thresholds(empty, truth, n_grid=5)
    fast = tune_two_thresholds_fast(empty, truth, n_grid=5)
    assert ref == fast


def test_all_identical_scores():
    """Every candidate ties; the threshold either takes all of them or none."""
    frame = pd.DataFrame(
        {
            "pair_key": ["S1-0::A", "S1-0::B", "S1-1::C"],
            "s1_id": ["S1-0", "S1-0", "S1-1"],
            "candidate_id": ["A", "B", "C"],
            "p_cal": np.array([0.5, 0.5, 0.5], dtype="float32"),
            "is_oof": [True, True, True],
        }
    )
    truth = {"S1-0": {"A"}, "S1-1": set()}
    ref = tune_two_thresholds(frame, truth, n_grid=6)
    fast = tune_two_thresholds_fast(frame, truth, n_grid=6)
    assert ref == fast


def test_duplicate_candidate_ids_do_not_inflate():
    """A repeated candidate must not count twice, in either implementation."""
    frame = pd.DataFrame(
        {
            "pair_key": ["S1-0::A", "S1-0::A", "S1-0::A"],
            "s1_id": ["S1-0", "S1-0", "S1-0"],
            "candidate_id": ["A", "A", "A"],
            "p_cal": np.array([0.9, 0.9, 0.9], dtype="float32"),
            "is_oof": [True, True, True],
        }
    )
    truth = {"S1-0": {"A"}}
    ref = tune_two_thresholds(frame, truth, n_grid=5)
    fast = tune_two_thresholds_fast(frame, truth, n_grid=5)
    assert ref == fast
    # One emitted match out of one truth: a perfect 1.0, not a diluted precision.
    tally = _EntityTally(frame, truth, "p_cal")
    keep, tp = tally.keep_and_tp(0.0)
    assert tally.macro_f05_for(0.0, 0.0, keep, tp) == pytest.approx(1.0)
