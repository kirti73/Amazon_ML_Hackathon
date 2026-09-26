"""Binary training labels and hard negatives for candidate pairs.

The label of a candidate pair is decided entirely by whether the pair appears in
ground truth:

* ``y == 1`` â€” the pair is a true match.
* ``y == 0`` â€” the pair was retrieved by some route but is not a true match. These are
  the *hard negatives*: they looked similar enough to survive retrieval, so they are the
  negatives that actually train a useful classifier.

Two invariants matter more than anything else here, because the rest of the pipeline
depends on them:

1. **Row count is preserved exactly.** Labeling never drops, merges, deduplicates or
   invents rows. The output has exactly one row per input candidate row, in the input
   order. A dropped row is an unscoreable pair; an invented row is a fabricated training
   example.
2. **Identifiers stay strings.** ``S1-â€¦``/``S2-â€¦``/``S3-â€¦`` ids are never coerced to a
   numeric dtype.

Ground truth is accepted in any of three shapes so this works against both the real
dataset and any upstream loader:

* **Wide** â€” the real ``train_ground_truth.tsv`` layout: one row per S1 entity with a
  comma-separated ``matched_entity_ids`` field. An empty field is a true singleton.
* **Long** â€” one row per true pair, ``s1_id`` + ``candidate_id``.
* **Mapping** â€” ``{s1_id: {candidate_id, ...}}``, the shape already used by
  ``src/validation/metrics.py`` and ``src/validation/splits.py``.

Note on ``metrics.build_ground_truth_from_tsv``: that helper assumes a long one-row-per-pair
file with ``entity_id`` columns and therefore does not match the real wide ground-truth
file. It is left untouched as part of the protected 20/20 baseline and is not used here.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Set, Union

import pandas as pd

PAIR_KEY_SEPARATOR = "::"

GT_S1_LONG = "s1_id"
GT_CANDIDATE_LONG = "candidate_id"
GT_S1_WIDE = "source1_entity_id"
GT_MATCHES_WIDE = "matched_entity_ids"

GroundTruthInput = Union[pd.DataFrame, Mapping[str, Iterable[str]]]


def make_pair_key(s1_id: str, candidate_id: str) -> str:
    """Build the canonical ``pair_key`` for a candidate pair."""
    return f"{s1_id}{PAIR_KEY_SEPARATOR}{candidate_id}"


def _normalize_ground_truth(ground_truth: GroundTruthInput) -> Set[str]:
    """Return the set of true positive ``pair_key`` values.

    Accepts the wide comma-separated layout, the long one-row-per-pair layout, or a
    mapping. The wide parse is fully vectorized.
    """
    if isinstance(ground_truth, Mapping):
        pair_keys: Set[str] = set()
        for s1_id, candidates in ground_truth.items():
            s1_id = str(s1_id).strip()
            if not s1_id:
                continue
            for candidate_id in candidates or ():
                candidate_id = str(candidate_id).strip()
                if candidate_id:
                    pair_keys.add(make_pair_key(s1_id, candidate_id))
        return pair_keys

    if not isinstance(ground_truth, pd.DataFrame):
        raise TypeError(
            "ground_truth must be a pandas DataFrame or a Mapping of "
            f"s1_id -> candidate ids, got {type(ground_truth).__name__}"
        )

    df = ground_truth
    if GT_S1_WIDE in df.columns and GT_MATCHES_WIDE in df.columns:
        raw = df[GT_MATCHES_WIDE]
        if raw.isna().any():
            raw = raw.fillna("")
        non_empty = raw.astype(str).str.strip().str.len() > 0
        subset = df.loc[non_empty, [GT_S1_WIDE, GT_MATCHES_WIDE]].copy()
        subset["candidate_id"] = subset[GT_MATCHES_WIDE].astype(str).str.split(",")
        subset = subset.explode("candidate_id")
        subset["candidate_id"] = subset["candidate_id"].astype(str).str.strip()
        subset = subset[subset["candidate_id"].str.len() > 0]
        s1 = subset[GT_S1_WIDE].astype(str).str.strip()
        return set(make_pair_key(s1_id, cand) for s1_id, cand in zip(s1, subset["candidate_id"]))

    if GT_S1_LONG in df.columns and GT_CANDIDATE_LONG in df.columns:
        s1 = df[GT_S1_LONG].astype(str).str.strip()
        cand = df[GT_CANDIDATE_LONG].astype(str).str.strip()
        mask = (s1.str.len() > 0) & (cand.str.len() > 0)
        return set(make_pair_key(s1_id, c) for s1_id, c in zip(s1[mask], cand[mask]))

    raise ValueError(
        "ground_truth DataFrame must contain either "
        f"({GT_S1_WIDE!r}, {GT_MATCHES_WIDE!r}) or "
        f"({GT_S1_LONG!r}, {GT_CANDIDATE_LONG!r}) columns; got {list(df.columns)}"
    )


def build_true_positive_pairs(ground_truth: GroundTruthInput) -> Set[str]:
    """Return the set of ``pair_key`` values that are true matches."""
    return _normalize_ground_truth(ground_truth)


def build_train_labels(
    candidates_df: pd.DataFrame,
    ground_truth_df: GroundTruthInput,
    pair_key_column: str = "pair_key",
) -> pd.DataFrame:
    """Label candidate pairs as true matches (``y=1``) or hard negatives (``y=0``).

    Parameters
    ----------
    candidates_df
        Candidate pairs. Must contain a ``pair_key`` column. Row order and row count are
        preserved exactly.
    ground_truth_df
        Ground truth in wide, long or mapping form; see the module docstring.
    pair_key_column
        Name of the pair key column in ``candidates_df``.

    Returns
    -------
    pandas.DataFrame
        Columns ``pair_key`` (string) and ``y`` (int8), one row per input candidate row.
    """
    if not isinstance(candidates_df, pd.DataFrame):
        raise TypeError(
            f"candidates_df must be a pandas DataFrame, got {type(candidates_df).__name__}"
        )
    if pair_key_column not in candidates_df.columns:
        raise ValueError(
            f"candidates_df must contain a {pair_key_column!r} column; "
            f"got {list(candidates_df.columns)}"
        )
    if candidates_df[pair_key_column].isna().any():
        raise ValueError(f"{pair_key_column!r} contains missing values")

    true_positive_pairs = _normalize_ground_truth(ground_truth_df)

    pair_keys = candidates_df[pair_key_column]
    # Vectorized membership test against the true positive pair keys.
    y = pair_keys.isin(true_positive_pairs).astype("int8")

    labels = pd.DataFrame(
        {
            "pair_key": pair_keys.astype(str).to_numpy(),
            "y": y.to_numpy(),
        }
    )
    return labels


def candidate_recall(
    candidates_df: pd.DataFrame,
    ground_truth_df: GroundTruthInput,
    pair_key_column: str = "pair_key",
) -> "tuple[int, int, float]":
    """Report how many true match pairs the candidate set actually contains.

    This is the retrieval ceiling: no model can recover a true pair that retrieval never
    produced. The master plan requires candidate recall to be reported as a release gate,
    so it is exposed here next to the label builder that consumes ground truth.

    Returns
    -------
    tuple
        ``(covered_true_pairs, total_true_pairs, recall)`` where ``recall`` is
        ``covered / total``, or ``1.0`` when there are no true pairs at all.
    """
    true_positive_pairs = _normalize_ground_truth(ground_truth_df)
    total = len(true_positive_pairs)
    if total == 0:
        return 0, 0, 1.0
    if not isinstance(candidates_df, pd.DataFrame) or pair_key_column not in candidates_df.columns:
        return 0, total, 0.0
    covered = int(candidates_df[pair_key_column].isin(true_positive_pairs).sum())
    return covered, total, covered / total
