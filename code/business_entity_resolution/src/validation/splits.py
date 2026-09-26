"""5-fold GroupKFold splitter for entity-level entity resolution.

All candidate pairs belonging to the same s1_id are assigned to the same fold.
The stratification key is (country, has_match, match_count_bucket) at the S1 entity level.

Group integrity is always preserved: no s1_id appears in more than one fold.
Stratification is approximate: the composite key is approximately balanced
across the 5 folds.
"""

from __future__ import annotations

from collections import defaultdict
from random import Random
from typing import Dict, List, Optional, Set, Tuple


def _compute_stratification_key(
    s1_country: str,
    s1_has_match: bool,
    s1_match_count: int,
) -> Tuple[str, bool, int]:
    """Compute the stratification key for an S1 entity.

    Parameters
    ----------
    s1_country : str
        Country string (e.g. "US", "India", "France").
    s1_has_match : bool
        Whether the entity has at least one true match in ground truth.
    s1_match_count : int
        Total number of true matches for this S1 entity in ground truth.

    Returns
    -------
    Tuple[str, bool, int]
        (country, has_match, match_count_bucket) where match_count_bucket
        is 0, 1, 2-3, or 4+.
    """
    # Determine match-count bucket
    if s1_match_count == 0:
        bucket = 0
    elif s1_match_count == 1:
        bucket = 1
    elif 2 <= s1_match_count <= 3:
        bucket = 2  # "2-3"
    else:
        bucket = 3  # "4+", consistent with _bucket_from_match_count

    return (s1_country, s1_has_match, bucket)


def _bucket_from_match_count(count: int) -> int:
    """Convert match count to stratification bucket integer.

    bucket:
        0 -> 0 matches
        1 -> 1 match
        2 -> 2-3 matches
        3 -> 4+ matches (this mapping is used internally)

    Note: The public API uses match_count_bucket with values 0, 1, 2-3, 4+;
    this helper maps to compact integers for algorithmic use.
    """
    if count == 0:
        return 0
    elif count == 1:
        return 1
    elif 2 <= count <= 3:
        return 2
    else:
        return 3


def _assign_folds_stratified(
    s1_ids: List[str],
    countries: Dict[str, str],
    match_counts: Dict[str, int],
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, int]:
    """Assign each s1_id to one of n_folds using stratified round-robin.

    This is a deterministic, custom GroupKFold assignment that:
    - Groups s1_ids by their stratification key (country, has_match, bucket)
    - Within each stratification key, distributes groups across folds
      round-robin to approximately balance the key distribution
    - Preserves group integrity: each s1_id appears in exactly one fold

    Parameters
    ----------
    s1_ids : List[str]
        List of all S1 entity IDs.
    countries : Dict[str, str]
        Mapping from s1_id to country string.
    match_counts : Dict[str, int]
        Mapping from s1_id to total match count in ground truth.
    n_folds : int
        Number of folds (default 5).
    random_state : Optional[int]
        Seed for deterministic shuffling within each stratification key.

    Returns
    -------
    Dict[str, int]
        Mapping from s1_id to fold_id (0 .. n_folds-1).
    """
    if random_state is not None:
        import random
        random.seed(random_state)

    # Build stratification key per s1_id
    key_to_s1ids: Dict[Tuple[str, bool, int], List[str]] = defaultdict(list)
    for s1_id in s1_ids:
        country = countries.get(s1_id, "UNKNOWN")
        has_match = match_counts.get(s1_id, 0) > 0
        bucket = _bucket_from_match_count(match_counts.get(s1_id, 0))
        key = (country, has_match, bucket)
        key_to_s1ids[key].append(s1_id)

    # Within each stratification key, shuffle and round-robin assign to folds
    fold_assignments: Dict[str, int] = {}

    rng = Random(random_state)
    current_fold = 0

    for key in sorted(key_to_s1ids.keys()):
        s1ids = key_to_s1ids[key]
        # Shuffle for randomness (deterministic via seed)
        rng.shuffle(s1ids)

        # Round-robin assign across folds
        for s1_id in s1ids:
            fold_assignments[s1_id] = current_fold % n_folds
            current_fold += 1

    # Verify: every s1_id in the input appears exactly once
    assigned_ids = set(fold_assignments.keys())
    missing = set(s1_ids) - assigned_ids
    if missing:
        raise ValueError(f"Fold assignment missing S1 IDs: {missing}")

    extra = assigned_ids - set(s1_ids)
    if extra:
        raise ValueError(f"Fold assignment has extra S1 IDs: {extra}")

    # Verify no s1_id appears in multiple folds (should be impossible with dict)
    # but verify fold counts are consistent
    fold_counts = defaultdict(int)
    for fid in fold_assignments.values():
        fold_counts[fid] += 1

    return fold_assignments


def assign_folds_from_ground_truth(
    s1_ids: List[str],
    ground_truth_by_s1: Dict[str, Set[str]],
    countries: Dict[str, str],
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, int]:
    """Convenience wrapper: compute match counts from ground truth and assign folds.

    Parameters
    ----------
    s1_ids : List[str]
        List of all S1 entity IDs that should receive a fold assignment.
    ground_truth_by_s1 : Dict[str, Set[str]]
        Mapping from s1_id to the set of true candidate IDs (from ground truth).
        The match count is len(ground_truth_by_s1[s1_id]).
    countries : Dict[str, str]
        Mapping from s1_id to country string.
    n_folds : int
        Number of folds (default 5).
    random_state : Optional[int]
        Seed for deterministic assignment.

    Returns
    -------
    Dict[str, int]
        Mapping from s1_id to fold_id (0 .. n_folds-1).
    """
    match_counts: Dict[str, int] = {}
    for s1_id in s1_ids:
        true_matches = ground_truth_by_s1.get(s1_id, set())
        match_counts[s1_id] = len(true_matches) if true_matches else 0

    return _assign_folds_stratified(
        s1_ids=s1_ids,
        countries=countries,
        match_counts=match_counts,
        n_folds=n_folds,
        random_state=random_state,
    )


def folds_to_parquet(
    fold_assignments: Dict[str, int],
    output_path: str,
) -> None:
    """Write fold assignments to a Parquet file.

    Parameters
    ----------
    fold_assignments : Dict[str, int]
        Mapping from s1_id to fold_id (0 to n_folds-1).
    output_path : str
        Path where the Parquet file will be written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    s1_ids = sorted(fold_assignments.keys())
    fold_ids = [fold_assignments[s1] for s1 in s1_ids]

    table = pa.table(
        {
            "s1_id": pa.array(s1_ids, type=pa.string()),
            "fold_id": pa.array(fold_ids, type=pa.int8()),
        }
    )

    pq.write_table(table, output_path)