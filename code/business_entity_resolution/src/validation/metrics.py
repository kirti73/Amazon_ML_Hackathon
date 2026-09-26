"""Official macro-F0.5 metric for entity-level entity resolution evaluation.

Computes F0.5 per S1 entity and macro-averages across all S1 entities.

Definitions (from Amazon ML Challenge 2026 planning documents):

- T_i = ground-truth set of matched candidate IDs for S1 entity i
- P_i = predicted set of matched candidate IDs for S1 entity i
- TP = |P_i âˆ© T_i| (true positives, deduplicated)

Per-entity score rules:
  If |T_i| == 0:       # zero-match entity (true singleton)
    if |P_i| == 0:     # predicted empty
      score = 1.0
    else:              # predicted some false matches
      score = 0.0

  If |T_i| > 0:        # non-singleton entity (has at least one true match)
    if |P_i| == 0:     # predicted empty but has true matches
      score = 0.0
    else:
      precision = TP / |P_i|
      recall    = TP / |T_i|
      if precision + recall == 0:
        score = 0.0
      else:
        F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
        score = F0.5

Macro-F0.5 = (1 / N_S1) * sum(scores for all S1 entities)

Key design decisions:
- IDs are treated as strings; duplicate predicted IDs do not inflate TP
- Prediction order does not matter (sets, not lists)
- Every S1 entity is included in evaluation, including zero-match entities
- No ground-truth-derived features are used in the metric itself
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple


def compute_per_entity_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute per-entity F0.5 scores.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Mapping from s1_id to the set of true matched candidate IDs.
        Every S1 entity in the test set must appear as a key (even if the
        match set is empty).
    predictions : Dict[str, Set[str]]
        Mapping from s1_id to the set of predicted candidate IDs.
        May be missing keys for entities with no predictions; those are
        treated as empty predictions.

    Returns
    -------
    Dict[str, float]
        Mapping from s1_id to its F0.5 score.
    """
    n_total = 0
    scores: Dict[str, float] = {}

    # Every S1 entity in ground truth must be evaluated.
    # If an entity is missing from predictions, treat as empty prediction.
    for s1_id in ground_truth:
        n_total += 1
        true_matches = ground_truth[s1_id]  # already a Set[str]
        pred_matches = predictions.get(s1_id, set())  # Set[str], default empty

        # Normalise to sets of strings
        true_matches = set(str(x) for x in true_matches)
        pred_matches = set(str(x) for x in pred_matches)

        n_true = len(true_matches)
        n_pred = len(pred_matches)

        if n_true == 0:
            # Zero-match entity (singleton)
            if n_pred == 0:
                score = 1.0  # correctly predicted empty
            else:
                score = 0.0  # false positive(s)
        else:
            # Non-singleton entity has at least one true match
            if n_pred == 0:
                score = 0.0  # missed all true matches
            else:
                # Compute TP = |P_i âˆ© T_i|, deduplicated inherently by set intersection
                tp = len(true_matches & pred_matches)
                precision = tp / n_pred if n_pred > 0 else 0.0
                recall = tp / n_true if n_true > 0 else 0.0

                if precision + recall == 0:
                    score = 0.0
                else:
                    F05_NUMERATOR = 1.25 * precision * recall
                    F05_DENOM = 0.25 * precision + recall
                    if F05_DENOM == 0:
                        score = 0.0
                    else:
                        score = F05_NUMERATOR / F05_DENOM

        scores[s1_id] = score

    # Sanity: if ground truth was empty, return empty dict
    if n_total == 0:
        return {}

    return scores


def macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> float:
    """Compute macro-averaged F0.5 across all S1 entities.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Mapping from s1_id to the set of true matched candidate IDs.
        Every S1 entity in the test set must appear as a key (even if the
        match set is empty).
    predictions : Dict[str, Set[str]]
        Mapping from s1_id to the set of predicted candidate IDs.
        May be missing keys for entities with no predictions; those are
        treated as empty predictions.

    Returns
    -------
    float
        Macro-averaged F0.5 in [0, 1].
    """
    per_entity_scores = compute_per_entity_f05(ground_truth, predictions)

    if not per_entity_scores:
        return 0.0

    return sum(per_entity_scores.values()) / len(per_entity_scores)


def build_ground_truth_from_tsv(
    tsv_path: str,
    s1_id_column: str = "entity_id",
    candidate_id_column: str = "entity_id",
) -> Dict[str, Set[str]]:
    """Build ground-truth mapping from a ground-truth TSV file.

    The expected format (as in train_ground_truth.tsv) has one row per
    (s1_id, candidate_id) true match pair.

    Parameters
    ----------
    tsv_path : str
        Path to the ground-truth TSV file.
    s1_id_column : str
        Column name for the S1 entity ID.
    candidate_id_column : str
        Column name for the candidate entity ID.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from s1_id to the set of true candidate IDs.
    """
    import csv

    gt: Dict[str, Set[str]] = defaultdict(set)

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row[s1_id_column].strip()
            cand = row[candidate_id_column].strip()
            gt[s1].add(cand)

    # Ensure every key is a string
    return {str(s1): matches for s1, matches in gt.items()}


def build_predictions_from_dict(
    predictions_dict: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Ensure every S1 entity has an entry, defaulting to empty set.

    Parameters
    ----------
    predictions_dict : Dict[str, Set[str]]
        May be missing keys for some S1 entities.

    Returns
    -------
    Dict[str, Set[str]]
        All S1 entities from the union of keys have entries; missing keys
        map to empty set.
    """
    # If empty, return empty dict
    if not predictions_dict:
        return {}

    return {str(s1): (pred_set if pred_set is not None else set())
            for s1, pred_set in predictions_dict.items()}