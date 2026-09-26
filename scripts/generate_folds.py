#!/usr/bin/env python
"""Generate and freeze the canonical entity-level 5-fold assignment.

Reads the real training data, assigns every S1 entity to exactly one of five folds
using the protected, committed splitter in ``src/validation/splits.py``, and writes
the frozen assignment to ``artifacts/folds.parquet``.

Fold assignment rules (already implemented and unit-tested in
``src/validation/splits.py``):

* exactly ``n_folds`` (default 5) folds, labelled ``0 .. n_folds - 1``
* every S1 entity in ``train_source1.tsv`` is assigned exactly one fold
* stratification key is ``(country, has_match, match_count_bucket)`` with buckets
  ``0`` (zero matches), ``1``, ``2-3`` and ``4+``
* candidate pairs inherit the fold of their S1 entity, so no entity is split
  across folds
* deterministic given ``random_state``

The resulting Parquet schema is fixed by contract:

===========  ========  ===========================================
column       type      meaning
===========  ========  ===========================================
``s1_id``    string    S1 entity identifier (e.g. ``S1-925783039``)
``fold_id``  int8      fold index in ``[0, n_folds)``
===========  ========  ===========================================

Usage
-----
    python scripts/generate_folds.py
    python scripts/generate_folds.py --n-folds 5 --random-state 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = REPO_ROOT / "code" / "business_entity_resolution"
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.splits import assign_folds_from_ground_truth, folds_to_parquet

DEFAULT_S1_PATH = REPO_ROOT / "dataset" / "train" / "train_source1.tsv"
DEFAULT_GT_PATH = REPO_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "folds.parquet"

CHUNK_SIZE = 200_000

S1_ID_COLUMN = "entity_id"
S1_COUNTRY_COLUMN = "country"
GT_S1_COLUMN = "source1_entity_id"
GT_MATCHES_COLUMN = "matched_entity_ids"


def load_s1_pool(s1_path: Path) -> "tuple[List[str], Dict[str, str]]":
    """Stream ``train_source1.tsv`` and collect S1 ids and their countries.

    Only ``entity_id`` and ``country`` are read. Country values are interned so the
    country mapping stays small for millions of entities.
    """
    s1_ids: List[str] = []
    countries: Dict[str, str] = {}
    seen: Set[str] = set()
    n_rows = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=[S1_ID_COLUMN, S1_COUNTRY_COLUMN],
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        n_rows += len(chunk)
        for s1_id, country in zip(chunk[S1_ID_COLUMN], chunk[S1_COUNTRY_COLUMN]):
            s1_id = s1_id.strip()
            if not s1_id:
                continue
            if s1_id in seen:
                continue
            seen.add(s1_id)
            s1_ids.append(s1_id)
            countries[s1_id] = sys.intern(country.strip())

    print(f"[load_s1_pool] rows read: {n_rows:,}")
    print(f"[load_s1_pool] unique S1 entities: {len(s1_ids):,}")
    if n_rows != len(s1_ids):
        print(
            f"[load_s1_pool] WARNING: {n_rows - len(s1_ids):,} duplicate S1 ids were skipped"
        )
    return s1_ids, countries


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Stream ``train_ground_truth.tsv`` into ``s1_id -> set of true candidate ids``.

    The file is wide: one row per S1 entity with comma-separated candidate ids.
    An empty ``matched_entity_ids`` field means a true singleton (zero-match entity).
    """
    ground_truth: Dict[str, Set[str]] = {}
    n_rows = 0
    n_zero_match = 0
    n_total_pairs = 0

    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        usecols=[GT_S1_COLUMN, GT_MATCHES_COLUMN],
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        n_rows += len(chunk)
        for s1_id, raw_matches in zip(chunk[GT_S1_COLUMN], chunk[GT_MATCHES_COLUMN]):
            s1_id = s1_id.strip()
            if not s1_id:
                continue
            raw_matches = raw_matches.strip()
            if not raw_matches:
                matches: Set[str] = set()
                n_zero_match += 1
            else:
                matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
                n_total_pairs += len(matches)
            if s1_id in ground_truth:
                ground_truth[s1_id].update(matches)
            else:
                ground_truth[s1_id] = matches

    print(f"[load_ground_truth] rows read: {n_rows:,}")
    print(f"[load_ground_truth] S1 entities with a ground-truth row: {len(ground_truth):,}")
    print(f"[load_ground_truth] zero-match (singleton) S1 entities: {n_zero_match:,}")
    print(f"[load_ground_truth] total true match pairs: {n_total_pairs:,}")
    return ground_truth


def validate_parquet(
    output_path: Path, expected_s1_ids: Set[str], n_folds: int
) -> "tuple[Dict[int, int], Dict[int, Dict[str, int]]]":
    """Validate the frozen artifact against the required contract.

    Checks exact Parquet schema, exact S1 coverage, fold-id range and reports
    the per-fold entity and stratum breakdown.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(output_path, columns=["s1_id", "fold_id"])
    schema = table.schema

    expected_schema = pa.schema(
        [pa.field("s1_id", pa.string()), pa.field("fold_id", pa.int8())]
    )
    if not schema.equals(expected_schema):
        raise AssertionError(
            f"fold schema mismatch.\n  expected: {expected_schema}\n  actual:   {schema}"
        )
    print(f"[validate] schema OK: {schema}")

    s1_col = table.column("s1_id").to_pylist()
    fold_col = table.column("fold_id").to_pylist()

    if len(s1_col) != len(expected_s1_ids):
        raise AssertionError(
            f"row count mismatch: parquet has {len(s1_col):,} rows, "
            f"expected {len(expected_s1_ids):,} S1 entities"
        )
    if len(set(s1_col)) != len(s1_col):
        raise AssertionError("duplicate s1_id values found in folds.parquet")

    written = set(s1_col)
    missing = expected_s1_ids - written
    extra = written - expected_s1_ids
    if missing:
        raise AssertionError(
            f"{len(missing):,} S1 entities are missing from folds.parquet "
            f"(first 5: {sorted(missing)[:5]})"
        )
    if extra:
        raise AssertionError(
            f"{len(extra):,} unexpected s1_id values in folds.parquet "
            f"(first 5: {sorted(extra)[:5]})"
        )
    print(f"[validate] coverage OK: all {len(expected_s1_ids):,} S1 entities assigned")

    bad_folds = {f for f in fold_col if not (0 <= f < n_folds)}
    if bad_folds:
        raise AssertionError(f"fold_id values outside [0, {n_folds}): {sorted(bad_folds)}")
    print(f"[validate] fold_id range OK: [0, {n_folds - 1}]")

    fold_counts: Dict[int, int] = {}
    for fold_id in fold_col:
        fold_counts[fold_id] = fold_counts.get(fold_id, 0) + 1

    missing_folds = set(range(n_folds)) - set(fold_counts)
    if missing_folds:
        raise AssertionError(f"folds with no entities assigned: {sorted(missing_folds)}")
    print(f"[validate] all {n_folds} folds are non-empty")

    return fold_counts, {}


def report_fold_sizes(fold_counts: Dict[int, int]) -> None:
    """Print a per-fold entity-count summary."""
    total = sum(fold_counts.values())
    print("\nFold sizes")
    print("-" * 46)
    print(f"{'fold':>6} {'entities':>14} {'share':>8}")
    for fold_id in sorted(fold_counts):
        count = fold_counts[fold_id]
        print(f"{fold_id:>6} {count:>14,} {count / total:>7.2%}")
    print("-" * 46)
    print(f"{'total':>6} {total:>14,}")
    imbalance = (max(fold_counts.values()) - min(fold_counts.values())) / total
    print(f"relative fold imbalance: {imbalance:.4%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-path", type=Path, default=DEFAULT_S1_PATH)
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    for path in (args.s1_path, args.gt_path):
        if not path.is_file():
            raise SystemExit(f"ERROR: required input not found: {path}")

    started = time.time()
    print("=" * 70)
    print("Canonical fold generation")
    print("=" * 70)
    print(f"s1 input   : {args.s1_path}")
    print(f"gt input   : {args.gt_path}")
    print(f"output     : {args.output_path}")
    print(f"n_folds    : {args.n_folds}")
    print(f"random_state: {args.random_state}")
    print("=" * 70)

    s1_ids, countries = load_s1_pool(args.s1_path)
    ground_truth_by_s1 = load_ground_truth(args.gt_path)

    missing_gt = [s1 for s1 in s1_ids if s1 not in ground_truth_by_s1]
    print(
        f"[check] S1 entities with no ground-truth row (treated as zero-match): "
        f"{len(missing_gt):,}"
    )
    for s1 in missing_gt:
        ground_truth_by_s1[s1] = set()

    print("[assign] running assign_folds_from_ground_truth ...")
    fold_assignments = assign_folds_from_ground_truth(
        s1_ids=s1_ids,
        ground_truth_by_s1=ground_truth_by_s1,
        countries=countries,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    print(f"[assign] assigned {len(fold_assignments):,} S1 entities to folds")

    if len(fold_assignments) != len(s1_ids):
        raise AssertionError(
            f"assignment covers {len(fold_assignments):,} of {len(s1_ids):,} S1 entities"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    folds_to_parquet(fold_assignments, str(args.output_path))
    print(f"[write] wrote {args.output_path}")

    fold_counts, _ = validate_parquet(
        args.output_path, set(s1_ids), args.n_folds
    )
    report_fold_sizes(fold_counts)

    print(f"\ndone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
