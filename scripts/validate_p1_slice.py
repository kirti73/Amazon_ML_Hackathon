"""Validate the finalized P1 slice artifacts and report blocking recall.

``scripts/run_p1_parallel.py --mode finalize`` *produces* the candidate set. This script is
the gate that decides whether the result is trustworthy, and it is deliberately a separate
program: a producer that also marks its own homework tends to define "correct" as whatever
it happened to emit.

Every check is independent of the producer and re-derives what it expects from the raw
slice inputs. The recall figure is the one that actually matters downstream: P2 can only
build features for candidates that exist, so a candidate set with poor recall caps the
whole pipeline no matter how good the model is.

Usage
-----
    python scripts/validate_p1_slice.py \
        --candidates artifacts/slice/out/candidates.parquet \
        --events artifacts/slice/out/retrieval_events.parquet \
        --slice-dir artifacts/slice/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.blocking.spill import CANONICAL_CANDIDATE_COLUMNS  # noqa: E402
from src.models.labels import make_pair_key  # noqa: E402

EXPECTED_ROUTES = (
    "exact_name",
    "tfidf_name",
    "rare_token_name",
    "numeric_address",
    "rare_token_address",
    "tfidf_address",
    "reverse_retrieval",
)
PAIR_KEY_SEPARATOR = "::"

_failures: List[str] = []
_notes: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}" + (f" :: {detail}" if detail else "")
    print(line, flush=True)
    if not ok:
        _failures.append(f"{name}: {detail}")
    return ok


def _read_ids(path: Path, column: str = "entity_id") -> Set[str]:
    """Read one id column from a slice source file (parquet or TSV)."""
    if path.suffix == ".parquet":
        return set(pq.read_table(path, columns=[column]).column(column).to_pylist())
    return set(
        pd.read_csv(path, sep="\t", usecols=[column], dtype=str, keep_default_na=False)[
            column
        ].tolist()
    )


def load_ground_truth(gt_path: Path) -> Dict[str, List[str]]:
    """Read the wide ground truth as ``s1_id -> [candidate_id, ...]``.

    Streams in chunks: the full file is 127 MB / 2.2 M rows, and the wide ``explode``
    would otherwise materialise every match at once.
    """
    out: Dict[str, List[str]] = {}
    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
        keep_default_na=False,
        chunksize=200_000,
    ):
        for s1_id, matches in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            out[s1_id] = matches.split(",") if matches else []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="artifacts/slice/out/candidates.parquet")
    ap.add_argument("--events", default="artifacts/slice/out/retrieval_events.parquet")
    ap.add_argument("--slice-dir", default="artifacts/slice/data")
    ap.add_argument("--metrics-path", default="artifacts/slice/out/p1_validation_metrics.json")
    ap.add_argument(
        "--skip-recall", action="store_true",
        help="skip the ground-truth recall check (e.g. on a test slice with no labels)",
    )
    args = ap.parse_args()

    cand_path, ev_path, slice_dir = Path(args.candidates), Path(args.events), Path(args.slice_dir)
    t_start = time.time()
    metrics: Dict[str, object] = {}

    print("=" * 72)
    print("P1 SLICE VALIDATION")
    print("=" * 72)

    # ---------------------------------------------------------------- schema / shape
    for label, path in (("candidates", cand_path), ("events", ev_path)):
        if not check(f"{label} artifact exists", path.exists(), str(path)):
            print("\nFATAL: cannot continue without both artifacts.")
            return 1

    cand_schema = pq.read_schema(cand_path).names
    check(
        "candidates schema matches docs/schemas.md section 6",
        cand_schema == list(CANONICAL_CANDIDATE_COLUMNS),
        f"got {cand_schema}",
    )
    ev_schema = pq.read_schema(ev_path).names
    check(
        "events schema",
        ev_schema == ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"],
        f"got {ev_schema}",
    )

    n_cand = pq.ParquetFile(cand_path).metadata.num_rows
    n_ev = pq.ParquetFile(ev_path).metadata.num_rows
    metrics["candidate_rows"] = n_cand
    metrics["event_rows"] = n_ev
    print(f"       candidates={n_cand:,}  events={n_ev:,}")

    # ----------------------------------------------------------------- candidates
    # Only the columns needed for the key/uniqueness/order gates; the float columns are
    # checked in a second pass so a failure still names the right thing.
    cand = pq.read_table(
        cand_path, columns=["pair_key", "s1_id", "candidate_id", "candidate_source", "n_routes"]
    ).to_pandas()
    s1 = cand["s1_id"].to_numpy()
    cand_id = cand["candidate_id"].to_numpy()
    pair_key = cand["pair_key"].to_numpy()

    check("candidates non-empty", n_cand > 0, f"{n_cand} rows")
    check("pair_key unique", not pd.Series(pair_key).duplicated().any())
    check(
        "candidates globally sorted by s1_id",
        bool(np.all(s1[1:] >= s1[:-1])),
        "required so downstream shard splitting can assume order",
    )
    check(
        "pair_key == s1_id + '::' + candidate_id",
        bool(np.all(pair_key == np.char.add(np.char.add(s1.astype(str), PAIR_KEY_SEPARATOR), cand_id.astype(str)))),
    )
    check("s1_id non-null", bool(pd.Series(s1).notna().all()))
    check("candidate_id non-null", bool(pd.Series(cand_id).notna().all()))

    allowed_sources = {"S2", "S3"}
    observed_sources = set(cand["candidate_source"].unique().tolist())
    check(
        "candidate_source in {S2, S3}",
        observed_sources <= allowed_sources,
        f"observed {sorted(observed_sources)}",
    )
    n_routes = cand["n_routes"].to_numpy()
    check(
        "1 <= n_routes <= 7",
        bool(n_routes.min() >= 1 and n_routes.max() <= len(EXPECTED_ROUTES)),
        f"min={n_routes.min()} max={n_routes.max()}",
    )
    del cand, pair_key

    # ------------------------------------------------------- id provenance vs slice
    pool_ids: Set[str] = set()
    for name in ("slice_source2.tsv", "slice_source3.tsv"):
        p = slice_dir / name
        if p.exists():
            pool_ids |= _read_ids(p)
    check("slice pool files readable", bool(pool_ids), f"{len(pool_ids):,} S2/S3 ids")

    cand = pq.read_table(cand_path, columns=["s1_id", "candidate_id"]).to_pandas()
    bad_s1 = set(cand["s1_id"].unique()) - _read_ids(slice_dir / "slice_source1.tsv")
    check("every s1_id is in the slice S1 set", not bad_s1, f"{len(bad_s1)} unknown")
    bad_cand = set(cand["candidate_id"].unique()) - pool_ids
    check("every candidate_id is in the slice S2/S3 set", not bad_cand, f"{len(bad_cand)} unknown")
    n_s1_in_candidates = cand["s1_id"].nunique()
    del cand

    # ------------------------------------------------------------------- events
    ev = pq.read_table(ev_path, columns=["s1_id", "route", "rank", "candidate_id"]).to_pandas()
    observed_routes = set(ev["route"].unique().tolist())
    check(
        "all 7 routes present in events",
        observed_routes == set(EXPECTED_ROUTES),
        f"observed {sorted(observed_routes)}",
    )
    rank = ev["rank"].to_numpy()
    check("rank >= 1", bool(rank.min() >= 1), f"min={rank.min()}")
    del ev, rank

    # The events artifact is written bucket-by-bucket (bucket = crc32(s1_id) % n_buckets),
    # so it is deliberately NOT globally s1_id-sorted. What must hold is that the buckets
    # partition cleanly -- every row of a given (s1_id, route) lands in exactly one bucket,
    # and within it rank is a dense 1..k. Both are checked below; asserting global s1_id
    # order here would be asserting a property the writer never promised.
    _notes.append(
        "retrieval_events.parquet is bucket-partitioned (crc32(s1_id) % n_buckets), "
        "not globally s1_id-sorted; candidate ordering is the sorted one"
    )

    # ------------------------------------------- events <-> candidates consistency
    ev_keys = pq.read_table(ev_path, columns=["pair_key"]).to_pandas()["pair_key"]
    cand_keys = pq.read_table(cand_path, columns=["pair_key"]).to_pandas()["pair_key"]
    ev_key_set = set(ev_keys.to_numpy())
    cand_key_set = set(cand_keys.to_numpy())
    events_not_in_candidates = len(ev_key_set - cand_key_set)
    candidates_without_events = len(cand_key_set - ev_key_set)
    check(
        "every event pair_key exists in candidates",
        events_not_in_candidates == 0,
        f"{events_not_in_candidates:,} orphan event pairs",
    )
    check(
        "every candidate has at least one event",
        candidates_without_events == 0,
        f"{candidates_without_events:,} candidates with no retrieval event",
    )
    metrics["events_rows_unique_pairs"] = len(ev_key_set)
    del ev_keys, cand_keys, ev_key_set, cand_key_set

    # -------------------------------------------------- per-entity rank monotonicity
    ev = pq.read_table(ev_path, columns=["s1_id", "route", "rank"]).to_pandas()

    # bucket is a pure function of s1_id, so map it over the unique entities once rather
    # than hashing all 36M rows.
    unique_s1 = np.unique(ev["s1_id"].to_numpy())
    bucket_of = pd.Series([_bucket_of(x) for x in unique_s1], index=pd.Index(unique_s1))
    ev_bucket = bucket_of.reindex(pd.Index(ev["s1_id"].to_numpy())).to_numpy()

    # The writer emits bucket by bucket, so the file must never go back to a lower bucket.
    check(
        "events are bucket-partitioned (non-decreasing bucket id)",
        bool(np.all(np.diff(ev_bucket) >= 0)),
        "written one bucket at a time, as the merge contract requires",
    )

    bad_ranks = 0
    s1_all = ev["s1_id"].to_numpy()
    rank_all = ev["rank"].to_numpy()
    for route, block in ev.groupby("route", sort=False):
        order = np.lexsort((block["rank"].to_numpy(), block["s1_id"].to_numpy()))
        s1 = block["s1_id"].to_numpy()[order]
        r = block["rank"].to_numpy()[order]
        # Dense 1..k per equal-s1 run: expected rank is the position within the run.
        change = np.empty(s1.shape[0], dtype=bool)
        change[0] = True
        change[1:] = s1[1:] != s1[:-1]
        starts = np.flatnonzero(change)
        sizes = np.diff(np.r_[starts, s1.shape[0]])
        expected = np.arange(1, s1.shape[0] + 1) - np.repeat(starts, sizes)
        bad_ranks += int((r != expected).sum())
    check(
        "rank is a dense 1..k within each (s1_id, route)",
        bad_ranks == 0,
        f"{bad_ranks} rows disagree",
    )
    del ev, s1_all, rank_all, ev_bucket

    # ------------------------------------------------------------------- recall
    #
    # IMPORTANT: this slice's ground truth is inherited verbatim from the full dataset, so
    # an S1 entity's `matched_entity_ids` still point at the *full* S2/S3 pool. Only a
    # minority of those ids exist in the sampled slice pool -- the rest are unreachable by
    # construction, because no blocking scheme over a 1.4M pool can return an entity that
    # is not in it. Measured on this slice, 86.4% of ground-truth pairs are pool-absent.
    #
    # Scoring recall against all of them therefore reports ~0.13 no matter how good the
    # blocking is, which says nothing about P1. The gate uses recall over the
    # *pool-resident* pairs; both numbers are reported so the gap is visible.
    gt_path = slice_dir / "slice_ground_truth.tsv"
    if not args.skip_recall and gt_path.exists():
        gt = load_ground_truth(gt_path)
        cand_keys = set(
            pq.read_table(cand_path, columns=["pair_key"]).to_pandas()["pair_key"].to_numpy()
        )
        total_pairs = 0
        resident_pairs = 0
        hit_pairs = 0
        resident_hit = 0
        s1_total = 0
        s1_with_resident = 0
        s1_resident_hit = 0
        for s1_id, matches in gt.items():
            s1_total += 1
            has_resident = False
            resident_hit_here = False
            for m in matches:
                total_pairs += 1
                in_pool = m in pool_ids
                got = make_pair_key(s1_id, m) in cand_keys
                if got:
                    hit_pairs += 1
                if in_pool:
                    resident_pairs += 1
                    has_resident = True
                    if got:
                        resident_hit += 1
                        resident_hit_here = True
            if has_resident:
                s1_with_resident += 1
                if resident_hit_here:
                    s1_resident_hit += 1

        pair_recall_all = hit_pairs / total_pairs if total_pairs else 0.0
        pair_recall_resident = resident_hit / resident_pairs if resident_pairs else 0.0
        entity_recall_resident = s1_resident_hit / s1_with_resident if s1_with_resident else 0.0
        coverage = resident_pairs / total_pairs if total_pairs else 0.0

        metrics.update(
            {
                "gt_entities": s1_total,
                "gt_pairs_total": total_pairs,
                "gt_pairs_pool_resident": resident_pairs,
                "gt_pool_coverage": round(coverage, 6),
                "gt_pairs_retrieved_all": hit_pairs,
                "pair_recall_all": round(pair_recall_all, 6),
                "pair_recall_pool_resident": round(pair_recall_resident, 6),
                "entity_recall_pool_resident": round(entity_recall_resident, 6),
                "entities_with_resident_gt": s1_with_resident,
            }
        )
        print(
            f"       ground truth is full-pool; only {coverage:.2%} of its pairs exist in the "
            f"slice pool ({resident_pairs:,} of {total_pairs:,})"
        )
        check(
            "candidate recall over POOL-RESIDENT ground-truth pairs",
            pair_recall_resident >= 0.90,
            f"pair_recall={pair_recall_resident:.4f} ({resident_hit:,}/{resident_pairs:,}) "
            f"entity_recall={entity_recall_resident:.4f} "
            f"({s1_resident_hit:,}/{s1_with_resident:,})",
        )
        print(f"       *** BLOCKING PAIR RECALL (pool-resident) = {pair_recall_resident:.4f} ***")
        _notes.append(
            f"pair_recall_all={pair_recall_all:.4f} is reported but is NOT the gate: "
            f"{1 - coverage:.2%} of this slice's ground-truth pairs reference entities "
            f"outside the sampled S2/S3 pool and are unreachable by construction"
        )
    else:
        _notes.append("recall check skipped (no slice ground truth or --skip-recall)")

    metrics["n_s1_with_candidates"] = int(n_s1_in_candidates)
    metrics["seconds"] = round(time.time() - t_start, 2)
    metrics["n_failures"] = len(_failures)
    metrics["notes"] = _notes

    out = Path(args.metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 72)
    if _failures:
        print(f"RESULT: FAIL ({len(_failures)} check(s))")
        for f in _failures:
            print(f"  - {f}")
    else:
        print("RESULT: PASS - all P1 checks green")
    for n in _notes:
        print(f"  note: {n}")
    print(f"metrics -> {out}")
    print("=" * 72)
    return 1 if _failures else 0


def _bucket_of(s1_id: str, n_buckets: int = 64) -> int:
    """The finalizer's stable bucket assignment, re-derived rather than imported."""
    return zlib.crc32(str(s1_id).encode("utf-8")) % n_buckets


if __name__ == "__main__":
    raise SystemExit(main())
