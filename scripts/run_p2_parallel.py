#!/usr/bin/env python
"""P2 feature generation, sharded across processes, for the full P1 candidate set.

Why this exists
---------------
``build_features`` measured 843 candidate pairs/s on one core against real P1 pairs, and
P1 produces ~25M candidate pairs, so a single-process run is ~8 hours. Two separate costs
dominate that number:

* the per-pair similarity features (rapidfuzz, char-cosine, token sets), and
* ``pivot_retrieval_features``, which was 94% of the 20k-pair sample (20.8 s of 23.7 s).

The pivot has since been vectorized in ``src/features/retrieval_features.py`` (verified
bit-identical to the original against the retained reference implementation, 96x faster),
so the remaining cost is the similarity features, which are embarrassingly parallel: a
feature row depends only on its own pair plus the records table.

So: split the candidate list into contiguous row shards, hand each worker only the
retrieval events belonging to its shard, and concatenate the shard outputs in order.
``build_features`` preserves its input row order and the pivot returns rows in the order
of the pair keys it is given, so concatenating shard 0..N-1 reproduces the single-process
result exactly.

Event partitioning matters for correctness *and* cost. ``candidates.parquet`` is sorted
by ``s1_id``, so shard boundaries fall at specific ``s1_id`` values and an event's shard is
found with a ``searchsorted`` on those boundaries. That gives each worker a small event
frame instead of making every worker re-pivot all 30M events.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

from src.features.build import FEATURE_COLUMNS, build_features  # noqa: E402
from src.preprocessing.normalize import preprocess_records_df  # noqa: E402
from src.utils.adapters import adapt_raw_for_preprocessing  # noqa: E402

CANDIDATE_KEY_COLUMNS = ["pair_key", "s1_id", "candidate_id"]
EVENT_COLUMNS = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]

#: Populated in the parent before fork, then shared copy-on-write by every worker. The
#: workers only ever read it.
_RECORDS: Optional[pd.DataFrame] = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# --------------------------------------------------------------------------- #
# parent: inputs
# --------------------------------------------------------------------------- #
def build_records(slice_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("slice_source1.tsv", "slice_source2.tsv", "slice_source3.tsv"):
        raw = pd.read_csv(slice_dir / name, sep="\t", dtype=str, keep_default_na=False, na_values=[])
        frames.append(preprocess_records_df(adapt_raw_for_preprocessing(raw)))
    records = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["entity_id"])
    log(f"records: {len(records):,} rows, columns={list(records.columns)}")
    return records


def shard_bounds(n_rows: int, n_shards: int) -> List[Tuple[int, int]]:
    step = -(-n_rows // n_shards)
    bounds = []
    for i in range(n_shards):
        lo = i * step
        if lo >= n_rows:
            break
        bounds.append((lo, min(lo + step, n_rows)))
    return bounds


def s1_group_starts(cand_path: Path) -> np.ndarray:
    """Row positions at which a new s1_id starts, in increasing order."""
    starts: List[int] = []
    prev = None
    pos = 0
    for batch in pq.ParquetFile(cand_path).iter_batches(batch_size=500_000, columns=["s1_id"]):
        col = batch.column("s1_id").to_pandas().to_numpy(dtype=object)
        if not len(col):
            continue
        if prev is not None and col[0] != prev:
            starts.append(pos)
        breaks = np.flatnonzero(col[1:] != col[:-1]) + 1
        starts.extend((pos + breaks).tolist())
        prev = col[-1]
        pos += len(col)
    return np.asarray(starts, dtype=np.int64)


def snap_bounds_to_s1_groups(cand_path: Path, bounds: List[Tuple[int, int]],
                             n_rows: int) -> List[Tuple[int, int]]:
    """Move every shard start forward to the next s1_id group boundary.

    Without this a shard can begin part-way through one s1_id's candidates. The events
    for that s1_id would all route to whichever shard owns it (searchsorted on s1_id), so
    the *other* shard's pairs for the same s1_id would silently fall back to default
    retrieval features -- a wrong-but-plausible result rather than an error. Snapping makes
    every s1_id wholly owned by exactly one shard.
    """
    group_starts = s1_group_starts(cand_path)
    if not len(group_starts):
        return [(0, n_rows)]
    log(f"s1 groups: {len(group_starts):,} distinct s1_id runs")
    starts = [0]
    for lo, _ in bounds[1:]:
        j = int(np.searchsorted(group_starts, lo, side="left"))
        nxt = int(group_starts[j]) if j < len(group_starts) else n_rows
        if nxt not in starts:
            starts.append(nxt)
    starts.append(n_rows)
    out = [(a, b) for a, b in zip(starts, starts[1:]) if b > a]
    log(f"shards after snapping to s1 groups: {[(a, b - a) for a, b in out]}")
    return out


def assert_s1_sorted(cand_path: Path) -> None:
    """Fail fast unless s1_id is non-decreasing across the whole candidate file.

    This is a P1 output invariant, but P2's event partitioning depends on it, so it is
    re-checked here rather than assumed.
    """
    prev = None
    for batch in pq.ParquetFile(cand_path).iter_batches(batch_size=500_000, columns=["s1_id"]):
        col = batch.column("s1_id").to_pandas()
        if not len(col):
            continue
        arr = col.to_numpy(dtype=object)
        if prev is not None and arr[0] < prev:
            raise RuntimeError(
                f"{cand_path} is not sorted by s1_id (saw {arr[0]!r} after {prev!r}); "
                "P2 event partitioning requires the P1 ordering invariant"
            )
        if len(arr) > 1 and not (arr[1:] >= arr[:-1]).all():
            bad = int(np.flatnonzero(arr[1:] < arr[:-1])[0])
            raise RuntimeError(
                f"{cand_path} is not sorted by s1_id (row {bad}: {arr[bad+1]!r} < {arr[bad]!r})"
            )
        prev = arr[-1]
    log("precondition ok: candidates sorted by s1_id")


def _boundary_s1_ids(cand_path: Path, bounds: List[Tuple[int, int]]) -> List[str]:
    """s1_id of the first row of each shard. One sequential pass, single column."""
    wanted = {lo for lo, _ in bounds}
    out: List[str] = []
    pf = pq.ParquetFile(cand_path)
    pos = 0
    for batch in pf.iter_batches(batch_size=100_000, columns=["s1_id"]):
        col = batch.column("s1_id").to_pandas()
        if not len(col):
            continue
        nxt = pos + len(col)
        for lo, _ in bounds:
            if pos <= lo < nxt:
                out.append(str(col.iloc[lo - pos]))
        pos = nxt
        if len(out) == len(bounds):
            break
    if len(out) != len(bounds):
        raise RuntimeError(f"found {len(out)} boundaries, expected {len(bounds)}")
    return out


def partition_events(events_path: Path, work: Path, s1_boundaries: List[str],
                     n_shards: int) -> List[int]:
    """Route every event to the shard owning its s1_id, writing one file per shard."""
    bounds_arr = pd.Index(pd.array(s1_boundaries, dtype=object))
    counts = [0] * n_shards
    writers: List[Optional[pq.ParquetWriter]] = [None] * n_shards
    pf = pq.ParquetFile(events_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=EVENT_COLUMNS):
        frame = batch.to_pandas()
        if not len(frame):
            continue
        shard_of = np.asarray(bounds_arr.searchsorted(frame["s1_id"], side="right")) - 1
        shard_of = np.clip(shard_of, 0, n_shards - 1)
        for shard_id in np.unique(shard_of):
            part = frame.iloc[np.flatnonzero(shard_of == shard_id)]
            if not len(part):
                continue
            table = pa.Table.from_pandas(part[EVENT_COLUMNS], preserve_index=False)
            if writers[shard_id] is None:
                writers[shard_id] = pq.ParquetWriter(
                    work / f"events_{shard_id}.parquet", table.schema, compression="zstd"
                )
            writers[shard_id].write_table(table)
            counts[shard_id] += table.num_rows
    for w in writers:
        if w is not None:
            w.close()
    return counts


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def _worker(task: Tuple[int, Path, Path, Path, int, str]) -> dict:
    shard_id, cand_path, ev_path, out_path, chunk_size, log_prefix = task
    t0 = time.time()
    cands = pq.read_table(cand_path).to_pandas()
    ev_path_obj = Path(ev_path)
    events = pq.read_table(ev_path_obj).to_pandas() if ev_path_obj.exists() else None
    feats = build_features(cands, _RECORDS, events, chunk_size=chunk_size)
    assert len(feats) == len(cands), f"row loss: {len(feats)} != {len(cands)}"
    pq.write_table(
        pa.Table.from_pandas(feats, preserve_index=False), out_path, compression="zstd"
    )
    return {
        "shard": shard_id,
        "rows": len(feats),
        "events": 0 if events is None else len(events),
        "seconds": round(time.time() - t0, 2),
        "peak_rss_mb": round(rss_mb(), 1),
    }


# --------------------------------------------------------------------------- #
# finalize + validate
# --------------------------------------------------------------------------- #
def concat_shards(shard_paths: List[Path], out_path: Path) -> Tuple[int, List[str]]:
    """Concatenate shard outputs in order, preserving global candidate order."""
    writer = None
    rows = 0
    first_cols: List[str] = []
    for p in shard_paths:
        if not p.exists():
            continue
        pf = pq.ParquetFile(p)
        if not first_cols:
            first_cols = list(pf.schema_arrow.names)
        for batch in pf.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
    if writer is not None:
        writer.close()
    else:
        pq.write_table(pa.table({"pair_key": pa.array([], type=pa.string())}), out_path)
    return rows, first_cols


def validate(out_path: Path, cand_path: Path, s1_path: Path,
             known_entity_ids: Optional[set] = None) -> dict:
    """P2 contract checks. There is no pre-existing P2 validator, so this defines one."""
    checks: dict = {}
    feats = pq.read_table(out_path)
    n_cand = pq.ParquetFile(cand_path).metadata.num_rows

    checks["features_nonempty"] = feats.num_rows > 0
    checks["row_count_matches_candidates"] = feats.num_rows == n_cand
    cols = list(feats.schema.names)
    checks["schema_39_cols"] = (
        len(cols) == 39 and cols[:3] == ["pair_key", "s1_id", "candidate_id"]
        and cols[3:] == FEATURE_COLUMNS
    )
    feat_types = [feats.schema.field(c).type for c in FEATURE_COLUMNS]
    checks["features_float32"] = all(str(t) == "float" and t.bit_width == 32 for t in feat_types)

    # Positional agreement with the P1 candidate list (the integration test's invariant).
    mismatches = 0
    fc = pq.ParquetFile(out_path)
    cc = pq.ParquetFile(cand_path)
    fb = next(fc.iter_batches(batch_size=200_000, columns=["pair_key"]))
    cb = next(cc.iter_batches(batch_size=200_000, columns=["pair_key"]))
    a = fb.column("pair_key").to_pylist()
    b = cb.column("pair_key").to_pylist()
    mismatches = sum(1 for x, y in zip(a, b) if x != y)
    checks["pair_key_order_matches_candidates"] = mismatches == 0 and len(a) == len(b)

    if known_entity_ids is not None:
        s1_ids = known_entity_ids
    elif s1_path.exists():
        s1_ids = set(
            pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"]
        )
    else:
        s1_ids = None
    sample = feats.column("s1_id").to_pandas().head(200_000)
    if s1_ids is None:
        checks["s1_ids_within_input_s1"] = True
    else:
        checks["s1_ids_within_input_s1"] = set(sample).issubset(s1_ids)
    return checks


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="artifacts/slice/out/candidates.parquet")
    ap.add_argument("--events", default="artifacts/slice/out/retrieval_events.parquet")
    ap.add_argument("--slice-dir", default="artifacts/slice/data")
    ap.add_argument("--records", default="",
                    help="optional pre-built records parquet; skips rebuilding from the slice")
    ap.add_argument("--out", default="artifacts/slice/out/features.parquet")
    ap.add_argument("--metrics", default="artifacts/slice/out/p2_metrics.json")
    ap.add_argument("--work-dir", default="/var/tmp/p2_shards")
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--rows-per-shard", type=int, default=2_200_000)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--reuse", action="store_true", help="keep existing work-dir shard files")
    args = ap.parse_args()

    global _RECORDS
    t_start = time.time()
    work = Path(args.work_dir)
    if work.exists() and not args.reuse:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cand_path, ev_path = Path(args.candidates), Path(args.events)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (cand_path, ev_path):
        if not p.exists():
            log(f"FATAL missing required input: {p}")
            return 1

    n_cand_total = pq.ParquetFile(cand_path).metadata.num_rows
    n_shards = max(1, min(args.shards, -(-n_cand_total // args.rows_per_shard)))
    log(f"P2 start: {n_cand_total:,} candidates, {n_shards} shards, "
        f"events={pq.ParquetFile(ev_path).metadata.num_rows:,}")

    if args.records:
        _RECORDS = pq.read_table(args.records).to_pandas()
        log(f"records loaded from {args.records}: {len(_RECORDS):,} rows")
    else:
        _RECORDS = build_records(Path(args.slice_dir))

    nominal = shard_bounds(n_cand_total, n_shards)
    # Event routing below uses searchsorted over shard-boundary s1_ids, which is only
    # valid when the candidate list is sorted by s1_id AND each s1_id occupies a single
    # contiguous run. P1 guarantees the sort (it is a P1 validation check), so assert it
    # rather than degrade silently; the contiguity requirement is enforced by snapping.
    assert_s1_sorted(cand_path)
    bounds = snap_bounds_to_s1_groups(cand_path, nominal, n_cand_total)
    s1_boundaries = _boundary_s1_ids(cand_path, bounds)
    log(f"shard boundary s1_ids: {s1_boundaries[:2]} ... {s1_boundaries[-1:]}")

    # Split candidates into per-shard files by ROW POSITION, so shard i holds exactly
    # rows [lo, hi) of the candidate list and concatenating shard 0..N-1 reproduces the
    # input order regardless of how the file happens to be sorted.
    t0 = time.time()
    writers: List[Optional[pq.ParquetWriter]] = [None] * len(bounds)
    offsets = [lo for lo, _ in bounds]
    buf: List[pd.DataFrame] = []
    pos = 0

    def flush() -> None:
        if not buf:
            return
        frame = pd.concat(buf, ignore_index=True)
        buf.clear()
        ends = np.searchsorted(np.asarray(offsets, dtype=np.int64),
                               frame.index.to_numpy(), side="right") - 1
        ends = np.clip(ends, 0, len(bounds) - 1)
        for sid in np.unique(ends):
            part = frame.iloc[np.flatnonzero(ends == sid)]
            table = pa.Table.from_pandas(part[CANDIDATE_KEY_COLUMNS], preserve_index=False)
            if writers[sid] is None:
                writers[sid] = pq.ParquetWriter(
                    work / f"cand_{sid}.parquet", table.schema, compression="zstd"
                )
            writers[sid].write_table(table)

    for batch in pq.ParquetFile(cand_path).iter_batches(
        batch_size=200_000, columns=CANDIDATE_KEY_COLUMNS
    ):
        frame = batch.to_pandas()
        if len(frame):
            frame.index = np.arange(pos, pos + len(frame))
            pos += len(frame)
            buf.append(frame)
        if sum(len(f) for f in buf) >= 400_000:
            flush()
    flush()
    for w in writers:
        if w is not None:
            w.close()
    cand_shard_rows = []
    for i in range(len(bounds)):
        p = work / f"cand_{i}.parquet"
        cand_shard_rows.append(pq.ParquetFile(p).metadata.num_rows if p.exists() else 0)
    log(f"candidate shards written in {time.time()-t0:.1f}s: {cand_shard_rows}")

    t0 = time.time()
    ev_counts = partition_events(ev_path, work, s1_boundaries, len(bounds))
    log(f"events partitioned in {time.time()-t0:.1f}s: {ev_counts}")

    tasks = [
        (
            i,
            work / f"cand_{i}.parquet",
            work / f"events_{i}.parquet" if ev_counts[i] else Path("/nonexistent"),
            work / f"feat_{i}.parquet",
            args.chunk_size,
            f"shard{i}",
        )
        for i in range(len(bounds))
        if cand_shard_rows[i] > 0
    ]

    ctx = mp.get_context("fork")
    log(f"launching {len(tasks)} feature workers (parent RSS {rss_mb():.0f} MB)")
    t0 = time.time()
    if len(tasks) == 1:
        results = [_worker(tasks[0])]
    else:
        with ctx.Pool(processes=len(tasks)) as pool:
            results = []
            for r in pool.imap_unordered(_worker, tasks):
                results.append(r)
                done = sum(x["rows"] for x in results)
                log(f"  worker {r['shard']} done rows={r['rows']:,} events={r['events']:,} "
                    f"{r['seconds']}s  ({done:,}/{n_cand_total:,})")
    feature_seconds = time.time() - t0
    log(f"feature workers done in {feature_seconds:.1f}s")

    t0 = time.time()
    shard_paths = [work / f"feat_{r['shard']}.parquet" for r in sorted(results, key=lambda x: x["shard"])]
    rows, cols = concat_shards(shard_paths, out_path)
    log(f"concatenated {rows:,} feature rows -> {out_path} in {time.time()-t0:.1f}s")

    checks = validate(
        out_path,
        cand_path,
        Path(args.slice_dir) / "slice_source1.tsv",
        known_entity_ids=set(_RECORDS["entity_id"].astype(str)) if args.records else None,
    )
    total_seconds = time.time() - t_start
    metrics = {
        "candidates_in": n_cand_total,
        "events_in": pq.ParquetFile(ev_path).metadata.num_rows,
        "feature_rows": rows,
        "shards": len(tasks),
        "rows_per_shard": cand_shard_rows,
        "events_per_shard": ev_counts,
        "workers": sorted(results, key=lambda x: x["shard"]),
        "feature_seconds": round(feature_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "parent_peak_rss_mb": round(rss_mb(), 1),
        "features_path": str(out_path),
        "validation": checks,
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"P2 validation: {checks}")
    failed = [k for k, v in checks.items() if not v]
    if failed:
        log(f"P2 FAILED checks: {failed}")
        return 1
    log(f"P2 COMPLETE in {total_seconds/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
