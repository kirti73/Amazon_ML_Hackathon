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
import os
import shutil
import sys
import time
from pathlib import Path
from subprocess import Popen, STDOUT
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
# Set per-process by main(); lets the worker helpers find the slice without a global.
SLICE_DIR: List[str] = [""]
REPO_ROOT = Path(__file__).resolve().parents[1]
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


def peak_rss_mb() -> float:
    """True high-water RSS for this process, in MB.

    ``rss_mb()`` samples only the current value, so a worker that peaked at 4 GB and then
    freed memory reported ~2.7 GB. ru_maxrss is the kernel's own high-water mark, which is
    the number that decides how many workers fit.
    """
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_available_mb() -> float:
    """MemAvailable from /proc/meminfo, in MB.

    Summing every process's RSS badly overstates usage, because shared pages (page cache,
    libc, the interpreter) are counted once per process. MemAvailable is the honest
    number for deciding whether N concurrent workers fit in RAM.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


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
def _load_records(records_path: Optional[Path]) -> pd.DataFrame:
    """Records frame from a parquet file if given, else rebuild it from the slice."""
    if records_path is not None and Path(records_path).exists():
        return pq.read_table(records_path).to_pandas()
    return build_records(Path(SLICE_DIR[0]))


def _worker(task: Tuple[int, Path, Path, Path, int, str, Optional[Path], Path]) -> dict:
    """Build features for one shard.

    Observability is the point of this rewrite. A worker now:
      * appends its own stdout/stderr to ``worker_<id>.log``;
      * records a ``done_<id>.json`` marker that says explicitly whether it succeeded,
        with row count / schema / peak RSS / elapsed time, so a partial write can never be
        mistaken for a finished shard on a later resume;
      * writes the parquet to a temp name and ``os.replace``s it into position, so the
        output file either does not exist or is complete;
      * re-opens and re-reads the file it just wrote before declaring success.
    Any exception is logged in full and re-raised, so the parent sees a real traceback
    instead of a silent disappearance.
    """
    shard_id, cand_path, ev_path, out_path, chunk_size, _prefix, records_path, log_dir = task
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = log_dir / f"worker_{shard_id}.log"
    marker = log_dir / f"done_{shard_id}.json"
    t0 = time.time()
    result = {
        "shard": shard_id, "rows": 0, "events": 0, "seconds": 0.0,
        "peak_rss_mb": 0.0, "ok": False, "log": str(logf),
    }
    try:
        records = _load_records(records_path)
        result["records_rows"] = int(len(records))
        cands = pq.read_table(cand_path).to_pandas()
        n_cand = len(cands)
        ev_path_obj = Path(ev_path)
        events = pq.read_table(ev_path_obj).to_pandas() if ev_path_obj.exists() else None
        result["events"] = 0 if events is None else int(len(events))
        result["candidates"] = int(n_cand)
        log_line(f"[worker {shard_id}] start cand={n_cand:,} events={result['events']:,} "
                 f"records={len(records):,} rss={rss_mb():.0f}MB")
        feats = build_features(cands, records, events, chunk_size=chunk_size)
        if len(feats) != n_cand:
            raise RuntimeError(f"row loss: {len(feats)} != {n_cand}")
        cols = list(feats.columns)
        if cols[:3] != ["pair_key", "s1_id", "candidate_id"] or cols[3:] != FEATURE_COLUMNS:
            raise RuntimeError(f"shard {shard_id} produced a bad schema: {cols[:5]}...")
        # Atomic: a killed worker must not leave a half-written file that a later resume
        # would happily accept.
        tmp = Path(str(out_path) + ".tmp")
        pq.write_table(
            pa.Table.from_pandas(feats, preserve_index=False), tmp, compression="zstd"
        )
        del feats, cands, events, records
        os.replace(tmp, out_path)
        # Verify what actually landed on disk rather than trusting the write.
        pf = pq.ParquetFile(out_path)
        disk_rows = pf.metadata.num_rows
        disk_cols = list(pf.schema_arrow.names)
        if disk_rows != n_cand or disk_cols != ["pair_key", "s1_id", "candidate_id"] + FEATURE_COLUMNS:
            raise RuntimeError(
                f"shard {shard_id} on-disk verification failed: rows={disk_rows} "
                f"cols={len(disk_cols)}"
            )
        result.update(
            rows=disk_rows,
            seconds=round(time.time() - t0, 2),
            peak_rss_mb=round(peak_rss_mb(), 1),
            ok=True,
            output_bytes=Path(out_path).stat().st_size,
        )
        log_line(f"[worker {shard_id}] OK rows={disk_rows:,} "
                 f"{result['seconds']}s peak_rss={result['peak_rss_mb']}MB")
    except BaseException as exc:  # noqa: BLE001 - we re-raise after recording
        import traceback
        tb = traceback.format_exc()
        result.update(
            ok=False, error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.time() - t0, 2), peak_rss_mb=round(peak_rss_mb(), 1),
        )
        log_line(f"[worker {shard_id}] FAILED {type(exc).__name__}: {exc}\n{tb}")
        marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def log_line(message: str) -> None:
    """Log to stdout *and* to this worker's own file, so output survives parent loss."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)
    try:
        cur = _WORKER_LOG.get()
        if cur is not None:
            with open(cur, "a", encoding="utf-8") as fh:
                fh.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


_WORKER_LOG: Optional[str] = None

POLL_SECONDS = 2.0


def shard_is_complete(shard_id: int, work: Path) -> Optional[dict]:
    """Return the marker for a shard only if its output is really present and sound.

    A marker alone is not enough: the marker and the parquet are written in that order, so
    a crash in between would otherwise leave a marker describing a file that is not there.
    """
    marker = work / f"done_{shard_id}.json"
    out = work / f"feat_{shard_id}.parquet"
    if not marker.exists() or not out.exists():
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not info.get("ok"):
        return None
    try:
        pf = pq.ParquetFile(out)
        if pf.metadata.num_rows != info.get("rows"):
            return None
        cols = list(pf.schema_arrow.names)
        if cols != ["pair_key", "s1_id", "candidate_id"] + FEATURE_COLUMNS:
            return None
    except Exception:
        return None
    return info


def launch_worker_subprocess(shard_id: int, work: Path, chunk_size: int,
                             records_path: Optional[Path]) -> Tuple[Popen, Path]:
    """Run one shard in its own process so its exit code and stderr are observable.

    Using real subprocesses rather than a multiprocessing.Pool is deliberate. A Pool
    reports a worker exception by re-raising in the parent, but it cannot tell the
    difference between "worker raised" and "worker was killed", and a parent that dies
    takes the whole Pool with it. A subprocess gives a genuine exit status (negative ==
    killed by a signal) and a per-worker log file that survives the parent.
    """
    logf = work / f"worker_{shard_id}.log"
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--run-worker", str(shard_id),
        "--work-dir", str(work),
        "--chunk-size", str(chunk_size),
    ]
    if records_path is not None:
        cmd += ["--records", str(records_path)]
    if SLICE_DIR[0]:
        cmd += ["--slice-dir", str(SLICE_DIR[0])]
    handle = open(logf, "a", encoding="utf-8")
    return Popen(cmd, stdout=handle, stderr=STDOUT, cwd=str(REPO_ROOT)), logf


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


def parquet_rows(path: Path) -> int:
    """Row count of a parquet file, or 0 if it does not exist."""
    path = Path(path)
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else 0


def validate(out_path: Path, cand_path: Path, s1_path: Path,
             known_entity_ids: Optional[set] = None,
             expected_rows: Optional[int] = None) -> dict:
    """P2 contract checks. There is no pre-existing P2 validator, so this defines one."""
    checks: dict = {}
    feats = pq.read_table(out_path)
    n_cand = pq.ParquetFile(cand_path).metadata.num_rows

    checks["features_nonempty"] = feats.num_rows > 0
    # expected_rows lets a subset run validate against the shards it actually built.
    # Comparing a 1-shard diagnostic against the full candidate file always fails, which
    # is noise, not signal. The full-coverage requirement is a separate check.
    checks["row_count_matches_candidates"] = (
        feats.num_rows == (n_cand if expected_rows is None else expected_rows))
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

def run_features_stage(args, work: Path, cand_path: Path, out_path: Path,
                       records_file: Path, wanted: Optional[List[int]] = None,
                       n_shards: Optional[int] = None, concat: bool = True) -> int:
    """Build features for the selected shards, then concatenate and validate.

    Safe to re-run: a shard whose ``done_<id>.json`` marker and parquet both check out is
    skipped, so a failure at shard 4 does not throw away shards 0-3.
    """
    global _WORKER_LOG
    work = Path(work)
    if n_shards is None:
        n_shards = len(sorted(int(p.stem.split("_")[1]) for p in work.glob("cand_*.parquet")))
    if wanted is None:
        wanted = list(range(n_shards))

    pending: List[int] = []
    results: Dict[int, dict] = {}
    for sid in wanted:
        done = shard_is_complete(sid, work)
        if done is not None:
            log(f"shard {sid} already complete ({done['rows']:,} rows, "
                f"{done.get('seconds')}s) - skipping")
            results[sid] = done
        else:
            pending.append(sid)

    max_workers = args.workers if args.workers and args.workers > 0 else max(1, len(pending))
    log(f"{len(pending)} shard(s) to build, {len(results)} already done, "
        f"concurrency={min(max_workers, max(1, len(pending)))}")

    t0 = time.time()
    running: Dict[int, Any] = {}
    queue = list(pending)
    min_available_mb = mem_available_mb()
    start_available_mb = min_available_mb
    while queue or running:
        while queue and len(running) < max_workers:
            sid = queue.pop(0)
            proc, logf = launch_worker_subprocess(sid, work, args.chunk_size, records_file)
            running[sid] = (proc, logf)
            log(f"  launched shard {sid} (pid {proc.pid}) -> {logf}")
        time.sleep(POLL_SECONDS)
        # The question is whether N concurrent workers fit in RAM, so track the low-water
        # mark of MemAvailable: that is the headroom we actually had to give away.
        avail = mem_available_mb()
        if avail == avail:  # not NaN
            min_available_mb = min(min_available_mb, avail)
        for sid in list(running):
            proc, logf = running[sid]
            rc = proc.poll()
            if rc is None:
                continue
            del running[sid]
            info = shard_is_complete(sid, work)
            if rc == 0 and info is not None:
                log(f"  shard {sid} OK rows={info['rows']:,} {info['seconds']}s "
                    f"peak_rss={info['peak_rss_mb']}MB "
                    f"(MemAvailable now {mem_available_mb():.0f}MB, low {min_available_mb:.0f}MB)")
                results[sid] = info
            else:
                tail = ""
                try:
                    tail = logf.read_text(encoding="utf-8")[-2000:]
                except Exception:
                    pass
                verdict = ("killed by signal %d" % -rc) if rc < 0 else f"exit {rc}"
                log(f"  shard {sid} FAILED ({verdict}); marker says "
                    f"{(info or {}).get('error', 'no marker')}")
                log(f"  ---- tail of {logf} ----\n{tail}\n  ---- end ----")
                results[sid] = {"shard": sid, "ok": False, "rows": 0,
                                "error": f"worker {verdict}"}
    log(f"feature workers finished in {time.time()-t0:.1f}s; "
        f"MemAvailable low-water {min_available_mb:.0f}MB (was {start_available_mb:.0f}MB "
        f"at start, {start_available_mb - min_available_mb:.0f}MB consumed)")

    failed = [sid for sid in results if not results[sid].get("ok")]
    if failed:
        log(f"INCOMPLETE shards: {sorted(failed)} - features.parquet NOT written")
        return 1
    if not concat:
        return 0

    ordered = [results[sid] for sid in sorted(results)]
    total_rows = sum(r["rows"] for r in ordered)
    log(f"all {len(ordered)} shard(s) complete, {total_rows:,} rows total")
    t1 = time.time()
    shard_paths = [work / f"feat_{r['shard']}.parquet" for r in ordered]
    rows, _cols = concat_shards(shard_paths, out_path)
    log(f"concatenated {rows:,} feature rows -> {out_path} in {time.time()-t1:.1f}s")
    if rows != total_rows:
        log(f"FATAL: concatenated {rows:,} != sum of shards {total_rows:,}")
        return 1

    # Derive the S1 allow-list from the records file when we have one. Reading the real
    # slice TSV instead would be wrong for any other input and needlessly slow: the
    # records frame is the authority on which entity ids exist in this run.
    known = None
    if Path(records_file).exists():
        known = set(
            pq.read_table(records_file, columns=["entity_id"])
            .to_pandas()["entity_id"].astype(str)
        )
    selected_cand_rows = sum(
        parquet_rows(work / f"cand_{r['shard']}.parquet") for r in ordered)
    all_shards = len(ordered) == n_shards
    n_cand_total = pq.ParquetFile(cand_path).metadata.num_rows
    checks = validate(
        out_path,
        cand_path,
        Path(SLICE_DIR[0] or "artifacts/slice/data") / "slice_source1.tsv",
        known_entity_ids=known,
        expected_rows=selected_cand_rows,
    )
    checks["all_shards_covered"] = all_shards
    checks["covers_full_candidate_set"] = (rows == n_cand_total) if all_shards else None
    log(f"P2 validation: {json.dumps(checks, indent=2)}")
    # Coverage is informational for a deliberately partial run: a one-shard diagnostic is
    # supposed to fail "covers_full_candidate_set", and treating that as a defect buries the
    # real signal. Only an explicit False is a failure; None means "not applicable".
    informational = {"all_shards_covered", "covers_full_candidate_set"}
    bad = [
        k for k, v in checks.items()
        if v is False and not (not all_shards and k in informational)
    ]
    # Keep the established metrics keys (rows_per_shard / events_per_shard / *_seconds)
    # so anything reading this file downstream keeps working; the new fields are additive.
    metrics = {
        "candidates_in": pq.ParquetFile(cand_path).metadata.num_rows,
        "events_in": sum(parquet_rows(work / f"events_{i}.parquet") for i in range(n_shards)),
        "feature_rows": rows,
        "shards": len(ordered),
        "concurrency": max_workers,
        "rows_per_shard": [parquet_rows(work / f"cand_{i}.parquet") for i in range(n_shards)],
        "events_per_shard": [parquet_rows(work / f"events_{i}.parquet") for i in range(n_shards)],
        "workers": ordered,
        "feature_seconds": round(time.time() - t0, 2),
        "parent_peak_rss_mb": round(peak_rss_mb(), 1),
        "min_mem_available_mb": round(min_available_mb, 1),
        "mem_available_at_start_mb": round(start_available_mb, 1),
        "features_path": str(out_path),
        "validation": checks,
    }
    try:
        Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"could not write metrics: {exc}")
    if bad:
        log(f"P2 FAILED checks: {bad}")
        return 1
    log(f"P2 COMPLETE: {rows:,} feature rows, 39 cols, validation green")
    return 0


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
    ap.add_argument(
        "--stage", choices=["all", "prepare", "features"], default="all",
        help="prepare: only write cand_*/events_* shards. features: only build features "
             "from existing shards (never touches them). all: both, as before.",
    )
    ap.add_argument(
        "--workers", type=int, default=0,
        help="max concurrent feature workers; 0 = one per shard. Separate from --shards "
             "on purpose, so concurrency can be tuned from measured memory.",
    )
    ap.add_argument(
        "--run-worker", type=int, default=None,
        help="internal: build exactly this one shard in this process, then exit. Must be "
             "checked before any orchestration, otherwise the child re-enters the stage "
             "and spawns children of its own (fork bomb).",
    )
    ap.add_argument(
        "--only-shards", default="",
        help="comma-separated shard ids to process, e.g. '0' for a single-shard diagnostic",
    )
    ap.add_argument(
        "--candidate-batch-rows", type=int, default=200_000,
        help="rows per candidate read batch (tests lower this to force several flushes)",
    )
    ap.add_argument(
        "--candidate-flush-rows", type=int, default=400_000,
        help="rows buffered before writing candidate shards (tests lower this to force "
             "multiple flushes, which is where shard routing was previously wrong)",
    )
    args = ap.parse_args()

    global _RECORDS
    t_start = time.time()
    global SLICE_DIR
    SLICE_DIR = [args.slice_dir]
    work = Path(args.work_dir)
    if work.exists() and not args.reuse and args.stage != "features" and args.run_worker is None:
        # Never rmtree when we are only consuming shards: that would destroy the very
        # inputs the run exists to use, and they are expensive to rebuild. A worker
        # process must never delete the directory it was launched into.
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cand_path, ev_path = Path(args.candidates), Path(args.events)
    out_path = Path(args.out)
    records_file = work / "records.parquet"

    if args.run_worker is not None:
        # A worker must never orchestrate. Everything above is bookkeeping only; the first
        # real work is this branch, so a child cannot recurse into a second generation.
        sid = args.run_worker
        rec = records_file if Path(records_file).exists() else (
            Path(args.records) if args.records and Path(args.records).exists() else None)
        task = (sid,
                work / f"cand_{sid}.parquet",
                work / f"events_{sid}.parquet",
                work / f"feat_{sid}.parquet",
                args.chunk_size, "", rec, work)
        try:
            info = _worker(task)
        except BaseException as exc:  # already logged with traceback by _worker
            log(f"[worker {sid}] aborting: {type(exc).__name__}: {exc}")
            return 1
        log(f"[worker {sid}] marker written ok={info['ok']}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (cand_path, ev_path):
        if not p.exists():
            log(f"FATAL missing required input: {p}")
            return 1

    n_cand_total = pq.ParquetFile(cand_path).metadata.num_rows

    if args.stage == "features":
        wanted = [int(x) for x in args.only_shards.split(",") if x.strip() != ""] or None
        return run_features_stage(args, work, cand_path, out_path, records_file,
                                  wanted=wanted)

    n_shards = max(1, min(args.shards, -(-n_cand_total // args.rows_per_shard)))
    log(f"P2 start: {n_cand_total:,} candidates, {n_shards} shards, "
        f"events={pq.ParquetFile(ev_path).metadata.num_rows:,}")

    if args.records:
        _RECORDS = pq.read_table(args.records).to_pandas()
        log(f"records loaded from {args.records}: {len(_RECORDS):,} rows")
    else:
        _RECORDS = build_records(Path(args.slice_dir))
    if not records_file.exists():
        # Children read this instead of each re-normalising 1.7M text records.
        pq.write_table(pa.Table.from_pandas(_RECORDS, preserve_index=False),
                       records_file, compression="zstd")
        log(f"records cached for workers -> {records_file}")

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
        # Must NOT pass ignore_index=True: each frame's index was deliberately set to its
        # absolute row position above, and that index is what routes rows to shards via
        # searchsorted. Resetting it would re-map every flush to shard 0, which is
        # invisible whenever the whole input fits in one flush and silently puts the
        # entire candidate set in shard 0 as soon as it does not.
        frame = pd.concat(buf)
        buf.clear()
        if not np.array_equal(frame.index.to_numpy(), np.arange(pos - len(frame), pos)):
            raise RuntimeError(
                "candidate buffer lost its absolute row index; shard routing would be wrong"
            )
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
        batch_size=args.candidate_batch_rows, columns=CANDIDATE_KEY_COLUMNS
    ):
        frame = batch.to_pandas()
        if len(frame):
            frame.index = np.arange(pos, pos + len(frame))
            pos += len(frame)
            buf.append(frame)
        if sum(len(f) for f in buf) >= args.candidate_flush_rows:
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

    cand_shard_rows_all = cand_shard_rows
    ev_counts_all = ev_counts
    bounds_n = len(bounds)
    if args.stage == "prepare":
        log("P2 PREPARE COMPLETE (candidate + event shards written; no features built)")
        return 0

    if args.only_shards:
        wanted = [int(x) for x in args.only_shards.split(",") if x.strip() != ""]
    else:
        wanted = list(range(bounds_n))
    log(f"feature stage: shards {wanted} of {bounds_n}")

    # run_features_stage owns the whole feature phase: it builds the selected shards,
    # concatenates them in order, validates the result, writes metrics, and returns a
    # process exit code. Nothing concatenates or validates a second time.
    t0 = time.time()
    rc = run_features_stage(
        args, work, cand_path, out_path, records_file,
        wanted=wanted, n_shards=bounds_n, concat=True,
    )
    log(f"feature stage finished rc={rc} in {time.time()-t0:.1f}s "
        f"(total {time.time()-t_start:.1f}s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
