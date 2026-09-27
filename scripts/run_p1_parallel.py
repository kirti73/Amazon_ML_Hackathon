#!/usr/bin/env python
"""Parallel P1 route workers + bounded-memory finalization for the 300K slice.

Why this exists
---------------
``scripts/run_p1_slice.py`` runs the seven routes sequentially in one process and then
materializes the final candidate set through ``iter_canonical_candidates_merged``, which
holds one fully-materialized pandas frame per hash bucket at the same time (all 64
generators are advanced by ``heapq.merge``). At slice scale that is the whole candidate
set resident at once, which is the one remaining unbounded-memory path.

This driver splits the work three ways, none of which touches retrieval semantics:

``--mode worker``
    Runs a single route, optionally restricted to a contiguous slice of S1, and spills
    *events only* to its own directory. For the two vectorizer routes the
    ``CharTfidfVectorizer`` is fitted once per worker on the **full** ``pool + S1`` corpus
    -- byte-identical to what a single full run fits -- and then passed in through the
    routes' existing ``vectorizer=`` parameter. Because the vectorizer is fixed and every
    query row is transformed independently, the per-row result is identical to the
    un-sharded run; ``--mode selftest`` proves that against the real fixtures.

``--mode finalize``
    Unions every worker's spill, then produces the artifacts in two bounded passes:
    canonicalize one bucket at a time to a temp file, then ``heapq.merge`` the sorted
    buckets lazily. Peak memory is one bucket plus one batch per bucket, not the whole
    candidate set.

``--mode selftest``
    Proves the pre-fitted-vectorizer path reproduces the route's own internal fit exactly.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import shutil
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

from src.blocking.candidate_generation import DEFAULT_BLOCKING_CONFIG  # noqa: E402
from src.blocking.spill import (  # noqa: E402
    CANONICAL_CANDIDATE_COLUMNS,
    EventSpillWriter,
    RecordBuffer,
    canonicalize_frame,
)

ID_COL = "entity_id"
NAME_COL = "business_name"
ADDR_COL = "business_address"

#: Frozen route parameters. These mirror ``DEFAULT_BLOCKING_CONFIG`` and the route
#: signatures; they are passed through unchanged.
ROUTE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "exact_name": dict(max_block_size=DEFAULT_BLOCKING_CONFIG["exact_name_max_block"]),
    "tfidf_name": dict(
        top_k=DEFAULT_BLOCKING_CONFIG["tfidf_name_top_k"],
        chunk_size=DEFAULT_BLOCKING_CONFIG["tfidf_name_chunk_size"],
        query_block=DEFAULT_BLOCKING_CONFIG["tfidf_query_block"],
        candidate_block=DEFAULT_BLOCKING_CONFIG["tfidf_name_candidate_chunk"],
    ),
    "rare_token_name": dict(
        top_k=DEFAULT_BLOCKING_CONFIG["rare_name_top_k"],
        max_block_size=DEFAULT_BLOCKING_CONFIG["rare_name_max_block"],
    ),
    "tfidf_address": dict(
        top_k=DEFAULT_BLOCKING_CONFIG["tfidf_addr_top_k"],
        chunk_size=DEFAULT_BLOCKING_CONFIG["tfidf_addr_chunk_size"],
        query_block=DEFAULT_BLOCKING_CONFIG["tfidf_query_block"],
        candidate_block=DEFAULT_BLOCKING_CONFIG["tfidf_addr_candidate_chunk"],
    ),
    "numeric_address": dict(
        top_k=DEFAULT_BLOCKING_CONFIG["numeric_top_k"],
        max_block_size=DEFAULT_BLOCKING_CONFIG["numeric_max_block"],
    ),
    "rare_token_address": dict(
        top_k=DEFAULT_BLOCKING_CONFIG["rare_addr_top_k"],
        max_block_size=DEFAULT_BLOCKING_CONFIG["rare_addr_max_block"],
    ),
    "reverse_retrieval": dict(
        top_k_reverse=DEFAULT_BLOCKING_CONFIG["reverse_top_k"],
        chunk_size=DEFAULT_BLOCKING_CONFIG["reverse_chunk_size"],
        query_block=DEFAULT_BLOCKING_CONFIG["tfidf_query_block"],
        index_block=DEFAULT_BLOCKING_CONFIG["reverse_index_block"],
    ),
}

#: Routes whose vocabulary/IDF is fitted over ``pool + S1``. They accept a pre-fitted
#: vectorizer, which is what makes S1 sharding admissible for them.
VECTORIZER_ROUTES = {"tfidf_name", "tfidf_address"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def peak_rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_slice(slice_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("slice_source1.tsv", "slice_source2.tsv", "slice_source3.tsv"):
        frames.append(
            pd.read_csv(
                slice_dir / name, sep="\t", dtype=str, keep_default_na=False, na_values=[]
            )
        )
    return frames[0], frames[1], frames[2]


def _iter_pool_addrs(s2: pd.DataFrame, s3: pd.DataFrame) -> Iterator[str]:
    for frame in (s2, s3):
        for raw in frame[ADDR_COL].fillna("").astype(str).tolist():
            if raw:
                yield raw


def fit_shared_vectorizer(route: str, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame):
    """Fit exactly what an un-sharded run of ``route`` would fit.

    Mirrors the ``if vectorizer is None:`` branch of the route verbatim, including the
    ``effective_min_df`` rule that depends on the *full* S1 size.
    """
    from src.blocking import tfidf_address as ta
    from src.blocking import tfidf_name as tn
    from src.blocking.tfidf_common import iter_pool_names

    n_candidates = len(s2) + len(s3)
    n_queries = len(s1)

    # S1 texts must come from the route's own _build_s1_query_data, which applies the
    # route's normalizer. Using raw strings here silently changes the n-gram vocabulary
    # -- the selftest caught exactly that for the address route.
    if route == "tfidf_address":
        cls, ngram_range, min_df, sublinear = ta.CharTfidfVectorizer, (3, 4), 2, True
        s1_addrs = ta._build_s1_query_data(s1, id_col=ID_COL, addr_col=ADDR_COL)[1]
        build = lambda: chain(  # noqa: E731
            ta._iter_nonempty_addresses(s2, s3, ADDR_COL),
            (a for a in s1_addrs if a),
        )
    elif route == "tfidf_name":
        cls, ngram_range, min_df, sublinear = tn.CharTfidfVectorizer, (3, 4), 2, True
        s1_names = tn._build_s1_query_data(s1, id_col=ID_COL, name_col=NAME_COL)[1]
        build = lambda: chain(  # noqa: E731
            iter_pool_names(s2, s3, NAME_COL, tn.normalize_name),
            (n for n in s1_names if n),
        )
    else:
        raise ValueError(f"{route} is not a vectorizer route")

    effective_min_df = min_df if (n_candidates + n_queries) >= 100 else 1
    vec = cls(ngram_range=ngram_range, min_df=effective_min_df, sublinear_tf=sublinear)
    corpus = build()
    first = next(corpus, None)
    if first is None:
        raise RuntimeError("empty corpus")
    vec.fit(chain([first], corpus))
    return vec


def s1_shard(s1: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    """Contiguous S1 block. Contiguity keeps query blocks aligned with the full run."""
    return s1.iloc[start:stop].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# worker mode
# --------------------------------------------------------------------------- #
def run_worker(args) -> int:
    slice_dir = Path(args.slice_dir)
    spill_dir = Path(args.spill_dir)
    t_start = time.time()

    log(f"worker pid={os.getpid()} route={args.route} shard={args.shard_index}/{args.num_shards} "
        f"s1=[{args.start}:{args.stop}) spill={spill_dir}")
    s1, s2, s3 = load_slice(slice_dir)
    log(f"loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} RSS={rss_mb():.0f} MB")

    n_total = len(s1)
    if args.stop >= 0:
        s1_run = s1_shard(s1, args.start, min(args.stop, n_total))
    else:
        s1_run = s1
    log(f"this worker handles {len(s1_run):,} S1 rows ({len(s1_run)/max(n_total,1):.1%} of S1)")

    if spill_dir.exists() and args.reset:
        shutil.rmtree(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    n_buckets = args.buckets
    writer = EventSpillWriter(spill_dir, n_buckets=n_buckets)
    buf = RecordBuffer(writer, max_records=args.record_buffer)

    kwargs = dict(ROUTE_DEFAULTS[args.route])
    vectorizer = None
    if args.route in VECTORIZER_ROUTES:
        # Fitted on the FULL corpus, not the shard, so the vocabulary/IDF match the
        # un-sharded run exactly.
        t0 = time.time()
        vectorizer = fit_shared_vectorizer(args.route, s1, s2, s3)
        log(f"shared vectorizer fitted on full pool+S1 in {time.time()-t0:.1f}s "
            f"(vocab={len(vectorizer.vocab):,}) RSS={rss_mb():.0f} MB")
        kwargs["vectorizer"] = vectorizer

    if args.route == "exact_name":
        from src.blocking.exact_name import retrieve_exact_name as fn
        fn(s1_run, s2, s3, id_col=ID_COL, name_col=NAME_COL, sink=buf, **kwargs)
    elif args.route == "tfidf_name":
        from src.blocking.tfidf_name import retrieve_tfidf_name as fn
        fn(s1_run, s2, s3, id_col=ID_COL, name_col=NAME_COL, sink=buf, **kwargs)
    elif args.route == "tfidf_address":
        from src.blocking.tfidf_address import retrieve_tfidf_address as fn
        fn(s1_run, s2, s3, id_col=ID_COL, addr_col=ADDR_COL, sink=buf, **kwargs)
    elif args.route == "rare_token_name":
        from src.blocking.rare_token import retrieve_rare_token_name as fn
        fn(s1_run, s2, s3, id_col=ID_COL, name_col=NAME_COL, sink=buf, **kwargs)
    elif args.route == "rare_token_address":
        from src.blocking.rare_token import retrieve_rare_token_address as fn
        fn(s1_run, s2, s3, id_col=ID_COL, addr_col=ADDR_COL, sink=buf, **kwargs)
    elif args.route == "numeric_address":
        from src.blocking.numeric import retrieve_numeric as fn
        fn(s1_run, s2, s3, id_col=ID_COL, addr_col=ADDR_COL, sink=buf, **kwargs)
    elif args.route == "reverse_retrieval":
        from src.blocking.reverse import retrieve_reverse as fn
        fn(s1_run, s2, s3, id_col=ID_COL, name_col=NAME_COL, sink=buf,
           spill_dir=str(spill_dir), dedup_buckets=n_buckets, **kwargs)
    else:
        raise ValueError(f"unknown route {args.route}")

    buf.close()
    writer.close()
    gc.collect()

    stats = {
        "pid": os.getpid(),
        "route": args.route,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "start": args.start,
        "stop": args.stop,
        "s1_rows": len(s1_run),
        "events": writer.rows_written,
        "seconds": round(time.time() - t_start, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "spill_dir": str(spill_dir),
    }
    Path(args.stats_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_file).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"DONE {json.dumps(stats)}")
    return 0


# --------------------------------------------------------------------------- #
# selftest: pre-fitted vectorizer == the route's own internal fit
# --------------------------------------------------------------------------- #
def run_selftest(args) -> int:
    """Prove the sharded path is semantics-identical on real fixture data."""
    from src.blocking import tfidf_address as ta
    from src.blocking import tfidf_name as tn

    slice_dir = Path(args.slice_dir)
    s1, s2, s3 = load_slice(slice_dir)
    n = args.selftest_rows
    s1 = s1.iloc[:n].reset_index(drop=True)
    s2 = s2.iloc[: args.selftest_pool].reset_index(drop=True)
    s3 = s3.iloc[: args.selftest_pool].reset_index(drop=True)
    log(f"selftest on real fixture: S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    failures = []
    for route, mod, colkw in (("tfidf_address", ta, "addr_col"), ("tfidf_name", tn, "name_col")):
        kwargs = dict(ROUTE_DEFAULTS[route])
        colval = ADDR_COL if colkw == "addr_col" else NAME_COL
        # (a) route's own internal fit, whole S1
        sink_a = InMem()
        fn = mod.retrieve_tfidf_address if route == "tfidf_address" else mod.retrieve_tfidf_name
        fn(s1, s2, s3, id_col=ID_COL, **{colkw: colval}, sink=sink_a, **kwargs)
        full = sink_a.frame()

        # (b) pre-fitted shared vectorizer, same whole S1
        sink_b = InMem()
        vec = fit_shared_vectorizer(route, s1, s2, s3)
        fn(s1, s2, s3, id_col=ID_COL, **{colkw: colval}, sink=sink_b, vectorizer=vec, **kwargs)
        shared = sink_b.frame()

        same_full = full.equals(shared)
        log(f"  {route}: whole-S1 shared-vs-internal identical = {same_full} "
            f"({len(full):,} events)")

        # (c) sharded: two contiguous halves, each with its own pre-fitted vectorizer,
        #     unioned. Must equal the whole-S1 result.
        half = len(s1) // 2
        pieces = []
        for lo, hi in ((0, half), (half, len(s1))):
            sink_c = InMem()
            v2 = fit_shared_vectorizer(route, s1, s2, s3)  # fitted on FULL S1, not the shard
            fn(s1_shard(s1, lo, hi), s2, s3, id_col=ID_COL, **{colkw: colval},
               sink=sink_c, vectorizer=v2, **kwargs)
            pieces.append(sink_c.frame())
        sharded = pd.concat(pieces, ignore_index=True)
        key = ["s1_id", "candidate_id", "route", "rank", "score"]
        a = full.sort_values(key).reset_index(drop=True)
        b = sharded.sort_values(key).reset_index(drop=True)
        same_shard = len(a) == len(b) and a[key].equals(b[key])
        log(f"  {route}: 2-way-shard union == whole-S1 = {same_shard} "
            f"({len(sharded):,} vs {len(full):,} events)")

        if not (same_full and same_shard):
            failures.append(route)

    if failures:
        log(f"SELFTEST FAILED for {failures}")
        return 1
    log("SELFTEST PASSED: pre-fitted vectorizer is semantics-identical, and sharding is safe")
    return 0


class InMem:
    """Minimal sink that accumulates event rows (selftest only, small inputs)."""

    COLUMNS = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def append(self, record: dict) -> None:
        self.rows.append(record)

    def write_records(self, records: Sequence[dict]) -> None:
        self.rows.extend(records)

    emit = write_records

    def close(self) -> None:
        return None

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(self.rows, columns=self.COLUMNS)


# --------------------------------------------------------------------------- #
# finalize mode: bounded-memory union + external merge
# --------------------------------------------------------------------------- #
EVENT_COLUMNS = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]


def bucket_paths(spill_dir: Path, bucket: int) -> List[Path]:
    return sorted((spill_dir / f"bucket_{bucket:04d}").glob("part-*.parquet"))


def read_bucket_events(spill_dirs: Sequence[Path], bucket: int) -> Optional[pd.DataFrame]:
    """All events for one hash bucket, unioned across every worker's spill.

    Union is safe and idempotent: ``canonicalize_frame`` aggregates with ``max``/``min``/
    ``nunique`` over a groupby, so a pair contributed by two workers collapses to the same
    single row it would have had from one.
    """
    tables = []
    for d in spill_dirs:
        parts = bucket_paths(d, bucket)
        if parts:
            tables.append(pq.read_table(parts[0] if len(parts) == 1 else parts))
    if not tables:
        return None
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return table.select(EVENT_COLUMNS).to_pandas()


def iter_canonical_rows_lazy(
    canon_dir: Path, n_buckets: int, batch_size: int
) -> List[Iterator[Tuple[str, int, tuple]]]:
    """One lazy sorted row-iterator per canonical bucket file.

    Each file is already sorted by ``(s1_id, rank)``, which is the precondition
    ``heapq.merge`` needs. Reading in Arrow batches keeps only ``batch_size`` rows per
    bucket resident instead of the whole bucket.
    """
    full_cols = list(CANONICAL_CANDIDATE_COLUMNS) + ["rank"]
    streams: List[Iterator[Tuple[str, int, tuple]]] = []
    for b in range(n_buckets):
        path = canon_dir / f"bucket_{b:04d}.parquet"
        if not path.exists():
            continue

        def _gen(path=path) -> Iterator[Tuple[str, int, tuple]]:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size, columns=full_cols):
                df = batch.to_pandas()
                if df.empty:
                    continue
                # Guard the merge precondition rather than trusting the writer.
                order = df.sort_values(["s1_id", "rank"], kind="mergesort")
                s1v = order["s1_id"].tolist()
                rkv = order["rank"].tolist()
                # zip(*) builds all row tuples in one C-level pass; per-row .iloc would
                # dominate the runtime at slice scale.
                col_vals = [order[c].tolist() for c in full_cols]
                for i, tup in enumerate(zip(*col_vals)):
                    yield (s1v[i], int(rkv[i]), tup)

        streams.append(_gen())
    return streams


def run_finalize(args) -> int:
    t_start = time.time()
    spill_dirs = [Path(p) for p in args.spill_dirs]
    out_dir = Path(args.out_dir)
    canon_dir = Path(args.canon_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if canon_dir.exists():
        shutil.rmtree(canon_dir)
    canon_dir.mkdir(parents=True, exist_ok=True)

    n_buckets = args.buckets
    log(f"finalize: union of {len(spill_dirs)} spill dirs, {n_buckets} buckets")
    for d in spill_dirs:
        log(f"  {d}  parts={sum(1 for _ in d.rglob('part-*.parquet'))}")

    cand_path = out_dir / "candidates.parquet"
    events_path = out_dir / "retrieval_events.parquet"

    # ---- pass 1: events, one bucket at a time (bounded) --------------------
    t0 = time.time()
    ev_writer: Optional[pq.ParquetWriter] = None
    ev_rows = 0
    try:
        for b in range(n_buckets):
            frame = read_bucket_events(spill_dirs, b)
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(frame[EVENT_COLUMNS], preserve_index=False)
            if ev_writer is None:
                ev_writer = pq.ParquetWriter(events_path, table.schema, compression="zstd")
            ev_writer.write_table(table)
            ev_rows += table.num_rows
            del frame, table
    finally:
        if ev_writer is not None:
            ev_writer.close()
    log(f"pass1 events: {ev_rows:,} rows -> {events_path} in {time.time()-t0:.1f}s "
        f"RSS={rss_mb():.0f} MB")

    # ---- pass 2: canonicalize one bucket at a time to a temp file ---------
    t0 = time.time()
    canon_rows = 0
    for b in range(n_buckets):
        frame = read_bucket_events(spill_dirs, b)
        if frame is None or frame.empty:
            continue
        canon = canonicalize_frame(frame)
        del frame
        if canon.empty:
            continue
        # canonicalize_frame already collapsed per (pair, route) via max, so `rank`
        # survives as the best (minimum) rank that pair earned in that bucket.
        canon = canon.sort_values(["s1_id", "rank"], kind="mergesort").reset_index(drop=True)
        pq.write_table(
            pa.Table.from_pandas(canon, preserve_index=False),
            canon_dir / f"bucket_{b:04d}.parquet",
            compression="zstd",
        )
        canon_rows += len(canon)
        del canon
        gc.collect()
        if (b + 1) % 8 == 0:
            log(f"  pass2 bucket {b+1}/{n_buckets} canon_rows={canon_rows:,} "
                f"RSS={rss_mb():.0f} MB")
    log(f"pass2 canonicalized {canon_rows:,} candidate rows in {time.time()-t0:.1f}s")

    # ---- pass 3: lazy k-way merge -> globally s1_id-ordered candidates -----
    t0 = time.time()
    streams = iter_canonical_rows_lazy(canon_dir, n_buckets, args.merge_batch)
    full_cols = list(CANONICAL_CANDIDATE_COLUMNS) + ["rank"]
    out_cols = list(CANONICAL_CANDIDATE_COLUMNS)
    cand_writer: Optional[pq.ParquetWriter] = None
    written = 0
    buf: List[tuple] = []
    for item in heapq.merge(*streams, key=lambda it: (it[0], it[1])):
        buf.append(item[2])
        if len(buf) >= args.flush_rows:
            # One bulk DataFrame build per flush; a per-row pd.Series construction is
            # the other quadratic-ish cost at this row count.
            frame = pd.DataFrame(buf, columns=full_cols)[out_cols]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if cand_writer is None:
                cand_writer = pq.ParquetWriter(cand_path, table.schema, compression="zstd")
            cand_writer.write_table(table)
            written += table.num_rows
            buf = []
    if buf:
        frame = pd.DataFrame(buf, columns=full_cols)[out_cols]
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if cand_writer is None:
            cand_writer = pq.ParquetWriter(cand_path, table.schema, compression="zstd")
        cand_writer.write_table(table)
        written += table.num_rows
    if cand_writer is not None:
        cand_writer.close()
    else:
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(columns=out_cols)), cand_path, compression="zstd"
        )
    log(f"pass3 merged {written:,} candidates -> {cand_path} in {time.time()-t0:.1f}s")

    if not args.keep_canon and canon_dir.exists():
        shutil.rmtree(canon_dir)

    summary = {
        "events_rows": ev_rows,
        "candidate_rows": written,
        "spill_dirs": [str(d) for d in spill_dirs],
        "seconds": round(time.time() - t_start, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "events_path": str(events_path),
        "candidates_path": str(cand_path),
    }
    (out_dir / "finalize_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"FINALIZE DONE {json.dumps(summary)}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["worker", "finalize", "selftest"])
    ap.add_argument("--slice-dir", default="artifacts/slice/data")
    ap.add_argument("--route")
    ap.add_argument("--spill-dir")
    ap.add_argument("--spill-dirs", nargs="*")
    ap.add_argument("--out-dir", default="artifacts/slice/out")
    ap.add_argument("--canon-dir", default="/var/tmp/p1_canon")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--record-buffer", type=int,
                    default=DEFAULT_BLOCKING_CONFIG["record_buffer"])
    ap.add_argument("--buckets", type=int,
                    default=DEFAULT_BLOCKING_CONFIG["dedup_buckets"])
    ap.add_argument("--merge-batch", type=int, default=20_000)
    ap.add_argument("--flush-rows", type=int, default=50_000)
    ap.add_argument("--keep-canon", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--stats-file", default="")
    ap.add_argument("--selftest-rows", type=int, default=200)
    ap.add_argument("--selftest-pool", type=int, default=600)
    args = ap.parse_args()

    if args.mode == "worker":
        return run_worker(args)
    if args.mode == "finalize":
        return run_finalize(args)
    return run_selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
