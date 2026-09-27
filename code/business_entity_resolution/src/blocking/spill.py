"""Disk-backed event spill and external candidate deduplication (L4).

Why this exists
---------------
At full scale the seven P1 routes emit roughly **260 million** retrieval events, measured
at ~124 bytes per event row. The previous orchestration kept every route's DataFrame alive
in ``route_dfs``, then ``pd.concat``-ed them and deduplicated the concatenation, which holds
two to three live copies at once. That is ~32 GB of event rows against ~11.8 GB of
headroom, so it cannot fit regardless of how the TF-IDF routes are blocked.

Strategy
--------
Events are partitioned by a **stable hash of ``s1_id``** and streamed to Parquet shards on
disk, so no more than one shard group is resident. Because ``deduplicate_candidates``
aggregates and ranks *within* each ``s1_id``, and every event for a given ``s1_id`` lands in
the same bucket, each bucket can be deduplicated and ranked completely independently and the
union of the results is bit-identical to deduplicating everything at once.

``zlib.crc32`` is used for the bucket hash rather than Python's ``hash()``, because
``hash()`` is randomized per process (PYTHONHASHSEED) and would make the partitioning
non-reproducible across runs.

Note on rank
------------
``deduplicate_candidates`` *recomputes* ``rank`` globally across routes (it sorts by
``s1_id`` asc, ``score`` desc, ``candidate_id`` asc and numbers sequentially), so the
per-route ``rank`` carried on an event is intermediate information. Per-bucket aggregation
therefore reproduces the single-shot result exactly.
"""

from __future__ import annotations

import heapq
import json
import shutil
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "CANONICAL_CANDIDATE_COLUMNS",
    "DEFAULT_DEDUP_BUCKETS",
    "DEFAULT_RECORD_BUFFER",
    "EVENT_COLUMNS",
    "EventSpillWriter",
    "InMemoryEventSink",
    "RecordBuffer",
    "SpillManifest",
    "canonicalize_frame",
    "deduplicate_frame",
    "empty_candidate_df",
    "iter_spilled_buckets",
    "iter_canonical_candidates_merged",
    "resolve_spill_dir",
    "write_candidate_dataset",
    "write_events_stream",
]

DEFAULT_DEDUP_BUCKETS: int = 64

#: Sentinel written into a bucket directory once every shard for it has been flushed.
_SUCCESS_MARKER = "_SUCCESS"

#: Number of records a :class:`RecordBuffer` accumulates before flushing to its sink.
#: This is the hard upper bound on per-batch event memory during retrieval
#: (``50_000 * ~547 B`` of Python dicts ~= 27 MB before the sink converts it).
DEFAULT_RECORD_BUFFER: int = 50_000

EVENT_COLUMNS: List[str] = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "route",
    "rank",
    "score",
]

#: Canonical ``candidates.parquet`` schema, per ``docs/schemas.md`` section 6.
CANONICAL_CANDIDATE_COLUMNS: List[str] = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "n_routes",
    "best_rank",
    "best_score",
]

# Explicit dtypes keep the on-disk shards small (string columns dominate at ~260M rows) and
# make the final artifacts schema-stable.
_EVENT_SCHEMA = pa.schema(
    [
        pa.field("pair_key", pa.large_string()),
        pa.field("s1_id", pa.large_string()),
        pa.field("candidate_id", pa.large_string()),
        pa.field("candidate_source", pa.large_string()),
        pa.field("route", pa.large_string()),
        pa.field("rank", pa.int32()),
        pa.field("score", pa.float64()),
    ]
)


# Artifact schema for ``retrieval_events.parquet``. Pinned explicitly (rather than inferred
# from a DataFrame) so the streamed artifact matches the contract of the existing file
# byte-for-byte at the schema level: large_string for text, int64 ``rank``, float64 ``score``.
_ARTIFACT_EVENT_SCHEMA = pa.schema(
    [
        pa.field("pair_key", pa.large_string()),
        pa.field("s1_id", pa.large_string()),
        pa.field("candidate_id", pa.large_string()),
        pa.field("candidate_source", pa.large_string()),
        pa.field("route", pa.large_string()),
        pa.field("rank", pa.int64()),
        pa.field("score", pa.float64()),
    ]
)


def empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def bucket_of(s1_id: str, n_buckets: int) -> int:
    """Stable, process-independent bucket assignment for an ``s1_id``."""
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")
    return zlib.crc32(s1_id.encode("utf-8")) % n_buckets


def resolve_spill_dir(
    spill_dir: Optional[str | Path],
    reset: bool = False,
) -> Optional[Path]:
    """Validate and create the spill directory, or return None when spilling is disabled.

    Unlike the original implementation this **never deletes an existing spill directory**
    unless ``reset=True`` is passed explicitly. Destructive-by-default was a real defect: it
    made restart impossible, because a partially completed run could only ever be thrown
    away and recomputed from scratch. Callers that genuinely want a clean slate (the test
    suite) pass ``reset=True``; the production runner does not.
    """
    if spill_dir is None:
        return None
    path = Path(spill_dir)
    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class EventSpillWriter:
    """Streams candidate records into hash-bucketed Parquet shards.

    Memory is bounded by the largest single :meth:`emit` batch plus one open shard buffer
    per bucket; nothing accumulates across calls.
    """

    def __init__(
        self,
        spill_dir: str | Path,
        n_buckets: int = DEFAULT_DEDUP_BUCKETS,
        columns: Sequence[str] = tuple(EVENT_COLUMNS),
    ) -> None:
        self.spill_dir = Path(spill_dir)
        self.n_buckets = n_buckets
        self.columns = list(columns)
        self._shard_counts: Dict[int, int] = {}
        self._rows_written = 0
        self.schema = pa.schema([_EVENT_SCHEMA.field(name) for name in self.columns])
        for b in range(self.n_buckets):
            (self.spill_dir / f"bucket_{b:04d}").mkdir(parents=True, exist_ok=True)

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def write_frame(self, frame: pd.DataFrame) -> None:
        """Spill a DataFrame of event records."""
        if frame is None or len(frame) == 0:
            return
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing event columns: {missing}")
        buckets = [bucket_of(str(s), self.n_buckets) for s in frame["s1_id"].tolist()]
        bucket_series = pd.Series(buckets, index=frame.index, dtype="int32")
        for b, part in frame.groupby(bucket_series, sort=True):
            self._write_shard(int(b), part)

    def write_records(self, records: Sequence[dict]) -> None:
        """Spill a list of event record dicts (the :class:`RecordBuffer` sink protocol)."""
        if not records:
            return
        self.write_frame(pd.DataFrame(list(records), columns=self.columns))

    def emit(self, records: Sequence[dict]) -> None:
        """Alias matching the reverse route's historical sink protocol."""
        self.write_records(records)

    def _write_shard(self, bucket: int, part: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(
            part[self.columns].reset_index(drop=True), schema=self.schema, preserve_index=False
        )
        seq = self._shard_counts.get(bucket, 0)
        self._shard_counts[bucket] = seq + 1
        target = self.spill_dir / f"bucket_{bucket:04d}" / f"part-{seq:05d}.parquet"
        pq.write_table(table, target, compression="zstd")
        self._rows_written += table.num_rows

    def close(self) -> None:
        self._shard_counts.clear()
        for b in range(self.n_buckets):
            marker = self.spill_dir / f"bucket_{b:04d}" / _SUCCESS_MARKER
            marker.write_text("ok\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
class InMemoryEventSink:
    """Collects event frames in a list. Test/reference implementation of the sink protocol.

    Deliberately unbounded: it exists so the in-memory retrieval path stays available as the
    equivalence reference for the streaming path, not so it can be used at full scale.
    """

    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []
        self.rows = 0

    def write_frame(self, frame: pd.DataFrame) -> None:
        if frame is None or len(frame) == 0:
            return
        self.frames.append(frame)
        self.rows += len(frame)

    def write_records(self, records: Sequence[dict]) -> None:
        if not records:
            return
        self.write_frame(pd.DataFrame(list(records), columns=EVENT_COLUMNS))

    def emit(self, records: Sequence[dict]) -> None:
        """Alias matching the reverse route's historical sink protocol."""
        self.write_records(records)

    def result(self) -> pd.DataFrame:
        if not self.frames:
            return empty_candidate_df()
        return pd.concat(self.frames, ignore_index=True)


class RecordBuffer:
    """Bounded buffer that converts route output into a sink as it is produced.

    This is the single object that removes the per-route ``records = []`` list that made
    every route memory-fatal. Routes append one dict at a time; the buffer flushes to the
    sink every ``max_records`` appends, so peak retention is one threshold's worth of
    records regardless of how many events a route produces in total.

    A route that has no sink configured still needs to return a DataFrame, so ``None`` is
    accepted and simply accumulates -- that is the historical in-memory behaviour, kept for
    the equivalence tests.
    """

    def __init__(self, sink: Optional[object] = None, max_records: int = DEFAULT_RECORD_BUFFER) -> None:
        if max_records <= 0:
            raise ValueError(f"max_records must be positive, got {max_records}")
        self.sink = sink
        self.max_records = max_records
        self._buffer: List[dict] = []
        self.records_appended = 0
        self.flushes = 0
        self.rows_emitted = 0

    def append(self, record: dict) -> None:
        self._buffer.append(record)
        self.records_appended += 1
        if len(self._buffer) >= self.max_records:
            self.flush()

    def emit(self, records: Sequence[dict]) -> None:
        """Append a batch of records. Matches the reverse route's historical sink protocol."""
        for record in records:
            self.append(record)

    def flush(self) -> None:
        """Emit buffered records to the sink and release them."""
        if not self._buffer:
            return
        if self.sink is None:
            # No sink: retain. This is the in-memory equivalence path.
            return
        self.sink.write_records(self._buffer)
        self.rows_emitted += len(self._buffer)
        self.flushes += 1
        self._buffer = []

    def close(self) -> None:
        self.flush()

    def result(self) -> pd.DataFrame:
        """Materialize the retained records. Only meaningful when ``sink is None``."""
        if not self._buffer:
            return empty_candidate_df()
        return pd.DataFrame(self._buffer, columns=EVENT_COLUMNS)


class SpillManifest:
    """Records which routes have completed, so a restart can resume instead of recomputing.

    Backed by a small JSON file inside the spill directory. This exists because the original
    spill had no completion tracking at all, which together with the destructive
    ``resolve_spill_dir`` made every restart a full, fatal recompute.
    """

    def __init__(self, spill_dir: str | Path) -> None:
        self.path = Path(spill_dir) / "manifest.json"
        self._done: Dict[str, int] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._done = {
                        str(k): int(v) for k, v in loaded.get("completed_routes", {}).items()
                    }
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                # A corrupt manifest is treated as absent: correctness is unaffected because
                # the worst case is recomputing a route.
                self._done = {}

    def is_done(self, route: str) -> bool:
        return route in self._done

    def rows_for(self, route: str) -> int:
        return self._done.get(route, 0)

    def mark_done(self, route: str, rows: int) -> None:
        self._done[route] = int(rows)
        self._flush()

    def completed_routes(self) -> List[str]:
        return sorted(self._done)

    def _flush(self) -> None:
        payload = {
            "version": 1,
            "completed_routes": dict(sorted(self._done.items())),
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def iter_spilled_buckets(
    spill_dir: str | Path,
    n_buckets: int = DEFAULT_DEDUP_BUCKETS,
    columns: Sequence[str] = tuple(EVENT_COLUMNS),
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """Yield ``(bucket_id, frame)`` one bucket at a time, in deterministic order.

    Only one bucket is resident at a time, which is what bounds peak memory during the
    external deduplication pass.
    """
    root = Path(spill_dir)
    for b in range(n_buckets):
        bucket_dir = root / f"bucket_{b:04d}"
        if not bucket_dir.is_dir():
            continue
        parts = sorted(bucket_dir.glob("part-*.parquet"))
        if not parts:
            continue
        tables = [pq.read_table(p, columns=list(columns)) for p in parts]
        frame = pa.concat_tables(tables).to_pandas()
        del tables
        yield b, frame


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate one bucket of retrieval events into unique candidate pairs.

    Semantics are identical to
    :func:`~src.blocking.candidate_generation.deduplicate_candidates`:

    - group by ``(pair_key, s1_id, candidate_id, candidate_source)``
    - ``score`` = max across routes
    - ``route``  = comma-joined sorted set of retrieving routes
    - rank      = 1-based position within ``s1_id`` ordered by
      ``score`` desc, then ``candidate_id`` asc

    A bucket contains every event for its ``s1_id`` values, so the per-``s1_id`` ranking is
    complete and the union over buckets reproduces the single-shot result exactly.
    """
    if events_df is None or len(events_df) == 0:
        return empty_candidate_df()

    agg_df = (
        events_df.groupby(
            ["pair_key", "s1_id", "candidate_id", "candidate_source"], as_index=False
        )
        .agg(
            score=("score", "max"),
            route=("route", lambda routes: ",".join(sorted(set(routes)))),
        )
    )
    agg_df = agg_df.sort_values(
        by=["s1_id", "score", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    agg_df["rank"] = agg_df.groupby("s1_id").cumcount() + 1
    agg_df["rank"] = agg_df["rank"].astype("int64")
    return agg_df[EVENT_COLUMNS]


def write_candidate_dataset(
    bucket_frames: Iterator[pd.DataFrame],
    out_path: str | Path,
    columns: Sequence[str] = tuple(CANONICAL_CANDIDATE_COLUMNS),
) -> int:
    """Stream per-bucket canonical candidate frames into a single Parquet file.

    Uses a single ``ParquetWriter`` so the row count is unbounded while resident memory
    stays at one bucket.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    total = 0
    out_cols = list(columns)
    try:
        for frame in bucket_frames:
            if frame is None or len(frame) == 0:
                continue
            missing = [c for c in out_cols if c not in frame.columns]
            if missing:
                raise ValueError(f"missing candidate columns: {missing}")
            table = pa.Table.from_pandas(
                frame[out_cols].reset_index(drop=True), preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            total += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        # No buckets produced rows: still emit a valid, empty artifact with the schema.
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(columns=out_cols)), out, compression="zstd"
        )
    return total


# ---------------------------------------------------------------------------
# Bucket-local canonicalization (docs/schemas.md section 6)
# ---------------------------------------------------------------------------
def canonicalize_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate one bucket of events and attach canonical metadata.

    This is the bucket-local form of
    :func:`~src.utils.adapters.reconcile_candidates_schema`. It is valid because the spill
    partitions on ``crc32(s1_id)``: every event belonging to a given ``s1_id`` is confined
    to exactly one bucket, so

    - the per-``s1_id`` dedup/rank is complete within the bucket, and
    - ``n_routes`` / ``best_rank`` / ``best_score``, which are per-``pair_key`` aggregates
      over retrieval events, never straddle a bucket boundary.

    Equivalence with the global implementation is **explicitly tested** in
    ``tests/test_blocking_spill.py::test_bucket_local_canonicalization_equals_global``;
    it is not merely asserted here.

    Returns columns ``CANONICAL_CANDIDATE_COLUMNS``, sorted by ``(s1_id, rank)``.
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame(columns=CANONICAL_CANDIDATE_COLUMNS)

    # Candidate identity: (pair_key, s1_id, candidate_id, candidate_source).
    agg = (
        events_df.groupby(
            ["pair_key", "s1_id", "candidate_id", "candidate_source"], as_index=False
        )
        .agg(
            best_score=("score", "max"),
            best_rank=("rank", "min"),
            n_routes=("route", "nunique"),
        )
    )

    # Global union rank per s1_id: score desc, then candidate_id asc (the frozen ordering).
    agg = agg.sort_values(
        by=["s1_id", "best_score", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    agg["rank"] = agg.groupby("s1_id").cumcount() + 1

    agg["n_routes"] = agg["n_routes"].astype("int64")
    agg["best_rank"] = agg["best_rank"].astype("int64")
    agg["best_score"] = agg["best_score"].astype("float64")
    return agg[CANONICAL_CANDIDATE_COLUMNS + ["rank"]].sort_values(
        by=["s1_id", "rank"], ascending=[True, True]
    ).reset_index(drop=True)


def iter_single_bucket(
    bucket_dir: str | Path,
    columns: Sequence[str] = tuple(EVENT_COLUMNS),
) -> Iterator[pd.DataFrame]:
    """Yield the raw event frames of a single bucket directory, in shard order."""
    root = Path(bucket_dir)
    parts = sorted(root.glob("part-*.parquet"))
    if not parts:
        return
    tables = [pq.read_table(part, columns=list(columns)) for part in parts]
    frame = pa.concat_tables(tables).to_pandas()
    del tables
    yield frame


def _canonical_rows(
    spill_dir: str | Path,
    n_buckets: int = DEFAULT_DEDUP_BUCKETS,
) -> List[Iterator[tuple]]:
    """One row iterator per non-empty bucket, each yielding ``(s1_id, rank, row_tuple)``.

    Each bucket is canonicalized and sorted internally, so its rows are already in
    non-decreasing ``s1_id`` order. That is what makes the downstream k-way merge correct.
    """
    root = Path(spill_dir)
    streams: List[Iterator[tuple]] = []
    for b in range(n_buckets):
        bucket_dir = root / f"bucket_{b:04d}"
        if not bucket_dir.is_dir():
            continue

        def _gen(bucket_dir: Path = bucket_dir) -> Iterator[tuple]:
            for frame in iter_single_bucket(bucket_dir):
                if frame is None or len(frame) == 0:
                    continue
                canon = canonicalize_frame(frame)
                if len(canon) == 0:
                    continue
                cols = list(canon.columns)
                s1_values = canon["s1_id"].tolist()
                rank_values = canon["rank"].tolist()
                for i in range(len(canon)):
                    yield (
                        s1_values[i],
                        rank_values[i],
                        tuple(canon.iloc[i][c] for c in cols),
                    )

        streams.append(_gen())
    return streams


def iter_canonical_candidates_merged(
    spill_dir: str | Path,
    n_buckets: int = DEFAULT_DEDUP_BUCKETS,
    columns: Sequence[str] = tuple(CANONICAL_CANDIDATE_COLUMNS),
    flush_rows: int = DEFAULT_RECORD_BUFFER,
) -> Iterator[pd.DataFrame]:
    """Yield canonical candidate frames globally ordered by ``s1_id``, then ``rank``.

    ``candidates.parquet`` is contractually sorted by ``s1_id`` ascending then rank (verified
    against the existing artifact), so a bucket-major concatenation would silently regress
    that invariant.

    Buckets cannot simply be yielded whole: bucket *b* may hold ``S1-0005`` and ``S1-0099``
    while bucket *c* holds ``S1-0007``, so frame-granular yielding interleaves incorrectly.
    This performs a true **row-level k-way merge** with ``heapq.merge`` over the buckets,
    each of which is internally sorted, and re-buffers the merged rows into ``flush_rows``
    sized frames. Peak memory is one frame per bucket plus one output frame -- O(n_buckets +
    flush_rows), not O(rows).
    """
    root = Path(spill_dir)
    streams = _canonical_rows(root, n_buckets)
    if not streams:
        return

    merged = heapq.merge(*streams, key=lambda item: (item[0], item[1]))
    full_cols = list(CANONICAL_CANDIDATE_COLUMNS) + ["rank"]
    out_cols = list(columns)
    buf: List[pd.Series] = []
    for item in merged:
        buf.append(pd.Series(item[2], index=full_cols))
        if len(buf) >= flush_rows:
            frame = pd.DataFrame(buf)
            yield frame[out_cols].reset_index(drop=True)
            buf = []
    if buf:
        yield pd.DataFrame(buf)[out_cols].reset_index(drop=True)


def write_events_stream(
    bucket_frames: Iterable[pd.DataFrame],
    out_path: str | Path,
    columns: Sequence[str] = tuple(EVENT_COLUMNS),
) -> int:
    """Stream per-bucket event frames into a single ``retrieval_events.parquet``.

    Preserves the single-file contract downstream P2/P3 code already reads from, while
    never holding more than one bucket resident. This replaces the previous
    ``_collect_spilled_events`` which read every bucket and ``pd.concat``-ed them, i.e. it
    re-created the entire ~32 GB event spike the spill existed to avoid.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    total = 0
    schema = _ARTIFACT_EVENT_SCHEMA
    try:
        for frame in bucket_frames:
            if frame is None or len(frame) == 0:
                continue
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise ValueError(f"missing event columns: {missing}")
            table = pa.Table.from_pandas(
                frame[list(columns)].reset_index(drop=True),
                schema=schema,
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(out, schema, compression="zstd")
            writer.write_table(table)
            total += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(columns=list(columns)), schema=schema),
            out,
            compression="zstd",
        )
    return total
