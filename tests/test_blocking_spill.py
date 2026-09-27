"""Unit tests for the P1 bounded-memory spill primitives (Phase 7).

These lock in the properties the streaming refactor actually depends on:

* a route can never retain more than ``max_records`` dicts, no matter how many
  events it produces in total;
* spilled events round-trip with the declared schema and can be reconciled
  bucket-locally or globally with identical results;
* candidate output does not depend on how the pool was decomposed into blocks.

The last point is the one that is easy to assume and hard to get right, so it is
tested explicitly against tie-heavy data where a block boundary can straddle the
k-th rank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.blocking.spill import (
    CANONICAL_CANDIDATE_COLUMNS,
    DEFAULT_RECORD_BUFFER,
    EventSpillWriter,
    InMemoryEventSink,
    RecordBuffer,
    SpillManifest,
    bucket_of,
    canonicalize_frame,
    iter_single_bucket,
    resolve_spill_dir,
    write_candidate_dataset,
)
from src.blocking.tfidf_common import canonical_top_k, merge_top_k

EVENT_COLUMNS = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "route",
    "rank",
    "score",
]

#: The production threshold the refactor was designed around.
assert DEFAULT_RECORD_BUFFER == 50_000, "the 50,000 record budget is a fixed contract"


# --------------------------------------------------------------------------- #
# RecordBuffer: the guarantee that makes bounded memory possible
# --------------------------------------------------------------------------- #


def test_record_buffer_never_retains_more_than_threshold() -> None:
    """Peak retention must be capped by the threshold, not by the route's output size."""
    sink = InMemoryEventSink()
    buf = RecordBuffer(sink=sink, max_records=100)

    observed_peaks: list[int] = []
    for i in range(10_000):
        buf.append(
            {
                "pair_key": f"S1-0001::S2-{i:06d}",
                "s1_id": "S1-0001",
                "candidate_id": f"S2-{i:06d}",
                "candidate_source": "S2",
                "route": "exact_name",
                "rank": 1,
                "score": 1.0,
            }
        )
        observed_peaks.append(len(buf._buffer))

    buf.close()

    assert max(observed_peaks) <= 100, "buffer grew past its configured ceiling"
    assert buf.flushes == 100, "10,000 records at a threshold of 100 should flush 100 times"
    assert buf.rows_emitted == 10_000
    assert sink.rows == 10_000
    assert max(len(f) for f in sink.frames) <= 100, "a sink batch exceeded the threshold"


def test_record_buffer_splits_an_oversized_batch() -> None:
    """The reverse route emits whole batches; one batch may exceed the threshold.

    It must be split across flushes rather than retained whole, otherwise a single
    large batch silently defeats the memory bound.
    """
    sink = InMemoryEventSink()
    buf = RecordBuffer(sink=sink, max_records=250)

    batch = [
        {
            "pair_key": f"S1-0001::S2-{i:06d}",
            "s1_id": "S1-0001",
            "candidate_id": f"S2-{i:06d}",
            "candidate_source": "S2",
            "route": "reverse_retrieval",
            "rank": 1,
            "score": 0.5,
        }
        for i in range(1_000)
    ]
    buf.emit(batch)
    buf.close()

    assert max(len(f) for f in sink.frames) <= 250, "one batch was retained whole"
    assert buf.rows_emitted == 1_000
    assert sink.rows == 1_000
    assert len(sink.frames) == 4, "1,000 records at a threshold of 250 should span 4 flushes"


def test_record_buffer_in_memory_mode_retains_everything() -> None:
    """The in-memory path must stay available for the equivalence tests."""
    buf = RecordBuffer(sink=None, max_records=10)
    for i in range(25):
        buf.append(
            {
                "pair_key": f"S1-0001::S2-{i:06d}",
                "s1_id": "S1-0001",
                "candidate_id": f"S2-{i:06d}",
                "candidate_source": "S2",
                "route": "exact_name",
                "rank": 1,
                "score": 1.0,
            }
        )
    buf.close()

    frame = buf.result()
    assert len(frame) == 25, "with no sink the buffer is the accumulator and must not drop rows"
    assert list(frame.columns) == EVENT_COLUMNS
    assert buf.flushes == 0, "nothing should have been emitted to a sink that does not exist"


def test_record_buffer_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="max_records must be positive"):
        RecordBuffer(sink=InMemoryEventSink(), max_records=0)


# --------------------------------------------------------------------------- #
# Spill directory handling
# --------------------------------------------------------------------------- #


def test_resolve_spill_dir_is_non_destructive_by_default(tmp_path) -> None:
    """Default behaviour must never delete, or a rerun destroys completed work."""
    spill = tmp_path / "spill"
    spill.mkdir()
    keep = spill / "part.parquet"
    keep.write_text("precious")

    assert resolve_spill_dir(spill, reset=False) == spill
    assert keep.exists(), "resolve_spill_dir(reset=False) deleted an existing file"

    resolve_spill_dir(spill, reset=True)
    assert not keep.exists(), "resolve_spill_dir(reset=True) did not clear the directory"


def test_resolve_spill_dir_none_passes_through() -> None:
    assert resolve_spill_dir(None) is None


def test_bucket_of_is_stable_and_spreads_ids() -> None:
    """Every id must land in exactly one deterministic bucket, with no id hot-spotting."""
    for s1_id in ("S1-0001", "S1-0002", "S1-9999"):
        first = bucket_of(s1_id, 64)
        assert first == bucket_of(s1_id, 64), "bucket assignment is not deterministic"
        assert 0 <= first < 64

    counts = np.zeros(64, dtype=np.int64)
    for i in range(64_000):
        counts[bucket_of(f"S1-{i:06d}", 64)] += 1
    assert counts.min() > 0, "at least one bucket is empty, which wastes a whole merge stream"


# --------------------------------------------------------------------------- #
# Spill round-trip
# --------------------------------------------------------------------------- #


def _synthetic_events(n_s1: int, per_row: int) -> list[dict]:
    """Tie-heavy events: many rows share the same score so ranking is decided by id."""
    out = []
    for s in range(n_s1):
        s1_id = f"S1-{s:06d}"
        for r in range(per_row):
            cand = f"S2-{(s * per_row + r) % 5000:06d}"
            out.append(
                {
                    "pair_key": f"{s1_id}::{cand}",
                    "s1_id": s1_id,
                    "candidate_id": cand,
                    "candidate_source": "S2",
                    "route": ["exact_name", "tfidf_name", "numeric_address"][r % 3],
                    "rank": r + 1,
                    # Only three distinct scores -> lots of exact ties.
                    "score": [0.9, 0.5, 0.5][r % 3],
                }
            )
    return out


def _read_all_events(spill_dir) -> list[pd.DataFrame]:
    """Read every bucket of a spill, one bucket at a time, as raw event frames."""
    root = Path(spill_dir)
    out = []
    for bucket_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        out.extend(iter_single_bucket(bucket_dir, columns=EVENT_COLUMNS))
    return out


def test_event_spill_writer_roundtrips_and_marks_success(tmp_path) -> None:
    spill = resolve_spill_dir(tmp_path / "spill", reset=True)
    records = _synthetic_events(n_s1=64, per_row=10)

    writer = EventSpillWriter(spill, n_buckets=8)
    for chunk in (records[:300], records[300:]):
        writer.write_records(chunk)
    writer.close()

    parts = sorted(spill.rglob("part-*.parquet"))
    assert parts, "no Parquet parts were written"
    for bucket_dir in sorted(p for p in spill.iterdir() if p.is_dir()):
        assert (bucket_dir / "_SUCCESS").exists(), f"missing completion marker in {bucket_dir}"

    read_back = _read_all_events(spill)
    frame = pd.concat(read_back, ignore_index=True)
    assert len(frame) == len(records)
    assert list(frame.columns) == EVENT_COLUMNS
    assert set(frame["route"]) == {"exact_name", "tfidf_name", "numeric_address"}


def test_spill_manifest_tracks_route_completion(tmp_path) -> None:
    """The manifest is what makes a restart a resume rather than a full recompute.

    It records *routes*, not buckets: bucket completion is already covered by the
    per-bucket ``_SUCCESS`` markers, whereas route completion is what decides which
    routes may be skipped on the next attempt.
    """
    spill = resolve_spill_dir(tmp_path / "spill", reset=True)

    manifest = SpillManifest(spill)
    assert manifest.completed_routes() == []
    assert manifest.is_done("exact_name") is False
    assert manifest.rows_for("exact_name") == 0

    manifest.mark_done("exact_name", 12_345)
    assert manifest.is_done("exact_name") is True
    assert manifest.rows_for("exact_name") == 12_345
    assert manifest.completed_routes() == ["exact_name"]

    # Re-read from disk: the manifest must survive a process restart.
    assert SpillManifest(spill).rows_for("exact_name") == 12_345

    # A corrupt manifest is treated as absent, so the worst case is a recompute.
    (spill / "manifest.json").write_text("{not json", encoding="utf-8")
    assert SpillManifest(spill).completed_routes() == []


# --------------------------------------------------------------------------- #
# Canonicalization: bucket-local must equal global
# --------------------------------------------------------------------------- #


def test_canonicalize_frame_is_independent_of_input_order() -> None:
    """Reconciling a bucket must not depend on the order rows were spilled in."""
    events = pd.DataFrame(_synthetic_events(n_s1=32, per_row=8))

    forward = canonicalize_frame(events)
    shuffled = canonicalize_frame(events.sample(frac=1.0, random_state=7))
    reversed_ = canonicalize_frame(events.iloc[::-1].reset_index(drop=True))

    key = ["s1_id", "candidate_id"]
    assert (
        forward.sort_values(key).reset_index(drop=True)[list(forward.columns)].to_dict("list")
        == shuffled.sort_values(key).reset_index(drop=True)[list(shuffled.columns)].to_dict("list")
    )
    assert (
        forward.sort_values(key).reset_index(drop=True)[list(forward.columns)].to_dict("list")
        == reversed_.sort_values(key).reset_index(drop=True)[list(reversed_.columns)].to_dict("list")
    )


def test_canonicalize_frame_emits_the_declared_columns() -> None:
    """``canonicalize_frame`` returns the contract plus a ``rank`` sort helper.

    ``rank`` exists so the frame can be ordered by ``(s1_id, rank)``. It is deliberately
    *not* part of ``CANONICAL_CANDIDATE_COLUMNS`` and is dropped when the artifact is
    written, which keeps the streamed output identical to the historical 7-column
    contract in ``artifacts/candidates.parquet``.
    """
    events = pd.DataFrame(_synthetic_events(n_s1=16, per_row=6))
    out = canonicalize_frame(events)

    assert list(out.columns) == list(CANONICAL_CANDIDATE_COLUMNS) + ["rank"]
    assert set(CANONICAL_CANDIDATE_COLUMNS).issubset(out.columns)
    assert out["n_routes"].min() >= 1
    assert out["best_rank"].min() >= 1

    # rank must be a dense 1..n sequence restarting at every s1_id
    for _, grp in out.groupby("s1_id"):
        assert list(grp["rank"]) == list(range(1, len(grp) + 1))

    # ordered by (s1_id, rank) and s1_id ascending
    assert list(out["s1_id"]) == sorted(out["s1_id"])


def test_write_candidate_dataset_matches_the_historical_contract(tmp_path) -> None:
    """The written artifact must have exactly the 7 historical columns, no ``rank``."""
    import pyarrow.parquet as pq

    spill = resolve_spill_dir(tmp_path / "spill", reset=True)
    writer = EventSpillWriter(spill, n_buckets=4)
    writer.write_records(_synthetic_events(n_s1=64, per_row=8))
    writer.close()

    out_path = tmp_path / "candidates.parquet"
    written = write_candidate_dataset(
        (canonicalize_frame(chunk) for chunk in _read_all_events(spill)), out_path
    )

    schema = [f.name for f in pq.ParquetFile(out_path).schema_arrow]
    assert schema == list(CANONICAL_CANDIDATE_COLUMNS)
    assert "rank" not in schema
    assert written == len(pd.read_parquet(out_path))


def test_canonicalization_of_whole_spill_matches_bucket_local_sum(tmp_path) -> None:
    """The streamed whole-spill reconciliation must equal per-bucket reconciliation."""
    spill = resolve_spill_dir(tmp_path / "spill", reset=True)
    records = _synthetic_events(n_s1=200, per_row=9)
    writer = EventSpillWriter(spill, n_buckets=8)
    writer.write_records(records)
    writer.close()

    per_bucket = pd.concat(
        [canonicalize_frame(chunk) for chunk in _read_all_events(spill)],
        ignore_index=True,
    )
    events = pd.DataFrame(records)
    whole = canonicalize_frame(events)

    key = ["s1_id", "candidate_id"]
    lhs = per_bucket.sort_values(key).reset_index(drop=True)
    rhs = whole.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(lhs, rhs, check_dtype=False)


# --------------------------------------------------------------------------- #
# Block-partition invariance of top-k selection (the tie-breaking claim)
# --------------------------------------------------------------------------- #


def _f32(x: float) -> float:
    """Scores travel as float32 inside the sparse similarity matrix.

    The reference implementation has to use the same representation, otherwise a plain
    0.9 arrives as 0.8999999761581421 and the comparison fails on representation rather
    than on selection logic.
    """
    return float(np.float32(x))


def _global_top_k(pairs, top_k, min_score):
    """Reference: sort every surviving candidate, then truncate."""
    floor = _f32(min_score)
    items = [(_f32(sc), cid) for cid, sc in pairs if _f32(sc) >= floor]
    items.sort(key=lambda item: (-item[0], item[1]))
    return items[:top_k]


def _blocked_top_k(pairs, top_k, min_score, block_sizes):
    """Production path: per-block selection, merged, then selected again."""
    merged: list[tuple[float, str, str]] = []
    pos = 0
    for size in block_sizes:
        block = pairs[pos : pos + size]
        pos += size
        if not block:
            continue
        n = len(block)
        local = canonical_top_k(
            np.arange(n, dtype=np.int32),
            np.array([sc for _, sc in block], dtype=np.float32),
            [cid for cid, _ in block],
            "S2",
            top_k,
            min_score,
        )
        merged = merge_top_k(merged, local, top_k)

    if not merged:
        return []
    final = canonical_top_k(
        np.arange(len(merged), dtype=np.int32),
        np.array([m[0] for m in merged], dtype=np.float32),
        [m[1] for m in merged],
        "S2",
        top_k,
        min_score,
    )
    return [(sc, cid) for sc, cid, _ in final]


@pytest.mark.parametrize("top_k", [1, 5, 20])
def test_top_k_selection_is_invariant_to_block_partition(top_k: int) -> None:
    """A block boundary straddling the k-th rank must not change the selected set.

    The data is deliberately tie-heavy: 10 candidates at 0.9 and 150 at exactly 0.5,
    so the k-th score is 0.5 and the members of the tie group chosen depend entirely
    on the canonical ``(-score, candidate_id)`` ordering. A naive ``argpartition``
    would return different ids for different block sizes here.
    """
    scores = [0.9] * 10 + [0.5] * 150 + [0.1] * 40
    pairs = [(f"S2-{i:06d}", sc) for i, sc in enumerate(scores)]

    expected = _global_top_k(pairs, top_k, 0.05)

    for sizes in ([7] * 29, [13] * 16, [1] * 200, [100, 100], [50, 50, 50, 50], [200]):
        blocks = []
        pos = 0
        for size in sizes:
            blocks.append(size)
            pos += size
            if pos >= len(pairs):
                break
        got = _blocked_top_k(pairs, top_k, 0.05, blocks)
        assert got == expected, f"block partition {blocks} changed the top-k selection"


def test_top_k_selection_is_invariant_to_candidate_order_within_a_block() -> None:
    """Reordering candidates inside a block must not change the result either."""
    pairs = [(f"S2-{i:06d}", sc) for i, sc in enumerate([0.9] * 5 + [0.5] * 50 + [0.2] * 45)]
    expected = _global_top_k(pairs, 10, 0.05)

    rng = np.random.RandomState(0)
    for _ in range(5):
        order = rng.permutation(len(pairs))
        shuffled = [pairs[i] for i in order]
        assert _blocked_top_k(shuffled, 10, 0.05, [13] * 8) == expected


def test_min_score_filter_excludes_candidates() -> None:
    """0.04 is below the floor and must be dropped; exactly-0.05 is inclusive."""
    pairs = [(f"S2-{i:06d}", sc) for i, sc in enumerate([0.9, 0.04, 0.05, 0.2])]
    got = _blocked_top_k(pairs, 10, 0.05, [2, 2])
    assert got == [
        (_f32(0.9), "S2-000000"),
        (_f32(0.2), "S2-000003"),
        (_f32(0.05), "S2-000002"),
    ]
    assert (_f32(0.04), "S2-000001") not in got
