"""Tests for the sharded P2 runner (scripts/run_p2_parallel.py).

P2 has to cover the whole P1 candidate set (~25M pairs), and ``build_features`` preserves
its input row order, which makes position-based sharding the only way to keep the output
aligned with ``candidates.parquet``. Two things can silently break that alignment:

* a shard boundary that lands in the middle of one ``s1_id``'s candidates. Events are
  routed to shards by ``s1_id``, so every event for that ``s1_id`` would go to one shard
  while the *other* shard's pairs for it silently fall back to default retrieval
  features -- wrong output, not an error. ``snap_bounds_to_s1_groups`` prevents that.
* assuming the candidate file is ``s1_id``-sorted without checking, which would make
  ``searchsorted`` event routing meaningless.

These tests pin both, and assert the sharded result is bit-identical to a single-process
``build_features`` call on the same input.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

from src.features.build import FEATURE_COLUMNS, build_features  # noqa: E402

EVENT_COLUMNS = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]
ROUTES = ["exact_name", "tfidf_name", "rare_token_name", "numeric_address", "reverse_retrieval"]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "p2_parallel", REPO / "scripts" / "run_p2_parallel.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["p2_parallel"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _make_dataset(tmp_path: Path, n_s1: int = 120, cand_per_s1: int = 6):
    """Synthetic but contract-shaped P1 outputs: s1-sorted candidates, one row per pair."""
    rng = np.random.default_rng(11)
    s1_ids = [f"S1-{i:05d}" for i in range(n_s1)]
    cand_ids = [f"S2-{i:06d}" for i in range(n_s1 * cand_per_s1)]
    rows, events = [], []
    ci = 0
    for s1 in s1_ids:
        for _ in range(cand_per_s1):
            cid = cand_ids[ci]
            pair = f"{s1}::{cid}"
            rows.append(dict(pair_key=pair, s1_id=s1, candidate_id=cid,
                             candidate_source="S2", n_routes=2, best_rank=1, best_score=0.75))
            for r, route in enumerate(rng.choice(ROUTES, size=2, replace=False), start=1):
                events.append(dict(pair_key=pair, s1_id=s1, candidate_id=cid,
                                   candidate_source="S2", route=str(route), rank=r,
                                   score=round(float(rng.random()), 6)))
            ci += 1
    cands = pd.DataFrame(rows).sort_values("s1_id", kind="mergesort").reset_index(drop=True)
    evs = pd.DataFrame(events)

    records = pd.DataFrame(
        {
            "entity_id": s1_ids + cand_ids,
            "source": ["S1"] * n_s1 + ["S2"] * len(cand_ids),
            "raw_name": [f"{'alpha bravo delta echo foxtrot'[i % 30: i % 30 + 5]} holdings {i}"
                         for i in range(n_s1 + len(cand_ids))],
            "raw_address": [f"{i} {i % 97} main street boulevard" for i in range(n_s1 + len(cand_ids))],
            "raw_country": ["IN"] * (n_s1 + len(cand_ids)),
            "norm_name_cons": [f"alpha bravo delta echo foxtrot {i % 7}" for i in range(n_s1 + len(cand_ids))],
            "norm_name_aggr": [f"alpha bravo delta echo {i % 5}" for i in range(n_s1 + len(cand_ids))],
            "norm_address_cons": [f"{i % 11} main street" for i in range(n_s1 + len(cand_ids))],
            "country_norm": ["in"] * (n_s1 + len(cand_ids)),
        }
    )

    inp = tmp_path / "in"
    inp.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(cands, preserve_index=False), inp / "candidates.parquet")
    pq.write_table(pa.Table.from_pandas(evs, preserve_index=False), inp / "retrieval_events.parquet")
    pq.write_table(pa.Table.from_pandas(records, preserve_index=False), inp / "records.parquet")
    return inp, cands, evs, records


def _run(runner_module, tmp_path, inp, shards, rows_per_shard):
    out = tmp_path / "features.parquet"
    metrics = tmp_path / "p2_metrics.json"
    argv = [
        sys.executable, "-u", str(REPO / "scripts" / "run_p2_parallel.py"),
        "--candidates", str(inp / "candidates.parquet"),
        "--events", str(inp / "retrieval_events.parquet"),
        "--records", str(inp / "records.parquet"),
        "--out", str(out), "--metrics", str(metrics),
        "--work-dir", str(tmp_path / "work"),
        "--shards", str(shards), "--rows-per-shard", str(rows_per_shard),
    ]
    # Drive main() in-process so the test can reuse the loaded module.
    old_argv = sys.argv
    sys.argv = ["run_p2_parallel.py"] + argv[3:]
    try:
        rc = runner_module.main()
    finally:
        sys.argv = old_argv
    assert rc == 0, f"runner exited {rc}"
    return out, metrics


def test_shards_never_split_an_s1_id(runner, tmp_path):
    inp, cands, _, _ = _make_dataset(tmp_path)
    bounds = runner.snap_bounds_to_s1_groups(
        inp / "candidates.parquet", runner.shard_bounds(len(cands), 5), len(cands)
    )
    s1 = cands["s1_id"].to_numpy()
    for lo, hi in bounds:
        inside = s1[lo:hi]
        # every s1_id in the shard appears nowhere else
        assert pd.Series(inside).isin(s1).all()
        for value in pd.unique(inside):
            assert np.count_nonzero(s1 == value) == np.count_nonzero(inside == value)
    # shards tile the candidate list exactly, in order
    assert bounds[0][0] == 0 and bounds[-1][1] == len(cands)
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


def test_rejects_unsorted_candidates(runner, tmp_path):
    inp, cands, _, _ = _make_dataset(tmp_path, n_s1=40)
    shuffled = cands.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(shuffled, preserve_index=False),
                   inp / "candidates.parquet")
    with pytest.raises(RuntimeError, match="not sorted by s1_id"):
        runner.assert_s1_sorted(inp / "candidates.parquet")


def test_sharded_features_are_bit_identical_to_single_process(runner, tmp_path):
    inp, cands, evs, records = _make_dataset(tmp_path, n_s1=140, cand_per_s1=7)
    out, metrics = _run(runner, tmp_path, inp, shards=4, rows_per_shard=120)

    ref = build_features(cands, records, evs, chunk_size=200)
    got = pq.read_table(out).to_pandas()

    assert len(got) == len(ref) == len(cands)
    assert list(got.columns) == ["pair_key", "s1_id", "candidate_id"] + FEATURE_COLUMNS
    assert (got["pair_key"].to_numpy() == ref["pair_key"].to_numpy()).all()
    for col in FEATURE_COLUMNS:
        a, b = ref[col].to_numpy(), got[col].to_numpy()
        assert ((a == b) | (pd.isna(a) & pd.isna(b))).all(), f"{col} differs"
        assert str(got[col].dtype) == "float32"

    import json

    checks = json.loads(metrics.read_text())["validation"]
    assert all(checks.values()), checks


def test_no_retrieval_features_fall_back_to_defaults(runner, tmp_path):
    """Every candidate pair has events here, so no row may show the no-events default
    (n_routes == 1.0 with all-zero route flags and NaN best_rank/best_score)."""
    inp, cands, evs, records = _make_dataset(tmp_path, n_s1=100, cand_per_s1=5)
    out, _ = _run(runner, tmp_path, inp, shards=3, rows_per_shard=90)
    got = pq.read_table(out).to_pandas()
    assert (got["n_routes"] > 1.0).all(), "some pairs lost their retrieval events"
    assert got["retrieval_best_rank"].notna().all()
    assert got["retrieval_best_score"].notna().all()
    flags = [c for c in got.columns if c.startswith("retrieved_by_")]
    assert (got[flags].sum(axis=1) > 0).all()


def test_multiple_candidate_flushes_still_route_rows_to_the_right_shard(
    runner, tmp_path
):
    """Regression: shard routing must survive more than one candidate flush.

    The candidate buffer is concatenated with ``pd.concat`` so that each frame keeps the
    absolute row index assigned while reading, because that index is what ``searchsorted``
    uses to route rows to shards. Using ``ignore_index=True`` silently rewrites every
    buffer to start at 0, so every flush after the first lands in shard 0 and the entire
    candidate set ends up in one shard. That is invisible on a small input, where the
    whole file fits in a single flush, and only appears once the input spans several
    flushes -- so this test shrinks the flush size instead of generating 400k rows.
    """
    inp, cands, _, _ = _make_dataset(tmp_path)
    n = len(cands)
    # Both the read batch and the flush threshold must be small: the buffer only flushes
    # between read batches, so shrinking one alone still yields a single flush.
    batch_rows = max(1, n // 12)
    flush_rows = max(1, n // 12)

    out = tmp_path / "features.parquet"
    metrics = tmp_path / "p2_metrics.json"
    old_argv = sys.argv
    sys.argv = [
        "run_p2_parallel.py",
        "--candidates", str(inp / "candidates.parquet"),
        "--events", str(inp / "retrieval_events.parquet"),
        "--records", str(inp / "records.parquet"),
        "--out", str(out), "--metrics", str(metrics),
        "--work-dir", str(tmp_path / "work_multi"),
        "--shards", "4", "--rows-per-shard", str(n // 4),
        "--candidate-batch-rows", str(batch_rows),
        "--candidate-flush-rows", str(flush_rows),
    ]
    try:
        assert runner.main() == 0
    finally:
        sys.argv = old_argv

    # Every row must land in exactly one shard, and the shards must partition the input.
    shard_rows = []
    for i in range(4):
        p = tmp_path / "work_multi" / f"cand_{i}.parquet"
        shard_rows.append(pq.ParquetFile(p).metadata.num_rows if p.exists() else 0)
    assert sum(shard_rows) == n, f"rows lost or duplicated across shards: {shard_rows}"
    assert max(shard_rows) > 0 and min(x for x in shard_rows if x) > 0, (
        f"expected every shard to be populated, got {shard_rows}"
    )
    assert len([x for x in shard_rows if x > 0]) >= 3, (
        f"routing collapsed into too few shards: {shard_rows}"
    )

    # And the features must still match a single-process build exactly.
    got = pq.read_table(out).to_pandas()
    records = pq.read_table(inp / "records.parquet").to_pandas()
    events = pq.read_table(inp / "retrieval_events.parquet").to_pandas()
    expected = build_features(cands, records, events)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_worker_process_cannot_spawn_more_workers(runner, tmp_path):
    """Regression: a feature worker must never orchestrate other workers.

    The first recovery run used ``multiprocessing.Pool``, which reported nothing useful
    when the box was reset: no exit code, no stderr, no output. Switching to one
    subprocess per shard fixed the observability but introduced a fork bomb, because the
    child was launched with ``--stage features`` and therefore ran the *parent* stage
    again, ignoring ``--only-shards`` and spawning a fresh generation of children per
    worker. Six shards became thirty-six, then two hundred and sixteen.

    The fix is structural: a child is launched with ``--run-worker N``, a mode that
    handles exactly one shard and returns before any orchestration is reachable. This test
    fails if that ordering ever regresses.
    """
    inp, cands, _evs, _records = _make_dataset(tmp_path, n_s1=20, cand_per_s1=3)
    work = tmp_path / "work_single"
    work.mkdir(parents=True, exist_ok=True)
    # A single-shard work dir, as the launcher would have prepared it.
    pq.write_table(pa.Table.from_pandas(cands, preserve_index=False), work / "cand_0.parquet")
    for name in ("records.parquet",):
        pq.write_table(pa.Table.from_pandas(pq.read_table(inp / name).to_pandas(),
                                           preserve_index=False), work / name)
    evs = pq.read_table(inp / "retrieval_events.parquet").to_pandas()
    pq.write_table(pa.Table.from_pandas(evs, preserve_index=False), work / "events_0.parquet")

    spawned = []
    original = runner.launch_worker_subprocess

    def _tripwire(*a, **kw):
        spawned.append(a[0] if a else kw.get("shard_id"))
        raise AssertionError(f"worker mode tried to launch worker {spawned[-1]}")

    runner.launch_worker_subprocess = _tripwire
    old_argv = sys.argv
    sys.argv = [
        "run_p2_parallel.py", "--run-worker", "0",
        "--work-dir", str(work), "--records", str(work / "records.parquet"),
        "--candidates", str(inp / "candidates.parquet"),
        "--out", str(tmp_path / "f.parquet"), "--metrics", str(tmp_path / "m.json"),
    ]
    try:
        rc = runner.main()
    finally:
        sys.argv = old_argv
        runner.launch_worker_subprocess = original

    assert spawned == [], f"fork bomb: worker mode launched {spawned}"
    assert rc == 0, f"worker mode exited {rc}"

    # It must have produced exactly one complete, validated shard plus a marker.
    out = work / "feat_0.parquet"
    assert out.exists(), "worker produced no output"
    assert pq.ParquetFile(out).metadata.num_rows == len(cands)
    assert (work / "done_0.json").exists(), "worker wrote no completion marker"
    assert not list(work.glob("*.tmp")), "worker left a temp file behind"


def test_completed_shard_is_skipped_on_resume(runner, tmp_path):
    """A shard with a valid marker and matching parquet must not be rebuilt.

    This is what makes the 6-shard run survivable: if shard 3 dies, shards 0-2 are not
    thrown away, and a marker that disagrees with the file on disk is ignored so a
    half-written output can never pass as finished.
    """
    inp, cands, _evs, _records = _make_dataset(tmp_path, n_s1=20, cand_per_s1=3)
    work = tmp_path / "work_resume"
    work.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(cands, preserve_index=False), work / "cand_0.parquet")
    pq.write_table(pa.Table.from_pandas(pq.read_table(inp / "records.parquet").to_pandas(),
                                       preserve_index=False), work / "records.parquet")
    evs = pq.read_table(inp / "retrieval_events.parquet").to_pandas()
    pq.write_table(pa.Table.from_pandas(evs, preserve_index=False), work / "events_0.parquet")

    argv_base = [
        "run_p2_parallel.py", "--work-dir", str(work),
        "--records", str(work / "records.parquet"),
        "--candidates", str(inp / "candidates.parquet"),
        "--out", str(tmp_path / "f.parquet"), "--metrics", str(tmp_path / "m.json"),
    ]
    old_argv = sys.argv
    sys.argv = argv_base + ["--run-worker", "0"]
    try:
        assert runner.main() == 0
    finally:
        sys.argv = old_argv
    stamp = (work / "feat_0.parquet").stat().st_mtime_ns

    # Valid marker -> skipped.
    assert runner.shard_is_complete(0, work) is not None
    # Marker disagreeing with the file -> not trusted.
    info = json.loads((work / "done_0.json").read_text())
    info["rows"] = info["rows"] + 1
    (work / "done_0.json").write_text(json.dumps(info))
    assert runner.shard_is_complete(0, work) is None, "stale marker was trusted"
    (work / "done_0.json").write_text(json.dumps({**info, "rows": info["rows"] - 1}))

    sys.argv = argv_base + ["--run-worker", "0"]
    try:
        assert runner.main() == 0
    finally:
        sys.argv = old_argv
    assert (work / "feat_0.parquet").stat().st_mtime_ns != stamp, "rebuilt a complete shard"
