"""Tests for the parallel-worker finalization path (scripts/run_p1_parallel.py).

Two properties are load-bearing and neither is obvious:

1. **A pre-fitted vectorizer is interchangeable with the route's internal fit.**
   ``run_p1_parallel.py`` fits the ``CharTfidfVectorizer`` itself and hands it to the
   route through the existing ``vectorizer=`` parameter, which is what makes S1 sharding
   admissible for the two TF-IDF routes. If the fit differs from what the route would have
   built -- and it did differ, because the routes fit on *normalized* S1 text -- the
   vocabulary changes and the retrieval output silently changes with it.

2. **Finalizing a union of spills equals finalizing their concatenation.**
   The bounded finalizer canonicalizes one hash bucket at a time and then k-way merges the
   sorted buckets, so it never holds more than one bucket plus one batch per bucket. It
   must still reproduce ``iter_canonical_candidates_merged`` exactly, including global
   ``s1_id`` ordering, when the same route's events arrive in several separate spills.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.blocking.spill import (
    CANONICAL_CANDIDATE_COLUMNS,
    EventSpillWriter,
    RecordBuffer,
    canonicalize_frame,
    iter_canonical_candidates_merged,
)
from src.blocking.tfidf_address import retrieve_tfidf_address
from src.blocking.tfidf_name import retrieve_tfidf_name

REPO = Path(__file__).resolve().parents[1]
EVENT_COLUMNS = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "route",
    "rank",
    "score",
]


def _load_driver():
    """Import scripts/run_p1_parallel.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location(
        "p1_parallel", REPO / "scripts" / "run_p1_parallel.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["p1_parallel"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


def _frames(n_s1: int, n_pool: int, seed: int = 7):
    """Small but real-shaped frames: shared name/address stems so retrieval is non-empty."""
    stems = [f"{w} corp holdings" for w in ("alpha", "bravo", "delta", "echo", "foxtrot")]
    k = len(stems)
    s1 = pd.DataFrame(
        {
            "entity_id": [f"S1-{i:05d}" for i in range(n_s1)],
            "business_name": [f"{stems[i % k]} {i}" for i in range(n_s1)],
            "business_address": [f"{i % 97} {stems[(i + 3) % k]} street" for i in range(n_s1)],
        }
    )
    pool = pd.DataFrame(
        {
            "entity_id": [f"S2-{i:05d}" for i in range(n_pool)],
            "business_name": [f"{stems[i % k]} {i}" for i in range(n_pool)],
            "business_address": [f"{i % 97} {stems[(i + 3) % k]} street" for i in range(n_pool)],
        }
    )
    return s1, pd.concat([pool, pool.assign(entity_id=[f"S3-{i:05d}" for i in range(n_pool)])],
                        ignore_index=True)


class CollectingSink:
    """Captures emitted records; mirrors RecordBuffer's append/emit/close protocol."""

    def __init__(self) -> None:
        self.rows = []

    def append(self, record: dict) -> None:
        self.rows.append(record)

    def write_records(self, records) -> None:
        self.rows.extend(records)

    emit = write_records

    def close(self) -> None:
        return None

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.DataFrame(self.rows, columns=EVENT_COLUMNS)


# --------------------------------------------------------------------------- #
# 1. pre-fitted vectorizer equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "route,call,colkw",
    [
        ("tfidf_address", retrieve_tfidf_address, "addr_col"),
        ("tfidf_name", retrieve_tfidf_name, "name_col"),
    ],
)
def test_prefixed_vectorizer_matches_internal_fit(driver, route, call, colkw):
    """The driver's own fit must equal the route's internal fit, bit for bit."""
    s1, pool = _frames(60, 150)
    kwargs = dict(driver.ROUTE_DEFAULTS[route])
    s2, s3 = pool.iloc[:150], pool.iloc[150:].reset_index(drop=True)

    internal = CollectingSink()
    call(s1, s2, s3, id_col="entity_id", **{colkw: "business_address" if colkw == "addr_col" else "business_name"},
         sink=internal, **kwargs)

    shared = CollectingSink()
    vec = driver.fit_shared_vectorizer(route, s1, s2, s3)
    call(s1, s2, s3, id_col="entity_id", **{colkw: "business_address" if colkw == "addr_col" else "business_name"},
         sink=shared, vectorizer=vec, **kwargs)

    assert not internal.frame().empty
    assert internal.frame().equals(shared.frame())


def test_s1_sharding_is_lossless_for_vectorizer_routes(driver):
    """A contiguous S1 split, each half given a vectorizer fitted on the FULL S1, unions
    back to the un-sharded result."""
    s1, pool = _frames(80, 200)
    s2, s3 = pool.iloc[:200], pool.iloc[200:].reset_index(drop=True)
    kwargs = dict(driver.ROUTE_DEFAULTS["tfidf_address"])

    whole = CollectingSink()
    retrieve_tfidf_address(s1, s2, s3, id_col="entity_id", addr_col="business_address",
                           sink=whole, **kwargs)

    half = len(s1) // 2
    pieces = []
    for lo, hi in ((0, half), (half, len(s1))):
        sink = CollectingSink()
        vec = driver.fit_shared_vectorizer("tfidf_address", s1, s2, s3)  # fitted on FULL s1
        retrieve_tfidf_address(driver.s1_shard(s1, lo, hi), s2, s3, id_col="entity_id",
                               addr_col="business_address", sink=sink, vectorizer=vec, **kwargs)
        pieces.append(sink.frame())

    key = ["s1_id", "candidate_id", "route", "rank", "score"]
    a = whole.frame().sort_values(key).reset_index(drop=True)
    b = pd.concat(pieces, ignore_index=True).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# 2. bounded finalizer == reference
# --------------------------------------------------------------------------- #
def _spill_events(spill_dir: Path, frame: pd.DataFrame, n_buckets: int = 8) -> None:
    writer = EventSpillWriter(spill_dir, n_buckets=n_buckets)
    buf = RecordBuffer(writer, max_records=64)
    for rec in frame.to_dict("records"):
        buf.append(rec)
    buf.close()
    writer.close()


def _run_finalize(driver, spill_dirs, out_dir, canon_dir, **kw):
    class Args:
        pass

    a = Args()
    a.spill_dirs = [str(p) for p in spill_dirs]
    a.out_dir = str(out_dir)
    a.canon_dir = str(canon_dir)
    a.buckets = 8
    a.merge_batch = 7
    a.flush_rows = 11
    a.keep_canon = False
    for key, value in kw.items():
        setattr(a, key, value)
    return driver.run_finalize(a)


def _reference_candidates(spill_dir: Path, n_buckets: int = 8) -> pd.DataFrame:
    frames = list(iter_canonical_candidates_merged(spill_dir, n_buckets))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=CANONICAL_CANDIDATE_COLUMNS
    )


def test_bounded_finalize_matches_reference_across_split_spills(driver, tmp_path):
    """One route's events split across three spills must finalize identically."""
    s1, pool = _frames(50, 120)
    s2, s3 = pool.iloc[:120], pool.iloc[120:].reset_index(drop=True)
    sink = CollectingSink()
    retrieve_tfidf_name(s1, s2, s3, id_col="entity_id", name_col="business_name",
                        sink=sink, **driver.ROUTE_DEFAULTS["tfidf_name"])
    events = sink.frame()
    assert len(events) > 0

    thirds = np.array_split(np.arange(len(events)), 3)
    spills = []
    for i, idx in enumerate(thirds):
        d = tmp_path / f"spill_{i}"
        _spill_events(d, events.iloc[idx].reset_index(drop=True))
        spills.append(d)

    out = tmp_path / "out"
    _run_finalize(driver, spills, out, tmp_path / "canon")
    got = pd.read_parquet(out / "candidates.parquet")

    # Reference: the same events in a single spill, through the original code path.
    single = tmp_path / "single"
    _spill_events(single, events)
    ref = _reference_candidates(single)

    assert list(got.columns) == list(CANONICAL_CANDIDATE_COLUMNS)
    assert len(got) == len(ref)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), ref[list(CANONICAL_CANDIDATE_COLUMNS)].reset_index(drop=True)
    )
    assert got["s1_id"].is_monotonic_increasing
    assert not got.duplicated(subset=["pair_key", "s1_id", "candidate_id"]).any()


def test_bounded_finalize_dedupes_overlapping_spills(driver, tmp_path):
    """A pair emitted by two workers must collapse to one candidate row."""
    events = pd.DataFrame(
        [
            dict(pair_key="S1-1::S2-1", s1_id="S1-1", candidate_id="S2-1",
                 candidate_source="S2", route="r", rank=2, score=0.4),
            dict(pair_key="S1-1::S2-1", s1_id="S1-1", candidate_id="S2-1",
                 candidate_source="S2", route="r", rank=1, score=0.9),
            dict(pair_key="S1-2::S2-2", s1_id="S1-2", candidate_id="S2-2",
                 candidate_source="S2", route="r", rank=1, score=0.5),
        ]
    )
    a, b = tmp_path / "a", tmp_path / "b"
    _spill_events(a, events)                       # worker A saw everything
    _spill_events(b, events.iloc[:2].reset_index(drop=True))  # worker B overlapped

    out = tmp_path / "out"
    _run_finalize(driver, [a, b], out, tmp_path / "canon")
    got = pd.read_parquet(out / "candidates.parquet")

    assert len(got) == 2
    row = got[got["s1_id"] == "S1-1"].iloc[0]
    assert row["best_rank"] == 1
    assert row["best_score"] == pytest.approx(0.9)
    assert row["n_routes"] == 1


def test_bounded_finalize_event_schema_and_order(driver, tmp_path):
    events = pd.DataFrame(
        [
            dict(pair_key=f"S1-{i:03d}::S2-{i:03d}", s1_id=f"S1-{i:03d}",
                 candidate_id=f"S2-{i:03d}", candidate_source="S2", route="r",
                 rank=1, score=0.5)
            for i in (5, 1, 3, 0, 4, 2)
        ]
    )
    spill = tmp_path / "s"
    _spill_events(spill, events)
    out = tmp_path / "out"
    _run_finalize(driver, [spill], out, tmp_path / "canon")

    ev = pd.read_parquet(out / "retrieval_events.parquet")
    assert list(ev.columns) == EVENT_COLUMNS
    assert len(ev) == len(events)
    got = pd.read_parquet(out / "candidates.parquet")
    assert list(got["s1_id"]) == sorted(got["s1_id"])


def test_canonicalize_frame_keeps_min_rank_for_merge_key():
    """Pass 2 sorts on `rank`; canonicalize_frame must preserve it as the best rank."""
    events = pd.DataFrame(
        [
            dict(pair_key="a::1", s1_id="a", candidate_id="1", candidate_source="S2",
                 route="r", rank=3, score=0.1),
            dict(pair_key="a::1", s1_id="a", candidate_id="1", candidate_source="S2",
                 route="r", rank=1, score=0.7),
        ]
    )
    canon = canonicalize_frame(events)
    assert canon["rank"].tolist() == [1]
    assert canon["best_rank"].tolist() == [1]
    assert canon["best_score"].tolist() == [pytest.approx(0.7)]
