"""Phase 7 tests: bounded memory and in-memory/streaming equivalence for P1 blocking.

The refactor's central claim is that routing every route through a sink changes *where*
results are accumulated, never *what* they are. These tests hold that claim to account on
real fixture data:

* the streaming path reproduces the in-memory candidates and events exactly;
* candidate/event output is invariant to the block and buffer sizes in use;
* the streamed artifacts keep the same 7-column contract as before;
* a live probe confirms the route buffer never retains more than its threshold.

The 300K real-data slice in ``scripts/run_p1_slice.py`` is the authoritative memory
measurement; the RSS check here is only a coarse regression guard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.blocking.candidate_generation import generate_candidates
from src.blocking.spill import CANONICAL_CANDIDATE_COLUMNS

DATA = Path(__file__).resolve().parents[1] / "dataset" / "train"
EVENT_COLUMNS = [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "route",
    "rank",
    "score",
]
ALL_ROUTES = [
    "exact_name",
    "tfidf_name",
    "rare_token_name",
    "tfidf_address",
    "numeric_address",
    "rare_token_address",
    "reverse_retrieval",
]

#: Thresholds are deliberately tiny so a few hundred rows produce many flushes. The
#: streaming path is then exercised far harder than at the production settings of 20/500.
STRESS_CONFIG = {
    "enabled_routes": ALL_ROUTES,
    "exact_name_max_block": 2,
    "tfidf_name_top_k": 3,
    "rare_name_top_k": 3,
    "rare_name_max_block": 2,
    "tfidf_addr_top_k": 3,
    "numeric_top_k": 3,
    "numeric_max_block": 2,
    "rare_addr_top_k": 3,
    "rare_addr_max_block": 2,
    "reverse_top_k": 2,
    "record_buffer": 100,
    "dedup_buckets": 8,
}


def _load(n_rows: int = 300):
    frames = []
    for name in ("s1_sample.tsv", "s2_sample.tsv", "s3_sample.tsv"):
        frame = pd.read_csv(DATA / name, sep="\t", dtype=str, keep_default_na=False, nrows=n_rows)
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def small_pool():
    return _load(300)


def _run_in_memory(pool, **overrides):
    """Run the legacy in-memory path and return (raw_candidates, events).

    The in-memory path returns the *legacy* candidate shape
    ``[pair_key, s1_id, candidate_id, candidate_source, route, rank, score]``.
    ``docs/schemas.md`` Section 6 requires the canonical shape with
    ``n_routes/best_rank/best_score``, and
    :func:`~src.utils.adapters.reconcile_candidates_schema` is the documented bridge.
    Equivalence is therefore asserted against the reconciled reference, not against the
    raw legacy frame -- comparing them directly would be comparing two different schemas.
    """
    config = {**STRESS_CONFIG, **overrides, "spill_dir": None}
    candidates, events = generate_candidates(*pool, config=config, return_events=True)
    return candidates, events


def _reconcile(candidates, events) -> pd.DataFrame:
    from src.utils.adapters import reconcile_candidates_schema

    return reconcile_candidates_schema(candidates, events)


def _run_streaming(pool, tmp_path, **overrides):
    config = {
        **STRESS_CONFIG,
        **overrides,
        "spill_dir": str(tmp_path / "spill"),
        "spill_output_path": str(tmp_path / "candidates.parquet"),
        "spill_events_path": str(tmp_path / "events.parquet"),
        "reset_spill_dir": True,
    }
    generate_candidates(*pool, config=config, return_events=True)
    return pd.read_parquet(config["spill_output_path"]), pd.read_parquet(
        config["spill_events_path"]
    )


# --------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------- #


def test_streaming_reproduces_in_memory_candidates(small_pool, tmp_path) -> None:
    """The streamed artifact must equal the reconciled in-memory result, row for row."""
    mem_cand, mem_events = _run_in_memory(small_pool)
    reference = _reconcile(mem_cand, mem_events)
    str_cand, _ = _run_streaming(small_pool, tmp_path)

    assert len(reference) > 0, "the fixture must produce candidates for the test to mean anything"
    assert list(str_cand.columns) == list(CANONICAL_CANDIDATE_COLUMNS)
    assert len(str_cand) == len(reference)
    pd.testing.assert_frame_equal(
        str_cand.reset_index(drop=True), reference.reset_index(drop=True)
    )


def test_in_memory_path_returns_the_legacy_candidate_shape(small_pool) -> None:
    """Pin the legacy/canonical split so it cannot drift unnoticed.

    The in-memory path hands back ``route``/``rank``/``score``; the artifact contract is
    ``n_routes``/``best_rank``/``best_score``. If a future change made the in-memory path
    emit the canonical names directly, the adapter would start silently mis-mapping
    columns -- so the divergence is asserted rather than assumed.
    """
    mem_cand, _ = _run_in_memory(small_pool)
    assert list(mem_cand.columns) == [
        "pair_key",
        "s1_id",
        "candidate_id",
        "candidate_source",
        "route",
        "rank",
        "score",
    ]


def test_streaming_reproduces_in_memory_events_ignoring_row_order(small_pool, tmp_path) -> None:
    """Events are order-independent: they land in 8 buckets, not in generation order.

    The sort key is the full row, so this is an exact content comparison rather than a
    containment or row-count check.
    """
    _, mem_events = _run_in_memory(small_pool)
    _, str_events = _run_streaming(small_pool, tmp_path)

    assert len(mem_events) > 0
    assert list(str_events.columns) == EVENT_COLUMNS

    key = EVENT_COLUMNS
    lhs = str_events.sort_values(key).reset_index(drop=True)
    rhs = mem_events.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(lhs, rhs)


def test_streaming_candidates_are_globally_ordered_by_s1_id(small_pool, tmp_path) -> None:
    """Output is one ascending ``s1_id`` stream, not 8 concatenated per-bucket runs.

    This is the property that lets downstream consumers stream the artifact instead of
    sorting it, so it is asserted on the written file rather than in memory.
    """
    str_cand, _ = _run_streaming(small_pool, tmp_path)
    assert str_cand["s1_id"].is_monotonic_increasing
    assert str_cand["s1_id"].nunique() > 8, "need more s1_ids than buckets for this to bite"


def test_candidate_and_event_pair_keys_agree(small_pool, tmp_path) -> None:
    """Every candidate must be backed by at least one retrieval event."""
    str_cand, str_events = _run_streaming(small_pool, tmp_path)
    assert set(str_cand["pair_key"]) <= set(str_events["pair_key"])
    assert len(str_events) >= len(str_cand), "candidates cannot outnumber their events"


def test_candidate_metadata_respects_the_aggregate_contract(small_pool, tmp_path) -> None:
    """``best_rank`` is the min rank *across routes*, so it is deliberately not dense.

    A pair retrieved by two routes at ranks 1 and 3 has ``best_rank == 1``, which means
    several candidates for one ``s1_id`` can share it. The dense per-s1 ordering lives in
    the ``rank`` column of ``canonicalize_frame`` and is not part of the artifact, so
    asserting density here would be asserting a property the schema does not have.
    """
    str_cand, _ = _run_streaming(small_pool, tmp_path)
    assert (str_cand["best_rank"] >= 1).all()
    assert (str_cand["n_routes"] >= 1).all()
    assert (str_cand["n_routes"] <= len(ALL_ROUTES)).all()
    assert str_cand["best_score"].notna().all()
    assert str_cand["best_score"].between(0.0, 1.0).all()
    assert str_cand["pair_key"].is_unique


# --------------------------------------------------------------------------- #
# Determinism across block configurations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,overrides",
    [
        ("fine_decomposition", {
            "tfidf_name_candidate_chunk": 25,
            "tfidf_addr_candidate_chunk": 25,
            "tfidf_query_block": 10,
            "reverse_index_block": 20,
        }),
        ("coarse_decomposition", {
            "tfidf_name_candidate_chunk": 100_000,
            "tfidf_addr_candidate_chunk": 100_000,
            "tfidf_query_block": 1_000,
            "reverse_index_block": 100_000,
        }),
        ("tiny_buffer_many_buckets", {"record_buffer": 7, "dedup_buckets": 64}),
        ("big_buffer_few_buckets", {"record_buffer": 5_000, "dedup_buckets": 2}),
    ],
)
def test_output_is_invariant_to_decomposition(small_pool, tmp_path, label, overrides) -> None:
    """The approved tie-break must make output independent of *decomposition*.

    "Decomposition" here means how the work is cut up -- candidate block, query block,
    reverse index block, buffer size and bucket count -- not the recall caps. Before the
    tie-break fix, which candidates survived the k-th rank depended on the block size, so
    a different block silently changed the candidate set. Pinning that here is what stops
    the regression from returning.
    """
    mem_cand, mem_events = _run_in_memory(small_pool)
    reference = _reconcile(mem_cand, mem_events)
    str_cand, _ = _run_streaming(small_pool, tmp_path / label, **overrides)

    assert len(str_cand) == len(reference), f"{label}: row count changed with decomposition"
    pd.testing.assert_frame_equal(
        str_cand.reset_index(drop=True),
        reference.reset_index(drop=True),
        obj=f"{label} candidates differ from the in-memory reference",
    )


@pytest.mark.parametrize("narrow,wider", [(1, 4), (2, 500)])
def test_max_block_size_is_a_recall_cap_not_a_decomposition_detail(small_pool, tmp_path, narrow, wider) -> None:
    """``max_block_size`` is *meant* to change recall, so it must not be pinned.

    ``retrieve_exact_name`` documents the parameter as "Maximum candidates per
    normalized-name bucket" and truncates the bucket with it, so widening the cap can only
    add pairs -- never remove one. This is asserted on Route 1 alone, which has no
    ``top_k``; for the top-k routes a wider cap re-runs the *selection*, so the outputs are
    related but not nested. Documenting the asymmetry stops anyone later reading the
    invariance test above and assuming these parameters should be invariant too.
    """
    only_exact = {"enabled_routes": ["exact_name"]}
    small, _ = _run_streaming(
        small_pool, tmp_path / f"n{narrow}", exact_name_max_block=narrow, **only_exact
    )
    large, _ = _run_streaming(
        small_pool, tmp_path / f"w{wider}", exact_name_max_block=wider, **only_exact
    )

    small_keys = set(small["pair_key"])
    large_keys = set(large["pair_key"])
    assert small_keys, "Route 1 found nothing on the fixture, so the test would be vacuous"
    assert small_keys <= large_keys, (
        f"widening the cap from {narrow} to {wider} dropped pairs the narrower cap had found"
    )
    assert len(large) >= len(small)


# --------------------------------------------------------------------------- #
# Bounded retention, measured live
# --------------------------------------------------------------------------- #


def test_route_buffer_never_exceeds_its_threshold(small_pool, tmp_path, monkeypatch) -> None:
    """Instrument the real writer and record the largest batch any flush produced.

    This observes the actual production code path rather than a stand-in, so it would
    catch a route that bypasses ``RecordBuffer`` and accumulates its own list.
    """
    from src.blocking import spill as spill_mod

    observed: list[int] = []
    original = spill_mod.EventSpillWriter.write_records

    def spy(self, records):
        observed.append(len(records))
        return original(self, records)

    monkeypatch.setattr(spill_mod.EventSpillWriter, "write_records", spy)

    config = {
        **STRESS_CONFIG,
        "record_buffer": 50,
        "spill_dir": str(tmp_path / "spill"),
        "spill_output_path": str(tmp_path / "candidates.parquet"),
        "spill_events_path": str(tmp_path / "events.parquet"),
        "reset_spill_dir": True,
    }
    generate_candidates(*small_pool, config=config, return_events=True)

    assert observed, "no flush happened, so retention was never bounded"
    assert max(observed) <= 50, f"a flush carried {max(observed)} rows, above the threshold of 50"


def test_streaming_peak_rss_stays_modest(small_pool, tmp_path) -> None:
    """Coarse guard only; the authoritative figure comes from the 300K slice run."""
    import resource

    _run_streaming(small_pool, tmp_path)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert peak_mb < 4096, f"peak RSS {peak_mb:.0f} MB is far above what a 300-row fixture needs"


def test_routes_return_empty_frames_when_a_sink_is_configured(small_pool, tmp_path) -> None:
    """Documented contract: a streaming route returns no rows, because the rows went to disk.

    A route returning rows *and* writing them would double-count on merge.
    """
    config = {
        **STRESS_CONFIG,
        "spill_dir": str(tmp_path / "spill"),
        "spill_output_path": str(tmp_path / "candidates.parquet"),
        "spill_events_path": str(tmp_path / "events.parquet"),
        "reset_spill_dir": True,
    }
    result = generate_candidates(*small_pool, config=config, return_events=False)
    assert len(result) == 0, "the orchestrator must not return rows it already streamed to disk"
    assert Path(config["spill_output_path"]).exists()
    assert Path(config["spill_events_path"]).exists()
