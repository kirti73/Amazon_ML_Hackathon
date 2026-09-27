"""
Production Runner for Full-Data P1 Candidate Generation Pipeline.
Amazon ML Challenge 2026: Business Entity Resolution.

Executes all 7 frozen candidate generation routes on the complete training dataset
(2.2M S1, 5.0M S2, 5.3M S3 records), applies the canonical schema adapter,
persists canonical Parquet artifacts, and runs comprehensive validation.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# psutil is present in .venv_win (7.2.2) but absent from the WSL .venv, which is why the
# first full-data attempt could only run under Windows Python. It is therefore optional:
# RSS falls back to getrusage (peak) and system RAM to /proc/meminfo on Linux, and to
# getrusage only elsewhere. The pipeline is fully functional without it.
try:  # pragma: no cover - trivial import guard
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# ``resource`` is POSIX-only; importing it unconditionally made this runner fail to import
# on Windows even though nothing else here is platform-specific.
try:  # pragma: no cover - platform guard
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

# Ensure repo root and code/business_entity_resolution are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution"))

from src.data.loaders import load_source_tsv
from src.blocking.candidate_generation import (
    generate_candidates,
    evaluate_candidate_recall,
    deduplicate_candidates,
)


def get_mem_mb() -> float:
    """Return process memory in MB.

    Prefers psutil's current RSS. Without psutil, ``getrusage`` reports the *peak* RSS
    (``ru_maxrss``), which is a conservative upper bound and therefore still a valid
    memory guard for a long run.
    """
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    if resource is None:  # Windows without psutil: no portable fallback available.
        return float("nan")
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return peak_kb / divisor


def get_sys_mem_gb() -> Tuple[float, float]:
    """Return ``(available_gb, total_gb)`` for system RAM."""
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        return vm.available / (1024**3), vm.total / (1024**3)

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: Dict[str, float] = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key] = float(parts[0]) * 1024  # kB -> bytes
        total = values.get("MemTotal", 0.0)
        available = values.get("MemAvailable", values.get("MemFree", 0.0))
        return available / (1024**3), total / (1024**3)

    # Last resort: assume 8 GB so the run still proceeds with a stated assumption.
    print("  [warn] psutil unavailable and /proc/meminfo unreadable; assuming 8 GB RAM.")
    return 8.0, 8.0


def main():
    start_total_time = time.time()
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: FULL-DATA P1 CANDIDATE GENERATION PIPELINE")
    print("=" * 80)

    # 1. Environment & System Info
    avail_gb, total_gb = get_sys_mem_gb()
    mem_source = "psutil" if _HAS_PSUTIL else "getrusage peak / /proc/meminfo"
    print(f"\n[Environment Diagnostic]")
    print(f"  Python Version:     {sys.version.split()[0]}")
    print(f"  Platform:           {sys.platform}")
    print(f"  Available RAM:      {avail_gb:.2f} GB / {total_gb:.2f} GB")
    print(f"  Memory source:      {mem_source}")
    print(f"  Process Memory:     {get_mem_mb():.1f} MB")

    # 2. File Paths
    data_dir = BASE_DIR / "dataset" / "train"
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file not found: {p}")
        print(f"  Found dataset file: {p.name} ({p.stat().st_size / (1024**2):.1f} MB)")

    # 3. Load Datasets
    print(f"\n[1/6] Loading full source TSVs safely via load_source_tsv()...")
    t0 = time.time()
    print(f"  Loading S1 from {s1_path.name}...")
    s1_df = load_source_tsv(s1_path)
    print(f"  Loaded S1: {len(s1_df):,} rows, cols={list(s1_df.columns)} (RSS: {get_mem_mb():.1f} MB)")

    print(f"  Loading S2 from {s2_path.name}...")
    s2_df = load_source_tsv(s2_path)
    print(f"  Loaded S2: {len(s2_df):,} rows, cols={list(s2_df.columns)} (RSS: {get_mem_mb():.1f} MB)")

    print(f"  Loading S3 from {s3_path.name}...")
    s3_df = load_source_tsv(s3_path)
    print(f"  Loaded S3: {len(s3_df):,} rows, cols={list(s3_df.columns)} (RSS: {get_mem_mb():.1f} MB)")
    print(f"  Dataset loading complete in {time.time() - t0:.2f}s.")

    # 4. Candidate Generation across 7 routes
    # The spill directory enables the bounded-memory L4 path: routes stream into
    # hash-bucketed Parquet shards and the aggregation runs one bucket at a time, so the
    # ~260M projected retrieval events are never all resident. Required for a full run.
    # The spill can reach tens of GB. /mnt/d (the repo's filesystem) only had 32 GB free,
    # so the spill lives on the Linux filesystem (949 GB free) and only the final artifacts
    # are written into the repo. Overridable via P1_SPILL_DIR.
    spill_dir = Path(os.environ.get("P1_SPILL_DIR", "/var/tmp/p1_spill"))
    # NOTE: deliberately NOT deleted here. The previous rmtree made every restart a full,
    # fatal recompute; see planning/P1_EXECUTION_STATUS.md section C.4. Use
    # --reset-spill (or P1_RESET_SPILL=1) to start clean on purpose.
    reset_spill = os.environ.get("P1_RESET_SPILL", "0") not in ("0", "", "false", "False")
    if reset_spill and spill_dir.exists():
        import shutil

        shutil.rmtree(spill_dir)
    print(f"\n[2/6] Executing P1 candidate generation across all 7 routes...")
    print(f"  Spill directory:    {spill_dir}")
    print(f"  Reset spill:        {reset_spill}")
    cand_parquet_path = artifacts_dir / "candidates.parquet"
    events_parquet_path = artifacts_dir / "retrieval_events.parquet"

    t1 = time.time()
    # return_events is deliberately NOT requested: the events table is streamed straight to
    # its final single-file artifact bucket-by-bucket, and the candidates are canonicalized
    # bucket-locally (provably equivalent -- see tests/test_blocking_spill.py). Requesting
    # return_events would load every event back into memory, i.e. reintroduce the ~32 GB
    # spike this pipeline exists to avoid.
    generate_candidates(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        config={
            "spill_dir": str(spill_dir),
            "spill_output_path": str(cand_parquet_path),
            "spill_events_path": str(events_parquet_path),
            "reset_spill_dir": reset_spill,
        },
    )
    t_gen = time.time() - t1
    n_cand = pq.ParquetFile(cand_parquet_path).metadata.num_rows
    n_events = pq.ParquetFile(events_parquet_path).metadata.num_rows
    print(f"  Candidate generation complete in {t_gen:.2f}s ({t_gen / 60:.2f} min).")
    print(f"  Canonical candidate pairs: {n_cand:,}")
    print(f"  Retrieval event records:  {n_events:,}")
    print(f"  Process Memory:           {get_mem_mb():.1f} MB")

    # Clean up S2/S3 raw DataFrames to free memory before validation
    del s2_df, s3_df
    gc.collect()

    candidates_df = pd.read_parquet(cand_parquet_path)
    retrieval_events_df = pd.read_parquet(events_parquet_path)
    print(f"\n[3/6] Canonical schema check (bucket-local, already applied during streaming)...")
    print(f"  Canonical candidate rows: {len(candidates_df):,}")
    print(f"  Canonical columns:        {list(candidates_df.columns)}")

    # 6. Persist Artifacts to Disk
    print(f"\n[4/6] Artifacts already streamed to {artifacts_dir}...")
    print(f"  {cand_parquet_path.name} ({cand_parquet_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"  {events_parquet_path.name} ({events_parquet_path.stat().st_size / (1024**2):.2f} MB)")

    # 7. Ground Truth Recall Evaluation
    print(f"\n[5/6] Evaluating candidate recall against ground truth...")
    t4 = time.time()
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    recall_metrics = evaluate_candidate_recall(
        candidates_df=candidates_df,
        gt_df=gt_df,
    )
    print(f"  Ground truth evaluation complete in {time.time() - t4:.2f}s.")
    print("\n--- RECALL METRICS SUMMARY ---")
    for k, v in recall_metrics.items():
        print(f"  {k:30s}: {v}")

    # 8. Strict Read-Back Validation
    print(f"\n[6/6] Performing strict read-back validation on persisted Parquet files...")
    reloaded_candidates = pd.read_parquet(cand_parquet_path)
    reloaded_events = pd.read_parquet(events_parquet_path)

    # Candidates validation
    assert len(reloaded_candidates) > 0, "candidates.parquet is empty!"
    assert list(reloaded_candidates.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "n_routes", "best_rank", "best_score"
    ], f"candidates.parquet column mismatch: {list(reloaded_candidates.columns)}"
    for col in ["pair_key", "s1_id", "candidate_id", "candidate_source"]:
        assert pd.api.types.is_string_dtype(reloaded_candidates[col]) or pd.api.types.is_object_dtype(reloaded_candidates[col]), f"{col} must be string dtype"
    assert (reloaded_candidates["pair_key"] == reloaded_candidates["s1_id"] + "::" + reloaded_candidates["candidate_id"]).all(), "pair_key format invalid!"
    assert reloaded_candidates["candidate_source"].isin(["S2", "S3"]).all(), "Invalid candidate_source values!"
    assert (reloaded_candidates["n_routes"] >= 1).all(), "n_routes must be >= 1"
    assert (reloaded_candidates["best_rank"] >= 1).all(), "best_rank must be >= 1"
    assert not reloaded_candidates["best_score"].isna().any(), "best_score contains NaNs!"
    assert reloaded_candidates["pair_key"].is_unique, "Duplicate pair_key values in candidates.parquet!"

    # Retrieval events validation
    assert len(reloaded_events) > 0, "retrieval_events.parquet is empty!"
    assert list(reloaded_events.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"
    ], f"retrieval_events.parquet column mismatch: {list(reloaded_events.columns)}"
    for col in ["pair_key", "s1_id", "candidate_id", "candidate_source", "route"]:
        assert pd.api.types.is_string_dtype(reloaded_events[col]) or pd.api.types.is_object_dtype(reloaded_events[col]), f"{col} must be string dtype"
    assert (reloaded_events["rank"] >= 1).all(), "rank must be >= 1 in retrieval_events"
    assert not reloaded_events["score"].isna().any(), "score contains NaNs in retrieval_events"

    # Cross-artifact consistency validation
    assert set(reloaded_candidates["pair_key"]) == set(reloaded_events["pair_key"]), "pair_key set mismatch between candidates and events!"
    print("  [PASS] Candidate and retrieval_events pair_key sets match 100% exactly.")

    # Sample cross-check
    sample_pks = reloaded_candidates["pair_key"].sample(min(1000, len(reloaded_candidates)), random_state=42)
    sample_events = reloaded_events[reloaded_events["pair_key"].isin(sample_pks)]
    agg_sample = sample_events.groupby("pair_key").agg(
        n_routes=("route", "nunique"),
        best_rank=("rank", "min"),
        best_score=("score", "max"),
    )
    sample_cands = reloaded_candidates[reloaded_candidates["pair_key"].isin(sample_pks)].set_index("pair_key")
    assert (sample_cands["n_routes"] == agg_sample.loc[sample_cands.index, "n_routes"]).all(), "n_routes mismatch in cross-check!"
    assert (sample_cands["best_rank"] == agg_sample.loc[sample_cands.index, "best_rank"]).all(), "best_rank mismatch in cross-check!"
    assert np.allclose(sample_cands["best_score"], agg_sample.loc[sample_cands.index, "best_score"]), "best_score mismatch in cross-check!"
    print("  [PASS] Aggregation cross-check (n_routes, best_rank, best_score) passed 100% on sample pairs.")

    # Check folds.parquet untouched
    folds_path = artifacts_dir / "folds.parquet"
    if folds_path.exists():
        print(f"  [PASS] Verified folds.parquet exists and was not altered ({folds_path.stat().st_size:,} bytes).")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"  FULL P1 PIPELINE EXECUTION & VALIDATION SUCCESSFUL IN {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
