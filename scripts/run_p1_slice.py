"""Bounded real-data slice runner for P1 candidate generation.

Why this exists
---------------
The full-data P1 run is memory-fatal in its current form and computationally infeasible
(~81 h per TF-IDF route, ~162 h for two). Running it again would crash WSL. This script
exists so the pipeline can be exercised on **real** data at a bounded scale, which:

  1. proves the bounded-memory streaming path works on real records, not just fixtures;
  2. measures true per-route wall-clock, so the full-run projection is based on data
     rather than on a similarity-density extrapolation;
  3. produces genuine P1 output so P2/P3 can be unblocked.

It is **not** a full run. It samples a fixed number of S1 rows plus a proportional slice of
the S2/S3 candidate pool, and writes everything to its own directory. It never touches
``artifacts/*.parquet`` or ``artifacts/folds.parquet``.

Sampling is **streaming** (``chunksize``): the full 210/489/504 MB TSVs are never held in
memory at once, which is exactly what ``scripts/make_sample.py`` does wrong.

Usage
-----
    # 300K S1 (the approved default) + proportional pool
    .venv/bin/python scripts/run_p1_slice.py

    # smaller / larger slice
    .venv/bin/python scripts/run_p1_slice.py --s1-n 50000

    # only build the slice, do not run retrieval
    .venv/bin/python scripts/run_p1_slice.py --build-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution"))

from src.blocking.candidate_generation import (  # noqa: E402
    DEFAULT_BLOCKING_CONFIG,
    generate_candidates,
)
from src.blocking.spill import (  # noqa: E402
    CANONICAL_CANDIDATE_COLUMNS,
    DEFAULT_RECORD_BUFFER,
    EventSpillWriter,
    RecordBuffer,
    resolve_spill_dir,
    write_candidate_dataset,
    write_events_stream,
    iter_canonical_candidates_merged,
    iter_spilled_buckets,
)

try:  # pragma: no cover - platform guard
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

try:  # pragma: no cover
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


DEFAULT_S1_N = 300_000
#: S2 and S3 are scaled by the *same* factor as S1 so the pool-to-query ratio of the real
#: dataset is preserved. This matters enormously: TF-IDF cost scales as O(|S1| x |pool|),
#: so shrinking the pool relative to S1 would badly understate the full-run projection.
#: Real ratios: 5,034,616/2,206,821 = 2.2813 and 5,285,603/2,206,821 = 2.3952.
S2_SCALE = 5_034_616 / 2_206_821
S3_SCALE = 5_285_603 / 2_206_821
CHUNK = 250_000
SEED = 42


def peak_rss_mb() -> float:
    """Peak RSS in MB. ``getrusage`` gives the high-water mark, which is the safe bound."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    if resource is None:
        return float("nan")
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def sys_mem_gb() -> Tuple[float, float]:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        vals: Dict[str, float] = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                vals[key] = float(parts[0]) * 1024
        return (
            vals.get("MemAvailable", 0.0) / (1024**3),
            vals.get("MemTotal", 0.0) / (1024**3),
        )
    return float("nan"), float("nan")


def stream_sample(src: Path, n: int, seed: int = SEED) -> pd.DataFrame:
    """Return ``n`` rows sampled from a TSV **without** loading the whole file.

    Uses a fixed RNG stream over per-chunk samples so the result is deterministic for a
    given ``(n, seed, source)`` regardless of chunk boundaries.
    """
    if n <= 0:
        return pd.DataFrame()
    keep: list[pd.DataFrame] = []
    have = 0
    # Two passes are avoided by sampling per chunk and stopping once we have enough.
    import numpy as np

    rng = np.random.default_rng(seed)
    for chunk in pd.read_csv(
        src,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=CHUNK,
    ):
        remaining = n - have
        if remaining <= 0:
            break
        if len(chunk) <= remaining:
            keep.append(chunk)
            have += len(chunk)
            continue
        idx = rng.choice(len(chunk), size=remaining, replace=False)
        keep.append(chunk.iloc[sorted(idx)])
        have += remaining
    if not keep:
        return pd.DataFrame()
    out = pd.concat(keep, ignore_index=True)
    return out.iloc[:n].reset_index(drop=True)


def build_slice(
    s1_n: int,
    slice_dir: Path,
    data_dir: Path,
    seed: int = SEED,
    log: Optional[list] = None,
) -> Dict[str, int]:
    """Materialize the slice TSVs + sliced ground truth. Idempotent; skips if present."""
    slice_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "s1": slice_dir / "slice_source1.tsv",
        "s2": slice_dir / "slice_source2.tsv",
        "s3": slice_dir / "slice_source3.tsv",
        "gt": slice_dir / "slice_ground_truth.tsv",
    }
    counts: Dict[str, int] = {}
    scales = {"s1": 1.0, "s2": S2_SCALE, "s3": S3_SCALE}
    for key, name in (("s1", "train_source1.tsv"), ("s2", "train_source2.tsv"), ("s3", "train_source3.tsv")):
        out = paths[key]
        if out.exists():
            counts[key] = sum(1 for _ in out.open(encoding="utf-8")) - 1
            if log is not None:
                log.append(f"  {key}: reusing existing slice ({counts[key]:,} rows)")
            continue
        t0 = time.monotonic()
        target = max(1, int(s1_n * scales[key]))
        # Derive the per-source seed from a stable digest, not built-in hash(): CPython
        # randomizes str hashing per process (PYTHONHASHSEED), so hash(key) would give a
        # different sample on every run and the slice would not be reproducible.
        offset = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
        df = stream_sample(data_dir / name, target, seed=seed + offset)
        df.to_csv(out, sep="\t", index=False)
        counts[key] = len(df)
        if log is not None:
            log.append(
                f"  {key}: sampled {len(df):,} rows -> {out.name} in {time.monotonic() - t0:.1f}s"
            )

    # Ground truth restricted to the sampled S1 ids.
    if paths["gt"].exists():
        counts["gt"] = sum(1 for _ in paths["gt"].open(encoding="utf-8")) - 1
    else:
        t0 = time.monotonic()
        s1_ids = set(pd.read_csv(paths["s1"], sep="\t", usecols=["entity_id"], dtype=str)["entity_id"])
        gt = pd.read_csv(
            data_dir / "train_ground_truth.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
        gt = gt[gt["source1_entity_id"].isin(s1_ids)]
        gt.to_csv(paths["gt"], sep="\t", index=False)
        counts["gt"] = len(gt)
        if log is not None:
            log.append(f"  gt : {len(gt):,} rows -> slice_ground_truth.tsv in {time.monotonic() - t0:.1f}s")
        del gt
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-n", type=int, default=DEFAULT_S1_N, help="S1 rows in the slice")
    ap.add_argument(
        "--slice-dir",
        default=None,
        help="slice input dir (default artifacts/slice/data)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="slice artifact dir (default artifacts/slice/out)",
    )
    ap.add_argument(
        "--spill-dir",
        default=os.environ.get("P1_SPILL_DIR", "/var/tmp/p1_spill_slice"),
        help="intermediate spill dir; must be on a filesystem with headroom",
    )
    ap.add_argument(
        "--record-buffer",
        type=int,
        default=DEFAULT_RECORD_BUFFER,
        help="records buffered before each flush to the spill",
    )
    ap.add_argument("--buckets", type=int, default=64, help="dedup hash buckets")
    ap.add_argument("--build-only", action="store_true", help="build the slice, skip retrieval")
    ap.add_argument(
        "--skip-full",
        action="store_true",
        help="with --per-route, stop after the per-route passes (halves total wall clock)",
    )
    ap.add_argument("--reset", action="store_true", help="delete the slice spill first")
    ap.add_argument(
        "--routes",
        default=None,
        help="comma-separated subset of routes to run (default: all 7)",
    )
    ap.add_argument(
        "--per-route",
        action="store_true",
        help="run each route separately to obtain per-route timings",
    )
    args = ap.parse_args()

    data_dir = BASE_DIR / "dataset" / "train"
    slice_dir = Path(args.slice_dir) if args.slice_dir else BASE_DIR / "artifacts" / "slice" / "data"
    out_dir = Path(args.out_dir) if args.out_dir else BASE_DIR / "artifacts" / "slice" / "out"
    spill_dir = Path(args.spill_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log: list = []
    metrics: Dict[str, Any] = {
        "s1_n": args.s1_n,
        "record_buffer": args.record_buffer,
        "buckets": args.buckets,
        "spill_dir": str(spill_dir),
    }

    print("=" * 78)
    print("  P1 BOUNDED REAL-DATA SLICE RUNNER")
    print("=" * 78)
    avail, total = sys_mem_gb()
    print(f"\n[env] RAM {avail:.2f} GB available / {total:.2f} GB total | psutil={_HAS_PSUTIL}")
    print(f"[env] python {sys.version.split()[0]} on {sys.platform}")

    print(f"\n[1/5] Building slice (S1 n={args.s1_n:,}, streaming sampler)...")
    counts = build_slice(args.s1_n, slice_dir, data_dir, log=log)
    for line in log:
        print(line)
    metrics["input_rows"] = counts
    print(f"  inputs: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    if args.build_only:
        print("\n--build-only set; stopping before retrieval.")
        return 0

    s1 = pd.read_csv(slice_dir / "slice_source1.tsv", sep="\t", dtype=str, keep_default_na=False, na_values=[])
    s2 = pd.read_csv(slice_dir / "slice_source2.tsv", sep="\t", dtype=str, keep_default_na=False, na_values=[])
    s3 = pd.read_csv(slice_dir / "slice_source3.tsv", sep="\t", dtype=str, keep_default_na=False, na_values=[])
    print(f"\n[2/5] Loaded slice: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}  RSS={peak_rss_mb():.0f} MB")
    metrics["rss_after_load_mb"] = round(peak_rss_mb(), 1)

    cand_path = out_dir / "candidates.parquet"
    events_path = out_dir / "retrieval_events.parquet"

    if reset_ok(args.reset, spill_dir):
        pass

    enabled = None
    if args.routes:
        enabled = [r.strip() for r in args.routes.split(",") if r.strip()]

    per_route: Dict[str, Any] = {}
    if args.per_route:
        # Run routes one at a time so each gets its own timing + event count. The final
        # artifacts are produced by the last (full) run, not by the per-route passes.
        all_routes = list(DEFAULT_BLOCKING_CONFIG["enabled_routes"])
        chosen = enabled or all_routes
        for route in chosen:
            rdir = Path(str(spill_dir) + "_route_" + route)
            if reset_ok(True, rdir):
                pass
            t0 = time.time()
            generate_candidates(
                s1_df=s1,
                s2_df=s2,
                s3_df=s3,
                config={
                    "spill_dir": str(rdir),
                    "spill_events_path": str(rdir / "events.parquet"),
                    "spill_output_path": str(rdir / "candidates.parquet"),
                    "enabled_routes": [route],
                    "record_buffer": args.record_buffer,
                    "dedup_buckets": args.buckets,
                    "reset_spill_dir": False,
                },
            )
            dt = time.time() - t0
            n_ev = pq.ParquetFile(rdir / "events.parquet").metadata.num_rows
            per_route[route] = {"seconds": round(dt, 2), "events": n_ev}
            print(f"  route {route:20s} {dt:8.1f}s  events={n_ev:,}  RSS={peak_rss_mb():.0f} MB")
        metrics["per_route"] = per_route

    if args.skip_full:
        metrics["generation_seconds"] = round(sum(v["seconds"] for v in per_route.values()), 2)
        metrics["per_route"] = per_route
        mpath = out_dir / "slice_metrics.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n--skip-full set; wrote {mpath}")
        return 0

    print(f"\n[3/5] Full 7-route streaming run -> {out_dir}")
    print(f"  spill: {spill_dir}")
    if args.reset and spill_dir.exists():
        import shutil

        shutil.rmtree(spill_dir)
    t0 = time.time()
    cfg: Dict[str, Any] = {
        "spill_dir": str(spill_dir),
        "spill_output_path": str(cand_path),
        "spill_events_path": str(events_path),
        "record_buffer": args.record_buffer,
        "dedup_buckets": args.buckets,
        "reset_spill_dir": False,
    }
    if enabled:
        cfg["enabled_routes"] = enabled
    generate_candidates(s1_df=s1, s2_df=s2, s3_df=s3, config=cfg)
    t_gen = time.time() - t0
    n_cand = pq.ParquetFile(cand_path).metadata.num_rows
    n_events = pq.ParquetFile(events_path).metadata.num_rows
    print(f"  done in {t_gen:.1f}s ({t_gen / 60:.2f} min)")
    print(f"  candidates={n_cand:,}  events={n_events:,}")
    print(f"  RSS={peak_rss_mb():.0f} MB ({peak_rss_mb() / 1024:.2f} GB)")

    print(f"\n[4/5] Validation")
    cands = pd.read_parquet(cand_path)
    checks: Dict[str, bool] = {}
    checks["candidates_nonempty"] = len(cands) > 0
    checks["candidates_schema"] = list(cands.columns) == CANONICAL_CANDIDATE_COLUMNS
    s1col = cands["s1_id"].tolist()
    checks["candidates_s1_globally_ordered"] = all(
        s1col[i] <= s1col[i + 1] for i in range(len(s1col) - 1)
    )
    checks["pair_key_format"] = bool(
        (cands["pair_key"] == cands["s1_id"] + "::" + cands["candidate_id"]).all()
    )
    checks["pair_key_unique"] = bool(cands["pair_key"].is_unique)
    checks["candidate_source_valid"] = bool(cands["candidate_source"].isin(["S2", "S3"]).all())
    checks["n_routes_ge_1"] = bool((cands["n_routes"] >= 1).all())
    checks["best_rank_ge_1"] = bool((cands["best_rank"] >= 1).all())
    checks["best_score_no_nan"] = bool(not cands["best_score"].isna().any())
    checks["candidates_within_input_s1"] = bool(set(cands["s1_id"]) <= set(s1["entity_id"]))

    # Candidate/event pair_key consistency without loading all events: compare counts and
    # use the fact that every candidate must have >=1 event.
    ev_meta = pq.ParquetFile(events_path).metadata
    checks["events_nonempty"] = ev_meta.num_rows > 0
    checks["events_ge_candidates"] = ev_meta.num_rows >= len(cands)
    ev_sample = pd.read_parquet(events_path).head(200_000)
    checks["events_schema"] = list(ev_sample.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score",
    ]
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    metrics["validation"] = checks

    print(f"\n[5/5] Writing metrics")
    metrics.update(
        {
            "generation_seconds": round(t_gen, 2),
            "candidates_rows": n_cand,
            "events_rows": n_events,
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "cand_bytes": cand_path.stat().st_size,
            "events_bytes": events_path.stat().st_size,
            "total_seconds": round(time.time() - t_start, 2),
        }
    )
    mpath = out_dir / "slice_metrics.json"
    mpath.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {mpath}")

    del s1, s2, s3
    gc.collect()
    failed = [k for k, v in checks.items() if not v]
    print("\n" + "=" * 78)
    if failed:
        print(f"  SLICE COMPLETED WITH {len(failed)} FAILED CHECK(S): {failed}")
    else:
        print(f"  SLICE COMPLETE in {time.time() - t_start:.1f}s - all checks passed")
    print("=" * 78)
    return 1 if failed else 0


def reset_ok(do_reset: bool, path: Path) -> bool:
    """Remove ``path`` when ``do_reset``; returns True if removed."""
    if do_reset and path.exists():
        import shutil

        shutil.rmtree(path)
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
