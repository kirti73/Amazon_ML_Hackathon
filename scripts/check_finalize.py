#!/usr/bin/env python
"""Validate the bounded finalizer in scripts/run_p1_parallel.py against the reference.

Builds a small multi-worker spill on real slice rows, finalizes it with the new
bounded-memory path, and asserts byte-level agreement with the existing
``iter_canonical_candidates_merged`` reference over a single shared spill.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

from src.blocking.spill import (  # noqa: E402
    CANONICAL_CANDIDATE_COLUMNS,
    iter_canonical_candidates_merged,
)

ROOT = Path("/var/tmp/p1_finalize_selftest")
PY = str(REPO / ".venv/bin/python")
DRIVER = str(REPO / "scripts/run_p1_parallel.py")
N_S1, N_POOL = 400, 900
EVENT_COLUMNS = ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]


def run(*args: str) -> None:
    r = subprocess.run([PY, "-u", DRIVER, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"driver failed: {' '.join(args)}")
    for line in r.stdout.strip().splitlines()[-3:]:
        print("   ", line)


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    slice_dir = ROOT / "slice"
    slice_dir.mkdir()
    src = REPO / "artifacts/slice/data"
    for name, n in (("slice_source1.tsv", N_S1), ("slice_source2.tsv", N_POOL), ("slice_source3.tsv", N_POOL)):
        pd.read_csv(src / name, sep="\t", dtype=str, keep_default_na=False, na_values=[]).head(n).to_csv(
            slice_dir / name, sep="\t", index=False
        )
    print(f"built {slice_dir} S1={N_S1} S2={N_POOL} S3={N_POOL}")

    # Three "workers" writing three separate spills, mimicking parallel route workers.
    work = [
        ("exact_name", 0, 200, ROOT / "spill_A"),
        ("exact_name", 200, 400, ROOT / "spill_B"),   # same route, disjoint S1 -> tests union
        ("rare_token_name", 0, 400, ROOT / "spill_C"),
    ]
    spills = []
    for route, lo, hi, d in work:
        print(f"worker {route} s1=[{lo}:{hi}) -> {d.name}")
        run("--mode", "worker", "--route", route, "--slice-dir", str(slice_dir),
            "--spill-dir", str(d), "--start", str(lo), "--stop", str(hi), "--reset",
            "--stats-file", str(ROOT / f"stats_{d.name}.json"))
        spills.append(str(d))

    out = ROOT / "out"
    print("finalizing (bounded path)...")
    run("--mode", "finalize", "--spill-dirs", *spills, "--out-dir", str(out),
        "--canon-dir", str(ROOT / "canon"), "--merge-batch", "37", "--flush-rows", "53")

    # Reference: union the same spills through the ORIGINAL single-spill code path.
    ref_dir = ROOT / "ref_spill"
    ref_dir.mkdir()
    for b in range(64):
        parts = [d / f"bucket_{b:04d}" for d in (ROOT / "spill_A", ROOT / "spill_B", ROOT / "spill_C")]
        tables = [pq.read_table(list(p.glob("part-*.parquet"))[0])
                  for p in parts if list(p.glob("part-*.parquet"))]
        if tables:
            import pyarrow as pa

            t = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
            (ref_dir / f"bucket_{b:04d}").mkdir(parents=True, exist_ok=True)
            pq.write_table(t, ref_dir / f"bucket_{b:04d}" / "part-00000.parquet")

    ref = pd.concat(
        list(iter_canonical_candidates_merged(ref_dir, 64)), ignore_index=True
    ) if True else None
    got = pq.read_table(out / "candidates.parquet").to_pandas()
    ev_got = pq.read_table(out / "retrieval_events.parquet").to_pandas()

    print(f"\nreference candidates={len(ref):,}  bounded candidates={len(got):,}  events={len(ev_got):,}")
    ok = True

    if list(got.columns) != list(CANONICAL_CANDIDATE_COLUMNS):
        print(f"FAIL column order: {list(got.columns)}")
        ok = False
    if list(ev_got.columns) != EVENT_COLUMNS:
        print(f"FAIL event columns: {list(ev_got.columns)}")
        ok = False
    if len(got) != len(ref):
        print("FAIL row count mismatch")
        ok = False
    else:
        r = ref[list(CANONICAL_CANDIDATE_COLUMNS)].reset_index(drop=True)
        g = got.reset_index(drop=True)
        same = r.equals(g)
        print(f"candidate frames identical (order-sensitive): {same}")
        ok &= same
        if not same:
            for c in CANONICAL_CANDIDATE_COLUMNS:
                if not r[c].equals(g[c]):
                    print(f"  differs in column {c}")
    if not got["s1_id"].is_monotonic_increasing:
        print("FAIL candidates not globally s1_id-ordered")
        ok = False
    else:
        print("candidates globally s1_id-ordered: True")

    dup = got.duplicated(subset=["pair_key", "s1_id", "candidate_id"]).sum()
    if dup:
        print(f"FAIL {dup} duplicate pair_keys")
        ok = False

    if ok:
        shutil.rmtree(ROOT)
    print("\n" + ("FINALIZE SELFTEST PASSED" if ok else "FINALIZE SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
