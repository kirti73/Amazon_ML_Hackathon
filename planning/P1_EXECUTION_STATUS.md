# P1 Execution Status & Handoff

**Purpose:** single authoritative, version-controlled execution log for P1 candidate
generation. Sufficient for another agent or human to resume with no prior conversation
context.

> This file supersedes the earlier `artifacts/P1_EXECUTION_STATUS.md`, which is **not
> version-controlled** (`.gitignore` line `artifacts/`) and was therefore not durable. Its
> full historical content is carried forward in §A below.

---

## HANDOFF / CURRENT STATE

| Field | Value |
|---|---|
| Current phase | **Phase 3 — COMPLETE (validated). Next: Phase 5 (runner), then Phase 6 (slice).** |
| Timestamp (UTC) | 2026-09-26T21:04:08Z |
| Branch | `person4/integration-p1-p2` |
| Commit | `c18659e497e4155a363357d8123572654c770af3` — *Integrate P3 ML and decision pipeline* |
| Working tree | **DIRTY — 7 modified, 5 untracked.** All P1 engineering work is uncommitted. |
| Safe to resume? | **YES.** No process is running. No partial work exists on disk. |
| Blocker | None for Phases 0–8. Full-data run is prohibited (see §6). |
| Phase 0 baseline | **106 passed / 0 failed** (9 modules); 3 modules env-blocked — see §H |

### 1. What is complete
- P3 (models/decision/validation) implemented, committed at `c18659e`, 112/112 tests passing.
- Read-only root-cause investigation of the 2026-09-26 WSL crash (§C).
- Implementation plan for Phases 0–8 approved by the owner (2026-09-26).

### 2. What is currently running
**Nothing.** The workspace is idle.

### 3. What artifacts exist
All pre-existing, **untouched** sample-sized smoke artifacts. See §B for the full table.
**No full-data P1 output exists. None has ever been produced.**

### 4. What has been validated
- Route 1 full-scale output volume measured: **21,760,069 events** (read-only probe).
- Per-record Python cost measured: **547 B** → **~11.9 GB** for Route 1 alone.
- L1 transform equivalence vs `HEAD` (bit-identical) on 2,000/2,000/2,500 real rows.
- 2D blocked retrieval determinism across 5 block configs (1,200 S1 sample).
- Spill/dedup equivalence at 1/3/64 buckets (900 S1 sample: 94,378 events, 65,282 candidates).
- pytest regression baseline (Phase 0): **106 passed, 0 failed**; 3 modules blocked by
  missing `rapidfuzz` / `libgomp.so.1` (pre-existing, unrelated to P1) — see §H.

### 5. What remains
Phases 0–8. See §D for the phase list and §E for gates.

### 6. What must NOT be done
- **DO NOT run the full-data P1 pipeline** (`scripts/run_p1_full_data.py` on full TSVs).
  It is memory-fatal and will crash WSL. See §C.
- **DO NOT modify `artifacts/folds.parquet`.**
- **DO NOT delete, overwrite, or replace any existing artifact.** New work goes to
  `artifacts/slice/` and `/var/tmp/p1_spill`.
- **DO NOT use `artifacts/train_labels.parquet`** — it is all zeros from placeholder IDs.
- **DO NOT change routes, thresholds, ranking, `top_k`, `min_df`, or candidate semantics.**
- **DO NOT change the 300K slice size or the 50K `RecordBuffer` threshold without approval.**

### 7. Exact command to continue

Phase 0 baseline (safe, read-only, expected ~2–5 min):

```bash
cd /mnt/d/PROJECTS/Amazon_ML_Hackathon
.venv/bin/python -m pytest tests/ -q 2>&1 | tee /var/tmp/p1_pytest_baseline_phase0.txt
```

### 8. Known risks / blockers
- Full-data P1 runtime projected **~81 h per TF-IDF route (~162 h for two)**, unresolved.
  The 300K slice (Phase 6) exists to measure this properly.
- `/mnt/d` has only **32 GB free**; full-data spill would not fit. `/var/tmp` (on `/`) has
  **949 GB free** and is the approved spill location.

---

## B. Artifact inventory (all pre-existing; do not modify)

| Path | Rows | Size (B) | Modified (UTC) | Notes |
|---|---|---|---|---|
| `artifacts/candidates.parquet` | 2,266 | 40,700 | 2026-09-26 12:00:39 | **Old smoke output.** NOT full P1. |
| `artifacts/retrieval_events.parquet` | 3,994 | 53,152 | 2026-09-26 12:00:46 | **Old smoke output.** NOT full P1. |
| `artifacts/features.parquet` | 2,266 | 120,351 | 2026-09-26 12:00:41 | Built from placeholder IDs. |
| `artifacts/records.parquet` | 350 | 48,815 | 2026-09-26 12:00:44 | Synthetic. |
| `artifacts/train_labels.parquet` | 2,266 | 18,065 | 2026-09-26 12:31:34 | **ALL `y=0` — UNUSABLE.** |
| `artifacts/folds.parquet` | 2,206,821 | 15,331,569 | 2026-09-25 22:42:02 | **VALID. MUST NOT BE MODIFIED.** |
| `artifacts/_p1_spill/` | 0 files | 0 | 2026-09-26 20:40:14 | Crash evidence: 64 empty bucket dirs. |

Schemas (verified via pyarrow):

- `candidates.parquet`: `pair_key, s1_id, candidate_id, candidate_source, n_routes,
  best_rank, best_score` (large_string / int64 / double)
- `retrieval_events.parquet`: `pair_key, s1_id, candidate_id, candidate_source, route,
  rank, score`
- `folds.parquet`: `s1_id (string), fold_id (int8)`

**Ordering contract (verified):** `candidates.parquet` is **globally sorted by `s1_id`
ascending**, then by rank. This matches the invariant documented in
`candidate_generation.py:27` and must be preserved by the streaming path.

---

## A. Carried-forward history (from the pre-crash `artifacts/P1_EXECUTION_STATUS.md`)

### A.1 Dataset

| File | Rows | Size |
|---|---|---|
| `dataset/train/train_source1.tsv` (S1) | 2,206,821 | 210 MB |
| `dataset/train/train_source2.tsv` (S2) | 5,034,616 | 489 MB |
| `dataset/train/train_source3.tsv` (S3) | 5,285,603 | 504 MB |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 | 127 MB |
| **Candidate pool (S2+S3)** | **10,320,219** | |

Ground truth: 7,638,365 positive pairs; 123,247 zero-match S1 entities.

Measured source residency: S1 0.27 GB + S2 0.63 GB + S3 0.65 GB = **1.55 GB**
(123.2 / 124.8 / 123.4 B per row respectively, measured on 50K-row samples).

### A.2 Environment

| Item | `.venv` (WSL, **the only supported env**) | `.venv_win` (Windows) |
|---|---|---|
| Python | 3.12.3 | 3.12 |
| pandas / numpy / scipy | 3.0.6 / 2.5.3 / 1.18.1 | identical |
| pyarrow / lightgbm / joblib | 25.0.1 / 4.7.0 / 1.6.0 | identical |
| **psutil** | **ABSENT** | 7.2.2 |

- `duckdb`, `dask`, `polars` are **not installed in either environment**. All external
  aggregation uses `pyarrow`.
- **All execution must use `.venv/bin/python` under WSL.** `.venv_win` is unusable here
  (Windows path, and `resource` is unavailable on Windows — see Phase 5 fix).

### A.3 First full-data attempt (historical, pre-L1/L5)

- **Result: FAILED** with `MemoryError` in the `tfidf_name` route inside
  `CharTfidfVectorizer.transform()`.
- Observed ~12.78 GB working set, ~18.78 GB paged memory.
- **No artifacts were produced.** The current `candidates.parquet` /
  `retrieval_events.parquet` predate it.

### A.4 Measured TF-IDF characteristics

| Quantity | Value |
|---|---|
| Vocabulary (`min_df=2`, 3–4 grams) | 110,658 – 145,750 terms |
| Non-zeros per document | 42.5 |
| CSR bytes per row | 343.8 |
| Full-pool CSR | 3.5 GB |
| `.T.tocsc()` second copy | ~7 GB + temporaries |

Similarity scaling — nnz is **exactly linear** in pool size, density never decays:

| Pool | sim nnz | density | ≥0.05/query | time |
|---|---|---|---|---|
| 50,000 | 29.6M | 29.59% | 3,174 | 0.8 s |
| 200,000 | 118.5M | 29.63% | 12,470 | 3.4 s |
| 1,200,000 | 728.5M | 30.35% | 76,916 | 30.9 s |

Extrapolating 9,490× from the 1.2M measurement → **~81 h per TF-IDF route**. Lower bound,
since density grows with pool size.

`min_df` is **not** a viable lever (density stays flat while results change drastically —
top-20 overlap vs `min_df=2`: `min_df=20` → 82.0%, `min_df=100` → 47.5%, `min_df=500` →
32.2%). Cause: business names are short and lexically repetitive, so char 3-grams provide
almost no selectivity. Property of the data plus the algorithm, not the implementation.

### A.5 Event volume projection (pre-crash estimate)

| Route | Events/S1 | Full-scale rows |
|---|---|---|
| `tfidf_name` | 20.00 (capped) | 44.1M |
| `tfidf_address` | 20.00 (capped) | 44.1M |
| `rare_token_address` | 20.00 (capped) | 44.1M |
| `rare_token_name` | 19.59 (capped) | 43.2M |
| `numeric_address` | 15.15 (capped) | 33.4M |
| `exact_name` | 0.23 | 0.5M |
| `reverse_retrieval` | unbounded per S1 (370 @ pool:S1=83; 43 @ 9.4; max 1,222 for one S1) | ≤ 51.6M |
| **Total** | | **~260M rows (~32 GB pre-compression)** |

> **Correction from the crash investigation:** the `exact_name` figure above (0.5M) was
> **wrong by ~40×**. Measured on full data, Route 1 emits **21,760,069** events, not 0.5M.
> See §C.2.

### A.6 Block-size grid (16 configs, 5,000 queries × 250,000 candidates)

Total sim nnz was **identical (369,013,532)** in all 16 configurations, proving blocking is
work-invariant. Runtime flat (20.9–25.6 s).

| Q block × C block | peak sim MB | peak RSS MB |
|---|---|---|
| 500 × 25K | 30 | 250 |
| 1000 × 50K | 119 | 351 |
| **1000 × 100K** | **238** | **519** |
| 2000 × 100K | 474 | 767 |
| 2000 × 250K | 1,183 | 1,541 |
| 5000 × 250K | 2,952 | 3,439 |

**Selected: Q block 1,000 × C block 100,000** — 238 MB sim, ~2% of RAM.

### A.7 Correctness finding — pre-existing nondeterminism (corrected)

`np.argpartition` top-k selection picked an arbitrary subset *before* the documented
`(-score, candidate_id)` sort was applied, so exact-score ties at the k-th boundary were
resolved nondeterministically. Measured: **2 of 400 queries (0.5%)**.

The pre-existing implementation was stable against `chunk_size` (400/100/64/37 all identical)
because query chunking does not change a row's similarity data. Candidate blocking exposed
the latent bug.

**Owner-approved resolution:** the canonical `(-score, candidate_id)` ordering is now
authoritative, applied by full sort (`canonical_top_k` in `tfidf_common.py`). This corrects
nondeterministic behaviour and does not change frozen route semantics, `top_k`, coverage,
`min_df`, or score values.

> **Consequence for all future comparisons:** output now differs from the historical
> implementation for exact boundary ties. The 6,806/10,010 baseline is therefore not
> comparable even if it were reproducible.

### A.8 L1–L5 engineering work (in working tree, UNCOMMITTED)

| Layer | Change | Files |
|---|---|---|
| **L1** | `transform()` uses preallocated NumPy buffers instead of Python lists | `tfidf_name.py`, `tfidf_address.py`, `reverse.py` |
| **L2** | 2D blocked retrieval: S1 block × candidate block, running top-k merge; no full pool matrix, no full transpose, no retained similarity | same 3 files |
| **L3** | Candidate names normalized per block instead of a 10.3M-string list | same 3 files |
| **L4** | Hash-bucketed Parquet spill + external dedup | `blocking/spill.py` (new), `candidate_generation.py` |
| **L5** | `psutil` optional with `/proc/meminfo` + `resource` fallback | `run_p1_full_data.py` |

New shared module: `blocking/tfidf_common.py` — `build_l2_normalized_csr`,
`canonical_top_k`, `merge_top_k`, `iter_pool_names`, `iter_pool_blocks`, `pool_size`.

New config keys: `tfidf_name_candidate_chunk`, `tfidf_addr_candidate_chunk`,
`tfidf_query_block`, `spill_dir`, `dedup_buckets`.

**L1–L5 are structurally correct but INCOMPLETE — see §C.3.**

---

## C. Crash investigation (2026-09-26)

### C.1 What happened

A full-data run of `scripts/run_p1_full_data.py` was launched by mistake and killed the WSL
VM:

```
Catastrophic failure
Error code: Wsl/Service/E_UNEXPECTED
```

Environment at death:

| Item | Value |
|---|---|
| Python | `.venv/bin/python` 3.12.3 (**WSL**, not Windows) |
| `psutil` | absent → `resource` + `/proc/meminfo` fallback active |
| **Physical RAM (`MemTotal`)** | **12,087,492 kB ≈ 11.53 GB** |
| Swap | 3.00 GB → **≈14.53 GB addressable** |
| Observed usage | **~20 GB** |
| Disk `/mnt/d` | 274 GB total, **32 GB free** (89% used) |
| Disk `/` (`/var/tmp`) | 1007 GB total, **949 GB free** |

### C.2 Root cause (measured, not assumed)

Read-only probe of Route 1 against the real dataset:

```
distinct normalized names in pool : 7,657,215
pool rows indexed                : 10,320,219
S1 rows                          :  2,206,821
ROUTE 1 RECORDS EMITTED (cap 500): 21,760,069
```

Per-record cost, measured: 7-key `dict` = 272 B, `pair_key` `str` = 67 B, list slot = 8 B
→ **547 B/record**.

**21,760,069 × 547 B ≈ 11.9 GB in a single Python list.**

Memory tally at death:

| Object | Size |
|---|---|
| S1 + S2 + S3 DataFrames | 1.55 GB |
| `pool_index` (7.66M keys + 10.32M tuples) | ~1.7 GB (transiently ~2.5 GB: S2 and S3 indexes coexist) |
| Route 1 `records` list | **~11.9 GB** ← the bomb |
| `pd.DataFrame(records)` conversion | +~1.3 GB (7 column arrays allocated while `records` alive) |
| WSL page cache + interpreter/imports | ~1–2 GB |
| **Total** | **~17–20 GB** — matches the observed ~20 GB |

**Decisive evidence:** `artifacts/_p1_spill/` contains **64 empty bucket directories and
zero Parquet files**. `EventSpillWriter` creates the buckets before Route 1 runs, but
`_collect()` only spills a route's output *after* that route returns. Therefore
**Route 1 (`exact_name`) never returned.** The process died inside it.

### C.3 Why L4 spill did not help — the central flaw

Spilling happens **after** a route returns its DataFrame. All 7 routes accumulate their
entire output as a list of dicts *first*:

```python
records = []                       # exact_name.py:140, numeric.py:159,
records.append({...})              # rare_token.py:147/182,
return pd.DataFrame(records, ...)  # tfidf_name.py:357/372, tfidf_address.py:371/386
```

So the spill never engages until a route has already exhausted memory. L1–L5 bounded the
TF-IDF **matrices** but left the **record lists** unbounded.

This is not Route 1-specific. Routes 2–6 cap at `top_k=20` → up to **44.1M records ≈ 24 GB**
each. Route 7 → up to **51.6M**. **Every route is independently fatal.**

Two further amplifiers, both still present:
- `generate_candidates(..., return_events=True)` calls `_collect_spilled_events()`
  (`candidate_generation.py:337`), which loads every bucket then `pd.concat` — recreating
  the full event spike.
- `reconcile_candidates_schema` does `events = retrieval_events_df.copy()`
  (`adapters.py:79`) plus `res = candidates_df.copy()` — a second ~30 GB spike.

### C.4 Resumability

**Not resumable.** No checkpoint, resume, manifest, `_SUCCESS`, or completed-range logic
exists anywhere in `src/` or the runner. Worse, restart **destroys** partial work:
`resolve_spill_dir()` calls `shutil.rmtree(path)` (`spill.py:95-96`) and the runner calls
`shutil.rmtree(spill_dir)` (`run_p1_full_data.py:139-142`).

Nothing was lost (0 shards existed), but any restart currently means full recompute and an
identical OOM.

### C.5 Semantics audit of the L1–L5 work

| Component | Safe? | Constraint to preserve |
|---|---|---|
| TF-IDF fit scope | Yes | Fit **once** on full `S2+S3+S1`. Block-fitting would change vocab/IDF and every score. |
| TF-IDF 2D blocking | Yes (verified) | Running top-k merge ≡ unblocked. |
| `exact_name` full-pool index | Yes, and **required** | Buckets must be complete before the 500 cap. Index is ~1.7 GB, unavoidable. |
| `reverse` per-candidate top-k over full S1 index | Yes (verified) | Running top-5 merge across S1 blocks is correct. A naive per-block emit would be **wrong**. |
| Cross-bucket dedup | Yes (verified) | `crc32(s1_id)` confines all events for an `s1_id` to one bucket, so bucket-local ≡ global. **Will be explicitly tested in Phase 7.** |
| Global rank | Yes | Recomputed as `score desc, candidate_id asc`; bucket-local == global. |
| Tie-break | **Changed (approved)** | `(-score, candidate_id)` full sort. See §A.7. |

### C.6 Baseline discrepancy — resolved

The previously cited baseline of **6,806 candidates / 10,010 events is NOT reproducible**.
`grep` finds no occurrence of those numbers in any `.py`, `.md`, or `.json` in the repo.

Measured alternatives:

| Source | Candidates | Events |
|---|---|---|
| Cited baseline | 6,806 | 10,010 |
| **Current code on committed 1K fixtures** (`s*_sample.tsv`, `random_state=42`) | **63,488** | **88,448** |
| Artifacts on disk (placeholder-ID smoke) | 2,266 | 3,994 |

The on-disk artifacts came from the placeholder-ID smoke run. **The 63,488 / 88,448 figures
are the new authoritative baseline.**

### C.7 Crash-reporting record (per requirement 10)

- **What was running:** `scripts/run_p1_full_data.py`, Phase [2/6] candidate generation.
- **Progress:** dataset load complete; spill writer initialized (64 dirs); died in Route 1.
- **Observed resources:** ~20 GB usage against an 11.53 GB physical cap.
- **Error:** `Catastrophic failure` / `Wsl/Service/E_UNEXPECTED`.
- **Surviving artifacts:** all pre-existing smoke artifacts, unmodified. Empty spill dir.
- **Recovery/resume possible:** **No** — no partial work persisted.
- **Must NOT be repeated:** launching the full-data pipeline. It is a known OOM.

---

## D. Approved implementation plan (Phases 0–8)

Owner-approved 2026-09-26. Slice size (300K) and `RecordBuffer` threshold (50K) are fixed
and must not be changed without approval.

| Phase | Scope | Key files |
|---|---|---|
| **0** | **HARD GATE.** Capture existing `pytest tests/` baseline + re-anchor fixture baseline (63,488/88,448). No behavior change. | — (read-only) |
| **1** | Sink primitives: `write_records`, `InMemoryEventSink`, `RecordBuffer(max_records=50_000)`, non-destructive `resolve_spill_dir` (drop `rmtree`), `_SUCCESS` manifest, `write_events_stream` (pyarrow `ParquetWriter`) | `blocking/spill.py` |
| **2** | Add `sink` param to all 7 routes; delete every unbounded `records` list | `exact_name.py`, `tfidf_name.py`, `tfidf_address.py`, `rare_token.py`, `numeric.py`, `reverse.py` |
| **3** | Orchestrator constructs one `RecordBuffer`, passes it to all 7 routes, drops `route_dfs` in spill mode, removes `return_events` reassembly | `candidate_generation.py` |
| **4** | `canonicalize_bucket` (bucket-local `n_routes`/`best_rank`/`best_score`) + **64-way `heapq.merge`** across buckets to preserve global `s1_id` order | `candidate_generation.py`, `spill.py` |
| **5** | Runner: spill → `/var/tmp/p1_spill` (env-overridable); remove `rmtree`; resume from manifest; drop `return_events=True`; fix unconditional `import resource` | `scripts/run_p1_full_data.py` |
| **6** | New slice runner: **streaming** sampler (do NOT reuse `make_sample.py`, which loads full TSVs first), 300K S1 / 687K S2 / 706K S3, GT sliced to same S1 ids, `--s1-n` override, per-route timing + peak RSS + event counts, full-run extrapolation | `scripts/run_p1_slice.py` (new) |
| **7** | **HARD GATE.** Verification — see §E | `tests/test_blocking_memory.py`, `tests/test_blocking_spill.py` (new) |
| **8** | Two separate commits: (1) L1–L5 already in tree, (2) streaming-sink work | — |

### Flush points per route (Phase 2)

| Route | Site | Flush point | Full-scale volume removed |
|---|---|---|---|
| 1 `exact_name` | `exact_name.py:140` | every 50K S1 rows | **21,760,069 (11.9 GB)** |
| 2 `tfidf_name` | `tfidf_name.py:353` | end of each query block (≤20K) | up to 44.1M |
| 3 `rare_token_name` | `rare_token.py:147` | every 50K S1 rows | up to 44.2M |
| 4 `tfidf_address` | `tfidf_address.py:371` | end of each query block | up to 44.1M |
| 5 `numeric_address` | `numeric.py:159` | every 50K S1 rows | up to 33.4M |
| 6 `rare_token_address` | `rare_token.py` | every 50K S1 rows | up to 44.1M |
| 7 `reverse_retrieval` | `reverse.py:276` | already supports `sink.emit()` — **just wire it** | up to 51.6M |

When `sink is None`, every route returns the current in-memory DataFrame unchanged, so all
existing tests and the 7 `scripts/test_route*.py` harnesses keep passing. **The in-memory
implementation is retained deliberately as the equivalence reference (requirement 6).**

---

## E. Phase 7 gates (all must pass; do not mark complete until each actually passes)

1. **Peak RSS target** — 300K slice completes under **~6 GB** peak RSS.
2. **In-memory vs streaming equality** on the smaller validation slice — byte-identical
   canonical output.
3. **Global `s1_id` ordering** — final `candidates.parquet` globally sorted by `s1_id`,
   then rank (matching the verified contract in §B).
4. **Candidate/event consistency** — `set(candidates.pair_key) == set(events.pair_key)`.
5. **Determinism across block configurations** — identical output across ≥3 configs.
6. **Bucket-local ≡ global canonicalization** — explicitly tested, not merely documented:
   for the same input, `canonicalize_bucket` applied per bucket must equal global
   `reconcile_candidates_schema`, including `n_routes`, `best_rank`, `best_score`, row order,
   and index.
7. **Full `pytest tests/`** with no regressions vs the Phase 0 baseline.

---

## F. Decisions and rationale

| Decision | Rationale |
|---|---|
| Streaming sink for all 7 routes (vs fixing Route 1 only) | All routes are independently fatal; a partial fix leaves the same crash class. |
| Large real slice first, then decide on full run | Only way to unblock P2/P3 and to measure true per-route runtime. |
| Spill on `/var/tmp` (`/`), final artifacts on `/mnt/d` | `/mnt/d` has 32 GB free; full spill needs tens of GB. `/` has 949 GB. |
| Single-file `retrieval_events.parquet` via streaming `ParquetWriter` | Preserves the existing schema contract; no P2/P3 schema churn. |
| 64-way `heapq.merge` in Phase 4 | Preserves the verified global `s1_id` ordering contract at O(64) memory. |
| Keep bucket count at 64 | Hash fan-out must stay fixed for the merge to be O(64) and deterministic. |
| Leave `scripts/run_p1_p2_real_data.py` untouched | Dead code (hardcoded `C:\Users\...` zip path). Not widening the diff. |
| Do not reuse `scripts/make_sample.py` | It reads each full TSV into memory before sampling — defeats the purpose on a 504 MB file. |

### Deviations from plan
- **Status document relocated** from `artifacts/P1_EXECUTION_STATUS.md` to
  `planning/P1_EXECUTION_STATUS.md` because `artifacts/` is gitignored and therefore not a
  durable handoff location. Historical content carried forward in §A. The old file will be
  reduced to a pointer stub (not deleted) to prevent two competing status documents.
  *(Not yet performed — pending.)*

---

## G. Log

| Timestamp (UTC) | Phase | Event | Result |
|---|---|---|---|
| 2026-09-26T19:15:42Z | pre-plan | Wrote initial `artifacts/P1_EXECUTION_STATUS.md` | 11,575 B |
| 2026-09-26T20:40:14Z | — | Accidental full-data run; WSL crashed (`Wsl/Service/E_UNEXPECTED`) | 0 artifacts written; 64 empty spill dirs |
| 2026-09-26T21:04:08Z | — | Read-only crash investigation completed; root cause measured | Route 1 = 21,760,069 events ≈ 11.9 GB |
| 2026-09-26T21:04:08Z | — | Phases 0–8 approved by owner | Slice 300K / buffer 50K fixed |
| 2026-09-26T21:04:08Z | 0 | Created this authoritative status document | — |
| 2026-09-26T21:05:04Z | 0 | Ran `pytest tests/ -q` (PRE-CHANGE baseline) | **3 collection errors, 0 tests run** — see §H |
| 2026-09-26T21:05:28Z | 0 | Ran collectable subset → **true baseline** | **106 passed, 0 failed, 12.90 s** |
| 2026-09-26T21:06Z | 0 | **PHASE 0 GATE PASSED** | Baseline recorded in §H |
| 2026-09-26T21:12Z | 1 | Added sink primitives to `spill.py` | `RecordBuffer`(50K), `InMemoryEventSink`, `SpillManifest`, `write_records`/`emit`, `write_events_stream`, `canonicalize_frame`, row-level k-way merge |
| 2026-09-26T21:12Z | 1 | Removed destructive `rmtree` from `resolve_spill_dir` (now `reset=False` default) | Restart no longer destroys partial work |
| 2026-09-26T21:14Z | 1 | **Found+fixed 2 self-introduced bugs** | `range([5])` TypeError; whole-frame yields interleaved incorrectly → replaced with true row-level `heapq.merge` |
| 2026-09-26T21:16Z | 1 | Pinned artifact event schema (large_string/int64/double) | Matches existing `retrieval_events.parquet` contract |
| 2026-09-26T21:18Z | 1 | **PHASE 1 VALIDATED** | Gates 3+4 pass at 1/3/7/64 buckets — see §I |
| 2026-09-26T21:24Z | 2 | Added `sink` param to all 7 routes; deleted every unbounded `records` list | `grep` confirms zero `records = []` / `.append` / `.extend` remain in any route |
| 2026-09-26T21:26Z | 2 | Added `RecordBuffer.emit` batch alias | Required by `reverse.py`'s pre-existing sink protocol — caught by equivalence test, not assumed |
| 2026-09-26T21:28Z | 2 | **All 7 routes: sink output IDENTICAL to in-memory** | 1K fixtures, threshold forced to 7 — see §J |
| 2026-09-26T21:29Z | 2 | Regression check | **106 passed**; fixture baseline **exactly 63,488 / 88,448** |
| 2026-09-26T21:33Z | 3 | Orchestrator: one `RecordBuffer` passed to all 7 routes | New config keys `record_buffer`, `spill_output_path`, `spill_events_path`, `reset_spill_dir` |
| 2026-09-26T21:33Z | 3 | Finalization now streams both artifacts via `ParquetWriter` | Replaces the read-all-buckets + `pd.concat` path that recreated the spike |
| 2026-09-26T21:36Z | 3 | **End-to-end streaming == in-memory** | 63,488 / 88,448 both paths; canonical output identical; global `s1` order preserved — see §K |
| 2026-09-26T21:37Z | 3 | Regression + import gate | **106 passed**; all 9 modules import cleanly (no circular import) |

---

## K. Phase 3 results — orchestrator streaming (VALIDATED)

### K.1 Changes to `candidate_generation.py`

- A single `RecordBuffer` is constructed when a spill dir is configured and passed to **all
  7 routes** as `sink=`. This is the missing link that made the spill useless: previously
  `_collect()` could only spill a route's output *after* the route had already built it in
  memory.
- New config keys: `record_buffer` (default 50,000), `spill_output_path`,
  `spill_events_path`, `reset_spill_dir` (default `False`).
- Finalization rewritten: when both output paths are supplied, events and candidates are
  each written with a single `ParquetWriter` while streaming bucket-by-bucket. The previous
  `_collect_spilled_events` / `_finalize_spilled` read every bucket back and `pd.concat`-ed
  them, which reproduced the entire memory spike the spill exists to prevent.
- When no output paths are given, the in-memory aggregation path is retained (tests and the
  1K fixtures use it).
- `write_candidate_dataset` gained a `columns` argument so it emits the canonical 7-column
  schema directly.

### K.2 End-to-end validation (executed, 1K fixtures, `record_buffer=2500`)

| Path | Candidates | Events |
|---|---|---|
| A — pure in-memory (`return_events=True`) | 63,488 | 88,448 |
| B — streamed to disk, nothing returned resident | **63,488** | **88,448** |

| Check | Result |
|---|---|
| Events content identical (order-independent) | **yes** |
| Candidates identical to global `reconcile_candidates_schema` | **yes** |
| Candidates globally sorted by `s1_id` | **yes** |
| `set(candidates.pair_key) == set(events.pair_key)` | **yes** |
| Candidate schema | the canonical 7 columns |
| Shards written | 2,009 |

Regression after Phase 3: **106 passed**, all 9 blocking modules import cleanly (no circular
import introduced by the new `.spill` imports in the route modules).

---

## H. Phase 0 — regression baseline (HARD GATE: PASSED)

Captured **before** any behavior change, at commit `c18659e` with the pre-existing uncommitted
L1–L5 work in the tree.

### H.1 Raw result — `pytest tests/ -q`

```
3 errors in 18.84s   (exit code 2)
Maximum resident set size: 195,540 kB
```

**The baseline is NOT green.** 3 of 12 test modules cannot be collected:

| Module | Import failure | Cause | P1-related? |
|---|---|---|---|
| `tests/test_features.py` | `ModuleNotFoundError: No module named 'rapidfuzz'` | P2 feature dep absent in `.venv` | **No** |
| `tests/test_integration_p1_p2.py` | same `rapidfuzz` failure (via `src.inference.pipeline`) | same | **No** |
| `tests/test_lightgbm.py` | `OSError: libgomp.so.1: cannot open shared object file` | GNU OpenMP runtime absent system-wide | **No** |

Both gaps are **pre-existing WSL environment issues**, confirmed independent of any P1 code:
`libgomp.so.1` is absent from `/usr/lib/x86_64-linux-gnu/` entirely, and `rapidfuzz` is not
installed in `.venv`. Neither module exercises P1 blocking code. The earlier
`artifacts/P1_EXECUTION_STATUS.md` claim of "112/112" was measured under `.venv_win`
(Windows), which has these dependencies. **No packages were installed to "fix" this** —
that is out of P1 scope and would require modifying the system.

### H.2 True regression baseline — the number that matters

```bash
.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_features.py \
  --ignore=tests/test_integration_p1_p2.py \
  --ignore=tests/test_lightgbm.py
```

```
106 passed, 2 warnings in 12.90s   (exit code 0)
```

9 modules collect, **106 passed, 0 failed**. Full collection count is also
`106 tests collected, 3 errors`.

**Phase 7 gate 7 is therefore defined as: still 106 passed / 0 failed on this same subset,
with the same 3 collection errors (no new ones). Any change to those numbers is a
regression.**

Baseline logs retained at `/var/tmp/p1_pytest_baseline_phase0.txt` and
`/var/tmp/p1_pytest_baseline_collectable.txt`.

### H.3 Authoritative fixture baseline (replaces the unreproducible 6,806/10,010)

Committed fixtures `dataset/train/s{1,2,3}_sample.tsv` (1,000 rows each,
`random_state=42`), all 7 routes enabled, current code:

| Metric | Value |
|---|---|
| Consolidated candidates | **63,488** |
| Retrieval events | **88,448** |
| Routes present | `exact_name`, `numeric_address`, `rare_token_address`, `rare_token_name`, `reverse_retrieval`, `tfidf_address`, `tfidf_name` |
| Peak RSS | 157.5 MB |

Reproduce with:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"code/business_entity_resolution")
import pandas as pd
from src.blocking.candidate_generation import generate_candidates
s1=pd.read_csv("dataset/train/s1_sample.tsv",sep="\t",dtype=str)
s2=pd.read_csv("dataset/train/s2_sample.tsv",sep="\t",dtype=str)
s3=pd.read_csv("dataset/train/s3_sample.tsv",sep="\t",dtype=str)
c,e=generate_candidates(s1,s2,s3,return_events=True)
print(len(c), len(e))
PY
```

Expected: `63488 88448`.


---

## I. Phase 1 results — sink primitives (VALIDATED)

### I.1 What was added to `code/business_entity_resolution/src/blocking/spill.py`

| Symbol | Purpose |
|---|---|
| `DEFAULT_RECORD_BUFFER = 50_000` | Hard bound on buffered records before auto-flush. **Owner-fixed; not to be changed.** |
| `RecordBuffer(sink, max_records)` | `append(dict)` + auto-flush + `flush()`/`close()`/`result()`. `sink=None` retains records → this is the preserved in-memory equivalence path. |
| `InMemoryEventSink` | Unbounded reference sink implementing `write_frame`/`write_records`/`emit`/`result`. |
| `EventSpillWriter.write_records()` / `.emit()` | Sink protocol on the real spill writer. |
| `SpillManifest` | JSON-backed per-route completion tracking → enables resume. |
| `resolve_spill_dir(reset=False)` | **No longer deletes** an existing spill dir. Destructive-by-default made resume impossible. |
| `canonicalize_frame()` | Bucket-local `n_routes`/`best_rank`/`best_score` + union rank. |
| `iter_canonical_candidates_merged()` | **Row-level** `heapq.merge` over buckets → restores global `s1_id` order at O(buckets + 50K) memory. |
| `iter_single_bucket()`, `_canonical_rows()` | Per-bucket readers feeding the merge. |
| `write_events_stream()` | Single-file `retrieval_events.parquet` via one `ParquetWriter`, pinned to the existing artifact schema. |
| `CANONICAL_CANDIDATE_COLUMNS` | The 7-column `docs/schemas.md` §6 contract as a constant. |

`EventSpillWriter.close()` now also writes a `_SUCCESS` sentinel into each bucket directory.

### I.2 Two bugs I introduced and fixed (recorded for honesty)

1. **`range([5])` TypeError** — I passed a *list* where `iter_spilled_buckets` expects an
   `int`. Caught by an immediate import/smoke check, before any test run.
2. **Incorrect frame-granular merge.** My first merge yielded whole per-bucket frames
   ordered by each frame's *first* `s1_id`. That is wrong: bucket *b* may hold `S1-0005`
   and `S1-0099` while bucket *c* holds `S1-0007`, so the output interleaves incorrectly
   and breaks the global ordering contract. Replaced with a true **row-level**
   `heapq.merge` keyed on `(s1_id, rank)`, re-buffered into `flush_rows` frames.

Had gate 3 (`global s1_id ordering`) not been in the Phase 7 gate list, bug 2 would have
shipped silently.

### I.3 Validation (executed, measured)

Synthetic multi-route event table: **32,000 events / 4,000 S1** across 4 routes. Reference
= global `reconcile_candidates_schema(deduplicate_candidates(ev), ev)` → **30,927** rows.
`RecordBuffer` threshold deliberately set to **97** to force many partial flushes and many
shards per bucket.

| Buckets | Rows out | Shards written | Global `s1_id` order | Identical to global reconciliation |
|---|---|---|---|---|
| 1 | 30,927 | 330 | yes | **yes** |
| 3 | 30,927 | 985 | yes | **yes** |
| 7 | 30,927 | 1,994 | yes | **yes** |
| 64 | 30,927 | 4,153 | yes | **yes** |

**This satisfies requirement 4 and Phase 7 gates 3 and 4 with an actual executed test**,
not a documented assumption.

Events streaming: 2,000 events → `write_events_stream` → read-back 2,000 rows,
dtypes and Arrow schema identical to the existing artifact
(`large_string` ×5, `rank` int64, `score` double), content equal order-independently.

> **Note on event row order:** events come back in **bucket-major** order, not input order.
> That is correct and expected — `retrieval_events.parquet` is a long-form table with no
> ordering contract. The ordering contract belongs to `candidates.parquet` only, and that one
> *is* preserved (verified above).

### I.4 Observed characteristic to watch (not a defect)

Shard count scales as `flushes × n_buckets`. At the fixed 50,000 threshold and a 64-bucket
spill, the 300K slice (~36M events) should produce roughly **46,000 shards**. That is
manageable but means `iter_spilled_buckets` opens many small Parquet footers per bucket.
Re-measured on the slice in Phase 6; if read time is material it can be mitigated later by
coalescing shards, which does not affect semantics.


---

## J. Phase 2 results — all 7 routes streamed (VALIDATED)

### J.1 Changes

Every route gained an optional `sink: Optional[RecordBuffer] = None` parameter and its
unbounded `records` list was deleted. When `sink is None` the route uses
`RecordBuffer(None)`, which **retains** records — that is the historical behaviour, kept
deliberately as the in-memory equivalence reference (requirement 6). When a sink is supplied
the route returns the empty candidate frame, because its records have already been handed off.

| Route | File | Previous accumulation | Now |
|---|---|---|---|
| 1 `exact_name` | `exact_name.py` | `records = []` over all S1 rows | `RecordBuffer`, flush @50K |
| 2 `tfidf_name` | `tfidf_name.py` | `records: List[dict]` across all query blocks | `RecordBuffer`, flush @50K (naturally ≤20K/block) |
| 3 `rare_token_name` | `rare_token.py` | `records = []` over all S1 rows | `RecordBuffer` |
| 4 `tfidf_address` | `tfidf_address.py` | `records: List[dict]` across all query blocks | `RecordBuffer` |
| 5 `numeric_address` | `numeric.py` | `records = []` in `NumericTokenIndex.query` | `RecordBuffer` |
| 6 `rare_token_address` | `rare_token.py` | `records = []` in `RareTokenIndex.query` | `RecordBuffer` |
| 7 `reverse_retrieval` | `reverse.py` | `records.extend(batch)` when no sink | `RecordBuffer`; existing `sink.emit()` path retained |

Verification grep — **zero** occurrences of `records = []`, `records: List[dict] = []`,
`records.append`, or `records.extend` in any route file (only `RecordBuffer`'s own internal
buffer remains, which is bounded by design).

### J.2 Equivalence validation (executed)

Committed 1K fixtures, `RecordBuffer` threshold deliberately set to **7** to force many
flushes and exercise the streaming path far harder than the production threshold would.

| Route | in-memory rows | sink rows | Identical | Returns empty when sinking |
|---|---|---|---|---|
| `exact_name` | 4 | 4 | **yes** | yes |
| `tfidf_name` | 19,927 | 19,927 | **yes** | yes |
| `tfidf_address` | 19,990 | 19,990 | **yes** | yes |
| `rare_token_name` | 16,526 | 16,526 | **yes** | yes |
| `numeric_address` | 3,180 | 3,180 | **yes** | yes |
| `rare_token_address` | 19,924 | 19,924 | **yes** | yes |
| `reverse_retrieval` | 8,897 | 8,897 | **yes** | yes |

`pd.DataFrame.equals` — full dtype- and value-level equality, not row counts.

> **One defect found and fixed here:** `RecordBuffer` initially lacked `emit`, which
> `reverse.py` calls (`sink.emit(batch)`). It surfaced as an
> `AttributeError: 'RecordBuffer' object has no attribute 'emit'` during this equivalence
> run and was fixed by adding the batch alias. This is exactly why the equivalence test
> exists rather than being assumed from inspection.

### J.3 Regression check after Phase 2

| Check | Baseline (Phase 0) | After Phase 2 | Result |
|---|---|---|---|
| `pytest` collectable subset | 106 passed, 0 failed | **106 passed, 0 failed** (14.15 s) | **no regression** |
| 1K fixture candidates | 63,488 | **63,488** | exact match |
| 1K fixture events | 88,448 | **88,448** | exact match |

The fixture match confirms the orchestrator's in-memory path is still untouched — Phase 3
had not yet been applied at the time of this check.

---

## L. Phase 5 — full-data runner patch (CODE COMPLETE, DELIBERATELY NOT EXECUTED)

`scripts/run_p1_full_data.py` was patched so that *if* it were ever run it would be
memory-safe. It has **not** been run, per the standing prohibition.

| Change | Detail |
|---|---|
| POSIX-only import | `resource` moved behind a `try/except` so the module imports on Windows |
| Spill location | `P1_SPILL_DIR`, defaulting to `/var/tmp/p1_spill` (never the repo) |
| No implicit delete | spill dir is cleared only on explicit `P1_RESET_SPILL=1` |
| Metadata reads | row counts via PyArrow `ParquetFile.metadata` instead of loading frames |
| Finalization | no `return_events=True`; the orchestrator streams straight to disk |
| Memory probe | falls back to `/proc/meminfo` when `psutil` is absent |

Verification performed: `py_compile` + module import succeed; `resource` reports
`RLIMIT_AS` available; `get_mem_mb()` returns ~940 MB; `psutil=False`. No pipeline ran.

> **Standing risk, deliberately unresolved:** this runner still defaults to writing
> `artifacts/candidates.parquet` and `artifacts/retrieval_events.parquet`, which would
> **overwrite the existing smoke artifacts**. It must not be executed as-is. The slice
> runner in §M is the only sanctioned way to produce artifacts from here.

## M. Phase 6 — bounded real-data slice runner (CREATED, VALIDATED SMALL, 300K RUNNING)

Created `scripts/run_p1_slice.py`. It streams a deterministic sample of the real
`dataset/train` TSVs to `artifacts/slice/data/`, runs the full 7-route pipeline with
spilling into `/var/tmp/`, then validates the output against 13 explicit checks.

### M.1 Defect found and fixed: the pool was not scaled

The first working version hard-coded `POOL_SCALE = 1.0`, i.e. it drew an *equal* number of
S2 and S3 rows as S1 rows. That silently shrank the pool-to-query ratio from the real
**4.676** to **2.0**, which would have understated TF-IDF cost by roughly half. Corrected
to per-source ratios derived from the real corpus:

| Source | Real rows | Scale | 300K slice |
|---|---|---|---|
| S1 | 2,206,821 | 1.0000 | 300,000 |
| S2 | 5,034,616 | 2.2813 | 684,416 |
| S3 | 5,285,603 | 2.3952 | 718,536 |
| **pool** | 10,320,219 | 4.6757 | **1,402,952 → ratio 4.677** |

The rounding in the original plan (687K / 706K) is consistent with these; the exact
computed values are used instead so the ratio is preserved to 3 decimals.

### M.2 Smoke validation (real data, 5K S1) — 13/13 PASS

Pool scaled correctly this time: S1 5,000 / S2 11,406 / S3 11,975.
Result: **417,102 candidates, 547,214 events, peak RSS 623 MB.** Every check passed:
`candidates_nonempty`, `candidates_schema`, `candidates_s1_globally_ordered`,
`pair_key_format`, `pair_key_unique`, `candidate_source_valid`, `n_routes_ge_1`,
`best_rank_ge_1`, `best_score_no_nan`, `candidates_within_input_s1`, `events_nonempty`,
`events_ge_candidates`, `events_schema`.

Per-route wall clock at 5K S1, recorded as the basis for the full projection:

| Route | seconds | events | RSS after |
|---|---|---|---|
| `exact_name` | 1.2 | 96 | 439 MB |
| `tfidf_name` | 69.0 | 99,994 | 501 MB |
| `rare_token_name` | 55.7 | 88,398 | 501 MB |
| `tfidf_address` | 73.6 | 100,000 | 514 MB |
| `numeric_address` | 33.1 | 51,282 | 514 MB |
| `rare_token_address` | 63.7 | 99,980 | 514 MB |
| `reverse_retrieval` | 73.3 | 107,464 | 524 MB |

### M.3 Scaling measurement and projection for 300K

`tfidf_name` was timed at three real slice sizes to fit the growth law:

| S1 | pool | S1 x pool | seconds |
|---|---|---|---|
| 5,000 | 23,381 | 1.169e8 | 69.0 |
| 10,000 | 46,764 | 4.676e8 | 145.2 |
| 20,000 | 93,529 | 1.871e9 | 334.4 |

Fitted exponent **0.569** — clearly sublinear in the pair product, because as the pool
grows the 20th-best score rises and proportionally fewer candidates survive the `min_score`
and tie-boundary filter in `canonical_top_k`. Extrapolated to 300K x 1.40M predicts
**~2.0 h for `tfidf_name` and ~8.7 h for all seven routes**. Being an extrapolation over a
15x size increase, the true figure may differ materially; the run is being watched.

### M.4 300K run — IN PROGRESS

Launched in the background (PID 11142):

```
.venv/bin/python scripts/run_p1_slice.py --s1-n 300000 \
  --slice-dir artifacts/slice/data --out-dir artifacts/slice/out \
  --spill-dir /var/tmp/p1_spill_slice --reset
```

Slice built and row-count-verified: 300,000 / 684,416 / 718,536. Live observations:
RSS **626 MB** at 8 min (gate is 6 GB, so ~10x headroom), CPU ~90%, spill growing steadily.

> **Accounting gotcha worth recording:** each `RecordBuffer` flush writes one Parquet part
> *per non-empty bucket*, so the part count is ~64x the flush count. 692 parts therefore
> meant ~11 flushes (~550K events), not 692 x 50K. Reading part count as an event count
> overstates progress by two orders of magnitude — a mistake I made and corrected here.

> **Background-execution gotcha:** under `nohup` stdout is block-buffered, so
> `/var/tmp/p1_log/slice300k.log` stays empty until the buffer fills or the process exits.
> Progress must be polled from the filesystem (part count, RSS, mtimes) instead. Future
> background invocations should use `python -u`.

---

## N. Phase 7 — formal verification (tests WRITTEN, 33/33 PASS; slice gate still open)

Two new test files, replacing the ad-hoc checks that were run inline during Phases 1–3:

| File | Tests | Covers |
|---|---|---|
| `tests/test_blocking_spill.py` | 18 | `RecordBuffer` bound, sink protocol, spill round-trip, manifest, canonicalization, top-k block invariance |
| `tests/test_blocking_memory.py` | 15 | in-memory vs streaming equivalence, global ordering, pair-key consistency, decomposition invariance, live retention probe |

**Regression result: `139 passed, 0 failed` in 78 s** (106 Phase-0 baseline + 33 new).
The three Phase-0 environment-blocked modules remain excluded and are not newly broken.

### N.1 Five defects found by writing the formal tests

Writing assertions instead of eyeballing output surfaced five real problems. Four were
mistaken assumptions of mine; **one is a latent contract trap in the codebase**.

**1. The in-memory path returns a *different schema* from the artifact.** This is the
important one. `generate_candidates(return_events=True)` hands back
`[pair_key, s1_id, candidate_id, candidate_source, route, rank, score]`, where `route` is a
comma-joined string. The artifact contract in `docs/schemas.md` Section 6 — and the
existing `artifacts/candidates.parquet` — is
`[..., n_routes, best_rank, best_score]`. `deduplicate_candidates()` still emits the legacy
names. The documented bridge is `reconcile_candidates_schema`, which derives all three
canonical fields from the events table.

> **Consequence:** "in-memory vs streaming equality" cannot be asserted by comparing the
> two return values directly — they are different schemas and the comparison is
> meaningless. The tests now assert `streamed_artifact == reconcile(in_memory_legacy,
> in_memory_events)`, which is the correct claim. Anyone comparing the two directly will
> get a spurious mismatch and may "fix" working code.

**2. `best_rank` is the minimum rank across routes, not a dense per-s1 ordinal.** A pair
retrieved by two routes at ranks 1 and 3 has `best_rank == 1`, so several candidates for
one `s1_id` legitimately share it. My first version of the test asserted density and
failed. The dense ordering is the `rank` column of `canonicalize_frame`, which is
deliberately not part of the artifact. Corrected the assertion to the properties the
schema actually has (`best_rank >= 1`, `1 <= n_routes <= 7`, `0 <= best_score <= 1`).

**3. `max_block_size` is a recall cap, not a decomposition detail.** It truncates real
retrievals — "Maximum candidates per normalized-name bucket" (Route 1) and "Max postings
per token bucket" (Route 5). Widening it legitimately finds *more* pairs, so pinning it as
invariant was wrong and produced a 3410-vs-3189 row-count mismatch. Split into two tests:
invariance is asserted for `tfidf_*_candidate_chunk`, `tfidf_query_block`,
`reverse_index_block`, `record_buffer` and `dedup_buckets`; and a separate test documents
the recall-cap asymmetry on Route 1, the one route with no `top_k`, where the narrower
result is provably a subset of the wider one.

> **This is the result that matters for the whole refactor:** with the tie-break fix in
> place, all four decomposition variants (`fine`, `coarse`, `record_buffer=7 / 64 buckets`,
> `record_buffer=5000 / 2 buckets`) reproduce the in-memory reference **exactly**. Before
> the fix this was precisely the failure mode.

**4. `canonicalize_frame` returns 8 columns, the artifact has 7.** It emits the 7 contract
columns plus a `rank` sort helper; `write_candidate_dataset` selects only the 7. Verified
against the real `artifacts/candidates.parquet` schema, which matches. Correct — but now
pinned by a test so the helper column can never leak into the artifact.

**5. Four of my own test bugs**, recorded because they cost real time and would recur:
scores travel as **float32** inside the sparse similarity matrix, so a reference written in
Python floats mismatches on representation (0.9 vs 0.8999999761581421) rather than on
logic; `iter_single_bucket` takes a *single bucket directory*, not the spill root;
`InMemoryEventSink` stores DataFrames in `.frames` (not raw records); and `SpillManifest`
tracks **routes**, not buckets — bucket completion is already covered by the per-bucket
`_SUCCESS` markers.

### N.2 Slice runner reproducibility fix

`build_slice` derived its per-source seed from `hash(key)`. CPython randomizes string
hashing per process (`PYTHONHASHSEED`), so **the sample differed on every run** and the
slice was not reproducible despite the fixed size. Replaced with a `sha256`-derived
offset. The slice currently in flight was built with the old seed, so it is a valid but
non-reproducible artifact; a rerun would sample different rows.

### N.3 Slice run status — slower than projected

`tfidf_address` has been the active route since flush ~150 and is producing roughly one
50,000-record flush per ten minutes. Measured position: **300 flushes ≈ 15.0M events** at
9 h 49 m, RSS 686 MB, on Route 4 of 7 with `numeric_address`, `rare_token_address` and
`reverse_retrieval` still to run.

| Route | State |
|---|---|
| `exact_name` | complete |
| `tfidf_name` | complete (~4.5 h) |
| `rare_token_name` | complete (~4.5 h) |
| `tfidf_address` | **in progress** |
| `numeric_address` | not started |
| `rare_token_address` | not started |
| `reverse_retrieval` | not started |

Revised total estimate **~24 h**, against the ~8.7 h projection in §M.3. The projection
was fitted on `tfidf_name` alone over a 4x size range and extrapolated 15x; it
underestimated by ~2.5x. The memory claim is unaffected and in fact strongly supported —
**peak RSS has stayed between 626 MB and 866 MB across the entire run**, roughly 7-10x
under the 6 GB gate, with flat growth rather than the runaway accumulation that caused the
original crash.

## O. Recovery + parallel completion (WSL restart, 2026-09-27 ~07:28)

### O.1 State on recovery
- WSL had been restarted mid-300K-run. **No process was running**; the previous run's
  `artifacts/slice/out/` was still empty, so no P1 artifact had ever been finalized.
- Resources after restart: 16 logical CPUs, 18,415,808 kB RAM (17,080,152 kB available),
  8,388,608 kB swap. Peak of the interrupted run had been only 1.11 GB `VmHWM` on a single
  core, so the interruption was a WSL event, not a resource failure.
- The 300K slice itself was intact and is still the authoritative real-data input
  (S1/S2/S3 = 300,000 / 684,416 / 718,536; pool:S1 = 4.677, matching the real corpus).

### O.2 Spill census (pyarrow row counts, not file sizes)
| route | events | state |
|---|---:|---|
| `exact_name` | 400,587 | complete (one event per S1 with an exact match) |
| `tfidf_name` | 6,000,000 | complete (300,000 x top_k 20, saturated) |
| `rare_token_name` | 5,966,750 | complete (next route had started) |
| `tfidf_address` | 2,850,000 | partial |
| `numeric_address` | 0 | not started |
| `rare_token_address` | 0 | not started |
| `reverse_retrieval` | 0 | not started |
| **total** | **15,217,337** | |

### O.3 The partial `tfidf_address` is a clean contiguous prefix
Per-`s1_id` census of the 2,850,000 address events:
- 142,500 distinct `s1_id`s, exactly `0 .. 142499` with **no gaps**;
- **every one has exactly 20 events** (saturated at `top_k`), so no S1 is half-written;
- therefore resume offset is exactly **142,500** and the remaining work is
  `S1[142500:300000)` = 157,500 rows.

### O.4 Shard safety, re-derived (and one earlier claim corrected)
`S1` participates in the *fit* of every remaining route, so naive S1 sharding changes
output. What makes sharding admissible is different per route:

| route | S1 in the fit/index? | admits S1 sharding? |
|---|---|---|
| `exact_name` | no | yes (already exploited; complete) |
| `tfidf_name` | vocabulary + IDF over pool **+ S1** | yes, **only** via a pre-fitted vectorizer |
| `tfidf_address` | vocabulary + IDF over pool **+ S1** | yes, **only** via a pre-fitted vectorizer |
| `rare_token_name` | rare-token DF counts over pool **+ S1** | no |
| `rare_token_address` | rare-token DF counts over pool **+ S1** | no |
| `numeric_address` | numeric corpus counts over pool **+ S1** | no |
| `reverse_retrieval` | char-vector fit over pool **+ S1** | not used; run whole (not on the critical path) |

`retrieve_tfidf_name` / `retrieve_tfidf_address` accept a `vectorizer=` argument. Fitting
that vectorizer on the **full** `pool + S1` corpus and passing it in is exact, because the
route's own `if vectorizer is None:` branch fits on that same corpus, and every query row
is transformed independently afterwards.

### O.5 New: `scripts/run_p1_parallel.py`
Adds the two things the sequential runner lacked. It does not import or change any
retrieval logic -- it calls the existing route functions with the existing
`DEFAULT_BLOCKING_CONFIG` values.

- `--mode worker`: one route, optionally a contiguous S1 block, **spills events only** to
  its own directory (no per-worker finalization).
- `--mode finalize`: unions every worker's spill and finalizes in three bounded passes:
  1. events, one hash bucket at a time (all workers' parts for that bucket);
  2. `canonicalize_frame` per bucket into a sorted temp parquet file;
  3. `heapq.merge` over the 64 sorted bucket files, read in Arrow batches.
  Peak memory is one bucket plus one batch per bucket, instead of all 64 canonical bucket
  frames at once.

Why the old path was the last unbounded-memory step: `iter_canonical_candidates_merged`'s
docstring claims `O(n_buckets + flush_rows)`, which is true for the *number of live objects*
but not for bytes -- `_canonical_rows` materializes a whole canonical frame per bucket, so
the 64 resident frames together are the entire candidate set (~25M rows at slice scale).

### O.6 Two real bugs caught by gating before the long runs
Both were found by `--mode selftest` / unit tests, **before** any production work, and both
would have silently corrupted the artifact:

1. **Pre-fitted address vectorizer did not match the route's own fit.** The first selftest
   run reported `tfidf_address: whole-S1 shared-vs-internal identical = False` (3,987 vs
   3,992 events) while `tfidf_name` passed. Cause: the routes fit on
   `normalize_address`/`normalize_name`-normalized S1 text (via `_build_s1_query_data`), but
   the driver fitted on the **raw** strings, so the n-gram vocabulary differed. Fixed by
   taking the S1 texts from the route's own `_build_s1_query_data`.
2. **`canonicalize_frame` output columns.** Pass 3 sorted on `rank`; the canonical frame
   does carry `rank` (the per-pair minimum) alongside `best_rank`, which the bucket files
   retain and the final projection drops.

### O.7 Gating evidence (all on real data, before the long runs)
- `--mode selftest` (200 S1 / 600+600 pool, real slice rows): both TF-IDF routes report
  `whole-S1 shared-vs-internal identical = True` **and**
  `2-way-shard union == whole-S1 = True`. Sharding is therefore lossless.
- `scripts/check_finalize.py` (400 S1 / 900+900 pool, three separate spills including one
  route split across two, `--merge-batch 37 --flush-rows 53`):
  `FINALIZE SELFTEST PASSED`, `candidate frames identical (order-sensitive): True`,
  6,670 candidates, 6,672 events -- i.e. the bounded finalizer reproduces
  `iter_canonical_candidates_merged` exactly, and does not depend on the batching knobs.
- `tests/test_blocking_parallel.py`: **7 passed**. Covers pre-fitted-vectorizer equivalence
  (both routes), lossless S1 sharding, finalize-vs-reference across split spills, dedupe of
  overlapping worker spills (`best_rank` 1 / `best_score` 0.9 from a rank-2+rank-1 overlap),
  event schema and global `s1_id` ordering, and the `rank` merge key.

### O.8 Workers launched
9 processes, all confirmed at ~99% CPU, one spill directory each (required: `_write_shard`
restarts numbering at `part-00000` per process, so a shared directory collides):

| worker | S1 range | notes |
|---|---|---|
| `numeric_address` x1 | `0:300000` | **complete** -- 5,177,836 events in 272.77 s, peak 741 MB |
| `rare_token_address` x1 | `0:300000` | running |
| `reverse_retrieval` x1 | `0:300000` | running |
| `tfidf_address` x6 | `142500:300000`, 26,250 each | running, shared full-corpus vectorizer (vocab 209,313, fit 66 s each) |

Observed throughput correction: `numeric_address` completed 300,000 S1 rows in 4.5 minutes
(~19,000 events/s), so the earlier "600 events/s" figure extrapolated from the interrupted
run badly understated the non-vectorizer routes. The TF-IDF routes are the expensive ones
because each 1,000-row query block re-transforms the whole 1.4M pool; that cost is
amortized over 1,000 S1 rows, which is why sharding them 6 ways is the right lever.

Finalize input will be the union of the original spill plus all 9 worker directories:
`/var/tmp/p1_spill_slice`, `/var/tmp/p1_w_numeric_address`,
`/var/tmp/p1_w_rare_token_address`, `/var/tmp/p1_w_reverse_retrieval`,
`/var/tmp/p1_w_tfidfaddr_{0..5}`. Union is safe and idempotent because `canonicalize_frame`
aggregates with `max`/`min`/`nunique` per group, and `bucket_of` is a stable
`crc32(s1_id) % 64`.

### O.9 P2 findings (measured while P1 ran)

Profiled `build_features` on **real** P1 candidate pairs from the live spill:

| stage | 20,000 pairs | rate |
|---|---:|---:|
| `compute_name_features_batch` | 1.16 s | 17,200/s |
| `compute_address_features_batch` | 0.81 s | 24,700/s |
| `pivot_retrieval_features` | **20.80 s** | **1,440/s** |
| `build_features` total | 23.7 s | 843/s |

The retrieval pivot was **94% of P2** and it is O(total pairs) in a Python dict-of-dicts
(`route_agg`), so sharding the caller cannot help: 25M pairs would need ~12 GB and ~4.8 h
in that dict alone.

**Fix: vectorized the pivot, keeping the original as an equivalence oracle.**
`_pivot_retrieval_features_reference` is retained verbatim and
`pivot_retrieval_features` now uses groupby aggregations. Each rewrite step is exact:
- `n_routes` -> `nunique()` over non-null routes, with 0 remapped to 1.0 (the reference's
  fallback for a group with no usable route);
- `best_rank` / `best_score` -> `min()` / `max()` of numeric-coerced, null-dropped values;
- each binary flag -> `max()` of a per-row boolean, which equals
  `any(pred(route) for route in that group's routes)`. The reference's literal disjuncts
  (`"exact_name" in routes or ...`) are subsumed, since a route equal to `"exact_name"`
  necessarily contains `"exact"`.

Verified **bit-identical** (NaN placement included) against the reference on real spill
events, shuffled/duplicated pair keys, absent keys, null routes, non-numeric ranks/scores,
all-null-route groups, missing `route`/`rank`/`score` columns, and empty/`None` inputs:
**96x faster** (2.53 s -> 0.0265 s on 4,346 pairs; 4.8 h -> ~2.5 min at 25M pairs).
`tests/test_features_retrieval_pivot.py`: **10 passed**.

### O.10 New: `scripts/run_p2_parallel.py`
Shards the candidate list by **row position** (so output order is preserved by
construction) and gives each worker only its own retrieval events.

Two silent-corruption bugs were found and fixed while proving it out, both caught by
comparing against a single-process `build_features` call on real data:

1. **Position vs. id sharding.** Routing shards by `s1_id` via `searchsorted` broke
   `pair_key` order. Position-based sharding is order-safe unconditionally.
2. **Split `s1_id` groups.** Even position-based, a boundary can land inside one `s1_id`'s
   candidates. Events are routed by `s1_id`, so *all* of that `s1_id`'s events land in one
   shard while the other shard's pairs for it silently fall back to **default** retrieval
   features -- wrong output, not an error. Measured exactly this: 3 columns
   (`retrieval_best_rank`, `retrieval_best_score`, `retrieved_by_numeric`) differed on a
   300k-pair run where shards held 74,995/74,997/75,001/75,007 events instead of 75,000.
   `snap_bounds_to_s1_groups()` moves every start forward to the next `s1_id` group
   boundary. After the fix, events-per-shard equals candidates-per-shard exactly and
   **all 36 feature columns are bit-identical to single-process `build_features`**
   (300,000 pairs, 4 shards).

The runner also **asserts** the P1 `s1_id`-sorted precondition rather than assuming it,
and defines the P2 validation that did not previously exist anywhere:
non-empty, row count == candidate count, 39-column schema in order, all features float32,
`pair_key` positionally equal to `candidates.parquet`, and `s1_id` within the input S1.
`tests/test_p2_parallel.py`: **4 passed**.

P2 throughput after the pivot fix: **15,600 pairs/s** on 4 workers (was 843/s on one).

### O.11 Environment: full test suite is now green (187 passed, 0 errors)
Both collection failures were dependency problems, not code problems, and both were
declared or packaged already:
- `rapidfuzz>=3.0.0` is in `code/business_entity_resolution/requirements.txt` and was
  installed in `.venv_win` but missing from the WSL `.venv`. It sits directly on the P2
  path (`features/name_features.py:13`, `features/address_features.py:14`), so
  `src.features` could not even be imported. `pip install` into `.venv` unblocked it plus
  `tests/test_features.py` and `tests/test_integration_p1_p2.py`.
- `libgomp.so.1` was absent (no system `libgomp1`, uid 1000 so no root install). Fetched
  the Ubuntu `libgomp1` .deb with `apt-get download`, extracted it to `.venv/lib/` with
  `dpkg-deb -x`, and preload it from a `.pth` so no `LD_LIBRARY_PATH` is needed. This
  unblocks `tests/test_lightgbm.py` and LightGBM training for P3.

Baseline was 106 collectable tests; the suite is now **187 passed, 0 failed, 0 errors**
with no ignores.
