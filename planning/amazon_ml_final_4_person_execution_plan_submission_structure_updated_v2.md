# AMAZON ML CHALLENGE — FINAL 4-PERSON EXECUTION PLAN

---

## 0. One-page overview

The pipeline is a two-stage system: cast a wide, cheap net (candidate generation), then score and decide precisely (features → model → threshold). The four people map onto four stages, but **nobody waits on the person "above" them** — everyone starts hour 0 against mocked/synthetic inputs and swaps to the real upstream artifact the moment it lands.

```
P1: Blocking / Candidate Generation
        ↓ (retrieval_events.parquet)
P2: Normalization + Pairwise Features         <- starts immediately on the 76 real
        ↓ (features.parquet)                     training examples, not P1's output
P3: ML / Validation / Decision Layer          <- starts immediately on synthetic
        ↓ (scores.parquet)                       features, not P2's output
P4: Integration / Output / Submission         <- starts immediately on fabricated
        ↓                                        predictions, not P3's output
matching_results.tsv + candidate_pairs.tsv
```

The real dependency chain only has to be real **once a day**, at the checkpoints in Section 6 — everyone else's first hours are spent building against mocks precisely so the chain being incomplete never blocks anyone. The first genuinely wired, end-to-end, real-data run happens at **Hour 6**, deliberately on a small subsample so integration bugs are cheap to find.

### 0A. Development repository vs final submission structure

The **development repository** may contain tests, logs, experiment trackers, configs, temporary artifacts, and scripts. The **final ZIP must match the competition structure exactly**. We therefore keep the production source tree aligned with the required `code/business_entity_resolution/src/` tree from the beginning, while keeping development-only material outside it.

Development repository:

```text
amazon-ml-challenge/
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── preprocessing/
│   ├── blocking/
│   ├── features/
│   ├── validation/
│   ├── models/
│   ├── decision/
│   ├── inference/
│   └── utils/
├── tests/
├── configs/
├── docs/
├── logs/
├── experiments/
├── scripts/
├── README.md
├── .gitignore
└── requirements-dev.txt              # optional development-only environment

```

The final submission is assembled from that repository into exactly:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   ├── preprocessing/
│       │   ├── blocking/
│       │   ├── features/
│       │   ├── validation/
│       │   ├── models/
│       │   ├── decision/
│       │   ├── inference/
│       │   └── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

**Important:** `requirements.txt` is inside `code/business_entity_resolution/`, alongside `README.md`. It is **not** at the ZIP root and it is **not** inside `output/`.

**Development `.gitignore` guidance:** do not ignore every TSV with a blanket `*.tsv` rule, because the required `output/matching_results.tsv` and `output/candidate_pairs.tsv` must remain visible to packaging. Ignore dataset/intermediate TSV locations explicitly (for example `dataset/**/*.tsv` and `tmp/**/*.tsv`) rather than all TSVs globally.

Production source ownership is:

- **P1:** `src/blocking/`
- **P2:** `src/preprocessing/` and `src/features/`
- **P3:** `src/validation/`, `src/models/`, and `src/decision/`
- **P4:** `src/data/`, `src/inference/`, and `src/utils/`

P4 owns the packaging scripts and development-only integration files, but does not own the ML logic inside P1/P2/P3's directories.

The final `code/business_entity_resolution/README.md` must describe how the files under `src/` are called to reproduce both official TSV outputs from the supplied training/test data. The final `requirements.txt` must pin the exact versions used by the final clean-environment run.

Evidence already established (not re-derived here, just enforced): candidate generation must include address-based routes, not just name routes (a true match can share almost no name similarity but match on address); missing fields are evidence-free, not negative evidence; numeric address mismatches are soft signals; training-data analysis suggests a one-to-one S2/S3-to-S1 structure that may support a reverse-consistency diagnostic/rule, but this is not an official task guarantee and is unverifiable on test.

---

## 0B. How each person uses the source tree

The four people work in separate production directories, with explicit handoffs:

```text
P1  src/blocking/
      │
      ├── retrieval_events.parquet
      └── candidates.parquet
                │
                ▼
P2  src/preprocessing/ + src/features/
      │
      └── features.parquet
                │
                ▼
P3  src/validation/ + src/models/ + src/decision/
      │
      └── scores.parquet + frozen decision config
                │
                ▼
P4  src/inference/ + src/data/ + src/utils/
      │
      ├── matching_results.tsv
      └── candidate_pairs.tsv
                │
                ▼
      final submission ZIP
```

- **P1 imports:** `src/data/loaders.py` and P2's normalized columns when they become available; P1 writes only candidate-generation artifacts.
- **P2 imports:** shared loaders plus P1's `retrieval_events.parquet`; P2 writes normalized records and pairwise features.
- **P3 imports:** P2's `features.parquet`, P1's `candidates.parquet`, and the official training ground truth; P3 owns labels, folds, model scores, and decision logic.
- **P4 imports:** all three upstream production modules/artifacts and is responsible for orchestration, output formatting, validation, and final packaging. P4 does not rewrite P1/P2/P3 logic.
- **All four:** may use `tests/`, `docs/`, `logs/`, `configs/`, and `experiments/` during development. These are coordination/development material, not automatically part of the final ZIP.

## 1. Shared Contracts

Every artifact below is a Parquet file with `pair_key = s1_id + "::" + candidate_id` as a string column wherever the row represents a pair. All ID columns are strings, never inferred numeric types.

**Candidate-set acceptance principle:** `candidate_pairs.tsv` is a first-class evaluated deliverable. The team should treat candidate generation as a constrained optimization problem: maintain the required recall/oracle-ceiling gate, then, among configurations that satisfy that gate, prefer smaller candidate sets per S1 / higher reduction ratio. Candidate volume is therefore both a runtime concern and a final submission-quality metric.

| Artifact | Owner | Columns | Consumed by | Must exist (real) by | Frozen at |
|---|---|---|---|---|---|
| `records.parquet` | P2 | `entity_id, source, raw_name, raw_address, raw_country, norm_name_cons, norm_name_aggr, norm_address_cons, country_norm` | P1, P2, P3 (via features) | Hour 4 (subsample), Hour 8 (full) | Normalization logic: Hour 24 |
| `folds.parquet` | P3 | `s1_id, fold_id` | P3, P4 | Hour 3 | Hour 3 — never touched again |
| `retrieval_events.parquet` | P1 | `pair_key, s1_id, candidate_id, candidate_source, route, rank, score` | P2 (pivots into features), P4 (provenance/debugging) | Hour 4 (subsample), Hour 10 (full train), Hour 20 (full test) | Route set: Hour 12 |
| `candidates.parquet` | P1 | `pair_key, s1_id, candidate_id, candidate_source, n_routes, best_rank, best_score` (aggregated view of `retrieval_events`) | P3 (recall eval), P4 | Same as `retrieval_events` | Same as `retrieval_events` |
| `features.parquet` | P2 | `pair_key, s1_id, candidate_id,` [name/address/country/missingness/retrieval features — full list in Section 3] | P3 | Hour 5 (subsample), Hour 11 (full train), Hour 22 (full test) | Feature list: Hour 24 |
| `train_labels.parquet` | P3 | `pair_key, y` (built by joining `candidates.parquet` against `train_ground_truth.tsv`) | P3 | Hour 6 | N/A (derived fresh whenever candidates change, up to Hour 12) |
| `scores.parquet` | P3 | `pair_key, s1_id, candidate_id, p_raw, p_cal, fold_id, is_oof, model_version` | P3 (decision layer), P4 | Hour 12 (LR), Hour 20 (LightGBM) | Model/hyperparameters: Hour 36 |
| `matching_results.tsv`, `candidate_pairs.tsv` | P4 | Per official spec | Validator, submission | First real pair: Hour 20 | Pipeline code: Hour 48 |
| `experiment_tracker.csv` | P4 (everyone writes rows) | `experiment_id, hypothesis, change, metric, candidate_recall, avg_candidates, runtime, result, decision, next_action` | Everyone | Hour 3, updated continuously | Never frozen |
| `logs/person{1-4}.md` | Each person owns their own | Free-form timestamped entries: what was tried, why, result, decision | P4 (consolidates into methodology doc) | Hour 0, updated continuously | Never frozen |
| `logs/submissions.md` | P4 | `submission_id, timestamp, commit, candidate_generation_version, model_version, decision_config, candidate_count, validation_status, leaderboard_result` | P4/team | First submission, updated per submission | Never frozen |

**Non-negotiable rules attached to these contracts:**
- `is_oof` must be `True` for any score used in threshold tuning or the reverse-consistency rule. Scores compared to each other in reverse-consistency must both be OOF.
- No feature or route may be derived from `train_ground_truth.tsv` match counts, match frequency, or which pairs are positive. Frequency features must come from raw text or from candidate *appearance*, never from label truth.
- `candidate_pairs.tsv` is generated from the exact same **final candidate DataFrame** that is passed into feature generation/model inference and produces `matching_results.tsv` — never regenerated independently. Any later candidate filtering, deduplication, ranking cut, or unioning changes the final set and must be reflected in `candidate_pairs.tsv`.

---

## 2. PERSON 1 — Candidate Generation / Blocking

**Owns:** candidate recall. Success is measured, not assumed.

### Hours 0–3
- Files: `src/blocking/exact_name.py`, `src/blocking/utils.py`.
- Write a temporary inline `normalize_stub()` (lowercase, strip punctuation, collapse whitespace) — do not wait for P2's real normalizer.
- Implement **Route 1: exact normalized-name retrieval** — hash-map block on `normalize_stub(name)`, cap block size at 500 records.
- Input: P4's 50k-row subsample (request it immediately; if not ready within 30 minutes, self-generate a 1,000-row `df.sample(n=1000, random_state=42)` from train to unblock yourself).
- Output: `retrieval_events.parquet` (subsample) with `route="exact_name"`.
- Success: route runs on the subsample in under 30 seconds; output matches the schema exactly.
- Do NOT: build address or TF-IDF routes yet; do not wait for P2.

### Hours 3–6
- Swap `normalize_stub()` for P2's real `records.parquet` normalized columns once available.
- Implement **Route 2: char n-gram TF-IDF, name** — n-gram range (3,4), top-k=20, chunked: process S1 in batches of 5,000 rows against the full S2/S3 TF-IDF matrix via sparse dot product, keep only per-chunk top-k via `np.argpartition`, discard the rest immediately. Never materialize a full dense or unbounded-sparse S1×S2/S3 matrix.
- Implement **Route 4: char n-gram TF-IDF, address** — same chunking discipline, top-k=20.
- Join the Hour 6 end-to-end checkpoint (Section 6) with real subsample-scale `retrieval_events.parquet` covering routes 1, 2, 4.
- Success: subsample any-match recall reported (no target yet — subsample is for pipeline correctness, not final recall numbers).
- Do NOT: run on full-scale data before Hour 6.

### Hours 6–12
- Scale Routes 1/2/4 to full train (2.2M S1 × 5.0M+5.3M pool) with the same chunking approach, batch size tuned for observed runtime/memory (start at 10,000 S1 rows/batch, adjust down if memory spikes).
- Implement **Route 5: numeric-token retrieval** — inverted index on digit-tokens (`\b\d+\b`) extracted from address; retrieve any candidate sharing ≥1 numeric token; cap block size. Do not require exact numeric match (evidence shows digits get corrupted, e.g. 703→70).
- Implement **Route 3: rare-token name retrieval** and **Route 6: rare-token address retrieval** — inverted index on tokens, restricted to document frequency below a cutoff computed from the actual df distribution of that file (e.g. below the 90th percentile) — do not hard-code a stopword list.
- Implement **Route 7: reverse retrieval** — run the TF-IDF routes with S2/S3 as queries against the S1 pool, union results in.
- Union all routes into `retrieval_events.parquet`; build the aggregated `candidates.parquet` view.
- Compute and report to the team: any-match recall, all-match recall, pair recall, oracle-ceiling macro-F0.5 (coordinate the metric call with P3), total candidate pairs, mean/median/p95/p99/max candidates per S1, reduction ratio, runtime, and memory — broken down by 0-match / 1-match / multi-match S1, and by US / India.
- Candidate-volume objective: first satisfy the team's recall/oracle-ceiling gate; among candidate configurations that pass that gate, prefer the smaller candidate set (lower total volume and lower mean/median/p95/p99/max per S1, with higher reduction ratio).
- Success: any-match recall ≥0.90 with no cliff in any breakdown slice, and the oracle-ceiling F0.5 meets the team's internal gate (for example 0.92; this is a team decision, not an official Amazon threshold).
- Do NOT: add phonetic or embedding retrieval — out of scope until justified by a specific measured miss.

### Hours 12–24
- Freeze the route set once Hour 12 numbers are reviewed by the team (Section 9).
- If a specific slice is short on recall (e.g. multi-match-8+, or India), tune that route's parameters narrowly (raise k for that route, lower the rare-token df cutoff) — not a blanket increase everywhere.
- Generate `retrieval_events.parquet` + `candidates.parquet` for the full test set, as a config path-swap of the same code, not separate code.
- Hand off frozen candidate-generation code to P4 for the single inference script.
- Success: full-train and full-test candidate generation both complete within the agreed runtime budget (report actual wall-clock to the team).
- Do NOT: keep experimenting with new routes after freeze — log ideas in `experiment_tracker.csv` under "next_action" for later only.

### Hours 24–36
- Support P3's error analysis: for every reported miss (true pair absent from candidates), classify which route should have caught it and why it didn't.
- Reopen a route only with a specific, measured recall gap behind it; implement a narrow fix (e.g. widen numeric-token matching tolerance), not a redesign.
- Confirm blocking routes are script-agnostic for the France/LOCO check (char n-gram and numeric-token routes require no code change; report non-ASCII-subset recall to P3).
- Do NOT: touch candidate-generation code without a specific measured reason.

### Hours 36–48
- Candidate-generation code is frozen. Verify with P4 that `candidate_pairs.tsv` content matches what was actually scored. Re-run full train+test generation only if a bug fix upstream requires it.
- Write up routes, parameters, and final recall numbers in `logs/person1.md`.
- Do NOT: start new experiments.

### Hours 48+
- On call for any pipeline bug touching candidate generation.
- Support P4's final reproducibility run.
- Finalize the candidate-generation section of the methodology document with real numbers.

---

## 3. PERSON 2 — Normalization / Feature Engineering

**Owns:** the descriptive vocabulary the model reasons over.

### Hours 0–3
- Files: `src/preprocessing/normalize.py`, `src/preprocessing/corpus_stats.py`, `src/features/build.py`.
- Implement **conservative normalization**: Unicode NFKC, case-fold, accent-fold, punctuation→space, whitespace collapse. This becomes the function P1 swaps in at their Hour 3–6.
- Implement **aggressive normalization**: conservative + legal-suffix stripping, where the suffix list is *mined* — compute frequency of the last alphabetic token across a name sample, keep the top N most frequent trailing tokens as candidate suffixes, save to `artifacts/suffix_tokens.json`. Do not hard-code a suffix list.
- Implement feature functions as pure functions `f(name1, addr1, country1, name2, addr2, country2) -> dict`: `name_char_cos`, `name_token_sort`, `name_containment_a_in_b`, `name_containment_b_in_a`, `addr_char_cos`, `addr_containment_a_in_b`, `addr_containment_b_in_a`, `addr_num_jaccard`, `country_eq`, missingness flags for name/address on each side.
- Unit-test these functions against the 76 real examples from `training_match_examples.md` used as a hand-built mock candidate table — no dependency on P1's output. Include at minimum: Example 66 (Kalyani, must score high on containment despite low symmetric similarity), Example 69 (NEXGILD, must correctly show near-zero name similarity — this is why address features exist), Example 71 (blank-address true match, must produce NaN not 0).
- Success: all unit tests pass.
- Do NOT: wait for `candidates.parquet` to exist.

### Hours 3–6
- Swap feature functions to run on P1's real `retrieval_events.parquet` (join on `pair_key`) once available.
- Implement retrieval-derived features: pivot `retrieval_events` into per-route rank/score columns, compute `n_routes`, `best_rank`, `best_score`.
- Join the Hour 6 checkpoint with real subsample-scale `features.parquet`.
- Success: no unexpected NaNs where evidence exists — a retrieved pair should have at least one non-NaN route score.
- Do NOT: add pool-frequency-of-true-matches or any other ground-truth-derived feature — this is a leak, flagged explicitly in Section 9's audit history.

### Hours 6–12
- Scale normalization + `corpus_stats` + feature computation to full train, vectorized (pandas/numpy/rapidfuzz batch functions — no `itertuples()` loops at this scale).
- Compute the corpus-derived suffix/DBA-marker list at full scale (frequency-based, label-free); save to `artifacts/` for both P1 and P2 to reference.
- Success: full feature computation completes within the agreed runtime budget; spot-check 10 known examples from `training_match_examples.md` for sane feature values.
- Do NOT: introduce any feature requiring ground-truth statistics.

### Hours 12–24
- Generate `features.parquet` for the full test set once P1's test candidates exist.
- Support P3: if error analysis flags a specific gap (e.g. a DBA marker not being caught), add a narrow corpus-derived feature (e.g. `dba_marker_present` using the mined marker list) — not a broad redesign.
- Feature list freezes at Hour 24 (Section 9). Document every feature's definition in `logs/person2.md`.
- Do NOT: add feature columns after Hour 24 without a team-wide announcement.

### Hours 24–36
- Targeted feature additions only if justified by P3's measured OOF error analysis (e.g. alias/DBA variant-max similarity, or a label-free candidate-appearance-frequency feature).
- Sanity-check features on a small manual sample of synthetic French-suffix-like names (SARL/SAS patterns) since France is unseen in training — confirm no feature silently breaks on unfamiliar suffixes.
- Do NOT: chase feature ideas without a measured OOF gain reported by P3.

### Hours 36–48
- Feature code frozen; bug fixes only.
- Finalize `logs/person2.md`: every feature, its definition, and why it was included or excluded — this feeds directly into `Documentation_template.md`.

### Hours 48+
- On call; support P4's reproducibility run; finalize documentation.

---

## 4. PERSON 3 — ML / Validation / Decision Layer

**Owns:** turning features into a defensible entity-level decision.

### Hours 0–3
- Files: `src/validation/metrics.py`, `src/validation/splits.py`, `src/models/logreg.py`.
- Implement and unit-test the macro-F0.5 metric exactly per spec, including: true singleton correctly predicted empty → 1.0; true singleton with a false positive → 0.0; matched entity predicted empty → 0.0.
- Implement the 5-fold GroupKFold-by-S1 split, stratified by country × has-match × match-count bucket (0, 1, 2, 3, 4, 5–7, 8+), using only `train_ground_truth.tsv` + `train_source1.tsv`'s country column — no dependency on candidates or features.
- Output: `folds.parquet`, committed and frozen immediately once written.
- Success: metric passes ≥5 hand-written edge-case tests; per-fold stratum counts reported and roughly balanced.
- Do NOT: wait for P2's features to start this.

### Hours 3–6
- Build a synthetic `features.parquet` (a few thousand fabricated rows with plausible values and known labels) to build and test the LR training loop end-to-end, independent of P2.
- Implement the OOF training harness: 5-fold loop, fit on 4 folds, predict on the held-out fold, concatenate OOF predictions.
- Join the Hour 6 checkpoint using P1+P2's real subsample-scale artifacts once available; swap mocks out immediately.
- Success: LR trains and produces OOF predictions on synthetic data without error.
- Do NOT: block on real features.

### Hours 6–12
- Swap to P2's real `features.parquet`. Build `train_labels.parquet` by joining `candidates.parquet` against `train_ground_truth.tsv`.
- Implement candidate-recall / oracle-ceiling evaluation code (uses the metric module; numbers are P1's to report, code lives here) on the frozen folds.
- Train the LR baseline on full features; report OOF macro-F0.5 with a simple global threshold.
- Success: LR clearly beats "predict nothing" (~5.58% train singleton rate); oracle-ceiling number reported at the Hour 12 checkpoint.
- Do NOT: tune anything on non-OOF predictions.

### Hours 12–24
- Train LightGBM: `learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1–10`, early stopping (~100 rounds patience) on average precision, two-pass procedure (find median best-iteration across folds, retrain each fold at that fixed round count for honest OOF).
- Implement the global-threshold decision layer, tuned on pooled OOF macro-F0.5.
- Test per-entity example weighting (`1 / candidates_for_that_S1`) against unweighted, since the metric is macro over entities.
- Output: `scores.parquet` with `is_oof` correctly set; tuned global threshold.
- Success: LightGBM OOF macro-F0.5 beats LR outside the noise floor (verify by rerunning with 2–3 seeds).
- Do NOT: hand-tune thresholds by eyeballing individual examples.

### Hours 24–36
- Implement the two-threshold decision rule (lower threshold to accept the first candidate per entity, higher for additional candidates); grid-search both jointly on OOF; compare to the global-threshold baseline via paired bootstrap over entities.
- If enabled, implement the reverse-consistency rule using **only** `is_oof=True` scores from comparable fold context. Treat the apparent one-to-one S2/S3 structure as a training-data observation, not an official test guarantee; verify the assumption empirically before using reverse-consistency suppression.
- Run the LOCO diagnostic (train on US, validate on India, and reverse) to estimate France-transfer risk.
- Run error analysis on OOF false positives/negatives, categorize by feature gap, hand findings to P1/P2.
- Success: two-threshold and reverse-consistency each independently show a bootstrap-CI-positive improvement, or are discarded.
- Do NOT: adopt a decision-layer rule without the bootstrap check.

### Hours 36–48
- Model type, hyperparameters, and decision rules freeze (Section 9). Retrain the final config on full training data (not just OOF folds); hand off the model artifact + decision function to P4.
- Optional: calibration check (reliability plot/Brier score) if time allows.
- Do NOT: start new modeling experiments.

### Hours 48+
- On call during test inference; verify test score distributions look sane, not degenerate.
- Document final model config and thresholds in `logs/person3.md`.

---

## 5. PERSON 4 — Integration / Pipeline / Submission / Tracking

**Owns:** the thing actually working, end to end, reproducibly.

### Hours 0–3
- Files: `src/data/loaders.py`, `docs/schemas.md` (development-only; final contracts are documented in README.md), `src/utils/submission.py`, `src/utils/validate_submission.py`, `configs/base.yaml` (development-only; final config values are copied into the runnable pipeline).
- Write the shared safe loader (`dtype=str, keep_default_na=False`, explicit tab delimiter, blank-field preservation, with chunked-read helpers where needed) — everyone imports this.
- **Top priority:** generate and commit the shared 50k-row deterministic train subsample (fixed seed) within the first hour — this unblocks everyone else.
- Write and circulate `docs/schemas.md` (development-only; final contracts are documented in README.md) (Section 1's table); get explicit sign-off from all three teammates before Hour 1 ends.
- Build the output writer and validator against fabricated data, including deliberately broken cases: duplicate IDs, an entity missing from output, a predicted ID absent from `candidate_pairs.tsv`. Assertions to bake in: exactly one row per target S1 id; no duplicate IDs within a `matched_entity_ids` cell; predicted IDs are a subset of `candidate_pairs.tsv`; no train-set ID appears in test output; empty match is an empty string, never NaN/None.
- Success: the validator correctly rejects every broken fabricated case and accepts a correct one.
- Do NOT: wait on anyone else.

### Hours 3–6
- Set up `experiment_tracker.csv`, the four `logs/person{1-4}.md` files, and `logs/submissions.md`; circulate to the team.
- Build `src/inference/pipeline.py` as the single production entrypoint calling P1's blocking → P2's features → P3's model/decision in sequence, initially against stub/mock modules for whichever pieces aren't ready, swapped for real ones as they land. A thin development wrapper may live under `scripts/` if convenient.
- Define the integration invariant early: the final candidate DataFrame used for feature generation/model inference is the same candidate DataFrame serialized into `candidate_pairs.tsv`; no hidden filtering or independent regeneration is allowed after the final-candidate freeze point.
- Coordinate the Hour 6 end-to-end checkpoint: run the full (possibly partially mocked) pipeline on the subsample with all four people present.
- Success: `run_pipeline.py` runs start to finish on the subsample with no manual intervention, producing output that passes the validator.
- Do NOT: let this slip past Hour 6 — it is the single most important checkpoint in the plan.

### Hours 6–12
- Scale `run_pipeline.py` to full train as P1/P2 scale up; profile runtime/memory; flag any dangerously slow stage (dense-matrix symptoms) back to P1 immediately.
- Enforce frequent-merge Git discipline: small PRs into `main`, at least one merge per person by Hour 12, no branch older than a few hours without merging.
- Prepare `code/business_entity_resolution/README.md`, `code/business_entity_resolution/requirements.txt`, and `Documentation_template.md` as empty shells/templates, filled in as dependencies/components land. Keep development-only notes in the repo's top-level README/docs.
- Success: full pipeline runs on full-train scale within the agreed runtime budget; ≥4 merged PRs (one per person) exist by Hour 12.
- Do NOT: let branches diverge for more than a few hours.

### Hours 12–24
- Run full test-set inference once P1/P2/P3 hand off frozen candidate generation, model, and decision layer; freeze the final candidate DataFrame immediately before feature/model inference, then generate both `matching_results.tsv` and `candidate_pairs.tsv` from that same final candidate set.
- Compute the release candidate-volume report: total test S1 entities, total candidate pairs, mean/median/p95/p99/max candidates per S1, reduction ratio, and count of S1 entities with zero candidates.
- Submit once, purely as a format/pipeline sanity check (Section 8) — record it in both `experiment_tracker.csv` and `logs/submissions.md`.
- Keep `experiment_tracker.csv` and the four logs current, pulled from teammates at each checkpoint.
- Success: first real submission passes the official validator with zero warnings and the candidate-volume report is recorded.
- Do NOT: use the leaderboard score to tune anything.

### Hours 24–36
- Support P1/P2/P3's targeted experiments: re-run the pipeline per change, report runtime/output diffs; maintain versioned config files so any run is reproducible from a config alone.
- Success: every adopted change has a corresponding re-run and tracker entry.
- Do NOT: modify feature or model logic yourself.

### Hours 36–48
- Enforce the pipeline-code freeze; coordinate the final full retrain + full test inference using the frozen configs.
- Begin drafting README.md and `Documentation_template.md` in earnest, using the tracker, all four logs, and `logs/submissions.md` as source material.
- Success: a reviewable README/methodology draft exists by end of Hour 48, with the final candidate-set invariant and clean-room execution convention explicitly documented.
- Do NOT: accept last-minute pipeline changes without team agreement.

### Hours 48+
- See Section 10.

---

## 6. Shared Timeline

| Hour | P1 | P2 | P3 | P4 | Team decision |
|---|---|---|---|---|---|
| **0** | Set up `blocking/`, request subsample | Set up `normalize.py`, start suffix mining | Set up metric + fold split | Build subsample, circulate schema doc | Sign off on `docs/schemas.md` (development-only; final contracts are documented in README.md) |
| **2** | Exact-name route on subsample | Feature functions unit-tested on 76 examples | Metric unit tests passing | Loader + validator against fabricated data | — |
| **4** | Real normalization swapped in; TF-IDF routes started | Real `retrieval_events.parquet` joined in | Fold split frozen; mock LR loop working | `run_pipeline.py` skeleton wired to mocks | — |
| **6** | Routes 1/2/4 on subsample | Features on subsample | LR on subsample scores | **End-to-end run on subsample, all real pieces** | **Go/no-go on architecture — does the wiring work at all?** |
| **12** | Full-train candidate gen complete, recall table reported | Full-train features complete | Oracle-ceiling + LR baseline reported | Runtime/memory profiled at full scale, ≥4 PRs merged | **Freeze route set + all schemas (Section 9)** |
| **24** | Test candidates generated; targeted tuning only | Feature list frozen; test features generated | LightGBM beats LR outside noise floor | First real test submission (format check) | **Freeze feature list; review Hour-12/24 numbers vs Section 7 decision tree** |
| **36** | Support error analysis; narrow fixes only | Targeted additions only, justified by P3 | Two-threshold + reverse-consistency validated via bootstrap; LOCO run | Track all adopted/discarded experiments | **Freeze model type, hyperparameters, decision rules** |
| **48** | Candidate gen fully frozen | Feature code fully frozen | Final model retrained on full data, handed off | Pipeline code frozen; README draft started | **Freeze pipeline code — bug fixes only from here** |
| **60** | On call | On call | Verify test scores non-degenerate | Full clean-environment dry run | Confirm packaging is on track |
| **72** | — | — | — | Final ZIP submitted | **Submit with buffer remaining, not at the deadline** |

---

## 7. Checkpoint Decision Tree

**At Hour 12 (candidate recall known):**
- IF any-match recall < 0.90, or any slice (multi-match-8+, India, non-ASCII) shows a cliff → P1 prioritizes the specific weak route/slice next; P2/P3 continue on schedule against whatever candidates currently exist, do not stall waiting for a recall fix.
- IF any-match recall ≥ 0.90 and oracle-ceiling F0.5 is high but volume (median candidates/S1) is excessive → P1 tightens block-size caps and k values before moving on, since F0.5's precision weighting makes volume a real cost, not just a runtime one.
- IF recall ≥ 0.90 and the oracle-ceiling gate is met → compare candidate-volume distributions; among configurations satisfying the recall gate, prefer the smaller candidate set / higher reduction ratio. Freeze the chosen route configuration once the trade-off is reviewed.

**At Hour 24 (model + decision layer known):**
- IF LightGBM OOF macro-F0.5 is close to the oracle ceiling from Hour 12 → the bottleneck has moved from blocking to precision; prioritize decision-layer work (two-threshold, reverse-consistency) over any further feature work.
- IF LightGBM OOF macro-F0.5 is well below the oracle ceiling → prioritize feature work (targeted, error-driven per P2's Hours 24–36 plan) over decision-layer tuning, since there's real signal being left on the table.
- IF LightGBM does not clearly beat LR (outside the seed-variance noise floor) → do not adopt LightGBM as-is; check for a data/feature bug before assuming the model itself is the problem.

**At Hour 36 (robustness known):**
- IF the LOCO gap (US↔India OOF F0.5) is small → France risk is lower than it could be; proceed to freeze as planned.
- IF the LOCO gap is large → this becomes the top priority for the remaining Hours 36–48 error-analysis window, ahead of any other polish, since it's the single biggest unquantifiable risk in the whole submission.

**At any point, IF pipeline runtime is too high:**
- First reduce candidate volume (tighter k, tighter block caps) — this is almost always the actual bottleneck, not the model.
- Only after that, optimize the retrieval implementation itself (better chunking, fewer redundant passes).
- Do not reach for infrastructure (more compute, distributed processing) before exhausting algorithmic fixes — see the AWS guidance in the master plan.

**IF everything above is acceptable by the relevant checkpoint:** stop optimizing that stage and move to the next one on schedule. A complete, working, on-time system beats a partially-optimized one.

---

## 8. Experiment / Submission Strategy

**What counts as an experiment:** any change to routes, features, model config, or decision rules that gets its own row in `experiment_tracker.csv`, with a hypothesis stated *before* running it.

**When to submit (5/day cap):**
- One submission at Hour ~20 (Section 6), purely to confirm the pipeline produces a validator-clean file at test scale — not a score check.
- After that, submit only when a *frozen* milestone is reached (e.g., after the Hour 24 or Hour 36 freeze) to get one external data point on the leaderboard — never mid-experiment.
- Reserve at least 2 of the 5 daily submissions on the final day for the actual final answer plus one buffer resubmission in case of a packaging error.

**What NOT to use leaderboard results for:** threshold tuning, feature selection, hyperparameter choices, or deciding between two candidate-generation configs. All of that is OOF-only, per Section 4. The leaderboard is a sanity check on your validation methodology, not a tuning signal — if leaderboard and OOF disagree sharply, that's a signal to re-examine the validation split for leakage, not to chase the leaderboard number.

**Version naming:** `expNN_<short-description>` (e.g. `exp07_two_threshold_decision`), matching the `experiment_id` in the tracker; each submission ZIP is named with the experiment ID that produced it.

**Submission history:** maintain `logs/submissions.md` with one entry per actual submission: submission ID, timestamp, commit/hash, candidate-generation version, model version, decision configuration/thresholds, candidate count, validator status, and leaderboard result.

**Metrics to record for every experiment:** OOF macro-F0.5, candidate recall (any-match, all-match), total candidate pairs, avg/median/p95/p99/max candidates per S1, reduction ratio, runtime, and the decision (kept/discarded) with the one-line reason.

**When to stop experimenting:** at each freeze point in Section 9 — no exceptions without a full-team discussion, since every hour spent past a freeze point is an hour not spent on packaging and documentation.

---

## 9. Freeze Plan

- **Hour 12:** candidate-generation route set (which routes exist, at what k/thresholds) and the candidate-volume trade-off; `docs/schemas.md` (development-only; final contracts are documented in README.md) reconfirmed as-built; `folds.parquet` (already frozen since Hour 3, reconfirmed unchanged).
- **Hour 24:** feature list (names + exact definitions) — no new feature columns without a team-wide announcement; final blocking parameters (k values, block-size caps) locked in.
- **Hour 36:** model type and hyperparameter search space — no further hyperparameter exploration past this point, only refits with the chosen config; decision-layer rule set (which of global/two-threshold/reverse-consistency ships in the final system).
- **Hour 48:** the entire pipeline codebase — bug fixes only from here. Final model retrained on full data; real test predictions generated from the frozen pipeline.

---

## 10. Final 12-Hour Plan (Hours 60–72)

### Clean-room execution convention
Run reproduction commands from the extracted `code/business_entity_resolution/` directory and write the official outputs to the ZIP-root `output/` directory. Use this convention consistently in the README and clean-room harness:

```bash
cd submission_staging/code/business_entity_resolution
python -m src.inference.pipeline --train-dir /path/to/train --test-dir /path/to/test --output-dir ../../output/
python src/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir /path/to/test
```

The clean-room check must verify exit code 0 and zero validator warnings.


- **P4:** Run the full pipeline from a genuinely clean checkout/virtualenv (`requirements.txt` pinned to exact versions, not `>=`) to catch "works on my machine" failures. Generate the final `matching_results.tsv` and `candidate_pairs.tsv` from the Hour-48 frozen pipeline against the full test set, with both files derived from the exact same final candidate DataFrame passed into inference. Run `validate_submission.py` and confirm zero warnings. Record final candidate-volume statistics and submission metadata in `logs/submissions.md`.
- **P3:** Spot-check the final test score distribution and predicted match-count distribution against train's (sanity, not tuning) — flag anything degenerate (e.g. near-zero predicted matches everywhere) immediately, since that's a pipeline bug, not a modeling result, this late.
- **P2 + P1:** Finalize `logs/person1.md` and `logs/person2.md` with final numbers (recall table, feature list) for direct inclusion in the methodology document; on call for any last bug.
- **P4:** Finalize `code/business_entity_resolution/README.md` (what the system does, exact reproduction commands, expected data layout, and output locations) and `Documentation_template.md` (1–2 page methodology: approach, candidate-generation routes and recall, features, model, decision layer, key experiments kept/discarded, France/robustness discussion) using the tracker, all four logs, and `logs/submissions.md` as source material — use the stricter 1–2 page guideline even if another task document uses "no page limit" wording, and keep the write-up dense and technical.
- **P4:** Assemble the exact required ZIP structure via script, not by hand. Copy only the production `src/` tree plus the final README/requirements into `code/business_entity_resolution/`; keep tests/logs/configs/experiments outside the ZIP unless explicitly required:
  ```
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       │   ├── data/
  │       │   ├── preprocessing/
  │       │   ├── blocking/
  │       │   ├── features/
  │       │   ├── validation/
  │       │   ├── models/
  │       │   ├── decision/
  │       │   ├── inference/
  │       │   └── utils/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
  ```
- **All four:** final review pass — confirm exactly one row per test S1 entity, no duplicate IDs, every predicted ID present in `candidate_pairs.tsv`, `candidate_pairs.tsv` contains exactly the candidates actually scored, and only test-set S2/S3 IDs used anywhere in the output.
- **Submit with hours of buffer remaining**, not at the deadline — reserve the last hour purely for a resubmission if the validator or a last review catches something.

---

## 11. Copy/Paste Tasks For Each Person

### PERSON 1 — SEND THIS TO THEM

You own candidate recall. Right now: create `src/blocking/exact_name.py`. Write a quick inline `normalize_stub()` (lowercase, strip punctuation, collapse whitespace) — don't wait for P2's real normalizer. Implement exact normalized-name retrieval: hash-map block on the normalized name, cap block size at 500. Get the 50k-row subsample from P4 (or self-sample 1,000 rows if it's not ready in 30 minutes) and run your route against it, writing `retrieval_events.parquet` with columns `pair_key, s1_id, candidate_id, candidate_source, route, rank, score`. Next up after that: char n-gram TF-IDF (3,4-gram) on name, then on address, top-k=20 each, chunked in batches of 5,000 S1 rows — never build a full similarity matrix. Full detailed phase plan is in Section 2 of the master plan.

### PERSON 2 — SEND THIS TO THEM

You own normalization and pairwise features — and you do not wait for P1. Right now: create `src/preprocessing/normalize.py` (conservative: NFKC, casefold, accent-fold, punctuation→space, whitespace collapse) and `src/preprocessing/corpus_stats.py` (mine a legal-suffix list from trailing-token frequency — don't hard-code one). Then write feature functions (`name_char_cos`, `name_containment_a_in_b/b_in_a`, `addr_char_cos`, `addr_containment`, `addr_num_jaccard`, `country_eq`, missingness flags) as pure functions, and unit-test them against the 76 real examples in `training_match_examples.md` — no dependency on P1's candidates needed for this. Full detailed phase plan is in Section 3.

### PERSON 3 — SEND THIS TO THEM

You own validation and the decision layer — and you do not wait for P2. Right now: create `src/validation/metrics.py` and implement + unit-test the exact macro-F0.5 metric (singleton correctly-empty = 1.0, singleton with a false positive = 0.0, matched entity predicted empty = 0.0). Then build `src/validation/splits.py`: 5-fold GroupKFold by S1 entity, stratified by country × has-match × match-count bucket, using only `train_ground_truth.tsv` and `train_source1.tsv`'s country column. Write `folds.parquet` and treat it as frozen the moment it's written. While waiting on real features, build a fabricated synthetic features table to get your LR training loop working end to end. Full detailed phase plan is in Section 4.

### PERSON 4 — SEND THIS TO THEM

You own integration and you have zero dependencies — start immediately. Top priority in the first hour: generate and commit the shared 50k-row deterministic train subsample (fixed seed), since everyone else needs it. Also right now: write the shared safe TSV loader (`dtype=str, keep_default_na=False`, explicit tab delimiter, blank-field preservation), circulate `docs/schemas.md` (development-only; final contracts are documented in README.md) and get sign-off from the other three before the first hour is out, and build the output writer + validator against fabricated data — including deliberately broken cases (duplicate IDs, a missing entity, an ID not in `candidate_pairs.tsv`) — so it's proven correct before it ever sees real output. Maintain `logs/submissions.md` for every actual submission. Full detailed phase plan is in Section 5.
