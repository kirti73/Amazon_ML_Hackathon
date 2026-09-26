# PERSON 4 LOG — Integration, Orchestration & Packaging

**Owner:** Person 4 (Integration & Packaging)  
**Role Context:** Amazon ML Challenge 2026: Business Entity Resolution  
**Current Branch:** `person4/integration-p1-p2`  
**Status:** P1 -> P2 Integration Complete, 20/20 Unit & Integration Tests Passed  

---

## 1. Scope and Mission Summary
Person 4 owns the end-to-end integration of subsystems produced by P1 (Candidate Generation), P2 (Preprocessing & Features), and P3 (ML & Decision Layer), culminating in canonical Parquet intermediate tables and submission outputs (`matching_results.tsv` and `candidate_pairs.tsv`).

### Milestone 1: P1 <-> P2 Integration
1. Unified codebase namespaces between root `src/` and `code/business_entity_resolution/src/` to prevent package shadowing.
2. Built P4 column adapter (`adapt_raw_for_preprocessing`) resolving the `business_name`/`business_address` vs `name`/`address` header mismatch.
3. Built candidate schema reconciler (`reconcile_candidates_schema`) aligning P1's deduplicated output with `docs/schemas.md` Section 6 (`n_routes`, `best_rank`, `best_score`).
4. Installed required environment dependencies (`rapidfuzz`, `pyarrow`, `pytest`).
5. Implemented `run_p1_p2_pipeline()` in `src/inference/pipeline.py` connecting `load_source_tsv()` -> P1 `generate_candidates(return_events=True)` -> P2 `preprocess_records_df()` -> P2 `build_features()`.
6. Verified with real competition records (`scripts/run_p1_p2_real_data.py`) and full unit/integration test suite.

### Milestone 2: Candidate Schema Reconciliation Fix
1. Refactored `reconcile_candidates_schema()` to derive `best_rank` as $\min(\text{rank})$ across retrieval routes from `retrieval_events_df`, avoiding the P1 union rank trap.
2. Derived `best_score` as $\max(\text{score})$ and `n_routes` as distinct route count per `pair_key`.
3. Attached canonical metadata to the exact candidate set, verifying:
   `set(candidate_pairs_before.pair_key) == set(candidate_pairs_after.pair_key)`.
4. Added multi-route regression test in `tests/test_integration_p1_p2.py` proving `best_rank != union_rank`.
5. Re-verified full test suite (21/21 passing) and real data integration pipeline.

### Milestone 3: Selective Integration of Person 3 (ML, Validation & Decision Layer)
1. **Isolated Selective Porting**:
   - Integrated P3-owned subsystems without wholesale merging or cherry-picking stale commits.
   - Preserved all existing P1/P2/P4 files without any overwrites.
   - Maintained single authoritative production tree under `code/business_entity_resolution/src/` with zero duplicate root `src/` files.
2. **Target File Placement**:
   - `src/models/*` -> `code/business_entity_resolution/src/models/` (`__init__.py`, `labels.py`, `lightgbm_model.py`, `logreg.py`, `train_cv.py`)
   - `src/decision/*` -> `code/business_entity_resolution/src/decision/` (`__init__.py`, `threshold.py`)
   - `src/validation/*` -> `code/business_entity_resolution/src/validation/` (`__init__.py`, `loco.py`, `metrics.py`, `splits.py`)
   - `utils/validate_submission.py` -> `code/business_entity_resolution/src/utils/validate_submission.py`
   - `scripts/generate_folds.py` -> `scripts/generate_folds.py`
   - Tests: `tests/conftest.py`, `tests/test_decision.py`, `tests/test_labels.py`, `tests/test_lightgbm.py`, `tests/test_loco.py`, `tests/test_logreg.py`, `tests/test_metrics.py`, `tests/test_splits.py`, `tests/test_train_cv.py`.
3. **Runtime & Dependency Configuration**:
   - Analyzed LightGBM OpenMP runtime on Windows: verified `import lightgbm` succeeds natively (version 4.7.0). Linux ELF binary `.vendor/libgomp/libgomp.so.1` is not required on Windows.
   - Made repo root resolution dynamic for `VENDORED_OPENMP_DIR` in `lightgbm_model.py` and `sys.path` in `scripts/generate_folds.py`.
   - Updated `code/business_entity_resolution/requirements.txt` to include `joblib>=1.3.0`.
4. **Verification**:
   - Existing 21 P1/P2 tests: 21/21 passed in 0.58s.
   - P3 test suite: 112/112 passed in 7.32s.
   - Complete combined test suite: 133/133 passed in 6.79s.
   - Real-data P1->P2 integration test: 100% passed with zero regressions.

---

## 2. Invariant & Contract Audit

| Contract | Required Schema | Implementation Status | Invariant Checks |
|---|---|---|---|
| `records.parquet` | `entity_id, source, raw_name, raw_address, raw_country, norm_name_cons, norm_name_aggr, norm_address_cons, country_norm` | Matched 100% | All string types, Devanagari safe, no dropped names |
| `retrieval_events.parquet` | `pair_key, s1_id, candidate_id, candidate_source, route, rank, score` | Matched 100% | Multi-route provenance preserved, 1-based ranks |
| `candidates.parquet` | `pair_key, s1_id, candidate_id, candidate_source, n_routes, best_rank, best_score` | Matched 100% via P4 reconciler | Unique `pair_key`, min route rank, max route score, distinct route count |
| `features.parquet` | `pair_key, s1_id, candidate_id` + 36 float32 features | Matched 100% | 39 columns total, float32, NaN for missing address |
| `folds.parquet` | `s1_id, fold_id` | Matched 100% (P3 `splits.py` & `generate_folds.py`) | Entity-level stratified split, no leakage |
| `train_labels.parquet` | `pair_key, s1_id, candidate_id, y` | Matched 100% (P3 `labels.py`) | Binary labels with hard negatives, singleton abstention safe |
| `scores.parquet` | `pair_key, s1_id, candidate_id, p_raw, p_cal, fold_id, is_oof, model_version` | Matched 100% (P3 `train_cv.py`) | Leak-free OOF predictions, Arrow string types |

---

## 3. Test Verification
- **P1/P2 Suite (21 tests)**:
  - `tests/test_preprocessing.py`: 11 passed (multilingual, Devanagari, accent normalization, corpus mining)
  - `tests/test_features.py`: 7 passed (lexical, n-gram, containment, address conflict, missing address NaNs)
  - `tests/test_integration_p1_p2.py`: 3 passed (full end-to-end pipeline + multi-route reconciliation regression)
- **P3 Suite (112 tests)**:
  - `tests/test_decision.py`: 21 passed (two-threshold optimization, singletons, boundary conditions)
  - `tests/test_labels.py`: 11 passed (label construction, singleton validation, negative generation)
  - `tests/test_lightgbm.py`: 17 passed (monotonicity sweeps, NaN handling, early stopping, serialization)
  - `tests/test_loco.py`: 18 passed (leave-one-country-out evaluation, cross-country robustness)
  - `tests/test_logreg.py`: 14 passed (scikit-learn baseline pipeline, missing value imputation)
  - `tests/test_metrics.py`: 11 passed (macro F0.5 precision-weighted evaluation, singleton handling)
  - `tests/test_splits.py`: 9 passed (stratified 5-fold assignment, country/match balance)
  - `tests/test_train_cv.py`: 11 passed (leak-free CV, OOF coverage, score schema compliance)
- **Combined Suite**: **133 passed** in 6.79s.
- **Pipeline Verification**: `scripts/run_p1_p2_real_data.py` confirmed 100% functional.

