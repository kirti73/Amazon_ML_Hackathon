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
5. Re-verified full test suite (20/20 passing) and real data integration pipeline.

---

## 2. Invariant & Contract Audit

| Contract | Required Schema | Implementation Status | Invariant Checks |
|---|---|---|---|
| `records.parquet` | `entity_id, source, raw_name, raw_address, raw_country, norm_name_cons, norm_name_aggr, norm_address_cons, country_norm` | Matched 100% | All string types, Devanagari safe, no dropped names |
| `retrieval_events.parquet` | `pair_key, s1_id, candidate_id, candidate_source, route, rank, score` | Matched 100% | Multi-route provenance preserved, 1-based ranks |
| `candidates.parquet` | `pair_key, s1_id, candidate_id, candidate_source, n_routes, best_rank, best_score` | Matched 100% via P4 reconciler | Unique `pair_key`, min route rank, max route score, distinct route count |
| `features.parquet` | `pair_key, s1_id, candidate_id` + 36 float32 features | Matched 100% | 39 columns total, float32, NaN for missing address |

---

## 3. Test Verification
- `tests/test_preprocessing.py`: 11 passed (multilingual, Devanagari, accent normalization, corpus mining)
- `tests/test_features.py`: 7 passed (lexical, n-gram, containment, address conflict, missing address NaNs)
- `tests/test_integration_p1_p2.py`: 2 passed (full end-to-end pipeline + multi-route reconciliation regression)
- `scripts/run_p1_p2_real_data.py`: Validated on real records from `student_resource` TSVs.
