import sys
import os
import time
import pandas as pd
import numpy as np

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import src.blocking.candidate_generation as cg

# 1. Load 1K Samples
print("Loading 1K dataset samples...")
s1 = pd.read_csv("dataset/train/s1_sample.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/s2_sample.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/s3_sample.tsv", sep="\t", dtype=str)
print(f"Loaded S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}")

# 2. Load Ground Truth
print("Loading ground truth...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
s1_sample_ids = set(s1["entity_id"])
gt_sample = gt[gt["source1_entity_id"].isin(s1_sample_ids)].copy()
print(f"Ground truth records for S1 sample: {len(gt_sample)}")

# 3. Run individual routes
print("\n--- Running Individual Routes ---")
routes = [
    ("Route 1: exact_name", ["exact_name"]),
    ("Route 2: tfidf_name", ["tfidf_name"]),
    ("Route 3: rare_token_name", ["rare_token_name"]),
    ("Route 4: tfidf_address", ["tfidf_address"]),
    ("Route 5: numeric_address", ["numeric_address"]),
    ("Route 6: rare_token_address", ["rare_token_address"]),
    ("Route 7: reverse_retrieval", ["reverse_retrieval"]),
]

route_counts = {}
for name, enabled in routes:
    df_route = cg.generate_candidates(s1, s2, s3, config={"enabled_routes": enabled})
    route_counts[name] = len(df_route)
    s1_cov = df_route["s1_id"].nunique()
    print(f"  {name:30s} -> Candidates: {len(df_route):6d}, S1 Coverage: {s1_cov:4d} / {len(s1)} ({s1_cov/len(s1)*100:5.1f}%)")

# 4. Run Full Union & Deduplication
print("\n--- Running Full 7-Route Integration (Union + Deduplication) ---")
t0 = time.time()
candidates_df, events_df = cg.generate_candidates(s1, s2, s3, return_events=True)
elapsed = time.time() - t0

print(f"Total Retrieval Events across 7 routes: {len(events_df)}")
print(f"Final Deduplicated Union Candidates:    {len(candidates_df)}")
print(f"Deduplication Reduction Ratio:          {len(candidates_df) / max(len(events_df), 1):.2f}")
print(f"Total Integration Runtime:              {elapsed:.3f} seconds")

# 5. Candidate Distribution Diagnostics
cands_per_s1 = candidates_df.groupby("s1_id").size()
s1_covered_count = candidates_df["s1_id"].nunique()
zero_candidate_s1 = len(s1) - s1_covered_count

print(f"\n--- Union Candidate Statistics ---")
print(f"  Unique S1 entities with candidates: {s1_covered_count} / {len(s1)} ({s1_covered_count/len(s1)*100:.1f}%)")
print(f"  Zero-candidate S1 entities:         {zero_candidate_s1}")
print(f"  Mean candidates per S1:             {len(candidates_df)/len(s1):.2f}")
print(f"  Median candidates per S1:           {cands_per_s1.median():.1f}")
print(f"  P95 candidates per S1:              {np.percentile(cands_per_s1, 95):.1f}")
print(f"  P99 candidates per S1:              {np.percentile(cands_per_s1, 99):.1f}")
print(f"  Max candidates per S1:              {cands_per_s1.max()}")
print(f"  Candidate Source breakdown:\n{candidates_df['candidate_source'].value_counts().to_string()}")

# 6. Determinism & Invariant Tests
print(f"\n--- Testing Invariants & Determinism ---")
assert list(candidates_df.columns) == ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]
assert (candidates_df["pair_key"] == candidates_df["s1_id"] + "::" + candidates_df["candidate_id"]).all(), "pair_key invalid"
assert len(candidates_df) == candidates_df["pair_key"].nunique(), "Duplicate candidate pairs found!"
assert candidates_df["candidate_source"].isin(["S2", "S3"]).all(), "Invalid candidate source"

# Second run for determinism
candidates_df2 = cg.generate_candidates(s1, s2, s3)
pd.testing.assert_frame_equal(candidates_df, candidates_df2)
print("  Determinism assertion: PASSED (identical output across consecutive runs)")
print("  Schema and deduplication assertions: ALL PASSED")

# 7. Ground Truth Candidate Recall Evaluation
print(f"\n--- Ground Truth Candidate Recall Evaluation ---")
metrics = cg.evaluate_candidate_recall(candidates_df, gt_sample, s1_ids_subset=list(s1_sample_ids))
for k, v in metrics.items():
    print(f"  {k:30s}: {v}")

# 8. Incremental Route Contribution Analysis
print(f"\n--- Incremental Route Recall Contribution Analysis ---")
cumulative_routes = []
route_names_list = [
    "exact_name",
    "tfidf_name",
    "rare_token_name",
    "tfidf_address",
    "numeric_address",
    "rare_token_address",
    "reverse_retrieval",
]

for r in route_names_list:
    cumulative_routes.append(r)
    df_cum = cg.generate_candidates(s1, s2, s3, config={"enabled_routes": cumulative_routes})
    m = cg.evaluate_candidate_recall(df_cum, gt_sample, s1_ids_subset=list(s1_sample_ids))
    print(f"  + {r:22s} -> Pairs: {len(df_cum):6d} | Pair Recall: {m['pair_recall']*100:5.2f}% | Any-Match: {m['any_match_recall']*100:5.2f}% | All-Match: {m['all_match_recall']*100:5.2f}%")

print("\n--- First 5 Final Canonical Candidate Rows ---")
print(candidates_df.head(5).to_string())
