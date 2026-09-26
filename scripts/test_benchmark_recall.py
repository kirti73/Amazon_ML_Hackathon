import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
import src.blocking.candidate_generation as cg

# 1. Read S1 subset
s1_full = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=200)
s1_ids = set(s1_full["entity_id"])
gt_full = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
gt_subset = gt_full[gt_full["source1_entity_id"].isin(s1_ids)]

needed_s2 = set()
needed_s3 = set()
for m in gt_subset["matched_entity_ids"]:
    if isinstance(m, str) and m:
        for x in m.split(","):
            if x.startswith("S2-"):
                needed_s2.add(x)
            elif x.startswith("S3-"):
                needed_s3.add(x)

# 2. Extract true partners from S2 and S3
s2_chunks = []
for chunk in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, chunksize=100000):
    matched = chunk[chunk["entity_id"].isin(needed_s2)]
    if len(matched) > 0:
        s2_chunks.append(matched)
    if len(s2_chunks) > 0 and sum(len(c) for c in s2_chunks) >= len(needed_s2):
        break
s2_aligned = pd.concat(s2_chunks)

s3_chunks = []
for chunk in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, chunksize=100000):
    matched = chunk[chunk["entity_id"].isin(needed_s3)]
    if len(matched) > 0:
        s3_chunks.append(matched)
    if len(s3_chunks) > 0 and sum(len(c) for c in s3_chunks) >= len(needed_s3):
        break
s3_aligned = pd.concat(s3_chunks)

# 3. Add 1K negative distractors to simulate realistic noise
s2_sample = pd.read_csv("dataset/train/s2_sample.tsv", sep="\t", dtype=str)
s3_sample = pd.read_csv("dataset/train/s3_sample.tsv", sep="\t", dtype=str)
s2_eval = pd.concat([s2_aligned, s2_sample]).drop_duplicates(subset=["entity_id"])
s3_eval = pd.concat([s3_aligned, s3_sample]).drop_duplicates(subset=["entity_id"])

route_list = [
    "exact_name",
    "tfidf_name",
    "rare_token_name",
    "tfidf_address",
    "numeric_address",
    "rare_token_address",
    "reverse_retrieval",
]

print("=== INDIVIDUAL ROUTE GROUND-TRUTH RECALL ===")
for r in route_list:
    df_r = cg.generate_candidates(s1_full, s2_eval, s3_eval, config={"enabled_routes": [r]})
    m = cg.evaluate_candidate_recall(df_r, gt_subset, s1_ids_subset=list(s1_ids))
    print(f"  {r:20s} -> Recall: {m['pair_recall']*100:6.2f}% ({m['recovered_pairs']}/{m['total_gt_pairs']}) | Pairs: {len(df_r):5d}")

print("\n=== CUMULATIVE ROUTE UNION GROUND-TRUTH RECALL ===")
cum = []
for r in route_list:
    cum.append(r)
    df_c = cg.generate_candidates(s1_full, s2_eval, s3_eval, config={"enabled_routes": cum})
    m = cg.evaluate_candidate_recall(df_c, gt_subset, s1_ids_subset=list(s1_ids))
    print(f"  + {r:20s} -> Cum Recall: {m['pair_recall']*100:6.2f}% ({m['recovered_pairs']}/{m['total_gt_pairs']}) | Any-Match: {m['any_match_recall']*100:5.2f}% | All-Match: {m['all_match_recall']*100:5.2f}% | Unique Pairs: {len(df_c):5d}")
