import importlib.util
import pandas as pd

# Load module directly
spec = importlib.util.spec_from_file_location(
    "rare_token",
    "code/business_entity_resolution/src/blocking/rare_token.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

s1 = pd.read_csv("dataset/train/s1_sample.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/s2_sample.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/s3_sample.tsv", sep="\t", dtype=str)

print("=== Testing Route 3 on 1K Samples ===")
print(f"S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}")

res1 = mod.retrieve_rare_token_name(s1, s2, s3, top_k=20, max_block_size=500)
print(f"Total candidates: {len(res1)}")
print(f"Columns: {list(res1.columns)}")
print(f"Candidate Source counts:\n{res1['candidate_source'].value_counts().to_string()}")
print(f"Unique S1 entities with candidates: {res1['s1_id'].nunique()} / {len(s1)}")
print(f"Mean candidates per S1: {len(res1) / len(s1):.2f}")
print(f"Score range: [{res1['score'].min():.4f}, {res1['score'].max():.4f}]")

# Test Determinism
res2 = mod.retrieve_rare_token_name(s1, s2, s3, top_k=20, max_block_size=500)
pd.testing.assert_frame_equal(res1, res2)
print("Determinism test: PASSED (consecutive runs produce identical results)")

# Check candidate schema & invariants
assert list(res1.columns) == [
    "pair_key",
    "s1_id",
    "candidate_id",
    "candidate_source",
    "route",
    "rank",
    "score",
], "Schema mismatch"
assert (res1["pair_key"] == res1["s1_id"] + "::" + res1["candidate_id"]).all(), "pair_key mismatch"
assert res1["candidate_source"].isin(["S2", "S3"]).all(), "Invalid candidate source"
assert (res1["route"] == "rare_token_name").all(), "Invalid route tag"
print("Schema & Invariant assertions: ALL PASSED")

print("\nTop 5 candidate rows:")
print(res1.head(5).to_string())
