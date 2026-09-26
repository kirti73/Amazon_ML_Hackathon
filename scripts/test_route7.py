import importlib.util
import pandas as pd

# Load module directly
spec = importlib.util.spec_from_file_location(
    "reverse",
    "code/business_entity_resolution/src/blocking/reverse.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

s1 = pd.read_csv("dataset/train/s1_sample.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/s2_sample.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/s3_sample.tsv", sep="\t", dtype=str)

print("=== Testing Route 7: Reverse Retrieval on 1K Samples ===")
print(f"S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}")

res1 = mod.retrieve_reverse(s1, s2, s3, top_k_reverse=5, chunk_size=250)
print(f"Total candidates: {len(res1)}")
print(f"Columns: {list(res1.columns)}")
print(f"Candidate Source counts:\n{res1['candidate_source'].value_counts().to_string()}")
s1_covered = res1['s1_id'].nunique()
print(f"Unique S1 entities with candidates: {s1_covered} / {len(s1)} ({s1_covered / len(s1) * 100:.1f}%)")
print(f"Mean candidates per S1 (for covered S1): {len(res1) / s1_covered:.2f}")
print(f"Score range: [{res1['score'].min():.4f}, {res1['score'].max():.4f}]")

# Test Determinism
res2 = mod.retrieve_reverse(s1, s2, s3, top_k_reverse=5, chunk_size=250)
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
assert (res1["route"] == "reverse_retrieval").all(), "Invalid route tag"
# Ensure no duplicate pair rows
assert len(res1) == res1["pair_key"].nunique(), "Duplicate pairs found"
print("Schema, Invariant & Deduplication assertions: ALL PASSED")

# Test Missing/Blank values
s1_blanks = s1.copy()
s1_blanks.loc[0:4, "business_name"] = ""
s1_blanks.loc[5:9, "business_name"] = None
res_blanks = mod.retrieve_reverse(s1_blanks, s2, s3, top_k_reverse=5, chunk_size=250)
print(f"Missing/Blank name edge case test: PASSED (Candidates: {len(res_blanks)})")

print("\nTop 5 candidate rows:")
print(res1.head(5).to_string())
