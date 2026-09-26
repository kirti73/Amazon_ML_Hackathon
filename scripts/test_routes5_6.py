import importlib.util
import pandas as pd

# Load modules directly
spec_num = importlib.util.spec_from_file_location(
    "numeric",
    "code/business_entity_resolution/src/blocking/numeric.py"
)
mod_num = importlib.util.module_from_spec(spec_num)
spec_num.loader.exec_module(mod_num)

spec_rare = importlib.util.spec_from_file_location(
    "rare_token",
    "code/business_entity_resolution/src/blocking/rare_token.py"
)
mod_rare = importlib.util.module_from_spec(spec_rare)
spec_rare.loader.exec_module(mod_rare)

s1 = pd.read_csv("dataset/train/s1_sample.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/s2_sample.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/s3_sample.tsv", sep="\t", dtype=str)

print("=== TESTING ROUTE 5: NUMERIC ADDRESS RETRIEVAL ===")
res5_1 = mod_num.retrieve_numeric(s1, s2, s3, top_k=20, max_block_size=500)
print(f"Route 5 Total Candidates: {len(res5_1)}")
print(f"Columns: {list(res5_1.columns)}")
print(f"Candidate Source counts:\n{res5_1['candidate_source'].value_counts().to_string()}")
s1_covered_5 = res5_1['s1_id'].nunique()
print(f"S1 entities covered: {s1_covered_5} / {len(s1)} ({s1_covered_5 / len(s1) * 100:.1f}%)")
print(f"Score range: [{res5_1['score'].min():.4f}, {res5_1['score'].max():.4f}]")

# Determinism test Route 5
res5_2 = mod_num.retrieve_numeric(s1, s2, s3, top_k=20, max_block_size=500)
pd.testing.assert_frame_equal(res5_1, res5_2)
print("Route 5 Determinism: PASSED")

# Invariant & schema test Route 5
assert list(res5_1.columns) == ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]
assert (res5_1["pair_key"] == res5_1["s1_id"] + "::" + res5_1["candidate_id"]).all()
assert res5_1["candidate_source"].isin(["S2", "S3"]).all()
assert (res5_1["route"] == "numeric_address").all()
print("Route 5 Schema & Invariants: ALL PASSED")

# Blank address edge case Route 5
s1_blanks = s1.copy()
s1_blanks.loc[0:4, "business_address"] = ""
s1_blanks.loc[5:9, "business_address"] = None
res5_blanks = mod_num.retrieve_numeric(s1_blanks, s2, s3, top_k=20)
print(f"Route 5 Blank address edge test: PASSED (Candidates: {len(res5_blanks)})")


print("\n=== TESTING ROUTE 6: RARE-TOKEN ADDRESS RETRIEVAL ===")
res6_1 = mod_rare.retrieve_rare_token_address(s1, s2, s3, top_k=20, max_block_size=500)
print(f"Route 6 Total Candidates: {len(res6_1)}")
print(f"Columns: {list(res6_1.columns)}")
print(f"Candidate Source counts:\n{res6_1['candidate_source'].value_counts().to_string()}")
s1_covered_6 = res6_1['s1_id'].nunique()
print(f"S1 entities covered: {s1_covered_6} / {len(s1)} ({s1_covered_6 / len(s1) * 100:.1f}%)")
print(f"Score range: [{res6_1['score'].min():.4f}, {res6_1['score'].max():.4f}]")

# Determinism test Route 6
res6_2 = mod_rare.retrieve_rare_token_address(s1, s2, s3, top_k=20, max_block_size=500)
pd.testing.assert_frame_equal(res6_1, res6_2)
print("Route 6 Determinism: PASSED")

# Invariant & schema test Route 6
assert list(res6_1.columns) == ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]
assert (res6_1["pair_key"] == res6_1["s1_id"] + "::" + res6_1["candidate_id"]).all()
assert res6_1["candidate_source"].isin(["S2", "S3"]).all()
assert (res6_1["route"] == "rare_token_address").all()
print("Route 6 Schema & Invariants: ALL PASSED")

# Blank address edge case Route 6
res6_blanks = mod_rare.retrieve_rare_token_address(s1_blanks, s2, s3, top_k=20)
print(f"Route 6 Blank address edge test: PASSED (Candidates: {len(res6_blanks)})")

print("\nSample Route 5 rows:")
print(res5_1.head(3).to_string())

print("\nSample Route 6 rows:")
print(res6_1.head(3).to_string())
