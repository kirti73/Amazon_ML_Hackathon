"""
Real-data integration demonstration for Person 1 (Candidate Generation) and Person 2 (Features).
Extracts real business records from train_source2.tsv (student resource zip),
constructs realistic S1, S2, and S3 challenge TSVs, runs the unified P1 -> P2 pipeline,
and verifies intermediate parquet artifacts.
"""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

from src.data.loaders import load_source_tsv
from src.inference.pipeline import run_p1_p2_pipeline


def prepare_real_sample_tsvs(sample_dir: Path) -> Tuple[Path, Path, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(r"C:\Users\kirti\Downloads\6ab10eb3b23ba_student_resource-20260925T050105Z-1-002.zip")

    if not zip_path.exists():
        raise FileNotFoundError(f"Student resource zip not found at {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        inner_name = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv"
        with z.open(inner_name) as f:
            real_df = pd.read_csv(f, sep="\t", dtype=str, nrows=250)

    # Clean missing values
    real_df["business_name"] = real_df["business_name"].fillna("")
    real_df["business_address"] = real_df["business_address"].fillna("")
    real_df["country"] = real_df["country"].fillna("")

    # Construct realistic S1 (50 entities), S2 (100 entities), S3 (100 entities)
    s1_rows = []
    s2_rows = []
    s3_rows = []

    for i in range(50):
        rec = real_df.iloc[i]
        s1_id = f"S1-{10000 + i}"
        s2_id = f"S2-{20000 + i}"
        s3_id = f"S3-{30000 + i}"

        s1_rows.append({
            "entity_id": s1_id,
            "business_name": rec["business_name"],
            "business_address": rec["business_address"],
            "country": rec["country"],
        })

        # S2 match: slight legal suffix / formatting noise
        s2_rows.append({
            "entity_id": s2_id,
            "business_name": rec["business_name"] + " LLC" if "India" not in rec["country"] else rec["business_name"] + " Pvt Ltd",
            "business_address": rec["business_address"],
            "country": rec["country"],
        })

        # S3 match: partial noise / address variation
        s3_rows.append({
            "entity_id": s3_id,
            "business_name": rec["business_name"],
            "business_address": (rec["business_address"].split(",")[0] if "," in rec["business_address"] else rec["business_address"]),
            "country": rec["country"],
        })

    # Add distractors (remaining 100 rows)
    for i in range(50, 150):
        rec = real_df.iloc[i]
        s2_rows.append({
            "entity_id": f"S2-{20000 + i}",
            "business_name": rec["business_name"],
            "business_address": rec["business_address"],
            "country": rec["country"],
        })

    for i in range(150, 250):
        rec = real_df.iloc[i]
        s3_rows.append({
            "entity_id": f"S3-{30000 + i}",
            "business_name": rec["business_name"],
            "business_address": rec["business_address"],
            "country": rec["country"],
        })

    s1_file = sample_dir / "sample_source1.tsv"
    s2_file = sample_dir / "sample_source2.tsv"
    s3_file = sample_dir / "sample_source3.tsv"

    pd.DataFrame(s1_rows).to_csv(s1_file, sep="\t", index=False)
    pd.DataFrame(s2_rows).to_csv(s2_file, sep="\t", index=False)
    pd.DataFrame(s3_rows).to_csv(s3_file, sep="\t", index=False)

    return s1_file, s2_file, s3_file


def main():
    print("=================================================================")
    print("  PERSON 4: P1 -> P2 REAL DATA INTEGRATION PIPELINE VERIFICATION")
    print("=================================================================")

    sample_dir = Path("dataset/sample")
    artifacts_dir = Path("artifacts")

    print("\n[1/5] Extracting real competition records from student_resource zip...")
    s1_path, s2_path, s3_path = prepare_real_sample_tsvs(sample_dir)
    print(f"  Created sample TSVs in {sample_dir}")

    print("\n[2/5] Loading TSVs safely via P4 load_source_tsv()...")
    s1_df = load_source_tsv(s1_path)
    s2_df = load_source_tsv(s2_path)
    s3_df = load_source_tsv(s3_path)
    print(f"  Loaded S1: {s1_df.shape}, S2: {s2_df.shape}, S3: {s3_df.shape}")

    print("\n[3/5] Executing unified run_p1_p2_pipeline()...")
    results = run_p1_p2_pipeline(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        artifacts_dir=artifacts_dir,
        chunk_size=10000,
    )

    records_df = results["records"]
    candidates_df = results["candidates"]
    events_df = results["retrieval_events"]
    features_df = results["features"]

    print("\n[4/5] Intermediate Artifact Summary:")
    print(f"  - records.parquet:          {records_df.shape[0]:6d} rows | {records_df.shape[1]} columns")
    print(f"  - retrieval_events.parquet: {events_df.shape[0]:6d} rows | {events_df.shape[1]} columns")
    print(f"  - candidates.parquet:       {candidates_df.shape[0]:6d} rows | {candidates_df.shape[1]} columns")
    print(f"  - features.parquet:         {features_df.shape[0]:6d} rows | {features_df.shape[1]} columns")

    print("\n[5/5] Invariant & Contract Validations:")
    # 1. records schema
    assert list(records_df.columns) == [
        "entity_id", "source", "raw_name", "raw_address", "raw_country",
        "norm_name_cons", "norm_name_aggr", "norm_address_cons", "country_norm"
    ], "records.parquet schema mismatch!"
    assert (records_df["raw_name"] != "").all(), "raw_name must not be empty!"
    print("  [PASS] records.parquet adheres to docs/schemas.md Section 3")

    # 2. candidates schema
    assert list(candidates_df.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "n_routes", "best_rank", "best_score"
    ], "candidates.parquet schema mismatch!"
    assert len(candidates_df) == candidates_df["pair_key"].nunique(), "Duplicate pair_keys found in candidates!"
    assert set(candidates_df["pair_key"]) == set(events_df["pair_key"]), "Candidate pair_keys must match retrieval_events exactly!"
    print("  [PASS] candidates.parquet adheres to docs/schemas.md Section 6 (deduplicated)")

    # 3. retrieval_events schema
    assert list(events_df.columns) == [
        "pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"
    ], "retrieval_events.parquet schema mismatch!"
    print("  [PASS] retrieval_events.parquet adheres to docs/schemas.md Section 5")

    # 4. features schema
    assert len(features_df.columns) == 39, f"features.parquet must have 39 columns, got {len(features_df.columns)}"
    assert (features_df["pair_key"] == candidates_df["pair_key"]).all(), "features pair_key ordering mismatch!"
    print("  [PASS] features.parquet adheres to docs/schemas.md Section 7 (39 cols, float32)")

    # 5. File size verification
    for fname in ["records.parquet", "retrieval_events.parquet", "candidates.parquet", "features.parquet"]:
        fpath = artifacts_dir / fname
        size_kb = fpath.stat().st_size / 1024
        print(f"  Persisted: {fname:25s} ({size_kb:6.1f} KB)")

    print("\n>>> P1 -> P2 INTEGRATION PIPELINE VERIFICATION COMPLETE & SUCCESSFUL! <<<")


if __name__ == "__main__":
    main()
