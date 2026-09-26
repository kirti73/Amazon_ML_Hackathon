import pandas as pd
from pathlib import Path

TRAIN_DIR = Path("dataset/train")

for source in [1, 2, 3]:
    input_file = TRAIN_DIR / f"train_source{source}.tsv"
    output_file = TRAIN_DIR / f"s{source}_sample.tsv"

    print(f"Reading {input_file}...")

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str
    )

    sample = df.sample(
        n=1000,
        random_state=42
    )

    sample.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(f"Created {output_file}: {len(sample)} rows")

print("\nDone! All 3 samples created.")