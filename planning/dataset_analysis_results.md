# Dataset Analysis Results — `script.py`

> [!NOTE]
> Script ran against the `dataset/train/` folder in the student resource directory. Exit code: **0** (success).

---

## Source File Shapes

| Source | Rows | Columns | Columns List |
|--------|------|---------|--------------|
| **S1** (train_source1.tsv) | 2,206,821 | 4 | `entity_id`, `business_name`, `business_address`, `country` |
| **S2** (train_source2.tsv) | 5,034,616 | 4 | `entity_id`, `business_name`, `business_address`, `country` |
| **S3** (train_source3.tsv) | 5,285,603 | 4 | `entity_id`, `business_name`, `business_address`, `country` |

---

## Data Quality Checks

| Check | S1 | S2 | S3 |
|-------|----|----|-----|
| Duplicate `entity_id` | 0 | 0 | 0 |
| Blank `business_name` | 0% | 0% | 0% |
| Blank `business_address` | 0% | **3.36%** | **3.33%** |

> [!IMPORTANT]
> All entity IDs are unique across each source. No duplicate IDs exist.

---

## Country Distribution

| Country | S1 | S2 | S3 |
|---------|----|----|-----|
| **US** | 1,323,633 | 3,016,817 | 3,170,056 |
| **India** | 883,188 | 2,017,799 | 2,115,547 |

---

## Ground Truth Analysis

| Metric | Value |
|--------|-------|
| Total rows | **2,206,821** |
| Unique S1 IDs in GT | **2,206,821** |
| S1 rows with no GT row | **0** (complete coverage) |

### Match Count Distribution

| # Matches | Count | Notes |
|-----------|-------|-------|
| 0 | 123,247 | **5.58%** singletons (no match) |
| 1 | 119,157 | |
| 2 | 375,212 | |
| 3 | 530,841 | ← **mode** |
| 4 | 484,115 | |
| 5 | 321,957 | |
| 6 | 164,868 | |
| 7 | 63,968 | |
| 8 | 18,680 | |
| 9 | 4,205 | |
| 10 | 534 | |
| 11 | 37 | |

---

## Structural Findings

| Check | Result |
|-------|--------|
| Max times a single S2/S3 ID appears across different S1 rows | **1** |
| S2/S3 IDs appearing >1 time in GT | **0** |
| GT IDs not present in S2/S3 | **0** |
| S2/S3 IDs never in GT (orphans) | **2,681,854** |

> [!TIP]
> **Key structural insight**: The matching is strictly **one-to-one** — no S2/S3 entity is matched to multiple S1 entities. This means the problem is a clean entity resolution task without many-to-many ambiguity.

> [!WARNING]
> **2,681,854 orphan IDs** exist in S2/S3 that are never referenced in the ground truth. These are distractors/negatives your blocking strategy must efficiently filter out.
