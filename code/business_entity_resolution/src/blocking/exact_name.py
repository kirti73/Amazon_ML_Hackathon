"""
Route 1 — Exact Normalized-Name Blocking
=========================================

Strategy
--------
Group S2/S3 records by their conservative-normalized business name.
For each S1 record, look up the matching bucket and emit every
(s1_id, candidate_id) pair as a candidate.

Block-size cap
--------------
If a normalized-name bucket contains > MAX_BLOCK_SIZE candidates,
retain only the first MAX_BLOCK_SIZE in *deterministic* order
(sorted by candidate_id ascending) so results are reproducible.

Empty-name handling
-------------------
Blank or whitespace-only names produce the normalized key "".
These are never indexed — they would create a huge uninformative block.

Normalization (P1 stub — isolated from P2's feature normalization)
------------------------------------------------------------------
NFKC -> casefold -> punctuation->space -> whitespace collapse
"""

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BLOCK_SIZE: int = 500      # plans specify 500 as the cap
ROUTE_NAME: str = "exact_name"


# ---------------------------------------------------------------------------
# P1 normalization stub
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_name(text: str) -> str:
    """Conservative P1 normalization for blocking keys.

    Steps:
        1. Unicode NFKC decomposition
        2. casefold (locale-agnostic lower-case)
        3. Replace punctuation characters with space
        4. Collapse whitespace and strip

    Returns empty string for blank/None input.
    """
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_name_index(
    df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    source_label: str = "S2",
) -> Dict[str, List[Tuple[str, str]]]:
    """Build a dict: normalized_name -> [(entity_id, source_label), ...]."""
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        key = normalize_name(str(row.get(name_col, "") or ""))
        if not key:
            continue
        eid = str(row[id_col])
        index[key].append((eid, source_label))
    return index


def build_combined_index(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
) -> Dict[str, List[Tuple[str, str]]]:
    """Merge S2 and S3 into a single normalized-name lookup map."""
    combined: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for src_df, label in [(s2_df, "S2"), (s3_df, "S3")]:
        idx = _build_name_index(src_df, id_col=id_col, name_col=name_col, source_label=label)
        for key, entries in idx.items():
            combined[key].extend(entries)

    # Sort each bucket deterministically by entity_id
    for key in combined:
        combined[key].sort(key=lambda x: x[0])

    return combined


# ---------------------------------------------------------------------------
# Route retrieval
# ---------------------------------------------------------------------------

def retrieve_exact_name(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    max_block_size: int = MAX_BLOCK_SIZE,
) -> pd.DataFrame:
    """Exact normalized-name blocking: Route 1.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with at least [id_col, name_col].
    id_col              : Column name for entity ID.
    name_col            : Column name for business name.
    max_block_size      : Maximum candidates per normalized-name bucket.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
    """
    pool_index = build_combined_index(s2_df, s3_df, id_col=id_col, name_col=name_col)

    records = []
    for _, row in s1_df.iterrows():
        s1_id = str(row[id_col])
        key = normalize_name(str(row.get(name_col, "") or ""))
        if not key:
            continue

        candidates = pool_index.get(key, [])
        if not candidates:
            continue

        # Apply block-size cap (already sorted deterministically)
        if len(candidates) > max_block_size:
            candidates = candidates[:max_block_size]

        for rank, (cand_id, cand_src) in enumerate(candidates, start=1):
            records.append(
                {
                    "pair_key": f"{s1_id}::{cand_id}",
                    "s1_id": s1_id,
                    "candidate_id": cand_id,
                    "candidate_source": cand_src,
                    "route": ROUTE_NAME,
                    "rank": rank,
                    "score": 1.0,
                }
            )

    if not records:
        return _empty_candidate_df()

    return pd.DataFrame(records, columns=_candidate_columns())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_columns():
    return ["pair_key", "s1_id", "candidate_id", "candidate_source", "route", "rank", "score"]


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_columns())
