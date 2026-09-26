"""
Route 5 — Numeric and Plot/Unit Address Token Retrieval
========================================================

Strategy
--------
1. Extract numeric and address tokens (house numbers, plot numbers,
   PIN/ZIP codes, unit/flat designations, e.g. "496/5b", "1064", "83/22")
   from business_address.
2. Build an inverted index mapping informative numeric tokens to S2 + S3 candidates.
3. Filter out overly common numeric tokens (e.g., "1", "2", or tokens with high DF)
   to prevent uninformative hubs.
4. For each S1 record, retrieve candidates sharing informative numeric tokens.
5. Soft retrieval route: records without numeric tokens are safely skipped
   without throwing errors or eliminating candidates from other routes.
6. Enforce block size caps (max 500) and top-k (default 20) per S1 record.
7. Deterministic tie-breaking: score desc, then candidate_id asc.

Candidate Schema
----------------
pair_key, s1_id, candidate_id, candidate_source, route, rank, score
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
ROUTE_NAME: str = "numeric_address"
DEFAULT_TOP_K: int = 20
DEFAULT_MAX_BLOCK_SIZE: int = 500
DEFAULT_MAX_DF_RATIO: float = 0.05
DEFAULT_MIN_DIGITS: int = 1


# ---------------------------------------------------------------------------
# Token Extraction & Normalization
# ---------------------------------------------------------------------------
# Matches tokens containing at least one digit: e.g., "1064", "496/5b", "u-1/22a", "3204", "126"
_NUMERIC_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9/\-#]{1,15}\d[a-zA-Z0-9/\-#]{0,15}\b")
_PUNCT_CLEAN_RE = re.compile(r"[^\w\s/\-#]", re.UNICODE)


def extract_numeric_tokens(address: str) -> List[str]:
    """Extract distinct numeric and alphanumeric address tokens from address text.

    Examples:
        "1064 Newton Rd, Unit 11" -> ["1064", "11"]
        "Plot No. U-1/22A, Phase-1" -> ["u-1/22a", "phase-1", "1"]
        "59/11, South Avani Moola St." -> ["59/11"]
    """
    if not address or not isinstance(address, str):
        return []
    if address.strip().lower() in ("nan", "none", "null"):
        return []

    norm = unicodedata.normalize("NFKC", address).casefold()
    # Replace non-structural punctuation with space
    cleaned = _PUNCT_CLEAN_RE.sub(" ", norm)

    matches = _NUMERIC_TOKEN_RE.findall(cleaned)
    tokens: List[str] = []
    for m in matches:
        # Strip trailing/leading non-alphanumerics
        t = m.strip(" /-#")
        # Ensure it contains at least one digit and length >= 1
        if any(c.isdigit() for c in t) and len(t) >= 1:
            tokens.append(t)
            # Also add the pure digits part if mixed (e.g. "unit11" -> "11")
            digits_only = re.sub(r"\D", "", t)
            if digits_only and digits_only != t and len(digits_only) >= 2:
                tokens.append(digits_only)

    return list(dict.fromkeys(tokens))  # preserve order, unique


# ---------------------------------------------------------------------------
# Numeric Token Inverted Index
# ---------------------------------------------------------------------------
class NumericTokenIndex:
    """Inverted index for address numeric tokens with IDF scoring."""

    def __init__(
        self,
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
        max_df_ratio: float = DEFAULT_MAX_DF_RATIO,
    ):
        self.max_block_size = max_block_size
        self.max_df_ratio = max_df_ratio
        self.index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0

    def fit_and_index(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        s1_df: Optional[pd.DataFrame] = None,
        id_col: str = "entity_id",
        addr_col: str = "business_address",
    ) -> "NumericTokenIndex":
        """Compute document frequencies and index S2+S3 numeric tokens."""
        df_counts: Counter = Counter()
        total_docs = 0

        sources = [(s2_df, "S2"), (s3_df, "S3")]
        if s1_df is not None:
            sources.append((s1_df, "S1"))

        for df, _ in sources:
            for _, row in df.iterrows():
                total_docs += 1
                raw = str(row.get(addr_col, "") or "")
                tokens = set(extract_numeric_tokens(raw))
                df_counts.update(tokens)

        self.n_docs = max(total_docs, 1)
        max_allowed_df = max(int(self.max_df_ratio * self.n_docs), self.max_block_size)

        # Smooth IDF
        for token, count in df_counts.items():
            # Higher weight for longer / more specific numeric codes
            specificity_bonus = min(len(token) * 0.1, 0.5)
            self.idf[token] = math.log((1.0 + self.n_docs) / (1.0 + count)) + 1.0 + specificity_bonus

        # Index S2 and S3 candidates
        for df, src_label in [(s2_df, "S2"), (s3_df, "S3")]:
            for _, row in df.iterrows():
                eid = str(row[id_col])
                raw = str(row.get(addr_col, "") or "")
                tokens = set(extract_numeric_tokens(raw))

                for token in tokens:
                    if df_counts[token] <= max_allowed_df:
                        self.index[token].append((eid, src_label))

        # Enforce deterministic ordering & cap
        for token, postings in list(self.index.items()):
            postings.sort(key=lambda x: x[0])
            if len(postings) > self.max_block_size:
                self.index[token] = postings[: self.max_block_size]

        return self

    def query(
        self,
        s1_df: pd.DataFrame,
        id_col: str = "entity_id",
        addr_col: str = "business_address",
        top_k: int = DEFAULT_TOP_K,
    ) -> pd.DataFrame:
        """Query S1 records against the numeric inverted index."""
        records = []

        for _, row in s1_df.iterrows():
            s1_id = str(row[id_col])
            raw = str(row.get(addr_col, "") or "")
            tokens = set(extract_numeric_tokens(raw))

            if not tokens:
                continue

            s1_idf_sum = sum(self.idf.get(t, 1.0) for t in tokens)
            if s1_idf_sum <= 0:
                s1_idf_sum = 1.0

            candidate_scores: Dict[Tuple[str, str], float] = defaultdict(float)

            for token in tokens:
                token_idf = self.idf.get(token, 1.0)
                postings = self.index.get(token, [])
                for cand_id, cand_src in postings:
                    candidate_scores[(cand_id, cand_src)] += token_idf

            if not candidate_scores:
                continue

            # Deterministic sort: score desc, candidate_id asc
            sorted_cands = sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
            top_cands = sorted_cands[:top_k]

            for rank, ((cand_id, cand_src), raw_score) in enumerate(top_cands, start=1):
                norm_score = min(raw_score / s1_idf_sum, 1.0)
                records.append(
                    {
                        "pair_key": f"{s1_id}::{cand_id}",
                        "s1_id": s1_id,
                        "candidate_id": cand_id,
                        "candidate_source": cand_src,
                        "route": ROUTE_NAME,
                        "rank": rank,
                        "score": round(float(norm_score), 6),
                    }
                )

        if not records:
            return _empty_candidate_df()

        return pd.DataFrame(records, columns=_candidate_columns())


# ---------------------------------------------------------------------------
# Route 5 Public Entrypoint
# ---------------------------------------------------------------------------
def retrieve_numeric(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    addr_col: str = "business_address",
    top_k: int = DEFAULT_TOP_K,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    max_df_ratio: float = DEFAULT_MAX_DF_RATIO,
) -> pd.DataFrame:
    """Route 5: Retrieve candidates using numeric/address tokens.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, addr_col].
    id_col              : Entity ID column name.
    addr_col            : Business address column name.
    top_k               : Max candidates to retrieve per S1.
    max_block_size      : Max postings per token bucket (default 500).
    max_df_ratio        : Document frequency ratio filter for common numbers.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
    """
    index = NumericTokenIndex(
        max_block_size=max_block_size,
        max_df_ratio=max_df_ratio,
    )
    index.fit_and_index(
        s2_df=s2_df,
        s3_df=s3_df,
        s1_df=s1_df,
        id_col=id_col,
        addr_col=addr_col,
    )
    return index.query(
        s1_df=s1_df,
        id_col=id_col,
        addr_col=addr_col,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _candidate_columns() -> List[str]:
    return [
        "pair_key",
        "s1_id",
        "candidate_id",
        "candidate_source",
        "route",
        "rank",
        "score",
    ]


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_columns())
