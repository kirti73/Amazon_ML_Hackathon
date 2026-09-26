"""
Route 3 & Route 6 — Rare-Token Inverted Index Retrieval
======================================================

Strategy (Route 3: Name / Route 6: Address)
-------------------------------------------
1. Conservative P1 normalization on input strings.
2. Tokenize into individual words/tokens.
3. Compute corpus Document Frequency (DF) and smooth IDF weights.
4. Filter out uninformative/high-frequency tokens to prevent giant blocks.
5. Build an inverted index mapping informative tokens to candidate IDs.
6. Retrieve candidate records sharing rare/high-IDF tokens with S1.
7. Score candidates by IDF-weighted token overlap.
8. Enforce per-bucket block caps (max 500) and top-k per S1 query.
9. Deterministic tie-breaking: score desc, candidate_id asc.

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
ROUTE_NAME_TOKEN_NAME: str = "rare_token_name"
ROUTE_NAME_TOKEN_ADDR: str = "rare_token_address"

DEFAULT_TOP_K: int = 20
DEFAULT_MAX_BLOCK_SIZE: int = 500
DEFAULT_MAX_DF_RATIO: float = 0.05  # Tokens in >5% docs excluded from indexing
DEFAULT_MIN_TOKEN_LEN: int = 2


# ---------------------------------------------------------------------------
# P1 Normalization Stub
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Conservative P1 normalization for token extraction."""
    if not text or not isinstance(text, str):
        return ""
    if text.strip().lower() in ("nan", "none", "null"):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: str, min_len: int = DEFAULT_MIN_TOKEN_LEN) -> List[str]:
    """Extract distinct alphanumeric tokens of minimum length."""
    norm = normalize_text(text)
    if not norm:
        return []
    return [t for t in norm.split() if len(t) >= min_len]


# ---------------------------------------------------------------------------
# Inverted Index Builder
# ---------------------------------------------------------------------------
class RareTokenIndex:
    """Inverted index with IDF weighting and block-size safeguards."""

    def __init__(
        self,
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
        max_df_ratio: float = DEFAULT_MAX_DF_RATIO,
        min_token_len: int = DEFAULT_MIN_TOKEN_LEN,
    ):
        self.max_block_size = max_block_size
        self.max_df_ratio = max_df_ratio
        self.min_token_len = min_token_len
        self.index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0

    def fit_and_index(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        s1_df: Optional[pd.DataFrame] = None,
        id_col: str = "entity_id",
        text_col: str = "business_name",
    ) -> "RareTokenIndex":
        """Compute corpus DF/IDF and index S2+S3 records."""
        df_counts: Counter = Counter()
        total_docs = 0

        sources_to_count = [(s2_df, "S2"), (s3_df, "S3")]
        if s1_df is not None:
            sources_to_count.append((s1_df, "S1"))

        for df, _ in sources_to_count:
            for _, row in df.iterrows():
                total_docs += 1
                raw = str(row.get(text_col, "") or "")
                tokens = set(tokenize(raw, min_len=self.min_token_len))
                df_counts.update(tokens)

        self.n_docs = max(total_docs, 1)

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1.0
        max_allowed_df = max(int(self.max_df_ratio * self.n_docs), self.max_block_size)
        for token, count in df_counts.items():
            self.idf[token] = math.log((1.0 + self.n_docs) / (1.0 + count)) + 1.0

        # Build inverted index for S2 and S3 candidates
        for df, src_label in [(s2_df, "S2"), (s3_df, "S3")]:
            for _, row in df.iterrows():
                eid = str(row[id_col])
                raw = str(row.get(text_col, "") or "")
                tokens = set(tokenize(raw, min_len=self.min_token_len))

                for token in tokens:
                    if df_counts[token] <= max_allowed_df:
                        self.index[token].append((eid, src_label))

        # Enforce deterministic sorting and block size cap on each bucket
        for token, postings in list(self.index.items()):
            postings.sort(key=lambda x: x[0])
            if len(postings) > self.max_block_size:
                self.index[token] = postings[: self.max_block_size]

        return self

    def query(
        self,
        s1_df: pd.DataFrame,
        id_col: str = "entity_id",
        text_col: str = "business_name",
        top_k: int = DEFAULT_TOP_K,
        route_name: str = ROUTE_NAME_TOKEN_NAME,
    ) -> pd.DataFrame:
        """Query S1 records against the inverted index and return top-k candidates."""
        records = []

        for _, row in s1_df.iterrows():
            s1_id = str(row[id_col])
            raw = str(row.get(text_col, "") or "")
            tokens = set(tokenize(raw, min_len=self.min_token_len))

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

            sorted_candidates = sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
            top_candidates = sorted_candidates[:top_k]

            for rank, ((cand_id, cand_src), raw_score) in enumerate(
                top_candidates, start=1
            ):
                norm_score = min(raw_score / s1_idf_sum, 1.0)
                records.append(
                    {
                        "pair_key": f"{s1_id}::{cand_id}",
                        "s1_id": s1_id,
                        "candidate_id": cand_id,
                        "candidate_source": cand_src,
                        "route": route_name,
                        "rank": rank,
                        "score": round(float(norm_score), 6),
                    }
                )

        if not records:
            return _empty_candidate_df()

        return pd.DataFrame(records, columns=_candidate_columns())


# ---------------------------------------------------------------------------
# Route 3 Public Function: Rare-Token Name Retrieval
# ---------------------------------------------------------------------------
def retrieve_rare_token_name(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    top_k: int = DEFAULT_TOP_K,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    max_df_ratio: float = DEFAULT_MAX_DF_RATIO,
) -> pd.DataFrame:
    """Route 3: Retrieve candidates using rare/informative business name tokens."""
    index = RareTokenIndex(
        max_block_size=max_block_size,
        max_df_ratio=max_df_ratio,
        min_token_len=DEFAULT_MIN_TOKEN_LEN,
    )
    index.fit_and_index(
        s2_df=s2_df,
        s3_df=s3_df,
        s1_df=s1_df,
        id_col=id_col,
        text_col=name_col,
    )
    return index.query(
        s1_df=s1_df,
        id_col=id_col,
        text_col=name_col,
        top_k=top_k,
        route_name=ROUTE_NAME_TOKEN_NAME,
    )


# ---------------------------------------------------------------------------
# Route 6 Public Function: Rare-Token Address Retrieval
# ---------------------------------------------------------------------------
def retrieve_rare_token_address(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    addr_col: str = "business_address",
    top_k: int = DEFAULT_TOP_K,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    max_df_ratio: float = DEFAULT_MAX_DF_RATIO,
) -> pd.DataFrame:
    """Route 6: Retrieve candidates using rare/informative address tokens (localities, landmarks)."""
    index = RareTokenIndex(
        max_block_size=max_block_size,
        max_df_ratio=max_df_ratio,
        min_token_len=DEFAULT_MIN_TOKEN_LEN,
    )
    index.fit_and_index(
        s2_df=s2_df,
        s3_df=s3_df,
        s1_df=s1_df,
        id_col=id_col,
        text_col=addr_col,
    )
    return index.query(
        s1_df=s1_df,
        id_col=id_col,
        text_col=addr_col,
        top_k=top_k,
        route_name=ROUTE_NAME_TOKEN_ADDR,
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
