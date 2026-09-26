"""
Route 7 — Reverse Retrieval ((S2 + S3) -> S1)
==============================================

Strategy
--------
1. Vectorize the S1 index (business name char 3,4-grams).
2. Query candidate records from S2 and S3 against the S1 index in memory-safe chunks.
3. For each S2/S3 query, retrieve top-k_reverse (default k=5) matching S1 records.
4. Convert every reverse match (candidate -> S1) into standard candidate schema
   (s1_id, candidate_id, candidate_source, pair_key).
5. Deterministic tie-breaking and rank calculation per S1 record:
   sort by score desc, then candidate_id asc.

Candidate Schema
----------------
pair_key, s1_id, candidate_id, candidate_source, route, rank, score
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
ROUTE_NAME: str = "reverse_retrieval"
DEFAULT_TOP_K_REVERSE: int = 5
DEFAULT_CHUNK_SIZE: int = 5000
DEFAULT_NGRAM_RANGE: Tuple[int, int] = (3, 4)
DEFAULT_MIN_DF: int = 2
MIN_SCORE_THRESHOLD: float = 0.10


# ---------------------------------------------------------------------------
# Normalization Stub
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_name(text: str) -> str:
    """Conservative P1 normalization for name strings."""
    if not text or not isinstance(text, str):
        return ""
    if text.strip().lower() in ("nan", "none", "null"):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Standalone Sparse Vectorizer
# ---------------------------------------------------------------------------
class CharTfidfVectorizer:
    """Fast, memory-safe sparse character n-gram TF-IDF vectorizer."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def _extract_ngrams(self, text: str) -> List[str]:
        ngrams: List[str] = []
        L = len(text)
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if L >= n:
                for i in range(L - n + 1):
                    ngrams.append(text[i : i + n])
        return ngrams

    def fit(self, raw_documents: List[str]) -> "CharTfidfVectorizer":
        df_counts: Counter = Counter()
        n_docs = len(raw_documents)
        for doc in raw_documents:
            if not doc:
                continue
            unique_ngrams = set(self._extract_ngrams(doc))
            df_counts.update(unique_ngrams)

        valid_terms = sorted(
            [term for term, count in df_counts.items() if count >= self.min_df]
        )
        self.vocab = {term: idx for idx, term in enumerate(valid_terms)}

        n_terms = len(self.vocab)
        self.idf = np.zeros(n_terms, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = df_counts[term]
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, raw_documents: List[str]) -> sp.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for row_idx, doc in enumerate(raw_documents):
            if not doc:
                continue
            ngrams = self._extract_ngrams(doc)
            if not ngrams:
                continue
            tf_counts = Counter(ngrams)
            for term, count in tf_counts.items():
                col_idx = self.vocab.get(term)
                if col_idx is not None:
                    tf = (
                        (1.0 + math.log(count))
                        if self.sublinear_tf
                        else float(count)
                    )
                    val = tf * self.idf[col_idx]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(val)

        n_docs = len(raw_documents)
        n_terms = len(self.vocab)
        mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32
        )

        for i in range(n_docs):
            r_start = mat.indptr[i]
            r_end = mat.indptr[i + 1]
            if r_start < r_end:
                r_data = mat.data[r_start:r_end]
                norm = float(np.sqrt(np.sum(r_data**2)))
                if norm > 0.0:
                    mat.data[r_start:r_end] /= norm

        return mat


# ---------------------------------------------------------------------------
# Route 7 Retrieval Pipeline
# ---------------------------------------------------------------------------
def retrieve_reverse(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    top_k_reverse: int = DEFAULT_TOP_K_REVERSE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    min_score: float = MIN_SCORE_THRESHOLD,
    vectorizer: Optional[CharTfidfVectorizer] = None,
) -> pd.DataFrame:
    """Retrieve candidate pairs in the reverse direction: (S2 + S3) queries -> S1 index.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, name_col].
    id_col              : Entity ID column name.
    name_col            : Business name column name.
    top_k_reverse       : Max S1 records to retrieve per S2/S3 entity.
    chunk_size          : Chunk size for S2/S3 querying.
    ngram_range         : Character n-gram range.
    min_df              : Min document frequency.
    min_score           : Min cosine similarity score.
    vectorizer          : Optional pre-fitted vectorizer.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
    """
    if len(s1_df) == 0 or (len(s2_df) == 0 and len(s3_df) == 0):
        return _empty_candidate_df()

    # 1. Prepare S1 index data
    s1_ids: List[str] = []
    s1_names: List[str] = []
    for _, row in s1_df.iterrows():
        s1_ids.append(str(row[id_col]))
        raw = str(row.get(name_col, "") or "")
        s1_names.append(normalize_name(raw))

    # 2. Prepare S2 and S3 query pool
    cand_ids: List[str] = []
    cand_sources: List[str] = []
    cand_names: List[str] = []
    for df, src_label in [(s2_df, "S2"), (s3_df, "S3")]:
        for _, row in df.iterrows():
            eid = str(row[id_col])
            raw = str(row.get(name_col, "") or "")
            cand_ids.append(eid)
            cand_sources.append(src_label)
            cand_names.append(normalize_name(raw))

    # 3. Fit or use vectorizer
    if vectorizer is None:
        effective_min_df = (
            min_df if (len(cand_names) + len(s1_names)) >= 100 else 1
        )
        vectorizer = CharTfidfVectorizer(
            ngram_range=ngram_range,
            min_df=effective_min_df,
            sublinear_tf=True,
        )
        all_corpus = [n for n in (s1_names + cand_names) if n]
        if all_corpus:
            vectorizer.fit(all_corpus)
        else:
            return _empty_candidate_df()

    # 4. Transform S1 index (N_s1, V) and transpose for CSC dot
    s1_matrix = vectorizer.transform(s1_names)
    s1_matrix_t = s1_matrix.T.tocsc()

    # 5. Query candidate pool in chunks against S1 index
    # Map from s1_id -> dict of {cand_id: (candidate_source, max_score)}
    s1_to_candidates: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)

    n_queries = len(cand_ids)
    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, n_queries)
        chunk_cids = cand_ids[start_idx:end_idx]
        chunk_csrcs = cand_sources[start_idx:end_idx]
        chunk_cnames = cand_names[start_idx:end_idx]

        # Vectorize chunk queries: (chunk_len, V)
        chunk_q = vectorizer.transform(chunk_cnames)

        # Dot product against S1: (chunk_len, N_s1)
        sim_matrix = chunk_q.dot(s1_matrix_t).tocsr()

        for i in range(sim_matrix.shape[0]):
            cand_id = chunk_cids[i]
            cand_src = chunk_csrcs[i]

            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i + 1]
            if row_start == row_end:
                continue

            row_indices = sim_matrix.indices[row_start:row_end]
            row_data = sim_matrix.data[row_start:row_end]

            mask = row_data >= min_score
            if not np.any(mask):
                continue

            valid_s1_indices = row_indices[mask]
            valid_scores = row_data[mask]

            if len(valid_scores) > top_k_reverse:
                top_part_idx = np.argpartition(-valid_scores, top_k_reverse)[
                    :top_k_reverse
                ]
                selected_s1_indices = valid_s1_indices[top_part_idx]
                selected_scores = valid_scores[top_part_idx]
            else:
                selected_s1_indices = valid_s1_indices
                selected_scores = valid_scores

            for s1_idx, score_val in zip(selected_s1_indices, selected_scores):
                s1_id = s1_ids[s1_idx]
                score_f = float(score_val)
                # Keep highest score if duplicate encountered
                if (
                    cand_id not in s1_to_candidates[s1_id]
                    or score_f > s1_to_candidates[s1_id][cand_id][1]
                ):
                    s1_to_candidates[s1_id][cand_id] = (cand_src, score_f)

    # 6. Format final candidate records sorted deterministically per S1
    records = []
    # Sort S1 keys for deterministic output ordering
    sorted_s1_ids = sorted(s1_to_candidates.keys())

    for s1_id in sorted_s1_ids:
        cand_dict = s1_to_candidates[s1_id]
        # Sort candidates: score desc, then candidate_id asc
        sorted_cand_items = sorted(
            cand_dict.items(),
            key=lambda item: (-item[1][1], item[0]),
        )

        for rank, (cand_id, (cand_src, score_val)) in enumerate(
            sorted_cand_items, start=1
        ):
            records.append(
                {
                    "pair_key": f"{s1_id}::{cand_id}",
                    "s1_id": s1_id,
                    "candidate_id": cand_id,
                    "candidate_source": cand_src,
                    "route": ROUTE_NAME,
                    "rank": rank,
                    "score": round(score_val, 6),
                }
            )

    if not records:
        return _empty_candidate_df()

    return pd.DataFrame(records, columns=_candidate_columns())


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
