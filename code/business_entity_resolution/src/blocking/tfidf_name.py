"""
Route 2 — Character N-Gram TF-IDF Retrieval (Business Name)
============================================================

Strategy
--------
1. Normalize names using the conservative P1 stub.
2. Build character 3,4-gram TF-IDF representations for the combined
   S2 + S3 candidate pool.
3. Query S1 in memory-safe chunks (e.g., 5,000 rows at a time).
4. For each S1 query, retrieve top-k candidates (default k=20) by cosine
   similarity without ever materializing the full S1 x (S2+S3) matrix.
5. Deterministic tie-breaking: sort by score desc, then candidate_id asc.

Candidate Schema
----------------
pair_key, s1_id, candidate_id, candidate_source, route, rank, score
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTE_NAME: str = "tfidf_name"
DEFAULT_TOP_K: int = 20
DEFAULT_CHUNK_SIZE: int = 5000
DEFAULT_NGRAM_RANGE: Tuple[int, int] = (3, 4)
DEFAULT_MIN_DF: int = 2
MIN_SCORE_THRESHOLD: float = 0.05  # Ignore negligible cosine similarities


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
    """
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Self-contained Sparse Character N-Gram TF-IDF Vectorizer
# ---------------------------------------------------------------------------
class CharTfidfVectorizer:
    """Fast, memory-safe sparse character n-gram TF-IDF vectorizer.

    Produces L2-normalized CSR sparse matrices using standard sublinear
    TF-IDF weighting: tf = 1 + log(tf), idf = log((1 + N) / (1 + df)) + 1.
    """

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

        # Sort terms alphabetically for deterministic vocabulary indexing
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

        # L2 normalize each row in-place
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
# Retrieval Pipeline
# ---------------------------------------------------------------------------
def _build_candidate_pool(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
) -> Tuple[List[str], List[str], List[str]]:
    """Extract and normalize candidate names from S2 and S3."""
    cand_ids: List[str] = []
    cand_sources: List[str] = []
    cand_names: List[str] = []

    for df, source_label in [(s2_df, "S2"), (s3_df, "S3")]:
        for _, row in df.iterrows():
            eid = str(row[id_col])
            raw_name = str(row.get(name_col, "") or "")
            norm_name = normalize_name(raw_name)
            cand_ids.append(eid)
            cand_sources.append(source_label)
            cand_names.append(norm_name)

    return cand_ids, cand_sources, cand_names


def retrieve_tfidf_name(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    min_score: float = MIN_SCORE_THRESHOLD,
    vectorizer: Optional[CharTfidfVectorizer] = None,
) -> pd.DataFrame:
    """Retrieve top-k candidates for each S1 record using char n-gram TF-IDF on name.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, name_col].
    id_col              : ID column name.
    name_col            : Business name column name.
    top_k               : Number of candidates to retrieve per S1 record.
    chunk_size          : Batch size of S1 queries to process at once.
    ngram_range         : Character n-gram range (default 3, 4).
    min_df              : Minimum document frequency for n-grams.
    min_score           : Minimum cosine similarity score threshold.
    vectorizer          : Optional pre-fitted CharTfidfVectorizer.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
    """
    # 1. Prepare candidate pool
    cand_ids, cand_sources, cand_names = _build_candidate_pool(
        s2_df, s3_df, id_col=id_col, name_col=name_col
    )
    n_candidates = len(cand_ids)
    if n_candidates == 0 or len(s1_df) == 0:
        return _empty_candidate_df()

    # 2. Prepare S1 query data
    s1_ids: List[str] = []
    s1_names: List[str] = []
    for _, row in s1_df.iterrows():
        s1_ids.append(str(row[id_col]))
        raw_name = str(row.get(name_col, "") or "")
        s1_names.append(normalize_name(raw_name))

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
        # Fit on candidate pool + S1 queries
        all_corpus = cand_names + s1_names
        vectorizer.fit(all_corpus)

    # 4. Transform candidate pool (l2-normalized)
    cand_matrix = vectorizer.transform(cand_names)  # (N_cand, V) CSR
    cand_matrix_t = cand_matrix.T.tocsc()          # (V, N_cand) CSC for fast dot

    # 5. Process S1 queries in memory-safe chunks
    records = []
    n_queries = len(s1_ids)

    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, n_queries)
        chunk_s1_ids = s1_ids[start_idx:end_idx]
        chunk_s1_names = s1_names[start_idx:end_idx]

        # Vectorize chunk queries: (chunk_len, V) CSR
        chunk_q = vectorizer.transform(chunk_s1_names)

        # Chunk similarity matrix: (chunk_len, N_cand)
        sim_matrix = chunk_q.dot(cand_matrix_t).tocsr()

        # Extract top-k for each query row
        for i in range(sim_matrix.shape[0]):
            curr_s1_id = chunk_s1_ids[i]
            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i + 1]

            if row_start == row_end:
                continue

            row_indices = sim_matrix.indices[row_start:row_end]
            row_data = sim_matrix.data[row_start:row_end]

            # Filter by min_score
            mask = row_data >= min_score
            if not np.any(mask):
                continue

            valid_indices = row_indices[mask]
            valid_scores = row_data[mask]

            # Select top-k
            if len(valid_scores) > top_k:
                top_part_idx = np.argpartition(-valid_scores, top_k)[:top_k]
                selected_cand_indices = valid_indices[top_part_idx]
                selected_scores = valid_scores[top_part_idx]
            else:
                selected_cand_indices = valid_indices
                selected_scores = valid_scores

            # Build candidate list with deterministic tie-breaking:
            # (-score, candidate_id)
            items = []
            for cand_idx, score_val in zip(selected_cand_indices, selected_scores):
                c_id = cand_ids[cand_idx]
                c_src = cand_sources[cand_idx]
                items.append((float(score_val), c_id, c_src))

            items.sort(key=lambda x: (-x[0], x[1]))
            items = items[:top_k]

            for rank, (score_val, c_id, c_src) in enumerate(items, start=1):
                records.append(
                    {
                        "pair_key": f"{curr_s1_id}::{c_id}",
                        "s1_id": curr_s1_id,
                        "candidate_id": c_id,
                        "candidate_source": c_src,
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
