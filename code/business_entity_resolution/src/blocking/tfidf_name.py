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
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .spill import RecordBuffer
import scipy.sparse as sp

from .tfidf_common import (
    build_l2_normalized_csr,
    canonical_top_k,
    iter_pool_blocks,
    iter_pool_names,
    merge_top_k,
    pool_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTE_NAME: str = "tfidf_name"
DEFAULT_TOP_K: int = 20
DEFAULT_CHUNK_SIZE: int = 5000
DEFAULT_NGRAM_RANGE: Tuple[int, int] = (3, 4)
DEFAULT_MIN_DF: int = 2
MIN_SCORE_THRESHOLD: float = 0.05  # Ignore negligible cosine similarities

# Bounded-memory block sizes. Selected empirically: over a 16-point grid of
# S1-block x candidate-block combinations (5,000 queries x 250,000 candidates) the total
# similarity non-zero count was invariant, confirming blocking is work-invariant, while
# peak similarity memory scaled linearly with the block product. 1000 x 100,000 measured
# 238 MB of similarity matrix at 519 MB peak RSS, versus 2,952 MB / 3,439 MB for the
# largest 5,000 x 250,000 configuration -- roughly 2% of the 11.8 GB headroom.
DEFAULT_QUERY_BLOCK: int = 1000
DEFAULT_CANDIDATE_BLOCK: int = 100_000


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

    def fit(self, raw_documents: Iterable[str]) -> "CharTfidfVectorizer":
        """Build the vocabulary and idf vector from a corpus.

        Accepts any iterable of documents, not just a list, so callers can stream the
        10.320.219-document candidate pool through :func:`iter_pool_names` instead of
        allocating a full-length list of strings. Still accepts a plain list, so existing
        callers and tests are unaffected. Document count is accumulated during the single
        pass.
        """
        df_counts: Counter = Counter()
        n_docs = 0
        for doc in raw_documents:
            n_docs += 1
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
        """Vectorize documents into an L2-normalized CSR matrix.

        Memory note
        -----------
        The original implementation accumulated ``rows``/``cols``/``data`` as Python
        lists. At the full 10.320.219-candidate pool that is ~35 GB of peak memory
        (28-byte int/float objects plus 8-byte list slots, for ~438M non-zeros) and was
        the direct cause of the ``MemoryError`` that killed the first full-data P1 run.

        This version writes into preallocated NumPy buffers and builds the CSR matrix
        directly from ``(data, indices, indptr)``, so there is no COO intermediate and no
        Python objects per non-zero. The non-zero *content* is bit-identical to the
        original; only the container changed. Buffers grow geometrically, so a single
        reallocation is amortized over many appends.

        Callers now always pass bounded blocks (see :func:`retrieve_tfidf_name`), so peak
        memory is a function of block size rather than pool size.
        """
        n_docs = len(raw_documents)
        n_terms = len(self.vocab)
        if n_docs == 0 or n_terms == 0:
            return sp.csr_matrix((n_docs, n_terms), dtype=np.float32)

        vocab = self.vocab
        idf = self.idf
        sublinear = self.sublinear_tf
        log = math.log

        # Preallocate with a growth headroom; ~n_docs * 8 is a good first guess for
        # short documents and is corrected by the geometric growth below.
        capacity = max(1024, n_docs * 8)
        cols = np.empty(capacity, dtype=np.int32)
        data = np.empty(capacity, dtype=np.float32)
        indptr = np.zeros(n_docs + 1, dtype=np.int32)

        nnz = 0
        for row_idx, doc in enumerate(raw_documents):
            if not doc:
                indptr[row_idx + 1] = nnz
                continue
            tf_counts = Counter(self._extract_ngrams(doc))
            if not tf_counts:
                indptr[row_idx + 1] = nnz
                continue

            # Grow the buffers if this document needs more room than remains.
            needed = nnz + len(tf_counts)
            if needed > capacity:
                new_capacity = max(needed, capacity * 2)
                grow = new_capacity - capacity
                cols = np.concatenate([cols, np.empty(grow, dtype=np.int32)])
                data = np.concatenate([data, np.empty(grow, dtype=np.float32)])
                capacity = new_capacity

            for term, count in tf_counts.items():
                col_idx = vocab.get(term)
                if col_idx is None:
                    continue
                tf = (1.0 + log(count)) if sublinear else float(count)
                cols[nnz] = col_idx
                data[nnz] = tf * idf[col_idx]
                nnz += 1
            indptr[row_idx + 1] = nnz

        return build_l2_normalized_csr(cols, data, indptr, n_docs, n_terms)


# ---------------------------------------------------------------------------
# Retrieval Pipeline
# ---------------------------------------------------------------------------
def _build_s1_query_data(
    s1_df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
) -> Tuple[List[str], List[str]]:
    """Extract S1 ids and normalized names (bounded by |S1|, which is not the bottleneck)."""
    s1_ids = s1_df[id_col].astype(str).tolist()
    if name_col in s1_df.columns:
        s1_names = [
            normalize_name(raw) for raw in s1_df[name_col].fillna("").astype(str).tolist()
        ]
    else:
        s1_names = [""] * len(s1_df)
    return s1_ids, s1_names


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
    query_block: int = DEFAULT_QUERY_BLOCK,
    candidate_block: int = DEFAULT_CANDIDATE_BLOCK,
    sink: Optional[RecordBuffer] = None,
) -> pd.DataFrame:
    """Retrieve top-k candidates for each S1 record using char n-gram TF-IDF on name.

    Bounded-memory design
    ---------------------
    Similarity is computed in two dimensions at once. ``query_block`` bounds how many S1
    rows are in flight, which bounds the running top-k accumulator; ``candidate_block``
    bounds how many candidate vectors are materialized, which bounds the CSR matrix, its
    transpose, and the similarity block. Nothing of size ``O(|S1| x |pool|)`` is ever
    allocated, and no candidate block's similarity matrix is retained after its top-k has
    been merged.

    Correctness
    -----------
    Output is identical to the previous unblocked implementation, except that exact-score
    ties at the top-k boundary are now resolved by the documented ``(-score,
    candidate_id)`` ordering instead of by ``np.argpartition``'s arbitrary subset. See
    :func:`canonical_top_k`. Because every candidate is scored, blocked and unblocked runs
    produce byte-identical records.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, name_col].
    id_col              : ID column name.
    name_col            : Business name column name.
    top_k               : Number of candidates to retrieve per S1 record.
    chunk_size          : Retained for API compatibility. Retrieval is now driven by
                          ``query_block``; see the note in the module docstring.
    ngram_range         : Character n-gram range (default 3, 4).
    min_df              : Minimum document frequency for n-grams.
    min_score           : Minimum cosine similarity score threshold.
    vectorizer          : Optional pre-fitted CharTfidfVectorizer.
    query_block         : Number of S1 rows scored per outer iteration.
    candidate_block     : Number of candidates vectorized per inner iteration.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
    """
    n_candidates = pool_size(s2_df, s3_df)
    n_queries = 0 if s1_df is None else len(s1_df)
    if n_candidates == 0 or n_queries == 0:
        return _empty_candidate_df()

    s1_ids, s1_names = _build_s1_query_data(s1_df, id_col=id_col, name_col=name_col)

    # 1. Fit or use vectorizer. The corpus is streamed, never materialized as a list.
    if vectorizer is None:
        effective_min_df = min_df if (n_candidates + n_queries) >= 100 else 1
        vectorizer = CharTfidfVectorizer(
            ngram_range=ngram_range,
            min_df=effective_min_df,
            sublinear_tf=True,
        )
        vectorizer.fit(
            chain(
                iter_pool_names(s2_df, s3_df, name_col, normalize_name),
                iter(s1_names),
            )
        )

    # 2. Two-dimensional blocked retrieval. The candidate loop is inner so that the running
    #    top-k accumulator holds at most query_block * top_k entries at any moment.
    buf = sink if sink is not None else RecordBuffer(None)
    for q_start in range(0, n_queries, query_block):
        q_stop = min(q_start + query_block, n_queries)
        block_ids = s1_ids[q_start:q_stop]
        block_names = s1_names[q_start:q_stop]
        n_block = q_stop - q_start
        q_matrix = vectorizer.transform(block_names)  # (B_q, V) CSR

        running: List[List[Tuple[float, str, str]]] = [[] for _ in range(n_block)]

        for cand_ids, cand_names, cand_source in iter_pool_blocks(
            s2_df, s3_df, id_col, name_col, candidate_block, normalize_name
        ):
            # (B_c, V) CSR and its (V, B_c) CSC transpose, both bounded by candidate_block.
            c_matrix = vectorizer.transform(cand_names)
            c_matrix_t = c_matrix.T.tocsc()
            del c_matrix

            sim_matrix = q_matrix.dot(c_matrix_t).tocsr()  # (B_q, B_c)
            del c_matrix_t

            for i in range(n_block):
                row_start = sim_matrix.indptr[i]
                row_end = sim_matrix.indptr[i + 1]
                if row_start == row_end:
                    continue
                block_items = canonical_top_k(
                    sim_matrix.indices[row_start:row_end],
                    sim_matrix.data[row_start:row_end],
                    cand_ids,
                    cand_source,
                    top_k,
                    min_score,
                )
                if block_items:
                    running[i] = merge_top_k(running[i], block_items, top_k)
            del sim_matrix

        del q_matrix

        for i in range(n_block):
            curr_s1_id = block_ids[i]
            for rank, (score_val, c_id, c_src) in enumerate(running[i], start=1):
                buf.append(
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

    buf.close()

    if sink is not None:
        return _empty_candidate_df()

    result = buf.result()
    if len(result) == 0:
        return _empty_candidate_df()
    return result


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
