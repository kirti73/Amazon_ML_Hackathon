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
from itertools import chain
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .spill import EventSpillWriter, RecordBuffer, iter_spilled_buckets
from .tfidf_common import (
    build_l2_normalized_csr,
    canonical_top_k,
    iter_pool_blocks,
    iter_pool_names,
    merge_top_k,
    pool_size,
)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
ROUTE_NAME: str = "reverse_retrieval"
DEFAULT_TOP_K_REVERSE: int = 5
DEFAULT_CHUNK_SIZE: int = 5000
DEFAULT_NGRAM_RANGE: Tuple[int, int] = (3, 4)
DEFAULT_MIN_DF: int = 2
MIN_SCORE_THRESHOLD: float = 0.10

# Bounded-memory block sizes for the 2D retrieval. The S1 index (2,206,821 rows) is
# vectorized once and then sliced by row block, so it is never re-vectorized per candidate
# block; the candidate (S2+S3) side is streamed in blocks instead.
DEFAULT_QUERY_BLOCK: int = 1000
DEFAULT_INDEX_BLOCK: int = 100_000
DEFAULT_DEDUP_BUCKETS: int = 64

# Columns of the intermediate reverse link table. Reverse emits one row per
# (candidate -> selected S1) link; the per-s1_id rank is assigned afterwards, in phase 2.
_REVERSE_LINK_COLUMNS: List[str] = ["s1_id", "candidate_id", "candidate_source", "score"]


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

    def fit(self, raw_documents: Iterator[str]) -> "CharTfidfVectorizer":
        """Build the vocabulary and idf vector from a streamed corpus.

        Accepts any iterable so the caller can avoid materializing the ~12.5M-document
        combined corpus as a list of strings. ``n_docs`` is accumulated in the single pass.
        """
        df_counts: Counter = Counter()
        n_docs = 0
        for doc in raw_documents:
            n_docs += 1
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
        """Vectorize documents into a row-sorted, L2-normalized CSR matrix.

        Memory note
        -----------
        The original accumulated ``rows``/``cols``/``data`` as Python lists, which costs
        ~28 bytes per non-zero plus 8 bytes of list slot instead of 4. This version writes
        into preallocated NumPy buffers and assembles the CSR directly. Non-zero content is
        bit-identical to the original; see
        :func:`~src.blocking.tfidf_common.build_l2_normalized_csr` for the two
        canonicalization details (row-sorted indices, float32 pairwise row norm) that are
        required to preserve exact bit patterns.
        """
        n_docs = len(raw_documents)
        n_terms = len(self.vocab)
        if n_docs == 0 or n_terms == 0:
            return sp.csr_matrix((n_docs, n_terms), dtype=np.float32)

        vocab = self.vocab
        idf = self.idf
        sublinear = self.sublinear_tf
        log = math.log

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
# Route 7 Retrieval Pipeline
# ---------------------------------------------------------------------------
def _iter_nonempty_chain(
    s1_names: List[str],
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    name_col: str,
) -> Iterator[str]:
    """Stream the non-empty fit corpus in the original order: S1 first, then S2, then S3.

    Dropping empties is load-bearing: ``n_docs`` feeds the idf formula, so filtering here
    is what keeps the fitted vectorizer identical to the previous list-based version.
    """
    return chain(
        (name for name in s1_names if name),
        (name for name in iter_pool_names(s2_df, s3_df, name_col, normalize_name) if name),
    )


def _iter_reverse_records(
    links: pd.DataFrame,
) -> Iterator[List[dict]]:
    """Yield ranked reverse event records in deterministic batches, per s1_id.

    ``links`` holds every ``(s1_id, candidate_id, candidate_source, score)`` link for a set
    of s1_id values (one spill bucket). Ranking is per ``s1_id``: score desc, then
    candidate_id asc. Duplicate links for the same ``(s1_id, candidate_id)`` keep the
    highest score, matching the previous in-memory behaviour.
    """
    if links is None or len(links) == 0:
        return
    best = (
        links.groupby(["s1_id", "candidate_id", "candidate_source"], as_index=False)
        .agg(score=("score", "max"))
        .sort_values(
            by=["s1_id", "score", "candidate_id"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    best["rank"] = best.groupby("s1_id").cumcount() + 1

    batch: List[dict] = []
    for row in best.itertuples(index=False):
        s1_id = str(row.s1_id)
        cand_id = str(row.candidate_id)
        batch.append(
            {
                "pair_key": f"{s1_id}::{cand_id}",
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": str(row.candidate_source),
                "route": ROUTE_NAME,
                "rank": int(row.rank),
                "score": round(float(row.score), 6),
            }
        )
        if len(batch) >= 50_000:
            yield batch
            batch = []
    if batch:
        yield batch


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
    query_block: int = DEFAULT_QUERY_BLOCK,
    index_block: int = DEFAULT_INDEX_BLOCK,
    sink: Optional[object] = None,
    spill_dir: Optional[str] = None,
    dedup_buckets: int = DEFAULT_DEDUP_BUCKETS,
) -> pd.DataFrame:
    """Retrieve candidate pairs in the reverse direction: (S2 + S3) queries -> S1 index.

    Bounded-memory design
    ---------------------
    Retrieval is two-dimensional: the S2+S3 pool is streamed in ``query_block`` blocks and
    each is scored against row blocks of the S1 index in ``index_block`` slices. The S1 index
    is vectorized exactly once (2,206,821 rows, ~750 MB) and only its slices are
    transposed, so it is never re-vectorized per candidate block.

    The second bottleneck is different from the other routes: after selecting each
    candidate's top-``top_k_reverse`` S1 records, the results are transposed into a
    per-``s1_id`` candidate list that is **unbounded** — measured up to 1,222 candidates for
    a single S1, projecting to <= 51.6M links from this route alone. That list is therefore
    materialized one hash bucket at a time, either on disk (``spill_dir``) or in memory when
    no sink is supplied, and the per-``s1_id`` rank is assigned bucket by bucket.

    Correctness
    -----------
    The selected S1 set per candidate is identical to before, except that exact-score ties
    at the ``top_k_reverse`` boundary are resolved by ``(-score, s1_id)`` rather than by
    ``np.argpartition``'s arbitrary subset. Output ordering, the keep-highest-score
    duplicate rule, and the per-``s1_id`` ranking are unchanged.

    Parameters
    ----------
    s1_df, s2_df, s3_df : DataFrames with [id_col, name_col].
    id_col              : Entity ID column name.
    name_col            : Business name column name.
    top_k_reverse       : Max S1 records to retrieve per S2/S3 entity.
    chunk_size          : Retained for API compatibility; ``query_block`` drives iteration.
    ngram_range         : Character n-gram range.
    min_df              : Min document frequency.
    min_score           : Min cosine similarity score.
    vectorizer          : Optional pre-fitted vectorizer.
    query_block         : Number of S2/S3 queries scored per outer iteration.
    index_block         : Number of S1 index rows scored per inner iteration.
    sink                : Optional object with ``emit(list_of_records)``. When supplied,
                          records are streamed out instead of accumulated, so the returned
                          DataFrame is empty.
    spill_dir           : Optional directory for the intermediate reverse link table.
    dedup_buckets       : Number of hash buckets for the link table.

    Returns
    -------
    DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source, route, rank, score
    """
    n_s1 = 0 if s1_df is None else len(s1_df)
    n_candidates = pool_size(s2_df, s3_df)
    if n_s1 == 0 or n_candidates == 0:
        return _empty_candidate_df()

    # 1. S1 index data. S1 is 2.2M rows rather than 10.3M, so these two lists are
    #    affordable and are needed for random access by index during scoring.
    s1_ids = s1_df[id_col].astype(str).tolist()
    if name_col in s1_df.columns:
        s1_names = [
            normalize_name(raw) for raw in s1_df[name_col].fillna("").astype(str).tolist()
        ]
    else:
        s1_names = [""] * n_s1

    # 2. Fit or use vectorizer, streaming the corpus.
    if vectorizer is None:
        effective_min_df = min_df if (n_candidates + n_s1) >= 100 else 1
        vectorizer = CharTfidfVectorizer(
            ngram_range=ngram_range,
            min_df=effective_min_df,
            sublinear_tf=True,
        )
        corpus = _iter_nonempty_chain(s1_names, s2_df, s3_df, name_col)
        first = next(corpus, None)
        if first is None:
            return _empty_candidate_df()
        vectorizer.fit(chain([first], corpus))

    # 3. Vectorize the S1 index once, then reuse it via row slices.
    s1_matrix = vectorizer.transform(s1_names)  # (N_s1, V) CSR

    # 4. Two-dimensional retrieval over the streamed candidate pool.
    #    Each link found is appended to link_frames; they are grouped into per-s1 buckets
    #    in phase 5 so that the unbounded transposed result never lives in one dict.
    link_frames: List[pd.DataFrame] = []
    use_disk = spill_dir is not None
    link_writer: Optional[EventSpillWriter] = None
    if use_disk:
        link_writer = EventSpillWriter(
            f"{spill_dir}/reverse_links",
            n_buckets=dedup_buckets,
            columns=tuple(_REVERSE_LINK_COLUMNS),
        )

    n_queries = n_candidates
    # One shared cursor over the candidate pool: consumed forward exactly once, so the
    # outer query loop stays O(pool) overall rather than O(pool) per block.
    block_cursor = iter_pool_blocks(
        s2_df, s3_df, id_col, name_col, query_block, normalize_name
    )
    for _q_start in range(0, n_queries, query_block):
        cand_ids: List[str] = []
        cand_sources: List[str] = []
        cand_names: List[str] = []
        while len(cand_ids) < query_block:
            try:
                block_ids, block_names, block_source = next(block_cursor)
            except StopIteration:
                break
            cand_ids.extend(block_ids)
            cand_sources.extend(block_source for _ in block_ids)
            cand_names.extend(block_names)
        n_block = len(cand_ids)
        if n_block == 0:
            continue
        q_matrix = vectorizer.transform(cand_names)  # (B_q, V) CSR

        # Running top-top_k_reverse S1 records per candidate, bounded by B_q * k.
        running: List[List[Tuple[float, str]]] = [[] for _ in range(n_block)]

        for lo in range(0, n_s1, index_block):
            hi = min(lo + index_block, n_s1)
            index_slice_t = s1_matrix[lo:hi].T.tocsc()  # (V, B_i) CSC
            sim_matrix = q_matrix.dot(index_slice_t).tocsr()  # (B_q, B_i)
            del index_slice_t

            s1_block_ids = s1_ids[lo:hi]
            for i in range(n_block):
                row_start = sim_matrix.indptr[i]
                row_end = sim_matrix.indptr[i + 1]
                if row_start == row_end:
                    continue
                # canonical_top_k sorts by (-score, <id>); here the selected entity is the
                # S1 record, so the tie-break key is s1_id.
                block_items = canonical_top_k(
                    sim_matrix.indices[row_start:row_end],
                    sim_matrix.data[row_start:row_end],
                    s1_block_ids,
                    "S1",
                    top_k_reverse,
                    min_score,
                )
                if block_items:
                    running[i] = merge_top_k(running[i], block_items, top_k_reverse)
            del sim_matrix

        del q_matrix

        if running:
            rows_s1: List[str] = []
            rows_cand: List[str] = []
            rows_src: List[str] = []
            rows_score: List[float] = []
            for i, items in enumerate(running):
                if not items:
                    continue
                cid = cand_ids[i]
                csrc = cand_sources[i]
                for score_val, s1_id, _index_source in items:
                    rows_s1.append(s1_id)
                    rows_cand.append(cid)
                    rows_src.append(csrc)
                    rows_score.append(score_val)
            if rows_s1:
                links = pd.DataFrame(
                    {
                        "s1_id": rows_s1,
                        "candidate_id": rows_cand,
                        "candidate_source": rows_src,
                        "score": rows_score,
                    }
                )
                if link_writer is not None:
                    link_writer.write_frame(links)
                else:
                    link_frames.append(links)

    del s1_matrix

    # 5. Phase 2: per-s1_id ranking, one hash bucket at a time.
    # ``buf`` is either the caller's sink or a sink-less buffer that retains records, which
    # keeps the in-memory equivalence path working while removing the unbounded list.
    buf = sink if sink is not None else RecordBuffer(None)

    def _handle(bucket: pd.DataFrame) -> None:
        for batch in _iter_reverse_records(bucket):
            if sink is not None:
                sink.emit(batch)
            else:
                for record in batch:
                    buf.append(record)

    if link_writer is not None:
        link_writer.close()
        for _bucket_id, frame in iter_spilled_buckets(
            f"{spill_dir}/reverse_links",
            n_buckets=dedup_buckets,
            columns=tuple(_REVERSE_LINK_COLUMNS),
        ):
            _handle(frame)
    else:
        if link_frames:
            _handle(pd.concat(link_frames, ignore_index=True))

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
