"""Shared helpers for bounded-memory TF-IDF blocking.

Two concerns are centralized here so that the name and address routes cannot drift apart:

1. :func:`canonical_top_k` — the *authoritative* top-k selection. It implements the
   documented ``(-score, candidate_id)`` ordering exactly, which the previous
   ``np.argpartition`` implementation did not (see :func:`canonical_top_k` docstring).

2. :func:`iter_pool_blocks` — L3 streaming. The candidate pool is ``S2 + S3`` and reaches
   10,320,219 rows on the full dataset. Neither the pool's normalized names nor its IDs are
   ever materialized as full-length Python lists; they are produced one bounded block at a
   time, and the source label is derived from the block's origin rather than stored.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "build_l2_normalized_csr",
    "canonical_top_k",
    "iter_pool_blocks",
    "iter_pool_names",
    "pool_size",
]


# ---------------------------------------------------------------------------
# L1: assemble a canonical, L2-normalized CSR matrix from raw buffers
# ---------------------------------------------------------------------------
def build_l2_normalized_csr(
    cols: np.ndarray,
    data: np.ndarray,
    indptr: np.ndarray,
    n_docs: int,
    n_terms: int,
) -> sp.csr_matrix:
    """Assemble a row-sorted, L2-normalized CSR matrix from raw buffers.

    Two details here are required to keep the result **bit-identical** to the original
    ``csr_matrix((data, (rows, cols)))`` implementation, and both were found by
    differential testing rather than assumed:

    1. **Row-sorted indices.** The original built a COO matrix and let SciPy canonicalize
       it, which sorts column indices ascending within every row. Constructing the CSR
       directly from ``indptr`` skips that step, so the indices would be left in
       n-gram-encounter order. The entry *set* is identical either way and the dot
       product is unaffected, but a permuted row changes the floating-point summation
       order in step 2, so the values would drift by ~1 ulp. This function applies the
       same within-row sort up front.

    2. **float32 pairwise summation for the row norm.** ``scipy.sparse.linalg.norm`` and
       ``np.bincount``-based row norms accumulate the sum of squares in float64, whereas
       the original used ``np.sqrt(np.sum(r_data ** 2))`` on a float32 slice, which uses
       float32 pairwise summation. The two disagree in the last ulp. The loop below
       reproduces the original expression verbatim, per row, so the normalized values
       match exactly. The loop is over rows of a *bounded block* (at most
       ``candidate_block``), and costs on the order of 2 us per row, so it is negligible
       next to the matrix product.

    Empty rows are skipped entirely: they have no entries, and every stored value is a
    positive ``tf * idf`` product (idf has a ``+1.0`` floor), so a row with entries always
    has a strictly positive norm and no zero-norm division is possible.
    """
    if n_docs == 0 or n_terms == 0:
        return sp.csr_matrix((n_docs, n_terms), dtype=np.float32)

    counts = np.diff(indptr)
    nnz = int(counts.sum()) if counts.size else 0
    if nnz == 0:
        return sp.csr_matrix((n_docs, n_terms), dtype=np.float32)

    if counts.size > 1 and not np.all(counts[:-1] <= counts[1:]):
        # Not expected (documents are appended in order), but keep the sort well-defined.
        order = np.lexsort((cols[:nnz], np.repeat(np.arange(n_docs, dtype=np.int64), counts)))
        cols = cols[:nnz][order]
        data = data[:nnz][order]
    else:
        # Sort column indices within each row, matching SciPy's COO canonicalization.
        if nnz > 1:
            row_ids = np.repeat(np.arange(n_docs, dtype=np.int32), counts)
            order = np.lexsort((cols[:nnz], row_ids))
            if not np.array_equal(order, np.arange(nnz)):
                cols = cols[:nnz][order]
                data = data[:nnz][order]

    mat = sp.csr_matrix(
        (data[:nnz], cols[:nnz], indptr), shape=(n_docs, n_terms), dtype=np.float32
    )

    # Reproduce the original per-row float32 pairwise sum exactly.
    mat_data = mat.data
    mat_indptr = mat.indptr
    for i in range(n_docs):
        r_start = mat_indptr[i]
        r_end = mat_indptr[i + 1]
        if r_start < r_end:
            r_data = mat_data[r_start:r_end]
            norm = float(np.sqrt(np.sum(r_data**2)))
            if norm > 0.0:
                mat_data[r_start:r_end] = r_data / norm

    return mat


# ---------------------------------------------------------------------------
# Canonical top-k selection
# ---------------------------------------------------------------------------
def canonical_top_k(
    row_indices: np.ndarray,
    row_data: np.ndarray,
    cand_ids: Sequence[str],
    cand_source: str,
    top_k: int,
    min_score: float,
) -> List[Tuple[float, str, str]]:
    """Select the top-k entries of one similarity row, with exact tie-breaking.

    Correctness note
    ----------------
    The original implementation used ``np.argpartition(-scores, top_k)[:top_k]`` to pick a
    *subset* of size ``k`` and only then sorted that subset. ``argpartition`` is not
    order-preserving, so when several candidates share a score that straddles the k-th
    rank, which of them landed in the subset depended on the layout of the array — i.e. on
    the candidate block size. The subsequent sort could not repair that, because the
    excluded-but-equal candidates were no longer available. This was latent in the original
    full-pool implementation and is *exposed* by candidate blocking.

    This function first takes the k-th largest score, retains **every** candidate scoring at
    least that value, then applies the full canonical ``(-score, candidate_id)`` sort and
    only then truncates to ``k``. The result is therefore independent of how the candidate
    pool is decomposed into blocks: blocked output is identical to unblocked output,
    including ties.

    ``cand_source`` is a scalar because candidate blocks never straddle the S2/S3 boundary
    (see :func:`iter_pool_blocks`); this also avoids materializing a per-candidate list of
    source labels, which for the full pool would be 10.320.219 pointers.

    Returns
    -------
    List of ``(score, candidate_id, candidate_source)``, already sorted by
    ``(-score, candidate_id)`` and truncated to ``top_k``.
    """
    if top_k <= 0 or row_data.size == 0:
        return []

    mask = row_data >= min_score
    n_valid = int(mask.sum())
    if n_valid == 0:
        return []

    valid_indices = row_indices[mask]
    valid_scores = row_data[mask]

    if n_valid > top_k:
        # k-th largest score. n_valid > top_k guarantees the kth index is in [1, n_valid).
        kth = np.partition(valid_scores, n_valid - top_k)[n_valid - top_k]
        # Retain all candidates at or above the k-th score so that exact ties at the
        # boundary remain available for the canonical sort to order by candidate_id.
        keep = valid_scores >= kth
        valid_indices = valid_indices[keep]
        valid_scores = valid_scores[keep]

    items = [
        (float(score_val), cand_ids[col], cand_source)
        for col, score_val in zip(valid_indices, valid_scores)
    ]
    items.sort(key=lambda item: (-item[0], item[1]))
    return items[:top_k]


def merge_top_k(
    existing: List[Tuple[float, str, str]],
    incoming: List[Tuple[float, str, str]],
    top_k: int,
) -> List[Tuple[float, str, str]]:
    """Merge one candidate block's top-k into a running top-k, keeping the best ``top_k``.

    Taking each block's top-k before merging is lossless: the canonical
    ``(-score, candidate_id)`` order is a total order, so if a candidate survives the
    global top-k then strictly fewer than ``top_k`` candidates precede it in that total
    order overall, and therefore fewer than ``top_k`` precede it within its own block — so
    it is necessarily in that block's top-k. Merging and re-truncating under the same total
    order is therefore exact, and the running list never exceeds ``top_k`` entries.
    """
    if not incoming:
        return existing
    if not existing:
        return incoming[:top_k]
    merged = existing + incoming
    merged.sort(key=lambda item: (-item[0], item[1]))
    return merged[:top_k]


# ---------------------------------------------------------------------------
# L3: stream the candidate pool in bounded blocks
# ---------------------------------------------------------------------------
def pool_size(s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> int:
    """Total number of candidates in the virtual ``S2 + S3`` pool."""
    n_s2 = 0 if s2_df is None else len(s2_df)
    n_s3 = 0 if s3_df is None else len(s3_df)
    return n_s2 + n_s3


def iter_pool_names(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    name_col: str,
    normalize: Callable[[str], str],
) -> Iterator[str]:
    """Yield every normalized candidate name in pool order, without materializing a list.

    Used to stream the fit corpus so that a 10.320.219-element list of strings is never
    allocated. Missing names and missing columns yield ``""`` (an empty document), which
    the vectorizer skips — matching the previous behaviour.
    """
    for df in (s2_df, s3_df):
        if df is None or len(df) == 0:
            continue
        if name_col in df.columns:
            for raw in df[name_col].fillna("").astype(str).tolist():
                yield normalize(raw)
        else:
            for _ in range(len(df)):
                yield ""


def iter_pool_blocks(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    id_col: str,
    name_col: str,
    block_size: int,
    normalize: Callable[[str], str],
) -> Iterator[Tuple[List[str], List[str], str]]:
    """Yield ``(candidate_ids, normalized_names, source_label)`` for bounded pool blocks.

    The pool is traversed in order — all of S2, then all of S3 — so a candidate's global
    position is derivable from the running offset, and no full-length ID or name list is
    ever built. Each yielded list holds at most ``block_size`` strings.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    for df, source_label in ((s2_df, "S2"), (s3_df, "S3")):
        if df is None or len(df) == 0:
            continue
        n_rows = len(df)
        id_series = df[id_col]
        has_name = name_col in df.columns
        name_series = df[name_col].fillna("").astype(str) if has_name else None

        for start in range(0, n_rows, block_size):
            stop = min(start + block_size, n_rows)
            block_ids = id_series.iloc[start:stop].astype(str).tolist()
            if name_series is not None:
                block_names = [
                    normalize(raw) for raw in name_series.iloc[start:stop].tolist()
                ]
            else:
                block_names = [""] * (stop - start)
            yield block_ids, block_names, source_label
