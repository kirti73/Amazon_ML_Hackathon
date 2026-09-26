"""
Pairwise name feature computation for business entity resolution.
Implements lexical, n-gram, token-sort/set, and containment features.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def extract_char_ngrams(text: str, n_low: int = 3, n_high: int = 4) -> Counter:
    """
    Extracts character 3-4 gram frequency counter from normalized text.
    Pads string with boundary markers for position awareness.
    """
    if not text:
        return Counter()
    padded = f"^{text}$"
    counts = Counter()
    for n in range(n_low, n_high + 1):
        if len(padded) >= n:
            for i in range(len(padded) - n + 1):
                counts[padded[i:i+n]] += 1
    return counts


def char_ngram_cosine_similarity(text1: str, text2: str) -> float:
    """
    Computes character 3-4 gram cosine similarity between two text strings.
    Returns 0.0 if either string is empty or has no n-grams.
    """
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0

    c1 = extract_char_ngrams(text1)
    c2 = extract_char_ngrams(text2)
    if not c1 or not c2:
        return 0.0

    # Dot product
    intersection = set(c1.keys()) & set(c2.keys())
    if not intersection:
        return 0.0

    dot = sum(c1[k] * c2[k] for k in intersection)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_name_feature_dict(
    name_s1_cons: str,
    name_cand_cons: str,
    name_s1_aggr: str = "",
    name_cand_aggr: str = "",
) -> Dict[str, float]:
    """
    Computes all name similarity features for a single candidate pair.
    """
    n1 = name_s1_cons or ""
    n2 = name_cand_cons or ""
    a1 = name_s1_aggr or n1
    a2 = name_cand_aggr or n2

    # Exact match flags
    name_exact_cons = 1.0 if (n1 and n2 and n1 == n2) else 0.0
    name_exact_aggr = 1.0 if (a1 and a2 and a1 == a2) else 0.0

    # Character cosine similarity
    name_char_cos = char_ngram_cosine_similarity(n1, n2)

    # RapidFuzz token similarities (normalized to [0, 1])
    if n1 and n2:
        name_token_sort_ratio = float(fuzz.token_sort_ratio(n1, n2)) / 100.0
        name_token_set_ratio = float(fuzz.token_set_ratio(n1, n2)) / 100.0
    else:
        name_token_sort_ratio = 0.0
        name_token_set_ratio = 0.0

    # Token containment features
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    if tokens1 and tokens2:
        intersection = tokens1 & tokens2
        name_containment_s1_in_cand = float(len(intersection)) / float(len(tokens1))
        name_containment_cand_in_s1 = float(len(intersection)) / float(len(tokens2))
    else:
        name_containment_s1_in_cand = 0.0
        name_containment_cand_in_s1 = 0.0

    # Length and token differences
    len1 = len(n1)
    len2 = len(n2)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = float(min(len1, len2)) / float(max(len1, len2)) if max(len1, len2) > 0 else 1.0

    tok_len1 = len(n1.split())
    tok_len2 = len(n2.split())
    name_token_count_diff = float(abs(tok_len1 - tok_len2))

    # First token match
    t1_first = n1.split()[0] if tok_len1 > 0 else ""
    t2_first = n2.split()[0] if tok_len2 > 0 else ""
    name_first_token_match = 1.0 if (t1_first and t2_first and t1_first == t2_first) else 0.0

    return {
        "name_exact_cons": name_exact_cons,
        "name_exact_aggr": name_exact_aggr,
        "name_char_cos": name_char_cos,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_containment_s1_in_cand": name_containment_s1_in_cand,
        "name_containment_cand_in_s1": name_containment_cand_in_s1,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_token_count_diff": name_token_count_diff,
        "name_first_token_match": name_first_token_match,
    }


def compute_name_features_batch(
    s1_names_cons: List[str] | np.ndarray,
    cand_names_cons: List[str] | np.ndarray,
    s1_names_aggr: Optional[List[str] | np.ndarray] = None,
    cand_names_aggr: Optional[List[str] | np.ndarray] = None,
) -> pd.DataFrame:
    """
    Vectorized / chunk-optimized computation of name features across arrays of pairs.
    """
    n_pairs = len(s1_names_cons)
    if s1_names_aggr is None:
        s1_names_aggr = s1_names_cons
    if cand_names_aggr is None:
        cand_names_aggr = cand_names_cons

    data: Dict[str, List[float]] = {
        "name_exact_cons": [0.0] * n_pairs,
        "name_exact_aggr": [0.0] * n_pairs,
        "name_char_cos": [0.0] * n_pairs,
        "name_token_sort_ratio": [0.0] * n_pairs,
        "name_token_set_ratio": [0.0] * n_pairs,
        "name_containment_s1_in_cand": [0.0] * n_pairs,
        "name_containment_cand_in_s1": [0.0] * n_pairs,
        "name_len_diff": [0.0] * n_pairs,
        "name_len_ratio": [1.0] * n_pairs,
        "name_token_count_diff": [0.0] * n_pairs,
        "name_first_token_match": [0.0] * n_pairs,
    }

    for i in range(n_pairs):
        feat = compute_name_feature_dict(
            s1_names_cons[i],
            cand_names_cons[i],
            s1_names_aggr[i],
            cand_names_aggr[i],
        )
        for k, v in feat.items():
            data[k][i] = v

    return pd.DataFrame(data)
