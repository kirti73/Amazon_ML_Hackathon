"""
Pairwise address feature computation for business entity resolution.
Implements lexical, numeric token, token-sort, containment, and digit conflict features.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .name_features import char_ngram_cosine_similarity


DIGIT_REGEX = re.compile(r"\b\d+\b")


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extracts isolated digit sequences (e.g. house numbers, pincodes, plot numbers).
    """
    if not text:
        return []
    return DIGIT_REGEX.findall(text)


def compute_address_feature_dict(
    addr_s1_cons: str,
    addr_cand_cons: str,
) -> Dict[str, float]:
    """
    Computes address similarity features for a candidate pair.
    Adheres strictly to the missingness guardrail: missing addresses produce NaN
    for similarity metrics rather than 0.0.
    """
    a1 = addr_s1_cons or ""
    a2 = addr_cand_cons or ""

    has_a1 = bool(a1.strip())
    has_a2 = bool(a2.strip())

    if not has_a1 or not has_a2:
        # One or both addresses are missing: produce NaN for similarities, not 0.0!
        return {
            "addr_exact_cons": 1.0 if (has_a1 and has_a2 and a1 == a2) else 0.0,
            "addr_char_cos": np.nan,
            "addr_token_sort_ratio": np.nan,
            "addr_containment_s1_in_cand": np.nan,
            "addr_containment_cand_in_s1": np.nan,
            "addr_len_diff": np.nan,
            "addr_num_jaccard": np.nan,
            "addr_num_conflict": 0.0,
            "addr_first_num_match": 0.0,
        }

    # Both addresses are present:
    addr_exact_cons = 1.0 if (a1 == a2) else 0.0
    addr_char_cos = char_ngram_cosine_similarity(a1, a2)
    addr_token_sort_ratio = float(fuzz.token_sort_ratio(a1, a2)) / 100.0

    tokens1 = set(a1.split())
    tokens2 = set(a2.split())

    if tokens1 and tokens2:
        intersection = tokens1 & tokens2
        addr_containment_s1_in_cand = float(len(intersection)) / float(len(tokens1))
        addr_containment_cand_in_s1 = float(len(intersection)) / float(len(tokens2))
    else:
        addr_containment_s1_in_cand = 0.0
        addr_containment_cand_in_s1 = 0.0

    addr_len_diff = float(abs(len(a1) - len(a2)))

    # Extract numeric tokens
    nums1_list = extract_numeric_tokens(a1)
    nums2_list = extract_numeric_tokens(a2)
    nums1 = set(nums1_list)
    nums2 = set(nums2_list)

    if nums1 or nums2:
        if nums1 and nums2:
            num_inter = nums1 & nums2
            num_union = nums1 | nums2
            addr_num_jaccard = float(len(num_inter)) / float(len(num_union))
            # Conflict occurs if both have digits but share none
            addr_num_conflict = 1.0 if len(num_inter) == 0 else 0.0
            # First numeric token match
            addr_first_num_match = 1.0 if (nums1_list[0] == nums2_list[0]) else 0.0
        else:
            # One has digits, other doesn't
            addr_num_jaccard = 0.0
            addr_num_conflict = 0.0
            addr_first_num_match = 0.0
    else:
        # Neither address contains numeric tokens -> NaN per specification
        addr_num_jaccard = np.nan
        addr_num_conflict = 0.0
        addr_first_num_match = 0.0

    return {
        "addr_exact_cons": addr_exact_cons,
        "addr_char_cos": addr_char_cos,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_containment_s1_in_cand": addr_containment_s1_in_cand,
        "addr_containment_cand_in_s1": addr_containment_cand_in_s1,
        "addr_len_diff": addr_len_diff,
        "addr_num_jaccard": addr_num_jaccard,
        "addr_num_conflict": addr_num_conflict,
        "addr_first_num_match": addr_first_num_match,
    }


def compute_address_features_batch(
    s1_addrs_cons: List[str] | np.ndarray,
    cand_addrs_cons: List[str] | np.ndarray,
) -> pd.DataFrame:
    """
    Vectorized / batch computation of address features across arrays of pairs.
    """
    n_pairs = len(s1_addrs_cons)
    data: Dict[str, List[float]] = {
        "addr_exact_cons": [0.0] * n_pairs,
        "addr_char_cos": [np.nan] * n_pairs,
        "addr_token_sort_ratio": [np.nan] * n_pairs,
        "addr_containment_s1_in_cand": [np.nan] * n_pairs,
        "addr_containment_cand_in_s1": [np.nan] * n_pairs,
        "addr_len_diff": [np.nan] * n_pairs,
        "addr_num_jaccard": [np.nan] * n_pairs,
        "addr_num_conflict": [0.0] * n_pairs,
        "addr_first_num_match": [0.0] * n_pairs,
    }

    for i in range(n_pairs):
        feat = compute_address_feature_dict(
            s1_addrs_cons[i],
            cand_addrs_cons[i],
        )
        for k, v in feat.items():
            data[k][i] = v

    return pd.DataFrame(data)
