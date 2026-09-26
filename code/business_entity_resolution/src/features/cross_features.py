"""
Pairwise country, missingness, and cross-field feature computation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def compute_cross_feature_dict(
    s1_name_raw: str,
    cand_name_raw: str,
    s1_addr_raw: str,
    cand_addr_raw: str,
    s1_country_norm: str,
    cand_country_norm: str,
    name_char_cos: float = 0.0,
    addr_char_cos: float = np.nan,
) -> Dict[str, float]:
    """
    Computes country match, missingness flags, and cross-field interaction features.
    """
    c1 = (s1_country_norm or "").strip()
    c2 = (cand_country_norm or "").strip()

    # Country equality: 1 if equal and present, 0 if different, NaN if either is missing
    if c1 and c2:
        country_eq = 1.0 if c1 == c2 else 0.0
    else:
        country_eq = np.nan

    # Missingness indicators
    name_missing_cand = 1.0 if not (cand_name_raw or "").strip() else 0.0
    addr_missing_s1 = 1.0 if not (s1_addr_raw or "").strip() else 0.0
    addr_missing_cand = 1.0 if not (cand_addr_raw or "").strip() else 0.0
    addr_missing_either = 1.0 if (addr_missing_s1 > 0.5 or addr_missing_cand > 0.5) else 0.0

    # Cross-field similarity combinations
    n_sim = 0.0 if np.isnan(name_char_cos) else float(name_char_cos)

    if np.isnan(addr_char_cos):
        min_name_addr_sim = n_sim
        max_name_addr_sim = n_sim
        both_strong_match = 0.0
    else:
        a_sim = float(addr_char_cos)
        min_name_addr_sim = min(n_sim, a_sim)
        max_name_addr_sim = max(n_sim, a_sim)
        both_strong_match = 1.0 if (n_sim > 0.8 and a_sim > 0.8) else 0.0

    return {
        "country_eq": country_eq,
        "name_missing_cand": name_missing_cand,
        "addr_missing_s1": addr_missing_s1,
        "addr_missing_cand": addr_missing_cand,
        "addr_missing_either": addr_missing_either,
        "min_name_addr_sim": min_name_addr_sim,
        "max_name_addr_sim": max_name_addr_sim,
        "both_strong_match": both_strong_match,
    }


def compute_cross_features_batch(
    s1_names_raw: List[str] | np.ndarray,
    cand_names_raw: List[str] | np.ndarray,
    s1_addrs_raw: List[str] | np.ndarray,
    cand_addrs_raw: List[str] | np.ndarray,
    s1_countries_norm: List[str] | np.ndarray,
    cand_countries_norm: List[str] | np.ndarray,
    name_char_cos_arr: List[float] | np.ndarray,
    addr_char_cos_arr: List[float] | np.ndarray,
) -> pd.DataFrame:
    """
    Vectorized / batch computation of country, missingness, and cross-field features.
    """
    n_pairs = len(s1_names_raw)
    data: Dict[str, List[float]] = {
        "country_eq": [np.nan] * n_pairs,
        "name_missing_cand": [0.0] * n_pairs,
        "addr_missing_s1": [0.0] * n_pairs,
        "addr_missing_cand": [0.0] * n_pairs,
        "addr_missing_either": [0.0] * n_pairs,
        "min_name_addr_sim": [0.0] * n_pairs,
        "max_name_addr_sim": [0.0] * n_pairs,
        "both_strong_match": [0.0] * n_pairs,
    }

    for i in range(n_pairs):
        feat = compute_cross_feature_dict(
            s1_names_raw[i],
            cand_names_raw[i],
            s1_addrs_raw[i],
            cand_addrs_raw[i],
            s1_countries_norm[i],
            cand_countries_norm[i],
            name_char_cos_arr[i],
            addr_char_cos_arr[i],
        )
        for k, v in feat.items():
            data[k][i] = v

    return pd.DataFrame(data)
