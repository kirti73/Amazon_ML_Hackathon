"""
Text normalization module for business entity resolution.
Implements conservative and aggressive multilingual-safe normalization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Set, Optional, List
import pandas as pd
import numpy as np


# Common default legal suffix phrases (multi-token and single-token)
DEFAULT_LEGAL_SUFFIXES: Set[str] = {
    # English / US / International
    "inc", "incorporated", "llc", "l l c", "corp", "corporation", "co", "company",
    "ltd", "limited", "pvt ltd", "private limited", "pvt", "lp", "l p", "llp", "l l p",
    "pllc", "p l l c", "pc", "p c", "pa", "p a", "services", "enterprises", "associates",
    "partners", "group", "holdings", "trust", "society",
    # French (Unseen test distribution)
    "sarl", "s a r l", "sas", "s a s", "sasu", "s a s u", "eurl", "e u r l",
    "sci", "s c i", "ei", "e i", "sa", "s a", "snc", "s n c", "gme",
    # German / European
    "gmbh", "g m b h", "ag", "a g", "bv", "b v", "sl", "s l",
}

# Common leading prefixes and DBA markers (matched on conservative normalized text)
DEFAULT_LEADING_MARKERS: List[str] = [
    r"^m\s+s\b",
    r"^m\s*/\s*s\b",
    r"^ms\b",
    r"^shree\b",
    r"^shri\b",
    r"^sri\b",
    r"^the\b",
    r"^c\s+o\b",
    r"^c\s*/\s*o\b",
    r"^d\s+b\s+a\b",
    r"^d\s*/\s*b\s*/\s*a\b",
    r"^dba\b",
    r"^f\s+k\s+a\b",
    r"^f\s*/\s*k\s*/\s*a\b",
    r"^fka\b",
    r"^t\s+a\b",
    r"^t\s*/\s*a\b",
    r"^trading as\b",
    r"^\*\*\*+",
    r"^#+",
    r"^-+",
]


def strip_combining_accents(text: str) -> str:
    """
    Decomposes characters and strips combining diacritical marks in the Latin/combining block
    (U+0300 to U+036F), e.g. 'é' -> 'e', 'ö' -> 'o', 'ú' -> 'u'.
    Guarantees that non-Latin scripts such as Devanagari (Hindi) vowel signs/matras
    are NOT corrupted.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    filtered = "".join(
        c for c in decomposed
        if not ("\u0300" <= c <= "\u036f")
    )
    return unicodedata.normalize("NFKC", filtered)


def normalize_conservative(text: Optional[str]) -> str:
    """
    Conservative normalization applied to all records for blocking and base features:
    1. Null/empty check -> returns ""
    2. Unicode NFKC normalization
    3. Full Unicode casefolding
    4. Accent folding (preserves non-Latin scripts like Devanagari)
    5. Replaces punctuation and symbols with spaces, preserving all Unicode Letters (L),
       Marks/matras (M), and Numbers (N).
    6. Collapses multiple whitespace and strips.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Full casefolding
    text = text.casefold()

    # Accent stripping (safe for Devanagari & Latin)
    text = strip_combining_accents(text)

    # Replace punctuation and non-alphanumeric characters with spaces
    # Preserve all Unicode Letters (L*), Marks/combining characters/matras (M*), Numbers (N*)
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith(("L", "M", "N")):
            chars.append(c)
        else:
            chars.append(" ")
    text = "".join(chars)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_aggressive(
    text: Optional[str],
    suffix_tokens: Optional[Set[str]] = None,
    leading_markers: Optional[List[str]] = None,
) -> str:
    """
    Aggressive normalization for business names:
    1. Applies conservative normalization.
    2. Strips leading DBA/honorific/trade markers ('m/s', 'sri', 'the', etc.).
    3. Iteratively strips corpus-derived or standard legal suffixes from the end.
    """
    norm = normalize_conservative(text)
    if not norm:
        return ""

    # Strip leading markers
    markers = leading_markers if leading_markers is not None else DEFAULT_LEADING_MARKERS
    for marker_pattern in markers:
        norm = re.sub(marker_pattern, " ", norm, flags=re.UNICODE).strip()

    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return ""

    suffixes = suffix_tokens if suffix_tokens is not None else DEFAULT_LEGAL_SUFFIXES

    # Sort suffixes by token length descending (e.g. 'private limited' before 'limited')
    sorted_suffixes = sorted(suffixes, key=lambda s: len(s.split()), reverse=True)

    changed = True
    iterations = 0
    while changed and iterations < 5:
        changed = False
        iterations += 1
        tokens = norm.split()
        if not tokens:
            break

        # Check multi-token and single-token suffixes at the end
        for suffix in sorted_suffixes:
            s_tokens = suffix.split()
            s_len = len(s_tokens)
            if len(tokens) > s_len:  # Do not strip if it would leave the name empty!
                if tokens[-s_len:] == s_tokens:
                    tokens = tokens[:-s_len]
                    norm = " ".join(tokens).strip()
                    changed = True
                    break

    return norm


def normalize_country(country: Optional[str]) -> str:
    """
    Standardizes country string: uppercase, stripped, empty if missing.
    """
    if country is None:
        return ""
    if not isinstance(country, str):
        if pd.isna(country):
            return ""
        country = str(country)
    return country.strip().upper()


def preprocess_record(
    entity_id: str,
    raw_name: Optional[str],
    raw_address: Optional[str],
    raw_country: Optional[str],
    suffix_tokens: Optional[Set[str]] = None,
) -> dict:
    """
    Normalizes a single entity record.
    """
    # Determine source from entity_id prefix
    source = entity_id.split("-")[0] if "-" in str(entity_id) else "UNKNOWN"

    r_name = "" if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)) else str(raw_name)
    r_addr = "" if raw_address is None or (isinstance(raw_address, float) and np.isnan(raw_address)) else str(raw_address)
    r_ctry = "" if raw_country is None or (isinstance(raw_country, float) and np.isnan(raw_country)) else str(raw_country)

    return {
        "entity_id": str(entity_id),
        "source": source,
        "raw_name": r_name,
        "raw_address": r_addr,
        "raw_country": r_ctry,
        "norm_name_cons": normalize_conservative(r_name),
        "norm_name_aggr": normalize_aggressive(r_name, suffix_tokens=suffix_tokens),
        "norm_address_cons": normalize_conservative(r_addr),
        "country_norm": normalize_country(r_ctry),
    }


def preprocess_records_df(
    df: pd.DataFrame,
    suffix_tokens: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Vectorized preprocessing of an entity DataFrame to produce canonical records.parquet schema:
    - entity_id (string)
    - source (string)
    - raw_name (string)
    - raw_address (string)
    - raw_country (string)
    - norm_name_cons (string)
    - norm_name_aggr (string)
    - norm_address_cons (string)
    - country_norm (string)
    """
    assert "entity_id" in df.columns, "DataFrame must contain 'entity_id' column"

    res = pd.DataFrame()
    res["entity_id"] = df["entity_id"].astype(str)

    # Infer source
    if "source" in df.columns:
        res["source"] = df["source"].astype(str)
    else:
        res["source"] = res["entity_id"].apply(lambda x: x.split("-")[0] if "-" in str(x) else "UNKNOWN")

    # Name column
    name_col = "name" if "name" in df.columns else ("raw_name" if "raw_name" in df.columns else None)
    if name_col:
        res["raw_name"] = df[name_col].fillna("").astype(str)
    else:
        res["raw_name"] = ""

    # Address column
    addr_col = "address" if "address" in df.columns else ("raw_address" if "raw_address" in df.columns else None)
    if addr_col:
        res["raw_address"] = df[addr_col].fillna("").astype(str)
    else:
        res["raw_address"] = ""

    # Country column
    ctry_col = "country" if "country" in df.columns else ("raw_country" if "raw_country" in df.columns else None)
    if ctry_col:
        res["raw_country"] = df[ctry_col].fillna("").astype(str)
    else:
        res["raw_country"] = ""

    # Apply normalizations
    res["norm_name_cons"] = res["raw_name"].apply(normalize_conservative)
    res["norm_name_aggr"] = res["raw_name"].apply(
        lambda x: normalize_aggressive(x, suffix_tokens=suffix_tokens)
    )
    res["norm_address_cons"] = res["raw_address"].apply(normalize_conservative)
    res["country_norm"] = res["raw_country"].apply(normalize_country)

    return res
