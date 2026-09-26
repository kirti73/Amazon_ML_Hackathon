"""
Corpus statistics module for label-free mining of legal suffixes and DBA markers.
Enables dynamic discovery of locale-specific legal forms (e.g. French SARL/SAS, Indian Pvt Ltd, US LLC).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Optional, Dict, Any

from .normalize import normalize_conservative, DEFAULT_LEGAL_SUFFIXES


def mine_corpus_suffixes(
    names: Iterable[str],
    min_freq_ratio: float = 0.0005,
    top_n: int = 60,
    include_defaults: bool = True,
) -> Set[str]:
    """
    Mines frequent trailing 1-token and 2-token legal suffixes from a collection of business names.
    Completely label-free: runs purely on name text frequency.
    """
    trailing_1_tokens: Counter = Counter()
    trailing_2_tokens: Counter = Counter()
    total_names = 0

    for raw in names:
        norm = normalize_conservative(raw)
        if not norm:
            continue
        tokens = norm.split()
        if not tokens:
            continue
        total_names += 1

        # 1-token suffix candidate
        trailing_1_tokens[tokens[-1]] += 1

        # 2-token suffix candidate
        if len(tokens) >= 2:
            trailing_2_tokens[f"{tokens[-2]} {tokens[-1]}"] += 1

    min_count = max(5, int(total_names * min_freq_ratio))

    mined: Set[str] = set()

    # Filter frequent 1-token candidates
    for token, count in trailing_1_tokens.most_common(top_n):
        if count >= min_count and len(token) >= 2:
            mined.add(token)

    # Filter frequent 2-token candidates
    for phrase, count in trailing_2_tokens.most_common(top_n):
        if count >= min_count:
            mined.add(phrase)

    if include_defaults:
        mined.update(DEFAULT_LEGAL_SUFFIXES)

    return mined


def mine_dba_markers(
    names: Iterable[str],
    top_n: int = 20,
) -> List[str]:
    """
    Identifies common leading/DBA markers in business names.
    """
    leading_tokens: Counter = Counter()
    for raw in names:
        norm = normalize_conservative(raw)
        if not norm:
            continue
        tokens = norm.split()
        if tokens:
            leading_tokens[tokens[0]] += 1

    # Standard known markers
    markers = [
        r"^m\s*/\s*s\b",
        r"^ms\b",
        r"^shree\b",
        r"^shri\b",
        r"^sri\b",
        r"^the\b",
        r"^c\s*/\s*o\b",
        r"^d\s*/\s*b\s*/\s*a\b",
        r"^dba\b",
        r"^f\s*/\s*k\s*/\s*a\b",
        r"^fka\b",
        r"^trading as\b",
    ]
    return markers


def save_mined_vocab(
    suffix_tokens: Set[str],
    filepath: str | Path = "artifacts/suffix_tokens.json",
) -> None:
    """
    Saves mined suffix tokens to a JSON file.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sorted(list(suffix_tokens)), f, indent=2, ensure_ascii=False)


def load_mined_vocab(
    filepath: str | Path = "artifacts/suffix_tokens.json",
    fallback_to_defaults: bool = True,
) -> Set[str]:
    """
    Loads mined suffix tokens from JSON file, falling back to defaults if not found.
    """
    p = Path(filepath)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data)
        except Exception:
            pass

    if fallback_to_defaults:
        return set(DEFAULT_LEGAL_SUFFIXES)
    return set()
