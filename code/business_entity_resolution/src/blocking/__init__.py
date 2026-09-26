"""
Blocking package — P1: Candidate Generation.

Public API
----------
from src.blocking import generate_candidates

generate_candidates(s1_df, s2_df, s3_df, cfg) -> pd.DataFrame
    Returns the canonical final candidate DataFrame with columns:
        pair_key, s1_id, candidate_id, candidate_source,
        route, rank, score
"""

from .candidate_generation import generate_candidates  # noqa: F401

__all__ = ["generate_candidates"]
