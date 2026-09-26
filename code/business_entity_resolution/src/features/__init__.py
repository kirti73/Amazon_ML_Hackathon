"""
Feature engineering package for Amazon ML Challenge 2026: Business Entity Resolution.
Owned by Person 2 (Normalization & Feature Engineering).
"""

from .name_features import compute_name_feature_dict, compute_name_features_batch
from .address_features import compute_address_feature_dict, compute_address_features_batch
from .cross_features import compute_cross_feature_dict, compute_cross_features_batch
from .retrieval_features import pivot_retrieval_features
from .build import build_features

__all__ = [
    "compute_name_feature_dict",
    "compute_name_features_batch",
    "compute_address_feature_dict",
    "compute_address_features_batch",
    "compute_cross_feature_dict",
    "compute_cross_features_batch",
    "pivot_retrieval_features",
    "build_features",
]
