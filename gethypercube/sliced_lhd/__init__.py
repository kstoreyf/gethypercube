"""Sliced Latin hypercube designs (SLHD) and utilities."""

from __future__ import annotations

from .design import maximinSLHD, SLHDResult
from .utils import lhs_degree, ks_test_uniform, is_valid_lhd, is_valid_slhd

__all__ = [
    "maximinSLHD",
    "SLHDResult",
    "lhs_degree",
    "ks_test_uniform",
    "is_valid_lhd",
    "is_valid_slhd",
]

