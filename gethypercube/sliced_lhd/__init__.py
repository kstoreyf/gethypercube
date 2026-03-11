"""Sliced Latin hypercube designs (SLHD) and utilities."""
from __future__ import annotations

from .design import sliced_lhd
from .objective import compute_phi
from .utils import lhs_degree, ks_test_uniform, is_valid_lhd, is_valid_slhd

__all__ = [
    "sliced_lhd",
    "compute_phi",
    "lhs_degree",
    "ks_test_uniform",
    "is_valid_lhd",
    "is_valid_slhd",
]
