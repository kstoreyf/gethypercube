"""gethypercube: sliced and nested Latin hypercube designs."""

from __future__ import annotations

from .utils import scale_LH
from .sliced_lhd import sliced_lhd, compute_phi, lhs_degree, ks_test_uniform
from .nested_lhd import (
    nested_lhd,
    nested_maximin_lhd,
    build_nested_lhd,
    extend_to_layer,
    two_layer_nested_lhd,
    validate_m_layers_qian,
    validate_m_layers_rennen,
    validate_m_layers,
    check_valid_lhd,
    check_nested,
    validate_result,
    suggest_valid_layers_qian,
    suggest_valid_layers_rennen,
    suggest_valid_layers,
    m_layers_from_rennen,
    scaling_factor,
    integer_to_continuous,
    integer_to_continuous_qian,
    integer_to_continuous_midpoint,
    integer_to_continuous_stratum,
    scramble_layer_midpoint,
)

__all__ = [
    # Post-processing
    "scale_LH",
    # Sliced LHD
    "sliced_lhd",
    "compute_phi",
    "lhs_degree",
    "ks_test_uniform",
    # Nested LHD (Qian / Rennen)
    "nested_lhd",
    "nested_maximin_lhd",
    "build_nested_lhd",
    "extend_to_layer",
    "two_layer_nested_lhd",
    "validate_m_layers_qian",
    "validate_m_layers_rennen",
    "validate_m_layers",
    "check_valid_lhd",
    "check_nested",
    "validate_result",
    "suggest_valid_layers_qian",
    "suggest_valid_layers_rennen",
    "suggest_valid_layers",
    "m_layers_from_rennen",
    "scaling_factor",
    "integer_to_continuous",
    "integer_to_continuous_qian",
    "integer_to_continuous_midpoint",
    "integer_to_continuous_stratum",
    "scramble_layer_midpoint",
]

__version__ = "0.1.0"

