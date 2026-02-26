"""Nested Latin hypercube designs: Qian (2009) and Rennen et al. (2010)."""

from .validate import (
    validate_m_layers_qian,
    validate_m_layers_rennen,
    validate_m_layers,
    check_valid_lhd,
    check_nested,
    validate_result,
)
from .qian import nested_lhd
from .multi_layer import nested_maximin_lhd, build_nested_lhd, extend_to_layer
from .two_layer import two_layer_nested_lhd
from .utils import (
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

