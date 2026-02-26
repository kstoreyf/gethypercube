"""Tests for nested LHD validation."""

import numpy as np
import pytest
from gethypercube.nested_lhd import (
    validate_m_layers,
    check_valid_lhd,
    check_nested,
    validate_result,
)


class TestValidateMLayers:
    def test_valid_two_layers(self):
        validate_m_layers([2, 4])
        validate_m_layers([3, 5, 9])
        validate_m_layers([2, 3, 5, 9, 17])

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least 2 entries"):
            validate_m_layers([5])
        with pytest.raises(ValueError, match="at least 2 entries"):
            validate_m_layers([])

    def test_not_strictly_increasing_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_m_layers([4, 4, 8])
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_m_layers([5, 3, 9])

    def test_divisibility_violation_raises(self):
        with pytest.raises(ValueError, match="divisibility"):
            validate_m_layers([2, 4, 8])  # (8-1)/(4-1) = 7/3
        with pytest.raises(ValueError, match="divisibility"):
            validate_m_layers([3, 10, 30])

    def test_layer_size_less_than_two_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            validate_m_layers([1, 5])


class TestCheckValidLhd:
    def test_two_points(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert check_valid_lhd(X) is True
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert check_valid_lhd(X) is True

    def test_three_points(self):
        X = np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.5]]).T
        assert check_valid_lhd(X) is True

    def test_invalid_duplicate_level(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert check_valid_lhd(X) is False

    def test_invalid_wrong_levels(self):
        X = np.array([[0.0, 0.5], [0.5, 1.0]])
        assert check_valid_lhd(X) is False  # levels should be 0, 1 for n=2


class TestCheckNested:
    def test_subset_passes(self):
        inner = np.array([[0.0, 0.0], [1.0, 1.0]])
        outer = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert check_nested(inner, outer) is True

    def test_not_subset_fails(self):
        inner = np.array([[0.0, 0.0], [1.0, 1.0]])
        outer = np.array([[0.0, 0.5], [0.5, 0.0], [1.0, 1.0]])
        assert check_nested(inner, outer) is False

    def test_same_matrix_passes(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert check_nested(X, X) is True


class TestValidateResult:
    def test_valid_layers_pass(self):
        layers = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ]
        validate_result(layers, [2, 3], 2)

    def test_shape_mismatch_raises(self):
        # Layer 0 valid LHD; layer 1 has wrong number of rows
        layers = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.zeros((4, 2)),  # expected (3, 2)
        ]
        with pytest.raises(AssertionError, match="shape mismatch"):
            validate_result(layers, [2, 3], 2)

    def test_invalid_lhd_raises(self):
        layers = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, 0.5], [1.0, 1.0]]),  # col0 not perm
        ]
        with pytest.raises(AssertionError, match="not a valid LHD"):
            validate_result(layers, [2, 3], 2)
