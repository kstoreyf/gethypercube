"""Tests for two-layer nested LHD construction."""

import numpy as np
import pytest
from gethypercube.nested_lhd import (
    two_layer_nested_lhd,
    check_valid_lhd,
    check_nested,
    validate_m_layers,
)


def rng(seed=0):
    return np.random.default_rng(seed)


class TestTwoLayerNestedLhd:
    def test_output_shapes(self):
        X1, X2 = two_layer_nested_lhd(
            k=3, n_1=2, n_2=4, rng=rng(42),
            n_restarts=1, max_outer_iters=10,
        )
        assert X1.shape == (2, 3)
        assert X2.shape == (4, 3)

    def test_inner_is_valid_lhd(self):
        X1, X2 = two_layer_nested_lhd(
            k=2, n_1=3, n_2=5, rng=rng(1),
            n_restarts=1, max_outer_iters=15,
        )
        # Inner layer has n_1 rows but values are on the n_2 grid (first rows of X2)
        assert check_valid_lhd(X1, convention="stratum", n_full=X2.shape[0]) is True

    def test_outer_is_valid_lhd(self):
        X1, X2 = two_layer_nested_lhd(
            k=2, n_1=3, n_2=5, rng=rng(2),
            n_restarts=1, max_outer_iters=15,
        )
        assert check_valid_lhd(X2, convention="stratum") is True

    def test_nested(self):
        X1, X2 = two_layer_nested_lhd(
            k=2, n_1=3, n_2=5, rng=rng(3),
            n_restarts=1, max_outer_iters=15,
        )
        assert check_nested(X1, X2) is True

    def test_in_unit_interval(self):
        X1, X2 = two_layer_nested_lhd(
            k=4, n_1=2, n_2=4, rng=rng(4),
            n_restarts=1, max_outer_iters=10,
        )
        # Stratum convention: [0, 1)
        assert np.all(X1 >= 0) and np.all(X1 < 1)
        assert np.all(X2 >= 0) and np.all(X2 < 1)

    def test_reproducible_with_same_seed(self):
        X1a, X2a = two_layer_nested_lhd(
            k=2, n_1=3, n_2=5, rng=rng(99),
            n_restarts=1, max_outer_iters=20,
        )
        X1b, X2b = two_layer_nested_lhd(
            k=2, n_1=3, n_2=5, rng=rng(99),
            n_restarts=1, max_outer_iters=20,
        )
        np.testing.assert_array_almost_equal(X1a, X1b)
        np.testing.assert_array_almost_equal(X2a, X2b)
