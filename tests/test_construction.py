"""Tests for random SLHD construction."""

import numpy as np
import pytest
from slicedlhd.construction import random_slhd
from slicedlhd.utils import is_valid_lhd, is_valid_slhd


def rng(seed=0):
    return np.random.default_rng(seed)


class TestRandomSlhd:
    def test_shape(self):
        D = random_slhd(t=3, m=4, k=5, rng=rng())
        assert D.shape == (12, 5)

    def test_shape_single_slice(self):
        D = random_slhd(t=1, m=10, k=3, rng=rng())
        assert D.shape == (10, 3)

    def test_dtype_is_integer(self):
        D = random_slhd(t=2, m=5, k=3, rng=rng())
        assert np.issubdtype(D.dtype, np.integer)

    def test_full_lhd_property(self):
        """Each column must be a permutation of 1..n."""
        t, m, k = 3, 4, 6
        D = random_slhd(t=t, m=m, k=k, rng=rng())
        assert is_valid_lhd(D, t * m)

    def test_full_slhd_property(self):
        """Each slice must also be a valid LHD over its stratum."""
        t, m, k = 4, 3, 5
        D = random_slhd(t=t, m=m, k=k, rng=rng())
        assert is_valid_slhd(D, t, m)

    def test_single_slice_equals_lhd(self):
        """t=1 should produce a standard LHD: col is permutation of 1..m."""
        t, m, k = 1, 8, 4
        D = random_slhd(t=t, m=m, k=k, rng=rng())
        assert is_valid_lhd(D, m)

    def test_slice_values_in_correct_stratum(self):
        """Values in slice s, any column, must lie in {s*m+1 ... (s+1)*m}."""
        t, m, k = 3, 4, 3
        D = random_slhd(t=t, m=m, k=k, rng=rng())
        for s in range(t):
            lo, hi = s * m + 1, (s + 1) * m
            block = D[s * m: (s + 1) * m, :]
            assert block.min() >= lo
            assert block.max() <= hi

    def test_different_seeds_give_different_designs(self):
        t, m, k = 2, 5, 3
        D1 = random_slhd(t=t, m=m, k=k, rng=rng(0))
        D2 = random_slhd(t=t, m=m, k=k, rng=rng(1))
        assert not np.array_equal(D1, D2)

    def test_same_seed_gives_same_design(self):
        t, m, k = 2, 5, 3
        D1 = random_slhd(t=t, m=m, k=k, rng=rng(42))
        D2 = random_slhd(t=t, m=m, k=k, rng=rng(42))
        assert np.array_equal(D1, D2)

    @pytest.mark.parametrize("t,m,k", [(1, 2, 1), (2, 2, 2), (5, 10, 8)])
    def test_various_shapes_are_valid(self, t, m, k):
        D = random_slhd(t=t, m=m, k=k, rng=rng())
        assert is_valid_slhd(D, t, m)
