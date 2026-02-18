"""Tests for validation utilities."""

import numpy as np
import pytest
from slicedlhd.utils import is_valid_lhd, is_valid_slhd


class TestIsValidLhd:
    def test_valid_lhd(self):
        D = np.array([[1, 3], [2, 1], [3, 2]])
        assert is_valid_lhd(D, n=3)

    def test_invalid_lhd_repeated_value(self):
        D = np.array([[1, 1], [2, 2], [2, 3]])  # col 0 has 2 twice
        assert not is_valid_lhd(D, n=3)

    def test_invalid_lhd_wrong_range(self):
        D = np.array([[0, 1], [1, 2], [2, 3]])  # col 0 has 0 instead of 3
        assert not is_valid_lhd(D, n=3)

    def test_single_column(self):
        D = np.array([[3], [1], [2]])
        assert is_valid_lhd(D, n=3)

    def test_single_row(self):
        D = np.array([[1, 1, 1]])
        assert is_valid_lhd(D, n=1)


class TestIsValidSlhd:
    def test_valid_slhd(self):
        # t=2, m=2, k=2: slice 0 gets {1,2}, slice 1 gets {3,4}
        D = np.array([
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ])
        assert is_valid_slhd(D, t=2, m=2)

    def test_invalid_slice_stratum(self):
        # slice 0 has values from slice 1's stratum
        D = np.array([
            [1, 3],  # col 1: 3 is in slice 1's stratum
            [2, 4],
            [3, 1],
            [4, 2],
        ])
        # full design col 1 = [3,4,1,2] is valid permutation of 1..4,
        # but slice 0 col 1 = [3,4] which are NOT in {1,2}
        assert not is_valid_slhd(D, t=2, m=2)

    def test_valid_single_slice(self):
        D = np.array([[2, 1], [1, 3], [3, 2]])
        assert is_valid_slhd(D, t=1, m=3)

    def test_invalid_not_lhd_globally(self):
        # Full column isn't a permutation of 1..4
        D = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [3, 4],  # duplicate 3 in col 0
        ])
        assert not is_valid_slhd(D, t=2, m=2)
