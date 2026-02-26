"""Tests for validation utilities."""

import numpy as np
import pytest
from gethypercube.sliced_lhd import is_valid_lhd, is_valid_slhd


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
        # t=2, m=2, k=2: each slice has one value from {1,2} and one from {3,4}
        # (paper's def: ceil(v/t) gives {1,2} per slice per column)
        D = np.array([
            [1, 3],
            [3, 1],
            [2, 4],
            [4, 2],
        ])
        assert is_valid_slhd(D, t=2, m=2)

    def test_invalid_slice_stratum(self):
        # slice 0 col 0 has [1,2] - both in level 1 (ceil 1,1) -> only {1}, not {1,2}
        D = np.array([
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ])
        assert not is_valid_slhd(D, t=2, m=2)

    def test_valid_single_slice(self):
        D = np.array([[2, 1], [1, 3], [3, 2]])
        assert is_valid_slhd(D, t=1, m=3)

    def test_invalid_not_lhd_globally(self):
        D = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [3, 4],
        ])
        assert not is_valid_slhd(D, t=2, m=2)
