"""Utility helpers."""

import numpy as np


def is_valid_lhd(D: np.ndarray, n: int) -> bool:
    """
    Check whether integer matrix D (shape n x k) is a valid LHD.
    Each column must be a permutation of 1..n.
    """
    expected = set(range(1, n + 1))
    for j in range(D.shape[1]):
        if set(D[:, j]) != expected:
            return False
    return True


def is_valid_slhd(D: np.ndarray, t: int, m: int) -> bool:
    """
    Check whether D (shape n x k, n = m*t) satisfies the SLHD constraint:
    - Full design is a valid LHD.
    - Each slice is a valid LHD of size m.
    """
    n = m * t
    k = D.shape[1]

    if not is_valid_lhd(D, n):
        return False

    for s in range(t):
        rows = slice(s * m, (s + 1) * m)
        expected = set(range(s * m + 1, (s + 1) * m + 1))
        for j in range(k):
            if set(D[rows, j]) != expected:
                return False

    return True
