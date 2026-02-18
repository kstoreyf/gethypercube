"""Random construction of valid SLHD designs."""

import numpy as np


def random_slhd(t: int, m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random valid SLHD design matrix.

    The design is an integer array of shape (n, k) where n = m * t.
    Each column is a permutation of 1..n (full LHD property).
    Within slice s (rows s*m .. (s+1)*m - 1), column j contains a permutation
    of {s*m+1, ..., (s+1)*m} (per-slice LHD property).

    Parameters
    ----------
    t : int
        Number of slices.
    m : int
        Points per slice.
    k : int
        Number of dimensions.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    D : np.ndarray, shape (n, k), dtype int
        Integer design matrix (1-indexed values).
    """
    n = m * t
    D = np.empty((n, k), dtype=np.int64)

    for s in range(t):
        row_start = s * m
        row_end = row_start + m
        # The m values assigned to slice s in any column are {s*m+1, ..., (s+1)*m}
        base = np.arange(s * m + 1, (s + 1) * m + 1, dtype=np.int64)
        for j in range(k):
            D[row_start:row_end, j] = rng.permutation(base)

    return D
