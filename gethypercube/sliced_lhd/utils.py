"""Utility helpers for LHD / SLHD."""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest


def lhs_degree(X: np.ndarray) -> float:
    """
    Compute the LHS degree D(S) — fractional closeness to an LHS (Boschini et al. 2025).

    For N samples in P dimensions with values in [0, 1)^P, bins each dimension into N
    disjoint intervals [l/N, (l+1)/N). D = (fraction of bins with at least one sample).

    - D(S) = 1 iff S is a perfect LHS (exactly one sample per bin per dimension).
    - 0 < D(S) <= 1; higher values indicate closer to LHS distribution.

    Reference: "LHS in LHS" expansion strategy, https://arxiv.org/abs/2509.00159

    Parameters
    ----------
    X : np.ndarray, shape (N, P)
        Design points in [0, 1)^P (e.g. standardized LHD).

    Returns
    -------
    float
        LHS degree in (0, 1].
    """
    N, P = X.shape
    bins = np.floor(X * N).astype(np.int64)
    bins = np.clip(bins, 0, N - 1)
    total = 0.0
    for j in range(P):
        total += len(np.unique(bins[:, j]))
    return total / (N * P)


def ks_test_uniform(X: np.ndarray) -> list[tuple[float, float]]:
    """
    KS test each column against U(0, 1).

    Tests whether one-dimensional projections of the design follow the uniform
    distribution, which is a property of a good LHD (one sample per stratum).

    Parameters
    ----------
    X : np.ndarray, shape (N, P)
        Design points in [0, 1)^P.

    Returns
    -------
    list of (statistic, pvalue)
        Per-dimension results. Lower statistic and higher pvalue indicate
        better agreement with U(0, 1).
    """
    results = []
    for j in range(X.shape[1]):
        stat, pval = kstest(X[:, j], "uniform", args=(0, 1))
        results.append((stat, pval))
    return results


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
    Check whether D (shape n x k, n = m*t) satisfies the SLHD constraint
    (Ba et al. 2015 / Qian 2012):
    - Full design is a valid LHD (each column a permutation of 1..n).
    - Each slice s: for each column, ceil(D[i,j]/t) for i in slice = {1,...,m}.
    I.e. each slice has one value from each bin {1..t}, {t+1..2t}, ...,
    {(m-1)t+1..mt}.
    """
    n = m * t
    k = D.shape[1]

    if not is_valid_lhd(D, n):
        return False

    expected_levels = set(range(1, m + 1))
    for s in range(t):
        rows = slice(s * m, (s + 1) * m)
        for j in range(k):
            levels = set(np.ceil(D[rows, j].astype(float) / t).astype(int))
            if levels != expected_levels:
                return False

    return True

