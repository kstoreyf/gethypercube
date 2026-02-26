"""Two-layer nested maximin LHD construction."""

from __future__ import annotations

import numpy as np

from .ese import two_layer_ese
from .utils import integer_to_continuous_stratum


def two_layer_nested_lhd(
    k: int,
    n_1: int,
    n_2: int,
    rng: np.random.Generator,
    n_restarts: int = 5,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a two-layer nested maximin LHD: X_1 ⊂ X_2, both valid LHDs.

    Stratum convention: (level + u) / n with u drawn once for the full design,
    inherited by inner layer as row subset. Strata [k/n, (k+1)/n) tile [0, 1).

    Parameters
    ----------
    k : int
        Number of dimensions.
    n_1, n_2 : int
        Layer sizes (n_1 < n_2, (n_2-1) divisible by (n_1-1)).
    rng : np.random.Generator
        Random generator.
    n_restarts : int
        Number of ESE restarts (each for GROUPRAND and POINTRAND).
    max_outer_iters : int
        Max outer ESE iterations.
    inner_iters_per_point : int
        Inner iters = this * n_2 * k.
    scramble : bool
        If True (default), u ~ Uniform(0,1) per cell. If False, u = 0.5.

    Returns
    -------
    X1 : np.ndarray, shape (n_1, k), float64 in [0, 1)
        Inner layer (first n_1 rows of X2).
    X2 : np.ndarray, shape (n_2, k), float64 in [0, 1)
        Outer layer.
    """
    X2_int, I1 = two_layer_ese(
        n_1, n_2, k, rng,
        n_restarts=n_restarts,
        max_outer_iters=max_outer_iters,
        inner_iters_per_point=inner_iters_per_point,
    )
    if scramble:
        u = rng.random((n_2, k), dtype=np.float64)
    else:
        u = np.full((n_2, k), 0.5, dtype=np.float64)
    X2 = integer_to_continuous_stratum(X2_int, n_2, u)
    X1 = X2[I1].copy()
    return X1, X2
