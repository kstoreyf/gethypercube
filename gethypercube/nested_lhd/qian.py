"""Qian (2009) nested Latin hypercube design — algebraic construction."""

from __future__ import annotations

import numpy as np

from .validate import validate_m_layers_qian, validate_result
from .utils import random_integer_lhd


def _random_lhd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Build random LHD of n points in k dimensions on integer levels 0..n-1."""
    return random_integer_lhd(n, k, rng)


def _expand_qian(
    X_inner: np.ndarray,
    n_large: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Expand inner design (integer 0..n_small-1) to outer (integer 0..n_large-1).
    n_large = c * n_small. First n_small rows of result are embedding of X_inner;
    result has shape (n_large, k).

    Level v (0-based) in column j maps to v*c + u_j with u_j ~ Uniform{0, ..., c-1}.
    Row order is preserved: inner layer is always rows 0..n_small-1, complement is n_small..n_large-1.
    """
    n_small = X_inner.shape[0]
    c = n_large // n_small
    X2 = np.zeros((n_large, k), dtype=np.int64)
    for j in range(k):
        u_j = rng.integers(0, c)
        X2[:n_small, j] = X_inner[:, j] * c + u_j
    for j in range(k):
        used = set(X2[:n_small, j].tolist())
        available = np.array([x for x in range(n_large) if x not in used])
        rng.shuffle(available)
        X2[n_small:, j] = available
    return X2


def _verify_qian_layer_indices(X_int: np.ndarray, m_layers: list[int], k: int) -> None:
    """
    Verify that for each n in m_layers, the first n rows of X_int form the n-layer
    design: in each column, floor(value / (n_L / n)) must be a permutation of 0..n-1.
    Raises AssertionError if the row-order invariant is broken.
    """
    n_L = X_int.shape[0]
    for n in m_layers:
        if n > n_L:
            continue
        c = n_L // n  # stride for this layer in the full grid
        for j in range(k):
            # First n rows: values should be in blocks [0..c-1], [c..2c-1], ..., [(n-1)c .. n*c-1]
            block = np.floor_divide(X_int[:n, j], c)
            block = np.clip(block, 0, n - 1)
            expected = np.sort(np.unique(block))
            if not np.array_equal(expected, np.arange(n)):
                raise AssertionError(
                    f"Qian row-order invariant broken: first n={n} rows in column {j} "
                    f"do not form a valid n-layer (block indices {np.sort(block)} != 0..{n-1})"
                )


def nested_lhd(
    k: int,
    m_layers: list[int] | int,
    seed: int | None = None,
    optimise: bool = False,
    scramble: bool = True,
) -> list[np.ndarray]:
    """
    Construct a multi-layer nested LHD using Qian (2009).

    Divisibility: n_{i+1} % n_i == 0 for all consecutive pairs.
    Zero-indexed stratum convention: integer levels {0, 1, ..., n-1}, convert to
    continuous via (level + u) / n with u ~ Uniform(0,1) drawn once for the full
    design and inherited by inner layers as row subsets. Strata [k/n, (k+1)/n)
    tile [0, 1) symmetrically; CDF scatters evenly around the U(0,1) diagonal.

    Parameters
    ----------
    k : int
        Number of dimensions. Must be >= 1.
    m_layers : list[int]
        Strictly increasing layer sizes; each n_{i+1} must be a multiple of n_i.
        Minimum 2 layers.
    seed : int or None
        Random seed for reproducibility.
    optimise : bool
        If True, run post-hoc complement-only ESE to improve space-filling.
        Default False (fast algebraic construction).
    scramble : bool
        If True (default), u ~ Uniform(0,1) per cell. If False, u = 0.5 (midpoint of stratum).

    Returns
    -------
    list of np.ndarray
        layers[i] is (m_layers[i], k) float64 in [0, 1); layers[0] ⊂ ... ⊂ layers[L-1].
    """
    if isinstance(m_layers, int):
        m_layers = [m_layers]
    rng = np.random.default_rng(seed)

    if len(m_layers) == 1:
        n = m_layers[0]
        X = _random_lhd(n, k, rng)
        if scramble:
            u = rng.random((n, k), dtype=np.float64)
        else:
            u = np.full((n, k), 0.5, dtype=np.float64)
        full = (X.astype(np.float64) + u) / n
        return [full]

    validate_m_layers_qian(m_layers)
    n_1 = m_layers[0]
    X = _random_lhd(n_1, k, rng)
    for i in range(1, len(m_layers)):
        n_large = m_layers[i]
        X = _expand_qian(X, n_large, k, rng)

    n_L = m_layers[-1]
    # Explicit layer membership: by construction, inner layer for size n is rows 0..n-1
    layer_indices = {n: list(range(n)) for n in m_layers}
    _verify_qian_layer_indices(X, m_layers, k)

    # u drawn once for full design; inner layers use same u at their row indices
    if scramble:
        u = rng.random((n_L, k), dtype=np.float64)
    else:
        u = np.full((n_L, k), 0.5, dtype=np.float64)
    full = (X.astype(np.float64) + u) / n_L

    layers = [full[layer_indices[n], :].copy() for n in m_layers]
    validate_result(layers, m_layers, k, convention="stratum", nesting_check="exact")
    return layers
