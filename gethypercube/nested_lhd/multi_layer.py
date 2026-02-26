"""Multi-layer nested LHD: recursive construction and public API."""

from __future__ import annotations

import numpy as np

from .validate import validate_m_layers_rennen, validate_result
from .ese import two_layer_ese, run_ese_extend
from .utils import integer_to_continuous_stratum, m_layers_from_rennen, random_integer_lhd


def _verify_rennen_layer_indices(
    X_int: np.ndarray, m_layers: list[int], k: int, n_L: int
) -> None:
    """
    Verify that for each n in m_layers, the first n rows of X_int form the n-layer
    design: Rennen embedding uses stride c = (n_L-1)//(n-1); in each column,
    value // c for the first n rows must be a permutation of 0..n-1.
    """
    for n in m_layers:
        if n > n_L or n < 2:
            continue
        c = (n_L - 1) // (n - 1)
        for j in range(k):
            block = np.floor_divide(X_int[:n, j], c)
            block = np.clip(block, 0, n - 1)
            expected = np.sort(np.unique(block))
            if not np.array_equal(expected, np.arange(n)):
                raise AssertionError(
                    f"Rennen row-order invariant broken: first n={n} rows in column {j} "
                    f"do not form a valid n-layer (block indices {np.sort(block)} != 0..{n-1})"
                )


def extend_to_layer_int(
    X_inner_int: np.ndarray,
    n_small: int,
    n_large: int,
    k: int,
    rng: np.random.Generator,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    n_restarts: int = 3,
) -> np.ndarray:
    """
    Extend inner integer design (0..n_small-1) to outer (0..n_large-1).
    Returns X_outer_int (n_large x k). Used to build full integer design before
    single (level+u)/n conversion.
    """
    c = (n_large - 1) // (n_small - 1)
    X2 = np.zeros((n_large, k), dtype=np.int64)
    X2[:n_small, :] = X_inner_int * c
    for j in range(k):
        used = set(X2[:n_small, j].tolist())
        available = np.array([x for x in range(n_large) if x not in used])
        rng.shuffle(available)
        X2[n_small:, j] = available
    X2 = run_ese_extend(
        X2,
        n_1=n_small,
        n_2=n_large,
        k=k,
        rng=rng,
        max_outer=max_outer_iters,
        inner_iters_per_point=inner_iters_per_point,
        n_restarts=n_restarts,
    )
    return X2


def extend_to_layer(
    X_inner: np.ndarray,
    n_large: int,
    k: int,
    rng: np.random.Generator,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    n_restarts: int = 3,
) -> np.ndarray:
    """
    Extend an existing inner-layer design to a larger nested layer.

    Given X_inner (n_small × k) float in [0, 1], build X_outer (n_large × k)
    such that the first n_small rows of X_outer are X_inner re-mapped to the
    n_large-grid, and X_outer is a valid LHD. Uses ESE to maximise space-filling.

    Parameters
    ----------
    X_inner : np.ndarray, shape (n_small, k), float in [0, 1]
        Valid LHD on n_small-grid (e.g. from a previous layer).
    n_large : int
        Target outer layer size. (n_large - 1) must be divisible by (n_small - 1).
    k : int
        Number of dimensions.
    rng : np.random.Generator
        Random generator.
    max_outer_iters, inner_iters_per_point, n_restarts
        ESE tuning.

    Returns
    -------
    X_outer : np.ndarray, shape (n_large, k), float64 in [0, 1]
    """
    n_small = X_inner.shape[0]
    c = (n_large - 1) // (n_small - 1)
    # Stratum: x = (level+u)/n => level = floor(x*n); recover integer from continuous
    X_inner_int = np.floor(X_inner * n_small).astype(np.int64)
    X_inner_int = np.clip(X_inner_int, 0, n_small - 1)
    X2 = np.zeros((n_large, k), dtype=np.int64)
    X2[:n_small, :] = X_inner_int * c

    # Step B: fill outer complement (integer grid 0..n_large-1)
    for j in range(k):
        used = set(X2[:n_small, j].tolist())
        available = np.array([x for x in range(n_large) if x not in used])
        rng.shuffle(available)
        X2[n_small:, j] = available

    # Step C: optimise
    X2 = run_ese_extend(
        X2,
        n_1=n_small,
        n_2=n_large,
        k=k,
        rng=rng,
        max_outer=max_outer_iters,
        inner_iters_per_point=inner_iters_per_point,
        n_restarts=n_restarts,
    )
    # Convert with (level+u)/n; u drawn here for this layer only (API for single extend)
    u = rng.random((n_large, k), dtype=np.float64)
    return integer_to_continuous_stratum(X2, n_large, u)


def _apply_scramble_midpoint(
    layers: list[np.ndarray], m_layers: list[int], rng: np.random.Generator
) -> None:
    """Deprecated: stratum convention uses single u at conversion; no separate scramble."""
    pass


def _stratum_convert(
    X_int: np.ndarray,
    n_L: int,
    k: int,
    rng: np.random.Generator,
    scramble: bool,
) -> np.ndarray:
    """Convert full integer design to continuous with (level+u)/n, u drawn once."""
    if scramble:
        u = rng.random((n_L, k), dtype=np.float64)
    else:
        u = np.full((n_L, k), 0.5, dtype=np.float64)
    return integer_to_continuous_stratum(X_int, n_L, u)


def nested_maximin_lhd(
    k: int,
    m_layers: list[int] | None = None,
    *,
    m_init: int | None = None,
    n_layers: int | None = None,
    ratio: int | None = None,
    n_restarts: int = 5,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    seed: int | None = None,
    scramble: bool = True,
) -> list[np.ndarray]:
    """
    Construct a multi-layer nested maximin LHD (Rennen et al. 2010).

    Divisibility: (n_{i+1}-1) % (n_i-1) == 0 for all consecutive pairs.
    Levels (stratum convention): integer levels 0..n-1, convert (level+u)/n with u drawn
    once for the full design; strata [k/n, (k+1)/n) tile [0, 1) symmetrically.

    Parameters
    ----------
    k : int
        Number of dimensions. Must be >= 1.
    m_layers : list[int], optional
        Strictly increasing layer sizes; each (n_{i+1}-1) divisible by (n_i-1).
        Minimum 2 layers. If not provided, m_init, n_layers, and ratio must be set.
    m_init : int, optional
        Smallest layer size. Used only when m_layers is not provided.
        Then m_layers = [m_init, (m_init-1)*ratio+1, ...] with n_layers elements.
    n_layers : int, optional
        Number of layers. Used only when m_layers is not provided.
    ratio : int, optional
        Integer ratio between consecutive (n-1) values. Used only when m_layers
        is not provided.
    n_restarts : int
        Number of ESE restarts per two-layer step. Default 5.
    max_outer_iters : int
        Max outer ESE iterations. Default 200.
    inner_iters_per_point : int
        Inner iters = this * n_2 * k. Default 100.
    seed : int or None
        Random seed.
    scramble : bool
        If True (default), randomly place values within each LHD cell (as in
        scipy.stats.qmc.LatinHypercube(scramble=True)). If False, keep grid-aligned
        values.

    Returns
    -------
    list of np.ndarray
        layers[i] is (m_layers[i], k) float64 in [0, 1), stratum convention.
        layers[0] ⊂ layers[1] ⊂ ... ⊂ layers[L-1].
    """
    if m_layers is None:
        if m_init is None or n_layers is None or ratio is None:
            raise ValueError(
                "Either m_layers or all of (m_init, n_layers, ratio) must be provided."
            )
        m_layers = m_layers_from_rennen(m_init, n_layers, ratio)
    return build_nested_lhd(
        k, m_layers,
        n_restarts=n_restarts,
        max_outer_iters=max_outer_iters,
        inner_iters_per_point=inner_iters_per_point,
        seed=seed,
        scramble=scramble,
    )


def build_nested_lhd(
    k: int,
    m_layers: list[int] | int,
    n_restarts: int = 5,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    seed: int | None = None,
    scramble: bool = True,
) -> list[np.ndarray]:
    """
    Construct a multi-layer nested Latin hypercube design.

    Parameters
    ----------
    k : int
        Number of dimensions. Must be >= 1.
    m_layers : list[int]
        Strictly increasing list of layer sizes [n_1, n_2, ..., n_L].
        Must satisfy: for all consecutive pairs (n_i, n_{i+1}),
        (n_{i+1} - 1) must be divisible by (n_i - 1).
        Minimum 2 layers required.
    n_restarts : int
        Number of independent ESE restarts per two-layer optimisation step.
        The best result across restarts is returned.
    max_outer_iters : int
        Maximum outer ESE iterations before termination.
    inner_iters_per_point : int
        Number of inner ESE iterations = inner_iters_per_point * n_2 * k.
    seed : int or None
        Random seed for reproducibility.

    scramble : bool
        If True (default), randomly place values within each LHD cell (as in
        scipy.stats.qmc.LatinHypercube(scramble=True)). If False, keep grid-aligned
        values.

    Returns
    -------
    list of np.ndarray
        layers[i] is an m_layers[i] × k float64 array with values in [0, 1).
        layers[0] ⊂ layers[1] ⊂ ... ⊂ layers[L-1] (row subset relationship).
        Stratum convention: one value per stratum [k/n_i, (k+1)/n_i), u drawn once.

    Raises
    ------
    ValueError
        If m_layers is not strictly increasing, has fewer than 2 elements,
        or violates the divisibility constraint.

    Examples
    --------
    >>> layers = build_nested_lhd(k=3, m_layers=[2, 4, 8, 16], seed=42)
    >>> len(layers)
    4
    >>> layers[0].shape, layers[-1].shape
    ((2, 3), (16, 3))
    """
    if isinstance(m_layers, int):
        m_layers = [m_layers]
    rng = np.random.default_rng(seed)

    if len(m_layers) == 1:
        n = m_layers[0]
        X_int = random_integer_lhd(n, k, rng)
        if scramble:
            u = rng.random((n, k), dtype=np.float64)
        else:
            u = np.full((n, k), 0.5, dtype=np.float64)
        full = integer_to_continuous_stratum(X_int, n, u)
        return [full]

    validate_m_layers_rennen(m_layers)
    n_1, n_2 = m_layers[0], m_layers[1]
    X_int, _ = two_layer_ese(
        n_1, n_2, k, rng,
        n_restarts=n_restarts,
        max_outer_iters=max_outer_iters,
        inner_iters_per_point=inner_iters_per_point,
    )
    for i in range(2, len(m_layers)):
        n_large = m_layers[i]
        X_int = extend_to_layer_int(
            X_int,
            m_layers[i - 1],
            n_large,
            k,
            rng,
            max_outer_iters=max_outer_iters,
            inner_iters_per_point=inner_iters_per_point,
            n_restarts=n_restarts,
        )

    n_L = m_layers[-1]
    # Explicit layer membership: by construction each extend places inner design in first rows
    layer_indices = {n: list(range(n)) for n in m_layers}
    _verify_rennen_layer_indices(X_int, m_layers, k, n_L)

    if scramble:
        u = rng.random((n_L, k), dtype=np.float64)
    else:
        u = np.full((n_L, k), 0.5, dtype=np.float64)
    full = integer_to_continuous_stratum(X_int, n_L, u)
    layers = [full[layer_indices[n], :].copy() for n in m_layers]

    validate_result(layers, m_layers, k, convention="stratum")
    return layers
