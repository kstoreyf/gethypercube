"""Sliced Latin hypercube design — single public entry point."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .construction import random_slhd
from .optimize import run_sa

_VALID_OPTIMIZATIONS = frozenset({"sa"})


def _standardize(
    D: np.ndarray, n: int, scramble: bool, rng: np.random.Generator
) -> np.ndarray:
    """Convert integer design (1-indexed) to continuous [0, 1]."""
    Df = D.astype(float)
    if scramble:
        result = np.empty_like(Df)
        for j in range(D.shape[1]):
            ranks = rankdata(Df[:, j])
            u = rng.uniform(0, 1, size=len(ranks))
            result[:, j] = (ranks - 1 + u) / n
        return result
    return np.apply_along_axis(
        lambda col: (rankdata(col) - 0.5) / n, axis=0, arr=Df
    )


def sliced_lhd(
    t: int,
    m: int,
    k: int,
    optimization: str | None = None,
    power: int = 15,
    nstarts: int = 1,
    itermax: int = 100,
    total_iter: int = 1_000_000,
    seed: int | None = None,
    scramble: bool = True,
) -> list[np.ndarray]:
    """
    Generate a Sliced Latin Hypercube Design (SLHD).

    The full design (n = m * t points) and every individual slice (m points)
    are each valid LHDs.  When t=1, produces a standard maximin-distance LHD.

    Parameters
    ----------
    t : int
        Number of slices. Use t=1 for a plain LHD.
    m : int
        Points per slice (>= 2). Total n = m * t.
    k : int
        Number of input dimensions (>= 1).
    optimization : {'sa'} or None
        None (default) — random construction, no optimisation.
        'sa' — simulated annealing on the maximin distance criterion
               phi_r(X) (Ba et al. 2015).
    power : int
        Exponent r in the average reciprocal-distance criterion (default 15).
        Higher values approximate true maximin more closely.
        Only used when optimization='sa'.
    nstarts : int
        Independent random restarts; best result is returned.
        Only used when optimization='sa'.
    itermax : int
        Non-improving iterations before the SA temperature is cooled.
        Only used when optimization='sa'.
    total_iter : int
        Hard cap on total SA iterations (split across nstarts).
        Only used when optimization='sa'.
    seed : int or None
        Random seed for reproducibility.
    scramble : bool
        True (default) — uniform jitter within each stratum (like SciPy LHS).
        False — stratum midpoints: (rank - 0.5) / n.

    Returns
    -------
    list[np.ndarray]
        List of t arrays each of shape (m, k), continuous in [0, 1]^k.
        Slice i is slices[i].  Full design: np.vstack(slices).
    """
    if t < 1 or m < 2 or k < 1:
        raise ValueError("Require t >= 1, m >= 2, k >= 1")
    if optimization is not None and optimization not in _VALID_OPTIMIZATIONS:
        raise ValueError(
            f"optimization must be None or one of {sorted(_VALID_OPTIMIZATIONS)}"
        )

    n = m * t
    rng = np.random.default_rng(seed)

    if optimization == "sa":
        seed_seq = np.random.SeedSequence(seed)
        child_seeds = seed_seq.spawn(nstarts)
        best_D: np.ndarray | None = None
        best_phi = float("inf")
        iter_per_start = total_iter // nstarts
        for i in range(nstarts):
            child_rng = np.random.default_rng(child_seeds[i])
            D, phi_val, _ = run_sa(t, m, k, power, itermax, iter_per_start, child_rng)
            if phi_val < best_phi:
                best_phi = phi_val
                best_D = D
        assert best_D is not None
        D = best_D
    else:
        D = random_slhd(t, m, k, rng)

    X = _standardize(D, n, scramble=scramble, rng=rng)
    return [X[s * m : (s + 1) * m] for s in range(t)]
