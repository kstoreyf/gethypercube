"""Main entry point: maximinSLHD."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata

from .optimize import run_sa


@dataclass
class SLHDResult:
    """
    Result of a maximin SLHD optimization.

    Attributes
    ----------
    design : np.ndarray, shape (n, k) or (n, k+1) when t > 1
        Raw integer design (1-indexed). When t > 1, the first column contains
        the slice label (1..t) and the remaining k columns are the design
        variables.
    std_design : np.ndarray, same shape as design
        Design standardized to (0, 1) using (rank - 0.5) / n.
        The slice column (if present) is carried through unchanged.
    measure : float
        Final phi value (average reciprocal distance). Lower = better.
    temp0 : float
        Initial SA temperature used during optimization.
    n_slices : int
        t — number of slices.
    n_per_slice : int
        m — points per slice.
    n_dims : int
        k — number of input dimensions.
    """

    design: np.ndarray
    std_design: np.ndarray
    measure: float
    temp0: float
    n_slices: int
    n_per_slice: int
    n_dims: int


def _standardize(D: np.ndarray, n: int) -> np.ndarray:
    """Standardize each column to (0, 1) via (rank - 0.5) / n."""
    return np.apply_along_axis(
        lambda col: (rankdata(col) - 0.5) / n, axis=0, arr=D.astype(float)
    )


def maximinSLHD(
    t: int,
    m: int,
    k: int,
    power: int = 15,
    nstarts: int = 1,
    itermax: int = 100,
    total_iter: int = 1_000_000,
    random_state: int | None = None,
) -> SLHDResult:
    """
    Generate a maximin-distance Sliced Latin Hypercube Design (SLHD).

    When t=1, produces a standard maximin-distance LHD.
    When t>1, produces an SLHD where the full design and each slice are
    individually valid LHDs, optimized for the maximin distance criterion.

    Parameters
    ----------
    t : int
        Number of slices. Use t=1 for a standard LHD.
    m : int
        Number of design points per slice. Total n = m * t.
    k : int
        Number of input variables (dimensions).
    power : int, optional
        Exponent r in the average reciprocal distance criterion. Default 15.
        Higher values more closely approximate true maximin distance.
    nstarts : int, optional
        Number of independent random restarts. The best result is returned.
    itermax : int, optional
        Max non-improving iterations before cooling the SA temperature.
    total_iter : int, optional
        Hard cap on total SA iterations across the full run.
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    SLHDResult
        Dataclass containing the design, standardized design, phi measure,
        initial temperature, and design parameters.

    Examples
    --------
    Standard maximin LHD (t=1):

    >>> result = maximinSLHD(t=1, m=10, k=3, random_state=42)
    >>> result.design.shape
    (10, 3)

    Sliced LHD with 3 slices of 4 points each:

    >>> result = maximinSLHD(t=3, m=4, k=2, random_state=42)
    >>> result.design.shape  # first col is slice label
    (12, 3)
    """
    if t < 1 or m < 2 or k < 1:
        raise ValueError("Require t >= 1, m >= 2, k >= 1")
    if nstarts < 1:
        raise ValueError("nstarts must be >= 1")

    seed_seq = np.random.SeedSequence(random_state)
    child_seeds = seed_seq.spawn(nstarts)

    best_D: np.ndarray | None = None
    best_phi = float("inf")
    best_temp0 = 0.0

    iter_per_start = total_iter // nstarts

    for i in range(nstarts):
        rng = np.random.default_rng(child_seeds[i])
        D, phi_val, temp0 = run_sa(t, m, k, power, itermax, iter_per_start, rng)
        if phi_val < best_phi:
            best_phi = phi_val
            best_D = D
            best_temp0 = temp0

    assert best_D is not None
    n = m * t

    if t > 1:
        slice_col = np.repeat(np.arange(1, t + 1), m).reshape(-1, 1)
        design_out = np.hstack([slice_col, best_D])
        std_vars = _standardize(best_D, n)
        std_out = np.hstack([slice_col, std_vars])
    else:
        design_out = best_D
        std_out = _standardize(best_D, n)

    return SLHDResult(
        design=design_out,
        std_design=std_out,
        measure=best_phi,
        temp0=best_temp0,
        n_slices=t,
        n_per_slice=m,
        n_dims=k,
    )
