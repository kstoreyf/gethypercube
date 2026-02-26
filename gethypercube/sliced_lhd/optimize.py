"""Simulated annealing optimizer for SLHD designs."""
from __future__ import annotations

import math
import numpy as np

from .construction import random_slhd
from .objective import (
    init_dist_matrix,
    init_S_sum,
    phi_mm_from_dist_sq,
    phi_mm_delta,
    phi_mm_delta_incremental,
    update_dist_sq,
)


def _estimate_t0(
    D: np.ndarray,
    dist_sq: np.ndarray,
    t: int,
    m: int,
    k: int,
    r: int,
    rng: np.random.Generator,
    n_samples: int = 200,
) -> float:
    """
    Estimate initial SA temperature by sampling random swaps.
    """
    current_phi = phi_mm_from_dist_sq(dist_sq, r, t, m)
    deltas = []

    for _ in range(n_samples):
        if rng.random() <= 0.5:
            # Within-slice swap
            s = rng.integers(0, t)
            col = rng.integers(0, k)
            rows = np.arange(s * m, (s + 1) * m)
            i1, i2 = rng.choice(rows, size=2, replace=False)
        else:
            # Cross-slice swap: choose col, level l, swap two in same Π_l
            col = rng.integers(0, k)
            l = rng.integers(1, m + 1)
            lo, hi = (l - 1) * t + 1, l * t
            rows = np.where((D[:, col] >= lo) & (D[:, col] <= hi))[0]
            if len(rows) < 2:
                continue
            i1, i2 = rng.choice(rows, size=2, replace=False)

        old_val_i1 = float(D[i1, col])
        old_val_i2 = float(D[i2, col])

        new_phi, _ = phi_mm_delta(
            dist_sq, r, t, m, i1, i2, col, D, old_val_i1, old_val_i2
        )
        delta = new_phi - current_phi
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0
    return float(np.mean(deltas) / (-math.log(0.8)))


def run_sa(
    t: int,
    m: int,
    k: int,
    r: int,
    itermax: int,
    total_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """
    Run one simulated annealing pass from a random start.

    Parameters
    ----------
    t, m, k, r, itermax, total_iter : see maximinSLHD docstring
    rng : np.random.Generator

    Returns
    -------
    best_D : np.ndarray, shape (n, k)
    best_phi : float
    temp0 : float
    """
    D = random_slhd(t, m, k, rng)
    dist_sq = init_dist_matrix(D)
    S_full, S_slices = init_S_sum(dist_sq, r, t, m)
    current_phi = phi_mm_from_dist_sq(dist_sq, r, t, m)

    temp0 = _estimate_t0(D, dist_sq, t, m, k, r, rng)
    T = temp0
    cooling_rate = 0.9
    non_improving = 0
    total = 0
    p0 = 0.5

    best_D = D.copy()
    best_phi = current_phi

    while total < total_iter:
        z = rng.random()
        if z <= p0:
            # Type (ii): within-slice swap
            s = int(rng.integers(0, t))
            col = int(rng.integers(0, k))
            rows = np.arange(s * m, (s + 1) * m)
            i1, i2 = rng.choice(rows, size=2, replace=False)
            i1, i2 = int(i1), int(i2)
        else:
            # Type (iii): cross-slice swap within same Π_l
            col = int(rng.integers(0, k))
            l = int(rng.integers(1, m + 1))
            lo, hi = (l - 1) * t + 1, l * t
            rows = np.where((D[:, col] >= lo) & (D[:, col] <= hi))[0]
            if len(rows) < 2:
                total += 1
                continue
            i1, i2 = rng.choice(rows, size=2, replace=False)
            i1, i2 = int(i1), int(i2)

        old_val_i1 = float(D[i1, col])
        old_val_i2 = float(D[i2, col])

        new_phi, new_S_full, new_S_slices = phi_mm_delta_incremental(
            dist_sq, S_full, S_slices, r, t, m, i1, i2, col, D, old_val_i1, old_val_i2
        )
        delta = new_phi - current_phi

        accept = delta < 0 or (T > 1e-10 and rng.random() < math.exp(-delta / T))

        if accept:
            D[i1, col], D[i2, col] = D[i2, col], D[i1, col]
            update_dist_sq(dist_sq, D, i1, i2, col, old_val_i1, old_val_i2)
            S_full, S_slices = new_S_full, new_S_slices
            current_phi = new_phi

            if current_phi < best_phi:
                best_phi = current_phi
                best_D = D.copy()
                non_improving = 0
            else:
                non_improving += 1
        else:
            non_improving += 1

        total += 1

        if non_improving >= itermax:
            T *= cooling_rate
            non_improving = 0

    return best_D, best_phi, temp0

