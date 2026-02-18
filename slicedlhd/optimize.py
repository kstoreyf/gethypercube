"""Simulated annealing optimizer for SLHD designs."""

import math
import numpy as np

from .construction import random_slhd
from .objective import init_dist_matrix, phi_from_dist_sq, phi_delta, update_dist_sq


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

    Sets T0 so that a 'typical' worsening move is accepted with ~80% probability:
        T0 = mean(|delta_phi| for worsening moves) / -log(0.8)
    """
    current_phi = phi_from_dist_sq(dist_sq, r)
    deltas = []

    for _ in range(n_samples):
        s = rng.integers(0, t)
        col = rng.integers(0, k)
        rows = np.arange(s * m, (s + 1) * m)
        i1, i2 = rng.choice(rows, size=2, replace=False)

        old_val_i1 = float(D[i1, col])
        old_val_i2 = float(D[i2, col])

        new_phi, _ = phi_delta(dist_sq, r, i1, i2, col, D, old_val_i1, old_val_i2)
        delta = new_phi - current_phi
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0  # fallback

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
    current_phi = phi_from_dist_sq(dist_sq, r)

    temp0 = _estimate_t0(D, dist_sq, t, m, k, r, rng)
    T = temp0
    cooling_rate = 0.9
    non_improving = 0
    total = 0

    best_D = D.copy()
    best_phi = current_phi

    while total < total_iter:
        # Propose a within-slice column swap
        s = int(rng.integers(0, t))
        col = int(rng.integers(0, k))
        rows = np.arange(s * m, (s + 1) * m)
        i1, i2 = rng.choice(rows, size=2, replace=False)
        i1, i2 = int(i1), int(i2)

        old_val_i1 = float(D[i1, col])
        old_val_i2 = float(D[i2, col])

        new_phi, new_dist_sq = phi_delta(dist_sq, r, i1, i2, col, D, old_val_i1, old_val_i2)
        delta = new_phi - current_phi

        accept = delta < 0 or (T > 1e-10 and rng.random() < math.exp(-delta / T))

        if accept:
            D[i1, col], D[i2, col] = D[i2, col], D[i1, col]
            dist_sq = new_dist_sq
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
