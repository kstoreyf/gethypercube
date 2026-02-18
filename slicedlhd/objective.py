"""Objective function (average reciprocal distance) and incremental update."""

import numpy as np
from scipy.spatial.distance import cdist


def compute_phi(D: np.ndarray, r: int) -> float:
    """
    Compute the average reciprocal inter-point distance measure phi(D, r).

    phi(D) = [ mean_{i<j}( 1 / dist(xi, xj)^r ) ]^(1/r)

    Lower phi = better (more spread out design).

    Parameters
    ----------
    D : np.ndarray, shape (n, k)
        Design matrix (float or int).
    r : int
        Power parameter.

    Returns
    -------
    float
    """
    n = D.shape[0]
    d = cdist(D.astype(float), D.astype(float))
    upper_idx = np.triu_indices(n, k=1)
    upper = d[upper_idx]
    return float((np.mean(upper ** (-r))) ** (1.0 / r))


def init_dist_matrix(D: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise squared Euclidean distance matrix.

    Returns
    -------
    dist_sq : np.ndarray, shape (n, n), float64
    """
    D_f = D.astype(np.float64)
    diff = D_f[:, np.newaxis, :] - D_f[np.newaxis, :, :]  # (n, n, k)
    return np.sum(diff ** 2, axis=2)


def phi_from_dist_sq(dist_sq: np.ndarray, r: int) -> float:
    """
    Compute phi from a precomputed squared distance matrix.
    """
    n = dist_sq.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    upper_sq = dist_sq[upper_idx]
    # dist = sqrt(dist_sq), dist^r = dist_sq^(r/2)
    return float((np.mean(upper_sq ** (-r / 2))) ** (1.0 / r))


def update_dist_sq(
    dist_sq: np.ndarray,
    D: np.ndarray,
    i1: int,
    i2: int,
    col: int,
    old_val_i1: float,
    old_val_i2: float,
) -> None:
    """
    Incrementally update dist_sq after swapping D[i1, col] <-> D[i2, col].

    Modifies dist_sq in-place. Assumes D has already been updated with the new values.

    Parameters
    ----------
    dist_sq : np.ndarray, shape (n, n)
        Pairwise squared distance matrix to update in-place.
    D : np.ndarray, shape (n, k)
        Design matrix *after* the swap.
    i1, i2 : int
        Indices of the two swapped rows.
    col : int
        Column in which the swap occurred.
    old_val_i1, old_val_i2 : float
        Values of D[i1, col] and D[i2, col] *before* the swap.
    """
    n = D.shape[0]
    new_val_i1 = float(D[i1, col])
    new_val_i2 = float(D[i2, col])

    # Update distances from i1 to all other rows
    col_vals = D[:, col].astype(np.float64)
    delta_i1 = (new_val_i1 - col_vals) ** 2 - (old_val_i1 - col_vals) ** 2
    delta_i2 = (new_val_i2 - col_vals) ** 2 - (old_val_i2 - col_vals) ** 2

    dist_sq[i1, :] += delta_i1
    dist_sq[:, i1] += delta_i1
    dist_sq[i2, :] += delta_i2
    dist_sq[:, i2] += delta_i2

    # The delta was applied twice to the (i1, i2) / (i2, i1) entries (once from
    # each row update), so we need to undo the double-count and recompute cleanly.
    d12 = float(np.sum((D[i1].astype(float) - D[i2].astype(float)) ** 2))
    dist_sq[i1, i2] = d12
    dist_sq[i2, i1] = d12
    # Self-distances must stay 0
    dist_sq[i1, i1] = 0.0
    dist_sq[i2, i2] = 0.0


def phi_delta(
    dist_sq: np.ndarray,
    r: int,
    i1: int,
    i2: int,
    col: int,
    D: np.ndarray,
    old_val_i1: float,
    old_val_i2: float,
) -> tuple[float, np.ndarray]:
    """
    Compute the change in phi if we swap D[i1, col] <-> D[i2, col],
    without permanently modifying D.

    Returns
    -------
    new_phi : float
    new_dist_sq : np.ndarray  (a copy with the update applied)
    """
    new_dist_sq = dist_sq.copy()
    # Temporarily apply swap to compute new distances
    D_temp = D.copy()
    D_temp[i1, col], D_temp[i2, col] = D_temp[i2, col], D_temp[i1, col]
    update_dist_sq(new_dist_sq, D_temp, i1, i2, col, old_val_i1, old_val_i2)
    new_phi = phi_from_dist_sq(new_dist_sq, r)
    return new_phi, new_dist_sq
