"""Objective function (average reciprocal distance) and incremental update."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def init_S_sum(dist_sq: np.ndarray, r: int, t: int, m: int) -> tuple[float, np.ndarray]:
    """
    Compute initial S = sum(d^(-r/2)) for full matrix and each slice.
    Returns (S_full, S_slices) where S_slices has length t.
    """
    n = dist_sq.shape[0]
    exp = -r / 2.0

    upper_idx = np.triu_indices(n, k=1)
    S_full = float(np.sum(dist_sq[upper_idx] ** exp))

    S_slices = np.zeros(t, dtype=np.float64)
    for s in range(t):
        rows = np.arange(s * m, (s + 1) * m)
        sub = dist_sq[np.ix_(rows, rows)]
        u = np.triu_indices(m, k=1)
        S_slices[s] = float(np.sum(sub[u] ** exp))
    return S_full, S_slices


def _phi_from_dist_sq_block(
    dist_sq: np.ndarray, r: int, rows: np.ndarray
) -> float:
    """Compute phi from dist_sq for a subset of rows (e.g. one slice)."""
    sub = dist_sq[np.ix_(rows, rows)]
    n = len(rows)
    if n < 2:
        return 0.0
    upper_idx = np.triu_indices(n, k=1)
    upper_sq = sub[upper_idx]
    return float((np.mean(upper_sq ** (-r / 2))) ** (1.0 / r))


def phi_mm_from_dist_sq(
    dist_sq: np.ndarray, r: int, t: int, m: int
) -> float:
    """
    Compute φ_Mm (paper Eq. 3): (1/2)(φ_r(X) + (1/t)Σ φ_r(X_i)).
    """
    if t == 1:
        return phi_from_dist_sq(dist_sq, r)
    n = dist_sq.shape[0]
    phi_full = phi_from_dist_sq(dist_sq, r)
    phi_slices = []
    for s in range(t):
        rows = np.arange(s * m, (s + 1) * m)
        phi_slices.append(_phi_from_dist_sq_block(dist_sq, r, rows))
    phi_avg_slice = np.mean(phi_slices)
    return 0.5 * (phi_full + phi_avg_slice)


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


def phi_mm_delta(
    dist_sq: np.ndarray,
    r: int,
    t: int,
    m: int,
    i1: int,
    i2: int,
    col: int,
    D: np.ndarray,
    old_val_i1: float,
    old_val_i2: float,
) -> tuple[float, np.ndarray]:
    """
    Compute φ_Mm after swapping D[i1, col] <-> D[i2, col], without modifying D.
    """
    new_dist_sq = dist_sq.copy()
    D_temp = D.copy()
    D_temp[i1, col], D_temp[i2, col] = D_temp[i2, col], D_temp[i1, col]
    update_dist_sq(new_dist_sq, D_temp, i1, i2, col, old_val_i1, old_val_i2)
    new_phi_mm = phi_mm_from_dist_sq(new_dist_sq, r, t, m)
    return new_phi_mm, new_dist_sq


def phi_mm_delta_incremental(
    dist_sq: np.ndarray,
    S_full: float,
    S_slices: np.ndarray,
    r: int,
    t: int,
    m: int,
    i1: int,
    i2: int,
    col: int,
    D: np.ndarray,
    old_val_i1: float,
    old_val_i2: float,
) -> tuple[float, float, np.ndarray]:
    """
    Compute φ_Mm after swapping D[i1, col] <-> D[i2, col] using incremental S update.
    Returns (new_phi, new_S_full, new_S_slices). Does not modify any inputs.
    """
    n = dist_sq.shape[0]
    exp = -r / 2.0
    col_vals = D[:, col].astype(np.float64)
    new_val_i1 = old_val_i2
    new_val_i2 = old_val_i1

    delta_i1 = (new_val_i1 - col_vals) ** 2 - (old_val_i1 - col_vals) ** 2
    delta_i2 = (new_val_i2 - col_vals) ** 2 - (old_val_i2 - col_vals) ** 2

    # Affected pairs: (i1, j) and (i2, j) for all j (exclude diagonal to avoid 0^exp)
    d1 = np.delete(dist_sq[i1, :].astype(np.float64), i1)
    d2 = np.delete(dist_sq[i2, :].astype(np.float64), i2)
    old_sum_full = float(np.sum(d1 ** exp)) + float(np.sum(d2 ** exp))
    new_row1 = np.delete(dist_sq[i1, :].astype(np.float64) + delta_i1, i1)
    new_row2 = np.delete(dist_sq[i2, :].astype(np.float64) + delta_i2, i2)
    new_sum_full = float(np.sum(new_row1 ** exp)) + float(np.sum(new_row2 ** exp))

    new_S_full = S_full - old_sum_full + new_sum_full
    N_full = n * (n - 1) / 2
    phi_full = (new_S_full / N_full) ** (1.0 / r)

    if t == 1:
        new_S_slices = np.array([new_S_full], dtype=np.float64)
        return phi_full, new_S_full, new_S_slices

    new_S_slices = S_slices.copy()
    N_slice = m * (m - 1) / 2
    s1, s2 = i1 // m, i2 // m
    row1, row2 = i1 % m, i2 % m

    def slice_contrib(rows_in_slice: np.ndarray, idx: int, delta: np.ndarray) -> tuple[float, float]:
        """Old and new S contribution for pairs (idx, j) with j in slice."""
        old_vals = dist_sq[idx, rows_in_slice]
        mask = rows_in_slice != idx
        old_vals = old_vals[mask]
        old_s = float(np.sum(old_vals ** exp))
        new_vals = old_vals + delta[rows_in_slice[mask]]
        new_s = float(np.sum(new_vals ** exp))
        return old_s, new_s

    if s1 == s2:
        rows_s = np.arange(s1 * m, (s1 + 1) * m)
        old1, new1 = slice_contrib(rows_s, i1, delta_i1)
        old2, new2 = slice_contrib(rows_s, i2, delta_i2)
        d12_old = float(dist_sq[i1, i2] ** exp)
        d12_new = float((dist_sq[i1, i2] + delta_i1[i2]) ** exp)
        old_sum_s = old1 + old2 - d12_old
        new_sum_s = new1 + new2 - d12_new
        new_S_slices[s1] = S_slices[s1] - old_sum_s + new_sum_s
    else:
        rows1 = np.arange(s1 * m, (s1 + 1) * m)
        rows2 = np.arange(s2 * m, (s2 + 1) * m)
        old1, new1 = slice_contrib(rows1, i1, delta_i1)
        old2, new2 = slice_contrib(rows2, i2, delta_i2)
        new_S_slices[s1] = S_slices[s1] - old1 + new1
        new_S_slices[s2] = S_slices[s2] - old2 + new2

    phi_slices = np.array(
        [(new_S_slices[s] / N_slice) ** (1.0 / r) for s in range(t)]
    )
    phi_avg_slice = float(np.mean(phi_slices))
    new_phi = 0.5 * (phi_full + phi_avg_slice)
    return new_phi, new_S_full, new_S_slices

