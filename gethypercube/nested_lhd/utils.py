"""Distance calculations, grid utilities, and suggest_valid_layers for nested LHD."""

from __future__ import annotations

import numpy as np


def random_integer_lhd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Build random LHD of n points in k dimensions on integer levels 0..n-1."""
    X = np.zeros((n, k), dtype=np.int64)
    for j in range(k):
        X[:, j] = rng.permutation(np.arange(0, n, dtype=np.int64))
    return X


def scaling_factor(k: int, n: int) -> float:
    """
    Scaling factor s_j = 1 / (k * (n - 1))^(1/k) for layer with n points in k dims.
    Rennen et al. eq.(1).
    """
    if k < 1 or n < 2:
        return 1.0
    return 1.0 / ((k * (n - 1)) ** (1.0 / k))


def min_scaled_separation(
    X_int: np.ndarray, n: int, k: int
) -> float:
    """
    Minimum scaled separation distance for point set X_int (integer grid 0..n-1).

    d = min_{p != q} ||X[p] - X[q]||_2 / s
    where continuous coords are X_int / (n-1), and s = 1/(k*(n-1))^(1/k).

    So in integer space: dist_cont = dist_int / (n-1), and
    d = (dist_int / (n-1)) * (k*(n-1))^(1/k) = dist_int * k^(1/k) / (n-1)^(1-1/k).

    X_int : (n_pts, k) integer matrix
    n : layer size (grid is 0..n-1)
    k : number of dimensions
    """
    n_pts = X_int.shape[0]
    if n_pts < 2:
        return float("inf")
    # Pairwise distances in integer grid; then scale to get d
    from scipy.spatial.distance import pdist

    d_int = pdist(X_int.astype(np.float64), metric="euclidean")
    min_d_int = float(np.min(d_int))
    if min_d_int <= 0:
        return 0.0
    s = scaling_factor(k, n)
    # Continuous min dist = min_d_int / (n - 1)
    min_d_cont = min_d_int / (n - 1)
    return min_d_cont / s


def min_scaled_separation_incremental(
    dist_sq: np.ndarray, n: int, k: int
) -> float:
    """
    Given squared Euclidean distance matrix (in integer grid units), return
    min scaled separation d = (sqrt(min_dist_sq)/(n-1)) / s.
    """
    n_pts = dist_sq.shape[0]
    if n_pts < 2:
        return float("inf")
    upper = np.triu_indices(n_pts, k=1)
    min_d_sq = float(np.min(dist_sq[upper]))
    if min_d_sq <= 0:
        return 0.0
    min_d_int = np.sqrt(min_d_sq)
    s = scaling_factor(k, n)
    min_d_cont = min_d_int / (n - 1)
    return min_d_cont / s


def integer_to_continuous(X_int: np.ndarray, n: int) -> np.ndarray:
    """Map integer grid {0, 1, ..., n-1} to [0, 1]: x -> x / (n-1). Rennen convention."""
    return X_int.astype(np.float64) / (n - 1)


def integer_to_continuous_midpoint(X_int: np.ndarray, n: int) -> np.ndarray:
    """
    Map integer grid {0, 1, ..., n-1} to (0, 1): x -> (2*x + 1) / (2*n).
    Midpoint convention: levels at 1/(2n), 3/(2n), ..., (2n-1)/(2n).
    """
    return (2 * X_int.astype(np.float64) + 1) / (2 * n)


def integer_to_continuous_stratum(
    X_int: np.ndarray, n: int, u: np.ndarray
) -> np.ndarray:
    """
    Map integer grid {0, 1, ..., n-1} to [0, 1): x -> (x + u) / n.
    u has same shape as X_int; drawn once for full design so inner layers inherit.
    Strata [k/n, (k+1)/n) tile [0, 1) symmetrically.
    """
    return (X_int.astype(np.float64) + u) / n


def scramble_layer_rennen(
    layer: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly place values within each LHD cell (scipy QMC style).

    layer has shape (m, k), values are grid 0, 1/(n-1), ..., 1 (Rennen convention).
    Replaces each value in cell j with (j + u) / (n-1), u ~ Uniform(0, 1).
    """
    out = layer.copy()
    j = np.clip(np.round(layer * (n - 1)).astype(np.int64), 0, n - 1)
    u = rng.random(layer.shape, dtype=np.float64)
    out[:] = (j + u) / (n - 1)
    return out


def integer_to_continuous_qian(X_int: np.ndarray, n: int) -> np.ndarray:
    """Map integer grid {1, 2, ..., n} to (0, 1]: x -> x / n. Qian endpoint convention."""
    return X_int.astype(np.float64) / n


def integer_to_continuous_midpoint_1based(X_int: np.ndarray, n: int) -> np.ndarray:
    """
    Map integer grid {1, 2, ..., n} to (0, 1): x -> (2*x - 1) / (2*n).
    Midpoint convention: levels at 1/(2n), 3/(2n), ..., (2n-1)/(2n).
    """
    return (2 * X_int.astype(np.float64) - 1) / (2 * n)


def scramble_layer_midpoint(
    layer: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly place values within each LHD cell (midpoint convention).

    layer has shape (m, k), values are grid 1/(2n), 3/(2n), ..., (2n-1)/(2n).
    Replaces each value in cell i (0..n-1) with (i + u) / n, u ~ Uniform(0, 1),
    so results stay strictly in (0, 1).
    """
    out = layer.copy()
    # Cell index from midpoint value: x = (2*i+1)/(2n) => i = (2*n*x - 1) / 2
    i_cell = np.round((layer * 2 * n - 1) / 2).astype(np.int64)
    i_cell = np.clip(i_cell, 0, n - 1)
    u = rng.random(layer.shape, dtype=np.float64)
    out[:] = (i_cell + u) / n
    return out


def scramble_layer_qian(
    layer: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly place values within each LHD cell (scipy QMC style).

    layer has shape (m, k), values are grid 1/n, 2/n, ..., 1 (Qian convention).
    Replaces each value in cell v (1..n) with (v - 1 + u) / n, u ~ Uniform(0, 1).
    """
    out = layer.copy()
    v = np.clip(np.round(layer * n).astype(np.int64), 1, n)
    u = rng.random(layer.shape, dtype=np.float64)
    out[:] = (v - 1 + u) / n
    return out


def suggest_valid_layers_rennen(
    n_start: int,
    n_max: int,
    ratios: list[int] | None = None,
) -> list[list[int]]:
    """
    Valid m_layers for nested_maximin_lhd (Rennen).
    Pattern: n_i = (n_start - 1) * r^(i-1) + 1.
    """
    if ratios is None:
        ratios = [2, 3, 4, 5]
    sequences = []
    for r in ratios:
        seq = [n_start]
        while len(seq) < 20:
            next_n = (seq[-1] - 1) * r + 1
            if next_n > n_max:
                break
            seq.append(next_n)
        if len(seq) >= 2 and seq not in sequences:
            sequences.append(seq)
    return sequences


def m_layers_from_rennen(m_init: int, n_layers: int, ratio: int) -> list[int]:
    """
    Compute Rennen-valid m_layers from initial size, number of layers, and ratio.

    m_layers[i] = (m_init - 1) * ratio^i + 1 for i = 0 .. n_layers-1,
    so (n_{i+1}-1) / (n_i-1) = ratio for all consecutive pairs.

    Parameters
    ----------
    m_init : int
        Smallest layer size (first element of m_layers). Must be >= 2.
    n_layers : int
        Number of layers. Must be >= 2.
    ratio : int
        Integer ratio: (n_{i+1}-1) = (n_i-1) * ratio. Must be >= 2.

    Returns
    -------
    list[int]
        Strictly increasing list of layer sizes satisfying Rennen constraint.
    """
    if m_init < 2:
        raise ValueError(f"m_init must be >= 2; got {m_init}")
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2; got {n_layers}")
    if ratio < 2:
        raise ValueError(f"ratio must be >= 2; got {ratio}")
    m_layers = []
    n = m_init
    for _ in range(n_layers):
        m_layers.append(n)
        n = (n - 1) * ratio + 1
    return m_layers


def suggest_valid_layers_qian(
    n_start: int,
    n_max: int,
    ratios: list[int] | None = None,
) -> list[list[int]]:
    """
    Valid m_layers for nested_lhd (Qian).
    Pattern: n_i = n_start * r^(i-1).
    """
    if ratios is None:
        ratios = [2, 3, 4, 5, 10]
    sequences = []
    for r in ratios:
        seq = [n_start]
        while len(seq) < 20:
            next_n = seq[-1] * r
            if next_n > n_max:
                break
            seq.append(next_n)
        if len(seq) >= 2 and seq not in sequences:
            sequences.append(seq)
    return sequences


def suggest_valid_layers(
    n_max: int,
    n_min: int = 2,
    n_layers: int = 4,
) -> list[list[int]]:
    """
    Return valid m_layers for Rennen (backward compatible).
    Consecutive (n_{i+1} - 1) / (n_i - 1) are integers.
    """
    sequences = []
    for start in [n_min, 3, 5]:
        if start < 2:
            continue
        for r in [2, 3]:
            seq = [start]
            for _ in range(n_layers - 1):
                next_n = (seq[-1] - 1) * r + 1
                if next_n > n_max:
                    break
                seq.append(next_n)
            if len(seq) == n_layers and seq not in sequences:
                sequences.append(seq)
    return sequences
