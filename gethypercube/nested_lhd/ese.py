"""Enhanced Stochastic Evolutionary algorithm for two-layer nested maximin LHD.

GROUPRAND (point + group exchanges) and POINTRAND (point-only), with threshold
acceptance outer loop. Returns (X2, I1) with best d = min(d_1, d_2).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .utils import scaling_factor


def _dist_sq_matrix(X_int: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances in integer grid. Shape (n, n)."""
    d = pdist(X_int.astype(np.float64), metric="sqeuclidean")
    return squareform(d)


def _objective(
    dist_sq: np.ndarray,
    I1: np.ndarray,
    n_1: int,
    n_2: int,
    k: int,
) -> float:
    """
    d = min(d_1, d_2). dist_sq is (n_2 x n_2) for full X2.
    """
    s_1 = scaling_factor(k, n_1)
    s_2 = scaling_factor(k, n_2)
    # Continuous-space min dist = sqrt(dist_sq_int) / (n - 1)
    # d_j = min_dist_cont / s_j
    inner_idx = np.ix_(I1, I1)
    inner_d_sq = dist_sq[inner_idx]
    n_inner = len(I1)
    if n_inner < 2:
        d_1 = float("inf")
    else:
        upper = np.triu_indices(n_inner, k=1)
        min_inner = np.min(inner_d_sq[upper])
        if min_inner <= 0:
            d_1 = 0.0
        else:
            min_d_cont_1 = np.sqrt(min_inner) / (n_1 - 1)
            d_1 = min_d_cont_1 / s_1

    upper_full = np.triu_indices(n_2, k=1)
    min_full = np.min(dist_sq[upper_full])
    if min_full <= 0:
        d_2 = 0.0
    else:
        min_d_cont_2 = np.sqrt(min_full) / (n_2 - 1)
        d_2 = min_d_cont_2 / s_2

    return min(d_1, d_2)


def _update_dist_sq_rows(
    dist_sq: np.ndarray, X2: np.ndarray, rows: np.ndarray
) -> None:
    """Recompute dist_sq for given rows (in-place). Used after group exchange."""
    Xf = X2.astype(np.float64)
    for i in rows:
        i = int(i)
        diff = Xf[i] - Xf
        np.sum(diff * diff, axis=1, out=dist_sq[i, :])
        dist_sq[:, i] = dist_sq[i, :]
        dist_sq[i, i] = 0.0


def _update_dist_sq(
    dist_sq: np.ndarray,
    X2: np.ndarray,
    p: int,
    q: int,
    j: int,
    old_p: int,
    old_q: int,
) -> None:
    """Update dist_sq after swapping X2[p,j] and X2[q,j] (integer values old_p, old_q)."""
    n, k = X2.shape
    col = X2[:, j].astype(np.float64)
    new_p = float(X2[p, j])
    new_q = float(X2[q, j])
    # Rows p and q change only in column j: delta = (new - old)^2 for that dimension
    delta_p = (new_p - col) ** 2 - (old_p - col) ** 2
    delta_q = (new_q - col) ** 2 - (old_q - col) ** 2
    dist_sq[p, :] += delta_p
    dist_sq[:, p] += delta_p
    dist_sq[q, :] += delta_q
    dist_sq[:, q] += delta_q
    dist_sq[p, p] = 0.0
    dist_sq[q, q] = 0.0
    d_pq = np.sum((X2[p].astype(float) - X2[q].astype(float)) ** 2)
    dist_sq[p, q] = d_pq
    dist_sq[q, p] = d_pq


def _initial_design(
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build initial valid nested design. X2 is (n_2 x k), I1 = np.arange(n_1).
    """
    c = (n_2 - 1) // (n_1 - 1)
    X2 = np.zeros((n_2, k), dtype=np.int64)
    I1 = np.arange(n_1)

    inner_levels = np.arange(0, n_2, c)  # 0, c, 2c, ..., (n_1-1)*c
    assert len(inner_levels) == n_1
    mask = np.ones(n_2, dtype=bool)
    mask[inner_levels] = False
    outer_levels = np.where(mask)[0]

    for j in range(k):
        perm_inner = rng.permutation(inner_levels.copy())
        perm_outer = rng.permutation(outer_levels.copy())
        X2[:n_1, j] = perm_inner
        X2[n_1:, j] = perm_outer

    return X2, I1


def _point_exchange(
    X2: np.ndarray,
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
    outer_only: bool = False,
) -> tuple[int, int, int, int, int] | None:
    """
    Perform one POINT exchange. Returns (p, q, j, old_p, old_q) if done,
    else None. If outer_only=True, p and q are restricted to outer rows [n_1, n_2).
    I1 is always np.arange(n_1).
    """
    if outer_only:
        n_outer = n_2 - n_1
        if n_outer < 2:
            return None
        idx = rng.choice(n_outer, size=2, replace=False)
        p, q = int(n_1 + idx[0]), int(n_1 + idx[1])
    else:
        p = int(rng.integers(0, n_2))
        in_inner = p < n_1
        if in_inner:
            q_candidates = np.arange(n_1)
            q_candidates = q_candidates[q_candidates != p]
        else:
            q_candidates = np.arange(n_1, n_2)
            q_candidates = q_candidates[q_candidates != p]
        if len(q_candidates) == 0:
            return None
        q = int(rng.choice(q_candidates))
    j = int(rng.integers(0, k))
    old_p = int(X2[p, j])
    old_q = int(X2[q, j])
    return (p, q, j, old_p, old_q)


def _group_exchange(
    X2: np.ndarray,
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """
    Perform one GROUP exchange in a random dimension. Returns changed row indices
    if done, else None. I1 is always np.arange(n_1).
    """
    c = (n_2 - 1) // (n_1 - 1)
    if c <= 1:
        return None
    j = int(rng.integers(0, k))
    inner_vals = np.sort(X2[:n_1, j])
    outer_indices = np.arange(n_1, n_2)
    vals = X2[outer_indices, j]
    # Vectorized group assignment: -1 if val < min, n_1-1 if val >= max, else bin
    group_ids = np.searchsorted(inner_vals, vals, side="right") - 1
    groups: dict[int, list[int]] = {}
    for idx, g in enumerate(group_ids):
        g = int(g)
        groups.setdefault(g, []).append(int(outer_indices[idx]))
    group_keys = [g for g in groups if len(groups[g]) > 0]
    if len(group_keys) < 2:
        return None
    g1, g2 = rng.choice(group_keys, size=2, replace=False)
    list1 = groups[g1]
    list2 = groups[g2]
    if len(list1) != len(list2):
        return None
    vals1 = X2[list1, j].copy()
    vals2 = X2[list2, j].copy()
    rng.shuffle(vals1)
    rng.shuffle(vals2)
    X2[list1, j] = vals2
    X2[list2, j] = vals1
    return np.array(list1 + list2)


def _run_ese(
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
    max_outer: int,
    inner_iters_per_point: int,
    use_group_exchange: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run one ESE pass. Returns (X2, I1, best_d).
    """
    X2, I1 = _initial_design(n_1, n_2, k, rng)
    dist_sq = _dist_sq_matrix(X2)
    current_d = _objective(dist_sq, I1, n_1, n_2, k)
    best_d = current_d
    best_X2 = X2.copy()

    # Fewer inner iters for small designs (design space is small)
    nk = n_2 * k
    iip = inner_iters_per_point if nk >= 100 else max(5, (inner_iters_per_point * nk) // 50)
    inner_iters = iip * nk
    threshold = max(1e-12, 0.005 * current_d)
    patience = 5
    no_improve = 0

    for _ in range(max_outer):
        improved = False
        for _ in range(inner_iters):
            if use_group_exchange and rng.random() < (0.5 if n_2 >= 15 else 0.2):
                X2_cand = X2.copy()
                changed = _group_exchange(X2_cand, n_1, n_2, k, rng)
                if changed is not None:
                    dist_sq_cand = dist_sq.copy()
                    _update_dist_sq_rows(dist_sq_cand, X2_cand, changed)
                    cand_d = _objective(dist_sq_cand, I1, n_1, n_2, k)
                    delta = cand_d - current_d
                    if delta >= -threshold:
                        X2 = X2_cand
                        dist_sq = dist_sq_cand
                        current_d = cand_d
                        if delta > 0:
                            improved = True
                            if current_d > best_d:
                                best_d = current_d
                                best_X2 = X2.copy()
                continue

            move = _point_exchange(X2, n_1, n_2, k, rng, outer_only=False)
            if move is None:
                continue
            p, q, j, old_p, old_q = move
            X2[p, j], X2[q, j] = X2[q, j], X2[p, j]
            _update_dist_sq(dist_sq, X2, p, q, j, old_p, old_q)
            new_d = _objective(dist_sq, I1, n_1, n_2, k)
            delta = new_d - current_d
            if delta >= -threshold:
                current_d = new_d
                if delta > 0:
                    improved = True
                    if current_d > best_d:
                        best_d = current_d
                        best_X2 = X2.copy()
            else:
                X2[p, j], X2[q, j] = X2[q, j], X2[p, j]
                _update_dist_sq(dist_sq, X2, p, q, j, old_q, old_p)

        if improved:
            threshold *= 0.9
        else:
            no_improve += 1
            if no_improve > patience:
                break
            threshold *= 1.1

    return best_X2, I1, best_d


def two_layer_ese(
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
    n_restarts: int = 5,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GROUPRAND and POINTRAND each with n_restarts; return (X2, I1) with
    best d = min(d_1, d_2). I1 is always np.arange(n_1).
    """
    best_d = -1.0
    best_X2 = None
    best_I1 = None

    for use_group in [True, False]:
        for _ in range(n_restarts):
            X2, I1, d = _run_ese(
                n_1, n_2, k, rng,
                max_outer=max_outer_iters,
                inner_iters_per_point=inner_iters_per_point,
                use_group_exchange=use_group,
            )
            if d > best_d:
                best_d = d
                best_X2 = X2
                best_I1 = I1

    assert best_X2 is not None and best_I1 is not None
    return best_X2, best_I1


def run_ese_extend(
    X2_init: np.ndarray,
    n_1: int,
    n_2: int,
    k: int,
    rng: np.random.Generator,
    max_outer: int = 200,
    inner_iters_per_point: int = 100,
    n_restarts: int = 3,
) -> np.ndarray:
    """
    Run ESE with inner rows (first n_1) fixed. Only outer-complement rows
    are modified. Returns best X2 (integer grid).
    """
    I1 = np.arange(n_1)
    best_d = -1.0
    best_X2 = X2_init.copy()

    for use_group in [True, False]:
        for _ in range(n_restarts):
            X2 = best_X2.copy()
            dist_sq = _dist_sq_matrix(X2)
            current_d = _objective(dist_sq, I1, n_1, n_2, k)
            run_best_d = current_d
            run_best_X2 = X2.copy()

            nk = n_2 * k
            iip = inner_iters_per_point if nk >= 100 else max(5, (inner_iters_per_point * nk) // 50)
            inner_iters = iip * nk
            threshold = max(1e-12, 0.005 * current_d)
            patience = 5
            no_improve = 0

            for _ in range(max_outer):
                improved = False
                for _ in range(inner_iters):
                    if use_group and rng.random() < (0.5 if n_2 >= 15 else 0.2):
                        X2_cand = X2.copy()
                        changed = _group_exchange(X2_cand, n_1, n_2, k, rng)
                        if changed is not None:
                            dist_sq_cand = dist_sq.copy()
                            _update_dist_sq_rows(dist_sq_cand, X2_cand, changed)
                            cand_d = _objective(dist_sq_cand, I1, n_1, n_2, k)
                            delta = cand_d - current_d
                            if delta >= -threshold:
                                X2 = X2_cand
                                dist_sq = dist_sq_cand
                                current_d = cand_d
                                if delta > 0:
                                    improved = True
                                    if current_d > run_best_d:
                                        run_best_d = current_d
                                        run_best_X2 = X2.copy()
                        continue

                    move = _point_exchange(X2, n_1, n_2, k, rng, outer_only=True)
                    if move is None:
                        continue
                    p, q, j, old_p, old_q = move
                    X2[p, j], X2[q, j] = X2[q, j], X2[p, j]
                    _update_dist_sq(dist_sq, X2, p, q, j, old_p, old_q)
                    new_d = _objective(dist_sq, I1, n_1, n_2, k)
                    delta = new_d - current_d
                    if delta >= -threshold:
                        current_d = new_d
                        if delta > 0:
                            improved = True
                            if current_d > run_best_d:
                                run_best_d = current_d
                                run_best_X2 = X2.copy()
                    else:
                        X2[p, j], X2[q, j] = X2[q, j], X2[p, j]
                        _update_dist_sq(dist_sq, X2, p, q, j, old_q, old_p)

                if improved:
                    threshold *= 0.9
                else:
                    no_improve += 1
                    if no_improve > patience:
                        break
                    threshold *= 1.1

            if run_best_d > best_d:
                best_d = run_best_d
                best_X2 = run_best_X2.copy()

    return best_X2
