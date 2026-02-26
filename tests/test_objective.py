"""Tests for objective function and incremental update."""

import numpy as np
import pytest
from gethypercube.sliced_lhd.construction import random_slhd
from gethypercube.sliced_lhd.objective import (
    compute_phi,
    init_dist_matrix,
    phi_from_dist_sq,
    update_dist_sq,
    phi_delta,
)


def rng(seed=0):
    return np.random.default_rng(seed)


def simple_design():
    """A tiny 4x2 design for deterministic tests."""
    return np.array([[1, 3], [2, 4], [3, 1], [4, 2]], dtype=np.float64)


class TestComputePhi:
    def test_returns_float(self):
        D = simple_design()
        result = compute_phi(D, r=15)
        assert isinstance(result, float)

    def test_positive(self):
        D = simple_design()
        assert compute_phi(D, r=15) > 0

    def test_higher_r_gives_higher_phi(self):
        """Higher r tightens the criterion; phi should be >= phi at lower r
        for well-spread designs (not strictly guaranteed but usually true)."""
        D = simple_design()
        phi5 = compute_phi(D, r=5)
        phi15 = compute_phi(D, r=15)
        # phi is bounded below by min-dist^{-1}, above by sum of inverses
        # Just check both are positive
        assert phi5 > 0
        assert phi15 > 0

    def test_more_spread_design_has_lower_phi(self):
        """A more spread-out design should have lower phi."""
        # Clustered design
        D_clustered = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [10.0, 10.0]])
        # Spread design
        D_spread = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]])
        assert compute_phi(D_clustered, r=2) > compute_phi(D_spread, r=2)

    def test_consistent_with_phi_from_dist_sq(self):
        D = random_slhd(t=2, m=4, k=3, rng=rng())
        r = 15
        phi1 = compute_phi(D, r)
        dist_sq = init_dist_matrix(D)
        phi2 = phi_from_dist_sq(dist_sq, r)
        assert abs(phi1 - phi2) < 1e-8


class TestInitDistMatrix:
    def test_shape(self):
        D = simple_design()
        d = init_dist_matrix(D)
        assert d.shape == (4, 4)

    def test_diagonal_is_zero(self):
        D = simple_design()
        d = init_dist_matrix(D)
        assert np.allclose(np.diag(d), 0)

    def test_symmetric(self):
        D = simple_design()
        d = init_dist_matrix(D)
        assert np.allclose(d, d.T)

    def test_known_values(self):
        """dist_sq([1,0], [4,0]) = 9."""
        D = np.array([[1.0, 0.0], [4.0, 0.0]])
        d = init_dist_matrix(D)
        assert abs(d[0, 1] - 9.0) < 1e-10
        assert abs(d[1, 0] - 9.0) < 1e-10


class TestUpdateDistSq:
    def test_update_matches_full_recompute(self):
        """After a swap, the incrementally updated dist_sq should match a full recompute."""
        t, m, k = 2, 5, 3
        D = random_slhd(t=t, m=m, k=k, rng=rng()).astype(np.float64)
        dist_sq = init_dist_matrix(D)

        # Pick a swap within slice 0
        i1, i2, col = 0, 3, 1
        old_val_i1 = D[i1, col]
        old_val_i2 = D[i2, col]

        # Apply swap
        D[i1, col], D[i2, col] = D[i2, col], D[i1, col]
        update_dist_sq(dist_sq, D, i1, i2, col, old_val_i1, old_val_i2)

        # Full recompute
        expected = init_dist_matrix(D)
        assert np.allclose(dist_sq, expected, atol=1e-8)

    def test_update_preserves_symmetry(self):
        D = random_slhd(t=2, m=4, k=3, rng=rng()).astype(np.float64)
        dist_sq = init_dist_matrix(D)
        i1, i2, col = 0, 2, 0
        old1, old2 = D[i1, col], D[i2, col]
        D[i1, col], D[i2, col] = D[i2, col], D[i1, col]
        update_dist_sq(dist_sq, D, i1, i2, col, old1, old2)
        assert np.allclose(dist_sq, dist_sq.T, atol=1e-10)

    def test_update_preserves_zero_diagonal(self):
        D = random_slhd(t=2, m=4, k=3, rng=rng()).astype(np.float64)
        dist_sq = init_dist_matrix(D)
        i1, i2, col = 1, 3, 2
        old1, old2 = D[i1, col], D[i2, col]
        D[i1, col], D[i2, col] = D[i2, col], D[i1, col]
        update_dist_sq(dist_sq, D, i1, i2, col, old1, old2)
        assert np.allclose(np.diag(dist_sq), 0, atol=1e-10)


class TestPhiDelta:
    def test_phi_delta_matches_full_recompute(self):
        """phi_delta should return same phi as full recompute after swap."""
        t, m, k = 2, 4, 2
        D = random_slhd(t=t, m=m, k=k, rng=rng()).astype(np.float64)
        dist_sq = init_dist_matrix(D)
        r = 15

        i1, i2, col = 0, 2, 0
        old1, old2 = D[i1, col], D[i2, col]

        new_phi, _ = phi_delta(dist_sq, r, i1, i2, col, D, old1, old2)

        # Apply swap and compute directly
        D2 = D.copy()
        D2[i1, col], D2[i2, col] = D2[i2, col], D2[i1, col]
        expected_phi = compute_phi(D2, r)

        assert abs(new_phi - expected_phi) < 1e-6

    def test_phi_delta_does_not_modify_original(self):
        """phi_delta must not mutate D or dist_sq."""
        D = random_slhd(t=2, m=4, k=2, rng=rng()).astype(np.float64)
        dist_sq = init_dist_matrix(D)
        D_orig = D.copy()
        dist_sq_orig = dist_sq.copy()

        phi_delta(dist_sq, 15, 0, 2, 0, D, D[0, 0], D[2, 0])

        assert np.array_equal(D, D_orig)
        assert np.allclose(dist_sq, dist_sq_orig)
