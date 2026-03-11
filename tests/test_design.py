"""Integration tests for sliced_lhd (the public entry point)."""

import numpy as np
import pytest
from gethypercube import sliced_lhd
from gethypercube.sliced_lhd import is_valid_lhd, is_valid_slhd


class TestSlicedLhdOutput:
    """Return type and shape."""

    def test_returns_list_of_arrays(self):
        slices = sliced_lhd(t=2, m=5, k=3, seed=0)
        assert isinstance(slices, list)
        assert all(isinstance(s, np.ndarray) for s in slices)

    def test_single_slice_shape(self):
        slices = sliced_lhd(t=1, m=8, k=3, seed=0)
        assert len(slices) == 1
        assert slices[0].shape == (8, 3)

    def test_multi_slice_shapes(self):
        t, m, k = 3, 4, 2
        slices = sliced_lhd(t=t, m=m, k=k, seed=0)
        assert len(slices) == t
        for s in slices:
            assert s.shape == (m, k)

    def test_full_design_shape(self):
        t, m, k = 4, 3, 5
        slices = sliced_lhd(t=t, m=m, k=k, seed=0)
        full = np.vstack(slices)
        assert full.shape == (t * m, k)


class TestSlicedLhdValidity:
    """LHD / SLHD structural correctness via the integer-level utilities."""

    def test_single_slice_is_valid_lhd_random(self):
        """Random construction: full design is a valid LHD."""
        slices = sliced_lhd(t=1, m=10, k=4, seed=1, scramble=False)
        full = np.vstack(slices)
        n = 10
        for j in range(4):
            strata = np.floor(full[:, j] * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(np.unique(strata)) == n

    def test_multi_slice_is_valid_slhd_random(self):
        """Random construction: each slice is a valid LHD over its stratum."""
        t, m, k = 3, 4, 3
        slices = sliced_lhd(t=t, m=m, k=k, seed=2, scramble=False)
        full = np.vstack(slices)
        n = t * m
        for j in range(k):
            col = full[:, j]
            strata = np.floor(col * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(np.unique(strata)) == n

    def test_single_slice_is_valid_lhd_sa(self):
        slices = sliced_lhd(t=1, m=10, k=4, optimization="sa",
                            seed=1, total_iter=1000)
        full = np.vstack(slices)
        n = 10
        for j in range(4):
            col = full[:, j]
            strata = np.floor(col * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(np.unique(strata)) == n

    def test_multi_slice_is_valid_slhd_sa(self):
        t, m, k = 3, 4, 3
        slices = sliced_lhd(t=t, m=m, k=k, optimization="sa",
                            seed=2, total_iter=500)
        full = np.vstack(slices)
        n = t * m
        for j in range(k):
            col = full[:, j]
            strata = np.floor(col * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(np.unique(strata)) == n


class TestSlicedLhdUnitInterval:
    """Values should be in (0, 1)."""

    def test_in_unit_interval_random(self):
        slices = sliced_lhd(t=2, m=5, k=3, seed=0)
        full = np.vstack(slices)
        assert full.min() >= 0
        assert full.max() <= 1

    def test_in_unit_interval_sa(self):
        slices = sliced_lhd(t=2, m=5, k=3, optimization="sa",
                            seed=0, total_iter=500)
        full = np.vstack(slices)
        assert full.min() >= 0
        assert full.max() <= 1

    def test_single_slice_in_unit_interval(self):
        slices = sliced_lhd(t=1, m=10, k=3, seed=0)
        assert slices[0].min() >= 0
        assert slices[0].max() <= 1


class TestScramble:
    """scramble=True vs False standardization."""

    def test_scramble_false_centers_points(self):
        """scramble=False: points at stratum midpoints (rank - 0.5) / n."""
        slices = sliced_lhd(t=1, m=6, k=2, optimization="sa",
                            seed=0, total_iter=500, scramble=False)
        full = np.vstack(slices)
        n = 6
        expected_centers = (np.arange(1, n + 1) - 0.5) / n
        for j in range(2):
            np.testing.assert_array_almost_equal(
                np.sort(full[:, j]), expected_centers
            )

    def test_scramble_true_reproducible_with_seed(self):
        kwargs = dict(t=1, m=6, k=2, optimization="sa",
                      seed=42, total_iter=500, scramble=True)
        s1 = sliced_lhd(**kwargs)
        s2 = sliced_lhd(**kwargs)
        np.testing.assert_array_almost_equal(
            np.vstack(s1), np.vstack(s2)
        )

    def test_scramble_true_differs_from_centered(self):
        """scramble=True yields different values from scramble=False."""
        common = dict(t=1, m=8, k=2, optimization="sa",
                      seed=99, total_iter=500)
        s_scrambled = sliced_lhd(**common, scramble=True)
        s_centered = sliced_lhd(**common, scramble=False)
        assert not np.allclose(np.vstack(s_scrambled), np.vstack(s_centered))


class TestReproducibility:

    def test_same_seed_same_result(self):
        kwargs = dict(t=2, m=5, k=3, seed=42, total_iter=500, optimization="sa")
        s1 = sliced_lhd(**kwargs)
        s2 = sliced_lhd(**kwargs)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        common = dict(t=2, m=5, k=3, total_iter=500, optimization="sa")
        s1 = sliced_lhd(seed=0, **common)
        s2 = sliced_lhd(seed=99, **common)
        assert not np.array_equal(np.vstack(s1), np.vstack(s2))


class TestOptimization:
    """The optimization parameter selects SA vs random."""

    def test_none_returns_random_design(self):
        slices = sliced_lhd(t=2, m=5, k=3, optimization=None, seed=0)
        assert len(slices) == 2
        assert slices[0].shape == (5, 3)

    def test_sa_returns_design(self):
        slices = sliced_lhd(t=2, m=5, k=3, optimization="sa",
                            seed=0, total_iter=500)
        assert len(slices) == 2
        assert slices[0].shape == (5, 3)

    def test_invalid_optimization_raises(self):
        with pytest.raises(ValueError, match="optimization"):
            sliced_lhd(t=2, m=5, k=3, optimization="bogus")


class TestInputValidation:

    @pytest.mark.parametrize("t,m,k", [(0, 5, 3), (1, 1, 3), (1, 5, 0)])
    def test_invalid_inputs_raise(self, t, m, k):
        with pytest.raises(ValueError):
            sliced_lhd(t=t, m=m, k=k)
