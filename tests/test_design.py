"""Integration tests for maximinSLHD."""

import numpy as np
import pytest
from slicedlhd import maximinSLHD, SLHDResult
from slicedlhd.utils import is_valid_lhd, is_valid_slhd


class TestMaximinSLHD:
    # ------------------------------------------------------------------ #
    # Output shape and type                                                #
    # ------------------------------------------------------------------ #

    def test_returns_slhd_result(self):
        result = maximinSLHD(t=1, m=5, k=2, random_state=0, total_iter=500)
        assert isinstance(result, SLHDResult)

    def test_shape_single_slice(self):
        """t=1: design shape is (m, k), no slice column."""
        result = maximinSLHD(t=1, m=8, k=3, random_state=0, total_iter=500)
        assert result.design.shape == (8, 3)
        assert result.std_design.shape == (8, 3)

    def test_shape_multi_slice(self):
        """t>1: design shape is (n, k+1) where first col is slice label."""
        result = maximinSLHD(t=3, m=4, k=2, random_state=0, total_iter=500)
        assert result.design.shape == (12, 3)
        assert result.std_design.shape == (12, 3)

    def test_metadata_fields(self):
        result = maximinSLHD(t=3, m=4, k=2, random_state=0, total_iter=500)
        assert result.n_slices == 3
        assert result.n_per_slice == 4
        assert result.n_dims == 2

    # ------------------------------------------------------------------ #
    # LHD / SLHD correctness                                              #
    # ------------------------------------------------------------------ #

    def test_single_slice_is_valid_lhd(self):
        result = maximinSLHD(t=1, m=10, k=4, random_state=1, total_iter=1000)
        assert is_valid_lhd(result.design, n=10)

    def test_multi_slice_design_variables_valid_slhd(self):
        """Strip the slice column and check SLHD property on the k variable cols."""
        t, m, k = 3, 4, 3
        result = maximinSLHD(t=t, m=m, k=k, random_state=2, total_iter=500)
        D_vars = result.design[:, 1:]  # drop slice column
        assert is_valid_slhd(D_vars, t=t, m=m)

    def test_slice_column_values(self):
        """First column should be 1..t repeated m times."""
        t, m, k = 4, 3, 2
        result = maximinSLHD(t=t, m=m, k=k, random_state=0, total_iter=500)
        expected = np.repeat(np.arange(1, t + 1), m)
        np.testing.assert_array_equal(result.design[:, 0], expected)

    # ------------------------------------------------------------------ #
    # Standardized design                                                  #
    # ------------------------------------------------------------------ #

    def test_std_design_in_unit_interval(self):
        result = maximinSLHD(t=2, m=5, k=3, random_state=0, total_iter=500)
        # For t>1, skip the slice column (integer labels)
        std_vars = result.std_design[:, 1:]
        assert std_vars.min() > 0
        assert std_vars.max() < 1

    def test_std_design_single_slice_in_unit_interval(self):
        result = maximinSLHD(t=1, m=10, k=3, random_state=0, total_iter=500)
        assert result.std_design.min() > 0
        assert result.std_design.max() < 1

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #

    def test_reproducible_with_same_seed(self):
        kwargs = dict(t=2, m=5, k=3, random_state=42, total_iter=500)
        r1 = maximinSLHD(**kwargs)
        r2 = maximinSLHD(**kwargs)
        np.testing.assert_array_equal(r1.design, r2.design)
        assert r1.measure == r2.measure

    def test_different_seeds_differ(self):
        r1 = maximinSLHD(t=2, m=5, k=3, random_state=0, total_iter=500)
        r2 = maximinSLHD(t=2, m=5, k=3, random_state=99, total_iter=500)
        assert not np.array_equal(r1.design, r2.design)

    # ------------------------------------------------------------------ #
    # Measure                                                              #
    # ------------------------------------------------------------------ #

    def test_measure_is_positive_float(self):
        result = maximinSLHD(t=1, m=6, k=2, random_state=0, total_iter=500)
        assert isinstance(result.measure, float)
        assert result.measure > 0

    def test_temp0_is_positive(self):
        result = maximinSLHD(t=1, m=6, k=2, random_state=0, total_iter=500)
        assert result.temp0 > 0

    def test_multiple_starts_measure_le_single_start(self):
        """More starts should not yield a worse measure than 1 start."""
        kw = dict(t=2, m=4, k=2, random_state=7, total_iter=2000)
        r1 = maximinSLHD(nstarts=1, **kw)
        r3 = maximinSLHD(nstarts=3, **kw)
        # Not strictly guaranteed, but should hold in practice for a convex SA
        assert r3.measure <= r1.measure + 1e-6

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("t,m,k", [(0, 5, 3), (1, 1, 3), (1, 5, 0)])
    def test_invalid_inputs_raise(self, t, m, k):
        with pytest.raises(ValueError):
            maximinSLHD(t=t, m=m, k=k)

    def test_invalid_nstarts_raises(self):
        with pytest.raises(ValueError):
            maximinSLHD(t=1, m=5, k=2, nstarts=0)
