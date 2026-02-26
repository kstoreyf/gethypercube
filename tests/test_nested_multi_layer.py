"""Tests for multi-layer build_nested_lhd."""

import numpy as np
import pytest
from gethypercube import build_nested_lhd
from gethypercube.nested_lhd import (
    validate_m_layers,
    check_valid_lhd,
    check_nested,
    validate_result,
    suggest_valid_layers,
)


class TestBuildNestedLhd:
    def test_returns_list_of_arrays(self):
        layers = build_nested_lhd(
            k=2, m_layers=[2, 3, 5], seed=42,
            n_restarts=1, max_outer_iters=5,
        )
        assert isinstance(layers, list)
        assert len(layers) == 3
        for L in layers:
            assert isinstance(L, np.ndarray)
            assert L.dtype == np.float64

    def test_shapes(self):
        m_layers = [2, 3, 5, 9]
        layers = build_nested_lhd(
            k=4, m_layers=m_layers, seed=0,
            n_restarts=1, max_outer_iters=5,
        )
        for i, (L, n) in enumerate(zip(layers, m_layers)):
            assert L.shape == (n, 4), f"Layer {i}"

    def test_each_layer_valid_lhd(self):
        layers = build_nested_lhd(
            k=3, m_layers=[2, 3, 5, 9], seed=1,
            n_restarts=1, max_outer_iters=5,
            scramble=False,  # grid-aligned so check_valid_lhd passes
        )
        n_full = len(layers[-1])
        for i, L in enumerate(layers):
            assert check_valid_lhd(L, convention="stratum", n_full=n_full), (
                f"Layer {i} not valid LHD"
            )

    def test_nesting(self):
        layers = build_nested_lhd(
            k=2, m_layers=[2, 3, 5], seed=2,
            n_restarts=1, max_outer_iters=5,
            scramble=False,
        )
        for i in range(len(layers) - 1):
            assert check_nested(layers[i], layers[i + 1]), f"Layers {i}, {i+1}"

    def test_nesting_preserved_with_scramble(self):
        """With scramble=True, smaller layers are exact subsets of larger (same rows)."""
        layers = build_nested_lhd(
            k=2, m_layers=[2, 3, 5], seed=2,
            n_restarts=1, max_outer_iters=5,
            scramble=True,
        )
        for i in range(len(layers) - 1):
            assert check_nested(layers[i], layers[i + 1]), f"Layers {i}, {i+1}"

    def test_validate_result_passes(self):
        layers = build_nested_lhd(
            k=2, m_layers=[2, 3, 5, 9], seed=3,
            n_restarts=1, max_outer_iters=5,
            scramble=False,
        )
        validate_result(layers, [2, 3, 5, 9], 2, convention="stratum")

    def test_in_unit_interval(self):
        layers = build_nested_lhd(
            k=2, m_layers=[2, 4, 10], seed=4,
            n_restarts=1, max_outer_iters=5,
        )
        for L in layers:
            assert np.all(L >= 0) and np.all(L <= 1)

    def test_invalid_m_layers_raises(self):
        with pytest.raises(ValueError):
            build_nested_lhd(k=2, m_layers=[2, 4, 8], seed=0)
        with pytest.raises(ValueError, match="at least 2 entries"):
            build_nested_lhd(k=2, m_layers=[], seed=0)

    def test_reproducible_with_seed(self):
        L1 = build_nested_lhd(k=2, m_layers=[2, 3, 5], seed=100, n_restarts=1, max_outer_iters=10, scramble=True)
        L2 = build_nested_lhd(k=2, m_layers=[2, 3, 5], seed=100, n_restarts=1, max_outer_iters=10, scramble=True)
        for a, b in zip(L1, L2):
            np.testing.assert_array_almost_equal(a, b)

    def test_scramble_places_values_in_cells(self):
        """With scramble=True, values in [0,1) and one per stratum per column."""
        layers = build_nested_lhd(
            k=2, m_layers=[2, 3, 5], seed=42,
            n_restarts=1, max_outer_iters=5,
            scramble=True,
        )
        for i, L in enumerate(layers):
            n = L.shape[0]
            assert np.all(L >= 0) and np.all(L < 1)
            for j in range(L.shape[1]):
                col = L[:, j]
                # Stratum: floor(x*n) gives cell index 0..n-1
                cells = np.floor(col * n).astype(np.int64)
                cells = np.clip(cells, 0, n - 1)
                assert len(np.unique(cells)) == n, f"Layer {i} col {j} should have one value per cell"


class TestSuggestValidLayers:
    def test_returns_list_of_lists(self):
        seqs = suggest_valid_layers(20, n_min=2, n_layers=4)
        assert isinstance(seqs, list)
        for s in seqs:
            assert isinstance(s, list)
            assert len(s) == 4

    def test_sequences_are_valid(self):
        seqs = suggest_valid_layers(100, n_min=2, n_layers=4)
        for m in seqs:
            validate_m_layers(m)

    def test_strictly_increasing(self):
        seqs = suggest_valid_layers(50, n_min=2, n_layers=4)
        for m in seqs:
            assert m == sorted(m)
            assert len(m) == len(set(m))
