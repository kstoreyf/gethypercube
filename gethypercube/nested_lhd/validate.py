"""Validation for nested LHD: m_layers (Qian vs Rennen), LHD property, nesting."""

from __future__ import annotations

import numpy as np


def validate_m_layers_qian(m_layers: list[int]) -> None:
    """
    Validate m_layers for Qian (2009): n_{i+1} % n_i == 0 for all consecutive pairs.
    """
    if len(m_layers) < 2:
        raise ValueError(
            f"m_layers must have at least 2 entries; got {len(m_layers)}"
        )
    for i in range(len(m_layers) - 1):
        n_small, n_large = m_layers[i], m_layers[i + 1]
        if n_large <= n_small:
            raise ValueError(
                f"m_layers must be strictly increasing; "
                f"m_layers[{i}]={n_small} >= m_layers[{i+1}]={n_large}"
            )
        if n_small < 1:
            raise ValueError(
                f"Layer sizes must be >= 1; m_layers[{i}]={n_small}"
            )
        if n_large % n_small != 0:
            raise ValueError(
                f"Qian divisibility violated between layers {i} and {i+1}: "
                f"n_{i+2}={n_large} is not divisible by n_{i+1}={n_small}. "
                f"Ratio = {n_large / n_small:.4f}. "
                f"Valid next sizes from {n_small}: "
                f"{[n_small * r for r in [2, 3, 4, 5, 10]]}."
            )


def validate_m_layers_rennen(m_layers: list[int]) -> None:
    """
    Validate m_layers for Rennen et al. (2010): (n_{i+1}-1) % (n_i-1) == 0.
    """
    if len(m_layers) < 2:
        raise ValueError(
            f"m_layers must have at least 2 entries; got {len(m_layers)}"
        )
    for i in range(len(m_layers) - 1):
        n_small, n_large = m_layers[i], m_layers[i + 1]
        if n_large <= n_small:
            raise ValueError(
                f"m_layers must be strictly increasing; "
                f"m_layers[{i}]={n_small} >= m_layers[{i+1}]={n_large}"
            )
        if n_small < 2:
            raise ValueError(
                f"Layer sizes must be >= 2; m_layers[{i}]={n_small}"
            )
        num, den = n_large - 1, n_small - 1
        if num % den != 0:
            raise ValueError(
                f"Rennen divisibility violated between layers {i} and {i+1}: "
                f"(n_{i+2}-1)={num} is not divisible by (n_{i+1}-1)={den}. "
                f"Ratio = {num / den:.4f}. "
                f"Valid next sizes from {n_small}: "
                f"{[den * r + 1 for r in [2, 3, 4, 5, 10]]}."
            )


def validate_m_layers(m_layers: list[int]) -> None:
    """Alias for validate_m_layers_rennen (backward compatibility)."""
    validate_m_layers_rennen(m_layers)


def check_valid_lhd(
    X: np.ndarray, convention: str = "rennen", tol: float = 1e-10, n_full: int | None = None
) -> bool:
    """
    Verify that X is a valid LHD.

    convention="rennen": columns are permutations of {0, 1/(n-1), ..., 1}.
    convention="qian":   columns are permutations of {1/n, 2/n, ..., 1}.
    convention="midpoint": columns are permutations of {1/(2n), 3/(2n), ..., (2n-1)/(2n)}.
    convention="stratum": each column has one value per stratum [k/n, (k+1)/n), k=0..n-1
        (zero-indexed levels with (level+u)/n, u drawn once for full design).
    For qian/midpoint/stratum, if n_full is given, allow n distinct values from the full grid.
    """
    n, k = X.shape
    if n < 2:
        return True
    if convention == "rennen":
        expected = np.linspace(0.0, 1.0, n)
        for j in range(k):
            col = np.sort(X[:, j])
            if not np.allclose(col, expected, rtol=0, atol=tol):
                return False
        return True
    if convention == "midpoint":
        # Midpoint levels: (2*k-1)/(2*n) for k=1..n
        expected = (2 * np.arange(1, n + 1, dtype=np.float64) - 1) / (2 * n)
        if n_full is not None:
            levels_full = (2 * np.arange(1, n_full + 1, dtype=np.float64) - 1) / (2 * n_full)
            for j in range(k):
                col = X[:, j]
                # Round to nearest midpoint level on full grid: index i -> (2*i+1)/(2*n_full)
                idx = np.round((col * 2 * n_full - 1) / 2).astype(np.int64)
                idx = np.clip(idx, 0, n_full - 1)
                rounded = (2 * idx + 1) / (2 * n_full)
                uniq = np.unique(rounded)
                if len(uniq) != n:
                    return False
                for u in uniq:
                    if np.min(np.abs(levels_full - u)) > tol:
                        return False
            return True
        for j in range(k):
            col = np.sort(X[:, j])
            if not np.allclose(col, expected, rtol=0, atol=tol):
                return False
        return True
    if convention == "stratum":
        # Strata [k/n, (k+1)/n): each column has one value per stratum
        if n_full is not None:
            for j in range(k):
                col = X[:, j]
                stratum = np.floor(col * n_full).astype(np.int64)
                stratum = np.clip(stratum, 0, n_full - 1)
                uniq = np.unique(stratum)
                if len(uniq) != n:
                    return False
            return True
        for j in range(k):
            col = X[:, j]
            stratum = np.floor(col * n).astype(np.int64)
            stratum = np.clip(stratum, 0, n - 1)
            if not np.array_equal(np.sort(stratum), np.arange(n)):
                return False
        return True
    # qian
    if n_full is not None:
        # Nested layer: n distinct values from {1/n_full, ..., 1}
        levels_full = np.linspace(1.0 / n_full, 1.0, n_full)
        for j in range(k):
            col = X[:, j]
            rounded = np.round(col * n_full) / n_full
            rounded = np.clip(rounded, 1.0 / n_full, 1.0)
            uniq = np.unique(rounded)
            if len(uniq) != n:
                return False
            for u in uniq:
                if np.min(np.abs(levels_full - u)) > tol:
                    return False
        return True
    expected = np.linspace(1.0 / n, 1.0, n)
    for j in range(k):
        col = np.sort(X[:, j])
        if not np.allclose(col, expected, rtol=0, atol=tol):
            return False
    return True


def check_nested(
    X_inner: np.ndarray, X_outer: np.ndarray, tol: float = 1e-10
) -> bool:
    """Every row of X_inner must appear as a row of X_outer."""
    n_inner = X_inner.shape[0]
    n_outer = X_outer.shape[0]
    if n_inner > n_outer:
        return False
    for i in range(n_inner):
        row = X_inner[i]
        found = False
        for j in range(n_outer):
            if np.allclose(row, X_outer[j], rtol=0, atol=tol):
                found = True
                break
        if not found:
            return False
    return True


def _check_nested_midpoint(
    X_inner: np.ndarray, X_outer: np.ndarray, n_outer: int, tol: float = 1e-10
) -> bool:
    """
    For midpoint convention: the first n_inner rows of X_outer must be the
    re-embedding of X_inner onto the n_outer grid (same rule as extend_to_layer).
    So we recompute the re-embedding and check equality.
    """
    n_inner, k = X_inner.shape
    if n_inner > n_outer or n_inner > X_outer.shape[0]:
        return False
    c = (n_outer - 1) // (n_inner - 1) if n_inner > 1 else 1
    # Same as extend_to_layer: inner index from midpoint value, then outer index = index * c
    idx = np.round((X_inner * 2 * n_inner - 1) / 2).astype(np.int64)
    idx = np.clip(idx, 0, n_inner - 1)
    outer_idx = idx * c
    expected = (2 * outer_idx + 1) / (2 * n_outer)
    return np.allclose(expected, X_outer[:n_inner], rtol=0, atol=tol)


def validate_result(
    layers: list[np.ndarray],
    m_layers: list[int],
    k: int,
    convention: str = "rennen",
    nesting_check: str | None = None,
) -> None:
    """Run all post-construction checks. Raises AssertionError on first failure.

    For convention="midpoint", nesting_check can be:
    - "exact": layers[i] is the first m_layers[i] rows of layers[i+1] (Qian path).
    - "reembed": first m_layers[i] rows of layers[i+1] are re-embedding of layers[i] (Rennen path).
    Default "reembed" when convention is midpoint.
    """
    assert len(layers) == len(m_layers)
    n_full = m_layers[-1] if convention in ("qian", "midpoint", "stratum") else None
    for i, (layer, n) in enumerate(zip(layers, m_layers)):
        assert layer.shape == (n, k), f"Layer {i} shape mismatch"
        assert check_valid_lhd(
            layer, convention, n_full=n_full
        ), f"Layer {i} is not a valid LHD"
    for i in range(len(layers) - 1):
        if convention == "midpoint":
            if nesting_check == "exact":
                assert check_nested(
                    layers[i], layers[i + 1]
                ), f"Layer {i} is not a subset of layer {i+1}"
            else:
                assert _check_nested_midpoint(
                    layers[i], layers[i + 1], m_layers[i + 1]
                ), f"Layer {i} is not a subset of layer {i+1}"
        elif convention == "stratum":
            # Stratum: layers are leading rows of full design, exact subset
            assert check_nested(
                layers[i], layers[i + 1]
            ), f"Layer {i} is not a subset of layer {i+1}"
        else:
            assert check_nested(
                layers[i], layers[i + 1]
            ), f"Layer {i} is not a subset of layer {i+1}"
