# gethypercube

**Sliced and nested Latin hypercube designs in Python.**

- **Sliced LHD** (`sliced_lhd`): maximin-distance Sliced Latin Hypercube Designs (Ba, Brenneman & Myers, 2015). When `t=1`, produces a standard maximin LHD.
- **Nested LHD** (`nested_lhd`, `nested_maximin_lhd`): Qian (2009) algebraic nested LHD and Rennen et al. (2010) ESE-optimised nested maximin LHD.

## Overview

A **Sliced Latin Hypercube Design (SLHD)** has `n = m × t` points, is a valid LHD overall, and partitions into `t` slices of `m` points each where every slice is also a valid LHD — useful for experiments with qualitative and quantitative factors.

**Nested LHDs** provide multiple layers of sizes `n_1 < n_2 < ...` where each layer is a valid LHD and smaller sets are subsets of larger ones. Use `nested_lhd` for Qian divisibility (`n_{i+1} % n_i == 0`) or `nested_maximin_lhd` for Rennen divisibility (`(n_{i+1}-1) % (n_i-1) == 0`) with ESE space-filling optimisation.

## Installation

```bash
pip install gethypercube
```

Or from source:

```bash
git clone https://github.com/kstoreyf/gethypercube
cd gethypercube
pip install -e ".[dev]"
```

## Quick Start

### Sliced LHD

```python
from gethypercube import sliced_lhd
import numpy as np

# Standard maximin LHD (t=1): 20 points in 3 dimensions
slices = sliced_lhd(t=1, m=20, k=3, optimization="sa", seed=42)
X = np.vstack(slices)  # shape (20, 3), values in [0, 1]

# Sliced LHD: 3 slices of 10 points each
slices = sliced_lhd(t=3, m=10, k=2, optimization="sa", seed=42)
# slices[i].shape == (10, 2); np.vstack(slices).shape == (30, 2)
```

### Nested LHD

```python
from gethypercube import nested_lhd, nested_maximin_lhd

# Qian (2009): n_{i+1} % n_i == 0
layers = nested_lhd(k=3, m_layers=[100, 200, 400, 800], seed=42)

# Rennen et al. (2010): (n_{i+1}-1) % (n_i-1) == 0, space-filling
layers = nested_maximin_lhd(k=3, m_layers=[2, 3, 5, 9], seed=42)
# layers[i] is a valid LHD; layers[0] ⊂ layers[1] ⊂ ... ⊂ layers[-1]
```

### Scaling to physical bounds

```python
from gethypercube import nested_lhd, scale_LH

layers = nested_lhd(k=2, m_layers=[10, 20], seed=0)
layers_phys = scale_LH(layers, l_bounds=[0, 100], u_bounds=[1, 200])
```

## API

### `sliced_lhd(t, m, k, ...)`

Generate a Sliced Latin Hypercube Design. Returns a list of `t` arrays each of shape `(m, k)` with values in `[0, 1]`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t` | — | Number of slices (`t=1` → standard LHD) |
| `m` | — | Points per slice (>= 2); total `n = m * t` |
| `k` | — | Number of dimensions (>= 1) |
| `optimization` | `None` | `None` for random construction, `'sa'` for simulated annealing |
| `power` | `15` | Exponent in average reciprocal distance criterion (SA only) |
| `nstarts` | `1` | Random restarts; best result returned (SA only) |
| `itermax` | `100` | Non-improving iterations before SA cools (SA only) |
| `total_iter` | `1_000_000` | Cap on total SA iterations (SA only) |
| `seed` | `None` | Random seed for reproducibility |
| `scramble` | `True` | Uniform jitter within each stratum; `False` for midpoints |

### `nested_lhd(k, m_layers, ...)`

Algebraic nested LHD construction (Qian 2009). Requires `n_{i+1} % n_i == 0`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | — | Number of dimensions (>= 1) |
| `m_layers` | — | Strictly increasing layer sizes (list or single int) |
| `seed` | `None` | Random seed |
| `scramble` | `True` | Uniform jitter within each stratum; `False` for midpoints |
| `optimization` | `None` | `'cd'` for centered-discrepancy hill climbing on complement rows |
| `n_iter` | `None` | Swap budget per complement step (`None` = auto-scale) |

Returns a list of arrays: `layers[i]` has shape `(m_layers[i], k)` with values in `[0, 1)`.

### `nested_maximin_lhd(k, m_layers, ...)`

ESE-optimised nested maximin LHD (Rennen et al. 2010). Requires `(n_{i+1}-1) % (n_i-1) == 0`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | — | Number of dimensions (>= 1) |
| `m_layers` | `None` | Layer sizes; or use `m_init`/`n_layers`/`ratio` instead |
| `m_init` | `None` | Smallest layer size (when `m_layers` not provided) |
| `n_layers` | `None` | Number of layers (when `m_layers` not provided) |
| `ratio` | `None` | Ratio between consecutive `(n-1)` values |
| `n_restarts` | `5` | ESE restarts per two-layer step |
| `max_outer_iters` | `200` | Max outer ESE iterations |
| `inner_iters_per_point` | `100` | Inner iters = this × n₂ × k |
| `seed` | `None` | Random seed |
| `scramble` | `True` | Uniform jitter within each stratum; `False` for midpoints |

Returns a list of arrays: `layers[i]` has shape `(m_layers[i], k)` with values in `[0, 1)`.

### `scale_LH(X, l_bounds, u_bounds, *, reverse=False)`

Scale a design (or list of nested layers) between the unit hypercube and physical bounds. `reverse=True` maps physical units back to `[0, 1]`.

### Utilities

| Function | Description |
|----------|-------------|
| `compute_phi(D, r)` | Average reciprocal inter-point distance (lower = better) |
| `lhs_degree(X)` | Fractional closeness to a perfect LHS (1.0 = perfect) |
| `ks_test_uniform(X)` | Per-dimension KS test against U(0, 1) |
| `is_valid_lhd(D, n)` | Check integer design is a valid LHD |
| `is_valid_slhd(D, t, m)` | Check integer design is a valid SLHD |
| `check_valid_lhd(X, convention)` | Verify continuous design is a valid LHD |
| `check_nested(X_inner, X_outer)` | Verify nesting (inner rows appear in outer) |
| `suggest_valid_layers_qian(n_start, n_max)` | Suggest valid `m_layers` for `nested_lhd` |
| `suggest_valid_layers_rennen(n_start, n_max)` | Suggest valid `m_layers` for `nested_maximin_lhd` |

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## References

- Ba, S., Brenneman, W. A. and Myers, W. R. (2015). "Optimal Sliced Latin Hypercube Designs." *Technometrics*, 57(4), 479–487.
- Qian, P. Z. G. (2009). "Nested Latin hypercube designs." *Biometrika*, 96(4), 957–970.
- Rennen, G., Husslage, B., van Dam, E. R., den Hertog, D. (2010). "Nested maximin Latin hypercube designs." *Structural and Multidisciplinary Optimization*, 41, 371–395.
