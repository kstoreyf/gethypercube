# gethypercube

**Sliced and nested Latin hypercube designs in Python.**

- **Sliced LHD** (`gethypercube.sliced_lhd`): maximin-distance Sliced Latin Hypercube Designs (Ba, Brenneman & Myers, 2015). When `t=1`, standard maximin LHD.
- **Nested LHD** (`gethypercube.nested_lhd`): Qian (2009) algebraic nested LHD and Rennen et al. (2010) nested maximin LHD.

## Overview

A **Sliced Latin Hypercube Design (SLHD)** has `n = m × t` points, is a valid LHD overall, and partitions into `t` slices of `m` points each where every slice is also a valid LHD—useful for experiments with qualitative and quantitative factors.

**Nested LHDs** provide multiple layers of sizes `n_1 < n_2 < ...` where each layer is a valid LHD and smaller sets are subsets of larger ones (`nested_lhd` for Qian divisibility, `nested_maximin_lhd` for Rennen/ESE-optimised).

## Installation

```bash
pip install gethypercube
```

Or from source:

```bash
git clone https://github.com/your-org/gethypercube
cd gethypercube
pip install -e ".[dev]"
```

## Quick Start

**Sliced LHD (maximin SLHD):**

```python
from gethypercube import maximinSLHD

# Standard maximin LHD (t=1)
result = maximinSLHD(t=1, m=10, k=3, random_state=42)
print(result.design)        # integer design, shape (10, 3)
print(result.std_design)    # standardized to (0, 1)
print(result.measure)       # phi value (lower = better)

# Sliced LHD: 3 slices of 4 points each
result = maximinSLHD(t=3, m=4, k=2, random_state=42)
print(result.design)        # shape (12, 3): first col = slice label 1..3
```

**Nested LHD:**

```python
from gethypercube import nested_lhd, nested_maximin_lhd

# Qian (2009): n_{i+1} % n_i == 0, e.g. [2, 4, 8]
layers_qian = nested_lhd(k=2, m_layers=[2, 4, 8], seed=42)

# Rennen et al. (2010): (n_{i+1}-1) % (n_i-1) == 0, space-filling
layers_maximin = nested_maximin_lhd(k=2, m_layers=[2, 3, 5], seed=42)
```

## Package layout

- `gethypercube.sliced_lhd` — maximin SLHD: `maximinSLHD`, `SLHDResult`, `lhs_degree`, `ks_test_uniform`, `is_valid_lhd`, `is_valid_slhd`
- `gethypercube.nested_lhd` — nested designs: `nested_lhd` (Qian), `nested_maximin_lhd` (Rennen), `build_nested_lhd`, `extend_to_layer`, validation and layer helpers

Top-level imports: `from gethypercube import maximinSLHD, nested_lhd, nested_maximin_lhd, ...`

## API (sliced LHD)

### `maximinSLHD(t, m, k, power=15, nstarts=1, itermax=100, total_iter=1_000_000, random_state=None)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t` | — | Number of slices (`t=1` → standard LHD) |
| `m` | — | Points per slice; total `n = m * t` |
| `k` | — | Number of input dimensions |
| `power` | `15` | Exponent in average reciprocal distance criterion |
| `nstarts` | `1` | Random restarts; best result returned |
| `itermax` | `100` | Non-improving iterations before SA cools |
| `total_iter` | `1_000_000` | Cap on total SA iterations |
| `random_state` | `None` | Seed for reproducibility |

Returns `SLHDResult` with `design`, `std_design`, `measure`, `temp0`, `n_slices`, `n_per_slice`, `n_dims`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## References

- Ba, S., Brenneman, W. A. and Myers, W. R. (2015). "Optimal Sliced Latin Hypercube Designs." *Technometrics*, 57(4), 479–487.
- Qian, P. Z. G. (2009). "Nested Latin hypercube designs." *Biometrika*, 96(4), 957–970.
- Rennen, G., Husslage, B., van Dam, E. R., den Hertog, D. (2010). "Nested maximin Latin hypercube designs." *Structural and Multidisciplinary Optimization*, 41, 371–395.
