# slhd

**Maximin-distance Sliced Latin Hypercube Designs in Python.**

A Python reimplementation of the R package [`SLHD`](https://cran.r-project.org/package=SLHD) (Ba, Brenneman & Myers, 2015).

## Overview

A **Sliced Latin Hypercube Design (SLHD)** is a space-filling design of experiments where:

- The full design of `n = m × t` points is a valid Latin Hypercube Design (LHD)
- It can be partitioned into `t` **slices** of `m` points each, where every slice is also a valid LHD

This is useful for computer experiments with a mix of quantitative and qualitative factors. Each slice provides space-filling coverage for one level of the qualitative factor.

When `t=1`, the result is a standard maximin-distance LHD.

## Installation

```bash
pip install slicedlhd
```

Or from source:

```bash
git clone https://github.com/your-org/slhd
cd slhd
pip install -e ".[dev]"
```

## Quick Start

```python
from slicedlhd import maximinSLHD

# Standard maximin LHD (t=1)
result = maximinSLHD(t=1, m=10, k=3, random_state=42)
print(result.design)        # integer design, shape (10, 3)
print(result.std_design)    # standardized to (0, 1), shape (10, 3)
print(result.measure)       # phi value (lower = better)

# Sliced LHD: 3 slices of 4 points each, 2 dimensions
result = maximinSLHD(t=3, m=4, k=2, random_state=42)
print(result.design)        # shape (12, 3): first col = slice label 1..3
print(result.std_design)    # shape (12, 3)
```

## API

### `maximinSLHD(t, m, k, power=15, nstarts=1, itermax=100, total_iter=1_000_000, random_state=None)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t` | — | Number of slices (`t=1` → standard LHD) |
| `m` | — | Points per slice; total `n = m * t` |
| `k` | — | Number of input dimensions |
| `power` | `15` | Exponent `r` in the average reciprocal distance criterion |
| `nstarts` | `1` | Independent random restarts; best result is returned |
| `itermax` | `100` | Non-improving iterations before SA temperature cools |
| `total_iter` | `1_000_000` | Hard cap on total SA iterations |
| `random_state` | `None` | Integer seed for reproducibility |

Returns an `SLHDResult` dataclass with fields `design`, `std_design`, `measure`, `temp0`, `n_slices`, `n_per_slice`, `n_dims`.

## Algorithm

Uses simulated annealing with:
- **Within-slice column swaps** to maintain the SLHD constraint at every step
- **Incremental distance updates** (`O(nk)` per step vs `O(n²k)` for full recompute)
- **Average reciprocal distance** as the smooth surrogate for the maximin criterion

See [Ba et al. (2015)](https://doi.org/10.1080/00401706.2014.957867) for details.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Reference

Ba, S., Brenneman, W. A. and Myers, W. R. (2015). "Optimal Sliced Latin Hypercube Designs." *Technometrics*, 57(4), 479–487.
