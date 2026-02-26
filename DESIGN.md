# Nested Latin Hypercube Design — Implementation Specification

Package layout: **gethypercube** with subpackages `gethypercube.sliced_lhd` (maximin SLHD) and `gethypercube.nested_lhd` (Qian + Rennen nested LHD).

## Overview

Implement two multi-layer nested Latin hypercube design (NLHD) generators:

| Function | Algorithm | Divisibility constraint | Space-filling | Use when |
|---|---|---|---|---|
| `nested_lhd` | Qian (2009) | `n_{i+1}` divisible by `n_i` | Moderate (algebraic) | Round `n` values matter, e.g. `[1000, 2000, 4000]` |
| `nested_maximin_lhd` | Rennen et al. (2010) | `(n_{i+1}-1)` divisible by `(n_i-1)` | High (ESE-optimised) | Best space-filling, e.g. `[5, 9, 17, 33]` |

Both functions accept a number of dimensions `k` and an ordered list of layer sizes `m_layers = [n_1, n_2, ..., n_L]` where `n_1 < n_2 < ... < n_L`. Both return `L` nested point sets where each is a valid Latin hypercube design and each smaller set is a strict subset of all larger ones.

---

## Background: What is a Nested LHD?

A Latin hypercube design (LHD) of `n` points in `k` dimensions is an `n × k` matrix where each column is a permutation of equispaced levels. No two points share any coordinate value in any dimension.

A **nested LHD** with layers `X_1 ⊂ X_2` requires that both `X_1` and `X_2` are individually valid LHDs. The two algorithms use different grid conventions, which leads to different divisibility constraints:

**Qian convention** — levels are `{1/n, 2/n, ..., 1}`. For `X_1`'s levels to be a subset of `X_2`'s levels, `n_2` must be a multiple of `n_1`.

**Rennen convention** — levels are `{0, 1/(n-1), 2/(n-1), ..., 1}`. For `X_1`'s grid to be a sub-grid of `X_2`'s, `(n_2 - 1)` must be a multiple of `(n_1 - 1)`.

These are genuinely different constraints and accept different size sequences. For example `[2, 4, 8, 16]` satisfies Qian (each divides the next) but fails Rennen (`7/3` is not an integer). Conversely `[2, 3, 5, 9]` satisfies Rennen (all `(n-1)` ratios equal 2) but fails Qian (`3/2` is not an integer).

The multi-layer extension requires pairwise divisibility at every consecutive pair.

---

## Algorithm 1: `nested_lhd` — Qian (2009)

Based on Qian, P.Z.G. (2009). *Nested Latin hypercube designs*. Biometrika, 96(4), 957–970.

### Divisibility Constraint

For all consecutive pairs: `n_{i+1} % n_i == 0`.

### Grid Convention

Levels are `{1/n, 2/n, ..., 1}`. Internally work in integers `{1, 2, ..., n}` and divide by `n` only in the final output.

### Two-Layer Construction

Given `n_1` and `n_2 = c * n_1` (integer `c >= 2`):

**Step 1 — Build X_1:**
Generate an LHD of `n_1` points in `k` dimensions on integer levels `{1, ..., n_1}`. For each column, take a random permutation of `{1, ..., n_1}`.

**Step 2 — Expand X_1 to the fine grid:**
Each level `v ∈ {1, ..., n_1}` in `X_1` corresponds to a block of `c` consecutive fine-grid levels: `{(v-1)*c + 1, ..., v*c}`. For each row `i` of `X_1` and each column `j`, replace the value `v` with one value drawn from its block. To preserve the LHD property of `X_2`, the chosen values across all `n_1` rows in column `j` must all come from different blocks (guaranteed since `X_1` is an LHD) and each block contributes exactly one value. Use a random offset `u_j ∈ {1, ..., c}` per column, applied uniformly: the expanded value for level `v` in column `j` is `(v-1)*c + u_j`. This ensures the `n_1` expanded values in column `j` occupy `n_1` distinct positions, one per block.

**Step 3 — Fill the complement:**
For each column `j`, the `n_1` expanded inner values occupy one slot per block. The remaining `c - 1` slots per block must be filled by the `n_2 - n_1` complement rows. Concretely:
- Available levels in column `j`: all levels in `{1, ..., n_2}` not chosen in Step 2.
- Randomly permute these `n_2 - n_1` available levels and assign as column `j` of the complement rows.

**Step 4 — Stack:**
Concatenate the `n_1` expanded inner rows and `n_2 - n_1` complement rows to form `X_2` of shape `(n_2, k)`. The first `n_1` rows are the embedding of `X_1`.

### Multi-Layer Recursive Construction

Apply the two-layer construction recursively, bottom-up:

```
X_1 ← random LHD of n_1 points on {1, ..., n_1}
X_2 ← expand(X_1, c_1 = n_2 / n_1)        # n_2 × k, X_1 rows embedded within
X_3 ← expand(X_2, c_2 = n_3 / n_2)        # n_3 × k, contains X_2 which contains X_1
...
X_L ← expand(X_{L-1}, c_{L-1} = n_L / n_{L-1})
```

At each step, the previous layer is the inner design. The expand function places each of its points into one slot of a `c`-sized fine-grid block and fills the remaining slots with new complement points.

**Note:** Qian's construction is deterministic and algebraic — there is no iterative optimisation. The designs are valid nested LHDs but are not maximin-optimised.

### Optional Post-Hoc Improvement

After Qian construction, optionally run a complement-only ESE pass (see `extend_to_layer` in the Rennen section) to improve space-filling without disturbing the nested structure. Expose via `optimise=False` (default off, to keep `nested_lhd` fast and deterministic).

---

## Algorithm 2: `nested_maximin_lhd` — Rennen et al. (2010)

Based on Rennen, G., Husslage, B.G.M., van Dam, E.R., den Hertog, D. (2010). *Nested maximin Latin hypercube designs*. Structural and Multidisciplinary Optimization, 41, 371–395.

### Divisibility Constraint

For all consecutive pairs: `(n_{i+1} - 1) % (n_i - 1) == 0`.

### Grid Convention

Levels are `{0, 1/(n-1), ..., 1}`. Internally work in integers `{0, 1, ..., n-1}` and divide by `(n-1)` only in the final output.

### Grid Structure

When `c = (n_2 - 1) / (n_1 - 1)` is an integer, the **nested n_2-grid** is used:
- All points are placed on the fine integer grid `{0, 1, ..., n_2-1}`.
- The `n_1` inner points occupy every `c`-th level: `{0, c, 2c, ..., (n_1-1)*c}` in each column.
- The `n_2 - n_1` outer complement points occupy the remaining `n_2 - n_1` levels.

This guarantees both layers are valid LHDs simultaneously.

### Two-Layer Core Primitive

#### Data Representation

- `X2`: `n_2 × k` integer matrix, each column a permutation of `{0, ..., n_2-1}`.
- `I1`: set of `n_1` row indices — the rows of `X2` forming `X_1`.
- Constraint: for each column `j`, `{X2[i,j] : i in I1}` must equal `{0, c, 2c, ..., (n_1-1)*c}`.

#### Distance Objective

```
s_j = 1 / (k * (n_j - 1))^(1/k)       # scaling factor for layer j

d_j = min_{p != q in X_j}  ||X_j[p] - X_j[q]||_2 / s_j

d = min(d_1, d_2)                        # objective to maximise
```

#### Initialisation

For each column `j`:
- Randomly permute `{0, c, 2c, ..., (n_1-1)*c}` and assign to the `I1` rows.
- Randomly permute the remaining `n_2 - n_1` levels and assign to complement rows.

Optionally use a diagonal start: `X2[i, j] = (i * c) mod (n_2 - 1)` for inner rows.

#### GROUPRAND Inner Loop

At each inner iteration, randomly choose POINT exchange (p=0.5) or GROUP exchange (p=0.5):

**POINT exchange:**
- Pick random point `p` from `X_2`.
- If `p ∈ I1`: pick random `q ∈ I1`, pick random column `j`, swap `X2[p,j]` and `X2[q,j]`.
- If `p ∉ I1`: pick random `q ∉ I1`, pick random column `j`, swap `X2[p,j]` and `X2[q,j]`.
- Swapping within the same stratum always preserves the nested LHD structure.

**GROUP exchange** (only when `c >= 2`):
- Pick random column `j`.
- The `n_1` inner coordinate values in column `j` divide `{0, ..., n_2-1}` into `n_1 - 1` gaps, each containing exactly `c - 1` complement levels. These are the "groups".
- Pick two groups at random. Swap all complement points in group A with all complement points in group B, exchanging their column-`j` coordinate values. Equal group sizes guarantee the column remains a valid permutation.
- When `c = 1` all groups are empty — skip group exchange entirely.

#### ESE Outer Loop

```
threshold ← 0.005 * d_initial
no_improve_count ← 0
best_d ← d_initial
best_design ← copy(X2, I1)

for outer_iter in 1..MAX_OUTER:
    improved_in_inner ← False

    for inner_iter in 1..(inner_iters_per_point * n_2 * k):
        candidate ← grouprand_move(X2, I1)
        delta ← d(candidate) - d(X2, I1)

        if delta >= -threshold:
            X2, I1 ← candidate
            if delta > 0:
                improved_in_inner ← True
                if d(X2, I1) > best_d:
                    best_d ← d(X2, I1)
                    best_design ← copy(X2, I1)

    if improved_in_inner:
        threshold ← threshold * 0.9          # intensify: exploit good region
    else:
        no_improve_count += 1
        if no_improve_count > PATIENCE:       # e.g. PATIENCE = 5
            break
        threshold ← threshold * 1.1          # diversify: escape local optimum

return best_design
```

Also run **POINTRAND** (identical but always POINT exchange, never GROUP). Return whichever of GROUPRAND or POINTRAND achieves the higher `d` across all restarts.

### Multi-Layer Recursive Construction

```
X_1, X_2 ← two_layer_maximin(k, n_1=m_layers[0], n_2=m_layers[1])
layers ← [X_1, X_2]

for i in 2..L-1:
    X_new ← extend_to_layer(X_inner=layers[i-1], n_large=m_layers[i], k=k)
    layers.append(X_new)

return layers
```

#### `extend_to_layer(X_inner, n_large, k)`

Given a fixed `n_small × k` inner design (already on integer scale `{0, ..., n_small-1}`), extend to `n_large` points:

**Step A — Re-embed on fine grid:**
`c = (n_large - 1) / (n_small - 1)`. Map each value `v` in `X_inner` to `v * c`. The inner points now sit on every `c`-th level of `{0, ..., n_large-1}`.

**Step B — Initialise complement:**
For each column `j`, the available levels are `{0, ..., n_large-1} \ {v*c : v in X_inner[:,j]}`. Randomly permute these and assign to the `n_large - n_small` new complement rows.

**Step C — ESE on complement only:**
Run GROUPRAND with inner rows **fixed**. Only complement rows participate in POINT and GROUP exchanges. The objective is `d = min(d_inner_fixed, d_outer)`. Since `d_inner` is constant, this effectively maximises `d_outer` down to the floor set by `d_inner`.

---

## API Specification

```python
def nested_lhd(
    k: int,
    m_layers: list[int],
    seed: int | None = None,
    optimise: bool = False,
) -> list[np.ndarray]:
    """
    Construct a multi-layer nested Latin hypercube design using Qian (2009).

    Divisibility constraint: n_{i+1} % n_i == 0 for all consecutive pairs.

    Parameters
    ----------
    k : int
        Number of dimensions. Must be >= 1.
    m_layers : list[int]
        Strictly increasing list of layer sizes [n_1, n_2, ..., n_L].
        Each n_{i+1} must be an exact multiple of n_i.
        Minimum 2 layers required.
    seed : int or None
        Random seed for reproducibility.
    optimise : bool
        If True, apply a post-hoc complement-only ESE pass to improve
        space-filling. Default False (pure algebraic construction).

    Returns
    -------
    list of np.ndarray
        layers[i] is an m_layers[i] × k float64 array.
        Levels follow Qian's convention: {1/n, 2/n, ..., 1}.
        layers[0] ⊂ layers[1] ⊂ ... ⊂ layers[L-1] (row subset relationship).
        Each layers[i] is a valid LHD.

    Raises
    ------
    ValueError
        If m_layers violates the constraint n_{i+1} % n_i == 0.

    Examples
    --------
    # Round multiples — the primary use case for this function
    layers = nested_lhd(k=5, m_layers=[1000, 2000, 4000, 8000])

    layers = nested_lhd(k=3, m_layers=[10, 30, 90])
    # 30%10==0 ✓, 90%30==0 ✓

    # Will raise ValueError:
    # nested_lhd(k=3, m_layers=[10, 20, 30])
    # 30 % 20 = 10 != 0  ✗
    """
    ...


def nested_maximin_lhd(
    k: int,
    m_layers: list[int],
    n_restarts: int = 5,
    max_outer_iters: int = 200,
    inner_iters_per_point: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Construct a multi-layer nested maximin Latin hypercube design using
    Rennen et al. (2010). Maximises the minimum pairwise distance subject
    to the nested LHD structure.

    Divisibility constraint: (n_{i+1}-1) % (n_i-1) == 0 for all consecutive pairs.

    Parameters
    ----------
    k : int
        Number of dimensions. Must be >= 1.
    m_layers : list[int]
        Strictly increasing list of layer sizes [n_1, n_2, ..., n_L].
        Each (n_{i+1} - 1) must be divisible by (n_i - 1).
        Minimum 2 layers required.
    n_restarts : int
        Number of independent ESE restarts per two-layer step.
        Best result across restarts is returned. Default 5.
    max_outer_iters : int
        Maximum outer ESE iterations per restart. Default 200.
    inner_iters_per_point : int
        Inner ESE iterations = inner_iters_per_point * n_2 * k. Default 100.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of np.ndarray
        layers[i] is an m_layers[i] × k float64 array.
        Levels follow Rennen's convention: {0, 1/(n-1), ..., 1}.
        layers[0] ⊂ layers[1] ⊂ ... ⊂ layers[L-1] (row subset relationship).
        Each layers[i] is a valid LHD.

    Raises
    ------
    ValueError
        If m_layers violates the constraint (n_{i+1}-1) % (n_i-1) == 0.

    Examples
    --------
    # Ratio-2 sequence (n-1 values double each step)
    layers = nested_maximin_lhd(k=3, m_layers=[2, 3, 5, 9, 17, 33])

    layers = nested_maximin_lhd(k=5, m_layers=[5, 9, 17])
    # (9-1)%(5-1)==0 ✓, (17-1)%(9-1)==0 ✓

    # Will raise ValueError:
    # nested_maximin_lhd(k=3, m_layers=[2, 4, 8, 16])
    # (8-1)%(4-1) = 7%3 = 1 != 0  ✗
    """
    ...
```

---

## Validation

### Input validation

```python
def validate_m_layers_qian(m_layers: list[int]) -> None:
    if len(m_layers) < 2:
        raise ValueError(f"m_layers must have at least 2 entries; got {len(m_layers)}")
    for i in range(len(m_layers) - 1):
        n_small, n_large = m_layers[i], m_layers[i + 1]
        if n_large <= n_small:
            raise ValueError(
                f"m_layers must be strictly increasing; "
                f"m_layers[{i}]={n_small} >= m_layers[{i+1}]={n_large}"
            )
        if n_small < 1:
            raise ValueError(f"Layer sizes must be >= 1; m_layers[{i}]={n_small}")
        if n_large % n_small != 0:
            raise ValueError(
                f"Qian divisibility violated between layers {i} and {i+1}: "
                f"n_{i+2}={n_large} is not divisible by n_{i+1}={n_small}. "
                f"Ratio = {n_large / n_small:.4f}. "
                f"Valid next sizes from {n_small}: "
                f"{[n_small*r for r in [2,3,4,5,10]]}."
            )


def validate_m_layers_rennen(m_layers: list[int]) -> None:
    if len(m_layers) < 2:
        raise ValueError(f"m_layers must have at least 2 entries; got {len(m_layers)}")
    for i in range(len(m_layers) - 1):
        n_small, n_large = m_layers[i], m_layers[i + 1]
        if n_large <= n_small:
            raise ValueError(
                f"m_layers must be strictly increasing; "
                f"m_layers[{i}]={n_small} >= m_layers[{i+1}]={n_large}"
            )
        if n_small < 2:
            raise ValueError(f"Layer sizes must be >= 2; m_layers[{i}]={n_small}")
        num, den = n_large - 1, n_small - 1
        if num % den != 0:
            raise ValueError(
                f"Rennen divisibility violated between layers {i} and {i+1}: "
                f"(n_{i+2}-1)={num} is not divisible by (n_{i+1}-1)={den}. "
                f"Ratio = {num / den:.4f}. "
                f"Valid next sizes from {n_small}: "
                f"{[den*r + 1 for r in [2, 3, 4, 5, 10]]}."
            )
```

### Post-construction checks

```python
def check_valid_lhd(X: np.ndarray, convention: str = "rennen") -> bool:
    """
    Verify X is a valid LHD.
    convention="rennen": columns are permutations of {0, 1/(n-1), ..., 1}
    convention="qian":   columns are permutations of {1/n, 2/n, ..., 1}
    Check to floating-point tolerance (1e-10).
    """
    ...


def check_nested(X_inner: np.ndarray, X_outer: np.ndarray, tol: float = 1e-10) -> bool:
    """Every row of X_inner must appear as a row of X_outer."""
    ...


def validate_result(
    layers: list[np.ndarray],
    m_layers: list[int],
    k: int,
    convention: str,
) -> None:
    assert len(layers) == len(m_layers)
    for i, (layer, n) in enumerate(zip(layers, m_layers)):
        assert layer.shape == (n, k), f"Layer {i} shape mismatch"
        assert check_valid_lhd(layer, convention), f"Layer {i} is not a valid LHD"
    for i in range(len(layers) - 1):
        assert check_nested(layers[i], layers[i + 1]), \
            f"Layer {i} is not a subset of layer {i+1}"
```

---

## Helper: Valid Layer Size Suggestions

```python
def suggest_valid_layers_qian(
    n_start: int,
    n_max: int,
    ratios: list[int] = [2, 3, 4, 5, 10],
) -> list[list[int]]:
    """
    Return valid m_layers sequences for nested_lhd (Qian) starting at n_start.

    Pattern: n_i = n_start * r^(i-1) for constant integer ratio r.

    Examples from n_start=1000:
        r=2:  [1000, 2000, 4000, 8000, 16000]
        r=3:  [1000, 3000, 9000]
        r=5:  [1000, 5000]
        r=10: [1000, 10000]
    """
    ...


def suggest_valid_layers_rennen(
    n_start: int,
    n_max: int,
    ratios: list[int] = [2, 3, 4, 5],
) -> list[list[int]]:
    """
    Return valid m_layers sequences for nested_maximin_lhd (Rennen) starting at n_start.

    Pattern: (n_i - 1) = (n_start - 1) * r^(i-1), i.e. n_i = (n_start-1)*r^(i-1) + 1.

    Examples from n_start=5 (n-1=4):
        r=2: [5, 9, 17, 33, 65, 129]
        r=3: [5, 13, 37, 109]

    Examples from n_start=11 (n-1=10):
        r=2: [11, 21, 41, 81, 161]
        r=10: [11, 101, 1001]

    Note: for round n values (multiples of 10, 100, etc.), use nested_lhd instead.
    """
    ...
```

---

## File Structure

```
nested_lhd/
├── __init__.py              # public exports: nested_lhd, nested_maximin_lhd,
│                            #   suggest_valid_layers_qian, suggest_valid_layers_rennen
├── validate.py              # validate_m_layers_qian, validate_m_layers_rennen,
│                            #   check_valid_lhd, check_nested, validate_result
├── qian.py                  # nested_lhd — Qian (2009) algebraic construction
├── ese.py                   # ESE algorithm: GROUPRAND + POINTRAND (shared by both)
├── rennen.py                # nested_maximin_lhd — Rennen (2010), uses ese.py
├── utils.py                 # distance computation, grid utilities,
│                            #   suggest_valid_layers_qian/rennen
└── tests/
    ├── test_validate.py     # divisibility checks, error messages for both
    ├── test_qian.py         # nested_lhd correctness, nesting, LHD validity
    ├── test_rennen.py       # nested_maximin_lhd correctness, nesting, LHD validity
    └── test_shared.py       # check_valid_lhd, check_nested, both conventions
```

---

## Implementation Notes

### Grid convention difference in output

The two functions return coordinates on different scales:
- `nested_lhd`: values in `(0, 1]`, levels `{1/n, 2/n, ..., 1}`.
- `nested_maximin_lhd`: values in `[0, 1]`, levels `{0, 1/(n-1), ..., 1}`.

These are not interchangeable. Document this clearly. If downstream code needs a consistent convention, add a `rescale` utility, but do not silently transform the output.

### Distance computation (Rennen only)

Work in integer coordinates `{0, ..., n-1}` throughout the ESE inner loop. Use `scipy.spatial.distance.cdist` for the initial pairwise distance matrix, then update incrementally: a swap of rows `p` and `q` in column `j` only changes the `2*(n-1)` pairwise distances involving `p` or `q`. Recompute only those rather than the full `n*(n-1)/2` matrix.

### Scaling (Rennen only)

The scaled separation distance uses `s_j = 1 / (k * (n_j - 1))^(1/k)` from Rennen eq.(1). When comparing designs of different sizes, this normalisation makes the distances comparable. Work in raw integer coordinates internally; apply the scale only when evaluating the objective.

### Convergence and computational cost (Rennen only)

For small designs (`n_2 <= 50`, `k <= 5`) ESE typically converges within 50–100 outer iterations. For `n > 100` increase `n_restarts` to 10 or more. For `n > 500` Rennen's approach becomes computationally expensive and `nested_lhd(optimise=True)` is the better choice.

### Seeding

Both functions accept `seed`. Pass to `np.random.default_rng(seed)` and derive per-restart child seeds via `rng.integers(0, 2**31)`.

---

## Choosing Between the Two Functions

```
Need round n values (e.g. 1000, 2000, 4000)?
    → nested_lhd (Qian)

n values can be flexible?
    → nested_maximin_lhd (Rennen) for best space-filling

n > 500 and space-filling still matters?
    → nested_lhd(optimise=True)

Unsure which sizes are valid?
    → call suggest_valid_layers_qian() or suggest_valid_layers_rennen() first
```

---

## References

Qian, P.Z.G. (2009). Nested Latin hypercube designs. *Biometrika*, 96(4), 957–970.

Rennen, G., Husslage, B.G.M., van Dam, E.R., den Hertog, D. (2010). Nested maximin Latin hypercube designs. *Structural and Multidisciplinary Optimization*, 41, 371–395. https://doi.org/10.1007/s00158-009-0432-y

Pre-computed Rennen designs for dimensions up to 10: http://www.spacefillingdesigns.nl