"""Random construction of valid SLHD designs."""

import numpy as np


def random_slhd(t: int, m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random valid SLHD design matrix (Ba et al. 2015, Section 2).

    Slice-wise construction:
    - Step 1: Generate t small LHDs X_1,...,X_t, each m x k with levels 1..m
    - Step 2: In each column, replace the t entries of level l with a random
      permutation of {(l-1)t+1, ..., lt}

    Result: each slice has one value from each bin {1..t}, {t+1..2t}, ...,
    {(m-1)t+1..mt}, so each slice spans 1..n. Full design is LHD; each slice
    satisfies ceil(X_c/t) is an m-level LHD.

    Parameters
    ----------
    t : int
        Number of slices.
    m : int
        Points per slice.
    k : int
        Number of dimensions.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    D : np.ndarray, shape (n, k), dtype int
        Integer design matrix (1-indexed values 1..n).
    """
    n = m * t
    D = np.zeros((n, k), dtype=np.int64)

    # Step 1: Generate t small LHDs with levels 1..m
    X_small = []
    for s in range(t):
        X = np.empty((m, k), dtype=np.int64)
        for j in range(k):
            X[:, j] = rng.permutation(np.arange(1, m + 1, dtype=np.int64))
        X_small.append(X)

    # Step 2: For each column j, for each level l in 1..m:
    #   Find t positions (one per slice) where X_small[s][row,j] == l
    #   Assign them a random permutation of {(l-1)t+1, ..., lt}
    for j in range(k):
        for l in range(1, m + 1):
            indices = []
            for s in range(t):
                for row in range(m):
                    if X_small[s][row, j] == l:
                        indices.append(s * m + row)
                        break
            perm = rng.permutation(
                np.arange((l - 1) * t + 1, l * t + 1, dtype=np.int64)
            )
            for i, global_row in enumerate(indices):
                D[global_row, j] = perm[i]

    return D

