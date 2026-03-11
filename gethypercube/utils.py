"""General-purpose post-processing utilities for LHD designs."""

from __future__ import annotations

import numpy as np


def scale_LH(
    X: np.ndarray | list[np.ndarray],
    l_bounds: np.ndarray | list[float],
    u_bounds: np.ndarray | list[float],
    *,
    reverse: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """
    Scale a design (or list of nested-LHD layers) between the unit hypercube
    and physical bounds.

    Mirrors the behaviour of ``scipy.stats.qmc.scale`` without depending on
    scipy.  When ``reverse=False`` (the default) the design is mapped from
    ``[0, 1]^k`` to ``[l_bounds, u_bounds]``; when ``reverse=True`` the
    mapping is inverted so a design in physical units is returned to
    ``[0, 1]^k``.

    Parameters
    ----------
    X : np.ndarray of shape (n, k), or list of such arrays
        A single design matrix **or** a list of nested-LHD layers (as returned
        by ``nested_lhd``).  All arrays must share the same *k*.
        With ``reverse=False`` values must lie in ``[0, 1]^k``; with
        ``reverse=True`` values must lie within ``[l_bounds, u_bounds]``.
    l_bounds : array-like, length k
        Lower bounds of the physical space, one per dimension.
    u_bounds : array-like, length k
        Upper bounds of the physical space, one per dimension.
    reverse : bool, optional
        When *True*, invert the scaling (physical → unit cube).
        Default is False.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Scaled design(s) in the same form as the input: a single array when
        *X* is a single array, a list of arrays when *X* is a list.

    Raises
    ------
    ValueError
        If ``l_bounds`` or ``u_bounds`` are incompatible with the design
        dimensions, if any lower bound is not strictly less than the
        corresponding upper bound, or if the list contains arrays with
        inconsistent numbers of columns.

    Examples
    --------
    Scale a single layer:

    >>> import numpy as np
    >>> from gethypercube import nested_lhd, scale_LH
    >>> layers = nested_lhd(k=3, m_layers=[20, 40], seed=0)
    >>> X_phys = scale_LH(layers[-1], [0, 100, -1], [1, 200, 1])
    >>> X_back = scale_LH(X_phys, [0, 100, -1], [1, 200, 1], reverse=True)
    >>> np.allclose(layers[-1], X_back)
    True

    Scale all nested layers at once:

    >>> layers_phys = scale_LH(layers, [0, 100, -1], [1, 200, 1])
    >>> [arr.shape for arr in layers_phys]
    [(20, 3), (40, 3)]
    """
    is_list = isinstance(X, list)
    arrays = X if is_list else [X]

    # Validate all arrays are 2-D and agree on k
    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(
                f"Each layer must be a 2-D ndarray; "
                f"got {'list item ' + str(i) + ' with ' if is_list else ''}shape "
                f"{np.asarray(arr).shape}"
            )
    k = arrays[0].shape[1]
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape[1] != k:
            raise ValueError(
                f"All layers must have the same number of columns; "
                f"layer 0 has {k}, layer {i} has {arr.shape[1]}"
            )

    l = np.asarray(l_bounds, dtype=float).ravel()
    u = np.asarray(u_bounds, dtype=float).ravel()
    if l.shape != (k,):
        raise ValueError(
            f"l_bounds must have length {k} (one per dimension); got {l.shape[0]}"
        )
    if u.shape != (k,):
        raise ValueError(
            f"u_bounds must have length {k} (one per dimension); got {u.shape[0]}"
        )
    if np.any(l >= u):
        bad = np.where(l >= u)[0].tolist()
        raise ValueError(
            f"l_bounds must be strictly less than u_bounds; "
            f"violated at dimension(s) {bad}"
        )

    def _scale_one(arr: np.ndarray) -> np.ndarray:
        a = arr.astype(float)
        if reverse:
            return (a - l) / (u - l)
        return l + a * (u - l)

    scaled = [_scale_one(arr) for arr in arrays]
    return scaled if is_list else scaled[0]
