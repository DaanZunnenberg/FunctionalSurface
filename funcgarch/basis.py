"""Basis function implementations for the functional GARCH and GAS models."""

import typing
import numpy as np
from numba import jit
from scipy.interpolate import BSpline


@jit(nopython=True)
def bernstein_basis(u: typing.Any, M: int, k: int) -> float:
    r"""Bernstein polynomial basis: $\varphi_k^M(u) = \binom{M-1}{k-1} u^{k-1}(1-u)^{M-k}$.

    Args:
        u: Evaluation point(s) in [0, 1].
        M: Total number of basis functions.
        k: Basis index in {1, ..., M}.
    """
    def factorial(n):
        p = 1
        for i in range(1, n + 1):
            p *= i
        return p

    def comb(n, v):
        return factorial(n) / (factorial(v) * factorial(n - v))

    n = M - 1
    v = k - 1
    mul = comb(M - 1, k - 1)
    return mul * (u ** v) * (1 - u) ** (n - v)


def cubic_bspline_basis(
    x: np.ndarray,
    order: int,
    n_interior_knots: int,
    create_constant: bool = True,
) -> np.ndarray:
    """Evaluate B-spline basis functions on grid x.

    Args:
        x: Evaluation grid in [0, 1].
        order: Spline order (e.g. 4 for cubic splines).
        n_interior_knots: Number of interior knots.
        create_constant: If True, prepend an intercept column of ones.

    Returns:
        Array of shape (n_basis, len(x)).
    """
    knots = np.concatenate(([0] * order, np.linspace(0, 1, n_interior_knots), [1] * order))
    n_basis = len(knots) - order - 1
    basis_fns = [BSpline(knots, np.eye(n_basis)[i], order) for i in range(n_basis)]
    if create_constant:
        return np.array([[1.0] * x.size] + [b(x) for b in basis_fns])
    return np.array([b(x) for b in basis_fns])


def ou_kernel(
    grid: np.ndarray,
    delta: float = 1.0,
    measure: typing.Callable | None = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck covariance kernel on a 1D grid.

    Returns the n×n matrix K where K[i,j] = measure(grid[j] - grid[i]).
    The default measure is the exponential decay exp(-|s - t| / delta).

    Args:
        grid: 1D evaluation grid, e.g. np.linspace(0, 1, n).
        delta: Length-scale parameter (larger = smoother).
        measure: Custom kernel function; receives the pairwise difference matrix.
    """
    if measure is None:
        measure = lambda diff: np.exp(-np.abs(diff) / delta)
    diff = grid[np.newaxis, :] - grid[:, np.newaxis]
    return measure(diff)
