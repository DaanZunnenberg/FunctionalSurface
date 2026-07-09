"""Basis function library shared by both model families.

``bernstein_basis``
    JIT-compiled Bernstein polynomial φ_k^M(u), the building block for the
    Bernstein-basis functional GARCH in ``garch.py``.

``cubic_bspline_basis``
    B-spline basis matrix Φ of shape (n_basis, n_grid), used in ``gas.py`` to
    parametrise the log-volatility curve as log sigma_t(u) = Phi(u)^T b_t.

``ou_kernel``
    Ornstein-Uhlenbeck covariance matrix of shape (n_grid, n_grid), used in
    ``gas.py`` as the intraday correlation structure.
"""

from __future__ import annotations

import typing
import numpy as np
from numba import jit
from scipy.interpolate import BSpline


@jit(nopython=True)
def bernstein_basis(u: typing.Any, n_basis: int, k: int) -> float:
    r"""Bernstein polynomial basis function φ_k^M(u).

    Computes :math:`\varphi_k^M(u) = \binom{M-1}{k-1} u^{k-1}(1-u)^{M-k}`.

    JIT-compiled with Numba.  Accepts both scalar and array inputs for ``u``.
    Uses the ``init`` accumulator pattern in callers ``delta`` and
    ``kernel_operator`` to allow vectorised evaluation.

    Args:
        u: Evaluation point(s) in [0, 1].
        n_basis: Total number of basis functions (M).
        k: Basis index in {1, …, n_basis}.

    Returns:
        φ_k^M(u), same type/shape as ``u``.
    """
    def factorial(n):
        p = 1
        for i in range(1, n + 1):
            p *= i
        return p

    def comb(n, v):
        return factorial(n) / (factorial(v) * factorial(n - v))

    degree = n_basis - 1
    v = k - 1
    return comb(n_basis - 1, k - 1) * (u ** v) * (1 - u) ** (degree - v)


def cubic_bspline_basis(
    x: np.ndarray,
    order: int,
    n_interior_knots: int,
    create_constant: bool = True,
) -> np.ndarray:
    """Evaluate B-spline basis functions on a grid.

    Constructs a clamped (full-multiplicity boundary) B-spline basis of the
    given order on [0, 1] and evaluates each basis function at every point in
    ``x``.

    Args:
        x: Evaluation grid in [0, 1], length n_grid.
        order: Spline order (e.g. 4 for cubic splines).
        n_interior_knots: Number of interior knots, uniformly spaced.
        create_constant: If True, prepend a row of ones (intercept term).

    Returns:
        Array of shape (n_basis, n_grid), where n_basis = n_interior_knots +
        order - 1 (plus 1 if ``create_constant`` is True).
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
    """Ornstein-Uhlenbeck covariance kernel on a 1-D grid.

    Returns the N×N matrix K where K[i, j] = exp(-|u_i - u_j| / δ).

    Args:
        grid: 1-D evaluation grid, e.g. ``np.linspace(0, 1, n)``.
        delta: Length-scale parameter (larger → slower decay → smoother).
        measure: Optional replacement kernel; receives the (N, N) pairwise
                 difference matrix and must return an (N, N) array.

    Returns:
        Symmetric positive-definite matrix of shape (N, N).
    """
    if measure is None:
        measure = lambda diff: np.exp(-np.abs(diff) / delta)
    diff = grid[np.newaxis, :] - grid[:, np.newaxis]
    return measure(diff)
