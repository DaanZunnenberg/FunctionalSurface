"""Basis function library shared by both model families.

This module provides the three basis-related primitives used across the package:

``bernstein_basis``
    JIT-compiled Bernstein polynomial φ_k^M(u), the building block for the
    Bernstein-basis functional GARCH in ``garch.py``.  Indexed k = 1, …, M
    over the unit interval.

``cubic_bspline_basis``
    B-spline basis matrix Φ of shape (M, N), used in ``gas.py`` to
    parametrise the log-volatility curve as log σ_t(u) = Φ(u)ᵀ b_t.

``ou_kernel``
    Ornstein-Uhlenbeck covariance matrix Λ_δ of shape (N, N), used in
    ``gas.py`` as the intraday correlation structure.

Dependency note
───────────────
This module has no internal package dependencies.  It is imported by
``garch.py`` (via ``bernstein_basis``) and ``gas.py`` (via
``cubic_bspline_basis`` and ``ou_kernel``).
"""

import typing
import numpy as np
from numba import jit
from scipy.interpolate import BSpline


@jit(nopython=True)
def bernstein_basis(u: typing.Any, M: int, k: int) -> float:
    r"""Bernstein polynomial basis function.

    Computes :math:`\varphi_k^M(u) = \binom{M-1}{k-1} u^{k-1}(1-u)^{M-k}`.

    The full basis :math:`\{\varphi_k^M\}_{k=1}^M` forms a partition of unity
    on [0, 1] and is used in ``garch.py`` to approximate the level operator δ
    and the kernel operators α, β.

    JIT-compiled with Numba.  Accepts both scalar and array inputs for ``u``
    (Numba specialises on first call).  Uses the ``init`` accumulator pattern
    in the callers ``delta`` and ``kernel_operator`` to allow vectorised
    evaluation; see ``garch.py`` for details.

    Args:
        u: Evaluation point(s) in [0, 1].
        M: Total number of basis functions.
        k: Basis index in {1, …, M}.

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

    n = M - 1
    v = k - 1
    return comb(M - 1, k - 1) * (u ** v) * (1 - u) ** (n - v)


def cubic_bspline_basis(
    x: np.ndarray,
    order: int,
    n_interior_knots: int,
    create_constant: bool = True,
) -> np.ndarray:
    """Evaluate B-spline basis functions on a grid.

    Constructs a clamped (full-multiplicity boundary) B-spline basis of the
    given order on [0, 1] and evaluates each basis function at every point in
    ``x``.  Used in ``gas.py`` to build the (M, N) basis matrix that maps
    B-spline coefficients b_t to the log-volatility surface.

    Args:
        x: Evaluation grid in [0, 1], length N.
        order: Spline order (e.g. 4 for cubic splines).
        n_interior_knots: Number of interior knots, uniformly spaced.
        create_constant: If True, prepend a row of ones (intercept term).

    Returns:
        Array of shape (n_basis, N), where n_basis = n_interior_knots +
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
    This stationary exponential kernel captures short-range intraday
    autocorrelation with a single length-scale parameter δ (``ou_scale``
    in ``gas.py``).  Larger δ implies smoother intraday dependence.

    Used in ``gas.py`` as the covariance structure Λ_δ of the multivariate
    Student-t likelihood.  A custom kernel can be supplied via ``measure``.

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
