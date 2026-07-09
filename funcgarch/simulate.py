"""Simulation utilities for the functional GARCH model.

Generates synthetic (returns, variance_surface) pairs from a functional
GARCH(1,1) process.  Used for Monte Carlo studies and to verify that
``fit()`` recovers known parameters.

The GARCH recursion here is identical to the one in ``garch_estimator()``, so
any ``params`` vector valid for ``fit()`` can be passed directly to
``simulate()``.
"""

import warnings
import typing

import numpy as np
from numba import njit

from .garch import delta, kernel_operator

warnings.filterwarnings(action='ignore')


@njit
def brownian(u: np.ndarray) -> np.ndarray:
    """Scaled Brownian motion: $W(u) = 2^{400u} / \\ln 2$."""
    return (2 ** (400 * u)) / np.log(2)


def _brownian_noise(
    std_dev: np.ndarray,
    bm_incr: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Intraday noise path: cumulative sum of scaled normal increments."""
    return np.sqrt(1 / std_dev) * np.cumsum(np.sqrt(bm_incr) * eta)


def simulate(
    shape: tuple[int, int],
    n_basis: int,
    params: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a functional GARCH(1,1) process.

    Args:
        shape: Tuple (n_grid, n_days).
        n_basis: Number of Bernstein basis functions.
        params: Parameter vector [delta_coefs (n_basis) | alpha_coefs (n_basis²) | beta_coefs (n_basis²)].
        delta_fn: Level operator (default: delta).
        kernel_fn: Kernel operator (default: kernel_operator).

    Returns:
        Tuple (returns, variance_surface), each of shape (n_grid, n_days).
    """
    n_grid, n_days = shape
    grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)

    delta_coefs = params[:n_basis]
    alpha_coefs = params[n_basis: n_basis + n_basis ** 2]
    beta_coefs  = params[n_basis + n_basis ** 2:]

    variance         = np.ones(n_grid)
    returns          = np.ones((n_grid, n_days))
    variance_surface = np.zeros((n_grid, n_days))
    variance_surface[:, 0] = variance

    delta_hat = delta_fn(delta_coefs, grid, n_basis=n_basis, init=np.zeros(n_grid))
    alpha_hat = kernel_fn(grid, alpha_coefs, n_basis=n_basis, init=np.zeros((n_grid, n_grid))).T
    beta_hat  = kernel_fn(grid, beta_coefs,  n_basis=n_basis, init=np.zeros((n_grid, n_grid))).T
    returns[:, 0] = delta_hat

    bm_vals  = brownian(grid)
    bm_diffs = np.diff(bm_vals)
    bm_incr  = np.array([bm_diffs[0]] + list(bm_diffs))

    for t in range(1, n_days):
        eta = np.random.normal(0, 1, n_grid)
        eps = _brownian_noise(bm_vals, bm_incr, eta)
        variance = (
            delta_hat
            + (alpha_hat * returns[:, t - 1] ** 2) @ np.ones(n_grid) / n_grid
            + (beta_hat  * variance)               @ np.ones(n_grid) / n_grid
        )
        variance_surface[:, t] = variance
        returns[:, t] = np.sqrt(variance) * eps

    return returns, variance_surface
