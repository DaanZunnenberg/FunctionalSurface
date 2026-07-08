"""Simulation utilities for the functional GARCH model.

Module role
───────────
Generates synthetic (mY, vsigma2_mat) pairs from a functional GARCH(1,1)
process.  Used for Monte Carlo studies and to verify that ``fit()`` recovers
known parameters.

The GARCH recursion here is identical to the one in ``garch_estimator()``, so
any ``coefs`` vector valid for ``fit()`` can be passed directly to
``simulate()``.

The intraday noise at each time step is constructed as cumulative scaled
Brownian increments (``_brownian_noise``), driven by the function
``brownian(t) = 2^{400t} / ln 2``.

Dependencies
────────────
- ``garch.py`` — imports ``delta`` and ``kernel_operator`` as the default
  level and kernel callables.  An alternative basis pair can be injected via
  ``delta_fn`` / ``kernel_fn``.
"""

import warnings
import typing

import numpy as np
from numba import njit

from .garch import delta, kernel_operator

warnings.filterwarnings(action='ignore')


@njit
def brownian(t: np.ndarray) -> np.ndarray:
    """Scaled Brownian motion: $W(t) = 2^{400t} / \\ln 2$."""
    return (2 ** (400 * t)) / np.log(2)


def _brownian_noise(
    sigma: np.ndarray,
    bm_incr: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Intraday noise path: cumulative sum of scaled normal increments."""
    return np.sqrt(1 / sigma) * np.cumsum(np.sqrt(bm_incr) * eta)


def simulate(
    shape: tuple[int, int],
    M: int,
    coefs: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a functional GARCH(1,1) process.

    Generates a return matrix mY and the corresponding conditional variance
    surface vsigma2_mat. The driving noise is an intraday Brownian increment
    scaled by the conditional standard deviation.

    Args:
        shape: Tuple (N, T) where N is the intraday grid size and T the
               number of days.
        M: Number of Bernstein basis functions.
        coefs: Parameter vector [delta_coefs (M) | alpha_coefs (M²) | beta_coefs (M²)].
        delta_fn: Level operator (default: delta).
        kernel_fn: Kernel operator (default: kernel_operator).

    Returns:
        Tuple (mY, vsigma2_mat), each of shape (N, T).
    """
    N, T = shape
    grid = np.linspace(1 / N, 1 - 1 / N, N)
    coefs_delta = coefs[:M]
    coefs_alpha = coefs[M: M + M ** 2]
    coefs_beta  = coefs[M + M ** 2:]

    vsigma2     = np.ones(N)
    mY          = np.ones((N, T))
    vsigma2_mat = np.zeros((N, T))
    vsigma2_mat[:, 0] = vsigma2

    delta_hat = delta_fn(coefs_delta, grid, M=M, init=np.zeros(N))
    alpha_hat = kernel_fn(grid, coefs_alpha, M=M, init=np.zeros((N, N))).T
    beta_hat  = kernel_fn(grid, coefs_beta,  M=M, init=np.zeros((N, N))).T
    mY[:, 0]  = delta_hat

    bm_vals  = brownian(grid)
    bm_diffs = np.diff(bm_vals)
    bm_incr  = np.array([bm_diffs[0]] + list(bm_diffs))

    for t in range(1, T):
        eta = np.random.normal(0, 1, N)
        eps = _brownian_noise(bm_vals, bm_incr, eta)
        vsigma2 = (
            delta_hat
            + (alpha_hat * mY[:, t - 1] ** 2) @ np.ones(N) / N
            + (beta_hat  * vsigma2)            @ np.ones(N) / N
        )
        vsigma2_mat[:, t] = vsigma2
        mY[:, t] = np.sqrt(vsigma2) * eps

    return mY, vsigma2_mat
