"""Simulation utilities for the functional GARCH model."""

import typing
import numpy as np
from numba import njit

from .garch import delta, functional_operator

import warnings
warnings.filterwarnings(action='ignore')


@njit
def brownian(t: np.ndarray) -> np.ndarray:
    """Scaled Brownian motion: $W(t) = 2^{400t} / \\ln 2$."""
    return (2 ** (400 * t)) / np.log(2)


def simulate(
    geometry: tuple,
    M: int,
    coefs: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = functional_operator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a functional GARCH(1,1) process.

    Generates a return matrix mY and the corresponding conditional variance
    surface vsigma2_mat. The driving noise is an intraday Brownian increment
    scaled by the conditional standard deviation.

    Args:
        geometry: Tuple (N, T) where N is the intraday grid size and T the
                  number of days.
        M: Number of Bernstein basis functions.
        coefs: Parameter vector [delta_coefs (M) | alpha_coefs (M²) | beta_coefs (M²)].
        delta_fn: Level operator (default: delta).
        kernel_fn: Kernel operator (default: functional_operator).

    Returns:
        Tuple (mY, vsigma2_mat), each of shape (N, T).

    Note:
        The Bernstein basis projection is used by default.  To switch to a
        different basis, pass alternative delta_fn and kernel_fn callables
        with the same signatures.
    """
    N, T = geometry
    grid = np.linspace(1 / N, 1 - 1 / N, N)
    coefs_delta = coefs[:M]
    coefs_alpha = coefs[M: M + M ** 2]
    coefs_beta  = coefs[M + M ** 2:]

    vsigma2     = np.ones(N)
    mY          = np.ones((N, T))
    vsigma2_mat = np.zeros((N, T))
    vsigma2_mat[:, 0] = vsigma2

    alpha_hat = kernel_fn(grid, coefs_alpha, M=M, _ret=np.zeros((N, N))).T
    beta_hat  = kernel_fn(grid, coefs_beta,  M=M, _ret=np.zeros((N, N))).T
    delta_hat = delta_fn(coefs_delta, grid, M=M, _ret=np.zeros(N))
    mY[:, 0]  = delta_hat

    bm_vals    = brownian(grid)
    bm_diffs   = np.diff(bm_vals)
    bm_incr    = np.array([bm_diffs[0]] + list(bm_diffs))

    noise = lambda sig, inc, eta: np.sqrt(1 / sig) * np.cumsum(np.sqrt(inc) * eta)

    for t in range(1, T):
        eta    = np.random.normal(0, 1, N)
        eps    = noise(bm_vals, bm_incr, eta)
        vsigma2 = (
            delta_hat
            + (alpha_hat * mY[:, t - 1] ** 2) @ np.ones(N) / N
            + (beta_hat  * vsigma2)            @ np.ones(N) / N
        )
        vsigma2_mat[:, t] = vsigma2
        mY[:, t] = np.sqrt(vsigma2) * eps

    return mY, vsigma2_mat
