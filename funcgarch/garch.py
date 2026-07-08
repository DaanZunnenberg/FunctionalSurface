"""Functional GARCH model: operators, loss, estimator, and volatility filter."""

import warnings
import typing

import numpy as np
from numba import njit, jit
from scipy.optimize import minimize

from .basis import bernstein_basis
from .utils import ResultContainer

warnings.filterwarnings(action='ignore')

# Mutable module-level counter for convergence logging; reset to 0 after each fit().
_call_count: list[int] = [0]


def _log_step(loss: float, params: np.ndarray, log_every: int = 500) -> None:
    """Print an optimizer progress line every `log_every` function evaluations."""
    _call_count[0] += 1
    if _call_count[0] % log_every == 0:
        vals = ' '.join(f'{100 * p:+.2f}' for p in params)
        print(f'step {_call_count[0]:>5} | loss: {loss:.6f} | params: {vals}')


@njit
def delta(coefs: np.ndarray, t: float, M: int, init: float = 0.0) -> float:
    r"""Level operator $\delta(t) = \sum_{k=1}^M c_k \varphi_k^M(t)$.

    Args:
        coefs: Level coefficient vector, length M.
        t: Evaluation point(s) in [0, 1].  Pass a scalar or an array.
        M: Number of Bernstein basis functions.
        init: Starting value for the accumulation.  Pass ``np.zeros(N)``
              for vectorised evaluation over a grid of length N.
    """
    acc = init
    for k, c in enumerate(coefs):
        acc = acc + c * bernstein_basis(t, M, k + 1)
    return acc


@jit(nopython=True)
def kernel_operator(
    t: np.ndarray,
    coefs: np.ndarray,
    M: int,
    init: np.ndarray,
) -> np.ndarray:
    r"""Kernel operator $\mathcal{K}(t,s) = \sum_{k,l} c_{kl}\,\varphi_k^M(t)\,\varphi_l^M(s)$.

    Evaluates the M²-term double sum via matrix products of Bernstein column
    vectors, returning an N×N matrix.

    Args:
        t: Grid vector of length N; reshaped to a column internally.
        coefs: Flattened coefficient matrix, length M².
        M: Number of Bernstein basis functions per dimension.
        init: Zero-initialised N×N accumulator array (required by Numba).
    """
    acc = init
    col = t.reshape((len(t), 1))
    idx = 0
    for k in range(1, M + 1):
        bk = bernstein_basis(col, M, k)
        for l in range(1, M + 1):
            acc = acc + coefs[idx] * bk @ bernstein_basis(col, M, l).T
            idx += 1
    return acc


@jit(nopython=True)
def loss_func(
    returns: np.ndarray,
    sigma2: np.ndarray,
    M: int,
    grid: np.ndarray,
) -> float:
    """Bernstein-projected MSE between squared returns and conditional variance.

    $L = \sum_{k=1}^M \\mathbb{E}[(r^2 - \\sigma^2)^2 \\varphi_k^M]$

    Args:
        returns: Intraday return vector for one day, shape (N,).
        sigma2: Conditional variance vector, shape (N,).
        M: Number of Bernstein basis functions.
        grid: Evaluation grid, shape (N,).
    """
    total = 0.0
    for k in range(1, M + 1):
        w = bernstein_basis(grid, M, k)
        total += np.mean(((returns ** 2 - sigma2) * w) ** 2)
    return total


def _build_operators(
    vtheta: np.ndarray,
    M: int,
    n_grid: int,
    delta_fn: typing.Callable,
    kernel_fn: typing.Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack parameter vector into pre-evaluated operator matrices.

    Returns:
        (grid, delta_hat, alpha_hat, beta_hat) where delta_hat is shape (N,)
        and alpha_hat, beta_hat are (N, N).
    """
    grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)
    coefs_delta = vtheta[:M]
    coefs_alpha = vtheta[M: M + M ** 2]
    coefs_beta  = vtheta[M + M ** 2:]
    delta_hat = delta_fn(coefs_delta, grid, M=M, init=np.zeros(n_grid))
    alpha_hat = kernel_fn(grid, coefs_alpha, M=M, init=np.zeros((n_grid, n_grid))).T
    beta_hat  = kernel_fn(grid, coefs_beta,  M=M, init=np.zeros((n_grid, n_grid))).T
    return grid, delta_hat, alpha_hat, beta_hat


def garch_filter(
    mY: np.ndarray,
    n_grid: int,
    vtheta: np.ndarray,
    M: int,
    sigma2_init: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
) -> np.ndarray:
    """Extract the conditional variance surface from observed returns.

    Applies the fitted functional GARCH recursion forward through all T days.

    Args:
        mY: Return matrix, shape (N, T).
        n_grid: Number of intraday grid points N.
        vtheta: Parameter vector [delta_coefs (M) | alpha_coefs (M²) | beta_coefs (M²)].
        M: Number of Bernstein basis functions.
        sigma2_init: Initial variance vector, shape (N,).
        delta_fn: Level operator (injectable for alternative bases).
        kernel_fn: Kernel operator (injectable for alternative bases).

    Returns:
        Variance matrix of shape (N, T).
    """
    N, T = mY.shape
    grid, delta_hat, alpha_hat, beta_hat = _build_operators(
        vtheta, M, n_grid, delta_fn, kernel_fn
    )
    ones_N = np.ones(N)
    sigma2 = sigma2_init * np.ones(n_grid)
    sigma2_mat = np.zeros((N, T))
    sigma2_mat[:, 0] = sigma2

    for t in range(1, T):
        sigma2 = (
            delta_hat
            + (alpha_hat * mY[:, t - 1] ** 2) @ ones_N / N
            + (beta_hat  * sigma2)             @ ones_N / N
        )
        sigma2_mat[:, t] = sigma2
    return sigma2_mat


def garch_estimator(
    mY: np.ndarray,
    n_grid: int,
    vtheta: np.ndarray,
    M: int,
    sigma2_init: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
) -> float:
    """Compute the functional GARCH objective for a given parameter vector.

    Runs the GARCH recursion forward through all T days and accumulates the
    Bernstein-projected MSE loss.  Designed to be passed to scipy.minimize.

    Args:
        mY: Return matrix, shape (N, T).
        n_grid: Number of intraday grid points N.
        vtheta: Parameter vector [delta_coefs (M) | alpha_coefs (M²) | beta_coefs (M²)].
        M: Number of Bernstein basis functions.
        sigma2_init: Initial variance vector, shape (N,).
        delta_fn: Level operator.
        kernel_fn: Kernel operator.
        loss_fn: Per-day loss function (default: loss_func).
        print_convergence: If True, print progress every 500 evaluations.

    Returns:
        Scalar objective value.
    """
    N, T = mY.shape
    grid, delta_hat, alpha_hat, beta_hat = _build_operators(
        vtheta, M, n_grid, delta_fn, kernel_fn
    )
    ones_N = np.ones(N)
    sigma2 = sigma2_init * np.ones(n_grid)
    total_loss = 0.0

    for t in range(1, T):
        sigma2 = (
            delta_hat
            + ((alpha_hat * mY[:, t - 1] ** 2) @ ones_N
            +  (beta_hat  * sigma2)             @ ones_N) / N
        )
        total_loss += loss_fn(mY[:, t], sigma2, M, grid)

    if print_convergence:
        _log_step(total_loss, vtheta)
    return total_loss


def fit(
    mY: np.ndarray,
    sigma2_init: np.ndarray,
    n_grid: int,
    M: int = 1,
    estimator_fn: typing.Callable = garch_estimator,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
    options: dict | None = None,
    **kwargs,
) -> ResultContainer:
    """Estimate functional GARCH parameters by minimising the projected MSE loss.

    Wraps scipy.minimize; optimizer arguments (x0, bounds, method, …) are
    passed through **kwargs.

    Args:
        mY: Return matrix, shape (N, T).
        sigma2_init: Initial variance vector, shape (N,).
        n_grid: Number of intraday grid points N.
        M: Number of Bernstein basis functions.
        estimator_fn: Full-sequence objective to minimise.
        delta_fn: Level operator (override to use a different basis).
        kernel_fn: Kernel operator (override to use a different basis).
        loss_fn: Per-day loss function.
        print_convergence: If True, log optimizer progress.
        options: Options dict forwarded to scipy.minimize (merged with defaults).
        **kwargs: Remaining arguments passed to scipy.minimize
                  (x0, bounds, method, constraints, …).

    Returns:
        ResultContainer wrapping the scipy OptimizeResult.

    Example::

        result = fit(
            mY, sigma2_init=np.ones(N), n_grid=N, M=2,
            x0=np.zeros(M + 2 * M**2),
            bounds=[(-.99, .99)] * (M + 2 * M**2),
            method='SLSQP',
        )
        theta_hat = result.x
    """
    _options = {'disp': True}
    if options:
        _options.update(options)

    def _objective(vtheta: np.ndarray) -> float:
        return estimator_fn(
            mY, n_grid, vtheta,
            M=M, sigma2_init=sigma2_init,
            delta_fn=delta_fn, kernel_fn=kernel_fn, loss_fn=loss_fn,
            print_convergence=print_convergence,
        )

    try:
        opt = minimize(_objective, options=_options, **kwargs)
    finally:
        _call_count[0] = 0

    return ResultContainer(**{k: opt[k] for k in opt.__dir__()})


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    prices = (
        pd.read_csv('../price_data_example.csv', parse_dates=True, index_col='date')
        .open
        .loc['2023-12-14':'2024-02-22']
        .resample('300S').last()
    )
    returns = 100 * np.log(prices[1:] / prices[:-1].values)
    N, T = int(len(returns) / 70), 70
    mY = np.column_stack([returns.iloc[k * T: k * T + N].values for k in range(T)])

    M = 4
    result = fit(
        mY, sigma2_init=np.ones(N), n_grid=N, M=M,
        x0=np.array([0.001] * M + [np.random.uniform(-0.2, 1) for _ in range(2 * M ** 2)]),
        bounds=[(-.99, .99)] * (M + 2 * M ** 2),
        method='SLSQP',
    )

    grid = np.linspace(1 / N, 1 - 1 / N, N)
    fig, ax = plt.subplots()
    ax.plot(delta(result.x[:M], grid, M=M, init=np.zeros(N)), label='level δ')
    ax.plot(mY[:, 50], color='black', lw=1, alpha=0.5, label='returns day 50')
    ax.plot(
        garch_filter(mY, n_grid=N, vtheta=result.x, M=M, sigma2_init=np.ones(N))[:, 60],
        label='σ² day 60',
    )
    ax.legend()
    plt.show()
