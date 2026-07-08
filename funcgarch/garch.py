"""Functional GARCH model: operators, loss, estimator, and volatility filter."""

import typing
import numpy as np
from numba import njit, jit, float64
from scipy.optimize import minimize

from .basis import bernstein_basis
from .utils import ResultContainer

_KT = typing.TypeVar('_KT')
_VT = typing.TypeVar('_VT')

import warnings
warnings.filterwarnings(action='ignore')

# Module-level iteration counter used by _log_step.
_COUNTER = ResultContainer(counter=0)


def _log_step(loss: float, params: np.ndarray, log_every: int = 500) -> None:
    """Print a convergence update every `log_every` optimizer calls."""
    _COUNTER.counter += 1
    if _COUNTER.counter % log_every == 0:
        vals = ''.join(str(round(100 * p, 1)).ljust(2, '0') for p in params)
        print(
            f'step {_COUNTER.counter:>5} | loss: \33[41m'
            f'{str(loss).ljust(20, "0")}\33[0m | {vals}'
        )


@njit
def delta(coefs: list, t: float, M: int, _ret: float = 0.0) -> float:
    """Level operator: $\\delta(t) = \\sum_{k=1}^M c_k \\varphi_k^M(t)$.

    Args:
        coefs: Coefficient vector of length M.
        t: Evaluation point(s) in [0, 1].
        M: Number of Bernstein basis functions.
        _ret: Accumulator (default 0.0; pass an array for vectorised evaluation).
    """
    v = 1
    for c in coefs:
        _ret = _ret + c * bernstein_basis(t, M, v)
        v += 1
    return _ret


@jit(nopython=True)
def functional_operator(t: np.ndarray, coefs: list, M: int, _ret: np.ndarray) -> np.ndarray:
    """Kernel operator: $\\mathcal{K}(t,s) = \\sum_{k,l} c_{kl} \\varphi_k^M(t) \\varphi_l^M(s)$.

    Evaluates the M²-term double sum using matrix products of Bernstein vectors.

    Args:
        t: Grid vector of length N; reshaped to a column internally.
        coefs: Flattened coefficient matrix of length M².
        M: Number of Bernstein basis functions per dimension.
        _ret: Zero-initialised N×N accumulator array.
    """
    _final = _ret
    col = t.reshape((len(t), 1))
    idx = 0
    for k in range(1, M + 1):
        for l in range(1, M + 1):
            _final = _final + coefs[idx] * bernstein_basis(col, M, k) @ bernstein_basis(col, M, l).T
            idx += 1
    return _final


@jit(nopython=True)
def loss_func(mY: np.ndarray, vsigma2: np.ndarray, M: int, grid: np.ndarray) -> float64:
    """MSE loss between squared returns and conditional variance, projected on the Bernstein basis.

    Args:
        mY: Intraday return vector for one day, shape (N,).
        vsigma2: Conditional variance vector, shape (N,).
        M: Number of basis functions.
        grid: Evaluation grid, shape (N,).
    """
    total = 0.0
    for k in range(1, M + 1):
        w = bernstein_basis(grid, M=M, k=k)
        total += np.mean(((mY ** 2 - vsigma2) * w) ** 2)
    return total


def garch_filter(
    mY: np.ndarray,
    grid_length: int,
    vtheta: np.ndarray,
    M: int,
    sigma2_ini: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = functional_operator,
) -> np.ndarray:
    """Extract the conditional volatility surface from observed returns.

    Applies the fitted functional GARCH recursion forward in time and
    returns the full N×T variance matrix.

    Args:
        mY: Return matrix, shape (N, T).
        grid_length: Number of intraday grid points N.
        vtheta: Parameter vector [delta_coefs | alpha_coefs | beta_coefs].
        M: Number of Bernstein basis functions.
        sigma2_ini: Initial variance vector, shape (N,).
        delta_fn: Level operator (default: delta).
        kernel_fn: Kernel operator (default: functional_operator).

    Returns:
        Variance matrix of shape (N, T).
    """
    T = mY.shape[1]
    N = mY.shape[0]
    grid = np.linspace(1 / grid_length, 1 - 1 / grid_length, grid_length)
    coefs_delta = vtheta[:M]
    coefs_alpha = vtheta[M: M + M ** 2]
    coefs_beta  = vtheta[M + M ** 2:]

    vsigma2 = sigma2_ini * np.ones(grid_length)
    vsigma2_mat = np.zeros(mY.shape)
    vsigma2_mat[:, 0] = vsigma2

    alpha_hat = kernel_fn(grid, coefs_alpha, M=M, _ret=np.zeros((grid_length, grid_length))).T
    beta_hat  = kernel_fn(grid, coefs_beta,  M=M, _ret=np.zeros((grid_length, grid_length))).T
    delta_hat = delta_fn(coefs_delta, grid, M=M, _ret=np.zeros(grid_length))

    for t in range(1, T):
        vsigma2 = (
            delta_hat
            + (alpha_hat * mY[:, t - 1] ** 2) @ np.ones(N) / N
            + (beta_hat  * vsigma2)            @ np.ones(N) / N
        )
        vsigma2_mat[:, t] = vsigma2
    return vsigma2_mat


def garch_estimator(
    mY: np.ndarray,
    grid_length: int,
    vtheta: np.ndarray,
    M: int,
    sigma2_ini: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = functional_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
) -> float:
    """Compute the functional GARCH loss for a given parameter vector.

    Runs the GARCH recursion forward through all T days and accumulates
    the Bernstein-projected MSE loss. Intended to be passed to scipy.minimize.

    Args:
        mY: Return matrix, shape (N, T).
        grid_length: Number of intraday grid points N.
        vtheta: Parameter vector [delta_coefs | alpha_coefs | beta_coefs].
        M: Number of Bernstein basis functions.
        sigma2_ini: Initial variance vector, shape (N,).
        delta_fn: Level operator.
        kernel_fn: Kernel operator.
        loss_fn: Loss function accumulated over days.
        print_convergence: If True, print progress every 500 calls.

    Returns:
        Scalar loss value.
    """
    T = mY.shape[1]
    N = mY.shape[0]
    grid = np.linspace(1 / grid_length, 1 - 1 / grid_length, grid_length)
    coefs_delta = vtheta[:M]
    coefs_alpha = vtheta[M: M + M ** 2]
    coefs_beta  = vtheta[M + M ** 2:]

    vsigma2 = sigma2_ini * np.ones(grid_length)
    total_loss = 0.0

    alpha_hat = kernel_fn(grid, coefs_alpha, M=M, _ret=np.zeros((grid_length, grid_length))).T
    beta_hat  = kernel_fn(grid, coefs_beta,  M=M, _ret=np.zeros((grid_length, grid_length))).T
    delta_hat = delta_fn(coefs_delta, grid, M=M, _ret=np.zeros(grid_length))
    ones_N = np.ones(N)

    for t in range(1, T):
        vsigma2 = (
            delta_hat
            + ((alpha_hat * mY[:, t - 1] ** 2) @ ones_N
            +  (beta_hat  * vsigma2)            @ ones_N) / N
        )
        total_loss += loss_fn(mY=mY[:, t], vsigma2=vsigma2, grid=grid, M=M)

    if print_convergence:
        _log_step(total_loss, vtheta)
    return total_loss


def fit(
    mY: np.ndarray,
    sigma2_ini: np.ndarray,
    grid_length: int,
    M: int = 1,
    estimator_fn: typing.Callable = garch_estimator,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = functional_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
    **kwargs,
) -> ResultContainer:
    """Estimate the functional GARCH parameters by minimising the projected MSE loss.

    Wraps scipy.minimize; all optimizer arguments (x0, bounds, method, …)
    are passed through **kwargs.

    Args:
        mY: Return matrix, shape (N, T).
        sigma2_ini: Initial variance vector, shape (N,).
        grid_length: Number of intraday grid points N.
        M: Number of Bernstein basis functions.
        estimator_fn: Loss function to minimise (default: garch_estimator).
        delta_fn: Level operator.
        kernel_fn: Kernel operator.
        loss_fn: Per-day loss function.
        print_convergence: If True, log optimizer progress.
        **kwargs: Passed directly to scipy.minimize (x0, bounds, method, options, …).

    Returns:
        ResultContainer wrapping the scipy OptimizeResult.

    Example::

        result = fit(
            mY, sigma2_ini=np.ones(N), grid_length=N, M=2,
            x0=np.zeros(M + 2 * M**2),
            bounds=[(-.99, .99)] * (M + 2 * M**2),
            method='SLSQP',
        )
        theta_hat = result.x
    """
    def _objective(vtheta):
        return estimator_fn(
            mY, grid_length, vtheta,
            M=M, sigma2_ini=sigma2_ini,
            delta_fn=delta_fn, kernel_fn=kernel_fn, loss_fn=loss_fn,
            print_convergence=print_convergence,
        )

    opt_result = minimize(_objective, options={'disp': True}, **kwargs)
    _COUNTER.counter = 0  # reset iteration counter after each fit call
    return ResultContainer(**{key: opt_result[key] for key in opt_result.__dir__()})


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    prices = pd.read_csv('../price_data_example.csv', parse_dates=True, index_col='date').open
    prices = prices[(prices.index >= '2023-12-14') & (prices.index <= '2024-02-22')].resample('300S').last()
    returns = 100 * np.log(prices[1:] / prices[:-1].values)
    N, T = int(len(returns) / 70), 70
    mY = np.zeros((N, T))
    for k in range(T):
        mY[:, k] = returns.iloc[k * T: k * T + N].values

    M = 4
    result = fit(
        mY, sigma2_ini=np.ones(N), grid_length=N, M=M,
        x0=np.array([0.001] * M + [np.random.uniform(-0.2, 1) for _ in range(2 * M ** 2)]),
        bounds=[(-.99, .99)] * (M + 2 * M ** 2),
        method='SLSQP',
    )

    grid = np.linspace(1 / N, 1 - 1 / N, N)
    plt.plot(delta(result.x[:M], grid, M=M, _ret=np.zeros(N)))
    plt.plot(mY[:, 50], color='black', lw=1, alpha=1)
    plt.plot(garch_filter(mY, grid_length=N, vtheta=result.x, M=M, sigma2_ini=np.ones(N) * 0.2)[:, 60])
    plt.show()
