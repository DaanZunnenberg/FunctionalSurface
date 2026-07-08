"""Functional GAS-GARCH model — B-spline parametrisation.

Module role
───────────
Provides two estimators for functional volatility models that parametrise the
log-volatility curve with a B-spline basis Φ:

    log σ_t(u) = Φ(u)ᵀ b_t

``gas_garch_estimator``
    The full GAS model.  The coefficient vector b_t is updated at each day by
    the scaled score of the multivariate Student-t log-likelihood:

        b_t = ω + B b_{t-1} + A s_{t-1}

    Returns the negative average log-likelihood and the fitted log-volatility
    surface.

``func_garch_estimator``
    A B-spline GARCH baseline (no score updating).  The variance is evolved
    through the same GARCH(1,1) recursion as in ``garch.py``, but using
    B-spline basis matrices instead of Bernstein polynomials.  Useful as a
    performance baseline.

Both estimators are called directly inside a ``scipy.optimize.minimize`` loop
(there is no ``fit()`` wrapper).  See ``examples/`` for worked demonstrations.

Dependencies
────────────
- ``basis.py`` — ``cubic_bspline_basis`` builds the basis matrix Φ that both
  estimators expect as input; ``ou_kernel`` builds the OU covariance Λ_δ used
  in the GAS likelihood.
"""

import warnings
import typing

import numpy as np
from scipy.special import gammaln

from .basis import cubic_bspline_basis, ou_kernel

warnings.filterwarnings(action='ignore')


class _CallCounter:
    count: int = 0


_COUNT = _CallCounter()


def _log_step(name: str, counter: _CallCounter, log_loss: float, log_every: int = 1) -> None:
    """Print a convergence update for the GAS optimiser."""
    if counter.count % log_every == 0:
        msg = f'Running {name} :: call #{counter.count:<5} :: loss: {log_loss:.6f}'
        print(msg.ljust(len(msg) + 30, ' '), end='\r')


def gas_garch_estimator(
    mY: np.ndarray,
    vb_ini: np.ndarray,
    dK: int,
    n: int,
    basis_mat: np.ndarray,
    vtheta: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Functional GAS-GARCH log-likelihood and fitted volatility surface.

    Implements the score-driven update for the B-spline coefficient vector b_t:
        b_t = omega + B @ b_{t-1} + A @ score_t

    where the score is derived from the multivariate Student-t log-likelihood
    with OU-structured covariance.

    Args:
        mY: Return matrix, shape (n, T). Rows are intraday times, columns are days.
        vb_ini: Initial B-spline coefficient vector, shape (dK+1, 1).
        dK: Number of B-spline basis functions minus one.
        n: Number of intraday grid points.
        basis_mat: B-spline basis matrix, shape (dK+1, n).
        vtheta: Parameter vector:
            [nu, ou_scale, omega (M), vec(B) (M²), vec(A) (M²)]
            where M = dK + 1.

    Returns:
        Tuple of (negative average log-likelihood, fitted sigma matrix of shape (n, T)).
    """
    T = mY.shape[1]
    nu       = vtheta[0]
    ou_scale = vtheta[1]
    M        = dK + 1

    omega = np.array(vtheta[2: M + 2]).reshape((M, 1))
    mB    = np.array(vtheta[M + 2: M + 2 + M ** 2]).reshape((M, M))
    mA    = np.array(vtheta[-(M ** 2):]).reshape((M, M))

    cov_mat = ou_kernel(np.linspace(0, 1, mY.shape[0]), delta=ou_scale)
    cov_inv = np.linalg.inv(cov_mat)

    vb_now    = vb_ini.copy()
    vy_now    = mY[:, 0].copy().reshape((n, 1))
    sigma_mat = np.zeros(mY.shape)
    log_lik   = 0.0
    nu_scale  = (nu + n) / (2 * nu)

    for t in range(1, T):
        sigma_now = basis_mat.T @ vb_now  # (n, 1)
        sigma_mat[:, t] = sigma_now[:, 0]

        Y  = vy_now
        S  = np.exp(sigma_now / 2)
        R  = np.eye(n) / S

        A1 = float(np.sum(1 + (Y.T @ (R @ (cov_inv @ (R @ Y)))) / nu))
        A2 = (Y / S).T * basis_mat
        A3 = A2 @ (cov_inv @ (R @ Y))

        if t > 5:
            log_lik += -0.5 * float(np.sum(sigma_now)) - ((n + nu) / 2) * np.log(A1)

        score  = np.array(
            -0.5 * (basis_mat @ np.ones(n)).reshape((dK + 1, 1))
            + (nu_scale / A1) * A3
        )
        vb_now = omega + mB @ vb_now + mA @ score
        vy_now = mY[:, t].reshape((n, 1))

    log_lik += T * (
        gammaln((nu + n) / 2)
        - gammaln(nu / 2)
        - (n / 2) * np.log(np.pi * nu)
        - 0.5 * np.log(np.linalg.det(cov_mat))
    )
    _log_step('GAS estimator', _COUNT, -log_lik / T)
    _COUNT.count += 1
    return -log_lik / T, sigma_mat


def func_garch_estimator(
    mY: np.ndarray,
    basis_splines: np.ndarray,
    vtheta: np.ndarray,
    M: int,
    p: int = 1,
    q: int = 1,
) -> tuple[float, np.ndarray]:
    """Functional GARCH estimator using B-spline basis projections.

    Implements the functional GARCH(1,1) recursion in the B-spline coefficient
    space. The conditional variance operator is:
        sigma²_t = B^T delta + (B^T A B) * y²_{t-1} + sigma²_{t-1} (B^T C B)

    Args:
        mY: Return matrix, shape (N, T). Rows are intraday times, columns are days.
        basis_splines: B-spline basis matrix, shape (M, N).
        vtheta: Parameter vector [delta_coefs (M) | vec(alpha) (M²) | vec(beta) (M²)].
        M: Number of B-spline basis functions.
        p: AR order (currently only p=1 supported).
        q: MA order (currently only q=1 supported).

    Returns:
        Tuple of (MSE loss, fitted variance matrix of shape (N, T)).
    """
    if max(p, q) > 1:
        raise NotImplementedError('Order (p,q) must be (1,1)')

    N, T = mY.shape
    coefs_delta = np.array(vtheta[:M]).reshape(1, M)
    coefs_alpha = vtheta[M: M + M ** 2].reshape((M, M))
    coefs_beta  = vtheta[M + M ** 2:].reshape((M, M))

    vsigma2     = np.ones(N)
    vsigma2_mat = np.zeros(mY.shape)
    vsigma2_mat[:, 0] = vsigma2

    delta_hat = coefs_delta @ basis_splines                        # (1, N)
    alpha_hat = basis_splines.T @ (coefs_alpha @ basis_splines)   # (N, N)
    beta_hat  = basis_splines.T @ (coefs_beta  @ basis_splines)   # (N, N)

    loss = 0.0
    for t in range(1, T):
        vsigma2 = np.asarray(
            delta_hat
            + (alpha_hat * mY[:, t - 1] ** 2) @ np.ones(N) / N
            + (vsigma2 @ beta_hat) / N
        ).ravel()
        vsigma2_mat[:, t] = vsigma2
        loss += float(np.sum(((mY[:, t] ** 2 - vsigma2) * basis_splines) ** 2))

    _log_step('GARCH estimator', _COUNT, loss)
    _COUNT.count += 1
    return loss, vsigma2_mat
