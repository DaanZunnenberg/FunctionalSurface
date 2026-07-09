#!/usr/bin/env python3
"""
Functional GAS-GARCH volatility surface estimation
====================================================
Replicates the simulation experiment from func_garch_gas.ipynb.

Workflow
--------
1.  Simulate an (N_GRID, N_DAYS) return panel with a known intraday volatility
    surface whose shape and level vary sinusoidally across days.
2.  Estimate the diagonal GAS-GARCH model by maximum likelihood.
3.  Evaluate the fit with metrics (RMSE, R², residual calibration) and plots.

Model
-----
The log-variance curve is parametrised as

    log sigma2_t(u) = Phi(u)^T b_t

where Phi is the (M x N) B-spline basis matrix (built by basis.cubic_bspline_basis)
and b_t follows the diagonal score-driven update

    b_t = omega + b(*) b_{t-1} + a(*) s_{t-1},    b, a in R^M

where (*) denotes element-wise multiplication and s_t is the score of the
multivariate Student-t log-likelihood under the OU covariance structure
Lambda_delta (built by basis.ou_kernel).  This is the diagonal restriction
of the full matrix GAS model in gas.py, which uses M x M matrices B and A.

Parameter vector layout (length 2 + 3M):
    [ nu, delta, omega_1...omega_M, b_1...b_M, a_1...a_M ]

Usage
-----
    pip install -e .                     # install the package first
    python examples/gas_vol_surface.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection
from matplotlib import cm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy import stats  # used in goodness_of_fit for the KS test

from funcgarch.basis import cubic_bspline_basis, ou_kernel


# ── configuration ─────────────────────────────────────────────────────────────

N_GRID      = 25     # intraday grid points  (n)
N_DAYS      = 500    # trading days          (T)
DK          = 7      # spline parameter — M = DK + 1 = 8 basis functions
N_INT_KNOTS = 3      # interior B-spline knots
SEED        = 42
MAXITER     = 1000
WARMUP      = 10     # days dropped from diagnostics (filter initialisation)


# ── simulation ────────────────────────────────────────────────────────────────

def simulate_vol_surface(n: int, T: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns with a known intraday volatility surface.

    The true variance at grid point i on day t is

        sigma2_t(u_i) = 4 + 10*(i - c(t))^2 / (3*n/4)^2  +  2*sin(2*pi*t/T)

    where c(t) = n/2 + n/4 * sin(3*pi*t/T) oscillates sinusoidally, producing a
    U-shaped intraday profile whose trough shifts over time.

    Returns
    -------
    mY          Log-return matrix, shape (n, T).
    sigma2_true True variance matrix, shape (n, T).
    """
    rng = np.random.default_rng(seed)
    grid_i = np.arange(n)
    sigma2 = np.zeros((n, T))
    for t in range(T):
        centre = n / 2 + (n / 4) * np.sin(t * 3 * np.pi / T)
        sigma2[:, t] = (
            4
            + 10 * (grid_i - centre) ** 2 / (0.75 * n) ** 2
            + 2 * np.sin(t * 2 * np.pi / T)
        )
    mY = np.sqrt(sigma2) * rng.standard_normal((n, T))
    return mY, sigma2


# ── GAS-GARCH filter ──────────────────────────────────────────────────────────

def _gas_filter(
    mY: np.ndarray,
    vb_ini: np.ndarray,
    dK: int,
    basis_mat: np.ndarray,
    vtheta: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Diagonal GAS-GARCH filter — negative average log-likelihood and log-variance surface.

    The filter runs forward through all T days, updating the B-spline coefficient
    vector b_t using the score of the multivariate Student-t likelihood.

    The ``basis_mat`` argument is the (M, n) matrix Phi whose columns are the
    B-spline basis vectors at each intraday grid point.  It is built once in
    ``fit_gas`` via ``cubic_bspline_basis`` and reused across all optimizer calls.

    The OU covariance Lambda_delta and its inverse are recomputed from ``vtheta[1]``
    on each call; in practice the optimizer changes delta rarely so this is cheap.

    Parameters
    ----------
    mY        : (n, T) return matrix.
    vb_ini    : (M, 1) initial coefficient vector.
    dK        : B-spline order parameter (M = dK + 1).
    basis_mat : (M, n) B-spline basis matrix from cubic_bspline_basis.
    vtheta    : Parameter vector [nu, delta, omega (M), b (M), a (M)], length 2 + 3M.

    Returns
    -------
    neg_avg_loglik : Scalar to minimise.
    log_sigma2     : (n, T) matrix of log-variance values log(sigma2_t(u_i)).
                     Columns t = 0 and t <= 5 are zero (filter warm-up).
    """
    n, T = mY.shape
    M    = dK + 1
    nu       = vtheta[0]
    ou_scale = vtheta[1]
    omega = vtheta[2:     M + 2].reshape(M, 1)   # constant offset
    vb    = vtheta[M + 2: 2*M + 2].reshape(M, 1)  # diagonal persistence
    va    = vtheta[2*M + 2:].reshape(M, 1)         # diagonal score gain

    # OU covariance from basis.py — captures intraday correlation
    cov_mat  = ou_kernel(np.linspace(0, 1, n), delta=ou_scale)
    cov_inv  = np.linalg.inv(cov_mat)
    log_det  = np.log(np.linalg.det(cov_mat))
    nu_scale = (nu + n) / (2 * nu)

    vb_now     = vb_ini.copy()
    vy_now     = mY[:, 0].reshape(n, 1)
    log_sigma2 = np.zeros(mY.shape)
    log_lik    = 0.0

    for t in range(1, T):
        sigma_now = basis_mat.T @ vb_now       # (n, 1) — log sigma2_t at each grid point
        log_sigma2[:, t] = sigma_now[:, 0]

        S = np.exp(sigma_now / 2)              # (n, 1) — sigma_t (conditional std dev)
        R = np.eye(n) / S                      # (n, n) — diag(1/sigma_t)
        Y = vy_now                             # (n, 1) — returns on day t-1

        # Quadratic form in the Student-t density (scalar)
        A1 = float(np.sum(1 + Y.T @ R @ cov_inv @ R @ Y / nu))

        # Score of the log-likelihood w.r.t. b_t — see README for derivation.
        # r_tilde = S^{-1} y;  A3 = Phi * (r_tilde .* Lambda^{-1} r_tilde)
        A2 = (Y / S).T * basis_mat             # (M, n): Phi diag(r_tilde)
        A3 = A2 @ (cov_inv @ (R @ Y))          # (M, 1)
        score = (
            -0.5 * basis_mat.sum(axis=1, keepdims=True)
            + (nu_scale / A1) * A3
        )

        # Skip first five days from the likelihood (filter initialisation)
        if t > 5:
            log_lik += (
                -0.5 * float(np.sum(sigma_now))
                - (n + nu) / 2 * np.log(A1)
            )

        # Diagonal GAS update: element-wise (*) instead of matrix @ in gas.py
        vb_now = omega + vb * vb_now + va * score
        vy_now = mY[:, t].reshape(n, 1)

    # Constant term of the Student-t log-likelihood (added once, scaled by T)
    log_lik += T * (
        gammaln((nu + n) / 2)
        - gammaln(nu / 2)
        - (n / 2) * np.log(np.pi * nu)
        - 0.5 * log_det
    )
    return -log_lik / T, log_sigma2


# ── estimation ────────────────────────────────────────────────────────────────

def fit_gas(
    mY: np.ndarray,
    dK: int = DK,
    n_int_knots: int = N_INT_KNOTS,
    maxiter: int = MAXITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the diagonal GAS-GARCH model by maximum likelihood.

    Builds the B-spline basis matrix via ``cubic_bspline_basis`` from basis.py
    and passes it to ``_gas_filter`` inside a SLSQP minimisation loop.

    Returns
    -------
    vtheta_hat : Estimated parameter vector, length 2 + 3M.
    sigma_hat  : Estimated volatility surface (std dev = exp(log sigma2 / 2)),
                 shape (n, T).
    basis_mat  : (M, n) B-spline basis matrix used in estimation.
    """
    n = mY.shape[0]
    M = dK + 1
    vtau      = np.linspace(0, 1, n)
    basis_mat = cubic_bspline_basis(vtau, order=dK - 2, n_interior_knots=n_int_knots)

    vb0     = np.ones((M, 1))
    vtheta0 = np.concatenate((
        [2.1, 0.5], np.ones(M), 0.5 * np.ones(M), 0.05 * np.ones(M)
    ))
    LB = np.concatenate(([1.05, 1e-5], -5 * np.ones(M), -0.99 * np.ones(M), -0.5 * np.ones(M)))
    UB = np.concatenate(([50,   1.0],  15 * np.ones(M),  0.99 * np.ones(M),  0.5 * np.ones(M)))

    call_count = [0]

    def _objective(vtheta: np.ndarray) -> float:
        nll = _gas_filter(mY, vb0, dK, basis_mat, vtheta)[0]
        call_count[0] += 1
        if call_count[0] % 25 == 0:
            print(f'  call {call_count[0]:>5d} | neg-log-lik: {nll:.6f}', end='\r')
        return nll

    nll0 = _objective(vtheta0)
    print(f'Initial neg-log-lik: {nll0:.4f}')

    opt = minimize(
        _objective, vtheta0,
        bounds=list(zip(LB, UB)),
        method='SLSQP',
        options={'maxiter': maxiter, 'ftol': 1e-9},
    )
    print(
        f'\nFinal   neg-log-lik: {opt.fun:.4f}'
        f'  |  converged: {opt.success}'
        f'  |  iters: {opt.nit}'
    )

    _, log_sigma2_hat = _gas_filter(mY, vb0, dK, basis_mat, opt.x)
    sigma_hat = np.exp(log_sigma2_hat / 2)   # log sigma2 -> sigma (std dev)
    return opt.x, sigma_hat, basis_mat


# ── goodness-of-fit ───────────────────────────────────────────────────────────

def goodness_of_fit(
    mY: np.ndarray,
    sigma_hat: np.ndarray,
    sigma2_true: np.ndarray,
    nu_hat: float,
    warmup: int = WARMUP,
) -> dict:
    """Compute and print goodness-of-fit metrics.

    Metrics
    -------
    RMSE, MAE
        Error of the estimated volatility std dev vs the true std dev,
        after discarding the warmup period.

    Pearson r, R²
        Correlation and explained variance between sigma_hat and sigma_true.

    Mean z² vs E[z²] under t(nu)
        Standardised residuals z = r / sigma_hat should satisfy
        E[z²] ~= nu/(nu-2) under the Student-t model.  A ratio close to 1
        indicates good volatility calibration.

    KS test
        Kolmogorov-Smirnov test of z against t(nu_hat).  Marginal approximation:
        ignores intraday cross-sectional correlation, so p-values should be
        interpreted as indicative rather than exact.
    """
    sigma_true = np.sqrt(sigma2_true[:, warmup:])
    s_hat      = sigma_hat[:, warmup:]
    z          = (mY[:, warmup:] / s_hat).ravel()

    rmse = np.sqrt(np.mean((s_hat - sigma_true) ** 2))
    mae  = np.mean(np.abs(s_hat - sigma_true))
    r    = float(np.corrcoef(s_hat.ravel(), sigma_true.ravel())[0, 1])
    r2   = 1 - np.sum((s_hat - sigma_true) ** 2) / np.sum((sigma_true - sigma_true.mean()) ** 2)

    resid_var = float(np.mean(z ** 2))
    t_var     = nu_hat / (nu_hat - 2) if nu_hat > 2 else float('nan')
    ks_stat, ks_pval = stats.kstest(z, stats.t(df=nu_hat).cdf)

    print('\n── Goodness-of-fit ─────────────────────────────────────────')
    print(f'  Volatility RMSE:                    {rmse:.4f}')
    print(f'  Volatility MAE:                     {mae:.4f}')
    print(f'  Pearson r (sigma_hat vs sigma):     {r:.4f}')
    print(f'  R²        (sigma_hat vs sigma):     {r2:.4f}')
    print(f'  Mean z²  (E[z²] ~= {t_var:.3f} under t(nu)): {resid_var:.4f}')
    print(f'  KS stat / p-value:                  {ks_stat:.4f} / {ks_pval:.4f}')
    print('────────────────────────────────────────────────────────────')

    return dict(
        rmse=rmse, mae=mae, pearson_r=r, r2=r2,
        resid_var=resid_var, expected_var=t_var,
        ks_stat=ks_stat, ks_pval=ks_pval,
    )


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_surfaces(sigma_hat: np.ndarray, sigma2_true: np.ndarray, warmup: int = WARMUP) -> None:
    """Side-by-side 3D surface plots: true vs estimated volatility."""
    sigma_true = np.sqrt(sigma2_true[:, warmup:])
    s_hat      = sigma_hat[:, warmup:]
    n, T = s_hat.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(T))

    fig = plt.figure(figsize=(18, 7))
    for col, (surf, title) in enumerate([
        (sigma_true.T, 'True volatility  sigma_t(u)'),
        (s_hat.T,      'Estimated volatility  sigma_hat_t(u)'),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.plot_surface(X, Y, surf, cmap=cm.plasma, alpha=0.85)
        ax.set_xlabel('Intraday grid u', fontsize=11)
        ax.set_ylabel('Day t',           fontsize=11)
        ax.set_zlabel('Volatility',      fontsize=11)
        ax.set_title(title,              fontsize=13)
    plt.suptitle('Volatility surface: true vs GAS-GARCH estimate', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Simulate
    print('── Simulating data ──────────────────────────────────────────')
    mY, sigma2_true = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)
    print(f'Return matrix shape : {mY.shape}')
    print(f'True σ² range       : [{sigma2_true.min():.2f}, {sigma2_true.max():.2f}]')

    # 2. Estimate
    print('\n── Estimating diagonal GAS-GARCH ────────────────────────────')
    vtheta_hat, sigma_hat, basis_mat = fit_gas(mY)

    nu_hat    = vtheta_hat[0]
    delta_hat = vtheta_hat[1]
    print(f'\nEstimated ν (Student-t df): {nu_hat:.3f}')
    print(f'Estimated δ (OU scale):     {delta_hat:.5f}')

    # 3. Goodness-of-fit metrics
    metrics = goodness_of_fit(mY, sigma_hat, sigma2_true, nu_hat)

    # 4. Plot
    print('\n── Plotting ─────────────────────────────────────────────────')
    plot_surfaces(sigma_hat, sigma2_true)


if __name__ == '__main__':
    main()
