#!/usr/bin/env python3
"""
Functional GAS-GARCH volatility surface estimation
====================================================
Replicates the simulation experiment from the gas_garch notebooks.

Workflow
--------
1.  Simulate an (n_grid, n_days) return panel with a known intraday volatility
    surface whose shape and level vary sinusoidally across days.
2.  Estimate the diagonal GAS-GARCH model by maximum likelihood.
3.  Evaluate the fit with metrics (RMSE, R², residual calibration) and plots.

Model
-----
The log-variance curve is parametrised as

    log sigma2_t(u) = Phi(u)^T b_t

where Phi is the (n_basis x n_grid) B-spline basis matrix and b_t follows
the diagonal score-driven update

    b_t = omega + b_diag * b_{t-1} + a_diag * score_{t-1}

where b_diag, a_diag are element-wise (diagonal) multipliers and score_t is
the score of the multivariate Student-t log-likelihood under the OU covariance.

Parameter vector layout (length 2 + 3*n_basis):
    [ nu, ou_scale, omega_1...omega_M, b_1...b_M, a_1...a_M ]

Usage
-----
    pip install -e .
    python scripts/gas_vol_surface.py
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
from scipy import stats

from funcgarch.basis import cubic_bspline_basis, ou_kernel


# ── configuration ─────────────────────────────────────────────────────────────

N_GRID       = 25     # intraday grid points
N_DAYS       = 500    # trading days
N_BASIS      = 8      # number of B-spline basis functions (including constant)
N_INT_KNOTS  = 3      # interior B-spline knots
SEED         = 42
MAXITER      = 1000
WARMUP       = 10     # days dropped from diagnostics (filter initialisation)


# ── simulation ────────────────────────────────────────────────────────────────

def simulate_vol_surface(n_grid: int, n_days: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns with a known intraday volatility surface.

    The true variance at grid point i on day t is

        sigma2_t(u_i) = 4 + 10*(i - c(t))^2 / (3*n/4)^2  +  2*sin(2*pi*t/T)

    where c(t) = n/2 + n/4 * sin(3*pi*t/T) oscillates sinusoidally.

    Returns
    -------
    returns       Log-return matrix, shape (n_grid, n_days).
    true_variance True variance matrix, shape (n_grid, n_days).
    """
    rng = np.random.default_rng(seed)
    grid_i = np.arange(n_grid)
    true_variance = np.zeros((n_grid, n_days))
    for t in range(n_days):
        centre = n_grid / 2 + (n_grid / 4) * np.sin(t * 3 * np.pi / n_days)
        true_variance[:, t] = (
            4
            + 10 * (grid_i - centre) ** 2 / (0.75 * n_grid) ** 2
            + 2 * np.sin(t * 2 * np.pi / n_days)
        )
    returns = np.sqrt(true_variance) * rng.standard_normal((n_grid, n_days))
    return returns, true_variance


# ── diagonal GAS-GARCH filter ─────────────────────────────────────────────────

def _gas_filter(
    returns: np.ndarray,
    init_coefs: np.ndarray,
    basis_mat: np.ndarray,
    params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Diagonal GAS-GARCH filter — negative average log-likelihood and log-variance surface.

    Parameters
    ----------
    returns    : (n_grid, n_days) return matrix.
    init_coefs : (n_basis, 1) initial coefficient vector.
    basis_mat  : (n_basis, n_grid) B-spline basis matrix from cubic_bspline_basis.
    params     : Parameter vector [nu, ou_scale, omega (n_basis), b (n_basis), a (n_basis)].

    Returns
    -------
    neg_avg_loglik : Scalar to minimise.
    log_vol_surface : (n_grid, n_days) log-variance values log(sigma2_t(u_i)).
    """
    n_basis, n_grid = basis_mat.shape
    n_days = returns.shape[1]

    nu       = params[0]
    ou_scale = params[1]
    omega    = params[2:      n_basis + 2].reshape(n_basis, 1)
    b_diag   = params[n_basis + 2: 2 * n_basis + 2].reshape(n_basis, 1)
    a_diag   = params[2 * n_basis + 2:].reshape(n_basis, 1)

    cov_mat  = ou_kernel(np.linspace(0, 1, n_grid), delta=ou_scale)
    cov_inv  = np.linalg.inv(cov_mat)
    log_det  = np.log(np.linalg.det(cov_mat))
    nu_scale = (nu + n_grid) / (2 * nu)

    coef_vec        = init_coefs.copy()
    returns_prev    = returns[:, 0].reshape(n_grid, 1)
    log_vol_surface = np.zeros(returns.shape)
    log_lik         = 0.0

    for t in range(1, n_days):
        log_vol = basis_mat.T @ coef_vec          # (n_grid, 1)
        log_vol_surface[:, t] = log_vol[:, 0]

        std_dev   = np.exp(log_vol / 2)           # (n_grid, 1)
        scale_inv = np.eye(n_grid) / std_dev      # diag(1/sigma_t)

        student_denom = float(np.sum(
            1 + returns_prev.T @ scale_inv @ cov_inv @ scale_inv @ returns_prev / nu
        ))
        score_a = (returns_prev / std_dev).T * basis_mat
        score_b = score_a @ (cov_inv @ (scale_inv @ returns_prev))
        score = (
            -0.5 * basis_mat.sum(axis=1, keepdims=True)
            + (nu_scale / student_denom) * score_b
        )

        if t > 5:
            log_lik += (
                -0.5 * float(np.sum(log_vol))
                - (n_grid + nu) / 2 * np.log(student_denom)
            )

        coef_vec     = omega + b_diag * coef_vec + a_diag * score
        returns_prev = returns[:, t].reshape(n_grid, 1)

    log_lik += n_days * (
        gammaln((nu + n_grid) / 2)
        - gammaln(nu / 2)
        - (n_grid / 2) * np.log(np.pi * nu)
        - 0.5 * log_det
    )
    return -log_lik / n_days, log_vol_surface


# ── estimation ────────────────────────────────────────────────────────────────

def fit_gas(
    returns: np.ndarray,
    n_basis: int = N_BASIS,
    n_int_knots: int = N_INT_KNOTS,
    maxiter: int = MAXITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the diagonal GAS-GARCH model by maximum likelihood.

    Returns
    -------
    params_hat  : Estimated parameter vector, length 2 + 3*n_basis.
    vol_surface : Estimated volatility surface (std dev = exp(log_sigma2 / 2)),
                  shape (n_grid, n_days).
    basis_mat   : (n_basis, n_grid) B-spline basis matrix used in estimation.
    """
    n_grid = returns.shape[0]
    grid = np.linspace(0, 1, n_grid)
    spline_order = n_basis - 3          # order s.t. n_interior_knots + order - 1 + 1 = n_basis
    basis_mat = cubic_bspline_basis(grid, order=spline_order, n_interior_knots=n_int_knots)

    init_coefs  = np.ones((n_basis, 1))
    init_params = np.concatenate((
        [2.1, 0.5], np.ones(n_basis), 0.5 * np.ones(n_basis), 0.05 * np.ones(n_basis)
    ))
    lower_bounds = np.concatenate(([1.05, 1e-5], -5 * np.ones(n_basis), -0.99 * np.ones(n_basis), -0.5 * np.ones(n_basis)))
    upper_bounds = np.concatenate(([50,   1.0],  15 * np.ones(n_basis),  0.99 * np.ones(n_basis),  0.5 * np.ones(n_basis)))

    call_count = [0]

    def _objective(params: np.ndarray) -> float:
        nll = _gas_filter(returns, init_coefs, basis_mat, params)[0]
        call_count[0] += 1
        if call_count[0] % 25 == 0:
            print(f'  call {call_count[0]:>5d} | neg-log-lik: {nll:.6f}', end='\r')
        return nll

    nll0 = _objective(init_params)
    print(f'Initial neg-log-lik: {nll0:.4f}')

    result = minimize(
        _objective, init_params,
        bounds=list(zip(lower_bounds, upper_bounds)),
        method='SLSQP',
        options={'maxiter': maxiter, 'ftol': 1e-9},
    )
    print(
        f'\nFinal   neg-log-lik: {result.fun:.4f}'
        f'  |  converged: {result.success}'
        f'  |  iters: {result.nit}'
    )

    _, log_vol_surface = _gas_filter(returns, init_coefs, basis_mat, result.x)
    vol_surface = np.exp(log_vol_surface / 2)   # log sigma2 -> sigma (std dev)
    return result.x, vol_surface, basis_mat


# ── goodness-of-fit ───────────────────────────────────────────────────────────

def goodness_of_fit(
    returns: np.ndarray,
    vol_surface: np.ndarray,
    true_variance: np.ndarray,
    nu_hat: float,
    warmup: int = WARMUP,
) -> dict:
    """Compute and print goodness-of-fit metrics."""
    true_vol     = np.sqrt(true_variance[:, warmup:])
    vol_hat      = vol_surface[:, warmup:]
    std_residuals = (returns[:, warmup:] / vol_hat).ravel()

    rmse = np.sqrt(np.mean((vol_hat - true_vol) ** 2))
    mae  = np.mean(np.abs(vol_hat - true_vol))
    pearson_r = float(np.corrcoef(vol_hat.ravel(), true_vol.ravel())[0, 1])
    r2 = 1 - np.sum((vol_hat - true_vol) ** 2) / np.sum((true_vol - true_vol.mean()) ** 2)

    residual_var = float(np.mean(std_residuals ** 2))
    t_dist_var   = nu_hat / (nu_hat - 2) if nu_hat > 2 else float('nan')
    ks_stat, ks_pval = stats.kstest(std_residuals, stats.t(df=nu_hat).cdf)

    print('\n── Goodness-of-fit ─────────────────────────────────────────')
    print(f'  Volatility RMSE:                    {rmse:.4f}')
    print(f'  Volatility MAE:                     {mae:.4f}')
    print(f'  Pearson r (vol_hat vs true_vol):    {pearson_r:.4f}')
    print(f'  R²        (vol_hat vs true_vol):    {r2:.4f}')
    print(f'  Mean z²  (E[z²] ~= {t_dist_var:.3f} under t(nu)): {residual_var:.4f}')
    print(f'  KS stat / p-value:                  {ks_stat:.4f} / {ks_pval:.4f}')
    print('────────────────────────────────────────────────────────────')

    return dict(
        rmse=rmse, mae=mae, pearson_r=pearson_r, r2=r2,
        residual_var=residual_var, t_dist_var=t_dist_var,
        ks_stat=ks_stat, ks_pval=ks_pval,
    )


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_surfaces(vol_surface: np.ndarray, true_variance: np.ndarray, warmup: int = WARMUP) -> None:
    """Side-by-side 3D surface plots: true vs estimated volatility."""
    true_vol = np.sqrt(true_variance[:, warmup:])
    vol_hat  = vol_surface[:, warmup:]
    n_grid, n_days = vol_hat.shape
    intraday_grid, day_grid = np.meshgrid(np.arange(n_grid), np.arange(n_days))

    fig = plt.figure(figsize=(18, 7))
    for col, (surf, title) in enumerate([
        (true_vol.T, 'True volatility  sigma_t(u)'),
        (vol_hat.T,  'Estimated volatility  sigma_hat_t(u)'),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.plot_surface(intraday_grid, day_grid, surf, cmap=cm.plasma, alpha=0.85)
        ax.set_xlabel('Intraday grid u', fontsize=11)
        ax.set_ylabel('Day t',           fontsize=11)
        ax.set_zlabel('Volatility',      fontsize=11)
        ax.set_title(title,              fontsize=13)
    plt.suptitle('Volatility surface: true vs GAS-GARCH estimate', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('── Simulating data ──────────────────────────────────────────')
    returns, true_variance = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)
    print(f'Return matrix shape  : {returns.shape}')
    print(f'True variance range  : [{true_variance.min():.2f}, {true_variance.max():.2f}]')

    print('\n── Estimating diagonal GAS-GARCH ────────────────────────────')
    params_hat, vol_surface, basis_mat = fit_gas(returns)

    nu_hat      = params_hat[0]
    ou_scale_hat = params_hat[1]
    print(f'\nEstimated ν (Student-t df):  {nu_hat:.3f}')
    print(f'Estimated δ (OU scale):      {ou_scale_hat:.5f}')

    metrics = goodness_of_fit(returns, vol_surface, true_variance, nu_hat)

    print('\n── Plotting ─────────────────────────────────────────────────')
    plot_surfaces(vol_surface, true_variance)


if __name__ == '__main__':
    main()
