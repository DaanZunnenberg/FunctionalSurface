#!/usr/bin/env python3
"""
Same simulated data/seed as gas_vol_surface_container.py and
fgarch_vol_surface_container.py, but instead of comparing each estimator
against the true surface, this compares the two estimators directly against
each other: functional GARCH(1,1) on the left, GAS-GARCH on the right.
Same site theming, shapes, and color palette as the other two figures.

Usage
-----
    python scripts/garch_vs_gas_container.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from gas_vol_surface import N_GRID, N_DAYS, SEED, simulate_vol_surface, fit_gas
from gas_vol_surface_container import plot_two_surfaces
from funcgarch.garch import fit, garch_filter

N_BASIS = 3  # Bernstein basis functions per dimension, matching fgarch_vol_surface_container.py


def main() -> None:
    print('── Simulating data ──────────────────────────────────────────')
    mY, sigma2_true = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)
    print(f'Return matrix shape : {mY.shape}')

    print('\n── Estimating functional GARCH(1,1) (Bernstein QMLE) ─────────')
    n_params = N_BASIS + 2 * N_BASIS ** 2
    initial_variance = np.full(N_GRID, mY.var())
    garch_result = fit(
        mY, initial_variance=initial_variance, n_grid=N_GRID, n_basis=N_BASIS,
        x0=np.zeros(n_params),
        bounds=[(-0.99, 0.99)] * n_params,
        method='SLSQP',
        print_convergence=True,
        options={'maxiter': 500, 'disp': True},
    )
    sigma2_garch = garch_filter(mY, N_GRID, garch_result.x, N_BASIS, initial_variance)
    sigma_garch = np.sqrt(np.clip(sigma2_garch, 1e-6, None))

    print('\n── Estimating diagonal GAS-GARCH ────────────────────────────')
    vtheta_hat, sigma_gas, basis_mat = fit_gas(mY)

    diff_rmse = np.sqrt(np.mean((sigma_garch - sigma_gas) ** 2))
    print(f'\nGARCH-vs-GAS RMSE: {diff_rmse:.4f}')

    print('\n── Plotting (site theme) ─────────────────────────────────────')
    plot_two_surfaces(
        sigma_garch, sigma_gas,
        title_left='GARCH-estimate  $\\hat\\sigma_t(u)$',
        title_right='GAS-estimate  $\\hat\\sigma_t(u)$',
        out_path='garch_vs_gas_vol_surface.png',
        suptitle='Volatility surface: GARCH-estimate versus GAS-estimate',
    )


if __name__ == '__main__':
    main()
