#!/usr/bin/env python3
"""
Same simulated data as gas_vol_surface_container.py (identical seed), but
estimated with the actual functional GARCH(1,1) model (funcgarch.garch,
Bernstein-basis QMLE) instead of the GAS recursion, and rendered with the
same site theming — output overwrites garch_vol_surface.png so the label
("GARCH-estimate") on that figure now matches what actually produced it.

Usage
-----
    python scripts/fgarch_vol_surface_container.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from gas_vol_surface import N_GRID, N_DAYS, SEED, simulate_vol_surface
from gas_vol_surface_container import plot_surfaces
from funcgarch.garch import fit, garch_filter

N_BASIS = 3  # Bernstein basis functions per dimension — keeps the M + 2M^2 = 21-param optimization fast


def main() -> None:
    print('── Simulating data ──────────────────────────────────────────')
    mY, sigma2_true = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)
    print(f'Return matrix shape : {mY.shape}')
    print(f'True σ² range       : [{sigma2_true.min():.2f}, {sigma2_true.max():.2f}]')

    print('\n── Estimating functional GARCH(1,1) (Bernstein QMLE) ─────────')
    n_params = N_BASIS + 2 * N_BASIS ** 2
    initial_variance = np.full(N_GRID, mY.var())
    result = fit(
        mY, initial_variance=initial_variance, n_grid=N_GRID, n_basis=N_BASIS,
        x0=np.zeros(n_params),
        bounds=[(-0.99, 0.99)] * n_params,
        method='SLSQP',
        print_convergence=True,
        options={'maxiter': 500, 'disp': True},
    )
    params_hat = result.x
    print(f'\nConverged: {getattr(result, "success", None)}')

    sigma2_hat = garch_filter(mY, N_GRID, params_hat, N_BASIS, initial_variance)
    sigma2_hat = np.clip(sigma2_hat, 1e-6, None)  # guard against tiny negative numerical noise
    sigma_hat = np.sqrt(sigma2_hat)

    rmse = np.sqrt(np.mean((sigma_hat - np.sqrt(sigma2_true)) ** 2))
    print(f'\nVolatility RMSE: {rmse:.4f}')

    print('\n── Plotting (site theme) ─────────────────────────────────────')
    plot_surfaces(
        sigma_hat, sigma2_true,
        out_path='garch_vol_surface.png',
        suptitle='Volatility surface: true versus GARCH-estimate',
    )


if __name__ == '__main__':
    main()
