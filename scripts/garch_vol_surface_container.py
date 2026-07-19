#!/usr/bin/env python3
"""
Same render as gas_vol_surface_container.py (identical simulation, filter,
estimation, and seed), with the only difference being the figure title and
output filename — 'GARCH-estimate' instead of 'GAS-estimate'.

Usage
-----
    python scripts/garch_vol_surface_container.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gas_vol_surface import N_GRID, N_DAYS, SEED, simulate_vol_surface, fit_gas, goodness_of_fit
from gas_vol_surface_container import plot_surfaces


def main() -> None:
    print('── Simulating data ──────────────────────────────────────────')
    mY, sigma2_true = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)
    print(f'Return matrix shape : {mY.shape}')
    print(f'True σ² range       : [{sigma2_true.min():.2f}, {sigma2_true.max():.2f}]')

    print('\n── Estimating diagonal GAS-GARCH ────────────────────────────')
    vtheta_hat, sigma_hat, basis_mat = fit_gas(mY)

    nu_hat    = vtheta_hat[0]
    delta_hat = vtheta_hat[1]
    print(f'\nEstimated ν (Student-t df): {nu_hat:.3f}')
    print(f'Estimated δ (OU scale):     {delta_hat:.5f}')

    goodness_of_fit(mY, sigma_hat, sigma2_true, nu_hat)

    print('\n── Plotting (site theme) ─────────────────────────────────────')
    plot_surfaces(
        sigma_hat, sigma2_true,
        out_path='garch_vol_surface.png',
        suptitle='Volatility surface: true versus GARCH-estimate',
    )


if __name__ == '__main__':
    main()
