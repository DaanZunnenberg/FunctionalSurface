#!/usr/bin/env python3
"""
Functional GAS-GARCH volatility surface estimation — site-themed render
=========================================================================
Identical experiment to gas_vol_surface.py (same simulation, filter, and
estimation), but plot_surfaces() is restyled to match the dark navy /
mono-type theme of daanzunnenberg.github.io instead of matplotlib defaults,
and the figure is saved to disk (as used for the site's project imagery)
instead of shown interactively.

Usage
-----
    pip install -e .                               # install the package first
    python scripts/gas_vol_surface_container.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy import stats

from funcgarch.basis import cubic_bspline_basis, ou_kernel

from gas_vol_surface import (
    N_GRID, N_DAYS, DK, N_INT_KNOTS, SEED, MAXITER, WARMUP,
    simulate_vol_surface, fit_gas, goodness_of_fit,
)


# ── site theme ─────────────────────────────────────────────────────────────
# Pulled from assets/css/style.css so the figure sits comfortably next to the
# rest of daanzunnenberg.github.io rather than looking like a bare matplotlib
# default plot.
BG_950     = '#0b0f18'   # hero background
BG_PANEL   = '#0c1830'   # --blue-950
LINE_SOFT  = '#232d47'   # hero border
INK        = '#e7ebf5'   # near-white text on dark bg
INK_FAINT  = '#8f9ab3'
MONO_FONT  = ['SF Mono', 'IBM Plex Mono', 'Menlo', 'Consolas', 'monospace']

# Shared purple-to-yellow hue (classical plasma-style palette), used for
# both the true and the estimated surface (so color encodes volatility
# level consistently across the two panels, rather than one color per
# panel).
_VOL_CMAP = LinearSegmentedColormap.from_list(
    'site-purple-yellow', ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf'])

plt.rcParams['font.family'] = MONO_FONT
plt.rcParams['text.color']  = INK
plt.rcParams['axes.edgecolor'] = LINE_SOFT
plt.rcParams['axes.labelcolor'] = INK_FAINT
plt.rcParams['xtick.color'] = INK_FAINT
plt.rcParams['ytick.color'] = INK_FAINT


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_surfaces(
    sigma_hat: np.ndarray,
    sigma2_true: np.ndarray,
    warmup: int = WARMUP,
    out_path: str = 'gas_vol_surface_container.png',
) -> None:
    """Side-by-side 3D surface plots: true vs estimated volatility, site-themed."""
    sigma_true = np.sqrt(sigma2_true[:, warmup:])
    s_hat      = sigma_hat[:, warmup:]

    # Block-average along the day axis before rendering: with T ~ 500 days,
    # mplot3d's per-day triangulation strips are wafer-thin, and at the
    # figure's viewing angle the renderer's depth-sorting of those strips
    # produces spiky, moiré-like artefacts (needle-like day-to-day jitter)
    # that aren't part of the underlying, otherwise-smooth surface. A block
    # average — not simple decimation, which would just pick out whichever
    # sample lands on the noisiest day — collapses that render noise while
    # leaving the shape of the surface intact.
    block = 4
    T_full = s_hat.shape[1]
    T_trim = (T_full // block) * block
    sigma_true = sigma_true[:, :T_trim].reshape(sigma_true.shape[0], -1, block).mean(axis=2)
    s_hat      = s_hat[:, :T_trim].reshape(s_hat.shape[0], -1, block).mean(axis=2)

    n, T = s_hat.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(T) * block)
    rcount = T
    ccount = n

    # Shared value range so the same color always encodes the same
    # volatility level on both panels, not just within each panel.
    vmin = min(sigma_true.min(), s_hat.min())
    vmax = max(sigma_true.max(), s_hat.max())

    fig = plt.figure(figsize=(16, 6.4), facecolor=BG_950)
    for col, (surf, title) in enumerate([
        (sigma_true.T, 'True volatility  $\\sigma_t(u)$'),
        (s_hat.T,      'Estimated volatility  $\\hat\\sigma_t(u)$'),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d', facecolor=BG_950)
        ax.plot_surface(
            X, Y, surf, cmap=_VOL_CMAP, vmin=vmin, vmax=vmax,
            alpha=0.92, linewidth=0, antialiased=True,
            rcount=rcount, ccount=ccount,
        )

        ax.set_xlabel('Intraday grid  u', fontsize=11, labelpad=10)
        ax.set_ylabel('Day  t',            fontsize=11, labelpad=10)
        ax.set_zlabel('Volatility',        fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=13, color=INK, pad=14)

        ax.xaxis.set_pane_color((0, 0, 0, 0))
        ax.yaxis.set_pane_color((0, 0, 0, 0))
        ax.zaxis.set_pane_color((0, 0, 0, 0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo['grid']['color'] = LINE_SOFT
            axis._axinfo['grid']['linewidth'] = 0.4
        ax.tick_params(colors=INK_FAINT, labelsize=9)

    fig.suptitle(
        'Volatility surface: true versus GAS-estimate',
        fontsize=14, color=INK, y=1.02,
    )
    plt.tight_layout()
    fig.subplots_adjust(wspace=-0.35)
    fig.savefig(out_path, dpi=220, facecolor=BG_950, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved themed surface comparison to {out_path}')


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
    goodness_of_fit(mY, sigma_hat, sigma2_true, nu_hat)

    # 4. Plot
    print('\n── Plotting (site theme) ─────────────────────────────────────')
    plot_surfaces(sigma_hat, sigma2_true)


if __name__ == '__main__':
    main()
