import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funcgarch import cubic_bspline_basis, gas_garch_estimator, func_garch_estimator

# ── Simulated data ────────────────────────────────────────────────────────────
n_days = 1000
n_grid = 25

true_variance = np.zeros((n_grid, n_days))
for t in range(n_days):
    for i in range(n_grid):
        true_variance[i, t] = (
            1
            + 10 * (i - (n_grid / 2 + n_grid / 4 * np.sin(t * 3 * np.pi / n_days))) ** 2
            / (3 * n_grid / 4) ** 2
            + 3
            + 2 * np.sin(t * 2 * np.pi / n_days)
        )
returns = np.sqrt(true_variance) * np.random.standard_normal((n_grid, n_days))

# ── Basis ─────────────────────────────────────────────────────────────────────
spline_order     = 2   # dK - 2 in original; cubic_bspline_basis `order` argument
n_interior_knots = 3
n_basis          = spline_order + n_interior_knots + 2   # = 7 (including constant row)
grid             = np.linspace(0, 1, n_grid)
basis_mat        = cubic_bspline_basis(grid, spline_order, n_interior_knots)

# ── GAS estimation ────────────────────────────────────────────────────────────
init_coefs      = np.ones((n_basis, 1))
gas_init_params = np.concatenate((
    [2.1, 0.001],
    np.ones(n_basis),
    -0.05 * np.ones(n_basis ** 2),
     0.01 * np.ones(n_basis ** 2),
))
gas_lower_bounds = np.concatenate(([1.05, 0.00001], -5 * np.ones(n_basis), -2 * np.ones(n_basis ** 2), -0.9 * np.ones(n_basis ** 2)))
gas_upper_bounds = np.concatenate(([500,  1],        15 * np.ones(n_basis),  2 * np.ones(n_basis ** 2),  0.9 * np.ones(n_basis ** 2)))

gas_objective = lambda p: gas_garch_estimator(returns, init_coefs, basis_mat, p)[0]
gas_result    = minimize(
    gas_objective, gas_init_params,
    bounds=list(zip(gas_lower_bounds, gas_upper_bounds)),
    method='SLSQP',
    options={'maxiter': 200},
)
gas_params_hat = gas_result.x
print(gas_result)

gas_log_vol_surface = gas_garch_estimator(returns, init_coefs, basis_mat, gas_params_hat)[1]
gas_vol_surface     = np.exp(gas_log_vol_surface / 2)   # log sigma2 -> sigma (std dev)

# ── GARCH estimation ──────────────────────────────────────────────────────────
garch_bounds      = [(0, 100)] * n_basis + [(-.99, .99)] * (2 * n_basis ** 2)
garch_init_params = np.array(
    [0.0] * n_basis
    + (0.1 * np.eye(n_basis).flatten()).tolist()
    + (0.8 * np.eye(n_basis).flatten()).tolist()
)

garch_objective = lambda p: func_garch_estimator(returns, basis_mat, p)[0]
garch_result    = minimize(
    garch_objective, garch_init_params,
    bounds=garch_bounds,
    method='Nelder-Mead',
    options={'maxiter': 2000},
)
garch_params_hat = garch_result.x
print(garch_result)

garch_variance_surface = func_garch_estimator(returns, basis_mat, garch_params_hat)[1]
garch_vol_surface      = garch_variance_surface ** 0.5   # variance -> std dev

# ── 3-D comparison: real vs GAS vs GARCH ─────────────────────────────────────
z_true  = true_variance[:, 4:]
z_gas   = gas_vol_surface[:, 4:]
z_garch = garch_vol_surface[:, 4:]

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=['True variance', 'GAS volatility', 'GARCH volatility'],
)
fig.add_trace(go.Surface(z=z_true,  name='True'),  row=1, col=1)
fig.add_trace(go.Surface(z=z_gas,   name='GAS'),   row=1, col=2)
fig.add_trace(go.Surface(z=z_garch, name='GARCH'), row=1, col=3)
fig.show()
