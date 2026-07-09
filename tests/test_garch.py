"""Smoke tests for the functional GARCH model."""

import numpy as np
import pytest
from funcgarch import delta, kernel_operator, garch_filter, garch_estimator

N_GRID, N_DAYS, N_BASIS = 20, 30, 2
PARAMS = np.array([0.05] * N_BASIS + [0.1] * N_BASIS ** 2 + [0.8] * N_BASIS ** 2)


def _make_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, size=(N_GRID, N_DAYS))


def test_delta_scalar():
    coefs = np.array([0.1, 0.2])
    val = delta(coefs, 0.5, n_basis=2, init=0.0)
    assert np.isfinite(val)


def test_delta_vectorised():
    coefs = np.array([0.1, 0.2])
    grid = np.linspace(1 / N_GRID, 1 - 1 / N_GRID, N_GRID)
    vals = delta(coefs, grid, n_basis=2, init=np.zeros(N_GRID))
    assert vals.shape == (N_GRID,)
    assert np.all(np.isfinite(vals))


def test_kernel_operator_shape():
    grid = np.linspace(1 / N_GRID, 1 - 1 / N_GRID, N_GRID)
    coefs = np.ones(N_BASIS ** 2) * 0.1
    K = kernel_operator(grid, coefs, n_basis=N_BASIS, init=np.zeros((N_GRID, N_GRID)))
    assert K.shape == (N_GRID, N_GRID)


def test_garch_filter_shape():
    returns = _make_returns()
    variance_surface = garch_filter(
        returns, n_grid=N_GRID, params=PARAMS, n_basis=N_BASIS,
        initial_variance=np.ones(N_GRID),
    )
    assert variance_surface.shape == (N_GRID, N_DAYS)


def test_garch_filter_non_negative():
    returns = _make_returns()
    variance_surface = garch_filter(
        returns, n_grid=N_GRID, params=PARAMS, n_basis=N_BASIS,
        initial_variance=np.ones(N_GRID),
    )
    assert np.all(variance_surface >= 0)


def test_garch_estimator_scalar():
    returns = _make_returns()
    loss = garch_estimator(
        returns, n_grid=N_GRID, params=PARAMS, n_basis=N_BASIS,
        initial_variance=np.ones(N_GRID),
    )
    assert np.isfinite(loss)
    assert loss >= 0
