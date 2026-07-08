"""Smoke tests for the functional GARCH model."""

import numpy as np
import pytest
from funcgarch import delta, kernel_operator, garch_filter, garch_estimator


N, T, M = 20, 30, 2
_THETA = np.array([0.05] * M + [0.1] * M ** 2 + [0.8] * M ** 2)


def _make_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, size=(N, T))


def test_delta_scalar():
    coefs = np.array([0.1, 0.2])
    val = delta(coefs, 0.5, M=2, init=0.0)
    assert np.isfinite(val)


def test_delta_vectorised():
    coefs = np.array([0.1, 0.2])
    grid = np.linspace(1 / N, 1 - 1 / N, N)
    vals = delta(coefs, grid, M=2, init=np.zeros(N))
    assert vals.shape == (N,)
    assert np.all(np.isfinite(vals))


def test_kernel_operator_shape():
    grid = np.linspace(1 / N, 1 - 1 / N, N)
    coefs = np.ones(M ** 2) * 0.1
    K = kernel_operator(grid, coefs, M=M, init=np.zeros((N, N)))
    assert K.shape == (N, N)


def test_garch_filter_shape():
    mY = _make_returns()
    sigma2_mat = garch_filter(mY, n_grid=N, vtheta=_THETA, M=M, sigma2_init=np.ones(N))
    assert sigma2_mat.shape == (N, T)


def test_garch_filter_non_negative():
    mY = _make_returns()
    sigma2_mat = garch_filter(mY, n_grid=N, vtheta=_THETA, M=M, sigma2_init=np.ones(N))
    assert np.all(sigma2_mat >= 0)


def test_garch_estimator_scalar():
    mY = _make_returns()
    loss = garch_estimator(mY, n_grid=N, vtheta=_THETA, M=M, sigma2_init=np.ones(N))
    assert np.isfinite(loss)
    assert loss >= 0
