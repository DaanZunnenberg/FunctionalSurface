"""Smoke tests for basis functions."""

import numpy as np
import pytest
from funcgarch import bernstein_basis, ou_kernel
from funcgarch.basis import cubic_bspline_basis


def test_bernstein_partition_of_unity():
    """Bernstein basis must sum to 1 at every grid point."""
    M, N = 4, 50
    grid = np.linspace(0, 1, N)
    total = sum(bernstein_basis(grid, M, k) for k in range(1, M + 1))
    np.testing.assert_allclose(total, np.ones(N), atol=1e-12)


def test_bernstein_non_negative():
    M, N = 4, 100
    grid = np.linspace(0, 1, N)
    for k in range(1, M + 1):
        assert np.all(bernstein_basis(grid, M, k) >= 0)


def test_ou_kernel_shape_and_symmetry():
    grid = np.linspace(0, 1, 30)
    K = ou_kernel(grid, delta=0.5)
    assert K.shape == (30, 30)
    np.testing.assert_allclose(K, K.T, atol=1e-14)


def test_ou_kernel_positive_definite():
    grid = np.linspace(0, 1, 20)
    K = ou_kernel(grid, delta=1.0)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals > 0)


def test_bspline_basis_shape():
    grid = np.linspace(0, 1, 50)
    B = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=False)
    assert B.ndim == 2
    assert B.shape[1] == len(grid)


def test_bspline_basis_with_constant():
    grid = np.linspace(0, 1, 50)
    B_no_const  = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=False)
    B_with_const = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=True)
    assert B_with_const.shape[0] == B_no_const.shape[0] + 1
    np.testing.assert_allclose(B_with_const[0], 1.0)
