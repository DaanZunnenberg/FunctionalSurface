"""Smoke tests for basis functions."""

import numpy as np
import pytest
from funcgarch import bernstein_basis, ou_kernel
from funcgarch.basis import cubic_bspline_basis


def test_bernstein_partition_of_unity():
    """Bernstein basis must sum to 1 at every grid point."""
    n_basis, n_grid = 4, 50
    grid = np.linspace(0, 1, n_grid)
    total = sum(bernstein_basis(grid, n_basis, k) for k in range(1, n_basis + 1))
    np.testing.assert_allclose(total, np.ones(n_grid), atol=1e-12)


def test_bernstein_non_negative():
    n_basis, n_grid = 4, 100
    grid = np.linspace(0, 1, n_grid)
    for k in range(1, n_basis + 1):
        assert np.all(bernstein_basis(grid, n_basis, k) >= 0)


def test_ou_kernel_shape_and_symmetry():
    grid = np.linspace(0, 1, 30)
    kernel = ou_kernel(grid, delta=0.5)
    assert kernel.shape == (30, 30)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-14)


def test_ou_kernel_positive_definite():
    grid = np.linspace(0, 1, 20)
    kernel = ou_kernel(grid, delta=1.0)
    eigvals = np.linalg.eigvalsh(kernel)
    assert np.all(eigvals > 0)


def test_bspline_basis_shape():
    grid = np.linspace(0, 1, 50)
    basis = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=False)
    assert basis.ndim == 2
    assert basis.shape[1] == len(grid)


def test_bspline_basis_with_constant():
    grid = np.linspace(0, 1, 50)
    basis_no_const   = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=False)
    basis_with_const = cubic_bspline_basis(grid, order=4, n_interior_knots=3, create_constant=True)
    assert basis_with_const.shape[0] == basis_no_const.shape[0] + 1
    np.testing.assert_allclose(basis_with_const[0], 1.0)
