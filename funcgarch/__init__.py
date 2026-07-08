"""funcgarch — Functional GARCH and GAS-GARCH models.

Public API
----------
Basis functions:
    bernstein_basis       Bernstein polynomial basis function (JIT-compiled).
    cubic_bspline_basis   Cubic B-spline basis matrix.
    ou_kernel             Ornstein-Uhlenbeck covariance kernel.

Functional GARCH:
    delta                 Level operator.
    functional_operator   Kernel operator (double Bernstein sum).
    loss_func             Bernstein-projected MSE loss.
    garch_filter          Extract conditional variance surface from returns.
    garch_estimator       Compute loss for a given parameter vector.
    fit                   Estimate parameters via scipy.minimize.

Functional GAS-GARCH:
    gas_garch_estimator   Score-driven estimator (Student-t likelihood).
    func_garch_estimator  Functional GARCH via B-spline coefficient recursion.

Simulation:
    brownian              Scaled Brownian motion on [0, 1].
    simulate              Simulate a functional GARCH process.

Utilities:
    ResultContainer       Key-value container wrapping optimization results.
"""

from .basis import bernstein_basis, cubic_bspline_basis, ou_kernel
from .garch import delta, functional_operator, loss_func, garch_filter, garch_estimator, fit
from .gas import gas_garch_estimator, func_garch_estimator
from .simulate import brownian, simulate
from .utils import ResultContainer

__all__ = [
    'bernstein_basis',
    'cubic_bspline_basis',
    'ou_kernel',
    'delta',
    'functional_operator',
    'loss_func',
    'garch_filter',
    'garch_estimator',
    'fit',
    'gas_garch_estimator',
    'func_garch_estimator',
    'brownian',
    'simulate',
    'ResultContainer',
]
