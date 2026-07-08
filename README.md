# FuncGARCH

Research code for **functional GARCH** and **functional GAS-GARCH** models applied to intraday volatility surfaces. The models treat the within-day return curve as a functional observation and extend classical GARCH/GAS dynamics to the function space.

---

## Theory

### Setting

Fix a trading day and let $\{r_t(u) : u \in [0,1]\}_{t=1}^T$ denote the sequence of intraday log-return curves, where $u$ indexes time within the day ($u = 0$ is market open, $u = 1$ is market close) and $t$ indexes the day.

The conditional variance curve on day $t$ given the filtration $\mathcal{F}_{t-1}$ is the functional object $\sigma_t^2(\cdot) \in L^2[0,1]$.

---

### Functional GARCH

The **functional GARCH(1,1)** model specifies:

$$r_t(u) = \sigma_t(u)\,\varepsilon_t(u), \quad \varepsilon_t \overset{iid}{\sim} (0, \mathrm{Id})$$

$$\sigma_t^2(u) = \delta(u) + \int_0^1 \alpha(u,s)\,r_{t-1}^2(s)\,ds + \int_0^1 \beta(u,s)\,\sigma_{t-1}^2(s)\,ds$$

where:
- $\delta \in L^2[0,1]$ is the **level function** (unconditional variance shape).
- $\alpha, \beta \in L^2([0,1]^2)$ are **kernel operators** governing shock transmission and persistence across intraday times.

#### Bernstein Basis

The kernel operators and level function are approximated in a finite-dimensional **Bernstein polynomial basis** $\{\varphi_k^M\}_{k=1}^M$:

$$\varphi_k^M(u) = \binom{M-1}{k-1} u^{k-1}(1-u)^{M-k}, \quad u \in [0,1]$$

The parametrisation is:

$$\delta(u) = \sum_{k=1}^M c_k\,\varphi_k^M(u), \qquad \alpha(u,s) = \sum_{k=1}^M \sum_{l=1}^M a_{kl}\,\varphi_k^M(u)\,\varphi_l^M(s)$$

with an identical expansion for $\beta$. The full parameter vector is $\theta = (c_{1:M},\, a_{11:MM},\, b_{11:MM}) \in \mathbb{R}^{M + 2M^2}$.

#### Estimation

Parameters are estimated by minimising the **Bernstein-projected MSE**:

$$\hat{\theta} = \arg\min_\theta \sum_{t=2}^T \sum_{k=1}^M \int_0^1 \left(r_t(u)^2 - \sigma_t^2(u;\theta)\right)^2 \varphi_k^M(u)\,du$$

Integrals are approximated on a uniform intraday grid of length $N$. Optimisation uses `scipy.minimize` with SLSQP.

---

### Functional GAS-GARCH

The **Generalized Autoregressive Score (GAS)** extension replaces fixed kernel operators with a **score-driven update** on a low-dimensional B-spline coefficient vector $b_t$.

Let $\Phi(u) = (\phi_1(u), \ldots, \phi_M(u))^\top$ be a cubic B-spline basis evaluated at $u$. The log-volatility curve is parametrised as:

$$\log \sigma_t(u) = \Phi(u)^\top b_t$$

The coefficient vector evolves according to the GAS recursion:

$$b_t = \omega + B\,b_{t-1} + A\,s_{t-1}$$

where $s_{t-1}$ is the **scaled score** of the conditional log-likelihood with respect to $b_{t-1}$, and $(\omega, B, A)$ are $M$-dimensional and $M \times M$ parameter matrices.

#### Likelihood

The return vector $r_t \in \mathbb{R}^n$ is modelled as a multivariate Student-$t$ with $\nu$ degrees of freedom:

$$r_t \mid \mathcal{F}_{t-1} \sim t_\nu\!\left(0,\, S_t \Lambda_\delta S_t\right)$$

where $S_t = \mathrm{diag}(\exp(\sigma_t(u_i)/2))$ scales individual volatilities and $\Lambda_\delta$ is an **Ornstein-Uhlenbeck covariance kernel**:

$$[\Lambda_\delta]_{ij} = \exp\!\left(-\frac{|u_i - u_j|}{\delta}\right)$$

This captures the intraday autocorrelation structure with a single length-scale parameter $\delta$.

#### Score Update

The score contribution is:

$$s_t = -\frac{1}{2}\,\Phi\,\mathbf{1}_n + \frac{\nu + n}{2\nu} \cdot \frac{1}{A_1}\,\Phi\,(\Lambda_\delta^{-1} S_t^{-1} r_t)$$

where $A_1 = 1 + r_t^\top (S_t \Lambda_\delta S_t)^{-1} r_t\,/\,\nu$ is the scalar quadratic form. Parameters $(\nu, \delta, \omega, B, A)$ are estimated by maximising the average log-likelihood.

---

## Repository Layout

```
funcgarch/               Python package
  __init__.py            Public API
  basis.py               bernstein_basis, cubic_bspline_basis, ou_kernel
  garch.py               delta, functional_operator, loss_func, garch_filter,
                         garch_estimator, fit
  gas.py                 gas_garch_estimator, func_garch_estimator
  simulate.py            brownian, simulate
  utils.py               ResultContainer
  context_manager.py     Jupyter module hot-reload utility

data/
  taq_cleaner.py         Cleans WRDS TAQ CSV exports into return matrices
  taq_fetcher.ipynb      TAQ data download notebook

notebooks/
  example.ipynb              Minimal functional GARCH example
  func_garch_snp.ipynb       S&P 500 intraday volatility estimation
  func_garch_gas.ipynb       Functional GAS-GARCH (latest)
  func_garch_gas_example.ipynb  Extended example with diagnostics

reference/
  func_gas_volatility.m      MATLAB reference implementation
  construct_likelihood_repara.m
  fdaM/                      FDA toolbox (Ramsay et al.)

wrds/
  data_fetcher.sas       WRDS TAQ data extraction
  taq_cleaner.sas        Minute-bar resampling
  nbbo_minute_reader.sas NBBO quote aggregation
  dynamic_taq_minute.sas Dynamic TAQ extraction
  export.sas             SAS cloud export helper

price_data_example.csv   Sample price data for quick testing
setup.py
requirements.txt
```

---

## Quick Start

```python
import numpy as np
from funcgarch import fit, garch_filter

# mY: (N, T) log-return matrix — N intraday grid points, T days
N, T = mY.shape
M = 2  # number of Bernstein basis functions

result = fit(
    mY,
    sigma2_ini=np.ones(N),
    grid_length=N,
    M=M,
    x0=np.zeros(M + 2 * M**2),
    bounds=[(-.99, .99)] * (M + 2 * M**2),
    method='SLSQP',
)

theta_hat  = result.x
sigma2_hat = garch_filter(mY, grid_length=N, vtheta=theta_hat, M=M, sigma2_ini=np.ones(N))
```

For the GAS-GARCH model:

```python
import numpy as np
from funcgarch import gas_garch_estimator, cubic_bspline_basis
from scipy.optimize import minimize

dK = 7
n  = mY.shape[0]
vtau = np.linspace(0, 1, n)
basis_mat = cubic_bspline_basis(vtau, k=dK - 2, n_interior_knots=3)  # (M, n)
M = dK + 1

vb0     = np.ones((M, 1))
vtheta0 = np.concatenate(([2.1, 0.001], np.ones(M), -0.5 * np.ones(M**2), 0.1 * np.ones(M**2)))
LB      = np.concatenate(([1.05, 1e-5], -5 * np.ones(M), -2 * np.ones(M**2), -0.9 * np.ones(M**2)))
UB      = np.concatenate(([50,   1],    15 * np.ones(M),  2 * np.ones(M**2),  0.9 * np.ones(M**2)))

objective = lambda vtheta: gas_garch_estimator(mY, vb0, dK, n, basis_mat, vtheta)[0]
result    = minimize(objective, vtheta0, bounds=list(zip(LB, UB)), method='SLSQP')
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.10 required.

---

## References

- Aue, A., Norinho, D. D., & Hörmann, S. (2015). On the prediction of stationary functional time series. *Journal of the American Statistical Association*, 110(509), 378–392.
- Hörmann, S., Kidziński, Ł., & Hallin, M. (2015). Dynamic functional principal components. *Journal of the Royal Statistical Society: Series B*, 77(2), 319–348.
- Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777–795.
