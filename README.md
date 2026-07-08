# FuncGARCH

Research code for **functional GARCH** and **functional GAS-GARCH** models applied to intraday volatility surfaces. The models treat the within-day return curve as a functional observation and extend classical GARCH/GAS dynamics to function space.

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

The basis satisfies $\sum_{k=1}^M \varphi_k^M(u) = 1$ (partition of unity) and $\varphi_k^M(u) \geq 0$. The parametrisation is:

$$\delta(u) = \sum_{k=1}^M c_k\,\varphi_k^M(u), \qquad \alpha(u,s) = \sum_{k=1}^M \sum_{l=1}^M a_{kl}\,\varphi_k^M(u)\,\varphi_l^M(s)$$

with an identical expansion for $\beta$. The full parameter vector is $\theta = (c_{1:M},\, a_{11:MM},\, b_{11:MM}) \in \mathbb{R}^{M + 2M^2}$.

#### Estimation

Parameters are estimated by minimising the **Bernstein-projected MSE**:

$$\hat{\theta} = \arg\min_\theta \sum_{t=2}^T \sum_{k=1}^M \left\|\left(r_t^2 - \sigma_t^2(\cdot\,;\theta)\right)\varphi_k^M\right\|_{L^2}^2$$

where $\|f\|_{L^2}^2 = \int_0^1 f(u)^2\,du$. This is equivalent to weighting the squared residual $(r_t^2 - \sigma_t^2)^2$ by $(\varphi_k^M)^2$ and integrating, approximated on a uniform intraday grid of length $N$. Optimisation uses `scipy.minimize` with SLSQP.

---

### Functional GAS-GARCH

The **Generalized Autoregressive Score (GAS)** extension replaces fixed kernel operators with a **score-driven update** on a low-dimensional B-spline coefficient vector $b_t$.

Let $\Phi(u) = (\phi_1(u), \ldots, \phi_M(u))^\top$ be a cubic B-spline basis evaluated at $u$. The log-volatility curve is parametrised as:

$$\log \sigma_t(u) = \Phi(u)^\top b_t$$

The coefficient vector evolves according to the GAS recursion:

$$b_t = \omega + B\,b_{t-1} + A\,s_{t-1}$$

where $s_{t-1}$ is the **score** of the conditional log-likelihood with respect to $b_{t-1}$, and $(\omega, B, A)$ are $M$-dimensional and $M \times M$ parameter matrices.

#### Likelihood

The return vector $r_t \in \mathbb{R}^N$ is modelled as a multivariate Student-$t$ with $\nu$ degrees of freedom:

$$r_t \mid \mathcal{F}_{t-1} \sim t_\nu\!\left(0,\, S_t \Lambda_\delta S_t\right)$$

where $S_t = \mathrm{diag}(\exp(\sigma_t(u_i)/2))_{i=1}^N$ scales individual volatilities and $\Lambda_\delta$ is an **Ornstein-Uhlenbeck covariance kernel**:

$$[\Lambda_\delta]_{ij} = \exp\!\left(-\frac{|u_i - u_j|}{\delta}\right)$$

This captures the intraday autocorrelation structure with a single length-scale parameter $\delta$.

#### Score

Let $\tilde{r}_t = S_t^{-1} r_t$ be the element-wise standardised returns and $A_1 = 1 + \tilde{r}_t^\top \Lambda_\delta^{-1} \tilde{r}_t\,/\,\nu$ the scalar quadratic form. The score of the log-likelihood with respect to $b_t$ is:

$$s_t = -\frac{1}{2}\,\Phi\,\mathbf{1}_N + \frac{\nu + N}{2\nu\,A_1}\,\Phi\,\bigl(\tilde{r}_t \odot \Lambda_\delta^{-1}\tilde{r}_t\bigr)$$

where $\Phi = [\Phi(u_1)\;\cdots\;\Phi(u_N)]$ is the $(M \times N)$ basis matrix and $\odot$ denotes element-wise multiplication. Parameters $(\nu, \delta, \omega, B, A)$ are estimated by maximising the average log-likelihood.

---

## Repository Layout

### File tree

```
funcgarch/               Installable Python package
  __init__.py            Public API
  basis.py               bernstein_basis, cubic_bspline_basis, ou_kernel
  garch.py               delta, kernel_operator, loss_func, garch_filter,
                         garch_estimator, fit
  gas.py                 gas_garch_estimator, func_garch_estimator
  simulate.py            brownian, simulate
  utils.py               ResultContainer

tests/
  test_basis.py          Basis function correctness tests
  test_garch.py          GARCH operator and filter tests

examples/
  example.ipynb              Minimal functional GARCH example
  func_garch_snp.ipynb       S&P 500 intraday volatility estimation
  func_garch_gas.ipynb       Functional GAS-GARCH (latest)
  func_garch_gas_example.ipynb  Extended example with diagnostics
  gas_vol_surface.py         Standalone simulation + estimation script (see below)

scripts/
  taq_cleaner.py         Cleans WRDS TAQ CSV exports into return matrices
  taq_fetcher.ipynb      TAQ data download notebook

wrds/
  data_fetcher.sas       WRDS TAQ data extraction
  taq_cleaner.sas        Minute-bar resampling
  nbbo_minute_reader.sas NBBO quote aggregation
  dynamic_taq_minute.sas Dynamic TAQ extraction
  export.sas             SAS cloud export helper

reference/
  func_gas_volatility.m      MATLAB reference implementation
  construct_likelihood_repara.m
  fdaM/                      FDA toolbox (Ramsay et al.)

pyproject.toml
requirements.txt
```

### How the components connect

**Data pipeline** — from raw market data to the input matrix `mY`:

```
WRDS database
     │  (SAS scripts in wrds/)
     ▼
Raw TAQ exports (CSV)
     │  scripts/taq_fetcher.ipynb fetches; scripts/taq_cleaner.py cleans
     ▼
mY : ndarray (N, T)   ← N intraday grid points, T trading days
```

**Model estimation** — from `mY` to fitted parameters:

```
mY
 ├──▶ funcgarch/garch.py  ──  fit() → garch_filter()   [Bernstein GARCH]
 └──▶ funcgarch/gas.py    ──  gas_garch_estimator()    [B-spline GAS-GARCH]

examples/ notebooks demonstrate both paths end-to-end.
```

**Internal module dependencies** inside `funcgarch/`:

```
basis.py  ◀──── garch.py ◀──── simulate.py
    ▲                ▲
    └────── gas.py   └──────── utils.py   (ResultContainer)
```

`basis.py` has no internal dependencies and is the foundation everything else builds on.

---

## Quick Start

### Functional GARCH

```python
import numpy as np
from funcgarch import fit, garch_filter

# mY: (N, T) log-return matrix — N intraday grid points, T days
N, T = mY.shape
M = 2  # number of Bernstein basis functions

result = fit(
    mY,
    sigma2_init=np.ones(N),
    n_grid=N,
    M=M,
    x0=np.zeros(M + 2 * M**2),
    bounds=[(-.99, .99)] * (M + 2 * M**2),
    method='SLSQP',
)

theta_hat  = result.x
sigma2_hat = garch_filter(mY, n_grid=N, vtheta=theta_hat, M=M, sigma2_init=np.ones(N))
```

### Functional GAS-GARCH

```python
import numpy as np
from funcgarch import gas_garch_estimator, cubic_bspline_basis
from scipy.optimize import minimize

dK = 7
N  = mY.shape[0]
vtau = np.linspace(0, 1, N)
basis_mat = cubic_bspline_basis(vtau, order=dK - 2, n_interior_knots=3)  # (M, N)
M = dK + 1

vb0     = np.ones((M, 1))
vtheta0 = np.concatenate(([2.1, 0.001], np.ones(M), -0.5 * np.ones(M**2), 0.1 * np.ones(M**2)))
LB      = np.concatenate(([1.05, 1e-5], -5 * np.ones(M), -2 * np.ones(M**2), -0.9 * np.ones(M**2)))
UB      = np.concatenate(([50,   1],    15 * np.ones(M),  2 * np.ones(M**2),  0.9 * np.ones(M**2)))

objective = lambda vtheta: gas_garch_estimator(mY, vb0, dK, N, basis_mat, vtheta)[0]
result    = minimize(objective, vtheta0, bounds=list(zip(LB, UB)), method='SLSQP')
```

---

### GAS-GARCH simulation script

`examples/gas_vol_surface.py` is a self-contained script that:
1. Simulates a 500-day intraday return panel with a known, time-varying volatility surface.
2. Estimates the diagonal GAS-GARCH model (26 parameters for M = 8) by maximum likelihood.
3. Prints goodness-of-fit metrics: RMSE, R², Pearson correlation, residual variance calibration, and a KS test of standardised residuals against the fitted Student-t distribution.
4. Produces three figure windows: side-by-side 3D surfaces (true vs estimated), flattened time-series comparison, and residual diagnostics (histogram, QQ plot, ACF of squared residuals).

```bash
python examples/gas_vol_surface.py
```

The script uses `cubic_bspline_basis` and `ou_kernel` from the package and implements the **diagonal** score update `bₜ = ω + b⊙bₜ₋₁ + a⊙sₜ₋₁` (vectors b, a), which is the version used in `func_garch_gas.ipynb`.  The full matrix version (M×M matrices B, A) is in `gas_garch_estimator` in `gas.py`.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.10 required. Run the test suite with:

```bash
pytest tests/
```

---

## References

- Aue, A., Norinho, D. D., & Hörmann, S. (2015). On the prediction of stationary functional time series. *Journal of the American Statistical Association*, 110(509), 378–392.
- Hörmann, S., Kidziński, Ł., & Hallin, M. (2015). Dynamic functional principal components. *Journal of the Royal Statistical Society: Series B*, 77(2), 319–348.
- Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777–795.
