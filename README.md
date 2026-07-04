# FIONNA

This folder contains a compact R implementation of **Functional Inference with Orthogonal Neural Network Adjustment (FIONNA)** for a scalar outcome, one or more scalar exposures, multiple functional mediators, and baseline covariates.

The code is organized so the method file is separate from the simulated data and the runnable example.

## Files

- `R/fionna.R`: method-only FIONNA implementation.
- `data/simulate_shared_profile.R`: simulated four-functional-mediator data generator.
- `examples/run_example.R`: example script that sources the method and data files, then runs FIONNA.

## Description

The function `fionna_fit()` in `R/fionna.R` implements the estimation procedure:

1. Standardizes `Y`, the columns of `X`, and the columns of `Z`.
2. Centers each functional mediator.
3. Performs per-mediator FPCA and stacks standardized FPCA scores.
4. Uses cross-fitted ReLU neural networks to estimate the nuisance regressions of `Y`, `X`, and stacked mediator scores on `Z`.
5. Forms the doubly residualized exposure.
6. Fits the residualized outcome regression with optional block-ridge shrinkage on mediator score coefficients.
7. Returns total, direct, and indirect point estimates.
8. Provides Wald inference for the indirect effect only.
9. Reconstructs the mediator coefficient functions `beta_j(t)`.

Direct and total effects are returned as point estimates, but this example exposes inference only for the indirect effect.

## Example

From the completed simulation summaries, I ranked FIONNA null settings by closeness of 95% Wald coverage to 0.95 with priority:

1. indirect effect,
2. total effect,
3. direct effect.

The selected illustrative setting is the four-functional-mediator shared-profile design:

- `n = 1600`
- `J = 4` functional mediators
- `T = 60` grid points per mediator
- `p = 8` covariates
- `K_j = 2` FPCA components per mediator
- `L = 5` cross-fitting folds
- neural network architecture `(128, 64, 32)`

In the completed Monte Carlo results, this setting gave FIONNA indirect-effect coverage approximately `0.9467`, closest to the nominal 95% target among the FIONNA rows considered. The corresponding total and direct coverage were approximately `0.8900` and `0.7100`; therefore, this example should be interpreted primarily as an indirect-effect inference example.

## Setup

Install R package dependencies:

```r
install.packages("torch")
torch::install_torch()
```

The rest of the code uses base R and `stats`.

## Run The Example

From this folder:

```r
source("examples/run_example.R")
```

The default run uses the selected illustrative setting with `n = 1600` and four mediators. It prints:

- `theta_hat`: total-effect point estimate,
- `alpha_hat`: direct-effect point estimate,
- `delta_hat`: indirect-effect point estimate,
- `se_delta`: indirect-effect standard error,
- `p_delta_joint`: chi-square Wald p-value for the indirect effect,
- a 95% confidence interval for the indirect effect.

For a quick smoke test, use a smaller sample size and fewer neural-network epochs:

```r
source("R/fionna.R")
source("data/simulate_shared_profile.R")

dat <- simulate_fionna_shared_profile(n = 400)
fit <- fionna_fit(
  Y = dat$Y,
  X = dat$X,
  M_list = dat$M_list,
  Z = dat$Z,
  grid_list = dat$grid_list,
  K = c(2, 2, 2, 2),
  L = 5,
  epochs = 30
)
fit$indirect_inference$ci
```

## Personalized Data

Your inputs should have the following form:

```r
fit <- fionna_fit(
  Y = Y,                         # numeric vector, length n
  X = X,                         # n by d matrix of scalar exposure(s)
  M_list = list(M1, M2, M3, M4),  # each Mj is n by T_j
  Z = Z,                         # n by p matrix of confounders
  grid_list = list(t1, t2, t3, t4),
  K = c(2, 2, 2, 2),
  L = 5,
  hidden = c(128, 64, 32),
  epochs = 120
)
```

Then read the indirect-effect inference:

```r
fit$delta
fit$indirect_inference$se
fit$indirect_inference$ci
fit$indirect_inference$joint_p
```

The reconstructed coefficient functions are in:

```r
fit$beta
```

## Notes

- Categorical confounders should be dummy-coded before passing them to `Z`.
- The returned effects are on the standardized scale because the algorithm standardizes `Y` and `X`.
- If trajectories are irregular or noisy, smooth or reconstruct them on a common grid before calling `fionna_fit()`.
- The example uses `torch` neural networks for the nuisance functions. For large simulation studies, consider GPU acceleration or reducing `epochs` during preliminary runs.
