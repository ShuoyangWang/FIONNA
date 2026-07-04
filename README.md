# FIONNA

This folder contains a compact R implementation of **Functional Inference with Orthogonal Neural Network Adjustment (FIONNA)** for a scalar outcome, one or more scalar exposures, multiple functional mediators, and baseline covariates.

## Files

- `R/fionna.R`: FIONNA implementation.
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

## Personalized Data

The inputs should have the following form:

```r
fit <- fionna_fit(
  Y = Y,                         # numeric vector, length n
  X = X,                         # n by d matrix of scalar exposure(s)
  M_list = list(M1, M2, M3, M4),  # each Mj is n by T_j
  Z = Z,                         # n by p matrix of confounders
  grid_list = list(t1, t2, t3, t4),
  K = c(3, 3, 3, 3),
  L = 5,
  hidden = c(128, 64, 32),
  epochs = 300
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
- The example uses `torch` neural networks for the nuisance functions. For large simulation studies, consider GPU acceleration.
