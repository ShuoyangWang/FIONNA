# Run from the repository root with:
#   source("examples/run_example.R")
#
# Or run directly with:
#   Rscript examples/run_example.R
#
# For a quick check, set n = 400 and epochs = 30 below.


script_path <- tryCatch(
  normalizePath(sys.frames()[[1L]]$ofile),
  error = function(e) NA_character_
)
if (is.na(script_path)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1L])) else NA_character_
}
repo_root <- if (is.na(script_path)) getwd() else dirname(dirname(script_path))

source(file.path(repo_root, "R", "fionna.R"))
source(file.path(repo_root, "data", "simulate_shared_profile.R"))


# Selected illustrative setting from the completed simulations:
# multi_shared, n = 1600, J = 4, T = 60, K_j = 2.
dat <- simulate_fionna_shared_profile(
  n = 1600L,
  p = 8L,
  T = 60L,
  J = 4L,
  delta_target = 0,
  seed = 20260704L
)

fit <- fionna_fit(
  Y = dat$Y,
  X = dat$X,
  M_list = dat$M_list,
  Z = dat$Z,
  grid_list = dat$grid_list,
  K = c(2L, 2L, 2L, 2L),
  L = 5L,
  hidden = c(128L, 64L, 32L),
  epochs = 120L,
  lr = 5e-4,
  weight_decay = 1e-4,
  lambda_gamma = 0,
  seed = 20270704L
)


cat("\nTruth used in the simulated example:\n")
print(dat$truth)

cat("\nFIONNA point estimates, standardized scale:\n")
print(round(c(
  theta_hat = fit$theta,
  alpha_hat = fit$alpha,
  delta_hat = fit$delta
), 4))

cat("\nIndirect-effect inference:\n")
print(round(c(
  se_delta = fit$indirect_inference$se,
  p_delta_coordinate = fit$indirect_inference$coordinate_p,
  p_delta_joint = fit$indirect_inference$joint_p
), 4))

cat("\n95% confidence interval for the indirect effect:\n")
print(round(fit$indirect_inference$ci, 4))

cat("\nEstimated beta function names:\n")
print(names(fit$beta))
