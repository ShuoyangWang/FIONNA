# Functional Inference with Orthogonal Neural Network Adjustment (FIONNA)
#
# Method-only file.
#
# Main function:
#   fionna_fit(Y, X, M_list, Z, grid_list, K, ...)
#
# The function implements:
#   1. standardization of Y, X, and Z;
#   2. per-mediator FPCA and stacked standardized FPCA scores;
#   3. cross-fitted ReLU neural-network nuisance adjustment;
#   4. doubly residualized exposure construction;
#   5. residualized block-ridge outcome regression;
#   6. total, direct, and indirect point estimates;
#   7. Wald inference for the indirect effect.
#
# Required package:
#   install.packages("torch")
#   torch::install_torch()


trapz_weights <- function(grid) {
  grid <- as.numeric(grid)
  if (length(grid) < 2L) stop("grid must have at least two points.")
  dx <- diff(grid)
  if (any(dx <= 0)) stop("grid must be strictly increasing.")

  w <- numeric(length(grid))
  w[1L] <- dx[1L] / 2
  w[length(grid)] <- dx[length(dx)] / 2
  if (length(grid) > 2L) {
    w[2L:(length(grid) - 1L)] <- (dx[-length(dx)] + dx[-1L]) / 2
  }
  w
}


standardize_vector <- function(y, eps = 1e-8) {
  y <- as.numeric(y)
  center <- mean(y)
  scale <- stats::sd(y)
  if (!is.finite(scale) || scale < eps) scale <- 1
  list(x = (y - center) / scale, center = center, scale = scale)
}


standardize_matrix <- function(x, eps = 1e-8) {
  x <- as.matrix(x)
  center <- colMeans(x)
  scale <- apply(x, 2L, stats::sd)
  scale[!is.finite(scale) | scale < eps] <- 1
  x_std <- sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  list(x = x_std, center = center, scale = scale)
}


safe_solve <- function(a, b = NULL, ridge = 1e-8) {
  a <- as.matrix(a)
  a <- a + diag(ridge, nrow(a))
  if (is.null(b)) return(solve(a))
  solve(a, b)
}


make_folds <- function(n, L = 5L, seed = 1L) {
  set.seed(seed)
  id <- sample.int(n)
  split(id, rep(seq_len(L), length.out = n))
}


fpca_one_mediator <- function(M, grid, K) {
  M <- as.matrix(M)
  grid <- as.numeric(grid)
  n <- nrow(M)
  if (ncol(M) != length(grid)) {
    stop("ncol(M) must match length(grid).")
  }
  if (K > min(n - 1L, ncol(M))) {
    stop("K is larger than the available rank.")
  }

  w <- trapz_weights(grid)
  mean_curve <- colMeans(M)
  M_centered <- sweep(M, 2L, mean_curve, "-")

  # Discrete L2 FPCA: SVD after multiplying columns by sqrt(integration weight).
  M_weighted <- sweep(M_centered, 2L, sqrt(w), "*")
  sv <- svd(M_weighted, nu = K, nv = K)

  values <- (sv$d[seq_len(K)]^2) / (n - 1)
  V <- sv$v[, seq_len(K), drop = FALSE]
  phi <- sweep(V, 1L, sqrt(w), "/")
  scores_raw <- M_weighted %*% V

  # Deterministic eigenfunction orientation.
  for (k in seq_len(K)) {
    anchor <- which.max(abs(phi[, k]))
    if (phi[anchor, k] < 0) {
      phi[, k] <- -phi[, k]
      scores_raw[, k] <- -scores_raw[, k]
    }
  }

  scores_std <- sweep(scores_raw, 2L, sqrt(pmax(values, .Machine$double.eps)), "/")

  list(
    scores = scores_std,
    raw_scores = scores_raw,
    eigenvalues = values,
    eigenfunctions = phi,
    mean_curve = mean_curve,
    grid = grid,
    weights = w,
    K = K
  )
}


fit_mlp_predict <- function(x_train,
                            y_train,
                            x_test,
                            hidden = c(128L, 64L, 32L),
                            epochs = 120L,
                            lr = 5e-4,
                            weight_decay = 1e-4,
                            seed = 1L,
                            verbose = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "The R package 'torch' is required. Install with: ",
      "install.packages('torch'); torch::install_torch()"
    )
  }

  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  y_train <- as.matrix(y_train)
  if (nrow(y_train) != nrow(x_train)) {
    y_train <- matrix(y_train, nrow = nrow(x_train))
  }

  torch::torch_manual_seed(seed)

  mlp_module <- torch::nn_module(
    "FionnaMLP",
    initialize = function(input_dim, output_dim, hidden) {
      layers <- list()
      last_dim <- input_dim
      for (h in hidden) {
        layers[[length(layers) + 1L]] <- torch::nn_linear(last_dim, h)
        layers[[length(layers) + 1L]] <- torch::nn_relu()
        last_dim <- h
      }
      layers[[length(layers) + 1L]] <- torch::nn_linear(last_dim, output_dim)
      self$net <- do.call(torch::nn_sequential, layers)
    },
    forward = function(x) {
      self$net(x)
    }
  )

  xtr <- torch::torch_tensor(x_train, dtype = torch::torch_float())
  ytr <- torch::torch_tensor(y_train, dtype = torch::torch_float())
  xte <- torch::torch_tensor(x_test, dtype = torch::torch_float())

  model <- mlp_module(
    input_dim = ncol(x_train),
    output_dim = ncol(y_train),
    hidden = as.integer(hidden)
  )
  optimizer <- torch::optim_adam(
    model$parameters,
    lr = lr,
    weight_decay = weight_decay
  )

  model$train()
  for (epoch in seq_len(epochs)) {
    optimizer$zero_grad()
    pred <- model(xtr)
    loss <- torch::nnf_mse_loss(pred, ytr)
    loss$backward()
    optimizer$step()

    if (verbose && (epoch %% 25L == 0L || epoch == epochs)) {
      cat("epoch", epoch, "loss", as.numeric(loss$item()), "\n")
    }
  }

  model$eval()
  pred_test <- torch::with_no_grad({
    model(xte)
  })
  out <- torch::as_array(pred_test)
  if (is.null(dim(out))) out <- matrix(out, nrow = 1L)
  if (length(dim(out)) == 1L) out <- matrix(out, ncol = 1L)
  out
}


crossfit_nuisances <- function(Y,
                               X,
                               S,
                               Z,
                               L = 5L,
                               hidden = c(128L, 64L, 32L),
                               epochs = 120L,
                               lr = 5e-4,
                               weight_decay = 1e-4,
                               seed = 1L,
                               verbose = FALSE) {
  Y <- as.numeric(Y)
  X <- as.matrix(X)
  S <- as.matrix(S)
  Z <- as.matrix(Z)

  n <- length(Y)
  d <- ncol(X)
  K_total <- ncol(S)
  folds <- make_folds(n, L = L, seed = seed)

  UY <- numeric(n)
  UX <- matrix(NA_real_, n, d)
  US <- matrix(NA_real_, n, K_total)

  for (ell in seq_along(folds)) {
    test <- folds[[ell]]
    train <- setdiff(seq_len(n), test)

    pred_Y <- fit_mlp_predict(
      Z[train, , drop = FALSE],
      Y[train],
      Z[test, , drop = FALSE],
      hidden = hidden,
      epochs = epochs,
      lr = lr,
      weight_decay = weight_decay,
      seed = seed + 1000L + ell,
      verbose = verbose
    )
    pred_X <- fit_mlp_predict(
      Z[train, , drop = FALSE],
      X[train, , drop = FALSE],
      Z[test, , drop = FALSE],
      hidden = hidden,
      epochs = epochs,
      lr = lr,
      weight_decay = weight_decay,
      seed = seed + 2000L + ell,
      verbose = verbose
    )
    pred_S <- fit_mlp_predict(
      Z[train, , drop = FALSE],
      S[train, , drop = FALSE],
      Z[test, , drop = FALSE],
      hidden = hidden,
      epochs = epochs,
      lr = lr,
      weight_decay = weight_decay,
      seed = seed + 3000L + ell,
      verbose = verbose
    )

    UY[test] <- Y[test] - pred_Y[, 1L]
    UX[test, ] <- X[test, , drop = FALSE] - pred_X
    US[test, ] <- S[test, , drop = FALSE] - pred_S
  }

  list(UY = UY, UX = UX, US = US)
}


fionna_fit <- function(Y,
                       X,
                       M_list,
                       Z,
                       grid_list,
                       K,
                       L = 5L,
                       hidden = c(128L, 64L, 32L),
                       epochs = 120L,
                       lr = 5e-4,
                       weight_decay = 1e-4,
                       lambda_gamma = 0,
                       block_weights = NULL,
                       projection_ridge = 1e-8,
                       linear_ridge = 1e-8,
                       alpha_level = 0.05,
                       seed = 1L,
                       verbose = FALSE) {
  if (!is.list(M_list)) stop("M_list must be a list of mediator matrices.")
  J <- length(M_list)
  if (length(grid_list) != J) stop("grid_list must have one grid per mediator.")
  if (length(K) == 1L) K <- rep(as.integer(K), J)
  if (length(K) != J) stop("K must have length 1 or length equal to M_list.")
  if (is.null(block_weights)) block_weights <- rep(1, J)
  if (length(block_weights) != J) stop("block_weights must have length J.")

  Y_prep <- standardize_vector(Y)
  X_prep <- standardize_matrix(X)
  Z_prep <- standardize_matrix(Z)
  Y_std <- Y_prep$x
  X_std <- X_prep$x
  Z_std <- Z_prep$x

  n <- length(Y_std)
  d <- ncol(X_std)

  fpca <- vector("list", J)
  for (j in seq_len(J)) {
    fpca[[j]] <- fpca_one_mediator(M_list[[j]], grid_list[[j]], K[j])
  }
  S <- do.call(cbind, lapply(fpca, `[[`, "scores"))
  K_total <- ncol(S)

  cf <- crossfit_nuisances(
    Y = Y_std,
    X = X_std,
    S = S,
    Z = Z_std,
    L = L,
    hidden = hidden,
    epochs = epochs,
    lr = lr,
    weight_decay = weight_decay,
    seed = seed,
    verbose = verbose
  )
  UY <- cf$UY
  UX <- cf$UX
  US <- cf$US

  # Exposure projection and second residualization.
  G <- crossprod(US) / n
  R <- safe_solve(G, crossprod(US, UX) / n, ridge = projection_ridge)
  A <- UX - US %*% R

  # Residualized block-ridge regression for functional coefficients.
  D <- cbind(UX, US)
  gamma_penalty <- rep(0, K_total)
  start <- 1L
  for (j in seq_len(J)) {
    idx <- start:(start + K[j] - 1L)
    gamma_penalty[idx] <- lambda_gamma * block_weights[j]
    start <- start + K[j]
  }
  penalty <- c(rep(0, d), gamma_penalty)

  lhs <- crossprod(D) / n + diag(penalty, ncol(D))
  rhs <- crossprod(D, UY) / n
  coef <- safe_solve(lhs, rhs, ridge = linear_ridge)
  gamma_hat <- as.numeric(coef[(d + 1L):(d + K_total), , drop = FALSE])

  # Direct and total point estimates.
  H_alpha <- crossprod(A, UX) / n
  alpha_hat <- safe_solve(
    H_alpha,
    crossprod(A, UY - US %*% gamma_hat) / n,
    ridge = linear_ridge
  )

  H_theta <- crossprod(UX) / n
  theta_hat <- safe_solve(
    H_theta,
    crossprod(UX, UY) / n,
    ridge = linear_ridge
  )

  alpha_hat <- as.numeric(alpha_hat)
  theta_hat <- as.numeric(theta_hat)
  delta_hat <- theta_hat - alpha_hat

  # Indirect-effect inference.
  e_alpha <- as.numeric(UY - UX %*% alpha_hat - US %*% gamma_hat)
  e_theta <- as.numeric(UY - UX %*% theta_hat)
  H_alpha_inv <- safe_solve(H_alpha, ridge = linear_ridge)
  H_theta_inv <- safe_solve(H_theta, ridge = linear_ridge)

  psi_alpha <- (A * e_alpha) %*% t(H_alpha_inv)
  psi_theta <- (UX * e_theta) %*% t(H_theta_inv)
  psi_delta <- psi_theta - psi_alpha

  Sigma_delta <- crossprod(psi_delta) / n
  V_delta <- Sigma_delta / n
  se_delta <- sqrt(diag(V_delta))
  zcrit <- stats::qnorm(1 - alpha_level / 2)
  ci_delta <- cbind(
    lower = delta_hat - zcrit * se_delta,
    upper = delta_hat + zcrit * se_delta
  )
  coordinate_p_delta <- 2 * stats::pnorm(abs(delta_hat / se_delta), lower.tail = FALSE)
  wald_delta <- as.numeric(
    t(delta_hat) %*% safe_solve(V_delta, ridge = linear_ridge) %*% delta_hat
  )
  joint_p_delta <- stats::pchisq(wald_delta, df = d, lower.tail = FALSE)

  # Reconstruct beta_j(t) for each mediator.
  beta_hat <- vector("list", J)
  start <- 1L
  for (j in seq_len(J)) {
    idx <- start:(start + K[j] - 1L)
    beta_hat[[j]] <- as.numeric(
      fpca[[j]]$eigenfunctions %*%
        (gamma_hat[idx] / sqrt(pmax(fpca[[j]]$eigenvalues, .Machine$double.eps)))
    )
    start <- start + K[j]
  }
  names(beta_hat) <- names(M_list)

  list(
    theta = theta_hat,
    alpha = alpha_hat,
    delta = delta_hat,
    gamma = gamma_hat,
    beta = beta_hat,
    indirect_inference = list(
      se = se_delta,
      ci = ci_delta,
      coordinate_p = coordinate_p_delta,
      wald = wald_delta,
      joint_p = joint_p_delta,
      covariance = V_delta,
      influence = psi_delta
    ),
    fpca = fpca,
    residuals = list(UY = UY, UX = UX, US = US, A = A),
    preprocessing = list(Y = Y_prep, X = X_prep, Z = Z_prep),
    settings = list(
      K = K,
      L = L,
      hidden = hidden,
      epochs = epochs,
      lr = lr,
      weight_decay = weight_decay,
      lambda_gamma = lambda_gamma,
      block_weights = block_weights
    )
  )
}
