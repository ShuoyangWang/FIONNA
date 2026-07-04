# Example data generator for the FIONNA GitHub example.
#
# This file intentionally contains only simulation/data-construction code.
# The FIONNA method itself is in R/fionna.R.


fourier_basis <- function(grid, n_basis) {
  outer(grid, seq_len(n_basis), function(t, k) sqrt(2) * sin(k * pi * t))
}


fixed_vector <- function(length, scale, phase = 0) {
  j <- seq_len(length)
  scale * sin(j + phase) / sqrt(j)
}


fixed_matrix <- function(rows, cols, scale, phase = 0) {
  outer(seq_len(rows), seq_len(cols), function(r, c) {
    scale * sin(0.7 * r * c + phase) / sqrt(r)
  })
}


zcol <- function(Z, j) {
  if (j <= ncol(Z)) Z[, j] else rep(0, nrow(Z))
}


simulate_fionna_shared_profile <- function(n = 1600L,
                                           p = 8L,
                                           T = 60L,
                                           J = 4L,
                                           n_basis = 28L,
                                           delta_target = 0,
                                           alpha0 = 1,
                                           seed = 20260704L) {
  set.seed(seed)
  grid <- seq(0, 1, length.out = T)
  Phi <- fourier_basis(grid, n_basis)
  eigvals <- seq_len(n_basis)^(-1.15)

  Z <- matrix(stats::rnorm(n * p), n, p)
  shared1 <- sin(zcol(Z, 1L) * zcol(Z, 2L))
  shared2 <- zcol(Z, 3L)^2 - 1
  shared3 <- tanh(zcol(Z, 4L))
  q0 <- 0.55 * shared1 + 0.35 * shared2 + 0.45 * shared3
  X <- q0 + stats::rnorm(n)

  weights_a <- c(0.50, 0.42, 0.36, 0.30)
  weights_b <- c(0.48, 0.38, 0.34, 0.28)
  j_grid <- 0:(n_basis - 1L)
  a_base <- vector("list", J)
  b_blocks <- vector("list", J)
  for (m in seq_len(J)) {
    decay <- exp(-0.18 * j_grid)
    phase <- 0.35 * (m - 1L)
    a <- weights_a[m] * decay * (1 + 0.12 * sin(j_grid + phase))
    b <- weights_b[m] * decay * (1 + 0.10 * cos(0.7 * j_grid + phase))
    a[j_grid >= 14] <- 0
    b[j_grid >= 14] <- 0
    a_base[[m]] <- a
    b_blocks[[m]] <- b
  }

  base_delta <- sum(vapply(seq_len(J), function(m) {
    sum(a_base[[m]] * b_blocks[[m]])
  }, numeric(1L)))
  scale <- if (abs(delta_target) < 1e-14) 0 else delta_target / base_delta
  a_blocks <- lapply(a_base, function(a) scale * a)
  delta0 <- sum(vapply(seq_len(J), function(m) {
    sum(a_blocks[[m]] * b_blocks[[m]])
  }, numeric(1L)))

  M_list <- vector("list", J)
  M_coef_list <- vector("list", J)
  shared_noise <- sweep(
    matrix(stats::rnorm(n * n_basis), n, n_basis),
    2L,
    sqrt(eigvals),
    "*"
  )

  active <- min(p, 12L)
  for (m in seq_len(J)) {
    phase <- 0.6 * (m - 1L)
    nonlinear_g <-
      tcrossprod(shared1, fixed_vector(n_basis, 0.12 + 0.02 * (m - 1L), 0.4 + phase)) +
      tcrossprod(shared2, fixed_vector(n_basis, 0.08, 1.3 + phase)) +
      tcrossprod(shared3, fixed_vector(n_basis, 0.07, 2.1 + phase)) +
      tcrossprod(
        tanh(zcol(Z, 3L) * zcol(Z, 4L)),
        fixed_vector(n_basis, 0.04, 2.8 + phase)
      )

    linear_g <- Z[, seq_len(active), drop = FALSE] %*%
      fixed_matrix(active, n_basis, 0.035 + 0.005 * (m - 1L), 2.7 + phase)

    specific_noise <- sweep(
      matrix(stats::rnorm(n * n_basis), n, n_basis),
      2L,
      sqrt(eigvals),
      "*"
    )
    corr <- 0.82
    score_noise <- corr * shared_noise + sqrt(1 - corr^2) * specific_noise

    M_coef <- tcrossprod(X, a_blocks[[m]]) + nonlinear_g + linear_g + score_noise
    M <- M_coef %*% t(Phi)
    M <- M + (0.38 + 0.04 * (m - 1L)) * matrix(stats::rnorm(n * T), n, T)

    M_coef_list[[m]] <- M_coef
    M_list[[m]] <- M
  }
  names(M_list) <- paste0("M", seq_len(J))

  mediator_signal <- rep(0, n)
  for (m in seq_len(J)) {
    mediator_signal <- mediator_signal + as.numeric(M_coef_list[[m]] %*% b_blocks[[m]])
  }
  f0 <- 0.55 * sin(zcol(Z, 1L) * zcol(Z, 3L)) +
    0.35 * log1p(abs(zcol(Z, 4L) * zcol(Z, 5L))) +
    0.35 * (zcol(Z, 2L)^2 - 1) +
    0.20 * tanh(zcol(Z, 6L) * zcol(Z, 7L))
  Y <- alpha0 * X + mediator_signal + f0 + 0.55 * stats::rnorm(n)

  list(
    Y = Y,
    X = matrix(X, ncol = 1L),
    Z = Z,
    M_list = M_list,
    grid_list = replicate(J, grid, simplify = FALSE),
    truth = list(alpha = alpha0, delta = delta0, theta = alpha0 + delta0),
    setting = list(
      n = n,
      p = p,
      T = T,
      J = J,
      n_basis = n_basis,
      delta_target = delta_target
    )
  )
}
