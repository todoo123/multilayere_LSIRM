library(Rcpp)
library(RcppArmadillo)

# Compile the C++ kernel (must be sourced from data/)
sourceCpp(file.path(getwd(), "bipartite_SBM.cpp"))

# =========================================================
# build_distance_cube(): assemble (n x p x m) distance array from
# multilayered LSIRM posterior samples.
#
#   a_samps      : array (m x n x d), e.g. result$a from
#                  lsirm_sharedpos_layer5_grm_v6_cpp().
#   b_samps_list : list of arrays each (m x P_l x d) — typically
#                  list(result$b1, result$b2, ..., result$b5).
#                  Empty layers (P_l == 0) should be filtered out
#                  before passing.
#
# returns: array (n x p x m)
#   D[i, j, s] = || a_samps[s, i, ] - b_concat[s, j, ] ||_2
# =========================================================
build_distance_cube <- function(a_samps, b_samps_list) {
  if (length(b_samps_list) == 0) {
    stop("build_distance_cube: b_samps_list is empty.")
  }
  if (!requireNamespace("abind", quietly = TRUE)) {
    # manual concat if abind not available
    b_concat <- b_samps_list[[1]]
    if (length(b_samps_list) > 1) {
      for (k in 2:length(b_samps_list)) {
        bk <- b_samps_list[[k]]
        new_dim <- c(dim(b_concat)[1], dim(b_concat)[2] + dim(bk)[2], dim(b_concat)[3])
        tmp <- array(NA_real_, dim = new_dim)
        tmp[, 1:dim(b_concat)[2], ] <- b_concat
        tmp[, (dim(b_concat)[2] + 1):new_dim[2], ] <- bk
        b_concat <- tmp
      }
    }
  } else {
    b_concat <- do.call(abind::abind, c(b_samps_list, list(along = 2)))
  }

  # check shape
  m <- dim(a_samps)[1]
  n <- dim(a_samps)[2]
  d <- dim(a_samps)[3]
  if (dim(b_concat)[1] != m || dim(b_concat)[3] != d) {
    stop(sprintf("build_distance_cube: a_samps (%s) and b_concat (%s) shape mismatch.",
                 paste(dim(a_samps), collapse = "x"),
                 paste(dim(b_concat), collapse = "x")))
  }
  p <- dim(b_concat)[2]

  # aperm into (n x d x m) and (p x d x m) for arma::cube
  a_cube <- aperm(a_samps,  c(2, 3, 1))
  b_cube <- aperm(b_concat, c(2, 3, 1))

  D <- compute_distance_cube(a_cube, b_cube)   # (n x p x m)

  # sanity attribute
  dim(D) <- c(n, p, m)
  D
}

# =========================================================
# bipartite_sbm(): one SBM Gibbs/MH sweep per slice of D_arr.
#
#   D_arr   : array (n x p x m). Each slice [,,s] is the s-th LSIRM
#             distance matrix between n respondents and p items.
#   Q, L    : number of row / column clusters.
#   hyper   : list(r, mu_log_kappa, sd_log_kappa).
#   prop_sd : list(log_kappa) — RW proposal SD for log kappa.
#   init    : optional list(z, w, pi, rho, Lambda, log_kappa). If NULL,
#             random labels + uniform proportions + Lambda = 1/Xbar^(1).
#
# returns:
#   list(z [m x n int],
#        w [m x p int],
#        pi [m x Q],
#        rho [m x L],
#        Lambda [Q x L x m],
#        log_kappa [m],
#        Xbar [m],
#        acc_log_kappa,
#        Q, L, n, p, m)
#
# Note: per project spec, no iter/burnin/thinning are exposed —
# m sweeps are run, one per LSIRM sample.
# =========================================================
bipartite_sbm <- function(D_arr, Q, L,
                          hyper = list(r = 0.1,
                                       mu_log_kappa = 0,
                                       sd_log_kappa = 2),
                          prop_sd = list(log_kappa = 0.1),
                          init = NULL,
                          verbose = TRUE) {
  if (length(dim(D_arr)) != 3) {
    stop("bipartite_sbm: D_arr must be a 3D array (n x p x m).")
  }
  if (any(D_arr <= 0)) {
    # numerical guard — Gamma likelihood requires X > 0
    D_arr[D_arr <= 0] <- 1e-10
  }
  n <- dim(D_arr)[1]
  p <- dim(D_arr)[2]
  m <- dim(D_arr)[3]
  Q <- as.integer(Q); L <- as.integer(L)
  stopifnot(Q >= 1, L >= 1, n >= 1, p >= 1, m >= 1)

  # default init
  if (is.null(init)) {
    Xbar0 <- mean(D_arr[, , 1])
    if (Xbar0 <= 0) Xbar0 <- 1
    init <- list(
      z         = sample.int(Q, n, replace = TRUE) - 1L,   # 0-based for cpp
      w         = sample.int(L, p, replace = TRUE) - 1L,
      pi        = rep(1 / Q, Q),
      rho       = rep(1 / L, L),
      Lambda    = matrix(1 / Xbar0, Q, L),
      log_kappa = 0
    )
  } else {
    # accept 1-based labels and shift down
    if (min(init$z) >= 1) init$z <- as.integer(init$z) - 1L
    if (min(init$w) >= 1) init$w <- as.integer(init$w) - 1L
  }
  storage.mode(init$z) <- "integer"
  storage.mode(init$w) <- "integer"

  # default hyper / prop_sd
  if (is.null(hyper$r))            hyper$r            <- 0.1
  if (is.null(hyper$mu_log_kappa)) hyper$mu_log_kappa <- 0
  if (is.null(hyper$sd_log_kappa)) hyper$sd_log_kappa <- 2
  if (is.null(prop_sd$log_kappa))  prop_sd$log_kappa  <- 0.1

  if (verbose) {
    cat(sprintf(
      "Running bipartite SBM: n=%d, p=%d, m=%d, Q=%d, L=%d\n",
      n, p, m, Q, L
    ))
    cat(sprintf("  r=%g, mu_log_kappa=%g, sd_log_kappa=%g, prop_sd_log_kappa=%g\n",
                hyper$r, hyper$mu_log_kappa, hyper$sd_log_kappa,
                prop_sd$log_kappa))
  }

  res <- run_bipartite_sbm_cpp(
    D_arr,
    Q, L,
    hyper, prop_sd, init,
    verbose
  )

  # back to 1-based for R-side use
  res$z <- res$z + 1L
  res$w <- res$w + 1L
  res
}


# =========================================================
# compute_sbm_icl(): Integrated Completed Likelihood for a fitted
# bipartite SBM using its full posterior distance trajectory.
#
# Implements paper/bipartite_SBM.tex §1.2.7:
#   ICL(Q,L) = log p(D, z_hat, w_hat | Theta_hat, Q, L)
#              - 0.5 * nu * log(m * n * p)
#   nu = (Q-1) + (L-1) + Q*L + 1
#
# Representative partition (z_hat^(s), w_hat^(s)) per LSIRM sample s
# is taken as the saved SBM sample at sweep s. Theta_hat = (pi, rho,
# Lambda, kappa) are posterior means across the trajectory.
#
# returns list(icl, cll, penalty, nu)
# =========================================================
compute_sbm_icl <- function(fit, D_cube, eps = 1e-12) {
  stopifnot(length(dim(D_cube)) == 3)
  n <- dim(D_cube)[1]; p <- dim(D_cube)[2]; m <- dim(D_cube)[3]
  Q <- fit$Q; L <- fit$L

  pi_hat     <- colMeans(fit$pi)
  rho_hat    <- colMeans(fit$rho)
  Lambda_hat <- apply(fit$Lambda, c(1, 2), mean)
  kappa_hat  <- mean(exp(fit$log_kappa))

  log_pi  <- log(pi_hat + eps)
  log_rho <- log(rho_hat + eps)
  log_Lambda_hat <- log(Lambda_hat + eps)

  cll <- 0
  for (s in seq_len(m)) {
    z_s <- as.integer(fit$z[s, ])  # 1-based
    w_s <- as.integer(fit$w[s, ])
    cll <- cll + sum(log_pi[z_s]) + sum(log_rho[w_s])

    Lam_ij     <- Lambda_hat[z_s, w_s, drop = FALSE]      # n x p
    log_Lam_ij <- log_Lambda_hat[z_s, w_s, drop = FALSE]  # n x p
    X <- D_cube[, , s]
    X[X < eps] <- eps

    cll <- cll + sum(kappa_hat * log_Lam_ij
                     - lgamma(kappa_hat)
                     + (kappa_hat - 1) * log(X)
                     - Lam_ij * X)
  }

  nu      <- (Q - 1) + (L - 1) + Q * L + 1
  penalty <- 0.5 * nu * log(m * n * p)

  list(icl = cll - penalty,
       cll = cll,
       penalty = penalty,
       nu = nu)
}
