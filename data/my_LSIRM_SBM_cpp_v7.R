library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v7: joint LSIRM (5-layer, per-item kappa) + Bipartite SBM in a single chain
sourceCpp(file.path(getwd(), "my_LSIRM_SBM_v7.cpp"))

# =========================================================
# lsirm_sbm_v7_cpp(): wrapper around run_lsirm_sbm_v7_cpp.
#
#   - LSIRM and SBM hyperparameters / proposal SDs / inits are
#     kept in *separate* lists for readability.
#   - Each MCMC iteration runs all v6 LSIRM updates, then
#     a single SBM sweep on the current respondent-item
#     distance matrix D_ij = ||a_i - b_concat_j||.
#   - Standard burnin / thin therefore applies to BOTH the
#     LSIRM and SBM trajectories.
#   - Procrustes alignment (post-MCMC) is orthogonal+translation
#     only, so SBM cluster estimates are preserved.
# =========================================================
lsirm_sbm_v7_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    Q = 3, L = 3,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # ---- LSIRM-side hyperparameters (same set as v6) ----
    lsirm_hyper = list(
      a_sigma = 1, b_sigma = 0.1,
      a_tau1 = 1, b_tau1 = 0.1, a_tau2 = 1, b_tau2 = 0.1,
      a_tau3 = 1, b_tau3 = 0.1,
      a_sigma0 = 1, b_sigma0 = 1,
      mu_log_gamma1 = 0, sd_log_gamma1 = 1,
      mu_log_gamma2 = 0, sd_log_gamma2 = 1,
      mu_log_gamma3 = 0, sd_log_gamma3 = 1,
      mu_log_gamma4 = 0, sd_log_gamma4 = 1,
      mu_log_gamma5 = 0, sd_log_gamma5 = 1,
      # shared prior for per-item LSIRM log_kappa_j (count NB shape)
      mu_log_kappa = 0, sd_log_kappa = 0.1,
      mu_beta4 = 0, sd_beta4 = 2,
      mu_beta5 = 0, sd_beta5 = 2
    ),

    # ---- SBM-side hyperparameters ----
    sbm_hyper = list(
      r            = 0.1,
      mu_log_kappa = 0,
      sd_log_kappa = 2
    ),

    # ---- LSIRM-side proposal SDs ----
    lsirm_prop_sd = list(
      alpha1 = 0.1, alpha2 = 0.1, alpha3 = 0.1, alpha4 = 0.1, alpha5 = 0.1,
      log_gamma1 = 0.05, log_gamma2 = 0.05, log_gamma3 = 0.05,
      log_gamma4 = 0.05, log_gamma5 = 0.05,
      a = 0.1,
      beta1 = 0.1, beta2 = 0.1, beta3 = 0.1, beta4 = 0.3, beta5 = 0.3,
      b1 = 0.1, b2 = 0.1, b3 = 0.1, b4 = 0.1, b5 = 0.1,
      log_kappa = 0.05
    ),

    # ---- SBM-side proposal SDs ----
    sbm_prop_sd = list(
      log_kappa = 0.1
    ),

    # ---- LSIRM init (same fields as v6 init) ----
    lsirm_init = NULL,

    # ---- SBM init: list(z, w, pi, rho, Lambda, log_kappa). 1-based z, w. ----
    sbm_init = NULL,

    verbose = TRUE,
    fix_gamma = FALSE
) {

  # ----- Data preprocessing -----
  Y_bin  <- as.matrix(Y_bin)
  Y_con  <- as.matrix(Y_con)
  Y_cnt  <- as.matrix(Y_cnt)
  Y_ord1 <- as.matrix(Y_ord1)
  Y_ord2 <- as.matrix(Y_ord2)
  storage.mode(Y_ord1) <- "integer"
  storage.mode(Y_ord2) <- "integer"

  n  <- max(nrow(Y_bin), nrow(Y_con), nrow(Y_cnt), nrow(Y_ord1), nrow(Y_ord2))
  P1 <- ncol(Y_bin)
  P2 <- ncol(Y_con)
  P3 <- ncol(Y_cnt)
  P4 <- ncol(Y_ord1)
  P5 <- ncol(Y_ord2)
  P_total <- P1 + P2 + P3 + P4 + P5
  K1 <- if (P4 > 0) max(Y_ord1) else 2L
  K2 <- if (P5 > 0) max(Y_ord2) else 2L

  Q <- as.integer(Q); L <- as.integer(L)
  stopifnot(Q >= 1, L >= 1, P_total >= 1)

  # ----- LSIRM init defaults -----
  if (is.null(lsirm_init)) {
    init_grm_beta <- function(P, K) {
      if (P == 0 || K <= 1) return(matrix(0, nrow = 0, ncol = 0))
      Km1 <- K - 1
      t(sapply(seq_len(P), function(j) sort(rnorm(Km1, 0, 1), decreasing = TRUE)))
    }
    lsirm_init <- list(
      alpha1 = rnorm(n, 0, 0.1),
      alpha2 = rnorm(n, 0, 0.1),
      alpha3 = rnorm(n, 0, 0.1),
      alpha4 = rnorm(n, 0, 0.1),
      alpha5 = rnorm(n, 0, 0.1),
      beta1  = rnorm(P1, 0, 0.1),
      beta2  = rnorm(P2, 0, 0.1),
      beta3  = rnorm(P3, 0, 0.1),
      a  = matrix(rnorm(n*d,  0, 0.5), n,  d),
      b1 = matrix(rnorm(P1*d, 0, 0.5), P1, d),
      b2 = matrix(rnorm(P2*d, 0, 0.5), P2, d),
      b3 = matrix(rnorm(P3*d, 0, 0.5), P3, d),
      b4 = matrix(rnorm(P4*d, 0, 0.5), P4, d),
      b5 = matrix(rnorm(P5*d, 0, 0.5), P5, d),
      log_gamma1 = 0, log_gamma2 = 0, log_gamma3 = 0,
      log_gamma4 = 0, log_gamma5 = 0,
      log_kappa = rep(0, P3),
      sigma_alpha1_sq = 1, sigma_alpha2_sq = 1, sigma_alpha3_sq = 1,
      sigma_alpha4_sq = 1, sigma_alpha5_sq = 1,
      tau_beta1_sq = 1, tau_beta2_sq = 1, tau_beta3_sq = 1,
      sigma0_sq = 1,
      beta4 = init_grm_beta(P4, K1),
      beta5 = init_grm_beta(P5, K2)
    )
  } else {
    if (length(lsirm_init$log_kappa) == 1 && P3 > 1) {
      lsirm_init$log_kappa <- rep(as.numeric(lsirm_init$log_kappa), P3)
    } else if (length(lsirm_init$log_kappa) == 0 && P3 > 0) {
      lsirm_init$log_kappa <- rep(0, P3)
    }
  }

  # ----- SBM init defaults -----
  if (is.null(sbm_init)) {
    sbm_init <- list(
      z         = sample.int(Q, n,       replace = TRUE),  # 1-based
      w         = sample.int(L, P_total, replace = TRUE),  # 1-based
      pi        = rep(1 / Q, Q),
      rho       = rep(1 / L, L),
      Lambda    = matrix(1, Q, L),
      log_kappa = 0
    )
  } else {
    # length / type guards
    if (length(sbm_init$z) != n) stop("sbm_init$z must have length n.")
    if (length(sbm_init$w) != P_total)
      stop("sbm_init$w must have length P_total = P1+P2+P3+P4+P5.")
    if (length(sbm_init$pi) != Q)  stop("sbm_init$pi must have length Q.")
    if (length(sbm_init$rho) != L) stop("sbm_init$rho must have length L.")
    if (!is.matrix(sbm_init$Lambda) ||
        any(dim(sbm_init$Lambda) != c(Q, L)))
      stop("sbm_init$Lambda must be a Q x L matrix.")
  }
  # convert 1-based -> 0-based for cpp (idempotent if already 0-based)
  if (min(sbm_init$z) >= 1) sbm_init$z <- as.integer(sbm_init$z) - 1L
  if (min(sbm_init$w) >= 1) sbm_init$w <- as.integer(sbm_init$w) - 1L
  storage.mode(sbm_init$z) <- "integer"
  storage.mode(sbm_init$w) <- "integer"

  # ----- Run C++ MCMC -----
  cat(sprintf(
    "Running joint LSIRM+SBM v7 [n=%d, P_total=%d (%d/%d/%d/%d/%d), Q=%d, L=%d, nu2=%g]\n",
    n, P_total, P1, P2, P3, P4, P5, Q, L, nu2
  ))
  res <- run_lsirm_sbm_v7_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    Q, L,
    lsirm_hyper, sbm_hyper,
    lsirm_prop_sd, sbm_prop_sd,
    lsirm_init, sbm_init,
    verbose, fix_gamma, nu2
  )

  # ----- Procrustes match LSIRM positions -----
  cat("Performing Procrustes matching (LSIRM positions only)...\n")
  A_arr  <- res$a
  B1_arr <- res$b1; B2_arr <- res$b2; B3_arr <- res$b3
  B4_arr <- res$b4; B5_arr <- res$b5
  n_save <- dim(A_arr)[3]
  ref_idx <- n_save

  get_stacked_coords <- function(idx) {
    parts <- list(A_arr[,,idx])
    if (P1 > 0) parts <- c(parts, list(B1_arr[,,idx]))
    if (P2 > 0) parts <- c(parts, list(B2_arr[,,idx]))
    if (P3 > 0) parts <- c(parts, list(B3_arr[,,idx]))
    if (P4 > 0) parts <- c(parts, list(B4_arr[,,idx]))
    if (P5 > 0) parts <- c(parts, list(B5_arr[,,idx]))
    do.call(rbind, parts)
  }
  Target <- get_stacked_coords(ref_idx)
  idx_A <- 1:n; offset <- n
  if (P1 > 0) { idx_B1 <- (offset+1):(offset+P1); offset <- offset+P1 } else idx_B1 <- integer(0)
  if (P2 > 0) { idx_B2 <- (offset+1):(offset+P2); offset <- offset+P2 } else idx_B2 <- integer(0)
  if (P3 > 0) { idx_B3 <- (offset+1):(offset+P3); offset <- offset+P3 } else idx_B3 <- integer(0)
  if (P4 > 0) { idx_B4 <- (offset+1):(offset+P4); offset <- offset+P4 } else idx_B4 <- integer(0)
  if (P5 > 0) { idx_B5 <- (offset+1):(offset+P5); offset <- offset+P5 } else idx_B5 <- integer(0)

  for (i in 1:(n_save-1)) {
    Current <- get_stacked_coords(i)
    proc <- vegan::procrustes(X = Target, Y = Current,
                              scale = FALSE, symmetric = FALSE)
    Aligned <- fitted(proc)
    res$a[,,i] <- Aligned[idx_A, ]
    if (length(idx_B1) > 0) res$b1[,,i] <- Aligned[idx_B1, ]
    if (length(idx_B2) > 0) res$b2[,,i] <- Aligned[idx_B2, ]
    if (length(idx_B3) > 0) res$b3[,,i] <- Aligned[idx_B3, ]
    if (length(idx_B4) > 0) res$b4[,,i] <- Aligned[idx_B4, ]
    if (length(idx_B5) > 0) res$b5[,,i] <- Aligned[idx_B5, ]
  }

  # reshape arrays to (n_save x rows x d) for downstream plotting
  res$a  <- aperm(res$a,  c(3,1,2))
  res$b1 <- aperm(res$b1, c(3,1,2))
  res$b2 <- aperm(res$b2, c(3,1,2))
  res$b3 <- aperm(res$b3, c(3,1,2))
  res$b4 <- aperm(res$b4, c(3,1,2))
  res$b5 <- aperm(res$b5, c(3,1,2))
  if (!is.null(res$beta4))   res$beta4   <- aperm(res$beta4,   c(3,1,2))
  if (!is.null(res$beta5))   res$beta5   <- aperm(res$beta5,   c(3,1,2))
  if (!is.null(res$lambda2)) res$lambda2 <- aperm(res$lambda2, c(3,1,2))

  # SBM: back to 1-based labels for R-side use
  res$sbm_z <- res$sbm_z + 1L
  res$sbm_w <- res$sbm_w + 1L

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    Q = Q, L = L, nu2 = nu2, kappa_per_item = TRUE
  )
  return(res)
}
