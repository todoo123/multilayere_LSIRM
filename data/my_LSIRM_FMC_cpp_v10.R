library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v10: joint LSIRM (5-layer, per-item kappa)
#      + Probabilistic-PCA-style mixture clustering with
#        conjugate NIW prior on (mu_l, Sigma_l), collapsed
#        Gibbs c_j update (NIW posterior-predictive Student-t),
#        and Jain-Neal split-merge moves.
sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v10.cpp"))

# =========================================================
# lsirm_fmc_v10_cpp(): wrapper around run_lsirm_fmc_v10_cpp.
#
#   Differences from v9:
#     * fmc_hyper$V0 is removed; replaced by fmc_hyper$kappa0
#       (NIW prior precision for the component mean).
#     * c_j updates use NIW posterior-predictive Student-t
#       densities (rho is integrated out).
#     * After a full sweep of single-site collapsed Gibbs
#       updates, n_split_merge Jain-Neal restricted-Gibbs
#       split-merge proposals are made on the partition c.
#     * (mu_l, Sigma_l) are still drawn each iteration from
#       the NIW posterior, so eta_j updates remain Gaussian
#       conditional on a point value of (mu_l, Sigma_l).
#     * Returns a new list element fmc_split_merge with split
#       / merge attempt and acceptance counts.
# =========================================================
lsirm_fmc_v10_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    r_fac  = 4L,
    K_star = 10L,
    e0     = 0.1,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # ---- LSIRM-side hyperparameters (same as v6 / v7 / v9) ----
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
      mu_log_kappa = 0, sd_log_kappa = 0.1,
      mu_beta4 = 0, sd_beta4 = 2,
      mu_beta5 = 0, sd_beta5 = 2
    ),

    # ---- FMC-side hyperparameters (NIW + PPCA) ----
    #
    # Defaults tuned to roughly match the V9 prior on mu_l (V0 = 9 I_r):
    #   E[Sigma_l / kappa0] = S0 / ((nu0 - r - 1) * kappa0)
    # With nu0 = r + 10, S0 = 0.05 I_r, kappa0 = 1e-3:
    #   E[Var(mu_l)] approx 0.05 / (9 * 1e-3) I_r = 5.6 I_r,
    # i.e. similar order of magnitude to V9's V0 = 9 I_r.
    fmc_hyper = NULL,

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

    # ---- LSIRM init ----
    lsirm_init = NULL,

    # ---- FMC init: list(rho, eta, c, mu, Sigma, Lambda, delta, sigma_eps_sq, sigma_delta_sq) ----
    fmc_init = NULL,

    # Storage toggles
    save_lambda_full = FALSE,
    save_delta_full  = FALSE,
    save_eta_full    = TRUE,

    compute_co_cluster_online = TRUE,

    # FMC warm-up: run only LSIRM block for first fmc_warmup iterations.
    fmc_warmup = 0L,

    # NEW IN V10: number of split-merge proposals after each sweep of
    # single-site collapsed Gibbs c-updates. 0 = pure single-site Gibbs.
    n_split_merge = 1L,

    # Row-center the LSIRM-induced log-distance matrix before passing to PPCA.
    # TRUE (default, V8/V9/V10 main behaviour) absorbs respondent-mean
    # log-distance into delta_i deterministically; FALSE feeds raw log-
    # distances and lets the FA layer (delta + Lambda*eta) learn the
    # respondent mean as part of the model. Diagnostic experiments only.
    row_center = TRUE,

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

  r_fac  <- as.integer(r_fac)
  K_star <- as.integer(K_star)
  stopifnot(r_fac >= 1, K_star >= 1, P_total >= 1, e0 > 0)

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

  # ----- FMC hyperparameter defaults (V10 NIW) -----
  if (is.null(fmc_hyper)) {
    # S0 default 0.01 follows the elicitation note in
    # modeling_paper/model_v10_niw_split_merge.tex Sec. 5.1: it matches
    # the within-cluster eta scale empirically observed in the four-layer
    # simulation. Users on data with a different eta scale should
    # re-elicit S0 from a pilot chain (see Verification checklist).
    fmc_hyper <- list(
      e0            = e0,
      m0            = rep(0, r_fac),
      kappa0        = 1e-3,                  # NEW in v10 (replaces V0)
      nu0           = r_fac + 10,
      S0            = 0.01 * diag(r_fac),
      tau_lambda_sq = 4.0,
      a_eps         = 5, b_eps = 0.1,
      a_delta       = 2, b_delta = 1
    )
  } else {
    if (is.null(fmc_hyper$e0)) fmc_hyper$e0 <- e0
    if (is.null(fmc_hyper$kappa0)) {
      stop("fmc_hyper$kappa0 must be specified (V10 replaces V0 by kappa0).")
    }
  }
  fmc_hyper$m0  <- as.numeric(fmc_hyper$m0)
  fmc_hyper$S0  <- as.matrix(fmc_hyper$S0)
  stopifnot(length(fmc_hyper$m0) == r_fac,
            all(dim(fmc_hyper$S0) == c(r_fac, r_fac)),
            fmc_hyper$nu0 > r_fac - 1,
            fmc_hyper$kappa0 > 0)

  # ----- FMC init defaults -----
  if (is.null(fmc_init)) {
    fmc_init <- list(
      rho     = rep(1 / K_star, K_star),
      eta     = matrix(rnorm(P_total * r_fac, 0, 0.1), P_total, r_fac),
      c       = sample.int(K_star, P_total, replace = TRUE),
      mu      = matrix(0, K_star, r_fac),
      Sigma   = array(rep(diag(r_fac), K_star),
                      dim = c(r_fac, r_fac, K_star)),
      Lambda  = matrix(rnorm(n * r_fac, 0, 0.1), n, r_fac),
      delta   = rnorm(n, 0, 0.1),
      sigma_eps_sq   = 0.1,
      sigma_delta_sq = 1
    )
  } else {
    if (length(fmc_init$rho) != K_star) stop("fmc_init$rho must have length K_star.")
    if (!is.matrix(fmc_init$eta) ||
        any(dim(fmc_init$eta) != c(P_total, r_fac)))
      stop("fmc_init$eta must be a P_total x r_fac matrix.")
    if (length(fmc_init$c) != P_total) stop("fmc_init$c must have length P_total.")
    if (!is.matrix(fmc_init$mu) ||
        any(dim(fmc_init$mu) != c(K_star, r_fac)))
      stop("fmc_init$mu must be a K_star x r_fac matrix.")
    if (!is.array(fmc_init$Sigma) ||
        any(dim(fmc_init$Sigma) != c(r_fac, r_fac, K_star)))
      stop("fmc_init$Sigma must be an r_fac x r_fac x K_star array.")
    if (!is.matrix(fmc_init$Lambda) ||
        any(dim(fmc_init$Lambda) != c(n, r_fac)))
      stop("fmc_init$Lambda must be an n x r_fac matrix.")
    if (length(fmc_init$delta) != n) stop("fmc_init$delta must have length n.")
    if (length(fmc_init$sigma_eps_sq) != 1 || fmc_init$sigma_eps_sq <= 0)
      stop("fmc_init$sigma_eps_sq must be a single positive scalar.")
  }
  if (min(fmc_init$c) >= 1) fmc_init$c <- as.integer(fmc_init$c) - 1L
  storage.mode(fmc_init$c) <- "integer"

  # ----- Run C++ MCMC -----
  cat(sprintf(
    "Running joint LSIRM+PPCA-FMC v10 [n=%d, P_total=%d (%d/%d/%d/%d/%d), r_fac=%d, K_star=%d, e0=%g, kappa0=%g, M_SM=%d, row_center=%s]\n",
    n, P_total, P1, P2, P3, P4, P5, r_fac, K_star, e0, fmc_hyper$kappa0,
    as.integer(n_split_merge), row_center
  ))
  res <- run_lsirm_fmc_v10_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    r_fac, K_star,
    lsirm_hyper, fmc_hyper,
    lsirm_prop_sd,
    lsirm_init, fmc_init,
    verbose, fix_gamma, nu2,
    save_lambda_full, save_delta_full,
    save_eta_full,
    compute_co_cluster_online,
    as.integer(fmc_warmup),
    as.integer(n_split_merge),
    as.logical(row_center)
  )

  # ----- Procrustes match LSIRM positions -----
  cat("Performing Procrustes matching (LSIRM positions only; FMC parameters are\n",
      "  invariant under orthogonal+translation alignment)...\n", sep = "")
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

  if (n_save > 1) {
    for (i in 1:(n_save - 1)) {
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
  }

  res$a  <- aperm(res$a,  c(3,1,2))
  res$b1 <- aperm(res$b1, c(3,1,2))
  res$b2 <- aperm(res$b2, c(3,1,2))
  res$b3 <- aperm(res$b3, c(3,1,2))
  res$b4 <- aperm(res$b4, c(3,1,2))
  res$b5 <- aperm(res$b5, c(3,1,2))
  if (!is.null(res$beta4))   res$beta4   <- aperm(res$beta4,   c(3,1,2))
  if (!is.null(res$beta5))   res$beta5   <- aperm(res$beta5,   c(3,1,2))
  if (!is.null(res$lambda2)) res$lambda2 <- aperm(res$lambda2, c(3,1,2))

  # ----- FMC post-processing -----
  res$fmc_c <- res$fmc_c + 1L

  if (!is.null(res$fmc_Sigma)) {
    res$fmc_Sigma <- array(as.numeric(res$fmc_Sigma),
                           dim = c(r_fac, r_fac, K_star, n_save))
  }

  # Quick split-merge summary print
  sm <- res$fmc_split_merge
  if (!is.null(sm)) {
    cat(sprintf("[v10 split-merge] split: %d / %d (%.3f), merge: %d / %d (%.3f)\n",
                sm$split_accepts, sm$split_attempts, sm$split_rate,
                sm$merge_accepts, sm$merge_attempts, sm$merge_rate))
  }

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    r_fac = r_fac, K_star = K_star, e0 = e0, nu2 = nu2,
    fmc_kind = "ppca_mixture_niw_split_merge",
    save_lambda_full = save_lambda_full,
    save_delta_full  = save_delta_full,
    save_eta_full    = save_eta_full,
    compute_co_cluster_online = compute_co_cluster_online,
    coupling = "one_way_LSIRM_to_FMC",
    n_split_merge = as.integer(n_split_merge),
    row_center    = as.logical(row_center)
  )

  return(res)
}
