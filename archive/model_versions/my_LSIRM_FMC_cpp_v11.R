library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v11: joint LSIRM (5-layer, per-item kappa)
#      + Latent-position Gaussian mixture clustering with
#        conjugate NIW prior on (mu_l, Sigma_l), collapsed
#        Gibbs c_q update (NIW posterior-predictive Student-t),
#        and Jain-Neal split-merge moves -- all in R^d (the LSIRM
#        latent space).  The PPCA measurement layer is removed.
#
#   The LSIRM block is byte-identical to v10, except that the
#   b_j MH update prior term changes from N(0, I_d) to the
#   mixture prior N_d(b_j; mu_{c_q}, Sigma_{c_q}).
sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v11.cpp"))

# =========================================================
# lsirm_fmc_v11_cpp(): wrapper around run_lsirm_fmc_v11_cpp.
#
#   Differences from v10:
#     * No PPCA layer: drop r_fac, eta, Lambda, delta,
#       sigma_eps_sq, sigma_delta_sq, tau_lambda_sq,
#       a_eps, b_eps, a_delta, b_delta, row_center,
#       save_eta_full, save_lambda_full, save_delta_full.
#     * fmc_hyper now lives in R^d (d = LSIRM latent dim):
#         list(e0, m0 (length d), kappa0, nu0, S0 (d x d)).
#     * fmc_init now: list(rho, c, mu (K_star x d),
#                          Sigma (d x d x K_star)).
#     * b_j MH ratio includes the mixture prior log-density
#       N_d(b_j; mu_{c_q}, Sigma_{c_q}); LSIRM and clustering
#       are fully jointly coupled.
# =========================================================
lsirm_fmc_v11_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    K_star = 10L,
    e0     = 0.1,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # ---- LSIRM-side hyperparameters (same as v10) ----
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

    # ---- FMC-side hyperparameters (NIW in R^d) ----
    #
    # Defaults:
    #   m0     : 0_d
    #   kappa0 : 1e-3 (weak prior precision on cluster mean)
    #   nu0    : d + 10 (informative IW d.f.)
    #   S0     : 1.0 * I_d -- calibrate to b_j within-cluster
    #            spread; see model_v11 sec on layer-scale.
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

    # ---- FMC init: list(rho, c, mu, Sigma) ----
    fmc_init = NULL,

    compute_co_cluster_online = TRUE,

    # FMC warm-up: first fmc_warmup iterations skip the FMC
    # (c, mu, Sigma) updates, but the b MH ratio still uses
    # the *initial* mixture (from fmc_init) as the prior.
    fmc_warmup = 0L,

    # Number of split-merge proposals after each sweep of
    # single-site collapsed Gibbs c-updates.
    n_split_merge = 1L,

    # Variance inflation alpha for the cluster mixture prior in the b MH
    # update only (sampler-level decoupling of cluster identity vs LSIRM
    # geometry constraint):
    #   b ~ N_d(mu_{c_q}, alpha * Sigma_{c_q})  in b MH only.
    #   c, mu, Sigma updates use original Sigma (alpha = 1).
    # alpha = 1: full joint coupling (paper).
    # alpha > 1: looser b-prior pull, better LSIRM geometry recovery,
    #            cluster definition unchanged.
    # alpha = Inf: no pull, equivalent to v10-style decoupling.
    b_prior_inflation = 1.0,

    verbose = TRUE,
    fix_gamma = FALSE,

    # Optional Procrustes target (LSIRM positions only). List with
    # components named matching the position arrays whose number of
    # rows agree with (n, P1, P2, P3, P4, P5):
    #   list(a = <n x d>, b1 = <P1 x d>, b2 = ..., b3 = ..., b4 = ..., b5 = ...)
    # Mixture parameters (mu_l, Sigma_l) are co-rotated with positions
    # so that cluster labels remain valid after alignment.
    procrustes_target = NULL
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

  K_star <- as.integer(K_star)
  d      <- as.integer(d)
  stopifnot(K_star >= 1, P_total >= 1, e0 > 0, d >= 1)

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

  # ----- FMC hyperparameter defaults (V11 NIW in R^d) -----
  #
  # Two prior layers:
  #   (1) NIW component prior:  Sigma_l | S0 ~ IW(nu0, S0),
  #                             mu_l | Sigma_l ~ N_d(m0, Sigma_l / kappa0).
  #   (2) Wishart hyperprior:   S0 ~ W(nu_S0, Lambda_S0)         (Option A)
  #
  # The hyperprior makes the cluster covariance scale data-driven via a
  # conjugate Gibbs update on S0 inside the C++ sampler.  Disabled if the
  # user passes nu_S0 = NULL or omits it.
  if (is.null(fmc_hyper)) {
    fmc_hyper <- list(
      e0     = e0,
      m0     = rep(0, d),
      kappa0 = 1e-3,
      nu0    = d + 10,
      S0     = 1.0 * diag(d)
    )
  } else {
    if (is.null(fmc_hyper$e0))     fmc_hyper$e0     <- e0
    if (is.null(fmc_hyper$kappa0)) stop("fmc_hyper$kappa0 must be specified.")
    if (is.null(fmc_hyper$m0))     fmc_hyper$m0     <- rep(0, d)
    if (is.null(fmc_hyper$nu0))    fmc_hyper$nu0    <- d + 10
    if (is.null(fmc_hyper$S0))     fmc_hyper$S0     <- 1.0 * diag(d)
  }
  fmc_hyper$m0 <- as.numeric(fmc_hyper$m0)
  fmc_hyper$S0 <- as.matrix(fmc_hyper$S0)
  stopifnot(length(fmc_hyper$m0) == d,
            all(dim(fmc_hyper$S0) == c(d, d)),
            fmc_hyper$nu0 > d - 1,
            fmc_hyper$kappa0 > 0)

  # ----- Wishart hyperprior on S0 (Option A) -----
  # If nu_S0 is supplied (or NULL is replaced by the default below), the
  # sampler updates S0 via conjugate Wishart Gibbs each iteration.
  # Default: weakly-informative prior anchored at the user's S0
  #   nu_S0     = d + 2  (smallest d.f. with finite Var[S0])
  #   Lambda_S0 = S0 / nu_S0  =>  E[S0] = nu_S0 * Lambda_S0 = user's S0
  # Pass nu_S0 = NA (or omit) to disable and hold S0 fixed at its initial
  # value (back-compatible behaviour).
  if (!("nu_S0" %in% names(fmc_hyper))) {
    fmc_hyper$nu_S0     <- d + 2
    fmc_hyper$Lambda_S0 <- fmc_hyper$S0 / fmc_hyper$nu_S0
  } else if (is.na(fmc_hyper$nu_S0) || is.null(fmc_hyper$nu_S0)) {
    fmc_hyper$nu_S0 <- 0    # signals "disabled" to the C++ sampler
  } else {
    if (is.null(fmc_hyper$Lambda_S0))
      fmc_hyper$Lambda_S0 <- fmc_hyper$S0 / fmc_hyper$nu_S0
    fmc_hyper$Lambda_S0 <- as.matrix(fmc_hyper$Lambda_S0)
    stopifnot(all(dim(fmc_hyper$Lambda_S0) == c(d, d)),
              fmc_hyper$nu_S0 > d - 1)
  }

  # ----- FMC init defaults -----
  if (is.null(fmc_init)) {
    fmc_init <- list(
      rho   = rep(1 / K_star, K_star),
      c     = sample.int(K_star, P_total, replace = TRUE),
      mu    = matrix(rnorm(K_star * d, 0, 0.5), K_star, d),
      Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star))
    )
  } else {
    if (length(fmc_init$rho) != K_star) stop("fmc_init$rho must have length K_star.")
    if (length(fmc_init$c)   != P_total) stop("fmc_init$c must have length P_total.")
    if (!is.matrix(fmc_init$mu) || any(dim(fmc_init$mu) != c(K_star, d)))
      stop("fmc_init$mu must be a K_star x d matrix.")
    if (!is.array(fmc_init$Sigma) ||
        any(dim(fmc_init$Sigma) != c(d, d, K_star)))
      stop("fmc_init$Sigma must be an d x d x K_star array.")
  }
  if (min(fmc_init$c) >= 1) fmc_init$c <- as.integer(fmc_init$c) - 1L
  storage.mode(fmc_init$c) <- "integer"

  # ----- Run C++ MCMC -----
  cat(sprintf(
    "Running joint LSIRM + latent-position-mixture v11 [n=%d, P_total=%d (%d/%d/%d/%d/%d), d=%d, K_star=%d, e0=%g, kappa0=%g, M_SM=%d]\n",
    n, P_total, P1, P2, P3, P4, P5, d, K_star, e0, fmc_hyper$kappa0,
    as.integer(n_split_merge)
  ))
  res <- run_lsirm_fmc_v11_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    K_star,
    lsirm_hyper, fmc_hyper,
    lsirm_prop_sd,
    lsirm_init, fmc_init,
    verbose, fix_gamma, nu2,
    compute_co_cluster_online,
    as.integer(fmc_warmup),
    as.integer(n_split_merge),
    as.numeric(b_prior_inflation)
  )

  # ----- Procrustes match LSIRM positions -----
  # Same protocol as v10: per-iter Procrustes against the running
  # posterior mean (always), then optionally a single rigid re-anchor
  # to an external target. The mixture (mu_l, Sigma_l) draws are
  # co-rotated with the LSIRM positions so cluster labels remain valid.
  cat("Performing Procrustes matching (",
      if (is.null(procrustes_target)) "iterative posterior-mean refinement"
      else "external target supplied",
      "; LSIRM positions are aligned and (mu_l, Sigma_l) are co-rotated)...\n", sep = "")
  n_save <- dim(res$a)[3]

  get_stacked_coords <- function(idx) {
    parts <- list(res$a[,,idx])
    if (P1 > 0) parts <- c(parts, list(res$b1[,,idx]))
    if (P2 > 0) parts <- c(parts, list(res$b2[,,idx]))
    if (P3 > 0) parts <- c(parts, list(res$b3[,,idx]))
    if (P4 > 0) parts <- c(parts, list(res$b4[,,idx]))
    if (P5 > 0) parts <- c(parts, list(res$b5[,,idx]))
    do.call(rbind, parts)
  }
  idx_A <- 1:n; offset <- n
  if (P1 > 0) { idx_B1 <- (offset+1):(offset+P1); offset <- offset+P1 } else idx_B1 <- integer(0)
  if (P2 > 0) { idx_B2 <- (offset+1):(offset+P2); offset <- offset+P2 } else idx_B2 <- integer(0)
  if (P3 > 0) { idx_B3 <- (offset+1):(offset+P3); offset <- offset+P3 } else idx_B3 <- integer(0)
  if (P4 > 0) { idx_B4 <- (offset+1):(offset+P4); offset <- offset+P4 } else idx_B4 <- integer(0)
  if (P5 > 0) { idx_B5 <- (offset+1):(offset+P5); offset <- offset+P5 } else idx_B5 <- integer(0)

  compute_stacked_mean <- function() {
    parts <- list(apply(res$a, c(1, 2), mean))
    if (P1 > 0) parts <- c(parts, list(apply(res$b1, c(1, 2), mean)))
    if (P2 > 0) parts <- c(parts, list(apply(res$b2, c(1, 2), mean)))
    if (P3 > 0) parts <- c(parts, list(apply(res$b3, c(1, 2), mean)))
    if (P4 > 0) parts <- c(parts, list(apply(res$b4, c(1, 2), mean)))
    if (P5 > 0) parts <- c(parts, list(apply(res$b5, c(1, 2), mean)))
    do.call(rbind, parts)
  }

  build_external_target <- function(tgt) {
    parts <- list(as.matrix(tgt$a))
    stopifnot(nrow(parts[[1]]) == n, ncol(parts[[1]]) == d)
    if (P1 > 0) { stopifnot(!is.null(tgt$b1), nrow(tgt$b1) == P1); parts <- c(parts, list(as.matrix(tgt$b1))) }
    if (P2 > 0) { stopifnot(!is.null(tgt$b2), nrow(tgt$b2) == P2); parts <- c(parts, list(as.matrix(tgt$b2))) }
    if (P3 > 0) { stopifnot(!is.null(tgt$b3), nrow(tgt$b3) == P3); parts <- c(parts, list(as.matrix(tgt$b3))) }
    if (P4 > 0) { stopifnot(!is.null(tgt$b4), nrow(tgt$b4) == P4); parts <- c(parts, list(as.matrix(tgt$b4))) }
    if (P5 > 0) { stopifnot(!is.null(tgt$b5), nrow(tgt$b5) == P5); parts <- c(parts, list(as.matrix(tgt$b5))) }
    do.call(rbind, parts)
  }

  # Apply rigid map (R, t) to one iteration's positions AND co-rotate
  # the mixture parameters (mu_l -> R mu_l + t, Sigma_l -> R Sigma_l R^T).
  # Procrustes from vegan returns Aligned = scale * (Y %*% R) + t but with
  # scale = FALSE the scale is 1.
  apply_rigid_one_iter <- function(i, R_mat, trans_vec) {
    Current <- get_stacked_coords(i)
    Aligned <- Current %*% R_mat +
               matrix(trans_vec, nrow(Current), ncol(Current), byrow = TRUE)
    res$a[,,i]  <<- Aligned[idx_A, ]
    if (length(idx_B1) > 0) res$b1[,,i] <<- Aligned[idx_B1, ]
    if (length(idx_B2) > 0) res$b2[,,i] <<- Aligned[idx_B2, ]
    if (length(idx_B3) > 0) res$b3[,,i] <<- Aligned[idx_B3, ]
    if (length(idx_B4) > 0) res$b4[,,i] <<- Aligned[idx_B4, ]
    if (length(idx_B5) > 0) res$b5[,,i] <<- Aligned[idx_B5, ]
    # Co-rotate mixture parameters (only if they were stored)
    if (!is.null(res$fmc_mu)) {
      mu_old <- res$fmc_mu[,,i]                  # K_star x d
      mu_new <- mu_old %*% R_mat +
                matrix(trans_vec, nrow(mu_old), ncol(mu_old), byrow = TRUE)
      res$fmc_mu[,,i] <<- mu_new
    }
    if (!is.null(res$fmc_Sigma)) {
      # res$fmc_Sigma is laid out as (d, d, K_star * n_save); slice K_star
      # consecutive entries starting at K_star*(i-1)+1 are this iter's Sigmas.
      base <- K_star * (i - 1)
      for (l in seq_len(K_star)) {
        S_old <- res$fmc_Sigma[,, base + l]
        res$fmc_Sigma[,, base + l] <<- t(R_mat) %*% S_old %*% R_mat
        # Note: vegan's R is such that Aligned = Y %*% R; for covariance
        # we apply Sigma -> R^T Sigma R only if positions are row-vectors
        # transformed as x' = x %*% R. Then Cov(x') = R^T Cov(x) R.
      }
    }
  }

  align_all_to <- function(Target) {
    for (i in seq_len(n_save)) {
      Current <- get_stacked_coords(i)
      proc <- vegan::procrustes(X = Target, Y = Current,
                                scale = FALSE, symmetric = FALSE)
      apply_rigid_one_iter(i, proc$rotation, proc$translation)
    }
  }

  if (n_save > 1) {
    # ---- Step 1: iterative-mean refinement (always) ----
    n_proc_passes <- 3L
    prev_target <- NULL
    for (pass in seq_len(n_proc_passes)) {
      Target <- compute_stacked_mean()
      delta <- if (is.null(prev_target)) NA_real_
               else sqrt(mean((Target - prev_target)^2))
      align_all_to(Target)
      cat(sprintf("  iterative-mean pass %d/%d, target shift RMSE = %s\n",
                  pass, n_proc_passes,
                  if (is.na(delta)) "(initial)" else sprintf("%.4f", delta)))
      prev_target <- Target
    }

    # ---- Step 2 (optional): single rigid re-anchor to external target ----
    if (!is.null(procrustes_target)) {
      Target_ext <- build_external_target(procrustes_target)
      PM <- compute_stacked_mean()
      pr_anchor <- vegan::procrustes(X = Target_ext, Y = PM,
                                     scale = FALSE, symmetric = FALSE)
      for (i in seq_len(n_save)) {
        apply_rigid_one_iter(i, pr_anchor$rotation, pr_anchor$translation)
      }
      pre_rmse  <- sqrt(mean((PM - Target_ext)^2))
      PM2 <- compute_stacked_mean()
      post_rmse <- sqrt(mean((PM2 - Target_ext)^2))
      cat(sprintf("  external re-anchor: post-mean RMSE %.3f -> %.3f\n",
                  pre_rmse, post_rmse))
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
                           dim = c(d, d, K_star, n_save))
  }

  # Quick split-merge summary print
  sm <- res$fmc_split_merge
  if (!is.null(sm)) {
    cat(sprintf("[v11 split-merge] split: %d / %d (%.3f), merge: %d / %d (%.3f)\n",
                sm$split_accepts, sm$split_attempts, sm$split_rate,
                sm$merge_accepts, sm$merge_attempts, sm$merge_rate))
  }

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    d = d, K_star = K_star, e0 = e0, nu2 = nu2,
    fmc_kind = "latent_position_mixture_niw_split_merge",
    compute_co_cluster_online = compute_co_cluster_online,
    coupling = "fully_joint_LSIRM_and_FMC",
    n_split_merge = as.integer(n_split_merge),
    b_prior_inflation = as.numeric(b_prior_inflation),
    S0_hyperprior = isTRUE(res$fmc_S0_hyperprior),
    nu_S0         = fmc_hyper$nu_S0,
    Lambda_S0     = fmc_hyper$Lambda_S0
  )

  return(res)
}
