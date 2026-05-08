library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v12: joint LSIRM (5-layer max, per-item kappa)
#      + Ewens-Pitman Attraction (EPA) pairwise partition prior on items.
#
#   Differences from v11:
#     * Item-position mixture (NIW on (mu_l, Sigma_l)) is REMOVED.
#       Item latent positions have an independent Gaussian prior:
#         b_j^{(l)} ~ N_d(0, sigma_b^2 I_d).
#     * Clustering is performed by an EPA pmf with similarity
#         lambda_qr(z, tau) = exp(-tau ||z_q - z_r||^2),
#       conditioned on z = (b_1, ..., b_P).
#     * b MH ratio includes the EPA prior contribution because moving
#       z_q rescales pairwise similarities.
#     * New auxiliary state: partition c, allocation permutation sigma,
#       EPA hypers (alpha mass, tau temperature).  delta is fixed at 0.
#     * Updates: single-item EPA Gibbs on c, random-swap MH on sigma,
#       log-scale MH on (alpha, tau), Jain-Neal nonconjugate split-merge
#       (LSIRM likelihood cancels by the decoupling property).
sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v12.cpp"))

# =========================================================
# lsirm_epa_v12_cpp(): wrapper around run_lsirm_epa_v12_cpp.
# =========================================================
lsirm_epa_v12_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # ---- LSIRM-side hyperparameters (same conventions as v11) ----
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

    # ---- EPA-side hyperparameters ----
    # Defaults follow modeling_paper/model_v12_epa_partition.tex sec
    # "Hyperpriors and identifiability conventions":
    #   alpha ~ Gamma(1, 1),  tau ~ Gamma(1, 1),  delta = 0.
    epa_hyper = NULL,

    # ---- LSIRM proposal SDs ----
    lsirm_prop_sd = list(
      alpha1 = 0.1, alpha2 = 0.1, alpha3 = 0.1, alpha4 = 0.1, alpha5 = 0.1,
      log_gamma1 = 0.05, log_gamma2 = 0.05, log_gamma3 = 0.05,
      log_gamma4 = 0.05, log_gamma5 = 0.05,
      a = 0.1,
      beta1 = 0.1, beta2 = 0.1, beta3 = 0.1, beta4 = 0.3, beta5 = 0.3,
      b1 = 0.1, b2 = 0.1, b3 = 0.1, b4 = 0.1, b5 = 0.1,
      log_kappa = 0.05
    ),

    # ---- EPA proposal SDs ----
    epa_prop_sd = list(log_alpha = 0.5, log_tau = 0.5),

    # ---- LSIRM init ----
    lsirm_init = NULL,

    # ---- EPA init: list(c, sigma, alpha, tau) ----
    #   c     : P_total integer vector, 1-based (auto-converted to 0-based)
    #   sigma : P_total integer vector representing a permutation of
    #           {1,...,P_total} (auto-converted to 0-based)
    #   alpha : positive scalar (mass parameter)
    #   tau   : non-negative scalar (similarity temperature)
    epa_init = NULL,

    # ---- Item-position prior scale ----
    # b_j^{(l)} ~ N_d(0, sigma_b^2 I_d).
    # IMPORTANT: `sigma_b` here is the prior STANDARD DEVIATION, not the
    # variance.  The C++ sampler reads this value and uses inv_sigma_b_sq
    # = 1 / sigma_b^2 internally (so the squared value enters the prior
    # log-density).  Default sigma_b = 1 (paper §priors-non-clustering,
    # eq:b-prior; matches sigma_b^2 = 1).
    sigma_b = 1.0,

    compute_co_cluster_online = TRUE,

    # ---- EPA warm-up ----
    # First epa_warmup iterations skip the EPA partition / hyper updates,
    # but the b MH ratio still includes the EPA prior at the *initial*
    # (c, sigma, alpha, tau) -- giving the geometry time to settle before
    # the partition starts to move.
    epa_warmup = 0L,

    # ---- Number of EPA partition moves per outer sweep ----
    n_split_merge   = 1L,   # M_SM   (split-merge attempts per sweep)
    n_split_merge_R = 5L,   # R      (Jain-Neal launch scans)
    n_perm_swaps    = NULL, # M_perm (default = P_total)

    # ---- b update coupling toggle ----
    # TRUE  : full joint coupling (paper-defined v12 model).  b update's
    #         MH ratio includes the EPA prior contribution.
    # FALSE : sampler-level decoupling (1-way coupling).  b update uses
    #         LSIRM + N(0, sigma_b^2) prior only; the EPA prior is dropped
    #         from the b MH ratio.  c, sigma, alpha, tau updates still use
    #         the EPA pmf, so the partition follows b's geometry but b's
    #         do not feel the EPA collapse pull.  This corresponds to v11's
    #         b_prior_inflation = +Inf option and is a documented sampler
    #         approximation, not a full Bayesian sampler of the v12 joint
    #         posterior.
    b_epa_coupling = TRUE,

    # ---- EPA hyperparameter update toggles ----
    # TRUE  : update via log-scale MH each sweep (paper, v12 default).
    # FALSE : hold fixed at the value supplied in epa_init (DDT 2017-style).
    #         The Gamma prior is then ignored for that hyperparameter
    #         (since it is no longer being treated as random).
    # Use FALSE to do sensitivity analyses or to match the original
    # Dahl-Day-Tsai (2017) practice of fixing kernel temperature.
    update_alpha = TRUE,
    update_tau   = TRUE,

    verbose = TRUE,
    fix_gamma = FALSE,

    # Optional Procrustes target (LSIRM positions only).  See v11 for the
    # exact protocol; orthogonal invariance of the EPA similarity guarantees
    # cluster identities are unaffected.
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

  d <- as.integer(d)
  stopifnot(P_total >= 1, d >= 1, sigma_b > 0)

  if (is.null(n_perm_swaps)) n_perm_swaps <- P_total
  n_perm_swaps    <- as.integer(n_perm_swaps)
  n_split_merge   <- as.integer(n_split_merge)
  n_split_merge_R <- as.integer(n_split_merge_R)

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

  # ----- EPA hyperparameter defaults -----
  if (is.null(epa_hyper)) {
    epa_hyper <- list(
      a_alpha = 1.0, b_alpha = 1.0,
      a_tau   = 1.0, b_tau   = 1.0,
      delta   = 0.0
    )
  } else {
    if (is.null(epa_hyper$a_alpha)) epa_hyper$a_alpha <- 1.0
    if (is.null(epa_hyper$b_alpha)) epa_hyper$b_alpha <- 1.0
    if (is.null(epa_hyper$a_tau))   epa_hyper$a_tau   <- 1.0
    if (is.null(epa_hyper$b_tau))   epa_hyper$b_tau   <- 1.0
    if (is.null(epa_hyper$delta))   epa_hyper$delta   <- 0.0
  }
  stopifnot(epa_hyper$a_alpha > 0, epa_hyper$b_alpha > 0,
            epa_hyper$a_tau   > 0, epa_hyper$b_tau   > 0,
            epa_hyper$delta >= 0, epa_hyper$delta < 1)

  if (is.null(epa_prop_sd$log_alpha)) epa_prop_sd$log_alpha <- 0.5
  if (is.null(epa_prop_sd$log_tau))   epa_prop_sd$log_tau   <- 0.5

  # ----- EPA init defaults -----
  if (is.null(epa_init)) {
    # Default partition: every item in its own singleton.  Default sigma:
    # identity permutation.  Defaults for (alpha, tau) are the prior means
    # under Gamma(a_alpha, b_alpha) and Gamma(a_tau, b_tau).
    epa_init <- list(
      c     = seq_len(P_total),
      sigma = seq_len(P_total),
      alpha = epa_hyper$a_alpha / epa_hyper$b_alpha,
      tau   = epa_hyper$a_tau   / epa_hyper$b_tau
    )
  } else {
    if (is.null(epa_init$c))
      epa_init$c <- seq_len(P_total)
    if (is.null(epa_init$sigma))
      epa_init$sigma <- seq_len(P_total)
    if (is.null(epa_init$alpha))
      epa_init$alpha <- epa_hyper$a_alpha / epa_hyper$b_alpha
    if (is.null(epa_init$tau))
      epa_init$tau <- epa_hyper$a_tau / epa_hyper$b_tau
  }
  stopifnot(length(epa_init$c) == P_total,
            length(epa_init$sigma) == P_total,
            epa_init$alpha > 0, epa_init$tau >= 0)

  # 1-based -> 0-based for c and sigma
  epa_init$c     <- as.integer(epa_init$c)
  epa_init$sigma <- as.integer(epa_init$sigma)
  if (min(epa_init$c) >= 1)     epa_init$c     <- epa_init$c     - 1L
  if (min(epa_init$sigma) >= 1) epa_init$sigma <- epa_init$sigma - 1L
  storage.mode(epa_init$c)     <- "integer"
  storage.mode(epa_init$sigma) <- "integer"
  if (any(epa_init$c < 0 | epa_init$c >= P_total))
    stop("epa_init$c labels out of range [0, P_total) after 1->0 conversion")
  if (!setequal(epa_init$sigma, 0:(P_total - 1L)))
    stop("epa_init$sigma must be a permutation of {1,...,P_total}")

  # ----- Run C++ MCMC -----
  cat(sprintf(
    "Running joint LSIRM + EPA partition v12 [n=%d, P_total=%d (%d/%d/%d/%d/%d), d=%d, sigma_b=%g, M_SM=%d, R=%d, M_perm=%d]\n",
    n, P_total, P1, P2, P3, P4, P5, d, sigma_b,
    n_split_merge, n_split_merge_R, n_perm_swaps
  ))
  res <- run_lsirm_epa_v12_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    lsirm_hyper, epa_hyper,
    lsirm_prop_sd, epa_prop_sd,
    lsirm_init, epa_init,
    verbose, fix_gamma, nu2,
    compute_co_cluster_online,
    as.integer(epa_warmup),
    n_split_merge, n_split_merge_R, n_perm_swaps,
    as.numeric(sigma_b),
    as.logical(b_epa_coupling),
    as.logical(update_alpha),
    as.logical(update_tau)
  )

  # ----- Procrustes match LSIRM positions (POST-HOC ONLY) -----
  # IMPORTANT: This alignment is applied AFTER the C++ MCMC returns; it is
  # not invoked inside the sampler.  The C++ sampler does NOT perform any
  # within-MCMC orientation alignment, so detailed balance is preserved.
  # Paper §sec:invariance: "Within-MCMC orientation alignment is not
  # required for clustering correctness."  EPA pmf depends on positions
  # only through pairwise distances and is O(d)-invariant, so rotating the
  # whole configuration after sampling does not affect the partition
  # posterior.
  cat("Performing Procrustes matching (",
      if (is.null(procrustes_target)) "iterative posterior-mean refinement"
      else "external target supplied",
      "; only LSIRM positions a/b are aligned -- EPA is rotation-invariant)...\n", sep = "")
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

  # ----- EPA post-processing: 0-based -> 1-based -----
  res$epa_c     <- res$epa_c     + 1L
  res$epa_sigma <- res$epa_sigma + 1L

  # Quick diagnostics print
  diag <- res$epa_diagnostics
  if (!is.null(diag)) {
    cat(sprintf("[v12 EPA] split: %d / %d (%.3f), merge: %d / %d (%.3f)\n",
                diag$split_accepts, diag$split_attempts, diag$split_rate,
                diag$merge_accepts, diag$merge_attempts, diag$merge_rate))
    cat(sprintf("[v12 EPA] sigma swaps: %.0f / %.0f (%.3f)\n",
                diag$sigma_swap_accepts, diag$sigma_swap_attempts,
                diag$sigma_swap_rate))
    cat(sprintf("[v12 EPA] alpha MH: %d / %d (%.3f); tau MH: %d / %d (%.3f)\n",
                diag$alpha_epa_accepts, diag$alpha_epa_attempts,
                diag$alpha_epa_rate,
                diag$tau_epa_accepts, diag$tau_epa_attempts,
                diag$tau_epa_rate))
  }

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    d = d, nu2 = nu2,
    sigma_b = sigma_b,
    epa_kind = "EPA_pairwise_partition_with_attraction",
    epa_warmup = as.integer(epa_warmup),
    n_split_merge   = n_split_merge,
    n_split_merge_R = n_split_merge_R,
    n_perm_swaps    = n_perm_swaps,
    epa_hyper = epa_hyper,
    epa_prop_sd = epa_prop_sd,
    compute_co_cluster_online = compute_co_cluster_online,
    b_epa_coupling = b_epa_coupling,
    update_alpha   = update_alpha,
    update_tau     = update_tau,
    coupling = if (b_epa_coupling) "fully_joint_LSIRM_and_EPA"
               else                "1_way_LSIRM_to_EPA_decoupled_b_update"
  )

  return(res)
}
