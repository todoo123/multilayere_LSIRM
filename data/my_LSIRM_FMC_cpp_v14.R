library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v14: joint LSIRM (5-layer, per-item kappa)
#      + Hierarchical Mixture-of-Mixtures (MoM) on the global pool
#        {z_q = b_j^(l)}_{q=1..P_total}:
#
#          z_q | S_q = k, I_q = l ~ N_d(mu_{kl}, Sigma_{kl})
#          S_q ~ Cat(eta_K),   I_q | S_q = k ~ Cat(w_k)
#          eta_K ~ Dir_K(alpha/K),  w_k ~ Dir_L(d_0)
#          Sigma_{kl}^{-1} | C_0k    ~ W_d(c_0, C_0k)
#          mu_{kl} | b_0k, Lambda_k  ~ N_d(b_0k, Lambda_k^{1/2} B_0 Lambda_k^{1/2})
#          C_0k ~ W_d(g_0, G_0),  b_0k ~ N_d(m_0, M_0)
#          lambda_{kj} ~ Gamma(nu_gig, nu_gig)
#
#   (All Wishart distributions follow Fruhwirth-Schnatter 2006 convention,
#    p(X) prop |X|^{c-(d+1)/2} exp(-tr(CX)), E[X] = c C^{-1}.)
#
# STAGE 1 (this wrapper): K fixed (= K_max), alpha = alpha_const,
#   Variant B b_j MH. No telescoping.
#
# Differences from v13:
#   * fmc_hyper now requires (m_0, M_0, B_0_diag, G_0, g_0, c_0, nu_gig, d_0).
#     Old NIW (m0, kappa0, nu0, S0) and Sigma_b hyperparameters are GONE.
#   * fmc_init contains (S, I) instead of (rho, c, mu, Sigma, r, Sigma_b).
#   * n_split_merge is removed.
#   * Procrustes co-rotates positions (a, b_1..5, mu_kl, b_0k) only.
#     Sigma_kl, C_0k, Lambda_k are NOT co-rotated -- they live in their
#     own conjugacy chain; B_0 is assumed diagonal which a rotation would
#     break. See wrapper notes below.

sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v14.cpp"))

# ---- Ensure GIGrvg is available (used by C++ inner loop) ----
if (!requireNamespace("GIGrvg", quietly = TRUE)) {
  install.packages("GIGrvg", repos = "https://cloud.r-project.org")
}
suppressMessages(library(GIGrvg))


# Two-level initializer: k-means on global pool, then within-cluster
# sub-k-means with L means each.
init_two_level <- function(z_init, K_max, L, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  P <- nrow(z_init)
  K_use <- min(K_max, max(2, floor(P / 4)))
  km <- tryCatch(kmeans(z_init, centers = K_use, nstart = 5, iter.max = 50),
                 error = function(e) NULL)
  if (is.null(km)) {
    S <- sample.int(K_max, P, replace = TRUE) - 1L
    I <- sample.int(L, P, replace = TRUE) - 1L
    return(list(S = S, I = I))
  }
  S0 <- km$cluster
  # Pack labels into [0, K_max)
  S <- integer(P)
  S[] <- S0 - 1L
  # Sub-k-means within each cluster
  I <- integer(P)
  for (k in seq_len(K_use)) {
    idx <- which(S == (k - 1L))
    if (length(idx) <= 1) {
      I[idx] <- 0L
      next
    }
    Lk <- min(L, length(idx))
    if (Lk < 1) Lk <- 1
    if (Lk == 1) {
      I[idx] <- 0L
    } else {
      sub <- tryCatch(kmeans(z_init[idx, , drop = FALSE], centers = Lk,
                             nstart = 3, iter.max = 30),
                      error = function(e) NULL)
      if (is.null(sub)) I[idx] <- sample.int(L, length(idx), replace = TRUE) - 1L
      else              I[idx] <- (sub$cluster - 1L)
    }
  }
  list(S = as.integer(S), I = as.integer(I))
}


# Build default fmc_hyper from data-driven scale.
#   m_0  = column means of pooled b inits
#   M_0  = 10 * sample cov of pooled b inits
#   B_0  = phi_W (1 - phi_B) * diag(sample cov)
#   G_0  = (using MFG17 variance-decomposition recipe below)
#   c_0  = 2.5 + (d-1)/2
#   g_0  = 0.5 + (d-1)/2
#   nu_gig = 10
#   d_0  = 1
#
# MFG17 §1.2 G_0^{-1} = (1 - phi_W)(1 - phi_B)(c_0 - (d+1)/2)/g_0 * diag(S_z)
# We solve for G_0 by symmetric inversion of that diagonal.
.fmc_default_hyper <- function(z_init, d, phi_B = 0.5, phi_W = 0.1) {
  S_z <- cov(z_init)
  S_z_diag <- diag(S_z)
  if (any(S_z_diag <= 0)) S_z_diag <- pmax(S_z_diag, 1e-3)
  c_0 <- 2.5 + (d - 1) / 2
  g_0 <- 0.5 + (d - 1) / 2
  G0_inv_diag <- (1 - phi_W) * (1 - phi_B) * (c_0 - (d + 1) / 2) / g_0 * S_z_diag
  G_0 <- diag(1.0 / G0_inv_diag, d)
  B_0_diag <- phi_W * (1 - phi_B) * S_z_diag
  M_0 <- 10.0 * diag(S_z_diag, d)
  m_0 <- colMeans(z_init)
  list(
    m_0       = as.numeric(m_0),
    M_0       = as.matrix(M_0),
    B_0_diag  = as.numeric(B_0_diag),
    G_0       = as.matrix(G_0),
    g_0       = as.numeric(g_0),
    c_0       = as.numeric(c_0),
    nu_gig    = 10.0,
    d_0       = 1.0
  )
}


# =========================================================
# lsirm_fmc_v14_cpp(): wrapper around run_lsirm_fmc_v14_cpp.
# =========================================================
lsirm_fmc_v14_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    K_max = 10L,
    L     = 4L,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # Stage 1 / Stage 2 parameters
    #   telescoping_on = FALSE  -> Stage 1 (K fixed = K_max, alpha = alpha_const)
    #   telescoping_on = TRUE   -> Stage 2 (sample K from BNB(1,4,3), alpha from F(6,3))
    alpha_const     = 1.0,
    telescoping_on  = FALSE,
    alpha_init      = 1.0,      # Stage 2: initial alpha; used only when telescoping_on
    s_alpha         = 0.5,      # Stage 2: SD of log-alpha RWMH proposal
    b_variant       = "B",      # "B" only in Stage 1 / Stage 2

    # ---- LSIRM hyperparameters (same as v13) ----
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

    fmc_hyper = NULL,

    lsirm_prop_sd = list(
      alpha1 = 0.1, alpha2 = 0.1, alpha3 = 0.1, alpha4 = 0.1, alpha5 = 0.1,
      log_gamma1 = 0.05, log_gamma2 = 0.05, log_gamma3 = 0.05,
      log_gamma4 = 0.05, log_gamma5 = 0.05,
      a = 0.1,
      beta1 = 0.1, beta2 = 0.1, beta3 = 0.1, beta4 = 0.3, beta5 = 0.3,
      b1 = 0.1, b2 = 0.1, b3 = 0.1, b4 = 0.1, b5 = 0.1,
      log_kappa = 0.05
    ),

    lsirm_init = NULL,
    fmc_init   = NULL,

    compute_co_cluster_online = TRUE,
    fmc_warmup = 0L,

    verbose   = TRUE,
    fix_gamma = FALSE,

    procrustes_target = NULL
) {

  # Map b_variant string to int.
  b_variant_int <- switch(b_variant,
                          "B" = 1L,
                          "A" = 2L,
                          stop("b_variant must be 'A' or 'B'"))

  # Stage 1 / Stage 2 enforcement (Stage 3 not yet implemented)
  if (b_variant_int != 1L)
    stop("Stage 1/2 (current implementation) requires b_variant = 'B'")
  if (isTRUE(telescoping_on)) {
    stopifnot(alpha_init > 0, s_alpha > 0)
  }

  # ----- Data preprocessing -----
  Y_bin  <- as.matrix(Y_bin)
  Y_con  <- as.matrix(Y_con)
  Y_cnt  <- as.matrix(Y_cnt)
  Y_ord1 <- as.matrix(Y_ord1)
  Y_ord2 <- as.matrix(Y_ord2)
  storage.mode(Y_ord1) <- "integer"
  storage.mode(Y_ord2) <- "integer"

  n  <- max(nrow(Y_bin), nrow(Y_con), nrow(Y_cnt), nrow(Y_ord1), nrow(Y_ord2))
  P1 <- ncol(Y_bin); P2 <- ncol(Y_con); P3 <- ncol(Y_cnt)
  P4 <- ncol(Y_ord1); P5 <- ncol(Y_ord2)
  P_total <- P1 + P2 + P3 + P4 + P5
  K1 <- if (P4 > 0) max(Y_ord1) else 2L
  K2 <- if (P5 > 0) max(Y_ord2) else 2L

  K_max <- as.integer(K_max)
  L     <- as.integer(L)
  d     <- as.integer(d)
  stopifnot(K_max >= 1, L >= 1, P_total >= 1, d >= 1)

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

  # ----- Stack initial b-positions to drive fmc defaults -----
  stack_b <- function() {
    parts <- list()
    if (P1 > 0) parts <- c(parts, list(lsirm_init$b1))
    if (P2 > 0) parts <- c(parts, list(lsirm_init$b2))
    if (P3 > 0) parts <- c(parts, list(lsirm_init$b3))
    if (P4 > 0) parts <- c(parts, list(lsirm_init$b4))
    if (P5 > 0) parts <- c(parts, list(lsirm_init$b5))
    do.call(rbind, parts)
  }
  z_init <- stack_b()
  stopifnot(nrow(z_init) == P_total, ncol(z_init) == d)

  # ----- fmc_hyper defaults (MFG17 variance-decomposition recipe) -----
  if (is.null(fmc_hyper)) {
    fmc_hyper <- .fmc_default_hyper(z_init, d)
  } else {
    defaults <- .fmc_default_hyper(z_init, d)
    for (nm in c("m_0", "M_0", "B_0_diag", "G_0", "g_0", "c_0", "nu_gig", "d_0")) {
      if (is.null(fmc_hyper[[nm]])) fmc_hyper[[nm]] <- defaults[[nm]]
    }
  }
  # Validate shapes
  fmc_hyper$m_0 <- as.numeric(fmc_hyper$m_0)
  fmc_hyper$M_0 <- as.matrix(fmc_hyper$M_0)
  fmc_hyper$B_0_diag <- as.numeric(fmc_hyper$B_0_diag)
  fmc_hyper$G_0 <- as.matrix(fmc_hyper$G_0)
  stopifnot(length(fmc_hyper$m_0) == d)
  stopifnot(all(dim(fmc_hyper$M_0) == c(d, d)))
  stopifnot(length(fmc_hyper$B_0_diag) == d)
  stopifnot(all(dim(fmc_hyper$G_0) == c(d, d)))
  stopifnot(fmc_hyper$g_0 > (d - 1) / 2,
            fmc_hyper$c_0 > (d - 1) / 2,
            fmc_hyper$nu_gig > 0,
            fmc_hyper$d_0 > 0)
  # GIG shape check
  if (fmc_hyper$nu_gig - L / 2 < 0)
    stop(sprintf("GIG shape p = nu_gig - L/2 = %g < 0. Increase nu_gig or decrease L.",
                 fmc_hyper$nu_gig - L / 2))

  # ----- fmc_init defaults -----
  if (is.null(fmc_init)) {
    init_SI <- init_two_level(z_init, K_max, L)
    fmc_init <- list(S = init_SI$S, I = init_SI$I)
  } else {
    if (is.null(fmc_init$S) || is.null(fmc_init$I)) {
      init_SI <- init_two_level(z_init, K_max, L)
      if (is.null(fmc_init$S)) fmc_init$S <- init_SI$S
      if (is.null(fmc_init$I)) fmc_init$I <- init_SI$I
    }
    fmc_init$S <- as.integer(fmc_init$S)
    fmc_init$I <- as.integer(fmc_init$I)
    if (length(fmc_init$S) != P_total) stop("fmc_init$S must have length P_total")
    if (length(fmc_init$I) != P_total) stop("fmc_init$I must have length P_total")
    if (min(fmc_init$S) >= 1) fmc_init$S <- fmc_init$S - 1L
    if (min(fmc_init$I) >= 1) fmc_init$I <- fmc_init$I - 1L
  }
  # Clip ranges
  fmc_init$S[fmc_init$S < 0 | fmc_init$S >= K_max] <- 0L
  fmc_init$I[fmc_init$I < 0 | fmc_init$I >= L] <- 0L
  storage.mode(fmc_init$S) <- "integer"
  storage.mode(fmc_init$I) <- "integer"

  # ----- Run C++ MCMC -----
  stage_str <- if (isTRUE(telescoping_on)) "stage2 (telescoping)" else "stage1 (K fixed)"
  if (isTRUE(telescoping_on)) {
    cat(sprintf(
      "Running joint LSIRM + two-level MoM v14 %s [n=%d, P_total=%d (%d/%d/%d/%d/%d), d=%d, K_max=%d, L=%d, alpha_init=%g, s_alpha=%g, variant=%s]\n",
      stage_str, n, P_total, P1, P2, P3, P4, P5, d, K_max, L,
      alpha_init, s_alpha, b_variant
    ))
  } else {
    cat(sprintf(
      "Running joint LSIRM + two-level MoM v14 %s [n=%d, P_total=%d (%d/%d/%d/%d/%d), d=%d, K_max=%d, L=%d, alpha_const=%g, variant=%s]\n",
      stage_str, n, P_total, P1, P2, P3, P4, P5, d, K_max, L,
      alpha_const, b_variant
    ))
  }
  res <- run_lsirm_fmc_v14_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    K_max, L,
    lsirm_hyper, fmc_hyper,
    lsirm_prop_sd,
    lsirm_init, fmc_init,
    verbose, fix_gamma, nu2,
    compute_co_cluster_online,
    as.integer(fmc_warmup),
    alpha_const,
    telescoping_on,
    b_variant_int,
    alpha_init,
    s_alpha
  )

  # ----- Procrustes match LSIRM positions -----
  # Same pattern as v13 but extended to (mu_kl, b_0k). Sigma_kl, C_0k,
  # Lambda_k are NOT co-rotated (B_0 is diagonal; rotation would break
  # diag assumption used in conjugate updates).
  cat("Performing Procrustes matching (",
      if (is.null(procrustes_target)) "iterative posterior-mean refinement"
      else "external target supplied",
      "; (a, b_1..5, mu_kl, b_0k) co-rotated; Sigma_kl/C_0k/Lambda_k kept)\n", sep = "")
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

    # Co-rotate mixture means (mu_kl: K_max*L rows x d) and b_0k (K_max x d)
    if (!is.null(res$fmc_mu_kl)) {
      mu_old <- res$fmc_mu_kl[,,i]
      mu_new <- mu_old %*% R_mat +
                matrix(trans_vec, nrow(mu_old), ncol(mu_old), byrow = TRUE)
      res$fmc_mu_kl[,,i] <<- mu_new
    }
    if (!is.null(res$fmc_b_0k)) {
      b0_old <- res$fmc_b_0k[,,i]
      b0_new <- b0_old %*% R_mat +
                matrix(trans_vec, nrow(b0_old), ncol(b0_old), byrow = TRUE)
      res$fmc_b_0k[,,i] <<- b0_new
    }
    # Sigma_kl, C_0k, Lambda_k are NOT co-rotated. See note in header.
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

  # ----- MoM post-processing -----
  # S, I from 0-based to 1-based for R consumption.
  res$fmc_S <- res$fmc_S + 1L
  res$fmc_I <- res$fmc_I + 1L

  # Reshape Sigma_kl: was stored as (d, d, K_max*L*n_save) -> (d, d, K_max*L, n_save)
  if (!is.null(res$fmc_Sigma_kl)) {
    res$fmc_Sigma_kl <- array(as.numeric(res$fmc_Sigma_kl),
                              dim = c(d, d, K_max * L, n_save))
  }
  # C_0k: (d, d, K_max*n_save) -> (d, d, K_max, n_save)
  if (!is.null(res$fmc_C_0k)) {
    res$fmc_C_0k <- array(as.numeric(res$fmc_C_0k),
                          dim = c(d, d, K_max, n_save))
  }

  # Stage 2 diagnostics print.
  if (isTRUE(telescoping_on) && !is.null(res$fmc_alpha_mh_accept_rate)) {
    cat(sprintf("[v14 stage2] alpha-MH acceptance: %.3f (n=%g attempts)\n",
                res$fmc_alpha_mh_accept_rate, res$fmc_alpha_mh_n_attempts))
    if (!is.null(res$fmc_K)) {
      cat(sprintf("[v14 stage2] K trace: min=%d  median=%d  max=%d (K_max=%d)\n",
                  as.integer(min(res$fmc_K)), as.integer(median(res$fmc_K)),
                  as.integer(max(res$fmc_K)), K_max))
    }
  }

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    d = d, K_max = K_max, L = L, nu2 = nu2,
    alpha_const = alpha_const,
    alpha_init = alpha_init, s_alpha = s_alpha,
    telescoping_on = telescoping_on,
    b_variant = b_variant,
    fmc_kind = if (isTRUE(telescoping_on)) "MoM_two_level_v14_stage2" else "MoM_two_level_v14_stage1",
    compute_co_cluster_online = compute_co_cluster_online,
    fmc_hyper = fmc_hyper
  )

  return(res)
}
