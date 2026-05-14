library(Rcpp)
library(RcppArmadillo)
library(vegan)

# v9: joint LSIRM (5-layer, per-item kappa)
#     + Probabilistic-PCA-style mixture clustering (PPCA-FMC)
sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v9.cpp"))

# =========================================================
# lsirm_fmc_v9_cpp(): wrapper around run_lsirm_fmc_v9_cpp.
#
#   - Same one-way coupling (LSIRM -> FMC) as v8.
#
#   - Key change vs v8: per-respondent residual variances
#     psi_i (vector of length n) are replaced by a single shared
#     scalar sigma_eps_sq (homoscedastic noise = Probabilistic
#     PCA). This removes the Lambda->0 degenerate fixed point
#     that locked v8 fits to K_+ = 1 or 2 regardless of e_0.
#
#   - All other blocks (LSIRM, eta-mixture, delta) are unchanged.
#
#   - Procrustes alignment (post-MCMC) is orthogonal+translation
#     only, so all FMC parameters that depend on log-distances
#     are invariant under the alignment.
# =========================================================
lsirm_fmc_v9_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    r_fac  = 4L,
    K_star = 10L,
    e0     = 0.05,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    nu2 = 5,

    # ---- LSIRM-side hyperparameters (same set as v6 / v7) ----
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

    # ---- FMC-side hyperparameters (defaults: weakly informative) ----
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

    # ---- LSIRM init (same fields as v6 init) ----
    lsirm_init = NULL,

    # ---- FMC init: list(rho, eta, c, mu, Sigma, Lambda, delta, sigma_eps_sq, sigma_delta_sq) ----
    fmc_init = NULL,

    # Storage toggles (defaults match user's choice for typical MIDUS scale)
    save_lambda_full = FALSE,   # n x r x n_save  -- LARGE, default OFF
    save_delta_full  = FALSE,   # n_save x n      -- LARGE, default OFF
    save_eta_full    = TRUE,    # P_total x r x n_save -- small, default ON

    compute_co_cluster_online = TRUE,

    # Number of leading iterations during which only the LSIRM block runs
    # (FMC sweep is skipped). Lets LSIRM positions stabilise before clustering
    # starts; reduces the chance of FMC collapsing on random initial distances.
    # Set to 0 to disable. Recommended: ~burnin/4 for MIDUS-scale runs.
    fmc_warmup = 0L,

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

  # ----- FMC hyperparameter defaults -----
  if (is.null(fmc_hyper)) {
    fmc_hyper <- list(
      e0            = e0,
      m0            = rep(0, r_fac),
      V0            = diag(r_fac),
      nu0           = r_fac + 2,
      S0            = diag(r_fac),
      tau_lambda_sq = 1,
      # PPCA shared noise variance: sigma_eps_sq ~ IG(a_eps, b_eps).
      # Default mean = b_eps / (a_eps - 1) = 0.025; concentrated.
      a_eps         = 5, b_eps = 0.1,
      a_delta       = 2, b_delta = 1
    )
  } else {
    # If user supplied hyper without e0, fill it from the e0 argument.
    if (is.null(fmc_hyper$e0)) fmc_hyper$e0 <- e0
  }
  # Coerce hyper components to expected types
  fmc_hyper$m0  <- as.numeric(fmc_hyper$m0)
  fmc_hyper$V0  <- as.matrix(fmc_hyper$V0)
  fmc_hyper$S0  <- as.matrix(fmc_hyper$S0)
  stopifnot(length(fmc_hyper$m0) == r_fac,
            all(dim(fmc_hyper$V0) == c(r_fac, r_fac)),
            all(dim(fmc_hyper$S0) == c(r_fac, r_fac)),
            fmc_hyper$nu0 > r_fac - 1)

  # ----- FMC init defaults -----
  if (is.null(fmc_init)) {
    fmc_init <- list(
      rho     = rep(1 / K_star, K_star),
      eta     = matrix(rnorm(P_total * r_fac, 0, 0.1), P_total, r_fac),
      c       = sample.int(K_star, P_total, replace = TRUE),  # 1-based
      mu      = matrix(0, K_star, r_fac),
      Sigma   = array(rep(diag(r_fac), K_star),
                      dim = c(r_fac, r_fac, K_star)),
      Lambda  = matrix(rnorm(n * r_fac, 0, 0.1), n, r_fac),
      delta   = rnorm(n, 0, 0.1),
      sigma_eps_sq   = 0.1,
      sigma_delta_sq = 1
    )
  } else {
    # length / type guards
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
  # Convert 1-based c -> 0-based for cpp (idempotent if already 0-based)
  if (min(fmc_init$c) >= 1) fmc_init$c <- as.integer(fmc_init$c) - 1L
  storage.mode(fmc_init$c) <- "integer"

  # ----- Run C++ MCMC -----
  cat(sprintf(
    "Running joint LSIRM+PPCA-FMC v9 [n=%d, P_total=%d (%d/%d/%d/%d/%d), r_fac=%d, K_star=%d, e0=%g, nu2=%g]\n",
    n, P_total, P1, P2, P3, P4, P5, r_fac, K_star, e0, nu2
  ))
  res <- run_lsirm_fmc_v9_cpp(
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
    as.integer(fmc_warmup)
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

  # ----- FMC post-processing -----
  # 1-based labels for R-side use
  res$fmc_c <- res$fmc_c + 1L

  # Reshape packed Sigma trace from (r x r x K_star * n_save)
  # to a 4-D array (r x r x K_star x n_save).
  # NOTE: column-major flattening matches the C++ packing
  #       slice index = K_star * save_idx + l.
  if (!is.null(res$fmc_Sigma)) {
    res$fmc_Sigma <- array(as.numeric(res$fmc_Sigma),
                           dim = c(r_fac, r_fac, K_star, n_save))
  }

  # eta cube: keep as (P_total x r_fac x n_save) when full trace stored;
  # the field fmc_eta_postmean is always returned as (P_total x r_fac).

  res$info <- list(
    n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P5 = P5,
    P_total = P_total, K1 = K1, K2 = K2,
    r_fac = r_fac, K_star = K_star, e0 = e0, nu2 = nu2,
    fmc_kind = "ppca_mixture",
    save_lambda_full = save_lambda_full,
    save_delta_full  = save_delta_full,
    save_eta_full    = save_eta_full,
    compute_co_cluster_online = compute_co_cluster_online,
    coupling = "one_way_LSIRM_to_FMC"
  )

  return(res)
}
