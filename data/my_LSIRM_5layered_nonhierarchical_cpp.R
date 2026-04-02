library(Rcpp)
library(RcppArmadillo)
library(vegan)

# C++ 코드 컴파일
sourceCpp("data/my_LSIRM_5layered_nonhierarchical_v3.cpp")

lsirm_sharedpos_layer5_lsgrm_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    hyper = list(
      a_sigma=1, b_sigma=0.1,
      a_tau1=1, b_tau1=0.1, a_tau2=1, b_tau2=0.1, a_tau3=1, b_tau3=0.1,
      a_sigma0=1, b_sigma0=1,
      mu_log_gamma1=0, sd_log_gamma1=1,
      mu_log_gamma2=0, sd_log_gamma2=1,
      mu_log_gamma3=0, sd_log_gamma3=1,
      mu_log_gamma4=0, sd_log_gamma4=1,
      mu_log_gamma5=0, sd_log_gamma5=1,
      mu_log_kappa=0, sd_log_kappa=1,
      mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
    ),
    prop_sd = list(
      alpha1=0.1, alpha2=0.1, alpha3=0.1, alpha4=0.1, alpha5=0.1,
      log_gamma1=0.05, log_gamma2=0.05, log_gamma3=0.05, log_gamma4=0.05, log_gamma5=0.05, a=0.1,
      beta1=0.1, beta2=0.1, beta3=0.1,
      b1=0.1, b2=0.1, b3=0.1, b4=0.1, b5=0.1,
      log_kappa=0.05, u=0.1, delta=0.1, delta2=0.1
    ),
    init = NULL,
    verbose = TRUE,
    fix_gamma = FALSE
) {

  # --- Data Preprocessing ---
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
  K1 <- if (P4 > 0) max(Y_ord1) else 2L
  K2 <- if (P5 > 0) max(Y_ord2) else 2L

  # --- Initialization if NULL ---
  if (is.null(init)) {
    init <- list(
      alpha1 = rnorm(n, 0, 0.1),
      alpha2 = rnorm(n, 0, 0.1),
      alpha3 = rnorm(n, 0, 0.1),
      alpha4 = rnorm(n, 0, 0.1),
      alpha5 = rnorm(n, 0, 0.1),
      beta1 = rnorm(P1, 0, 0.1),
      beta2 = rnorm(P2, 0, 0.1),
      beta3 = rnorm(P3, 0, 0.1),
      a  = matrix(rnorm(n*d, 0, 0.5), n, d),
      b1 = matrix(rnorm(P1*d, 0, 0.5), P1, d),
      b2 = matrix(rnorm(P2*d, 0, 0.5), P2, d),
      b3 = matrix(rnorm(P3*d, 0, 0.5), P3, d),
      b4 = matrix(rnorm(P4*d, 0, 0.5), P4, d),
      b5 = matrix(rnorm(P5*d, 0, 0.5), P5, d),
      log_gamma1 = 0, log_gamma2 = 0, log_gamma3 = 0, log_gamma4 = 0, log_gamma5 = 0,
      log_kappa = 0,
      sigma_alpha_sq = 1, tau_beta1_sq = 1, tau_beta2_sq = 1, tau_beta3_sq = 1, sigma0_sq = 1,
      u = rnorm(P4, 0, 1),
      delta  = if(K1>2) matrix(rnorm(P4*(K1-2), 0, 1), P4, K1-2) else matrix(0, P4, 0),
      delta2 = if(K2>2) matrix(rnorm(P5*(K2-2), 0, 1), P5, K2-2) else matrix(0, P5, 0)
    )
  }

  # --- Run C++ MCMC ---
  cat("Running C++ MCMC (5-layered)...\n")
  res <- run_lsirm_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
    d, n_iter, burnin, thin,
    hyper, prop_sd, init, verbose, fix_gamma
  )

  # --- Procrustes Matching ---
  cat("Performing Procrustes matching...\n")

  A_arr  <- res$a
  B1_arr <- res$b1
  B2_arr <- res$b2
  B3_arr <- res$b3
  B4_arr <- res$b4
  B5_arr <- res$b5

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

  # Indices to split back (Pj==0 → integer(0) to avoid reverse-range bug)
  idx_A <- 1:n
  offset <- n
  if (P1 > 0) { idx_B1 <- (offset+1):(offset+P1); offset <- offset+P1 } else { idx_B1 <- integer(0) }
  if (P2 > 0) { idx_B2 <- (offset+1):(offset+P2); offset <- offset+P2 } else { idx_B2 <- integer(0) }
  if (P3 > 0) { idx_B3 <- (offset+1):(offset+P3); offset <- offset+P3 } else { idx_B3 <- integer(0) }
  if (P4 > 0) { idx_B4 <- (offset+1):(offset+P4); offset <- offset+P4 } else { idx_B4 <- integer(0) }
  if (P5 > 0) { idx_B5 <- (offset+1):(offset+P5); offset <- offset+P5 } else { idx_B5 <- integer(0) }

  for (i in 1:(n_save-1)) {
    Current <- get_stacked_coords(i)
    proc <- vegan::procrustes(X = Target, Y = Current, scale = FALSE, symmetric = FALSE)
    Aligned <- fitted(proc)

    res$a[,,i] <- Aligned[idx_A, ]
    if (length(idx_B1) > 0) res$b1[,,i] <- Aligned[idx_B1, ]
    if (length(idx_B2) > 0) res$b2[,,i] <- Aligned[idx_B2, ]
    if (length(idx_B3) > 0) res$b3[,,i] <- Aligned[idx_B3, ]
    if (length(idx_B4) > 0) res$b4[,,i] <- Aligned[idx_B4, ]
    if (length(idx_B5) > 0) res$b5[,,i] <- Aligned[idx_B5, ]
  }

  res$a  <- aperm(res$a,  c(3,1,2))
  res$b1 <- aperm(res$b1, c(3,1,2))
  res$b2 <- aperm(res$b2, c(3,1,2))
  res$b3 <- aperm(res$b3, c(3,1,2))
  res$b4 <- aperm(res$b4, c(3,1,2))
  res$b5 <- aperm(res$b5, c(3,1,2))
  if (!is.null(res$delta))  res$delta  <- aperm(res$delta,  c(3,1,2))
  if (!is.null(res$delta2)) res$delta2 <- aperm(res$delta2, c(3,1,2))
  if (!is.null(res$thr))    res$thr    <- aperm(res$thr,    c(3,1,2))
  if (!is.null(res$thr2))   res$thr2   <- aperm(res$thr2,   c(3,1,2))
  res$info <- list(n=n, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, K1=K1, K2=K2)
  return(res)
}
