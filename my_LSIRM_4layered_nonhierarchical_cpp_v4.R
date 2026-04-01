library(Rcpp)
library(RcppArmadillo)
library(vegan)

# C++ 코드 컴파일
sourceCpp("my_LSIRM_4layered_nonhierarchical_v4.cpp")

lsirm_sharedpos_layer4_lsgrm_cpp <- function(
    Y_bin, Y_con, Y_cnt, Y_ord,
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
      mu_log_kappa=0, sd_log_kappa=1,
      mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
    ),
    prop_sd = list(
      alpha=0.1, log_gamma1=0.05, log_gamma2=0.05, log_gamma3=0.05, log_gamma4=0.05, a=0.1,
      beta1=0.1, beta2=0.1, beta3=0.1,
      b1=0.1, b2=0.1, b3=0.1, b4=0.1,
      log_kappa=0.05, u=0.1, delta=0.1
    ),
    init = NULL,
    verbose = TRUE,
    fix_gamma = FALSE
) {

  # --- Data Preprocessing ---
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  Y_cnt <- as.matrix(Y_cnt)
  Y_ord <- as.matrix(Y_ord)
  storage.mode(Y_ord) <- "integer"

  n <- nrow(Y_bin)
  P1 <- ncol(Y_bin)
  P2 <- ncol(Y_con)
  P3 <- ncol(Y_cnt)
  P4 <- ncol(Y_ord)
  K <- max(Y_ord)

  # --- Initialization if NULL ---
  if (is.null(init)) {
    init <- list(
      alpha = rnorm(n, 0, 0.1),
      beta1 = rnorm(P1, 0, 0.1),
      beta2 = rnorm(P2, 0, 0.1),
      beta3 = rnorm(P3, 0, 0.1),
      a  = matrix(rnorm(n*d, 0, 0.5), n, d),
      b1 = matrix(rnorm(P1*d, 0, 0.5), P1, d),
      b2 = matrix(rnorm(P2*d, 0, 0.5), P2, d),
      b3 = matrix(rnorm(P3*d, 0, 0.5), P3, d),
      b4 = matrix(rnorm(P4*d, 0, 0.5), P4, d),
      log_gamma1 = 0, log_gamma2 = 0, log_gamma3 = 0, log_gamma4 = 0,
      log_kappa = 0,
      sigma_alpha_sq = 1, tau_beta1_sq = 1, tau_beta2_sq = 1, tau_beta3_sq = 1, sigma0_sq = 1,
      u = rnorm(P4, 0, 1),
      delta = if(K>2) matrix(rnorm(P4*(K-2), 0, 1), P4, K-2) else matrix(0, P4, 0)
    )
  }

  # --- Run C++ MCMC ---
  cat("Running C++ MCMC...\n")
  res <- run_lsirm_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord,
    d, n_iter, burnin, thin,
    hyper, prop_sd, init, verbose, fix_gamma
  )

  # --- Procrustes Matching ---
  cat("Performing Procrustes matching...\n")

  # Extract arrays (slices are 3rd dimension)
  A_arr  <- res$a
  B1_arr <- res$b1
  B2_arr <- res$b2
  B3_arr <- res$b3
  B4_arr <- res$b4

  n_save <- dim(A_arr)[3]

  ref_idx <- n_save

  get_stacked_coords <- function(idx) {
    rbind(
      A_arr[,,idx],
      B1_arr[,,idx],
      B2_arr[,,idx],
      B3_arr[,,idx],
      B4_arr[,,idx]
    )
  }

  Target <- get_stacked_coords(ref_idx)

  # Indices to split back
  idx_A  <- 1:n
  idx_B1 <- (n+1):(n+P1)
  idx_B2 <- (n+P1+1):(n+P1+P2)
  idx_B3 <- (n+P1+P2+1):(n+P1+P2+P3)
  idx_B4 <- (n+P1+P2+P3+1):(n+P1+P2+P3+P4)

  # Perform Matching
  for (i in 1:(n_save-1)) {
    Current <- get_stacked_coords(i)

    proc <- vegan::procrustes(X = Target, Y = Current, scale = FALSE, symmetric = FALSE)
    Aligned <- fitted(proc)

    res$a[,,i]  <- Aligned[idx_A, ]
    res$b1[,,i] <- Aligned[idx_B1, ]
    res$b2[,,i] <- Aligned[idx_B2, ]
    res$b3[,,i] <- Aligned[idx_B3, ]
    res$b4[,,i] <- Aligned[idx_B4, ]
  }

  res$a <- aperm(res$a, c(3,1,2))
  res$b1 <- aperm(res$b1, c(3,1,2))
  res$b2 <- aperm(res$b2, c(3,1,2))
  res$b3 <- aperm(res$b3, c(3,1,2))
  res$b4 <- aperm(res$b4, c(3,1,2))
  res$delta <- aperm(res$delta, c(3,1,2))
  res$thr <- aperm(res$thr, c(3,1,2))
  res$info <- list(n=n, P1=P1, P2=P2, P3=P3, P4=P4, K=K)
  return(res)
}
