library(Rcpp)
library(RcppArmadillo)
library(vegan)

# C++ 코드 컴파일
sourceCpp("test/lsirm_ordinal_only.cpp")

lsirm_ordinal_only_cpp <- function(
    Y_ord,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    hyper = list(
      a_sigma=1, b_sigma=0.1,
      mu_log_gamma=0, sd_log_gamma=1,
      mu_delta=0, sd_delta=1
    ),
    prop_sd = list(
      alpha=0.1, log_gamma=0.05, a=0.1,
      b=0.1, delta=0.1
    ),
    init = NULL,
    verbose = TRUE,
    fix_gamma = FALSE
) {

  # --- Data Preprocessing ---
  Y_ord <- as.matrix(Y_ord)
  storage.mode(Y_ord) <- "integer"

  n <- nrow(Y_ord)
  P <- ncol(Y_ord)
  K <- max(Y_ord)

  # --- Initialization if NULL ---
  if (is.null(init)) {
    init <- list(
      alpha = rnorm(n, 0, 0.1),
      a  = matrix(rnorm(n*d, 0, 0.5), n, d),
      b  = matrix(rnorm(P*d, 0, 0.5), P, d),
      log_gamma = 0,
      sigma_alpha_sq = 1,
      delta = if(K>2) matrix(rnorm(P*(K-2), 0, 1), P, K-2) else matrix(0, P, 0)
    )
  }

  # --- Run C++ MCMC ---
  cat("Running C++ MCMC (Ordinal-only)...\n")
  res <- run_lsirm_ordinal_only_cpp(
    Y_ord,
    d, n_iter, burnin, thin,
    hyper, prop_sd, init, verbose, fix_gamma
  )

  # --- Procrustes Matching ---
  cat("Performing Procrustes matching...\n")

  A_arr <- res$a
  B_arr <- res$b

  n_save <- dim(A_arr)[3]
  ref_idx <- n_save

  get_stacked_coords <- function(idx) {
    rbind(A_arr[,,idx], B_arr[,,idx])
  }

  Target <- get_stacked_coords(ref_idx)

  idx_A <- 1:n
  idx_B <- (n+1):(n+P)

  for (i in 1:(n_save-1)) {
    Current <- get_stacked_coords(i)
    proc <- vegan::procrustes(X = Target, Y = Current, scale = FALSE, symmetric = FALSE)
    Aligned <- fitted(proc)

    res$a[,,i] <- Aligned[idx_A, ]
    res$b[,,i] <- Aligned[idx_B, ]
  }

  res$a <- aperm(res$a, c(3,1,2))
  res$b <- aperm(res$b, c(3,1,2))
  if (!is.null(res$delta)) res$delta <- aperm(res$delta, c(3,1,2))
  if (!is.null(res$thr))   res$thr   <- aperm(res$thr,   c(3,1,2))
  res$info <- list(n=n, P=P, K=K)
  return(res)
}
