setwd('/Users/todoo/Desktop/학교/대학원/Research/joint_LSM')
source('my_LSIRM_4layered_nonhierarchical_cpp.R')

# =========================================================
# Coverage computation over repeated fits
# - For ALL parameters used in plot_group_box_ci:
#   alpha, beta1, beta2, beta3, u, thr, (optional) delta
# - coverage: TRUE value in 95% CI (inclusive boundaries)
# - repeats: fit -> compute -> store
# - output: coverage.csv in current directory
# =========================================================

# ---------- helper: compute elementwise 95% CI coverage + width ----------
coverage_from_samples <- function(samples_mat, true_vec, prob = c(0.025, 0.975)) {
  # samples_mat: (n_save x n_param) numeric
  # true_vec   : length n_param numeric (can include NA)
  stopifnot(is.matrix(samples_mat) || is.data.frame(samples_mat))
  samples_mat <- as.matrix(samples_mat)
  
  if (length(true_vec) != ncol(samples_mat)) {
    stop("true_vec length must match ncol(samples_mat).")
  }
  
  # CI per parameter
  q_lo <- apply(samples_mat, 2, quantile, probs = prob[1], na.rm = TRUE)
  q_hi <- apply(samples_mat, 2, quantile, probs = prob[2], na.rm = TRUE)
  
  widths <- q_hi - q_lo
  
  # inclusive boundary: (lo <= true <= hi)
  # NA true -> NA coverage
  cov_ind <- rep(NA_integer_, length(true_vec))
  ok <- !is.na(true_vec)
  cov_ind[ok] <- as.integer(q_lo[ok] <= true_vec[ok] & true_vec[ok] <= q_hi[ok])
  
  list(
    covered_ind = cov_ind,
    n_param     = sum(ok),
    n_covered   = sum(cov_ind[ok] == 1),
    pct_covered = if (sum(ok) == 0) NA_real_ else 100 * mean(cov_ind[ok] == 1),
    mean_width  = if (sum(ok) == 0) NA_real_ else mean(widths[ok]),
    sum_width   = if (sum(ok) == 0) 0 else sum(widths[ok])
  )
}

# ---------- helper: flatten (n_save x P x d) -> (n_save x (P*d)) ----------
flatten_3d_to_2d <- function(arr3d) {
  # arr3d: [n_save, P, d]
  stopifnot(length(dim(arr3d)) == 3)
  n_save <- dim(arr3d)[1]
  P <- dim(arr3d)[2]
  d <- dim(arr3d)[3]
  matrix(arr3d, nrow = n_save, ncol = P * d)
}

# =========================================================
# Distance coverage (a vs b-layer) with gamma scaling
# - a_samps: [S, n, d]
# - b_samps: [S, P, d]
# - log_gamma_samps: length S
# - D_true: [n, P] (Euclidean distances from TRUE positions)
# Coverage checks TRUE*(gamma_true) within posterior 95% CI of dist*(gamma_s)
# =========================================================
distance_coverage_layer <- function(a_samps, b_samps, log_gamma_samps,
                                    D_true, gamma_true = 1.0,
                                    prob = c(0.025, 0.975)) {
  stopifnot(length(dim(a_samps)) == 3, length(dim(b_samps)) == 3)
  S <- dim(a_samps)[1]
  n <- dim(a_samps)[2]
  d <- dim(a_samps)[3]
  P <- dim(b_samps)[2]
  stopifnot(dim(b_samps)[1] == S, dim(b_samps)[3] == d)
  stopifnot(all(dim(D_true) == c(n, P)))
  stopifnot(length(log_gamma_samps) == S)
  
  gamma_s <- exp(as.numeric(log_gamma_samps))  # length S
  true_scaled <- D_true * gamma_true           # n x P
  
  n_param <- n * P
  n_covered <- 0L
  sum_width <- 0.0
  
  # loop by item to keep memory small: build S x n matrix each time
  for (j in 1:P) {
    # dx,dy: S x n
    dx <- a_samps[, , 1] - matrix(b_samps[, j, 1], nrow = S, ncol = n)
    if (d >= 2) {
      dy <- a_samps[, , 2] - matrix(b_samps[, j, 2], nrow = S, ncol = n)
      dist_s <- sqrt(dx^2 + dy^2)
    } else {
      dist_s <- abs(dx)
    }
    
    # scale by gamma per draw
    dist_s <- dist_s * matrix(gamma_s, nrow = S, ncol = n)
    
    # CI per respondent (column-wise)
    q_lo <- apply(dist_s, 2, quantile, probs = prob[1], na.rm = TRUE)
    q_hi <- apply(dist_s, 2, quantile, probs = prob[2], na.rm = TRUE)
    
    sum_width <- sum_width + sum(q_hi - q_lo)
    
    # coverage for this item j across all respondents i
    tj <- true_scaled[, j]
    n_covered <- n_covered + sum(q_lo <= tj & tj <= q_hi, na.rm = TRUE)
  }
  
  list(
    n_param     = n_param,
    n_covered   = n_covered,
    pct_covered = 100 * (n_covered / n_param),
    mean_width  = sum_width / n_param,
    sum_width   = sum_width
  )
}

# ---------- helper: compute coverage summary for one fit ----------
compute_coverage_onefit <- function(fit, true_list) {
  # true_list: list(alpha, beta1, beta2, beta3, u, thr, delta)
  # fit: result object with fit$samples
  samps <- fit$samples
  
  out_rows <- list()
  
  # alpha
  tmp <- coverage_from_samples(samps$alpha, true_list$alpha)
  out_rows[["alpha"]] <- data.frame(
    param = "alpha", n_param = tmp$n_param, n_covered = tmp$n_covered,
    pct = tmp$pct_covered, mean_width = tmp$mean_width,
    sum_width = tmp$sum_width, stringsAsFactors = FALSE
  )
  
  # beta1
  tmp <- coverage_from_samples(samps$beta1, true_list$beta1)
  out_rows[["beta1"]] <- data.frame(
    param = "beta1", n_param = tmp$n_param, n_covered = tmp$n_covered,
    pct = tmp$pct_covered, mean_width = tmp$mean_width,
    sum_width = tmp$sum_width, stringsAsFactors = FALSE
  )
  
  # beta2
  tmp <- coverage_from_samples(samps$beta2, true_list$beta2)
  out_rows[["beta2"]] <- data.frame(
    param = "beta2", n_param = tmp$n_param, n_covered = tmp$n_covered,
    pct = tmp$pct_covered, mean_width = tmp$mean_width,
    sum_width = tmp$sum_width, stringsAsFactors = FALSE
  )
  
  # beta3
  tmp <- coverage_from_samples(samps$beta3, true_list$beta3)
  out_rows[["beta3"]] <- data.frame(
    param = "beta3", n_param = tmp$n_param, n_covered = tmp$n_covered,
    pct = tmp$pct_covered, mean_width = tmp$mean_width,
    sum_width = tmp$sum_width, stringsAsFactors = FALSE
  )
  
  # =======================================================
  # Distance coverage: dist(a, b1/b2/b3) * gamma
  # =======================================================
  if (!is.null(samps$a) && !is.null(samps$b1) && !is.null(samps$log_gamma)) {
    tmpd <- distance_coverage_layer(
      a_samps = samps$a, b_samps = samps$b1, log_gamma_samps = samps$log_gamma,
      D_true = D1_true, gamma_true = gamma_true
    )
    out_rows[["dist_bin"]] <- data.frame(
      param = "dist_bin", n_param = tmpd$n_param, n_covered = tmpd$n_covered,
      pct = tmpd$pct_covered, mean_width = tmpd$mean_width,
      sum_width = tmpd$sum_width, stringsAsFactors = FALSE
    )
  }
  
  if (!is.null(samps$a) && !is.null(samps$b2) && !is.null(samps$log_gamma)) {
    tmpd <- distance_coverage_layer(
      a_samps = samps$a, b_samps = samps$b2, log_gamma_samps = samps$log_gamma,
      D_true = D2_true, gamma_true = gamma_true
    )
    out_rows[["dist_con"]] <- data.frame(
      param = "dist_con", n_param = tmpd$n_param, n_covered = tmpd$n_covered,
      pct = tmpd$pct_covered, mean_width = tmpd$mean_width,
      sum_width = tmpd$sum_width, stringsAsFactors = FALSE
    )
  }
  
  if (!is.null(samps$a) && !is.null(samps$b3) && !is.null(samps$log_gamma)) {
    tmpd <- distance_coverage_layer(
      a_samps = samps$a, b_samps = samps$b3, log_gamma_samps = samps$log_gamma,
      D_true = D3_true, gamma_true = gamma_true
    )
    out_rows[["dist_cnt"]] <- data.frame(
      param = "dist_cnt", n_param = tmpd$n_param, n_covered = tmpd$n_covered,
      pct = tmpd$pct_covered, mean_width = tmpd$mean_width,
      sum_width = tmpd$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # Ordinal items distance
  if (!is.null(samps$a) && !is.null(samps$b4) && !is.null(samps$log_gamma)) {
    tmpd <- distance_coverage_layer(
      a_samps = samps$a, b_samps = samps$b4, log_gamma_samps = samps$log_gamma,
      D_true = D4_true, gamma_true = gamma_true
    )
    out_rows[["dist_ord"]] <- data.frame(
      param = "dist_ord", n_param = tmpd$n_param, n_covered = tmpd$n_covered,
      pct = tmpd$pct_covered, mean_width = tmpd$mean_width,
      sum_width = tmpd$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # u (ordinal)
  if (!is.null(samps$u) && !is.null(true_list$u)) {
    tmp <- coverage_from_samples(samps$u, true_list$u)
    out_rows[["u"]] <- data.frame(
      param = "u", n_param = tmp$n_param, n_covered = tmp$n_covered,
      pct = tmp$pct_covered, mean_width = tmp$mean_width,
      sum_width = tmp$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # delta (ordinal gaps) : [n_save, P4, K-2]
  if (!is.null(samps$delta) && !is.null(true_list$delta)) {
    delta_samps <- samps$delta
    n_save <- dim(delta_samps)[1]
    P4 <- dim(delta_samps)[2]
    Km2 <- dim(delta_samps)[3]
    delta_mat <- matrix(delta_samps, nrow = n_save, ncol = P4 * Km2)
    
    delta_true_vec <- as.vector(true_list$delta)
    
    tmp <- coverage_from_samples(delta_mat, delta_true_vec)
    out_rows[["delta"]] <- data.frame(
      param = "delta", n_param = tmp$n_param, n_covered = tmp$n_covered,
      pct = tmp$pct_covered, mean_width = tmp$mean_width,
      sum_width = tmp$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # combine
  do.call(rbind, out_rows)
}


# =========================================================
# Main loop: repeat fitting -> coverage -> append
# =========================================================

# (1) Define how many repetitions
n_rep <- 10 # 원하는 만큼 변경

# (2) True values used for coverage
true_list <- list(
  alpha = alpha_true,
  beta1 = beta1_true,
  beta2 = beta2_true,
  beta3 = beta3_true,
  u     = if (exists("u_true")) u_true else NULL,
  thr   = if (exists("thr_true_mat")) thr_true_mat else NULL,
  delta = if (exists("delta_true")) delta_true else NULL
)

# (3) Fit settings (same as your current call)
fit_args <- list(
  d = 2,
  n_iter = 50000,
  burnin = 10000,
  thin = 10,
  hyper = list(
    a_sigma=1, b_sigma=0.1,
    a_tau1=1, b_tau1=0.5, a_tau2=1, b_tau2=0.5, a_tau3=1, b_tau3=0.5,
    a_sigma0=1, b_sigma0=0.5,
    mu_log_gamma=0, sd_log_gamma=1,
    mu_log_kappa=0, sd_log_kappa=1,
    mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
  ),
  prop_sd = list(
    alpha=0.5, log_gamma=0.05, a=0.3,
    beta1=0.5, beta2=0.2, beta3=0.3,
    b1=0.5, b2=0.2, b3=0.3, b4=0.4,
    log_kappa=0.20, u=0.3, delta=0.3
  ),
  init = NULL,
  verbose = TRUE,
  fix_gamma = TRUE
)

# (4) Storage for all reps
coverage_all <- list()

for (r in 1:n_rep) {
  cat("\n==============================\n")
  cat(sprintf("Replication %d / %d\n", r, n_rep))
  cat("==============================\n")
  
  # --- Fit model (data already defined; same Y_* matrices used each time) ---
  fit_r <- do.call(
    lsirm_sharedpos_layer4_lsgrm_cpp,
    c(list(Y_bin, Y_con, Y_cnt, Y_ord), fit_args)
  )
  
  # Ensure samples nested
  if (is.null(fit_r$samples)) fit_r$samples <- fit_r
  
  # --- Compute coverage summary for this fit ---
  cov_r <- compute_coverage_onefit(fit_r, true_list)
  cov_r$rep <- r
  
  # store
  coverage_all[[r]] <- cov_r
}
ts.plot(fit_r$samples$alpha[,2])
coverage_df <- do.call(rbind, coverage_all)

# (optional) also compute overall coverage across all params within each rep
overall_by_rep <- aggregate(
  cbind(n_param, n_covered, sum_width) ~ rep,
  data = coverage_df,
  FUN = sum
)
overall_by_rep$pct <- 100 * overall_by_rep$n_covered / overall_by_rep$n_param
overall_by_rep$mean_width <- overall_by_rep$sum_width / overall_by_rep$n_param
overall_by_rep$param <- "OVERALL"
overall_by_rep <- overall_by_rep[, c("rep", "param", "n_param", "n_covered", "pct", "mean_width", "sum_width")]

coverage_df <- rbind(
  coverage_df[, c("rep", "param", "n_param", "n_covered", "pct", "mean_width", "sum_width")],
  overall_by_rep
)

# sort nicely
coverage_df <- coverage_df[order(coverage_df$rep, coverage_df$param), ]

# (5) Save to CSV in current directory
write.csv(coverage_df, file = "coverage.csv", row.names = FALSE)

cat("\nSaved: coverage.csv\n")
print(head(coverage_df, 20))

