rm(list = ls())

################################################################################
# simulation_FMC_LSIR_v10.R
#
# Coverage + cluster-recovery simulation for the joint multilayered LSIRM +
# PPCA-FMC model (v10), with side-by-side comparison against a unilayered
# (dichotomized) LSIRM-FMC fit.
#
# What this script does (per replication):
#   (1) Generate 4-layer (binary / continuous / count / ordinal) data with
#       K_true item meta-clusters built into the latent-position generator.
#   (2) Fit the full multilayered v10 model on the original data.
#   (3) Dichotomize every layer at its column mean and stack everything into
#       a single binary matrix; fit the same v10 sampler with only Y_bin
#       populated (the "unilayered" baseline).
#   (4) For both fits compute:
#         - 95% coverage and CI width of every shared parameter that has a
#           well-defined truth (alpha, beta, GRM thresholds when applicable),
#         - Distance coverage |a_i - b_j| * gamma vs D_true * gamma_true,
#         - Cluster-recovery DICE on the FMC partition vs true item cluster
#           (DICE is the pair-counting Sorensen index, label-invariant).
#
# Output (one folder, all CSV/RDS suffixed with _v10):
#   <OUTPUT_DIR>/coverage_per_rep_v10.csv      (long format: rep, model, param, ...)
#   <OUTPUT_DIR>/coverage_summary_v10.csv      (mean across reps)
#   <OUTPUT_DIR>/cluster_dice_v10.csv          (DICE per rep, both models)
#   <OUTPUT_DIR>/sim_settings_v10.rds          (frozen settings)
#
# Single-file design: copy this file + (data/my_LSIRM_FMC_cpp_v10.R,
# data/my_LSIRM_FMC_v10.cpp) to the server and update the two paths below.
################################################################################

################################################################################
# 0. Path setup --- update these two on the server
################################################################################
BASE_DIR <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
OUTPUT_DIR <- file.path(BASE_DIR, "simulation_FMC_LSIRM_v10")
MODEL_DIR  <- file.path(BASE_DIR, "data")  # location of my_LSIRM_FMC_cpp_v10.R / .cpp

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

################################################################################
# 1. Source the v10 sampler
################################################################################
old_wd <- getwd()
setwd(MODEL_DIR)
source(file.path(MODEL_DIR, "my_LSIRM_FMC_cpp_v10.R"))
setwd(old_wd)

library(Rcpp)
library(vegan)

################################################################################
# 2. Simulation configuration
################################################################################
N_REP   <- 10L          # number of replications
SEED0   <- 20260501L    # base seed; replication r uses SEED0 + r

n  <- 150L
P1 <- 10L     # binary
P2 <- 10L     # continuous (robust Student-t)
P3 <- 10L     # count (NB)
P4 <- 10L     # ordinal (GRM, K1=5)
P_total <- P1 + P2 + P3 + P4

d        <- 2L
r_fac    <- 2L
K_star   <- 10L
e0       <- 0.1
K_true   <- 4L
K1       <- 5L           # ordinal categories

gamma_true     <- 1.0
sigma0_sq_true <- 1.0
kappa_true     <- 1.0
nu2_true       <- 5
nu2_fit        <- 4

# True item meta-cluster centers (LSIRM latent space, d=2)
centers_meta <- rbind(
  c(-1.0,  1.0),
  c( 1.0,  1.0),
  c( 1.0, -1.0),
  c(-1.0, -1.0)
)
make_rot <- function(theta) {
  cs <- cos(theta); sn <- sin(theta)
  matrix(c(cs, -sn, sn, cs), 2, 2)
}
sigma_meta <- list(
  diag(c(0.18, 0.18)),
  diag(c(0.45, 0.07)),
  diag(c(0.18, 0.18)),
  make_rot(pi / 4) %*% diag(c(0.45, 0.07)) %*% t(make_rot(pi / 4))
)
centers_resp    <- centers_meta * 1.5
sd_cluster_resp <- 0.80

# True item clusters (round-robin assignment within each layer)
assign_meta <- function(P, K) rep_len(seq_len(K), P)
B1_meta <- assign_meta(P1, K_true)
B2_meta <- assign_meta(P2, K_true)
B3_meta <- assign_meta(P3, K_true)
B4_meta <- assign_meta(P4, K_true)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

# MCMC configuration (per fit). Two fits per replication, so be conservative.
common_mcmc <- list(d = d, n_iter = 30000L, burnin = 10000L, thin = 10L)
fmc_warmup_iter <- as.integer(common_mcmc$burnin / 4)
n_split_merge   <- 5L
S0_scale        <- 0.01
row_center_flag <- FALSE

# Hyperparameters / proposals (mirrored from simulation_4_layered_v10.R)
common_lsirm_hyper <- list(
  a_sigma = 1, b_sigma = 1,
  a_tau1 = 1, b_tau1 = 1, a_tau2 = 1, b_tau2 = 1, a_tau3 = 1, b_tau3 = 1,
  a_sigma0 = 1, b_sigma0 = 1,
  mu_log_gamma1 = 0, sd_log_gamma1 = 0.5,
  mu_log_gamma2 = 0, sd_log_gamma2 = 0.5,
  mu_log_gamma3 = 0, sd_log_gamma3 = 0.5,
  mu_log_gamma4 = 0, sd_log_gamma4 = 0.5,
  mu_log_gamma5 = 0, sd_log_gamma5 = 0.5,
  mu_log_kappa = 0, sd_log_kappa = 0.1,
  mu_beta4 = 0, sd_beta4 = 2,
  mu_beta5 = 0, sd_beta5 = 2
)
common_lsirm_prop_sd <- list(
  alpha1 = 0.5, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.5,
  log_gamma1 = 0.10, log_gamma2 = 0.05, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.05,
  a = 0.30,
  beta1 = 0.40, beta2 = 0.10, beta3 = 0.20, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.30, b2 = 0.20, b3 = 0.20, b4 = 0.20, b5 = 0.20,
  log_kappa = 0.20
)
common_fmc_hyper <- list(
  e0            = e0,
  m0            = rep(0, r_fac),
  kappa0        = 1e-3,
  nu0           = r_fac + 10,
  S0            = S0_scale * diag(r_fac),
  tau_lambda_sq = 4.0,
  a_eps         = 5, b_eps = 0.1,
  a_delta       = 2, b_delta = 1
)

################################################################################
# 3. Helpers
################################################################################
sample_around_centers <- function(centers, cluster_ids, sd) {
  P  <- length(cluster_ids)
  d_ <- ncol(centers)
  centers[cluster_ids, , drop = FALSE] +
    matrix(rnorm(P * d_, 0, sd), P, d_)
}

sample_around_centers_sigma <- function(centers, cluster_ids, sigma_list) {
  P  <- length(cluster_ids)
  d_ <- ncol(centers)
  out <- matrix(0, P, d_)
  for (k in seq_along(sigma_list)) {
    idx <- which(cluster_ids == k)
    if (length(idx) == 0) next
    L <- chol(sigma_list[[k]])
    z <- matrix(rnorm(length(idx) * d_), length(idx), d_)
    out[idx, ] <- matrix(centers[k, ], length(idx), d_, byrow = TRUE) + z %*% L
  }
  out
}

dist_mat <- function(A, B) {
  n_ <- nrow(A); P_ <- nrow(B)
  out <- matrix(0, n_, P_)
  for (j in 1:P_) {
    out[, j] <- sqrt(rowSums((A - matrix(B[j, ], n_, ncol(A), byrow = TRUE))^2))
  }
  out
}

invlogit <- function(x) 1 / (1 + exp(-x))

generate_grm_thresholds <- function(P, K) {
  Km1 <- K - 1
  out <- matrix(NA_real_, P, Km1)
  for (j in 1:P) out[j, ] <- sort(rnorm(Km1, 0, 1.5), decreasing = TRUE)
  out
}

generate_grm_data <- function(ETA, beta_thr, K_cat) {
  n_  <- nrow(ETA); P_ <- ncol(ETA)
  Y   <- matrix(NA_integer_, n_, P_)
  for (j in 1:P_) for (i in 1:n_) {
    p_ge <- invlogit(ETA[i, j] + beta_thr[j, ])
    p <- numeric(K_cat)
    p[1] <- 1 - p_ge[1]
    for (k in 2:(K_cat - 1)) p[k] <- p_ge[k - 1] - p_ge[k]
    p[K_cat] <- p_ge[K_cat - 1]
    p[p < 0] <- 0
    ps <- sum(p)
    if (ps <= 0) { p <- rep(0, K_cat); p[round(K_cat / 2)] <- 1 } else p <- p / ps
    Y[i, j] <- sample.int(K_cat, size = 1, prob = p)
  }
  storage.mode(Y) <- "integer"
  Y
}

# ----- coverage helpers -----
coverage_from_samples <- function(samples_mat, true_vec, prob = c(0.025, 0.975)) {
  samples_mat <- as.matrix(samples_mat)
  if (length(true_vec) != ncol(samples_mat))
    stop("true_vec length must match ncol(samples_mat).")
  q_lo <- apply(samples_mat, 2, quantile, probs = prob[1], na.rm = TRUE)
  q_hi <- apply(samples_mat, 2, quantile, probs = prob[2], na.rm = TRUE)
  widths <- q_hi - q_lo
  cov_ind <- rep(NA_integer_, length(true_vec))
  ok <- !is.na(true_vec)
  cov_ind[ok] <- as.integer(q_lo[ok] <= true_vec[ok] & true_vec[ok] <= q_hi[ok])
  list(
    n_param     = sum(ok),
    n_covered   = sum(cov_ind[ok] == 1),
    pct_covered = if (sum(ok) == 0) NA_real_ else 100 * mean(cov_ind[ok] == 1),
    mean_width  = if (sum(ok) == 0) NA_real_ else mean(widths[ok]),
    sum_width   = if (sum(ok) == 0) 0 else sum(widths[ok])
  )
}

coverage_from_scalar <- function(samp_vec, true_val, prob = c(0.025, 0.975)) {
  samp_vec <- as.numeric(samp_vec)
  q <- quantile(samp_vec, probs = prob, na.rm = TRUE)
  width <- q[2] - q[1]
  covered <- as.integer(q[1] <= true_val & true_val <= q[2])
  list(n_param = 1L, n_covered = covered,
       pct_covered = 100 * covered, mean_width = width, sum_width = width)
}

# distance coverage: posterior 95% CI of dist(a_i, b_j) * gamma_s vs D_true * gamma_true
distance_coverage_layer <- function(a_samps, b_samps, log_gamma_samps,
                                    D_true, gamma_true = 1.0,
                                    prob = c(0.025, 0.975)) {
  stopifnot(length(dim(a_samps)) == 3, length(dim(b_samps)) == 3)
  S <- dim(a_samps)[1]
  n_ <- dim(a_samps)[2]
  d_ <- dim(a_samps)[3]
  P_ <- dim(b_samps)[2]
  stopifnot(dim(b_samps)[1] == S, dim(b_samps)[3] == d_)
  stopifnot(all(dim(D_true) == c(n_, P_)))
  stopifnot(length(log_gamma_samps) == S)

  gamma_s <- exp(as.numeric(log_gamma_samps))
  true_scaled <- D_true * gamma_true
  n_param <- n_ * P_
  n_covered <- 0L
  sum_width <- 0.0

  for (j in 1:P_) {
    dx <- a_samps[, , 1] - matrix(b_samps[, j, 1], nrow = S, ncol = n_)
    if (d_ >= 2) {
      dy <- a_samps[, , 2] - matrix(b_samps[, j, 2], nrow = S, ncol = n_)
      dist_s <- sqrt(dx^2 + dy^2)
    } else {
      dist_s <- abs(dx)
    }
    dist_s <- dist_s * matrix(gamma_s, nrow = S, ncol = n_)

    q_lo <- apply(dist_s, 2, quantile, probs = prob[1], na.rm = TRUE)
    q_hi <- apply(dist_s, 2, quantile, probs = prob[2], na.rm = TRUE)
    sum_width <- sum_width + sum(q_hi - q_lo)

    tj <- true_scaled[, j]
    n_covered <- n_covered + sum(q_lo <= tj & tj <= q_hi, na.rm = TRUE)
  }

  list(
    n_param = n_param, n_covered = n_covered,
    pct_covered = 100 * (n_covered / n_param),
    mean_width = sum_width / n_param,
    sum_width = sum_width
  )
}

# pair-counting Sorensen-Dice index for two partitions of the same set of items.
# DICE = 2A / (2A + B + C)
#   A = # pairs together in both partitions
#   B = # pairs together in true only
#   C = # pairs together in est only
# Range [0, 1], 1 = perfect agreement; rotation/label-invariant.
pair_dice <- function(true_part, est_part) {
  stopifnot(length(true_part) == length(est_part))
  P <- length(true_part)
  T_mat <- outer(true_part, true_part, "==")
  E_mat <- outer(est_part,  est_part,  "==")
  upper <- upper.tri(T_mat)
  Tv <- T_mat[upper]
  Ev <- E_mat[upper]
  A <- sum(Tv & Ev)
  B <- sum(Tv & !Ev)
  C <- sum(!Tv & Ev)
  if (2 * A + B + C == 0) return(NA_real_)
  2 * A / (2 * A + B + C)
}

# adjusted Rand index (kept as a side metric since it is the standard companion)
adj_rand_index <- function(a, b) {
  tab <- table(a, b)
  n_  <- sum(tab)
  if (n_ < 2) return(NA_real_)
  sum_c <- sum(choose(rowSums(tab), 2))
  sum_k <- sum(choose(colSums(tab), 2))
  sum_t <- sum(choose(tab, 2))
  expected <- sum_c * sum_k / choose(n_, 2)
  max_idx  <- (sum_c + sum_k) / 2
  if (max_idx == expected) return(1)
  (sum_t - expected) / (max_idx - expected)
}

# ---------- final partition from posterior similarity matrix ----------
final_partition_from_psm <- function(co_cluster, K_target) {
  P <- nrow(co_cluster)
  K_use <- max(2, min(K_target, P - 1))
  hc <- hclust(as.dist(1 - co_cluster), method = "average")
  cutree(hc, k = K_use)
}

# ---------- PCA-smart FMC init (from simulation_4_layered_v10.R) ----------
init_eta_via_pca <- function(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, r_fac) {
  blocks <- list()
  if (ncol(Y_bin)  > 0) blocks <- c(blocks, list(scale(Y_bin)))
  if (ncol(Y_con)  > 0) blocks <- c(blocks, list(scale(Y_con)))
  if (ncol(Y_cnt)  > 0) blocks <- c(blocks, list(scale(log1p(Y_cnt))))
  if (ncol(Y_ord1) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord1))))
  if (ncol(Y_ord2) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord2))))
  X <- do.call(cbind, blocks)
  X[is.na(X)] <- 0
  pc <- prcomp(t(X), center = TRUE, scale. = FALSE)
  rk <- min(r_fac, ncol(pc$x))
  out <- matrix(0, nrow = ncol(X), ncol = r_fac)
  out[, seq_len(rk)] <- pc$x[, seq_len(rk), drop = FALSE]
  out
}

build_fmc_init_smart <- function(eta_init_pca, n_, K_star, r_fac) {
  km <- kmeans(eta_init_pca, centers = K_star, nstart = 25, iter.max = 50)
  list(
    rho            = rep(1 / K_star, K_star),
    eta            = eta_init_pca,
    c              = km$cluster,
    mu             = km$centers,
    Sigma          = array(rep(diag(r_fac), K_star),
                           dim = c(r_fac, r_fac, K_star)),
    Lambda         = matrix(rnorm(n_ * r_fac, 0, 0.5), n_, r_fac),
    delta          = rep(0, n_),
    sigma_eps_sq   = 0.1,
    sigma_delta_sq = 0.1
  )
}

################################################################################
# 4. Per-replication routines
################################################################################
generate_data <- function(seed) {
  set.seed(seed)

  resp_cl <- sample.int(nrow(centers_resp), n, replace = TRUE)
  A_true  <- sample_around_centers(centers_resp, resp_cl, sd_cluster_resp)
  B1_true <- sample_around_centers_sigma(centers_meta, B1_meta, sigma_meta)
  B2_true <- sample_around_centers_sigma(centers_meta, B2_meta, sigma_meta)
  B3_true <- sample_around_centers_sigma(centers_meta, B3_meta, sigma_meta)
  B4_true <- sample_around_centers_sigma(centers_meta, B4_meta, sigma_meta)

  alpha1_true <- rnorm(n, 0, 1.0)
  alpha2_true <- rnorm(n, 0, 1.0)
  alpha3_true <- rnorm(n, 0, 1.0)
  alpha4_true <- rnorm(n, 0, 1.0)
  beta1_true  <- rnorm(P1, 0, 1.0)
  beta2_true  <- rnorm(P2, 0, 1.0)
  beta3_true  <- rnorm(P3, 0, 0.5)
  beta4_true  <- generate_grm_thresholds(P4, K1)

  D1_true <- dist_mat(A_true, B1_true)
  D2_true <- dist_mat(A_true, B2_true)
  D3_true <- dist_mat(A_true, B3_true)
  D4_true <- dist_mat(A_true, B4_true)

  ETA1 <- outer(alpha1_true, rep(1, P1)) - outer(rep(1, n), beta1_true) - gamma_true * D1_true
  ETA2 <- outer(alpha2_true, rep(1, P2)) - outer(rep(1, n), beta2_true) - gamma_true * D2_true
  ETA3 <- outer(alpha3_true, rep(1, P3)) - outer(rep(1, n), beta3_true) - gamma_true * D3_true
  ETA4 <- outer(alpha4_true, rep(1, P4))                                 - gamma_true * D4_true

  Y_bin <- matrix(rbinom(n * P1, 1, as.vector(invlogit(ETA1))), n, P1)

  lambda_t <- matrix(rgamma(n * P2, shape = nu2_true / 2, rate = nu2_true / 2), n, P2)
  Y_con <- ETA2 + matrix(rnorm(n * P2, 0, sqrt(sigma0_sq_true)), n, P2) / sqrt(lambda_t)
  storage.mode(Y_con) <- "numeric"

  MU_cnt  <- exp(ETA3)
  size_nb <- 1 / kappa_true
  Y_cnt   <- matrix(rnbinom(n * P3, size = size_nb, mu = as.vector(MU_cnt)), n, P3)

  Y_ord1 <- generate_grm_data(ETA4, beta4_true, K1)
  Y_ord2 <- matrix(0L, nrow = n, ncol = 0)

  list(
    Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt,
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2,
    A_true = A_true,
    B1_true = B1_true, B2_true = B2_true, B3_true = B3_true, B4_true = B4_true,
    D1_true = D1_true, D2_true = D2_true, D3_true = D3_true, D4_true = D4_true,
    alpha1_true = alpha1_true, alpha2_true = alpha2_true,
    alpha3_true = alpha3_true, alpha4_true = alpha4_true,
    beta1_true  = beta1_true,  beta2_true  = beta2_true,
    beta3_true  = beta3_true,  beta4_true  = beta4_true,
    resp_cl = resp_cl
  )
}

# Dichotomize at column mean (binary unchanged; ordinal/count/cont -> {0,1}).
dichotomize_dataset <- function(dat) {
  bin_thresh <- function(M) {
    if (ncol(M) == 0) return(matrix(0L, nrow(M), 0))
    out <- matrix(0L, nrow(M), ncol(M))
    for (j in seq_len(ncol(M))) {
      mu_j <- mean(M[, j], na.rm = TRUE)
      out[, j] <- as.integer(M[, j] > mu_j)
    }
    out
  }
  Y_bin_all <- cbind(
    dat$Y_bin,
    bin_thresh(dat$Y_con),
    bin_thresh(dat$Y_cnt),
    bin_thresh(dat$Y_ord1)
  )
  storage.mode(Y_bin_all) <- "integer"
  Y_bin_all
}

fit_multilayer <- function(dat) {
  eta_init <- init_eta_via_pca(dat$Y_bin, dat$Y_con, dat$Y_cnt,
                               dat$Y_ord1, dat$Y_ord2, r_fac)
  fmc_init <- build_fmc_init_smart(eta_init, n, K_star, r_fac)

  lsirm_fmc_v10_cpp(
    Y_bin   = dat$Y_bin,
    Y_con   = dat$Y_con,
    Y_cnt   = dat$Y_cnt,
    Y_ord1  = dat$Y_ord1,
    Y_ord2  = dat$Y_ord2,
    r_fac   = r_fac, K_star = K_star, e0 = e0,
    d       = common_mcmc$d,
    n_iter  = common_mcmc$n_iter,
    burnin  = common_mcmc$burnin,
    thin    = common_mcmc$thin,
    nu2     = nu2_fit,
    lsirm_hyper   = common_lsirm_hyper,
    fmc_hyper     = common_fmc_hyper,
    lsirm_prop_sd = common_lsirm_prop_sd,
    lsirm_init    = NULL,
    fmc_init      = fmc_init,
    save_lambda_full = FALSE,
    save_delta_full  = FALSE,
    save_eta_full    = TRUE,
    compute_co_cluster_online = TRUE,
    fmc_warmup    = fmc_warmup_iter,
    n_split_merge = n_split_merge,
    row_center    = row_center_flag,
    verbose       = FALSE,
    fix_gamma     = TRUE
  )
}

fit_unilayer <- function(dat) {
  Y_bin_all <- dichotomize_dataset(dat)
  Y_empty_n <- matrix(0,  nrow = n, ncol = 0)
  Y_empty_i <- matrix(0L, nrow = n, ncol = 0)

  eta_init <- init_eta_via_pca(Y_bin_all, Y_empty_n, Y_empty_i,
                               Y_empty_i, Y_empty_i, r_fac)
  fmc_init <- build_fmc_init_smart(eta_init, n, K_star, r_fac)

  lsirm_fmc_v10_cpp(
    Y_bin   = Y_bin_all,
    Y_con   = Y_empty_n,
    Y_cnt   = Y_empty_i,
    Y_ord1  = Y_empty_i,
    Y_ord2  = Y_empty_i,
    r_fac   = r_fac, K_star = K_star, e0 = e0,
    d       = common_mcmc$d,
    n_iter  = common_mcmc$n_iter,
    burnin  = common_mcmc$burnin,
    thin    = common_mcmc$thin,
    nu2     = nu2_fit,
    lsirm_hyper   = common_lsirm_hyper,
    fmc_hyper     = common_fmc_hyper,
    lsirm_prop_sd = common_lsirm_prop_sd,
    lsirm_init    = NULL,
    fmc_init      = fmc_init,
    save_lambda_full = FALSE,
    save_delta_full  = FALSE,
    save_eta_full    = TRUE,
    compute_co_cluster_online = TRUE,
    fmc_warmup    = fmc_warmup_iter,
    n_split_merge = n_split_merge,
    row_center    = row_center_flag,
    verbose       = FALSE,
    fix_gamma     = TRUE
  )
}

################################################################################
# 5. Coverage extraction
################################################################################
make_row <- function(rep, model, param, info) {
  data.frame(
    rep        = rep,
    model      = model,
    param      = param,
    n_param    = info$n_param,
    n_covered  = info$n_covered,
    pct        = info$pct_covered,
    mean_width = info$mean_width,
    sum_width  = info$sum_width,
    stringsAsFactors = FALSE
  )
}

coverage_multilayer <- function(rep, fit, dat) {
  rows <- list()

  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "alpha1",
    coverage_from_samples(fit$alpha1, dat$alpha1_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "alpha2",
    coverage_from_samples(fit$alpha2, dat$alpha2_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "alpha3",
    coverage_from_samples(fit$alpha3, dat$alpha3_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "alpha4",
    coverage_from_samples(fit$alpha4, dat$alpha4_true))

  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "beta1",
    coverage_from_samples(fit$beta1, dat$beta1_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "beta2",
    coverage_from_samples(fit$beta2, dat$beta2_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "beta3",
    coverage_from_samples(fit$beta3, dat$beta3_true))

  if (!is.null(fit$beta4)) {
    b4 <- fit$beta4                       # [n_save, P4, K1-1]
    n_save <- dim(b4)[1]
    b4_mat <- matrix(b4, nrow = n_save, ncol = P4 * (K1 - 1))
    rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "beta4_grm_thr",
      coverage_from_samples(b4_mat, as.vector(dat$beta4_true)))
  }

  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "dist_bin",
    distance_coverage_layer(fit$a, fit$b1, fit$log_gamma1, dat$D1_true, gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "dist_con",
    distance_coverage_layer(fit$a, fit$b2, fit$log_gamma2, dat$D2_true, gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "dist_cnt",
    distance_coverage_layer(fit$a, fit$b3, fit$log_gamma3, dat$D3_true, gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "dist_ord",
    distance_coverage_layer(fit$a, fit$b4, fit$log_gamma4, dat$D4_true, gamma_true))

  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "gamma1",
    coverage_from_scalar(exp(fit$log_gamma1), gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "gamma2",
    coverage_from_scalar(exp(fit$log_gamma2), gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "gamma3",
    coverage_from_scalar(exp(fit$log_gamma3), gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "gamma4",
    coverage_from_scalar(exp(fit$log_gamma4), gamma_true))

  rows[[length(rows) + 1]] <- make_row(rep, "multilayer", "sigma0_sq",
    coverage_from_scalar(fit$sigma0_sq, sigma0_sq_true))

  do.call(rbind, rows)
}

coverage_unilayer <- function(rep, fit, dat) {
  # In the unilayered fit Y_bin holds all P_total items, so:
  #   fit$alpha1 = [n_save, n]        single respondent random effect
  #   fit$beta1  = [n_save, P_total]  per-item difficulty over the stacked binary matrix
  # The data generator has layer-specific (alpha_l_true, beta_l_true). We compare
  # the unilayer estimates to each non-ordinal layer's truth so the user can see
  # which layer the dichotomized fit best approximates. Ordinal (alpha4, beta4)
  # is omitted by design (no GRM in the dichotomized model).
  rows <- list()

  # --- alpha: single coverage row. The unilayer model has one respondent random
  # effect; we compare it to the per-respondent mean of (alpha1, alpha2, alpha3)_true
  # (non-ordinal layers only). This treats the unilayer alpha as the "average"
  # random effect across the dichotomized layers.
  alpha_mean_true <- (dat$alpha1_true + dat$alpha2_true + dat$alpha3_true) / 3
  rows[[length(rows) + 1]] <- make_row(rep, "unilayer", "alpha",
    coverage_from_samples(fit$alpha1, alpha_mean_true))

  # --- beta: single coverage row. Concatenate the non-ordinal layer truths
  # (binary / continuous / count) and compare to the matching first P1+P2+P3
  # columns of the unilayer beta1 samples. Ordinal block (last P4 columns) is
  # excluded by design.
  beta_all <- as.matrix(fit$beta1)                                # [n_save, P_total]
  idx_nonord <- 1:(P1 + P2 + P3)
  beta_true_nonord <- c(dat$beta1_true, dat$beta2_true, dat$beta3_true)
  rows[[length(rows) + 1]] <- make_row(rep, "unilayer", "beta",
    coverage_from_samples(beta_all[, idx_nonord, drop = FALSE], beta_true_nonord))

  # --- distance and gamma (kept) ---
  D_all_true <- cbind(dat$D1_true, dat$D2_true, dat$D3_true, dat$D4_true)
  rows[[length(rows) + 1]] <- make_row(rep, "unilayer", "dist_all",
    distance_coverage_layer(fit$a, fit$b1, fit$log_gamma1, D_all_true, gamma_true))
  rows[[length(rows) + 1]] <- make_row(rep, "unilayer", "gamma1",
    coverage_from_scalar(exp(fit$log_gamma1), gamma_true))

  do.call(rbind, rows)
}

cluster_recovery_metrics <- function(fit, true_part, K_target) {
  co <- fit$fmc_co_cluster
  est_partition <- final_partition_from_psm(co, K_target)
  list(
    dice          = pair_dice(true_part, est_partition),
    ari           = adj_rand_index(est_partition, true_part),
    K_plus_median = median(fit$fmc_K_plus),
    K_plus_mean   = mean(fit$fmc_K_plus),
    split_rate    = fit$fmc_split_merge$split_rate,
    merge_rate    = fit$fmc_split_merge$merge_rate
  )
}

################################################################################
# 6. Main loop
################################################################################
all_coverage <- list()
all_dice     <- list()

cat(sprintf("\n========== simulation_FMC_LSIR_v10  (n_rep = %d) ==========\n", N_REP))
cat(sprintf("  output dir : %s\n", OUTPUT_DIR))
cat(sprintf("  n=%d, P=%d/%d/%d/%d (total %d), K_true=%d, K*=%d, r_fac=%d\n",
            n, P1, P2, P3, P4, P_total, K_true, K_star, r_fac))
cat(sprintf("  MCMC: n_iter=%d, burnin=%d, thin=%d  (per fit; 2 fits per rep)\n",
            common_mcmc$n_iter, common_mcmc$burnin, common_mcmc$thin))

t_start <- Sys.time()

for (r in seq_len(N_REP)) {
  cat(sprintf("\n---- replication %d / %d ----\n", r, N_REP))
  rep_seed <- SEED0 + r
  dat <- generate_data(rep_seed)

  cat(sprintf("  [rep %d] fitting multilayer ...\n", r))
  fit_m <- fit_multilayer(dat)
  cov_m <- coverage_multilayer(r, fit_m, dat)
  rec_m <- cluster_recovery_metrics(fit_m, true_item_cluster, K_target = K_true)

  cat(sprintf("  [rep %d] fitting unilayer (dichotomized) ...\n", r))
  fit_u <- fit_unilayer(dat)
  cov_u <- coverage_unilayer(r, fit_u, dat)
  rec_u <- cluster_recovery_metrics(fit_u, true_item_cluster, K_target = K_true)

  all_coverage[[r]] <- rbind(cov_m, cov_u)
  all_dice[[r]] <- data.frame(
    rep                   = r,
    seed                  = rep_seed,
    multilayer_dice       = rec_m$dice,
    unilayer_dice         = rec_u$dice,
    multilayer_ari        = rec_m$ari,
    unilayer_ari          = rec_u$ari,
    multilayer_K_median   = rec_m$K_plus_median,
    unilayer_K_median     = rec_u$K_plus_median,
    multilayer_split_rate = rec_m$split_rate,
    unilayer_split_rate   = rec_u$split_rate,
    multilayer_merge_rate = rec_m$merge_rate,
    unilayer_merge_rate   = rec_u$merge_rate,
    stringsAsFactors      = FALSE
  )

  cat(sprintf("  [rep %d] DICE: multi=%.3f  uni=%.3f   ARI: multi=%.3f  uni=%.3f\n",
              r, rec_m$dice, rec_u$dice, rec_m$ari, rec_u$ari))

  # incremental save (so partial results survive a server interruption)
  write.csv(do.call(rbind, all_coverage),
            file.path(OUTPUT_DIR, "coverage_per_rep_v10.csv"), row.names = FALSE)
  write.csv(do.call(rbind, all_dice),
            file.path(OUTPUT_DIR, "cluster_dice_v10.csv"), row.names = FALSE)
}

t_end <- Sys.time()
cat(sprintf("\nTotal wall time: %.2f min\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

################################################################################
# 7. Aggregate summary
################################################################################
coverage_df <- do.call(rbind, all_coverage)
dice_df     <- do.call(rbind, all_dice)

agg_cov <- aggregate(
  cbind(n_param, n_covered, pct, mean_width, sum_width) ~ model + param,
  data = coverage_df,
  FUN  = function(x) c(mean = mean(x), sd = sd(x))
)
# Flatten
flat <- function(M, base) {
  if (is.matrix(M)) {
    out <- data.frame(M[, "mean"], M[, "sd"])
    names(out) <- paste0(base, c("_mean", "_sd"))
  } else {
    out <- data.frame(M); names(out) <- base
  }
  out
}
summary_df <- data.frame(
  model = agg_cov$model, param = agg_cov$param,
  flat(agg_cov$n_param,   "n_param"),
  flat(agg_cov$n_covered, "n_covered"),
  flat(agg_cov$pct,        "pct"),
  flat(agg_cov$mean_width, "mean_width"),
  flat(agg_cov$sum_width,  "sum_width"),
  stringsAsFactors = FALSE
)
write.csv(summary_df,
          file.path(OUTPUT_DIR, "coverage_summary_v10.csv"), row.names = FALSE)

dice_summary <- data.frame(
  model            = c("multilayer", "unilayer"),
  dice_mean        = c(mean(dice_df$multilayer_dice, na.rm = TRUE),
                       mean(dice_df$unilayer_dice,   na.rm = TRUE)),
  dice_sd          = c(sd(dice_df$multilayer_dice,   na.rm = TRUE),
                       sd(dice_df$unilayer_dice,     na.rm = TRUE)),
  ari_mean         = c(mean(dice_df$multilayer_ari, na.rm = TRUE),
                       mean(dice_df$unilayer_ari,   na.rm = TRUE)),
  ari_sd           = c(sd(dice_df$multilayer_ari,   na.rm = TRUE),
                       sd(dice_df$unilayer_ari,     na.rm = TRUE)),
  K_median_mean    = c(mean(dice_df$multilayer_K_median, na.rm = TRUE),
                       mean(dice_df$unilayer_K_median,   na.rm = TRUE)),
  split_rate_mean  = c(mean(dice_df$multilayer_split_rate, na.rm = TRUE),
                       mean(dice_df$unilayer_split_rate,   na.rm = TRUE)),
  stringsAsFactors = FALSE
)
write.csv(dice_summary,
          file.path(OUTPUT_DIR, "cluster_dice_summary_v10.csv"), row.names = FALSE)

saveRDS(list(
  N_REP = N_REP, SEED0 = SEED0,
  n = n, P1 = P1, P2 = P2, P3 = P3, P4 = P4, P_total = P_total,
  d = d, r_fac = r_fac, K_star = K_star, e0 = e0, K_true = K_true, K1 = K1,
  gamma_true = gamma_true, sigma0_sq_true = sigma0_sq_true,
  kappa_true = kappa_true, nu2_true = nu2_true, nu2_fit = nu2_fit,
  S0_scale = S0_scale, n_split_merge = n_split_merge,
  row_center_flag = row_center_flag,
  centers_meta = centers_meta, sigma_meta = sigma_meta,
  centers_resp = centers_resp, sd_cluster_resp = sd_cluster_resp,
  true_item_cluster = true_item_cluster,
  common_mcmc = common_mcmc,
  common_lsirm_hyper = common_lsirm_hyper,
  common_lsirm_prop_sd = common_lsirm_prop_sd,
  common_fmc_hyper = common_fmc_hyper
), file.path(OUTPUT_DIR, "sim_settings_v10.rds"))

cat("\n========== summary (mean across reps) ==========\n")
print(summary_df)
cat("\n========== cluster recovery summary ==========\n")
print(dice_summary)
cat(sprintf("\nArtifacts written to: %s\n", OUTPUT_DIR))
cat("  - coverage_per_rep_v10.csv\n")
cat("  - coverage_summary_v10.csv\n")
cat("  - cluster_dice_v10.csv\n")
cat("  - cluster_dice_summary_v10.csv\n")
cat("  - sim_settings_v10.rds\n")
