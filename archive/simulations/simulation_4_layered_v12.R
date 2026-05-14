rm(list = ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile / source v12 wrappers
#
#    v12 model (joint LSIRM + EPA pairwise partition prior):
#      - LSIRM block byte-identical to v11/v10 (5-layer max:
#        bin/con/cnt/ord1/ord2, per-item kappa, robust Student-t
#        continuous, GRM ordinal).
#      - Item position prior: independent Gaussian
#          b_j^{(l)} ~ N_d(0, sigma_b^2 I_d).
#      - Clustering: EPA pairwise partition prior (Dahl et al. 2017)
#        with similarity
#          lambda_qr(z, tau) = exp(-tau ||z_q - z_r||^2)
#        on z_q := b_{j(q)}^{(l(q))}.
#      - b_j MH ratio includes the EPA prior contribution
#        (LSIRM and EPA are FULLY JOINTLY coupled through z).
#      - Updates on the partition use the decoupling property:
#          single-item EPA Gibbs on c, random-swap MH on sigma,
#          log-scale MH on (alpha, tau), Jain-Neal nonconjugate
#          split-merge with restricted Gibbs scans.
#
#    Simulation:
#      - 4-layer simulation (binary / continuous / count / ordinal),
#        identical data-generating mechanism to simulation_4_layered_v11.R
#        so the only systematic difference is the partition sampler.
# =========================================================
data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v12.R"))   # joint LSIRM + EPA v12
source(file.path(proj_dir, "utils.R"))

set.seed(20260501)

# =========================================================
# 1) Simulation settings (identical to simulation_4_layered_v11.R)
# =========================================================
n  <- 150
P1 <- 30  # binary
P2 <- 30  # continuous (robust Student-t)
P3 <- 30  # count (NB)
P4 <- 30  # ordinal (GRM, K1=5 categories)
P_total <- P1 + P2 + P3 + P4

d        <- 2L
K_true   <- 4L

K1       <- 5L

gamma_true     <- 1.0
sigma0_sq_true <- 1.0
kappa_true     <- 1.0
nu2_true       <- 5
nu2_fit        <- 4

# Item position prior scale (paper §priors-non-clustering, eq:b-prior):
#   b_j^{(l)} ~ N_d(0, sigma_b^2 I_d).
# This `sigma_b` is the prior STANDARD DEVIATION (not variance).  Paper uses
# sigma_b^2 = 1, i.e. sigma_b = 1.  Same scale as the respondent prior.
sigma_b <- 1.0

# =========================================================
# 2) TRUE item meta-clusters in LSIRM latent space
# =========================================================
centers_meta <- rbind(
  c(-1.0,  1.0),
  c( 1.0,  1.0),
  c( 1.0, -1.0),
  c(-1.0, -1.0)
)
stopifnot(nrow(centers_meta) == K_true, ncol(centers_meta) == d)

sigma_meta <- list(
  diag(c(0.15, 0.15)),
  diag(c(0.20, 0.20)),
  diag(c(0.10, 0.10)),
  diag(c(0.18, 0.18))
)

assign_meta <- function(P, K) rep_len(seq_len(K), P)

B1_meta <- assign_meta(P1, K_true)
B2_meta <- assign_meta(P2, K_true)
B3_meta <- assign_meta(P3, K_true)
B4_meta <- assign_meta(P4, K_true)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

centers_resp <- centers_meta * 0.7
sd_cluster_resp <- 0.5

# =========================================================
# 3) Sampling helpers
# =========================================================
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

# =========================================================
# 4) TRUE positions
# =========================================================
resp_cl <- sample.int(nrow(centers_resp), n, replace = TRUE)
A_true  <- sample_around_centers(centers_resp, resp_cl, sd_cluster_resp)

B1_true <- sample_around_centers_sigma(centers_meta, B1_meta, sigma_meta)
B2_true <- sample_around_centers_sigma(centers_meta, B2_meta, sigma_meta)
B3_true <- sample_around_centers_sigma(centers_meta, B3_meta, sigma_meta)
B4_true <- sample_around_centers_sigma(centers_meta, B4_meta, sigma_meta)

# =========================================================
# 5) TRUE intercepts / difficulties / GRM thresholds
# =========================================================
true_sigma_alpha <- 1.0
true_tau_beta1   <- 1.0
true_tau_beta2   <- 1.0
true_tau_beta3   <- 0.5

alpha1_true <- rnorm(n, 0, true_sigma_alpha)
alpha2_true <- rnorm(n, 0, true_sigma_alpha)
alpha3_true <- rnorm(n, 0, true_sigma_alpha)
alpha4_true <- rnorm(n, 0, true_sigma_alpha)

beta1_true  <- rnorm(P1, 0, true_tau_beta1)
beta2_true  <- rnorm(P2, 0, true_tau_beta2)
beta3_true  <- rnorm(P3, 0, true_tau_beta3)

generate_grm_thresholds <- function(P, K) {
  Km1 <- K - 1
  out <- matrix(NA_real_, P, Km1)
  for (j in 1:P) out[j, ] <- sort(rnorm(Km1, 0, 1.5), decreasing = TRUE)
  out
}
beta4_true <- generate_grm_thresholds(P4, K1)

cat("Beta4 thresholds (first 3 items):\n")
print(round(beta4_true[1:3, ], 2))

# =========================================================
# 6) Linear predictors per layer
# =========================================================
D1_true <- dist_mat(A_true, B1_true)
D2_true <- dist_mat(A_true, B2_true)
D3_true <- dist_mat(A_true, B3_true)
D4_true <- dist_mat(A_true, B4_true)

ETA1_true <- outer(alpha1_true, rep(1, P1)) - outer(rep(1, n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha2_true, rep(1, P2)) - outer(rep(1, n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha3_true, rep(1, P3)) - outer(rep(1, n), beta3_true) - gamma_true * D3_true
ETA4_true <- outer(alpha4_true, rep(1, P4))                                 - gamma_true * D4_true

# =========================================================
# 7) Generate observations
# =========================================================
P_bern <- invlogit(ETA1_true)
Y_bin  <- matrix(rbinom(n * P1, 1, as.vector(P_bern)), n, P1)

lambda_true <- matrix(rgamma(n * P2, shape = nu2_true / 2, rate = nu2_true / 2), n, P2)
Y_con <- ETA2_true + matrix(rnorm(n * P2, 0, sqrt(sigma0_sq_true)), n, P2) / sqrt(lambda_true)
storage.mode(Y_con) <- "numeric"

MU_cnt  <- exp(ETA3_true)
size_nb <- 1 / kappa_true
Y_cnt   <- matrix(rnbinom(n * P3, size = size_nb, mu = as.vector(MU_cnt)), n, P3)

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
Y_ord1 <- generate_grm_data(ETA4_true, beta4_true, K1)

Y_ord2 <- matrix(0L, nrow = n, ncol = 0)

cat("\n=== Data Summary ===\n")
cat(sprintf("Y_bin:  %d x %d (binary)\n",     nrow(Y_bin),  ncol(Y_bin)))
cat(sprintf("Y_con:  %d x %d (continuous)\n", nrow(Y_con),  ncol(Y_con)))
cat(sprintf("Y_cnt:  %d x %d (count)\n",      nrow(Y_cnt),  ncol(Y_cnt)))
cat(sprintf("Y_ord1: %d x %d (ord, K=%d)\n",  nrow(Y_ord1), ncol(Y_ord1), K1))
cat(sprintf("\nOrdinal response table:\n")); print(table(Y_ord1))
cat(sprintf("\nTrue item cluster sizes: %s\n",
            paste(table(true_item_cluster), collapse = " ")))

item_names_full <- c(paste0("bin_",  seq_len(P1)),
                     paste0("con_",  seq_len(P2)),
                     paste0("cnt_",  seq_len(P3)),
                     paste0("ord1_", seq_len(P4)))
stopifnot(length(item_names_full) == P_total)

# =========================================================
# 8) Experiment dispatch + plot dir
# =========================================================
# Two dispatch modes:
#  1. A/B/C ablation (legacy):
#       V12_EXPERIMENT in {"A","B","C"}; sets (update_alpha, update_tau)
#       per the early ablation runs.  alpha_fix = tau_fix = 1.0.
#  2. (alpha, tau) sensitivity grid (new):
#       V12_SENS_LABEL = a free-form label (used in plot_dir name)
#       V12_ALPHA_FIX = numeric (fixed alpha value, used as init)
#       V12_TAU_FIX   = numeric (fixed tau   value, used as init)
#       V12_UPDATE_ALPHA = "TRUE"/"FALSE" (default "FALSE" for sensitivity)
#       V12_UPDATE_TAU   = "TRUE"/"FALSE" (default "FALSE")
# If V12_SENS_LABEL is set, sensitivity mode wins.  Otherwise use A/B/C.
sens_label <- Sys.getenv("V12_SENS_LABEL", "")
b_epa_coupling <- TRUE  # paper-canonical for all dispatches

if (nchar(sens_label) > 0) {
  alpha_fix    <- as.numeric(Sys.getenv("V12_ALPHA_FIX", "1.0"))
  tau_fix      <- as.numeric(Sys.getenv("V12_TAU_FIX",   "1.0"))
  update_alpha <- as.logical(Sys.getenv("V12_UPDATE_ALPHA", "FALSE"))
  update_tau   <- as.logical(Sys.getenv("V12_UPDATE_TAU",   "FALSE"))
  exp_label    <- sens_label
  cat(sprintf("\n[Sensitivity %s] alpha_fix=%g, tau_fix=%g, update_alpha=%s, update_tau=%s\n",
              exp_label, alpha_fix, tau_fix, update_alpha, update_tau))
  plot_dir <- file.path(
    proj_dir, "plot",
    sprintf("simulation_4_layered_v12_sens_%s", exp_label))
} else {
  exp_label <- toupper(Sys.getenv("V12_EXPERIMENT", "A"))
  if (!exp_label %in% c("A", "B", "C")) {
    stop(sprintf("Unknown V12_EXPERIMENT=%s; use A, B, or C, or set V12_SENS_LABEL.", exp_label))
  }
  alpha_fix    <- 1.0
  tau_fix      <- 1.0
  update_alpha <- switch(exp_label, A = TRUE,  B = TRUE,  C = FALSE)
  update_tau   <- switch(exp_label, A = TRUE,  B = FALSE, C = FALSE)
  cat(sprintf("\n[Experiment %s] update_alpha=%s, update_tau=%s, alpha_fix=%g, tau_fix=%g\n",
              exp_label, update_alpha, update_tau, alpha_fix, tau_fix))
  plot_dir <- file.path(
    proj_dir, "plot",
    sprintf("simulation_4_layered_v12_d%d_sigb_%g_exp%s",
            d, sigma_b, exp_label))
}
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal)
  if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]

draw_ellipse <- function(mu, Sigma, level = 0.95, col = 1, lwd = 1, lty = 2, n = 200) {
  t <- seq(0, 2 * pi, length.out = n)
  circle <- rbind(cos(t), sin(t))
  ev <- eigen(Sigma, symmetric = TRUE)
  scale <- sqrt(qchisq(level, 2))
  pts   <- t(ev$vectors %*% diag(sqrt(pmax(ev$values, 0))) %*% circle) * scale
  pts   <- sweep(pts, 2, mu, "+")
  lines(pts[, 1], pts[, 2], col = col, lwd = lwd, lty = lty)
}

plot_positions_v12 <- function(A, B_list, B_meta_list, A_resp_cl,
                               title = "TRUE latent positions (4-layer, v12)",
                               rng = NA, draw_true_ellipses = TRUE) {
  layer_names <- c("Binary", "Continuous", "Count", "Ordinal")
  pch_layer   <- c(21, 22, 23, 24)
  all_pts <- rbind(A, do.call(rbind, B_list))
  if (any(is.na(rng))) rng <- range(all_pts, na.rm = TRUE)
  pal_use <- expand_pal(K_true, cluster_pal)
  base::plot(A, pch = 21, bg = "gray80", col = "black",
             xlab = "Dim1", ylab = "Dim2", main = title,
             xlim = rng, ylim = rng, cex = 0.7)
  if (draw_true_ellipses) {
    for (k in seq_len(K_true))
      draw_ellipse(centers_meta[k, ], sigma_meta[[k]],
                   level = 0.95, col = pal_use[k], lwd = 2, lty = 2)
  }
  for (l in seq_along(B_list)) {
    points(B_list[[l]], pch = pch_layer[l],
           bg = pal_use[B_meta_list[[l]]], col = "black", cex = 1.6)
  }
  legend("topleft",
         legend = c("Respondents (a)",
                    paste(layer_names, "items"),
                    paste("Meta-cluster", seq_len(K_true))),
         pch    = c(21, pch_layer, rep(15, K_true)),
         pt.bg  = c("gray80", rep("white", 4), pal_use),
         col    = c("black", rep("black", 4), pal_use),
         bty = "n", cex = 0.7)
}

pdf(file.path(plot_dir, "true_positions.pdf"), width = 10, height = 8)
plot_positions_v12(A_true,
                   list(B1_true, B2_true, B3_true, B4_true),
                   list(B1_meta, B2_meta, B3_meta, B4_meta),
                   resp_cl)
dev.off()

# =========================================================
# 9) EPA initialisation
#
#    The EPA pmf is non-exchangeable in (sigma, c) so the initial
#    partition matters less than under v11's NIW mixture.  We
#    initialise:
#      c     : k-means on a cheap PC proxy of the response matrix,
#              giving a non-degenerate starting partition.
#      sigma : random permutation (Uniform on Perm(P_total))
#      alpha : prior mean = a_alpha / b_alpha
#      tau   : prior mean = a_tau / b_tau
# =========================================================
init_b_proxy_via_pca <- function(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, d) {
  blocks <- list()
  if (ncol(Y_bin)  > 0) blocks <- c(blocks, list(scale(Y_bin)))
  if (ncol(Y_con)  > 0) blocks <- c(blocks, list(scale(Y_con)))
  if (ncol(Y_cnt)  > 0) blocks <- c(blocks, list(scale(log1p(Y_cnt))))
  if (ncol(Y_ord1) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord1))))
  if (ncol(Y_ord2) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord2))))
  X <- do.call(cbind, blocks)
  X[is.na(X)] <- 0
  pc <- prcomp(t(X), center = TRUE, scale. = FALSE)
  rk <- min(d, ncol(pc$x))
  out <- matrix(0, nrow = ncol(X), ncol = d)
  out[, seq_len(rk)] <- pc$x[, seq_len(rk), drop = FALSE]
  out
}

b_init_proxy <- init_b_proxy_via_pca(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, d)
K_init <- K_true
km <- kmeans(b_init_proxy, centers = K_init, nstart = 25, iter.max = 50)

epa_init_smart <- list(
  c     = as.integer(km$cluster),       # 1-based; wrapper converts to 0-based
  sigma = sample.int(P_total, P_total),  # uniform random permutation
  alpha = alpha_fix,
  tau   = tau_fix
)
cat(sprintf("[EPA init] PCA-b-proxy sd per dim: %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))
cat(sprintf("[EPA init] k-means cluster sizes: %s\n",
            paste(table(km$cluster), collapse = " ")))

# =========================================================
# 10) Hyperparameters / proposal SDs / MCMC settings
# =========================================================
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
  alpha1 = 1.15, alpha2 = 0.48, alpha3 = 1.15, alpha4 = 0.92, alpha5 = 0.5,
  log_gamma1 = 0.07, log_gamma2 = 0.018, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.05,
  a = 0.26,
  beta1 = 0.44, beta2 = 0.13, beta3 = 0.37, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.33, b2 = 0.13, b3 = 0.30, b4 = 0.26, b5 = 0.20,
  log_kappa = 0.20
)

# EPA hyperpriors (paper §hyperpriors-and-identifiability):
#   alpha ~ Gamma(a_alpha, b_alpha)
#   tau   ~ Gamma(a_tau,   b_tau)
#   delta = 0  (fixed in current implementation)
# Paper recommends weakly-informative defaults (1, 1) for both.
common_epa_hyper <- list(
  a_alpha = 1.0, b_alpha = 1.0,
  a_tau   = 1.0, b_tau   = 1.0,
  delta   = 0.0
)

# Log-scale random-walk MH proposal SDs for (alpha, tau).
common_epa_prop_sd <- list(log_alpha = 0.5, log_tau = 0.5)

# MCMC settings.  EPA's per-sweep cost is O(P^2) per pmf evaluation; the
# partition block does O(K * P_total) of these per sweep.  Lambda matrix
# caching (v12 cpp) makes each evaluation a memory lookup loop instead of
# a fresh exp() over P^2 pairs.
common_mcmc <- list(
  d      = d,
  n_iter = as.integer(Sys.getenv("V12_N_ITER", "20000")),
  burnin = as.integer(Sys.getenv("V12_BURNIN", "8000")),
  thin   = as.integer(Sys.getenv("V12_THIN",   "5"))
)

# Paper has no warmup phase; all updates run every sweep from iteration 0.
epa_warmup_iter <- 0L

# Paper defaults: M_SM = 1, R = 5 (Jain-Neal launch scans).
# Override M_SM via env var V12_M_SM (paper §sec:sm: "default M_SM in {1, 5}").
n_split_merge   <- as.integer(Sys.getenv("V12_M_SM", "1"))
n_split_merge_R <- as.integer(Sys.getenv("V12_R_SM", "5"))

# Permutation random-swap attempts per sweep.  Default = P_total.
n_perm_swaps    <- as.integer(P_total)

# =========================================================
# 11) Run v12 joint MCMC
# =========================================================
cat(sprintf(
  "\n========== simulation_4_layered_v12 [exp %s] (n=%d, P=%d/%d/%d/%d, K_true=%d, sigma_b=%g, M_SM=%d, R=%d, M_perm=%d, update_alpha=%s, update_tau=%s) ==========\n",
  exp_label, n, P1, P2, P3, P4, K_true, sigma_b,
  n_split_merge, n_split_merge_R, n_perm_swaps,
  if (update_alpha) "TRUE" else "FALSE",
  if (update_tau)   "TRUE" else "FALSE"))

result <- lsirm_epa_v12_cpp(
  Y_bin   = Y_bin,
  Y_con   = Y_con,
  Y_cnt   = Y_cnt,
  Y_ord1  = Y_ord1,
  Y_ord2  = Y_ord2,
  d       = common_mcmc$d,
  n_iter  = common_mcmc$n_iter,
  burnin  = common_mcmc$burnin,
  thin    = common_mcmc$thin,
  nu2     = nu2_fit,
  lsirm_hyper   = common_lsirm_hyper,
  epa_hyper     = common_epa_hyper,
  lsirm_prop_sd = common_lsirm_prop_sd,
  epa_prop_sd   = common_epa_prop_sd,
  lsirm_init    = NULL,
  epa_init      = epa_init_smart,
  sigma_b       = sigma_b,
  compute_co_cluster_online = TRUE,
  epa_warmup    = epa_warmup_iter,
  n_split_merge   = n_split_merge,
  n_split_merge_R = n_split_merge_R,
  n_perm_swaps    = n_perm_swaps,
  b_epa_coupling  = b_epa_coupling,
  update_alpha    = update_alpha,
  update_tau      = update_tau,
  verbose       = TRUE,
  fix_gamma     = FALSE,
  procrustes_target = list(a  = A_true,
                           b1 = B1_true, b2 = B2_true,
                           b3 = B3_true, b4 = B4_true)
)

samps <- result

# =========================================================
# 12) Acceptance summary
# =========================================================
acc <- result$accept
cat("\n-- LSIRM Acceptance --\n")
cat(sprintf("  alpha1..4 mean : %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$alpha1), mean(acc$alpha2), mean(acc$alpha3), mean(acc$alpha4)))
cat(sprintf("  beta1..3 mean  : %.3f / %.3f / %.3f\n",
            mean(acc$beta1), mean(acc$beta2), mean(acc$beta3)))
cat(sprintf("  log_gamma1..4  : %.3f / %.3f / %.3f / %.3f\n",
            acc$log_gamma1, acc$log_gamma2, acc$log_gamma3, acc$log_gamma4))
cat(sprintf("  a / b1..4 mean : %.3f / %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$a), mean(acc$b1), mean(acc$b2), mean(acc$b3), mean(acc$b4)))
cat(sprintf("  log_kappa per-item mean: %.3f\n", mean(acc$log_kappa)))

diag <- result$epa_diagnostics
cat("\n-- EPA partition diagnostics --\n")
cat(sprintf("  split: %d / %d  (rate %.3f)\n",
            diag$split_accepts, diag$split_attempts, diag$split_rate))
cat(sprintf("  merge: %d / %d  (rate %.3f)\n",
            diag$merge_accepts, diag$merge_attempts, diag$merge_rate))
cat(sprintf("  sigma swaps: %.0f / %.0f  (rate %.3f)\n",
            diag$sigma_swap_accepts, diag$sigma_swap_attempts,
            diag$sigma_swap_rate))
cat(sprintf("  alpha MH: %d / %d  (rate %.3f)\n",
            diag$alpha_epa_accepts, diag$alpha_epa_attempts,
            diag$alpha_epa_rate))
cat(sprintf("  tau   MH: %d / %d  (rate %.3f)\n",
            diag$tau_epa_accepts, diag$tau_epa_attempts,
            diag$tau_epa_rate))

leg <- list(
  x      = "topright",
  legend = c("Posterior mean", "95% credible interval", "True value"),
  col    = c("darkgreen", "blue", "red"),
  lwd    = 2,
  lty    = c(1, 3, 2),
  bty    = "n",
  cex    = 0.8
)

# =========================================================
# 13) LSIRM traceplots: positions a, b1..b4
# =========================================================
pdf(file.path(plot_dir, "trace_a.pdf"), width = 8, height = 12)
par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
for (i in 1:min(8, dim(samps$a)[2])) for (j in 1:dim(samps$a)[3]) {
  ts.plot(samps$a[, i, j], main = paste0("a: ", i, "_", j))
  abline(h = A_true[i, j], col = "red", lty = 2)
}
dev.off()

B_true_list <- list(B1_true, B2_true, B3_true, B4_true)
for (l in 1:4) {
  bn <- paste0("b", l)
  bs <- samps[[bn]]
  if (is.null(bs) || dim(bs)[2] == 0) next
  pdf(file.path(plot_dir, paste0("trace_", bn, ".pdf")), width = 8, height = 12)
  par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
  for (i in 1:dim(bs)[2]) for (j in 1:dim(bs)[3]) {
    ts.plot(bs[, i, j], main = paste0(bn, ": item", i, " (cl=", true_item_cluster[
      switch(l, 1:P1, P1 + 1:P2, P1 + P2 + 1:P3, P1 + P2 + P3 + 1:P4)[i]
    ], ") d", j))
    abline(h = B_true_list[[l]][i, j], col = "red", lty = 2)
  }
  dev.off()
}

# =========================================================
# 14) LSIRM traceplots: alpha1..4, beta1..3, beta4 thresholds
# =========================================================
alpha_trues <- list(alpha1_true, alpha2_true, alpha3_true, alpha4_true)
for (l in 1:4) {
  an <- paste0("alpha", l)
  pdf(file.path(plot_dir, paste0("trace_", an, ".pdf")), width = 8, height = 12)
  plot_trace_vec(samps[[an]], alpha_trues[[l]], an, mfrow = c(3, 2), leg = leg)
  dev.off()
}

pdf(file.path(plot_dir, "trace_beta1.pdf"), width = 8, height = 12)
plot_trace_vec(samps$beta1, beta1_true, "beta1", mfrow = c(3, 2), leg = leg)
dev.off()
pdf(file.path(plot_dir, "trace_beta2.pdf"), width = 8, height = 12)
plot_trace_vec(samps$beta2, beta2_true, "beta2", mfrow = c(3, 2), leg = leg)
dev.off()
pdf(file.path(plot_dir, "trace_beta3.pdf"), width = 8, height = 12)
plot_trace_vec(samps$beta3, beta3_true, "beta3", mfrow = c(3, 2), leg = leg)
dev.off()

if (!is.null(samps$beta4)) {
  b4s <- samps$beta4
  n_save <- dim(b4s)[1]
  b4_mat <- matrix(b4s, nrow = n_save, ncol = P4 * (K1 - 1))
  b4_true_vec <- as.vector(beta4_true)
  pdf(file.path(plot_dir, "trace_beta4_thr.pdf"), width = 8, height = 12)
  plot_trace_vec(b4_mat, b4_true_vec, "beta4 (GRM thr)", mfrow = c(3, 2), leg = leg)
  dev.off()
}

# =========================================================
# 15) LSIRM traceplots: scalar hyperparameters
# =========================================================
pdf(file.path(plot_dir, "trace_lsirm_extra.pdf"), width = 8, height = 14)
par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
plot_trace_scalar(samps$sigma0_sq,  true = sigma0_sq_true, main = "sigma0_sq")
plot_trace_scalar(samps$log_gamma1, true = gamma_true,     main = "gamma1 (Bin)",  transform = exp)
plot_trace_scalar(samps$log_gamma2, true = gamma_true,     main = "gamma2 (Con)",  transform = exp)
plot_trace_scalar(samps$log_gamma3, true = gamma_true,     main = "gamma3 (Cnt)",  transform = exp)
plot_trace_scalar(samps$log_gamma4, true = gamma_true,     main = "gamma4 (Ord)",  transform = exp)
if (!is.null(samps$lambda2_mean))
  plot_trace_scalar(samps$lambda2_mean, true = 1.0, main = "lambda2_mean (robust)")
dev.off()

pdf(file.path(plot_dir, "trace_sigma_alpha.pdf"), width = 8, height = 8)
par(mfrow = c(2, 2), mar = c(3, 3, 2, 1))
for (l in 1:4) {
  sn <- paste0("sigma_alpha", l, "_sq")
  if (!is.null(samps[[sn]]))
    plot_trace_scalar(samps[[sn]], true = true_sigma_alpha^2, main = sn)
}
dev.off()

if (!is.null(samps$log_kappa) && is.matrix(samps$log_kappa) && ncol(samps$log_kappa) > 0) {
  lk <- samps$log_kappa
  P3d <- ncol(lk)
  pdf(file.path(plot_dir, "trace_kappa_per_item.pdf"), width = 10, height = 12)
  par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
  for (j in 1:P3d) {
    kx <- exp(lk[, j])
    ts.plot(kx, main = sprintf("kappa[cnt_%d]", j))
    abline(h = c(mean(kx), quantile(kx, c(.025, .975)), kappa_true),
           col = c("darkgreen", "blue", "blue", "red"),
           lwd = 2, lty = c(1, 3, 3, 2))
  }
  dev.off()
}

# =========================================================
# 16) LSIRM box-CI grouping plots
# =========================================================
pdf(file.path(plot_dir, "boxplot_groups.pdf"), width = 10, height = 8)
for (l in 1:4) {
  an <- paste0("alpha", l)
  plot_group_box_ci(samps[[an]], alpha_trues[[l]], an, per_page = 24)
}
plot_group_box_ci(samps$beta1, beta1_true, "beta1", per_page = 24)
plot_group_box_ci(samps$beta2, beta2_true, "beta2", per_page = 24)
plot_group_box_ci(samps$beta3, beta3_true, "beta3", per_page = 24)
if (!is.null(samps$beta4)) {
  b4s <- samps$beta4
  n_save <- dim(b4s)[1]
  b4_mat <- matrix(b4s, nrow = n_save, ncol = P4 * (K1 - 1))
  plot_group_box_ci(b4_mat, as.vector(beta4_true), "beta4 (GRM thr)", per_page = 24)
}
dev.off()

# =========================================================
# 17) Posterior mean LSIRM positions
# =========================================================
A_hat  <- apply(samps$a,  c(2, 3), mean)
B1_hat <- apply(samps$b1, c(2, 3), mean)
B2_hat <- apply(samps$b2, c(2, 3), mean)
B3_hat <- apply(samps$b3, c(2, 3), mean)
B4_hat <- apply(samps$b4, c(2, 3), mean)

pdf(file.path(plot_dir, "positions_true_vs_hat.pdf"), width = 14, height = 7)
par(mfrow = c(1, 2))
plot_positions_v12(A_true,
                   list(B1_true, B2_true, B3_true, B4_true),
                   list(B1_meta, B2_meta, B3_meta, B4_meta),
                   resp_cl, title = "TRUE positions")
plot_positions_v12(A_hat,
                   list(B1_hat, B2_hat, B3_hat, B4_hat),
                   list(B1_meta, B2_meta, B3_meta, B4_meta),
                   resp_cl, title = "Posterior mean positions (Procrustes-aligned)")
dev.off()

# =========================================================
# 18) Distance recovery
# =========================================================
gamma1_post <- mean(exp(samps$log_gamma1))
gamma2_post <- mean(exp(samps$log_gamma2))
gamma3_post <- mean(exp(samps$log_gamma3))
gamma4_post <- mean(exp(samps$log_gamma4))

D1_hat <- dist_mat(A_hat, B1_hat) * gamma1_post
D2_hat <- dist_mat(A_hat, B2_hat) * gamma2_post
D3_hat <- dist_mat(A_hat, B3_hat) * gamma3_post
D4_hat <- dist_mat(A_hat, B4_hat) * gamma4_post

vec <- function(M) as.vector(M)
D_true_all <- cbind(D1_true, D2_true, D3_true, D4_true) * gamma_true
D_hat_all  <- cbind(D1_hat,  D2_hat,  D3_hat,  D4_hat)
cor_all <- cor(vec(D_true_all), vec(D_hat_all))
cor_l   <- c(cor(vec(D1_true * gamma_true), vec(D1_hat)),
             cor(vec(D2_true * gamma_true), vec(D2_hat)),
             cor(vec(D3_true * gamma_true), vec(D3_hat)),
             cor(vec(D4_true * gamma_true), vec(D4_hat)))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("  All layers combined: %.3f\n", cor_all))
cat(sprintf("  L1 Binary:           %.3f\n", cor_l[1]))
cat(sprintf("  L2 Continuous:       %.3f\n", cor_l[2]))
cat(sprintf("  L3 Count:            %.3f\n", cor_l[3]))
cat(sprintf("  L4 Ordinal:          %.3f\n", cor_l[4]))

pdf(file.path(plot_dir, "distance_recovery_scatter.pdf"), width = 7, height = 7)
plot(vec(D_true_all), vec(D_hat_all),
     pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.4,
     xlab = "True dist x gamma_true",
     ylab = "Estimated dist x gamma_post",
     main = sprintf("Distance recovery, all layers (r=%.3f)", cor_all))
abline(a = 0, b = 1, col = "red", lwd = 2)
dev.off()

# =========================================================
# 19) Layer-scale diagnostic R_scale
# =========================================================
B_hat_list <- list(B1_hat, B2_hat, B3_hat, B4_hat)
layer_scale <- function(B) {
  Bc <- sweep(B, 2, colMeans(B), "-")
  sqrt(mean(rowSums(Bc^2)))
}
s_l       <- vapply(B_hat_list, layer_scale, numeric(1))
R_scale   <- max(s_l) / min(s_l)
cat("\n=== Layer-scale diagnostic ===\n")
cat(sprintf("  s_l per layer : %s\n", paste(round(s_l, 3), collapse = " / ")))
cat(sprintf("  R_scale       : %.3f  (>1.5 -> rescale concern)\n", R_scale))

# =========================================================
# 20) EPA traceplots
# =========================================================
n_save <- length(samps$epa_alpha)

pdf(file.path(plot_dir, "epa_trace_K_plus.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(samps$epa_K_plus, main = "Occupied cluster count K_+",
        ylab = expression(K["+"]^{(s)}))
abline(h = c(mean(samps$epa_K_plus), K_true),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2))
legend("topright", legend = c("posterior mean", "K_true"),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

pdf(file.path(plot_dir, "epa_trace_alpha_tau.pdf"), width = 9, height = 8)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
ts.plot(samps$epa_alpha, main = "EPA mass alpha",
        ylab = expression(alpha))
abline(h = c(mean(samps$epa_alpha), quantile(samps$epa_alpha, c(.025, .975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
ts.plot(samps$epa_tau, main = "EPA temperature tau",
        ylab = expression(tau))
abline(h = c(mean(samps$epa_tau), quantile(samps$epa_tau, c(.025, .975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
dev.off()

pdf(file.path(plot_dir, "epa_trace_log_pmf.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(samps$epa_log_pmf, main = "log p_EPA(P | sigma, z, alpha, tau)",
        ylab = expression(log~p[EPA]))
abline(h = mean(samps$epa_log_pmf), col = "darkgreen", lwd = 2)
dev.off()

# Cluster-size traces (over all labels in [1, P_total]).
n_l_trace <- t(apply(samps$epa_c, 1, function(v) tabulate(v, nbins = P_total)))
# Plot only labels that are ever occupied above some threshold.
ever_used <- which(colSums(n_l_trace > 0) > 0)
pal_used <- expand_pal(length(ever_used), cluster_pal)

pdf(file.path(plot_dir, "epa_trace_cluster_sizes.pdf"), width = 10, height = 5)
par(mar = c(4, 4, 3, 1))
matplot(n_l_trace[, ever_used], type = "l", lty = 1,
        col = pal_used,
        xlab = "saved iter", ylab = expression(n[l]^{(s)}),
        main = sprintf("Item cluster sizes (c) -- %d labels ever used",
                       length(ever_used)))
dev.off()

# =========================================================
# 21) EPA clustering recovery
# =========================================================
co_cluster <- samps$epa_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

mode_label <- function(v)
  as.integer(names(sort(table(v), decreasing = TRUE))[1])
item_cluster_mode <- apply(samps$epa_c, 2, mode_label)

median_K_plus   <- max(2, round(median(samps$epa_K_plus)))
hc_co           <- hclust(as.dist(1 - co_cluster), method = "average")
final_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

adj_rand_index <- function(a, b) {
  tab <- table(a, b)
  n_  <- sum(tab)
  if (n_ < 2) return(NA_real_)
  sum_c <- sum(choose(rowSums(tab), 2))
  sum_k <- sum(choose(colSums(tab), 2))
  sum_t <- sum(choose(tab, 2))
  expected <- sum_c * sum_k / choose(n_, 2)
  max_idx  <- (sum_c + sum_k) / 2
  # Tolerance-based equality: avoids floating-point near-equality
  # treating max_idx == expected as inequality and division-by-near-zero.
  if (abs(max_idx - expected) < 1e-12) return(1)
  (sum_t - expected) / (max_idx - expected)
}

# Dahl (2006) least-squares model-based clustering point estimate.
# Co-cluster matrix is symmetric with diagonal 1, so loss is computed on
# the upper triangle only (~2x faster, no change in argmin).
dahl_partition <- function(c_samples, C_post) {
  S <- nrow(c_samples)
  ut <- upper.tri(C_post)
  C_post_ut <- C_post[ut]
  loss <- numeric(S)
  for (s in seq_len(S)) {
    cs <- c_samples[s, ]
    Cs_ut <- (outer(cs, cs, FUN = "=="))[ut]
    loss[s] <- sum((Cs_ut - C_post_ut)^2)
  }
  s_star <- which.min(loss)
  cl_raw <- as.integer(c_samples[s_star, ])
  cl <- as.integer(factor(cl_raw, levels = unique(cl_raw)))
  list(partition = cl, iter = s_star, loss = loss[s_star],
       K_plus = length(unique(cl)))
}

# Binder loss (Lau & Green 2007) restricted point estimate.
binder_partition <- function(c_samples, C_post) {
  S <- nrow(c_samples)
  ut <- upper.tri(C_post)
  pi_ut <- C_post[ut]
  weight <- 1 - 2 * pi_ut
  loss <- numeric(S)
  for (s in seq_len(S)) {
    cs <- c_samples[s, ]
    Cs_ut <- (outer(cs, cs, FUN = "=="))[ut]
    loss[s] <- sum(weight * Cs_ut)
  }
  s_star <- which.min(loss)
  cl_raw <- as.integer(c_samples[s_star, ])
  cl <- as.integer(factor(cl_raw, levels = unique(cl_raw)))
  list(partition = cl, iter = s_star, loss = loss[s_star],
       K_plus = length(unique(cl)))
}

# Variation of Information (Meila 2007) restricted point estimate.
vi_distance <- function(a, b) {
  P <- length(a)
  if (P == 0) return(0)
  fa <- as.integer(factor(a))
  fb <- as.integer(factor(b))
  Ka <- max(fa); Kb <- max(fb)
  joint <- tabulate((fa - 1L) * Kb + fb, nbins = Ka * Kb)
  joint_mat <- matrix(joint, Ka, Kb, byrow = TRUE) / P
  pa <- rowSums(joint_mat)
  pb <- colSums(joint_mat)
  H_a <- -sum(pa[pa > 0] * log(pa[pa > 0]))
  H_b <- -sum(pb[pb > 0] * log(pb[pb > 0]))
  pj_pos_idx <- joint_mat > 0
  pj_pos <- joint_mat[pj_pos_idx]
  ratio  <- (pa %o% pb)[pj_pos_idx]
  I_ab <- sum(pj_pos * log(pj_pos / ratio))
  H_a + H_b - 2 * I_ab
}

vi_partition <- function(c_samples, vi_max_S = 500L) {
  S <- nrow(c_samples)
  if (S > vi_max_S) {
    keep <- sort(sample.int(S, vi_max_S))
    c_use <- c_samples[keep, , drop = FALSE]
  } else {
    c_use <- c_samples
    keep <- seq_len(S)
  }
  S_use <- nrow(c_use)
  D <- matrix(0, S_use, S_use)
  for (i in seq_len(S_use - 1)) {
    for (j in (i + 1):S_use) {
      v <- vi_distance(c_use[i, ], c_use[j, ])
      D[i, j] <- v; D[j, i] <- v
    }
  }
  exp_vi <- rowMeans(D)
  i_star <- which.min(exp_vi)
  cl_raw <- as.integer(c_use[i_star, ])
  cl <- as.integer(factor(cl_raw, levels = unique(cl_raw)))
  list(partition = cl, iter = keep[i_star],
       expected_vi = exp_vi[i_star],
       K_plus = length(unique(cl)),
       n_used = S_use)
}

dahl <- dahl_partition(samps$epa_c, co_cluster)
dahl_K_plus        <- dahl$K_plus
dahl_partition_vec <- dahl$partition

binder <- binder_partition(samps$epa_c, co_cluster)
binder_K_plus        <- binder$K_plus
binder_partition_vec <- binder$partition

cat("\n[VI] Computing pairwise VI distances ...\n")
vi <- vi_partition(samps$epa_c, vi_max_S = 500L)
vi_K_plus        <- vi$K_plus
vi_partition_vec <- vi$partition
cat(sprintf("[VI] Used %d MCMC samples for pairwise VI computation.\n", vi$n_used))

ari_partition <- adj_rand_index(final_partition,      true_item_cluster)
ari_mode      <- adj_rand_index(item_cluster_mode,    true_item_cluster)
ari_dahl      <- adj_rand_index(dahl_partition_vec,   true_item_cluster)
ari_binder    <- adj_rand_index(binder_partition_vec, true_item_cluster)
ari_vi        <- adj_rand_index(vi_partition_vec,     true_item_cluster)

cat("\n=== Clustering recovery ===\n")
cat(sprintf("  median K_+ : %.0f   (true K = %d)\n", median_K_plus, K_true))
cat(sprintf("  Dahl   K_+ : %.0f   (iter %d, L2 loss  = %.3f)\n",
            dahl_K_plus,   dahl$iter,   dahl$loss))
cat(sprintf("  Binder K_+ : %.0f   (iter %d, Binder   = %.3f)\n",
            binder_K_plus, binder$iter, binder$loss))
cat(sprintf("  VI     K_+ : %.0f   (iter %d, exp VI   = %.3f)\n",
            vi_K_plus,     vi$iter,     vi$expected_vi))
cat(sprintf("  ARI(hclust partition vs true): %.3f\n", ari_partition))
cat(sprintf("  ARI(Dahl   partition vs true): %.3f\n", ari_dahl))
cat(sprintf("  ARI(Binder partition vs true): %.3f\n", ari_binder))
cat(sprintf("  ARI(VI     partition vs true): %.3f\n", ari_vi))
cat(sprintf("  ARI(mode label       vs true): %.3f  (subject to label switching)\n",
            ari_mode))
cat("\nCross-tab: hclust partition vs true_item_cluster\n")
print(table(final_partition, true_item_cluster))
cat("\nCross-tab: Dahl partition vs true_item_cluster\n")
print(table(dahl_partition_vec, true_item_cluster))
cat("\nCross-tab: Binder partition vs true_item_cluster\n")
print(table(binder_partition_vec, true_item_cluster))
cat("\nCross-tab: VI partition vs true_item_cluster\n")
print(table(vi_partition_vec, true_item_cluster))
cat("\nCross-tab: item_cluster_mode vs true_item_cluster\n")
print(table(item_cluster_mode, true_item_cluster))

pdf(file.path(plot_dir, "epa_co_cluster_true_order.pdf"), width = 9, height = 8)
ord_true <- order(true_item_cluster)
par(mar = c(7, 7, 3, 1))
image(seq_len(P_total), seq_len(P_total), co_cluster[ord_true, ord_true],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("EPA co-clustering reordered by TRUE cluster  (ARI=%.3f)", ari_partition))
axis(1, at = seq_len(P_total), labels = item_names_full[ord_true],
     las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_true],
     las = 2, cex.axis = 0.55)
bnd <- cumsum(table(true_item_cluster[ord_true]))
abline(v = bnd + 0.5, col = "red", lty = 2)
abline(h = bnd + 0.5, col = "red", lty = 2)
box()
dev.off()

pdf(file.path(plot_dir, "epa_co_cluster_partition_order.pdf"), width = 9, height = 8)
ord_fmc <- order(final_partition)
par(mar = c(7, 7, 3, 1))
image(seq_len(P_total), seq_len(P_total), co_cluster[ord_fmc, ord_fmc],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = "EPA co-clustering reordered by hclust partition")
axis(1, at = seq_len(P_total), labels = item_names_full[ord_fmc],
     las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_fmc],
     las = 2, cex.axis = 0.55)
box()
dev.off()

# =========================================================
# 22) b-position cluster scatter
# =========================================================
B_hat_pm <- rbind(B1_hat, B2_hat, B3_hat, B4_hat)
rownames(B_hat_pm) <- item_names_full

K_for_pal <- max(K_true, max(item_cluster_mode))
pdf(file.path(plot_dir, "epa_b_postmean.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
pal_true <- expand_pal(K_true,    cluster_pal)
pal_fmc  <- expand_pal(K_for_pal, cluster_pal)
plot(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, cex = 1.3,
     col = pal_true[true_item_cluster],
     xlab = "b dim 1", ylab = "b dim 2",
     main = "b posterior mean (coloured by TRUE cluster)")
text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
plot(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, cex = 1.3,
     col = pal_fmc[item_cluster_mode],
     xlab = "b dim 1", ylab = "b dim 2",
     main = "b posterior mean (coloured by EPA mode)")
text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
dev.off()

# =========================================================
# 23) Biplots
# =========================================================
biplot_pdf <- function(A_hat, B_hat, item_partition, item_names,
                       title, file) {
  k_max <- max(item_partition)
  pal_use <- expand_pal(k_max, cluster_pal)
  pdf(file, width = 10, height = 8)
  par(mar = c(4, 4, 3, 1))
  xr <- range(A_hat[, 1], B_hat[, 1])
  yr <- range(A_hat[, 2], B_hat[, 2])
  plot(A_hat[, 1], A_hat[, 2], pch = 21,
       bg = adjustcolor("gray60", alpha.f = 0.30),
       col = "gray40", cex = 0.7,
       xlab = "Dim1", ylab = "Dim2", main = title,
       xlim = xr + c(-1, 1) * 0.1 * diff(xr),
       ylim = yr + c(-1, 1) * 0.1 * diff(yr))
  points(B_hat[, 1], B_hat[, 2], pch = 22,
         bg = pal_use[item_partition], col = "black", cex = 1.7)
  text(B_hat[, 1], B_hat[, 2], labels = item_names, pos = 4, cex = 0.55)
  uq <- sort(unique(item_partition))
  legend("topright",
         legend = c("Respondents (a_i)", sprintf("Item partition %d", uq)),
         pch    = c(21, rep(22, length(uq))),
         pt.bg  = c(adjustcolor("gray60", alpha.f = 0.30), pal_use[uq]),
         col    = c("gray40", rep("black", length(uq))),
         bty = "n", cex = 0.75)
  dev.off()
}

biplot_pdf(A_hat, B_hat_pm, final_partition, item_names_full,
           sprintf("Biplot — items coloured by hclust partition (ARI=%.3f)", ari_partition),
           file.path(plot_dir, "biplot_hclust_partition.pdf"))
biplot_pdf(A_hat, B_hat_pm, dahl_partition_vec, item_names_full,
           sprintf("Biplot — items coloured by Dahl partition (ARI=%.3f)", ari_dahl),
           file.path(plot_dir, "biplot_dahl_partition.pdf"))
biplot_pdf(A_hat, B_hat_pm, binder_partition_vec, item_names_full,
           sprintf("Biplot — items coloured by Binder partition (ARI=%.3f)", ari_binder),
           file.path(plot_dir, "biplot_binder_partition.pdf"))
biplot_pdf(A_hat, B_hat_pm, vi_partition_vec, item_names_full,
           sprintf("Biplot — items coloured by VI partition (ARI=%.3f)", ari_vi),
           file.path(plot_dir, "biplot_vi_partition.pdf"))
biplot_pdf(A_hat, B_hat_pm, true_item_cluster, item_names_full,
           "Biplot — items coloured by TRUE meta-cluster",
           file.path(plot_dir, "biplot_true_cluster.pdf"))

# =========================================================
# 24) CSV outputs
# =========================================================
write.csv(data.frame(item              = item_names_full,
                     true_cluster      = true_item_cluster,
                     epa_mode_cluster  = item_cluster_mode,
                     hclust_partition  = final_partition,
                     dahl_partition    = dahl_partition_vec,
                     binder_partition  = binder_partition_vec,
                     vi_partition      = vi_partition_vec),
          file.path(plot_dir, "epa_item_clusters.csv"), row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(plot_dir, "epa_co_cluster.csv"))

write.csv(data.frame(
            mean_K_plus   = mean(samps$epa_K_plus),
            median_K_plus = median(samps$epa_K_plus),
            sd_K_plus     = sd(samps$epa_K_plus),
            min_K_plus    = min(samps$epa_K_plus),
            max_K_plus    = max(samps$epa_K_plus),
            K_true        = K_true,
            n_save        = n_save,

            hclust_K_plus = median_K_plus,
            dahl_K_plus   = dahl_K_plus,
            dahl_iter     = dahl$iter,
            dahl_loss     = dahl$loss,
            binder_K_plus = binder_K_plus,
            binder_iter   = binder$iter,
            binder_loss   = binder$loss,
            vi_K_plus     = vi_K_plus,
            vi_iter       = vi$iter,
            vi_expected   = vi$expected_vi,
            vi_n_used     = vi$n_used,

            ari_hclust = ari_partition,
            ari_dahl   = ari_dahl,
            ari_binder = ari_binder,
            ari_vi     = ari_vi,
            ari_mode   = ari_mode,

            cor_dist_all   = cor_all,
            R_scale        = R_scale,

            mean_alpha_epa = mean(samps$epa_alpha),
            mean_tau_epa   = mean(samps$epa_tau),

            split_attempts = diag$split_attempts,
            split_accepts  = diag$split_accepts,
            split_rate     = diag$split_rate,
            merge_attempts = diag$merge_attempts,
            merge_accepts  = diag$merge_accepts,
            merge_rate     = diag$merge_rate,
            sigma_swap_rate = diag$sigma_swap_rate,
            alpha_epa_rate  = diag$alpha_epa_rate,
            tau_epa_rate    = diag$tau_epa_rate
          ),
          file.path(plot_dir, "epa_recovery_summary.csv"),
          row.names = FALSE)

cat(sprintf("\n=== EPA summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(samps$epa_K_plus), median(samps$epa_K_plus), sd(samps$epa_K_plus)))
cat(sprintf("  mean(alpha) = %.3f, mean(tau) = %.3f\n",
            mean(samps$epa_alpha), mean(samps$epa_tau)))
cat(sprintf("  ARI: hclust=%.3f, Dahl=%.3f, Binder=%.3f, VI=%.3f\n",
            ari_partition, ari_dahl, ari_binder, ari_vi))
cat(sprintf("  R_scale = %.3f\n", R_scale))
cat(sprintf("  split rate = %.3f, merge rate = %.3f, sigma swap rate = %.3f\n",
            diag$split_rate, diag$merge_rate, diag$sigma_swap_rate))
# =========================================================
# 25) LSIRM parameter coverage (95% credible interval)
# =========================================================
# For each true parameter, check whether it falls within the 95% credible
# interval of the corresponding posterior trace.  Returns mean coverage
# (fraction in CI) per parameter family.  Uses Procrustes-aligned positions
# so 'a' / 'b' coverage is meaningful.
compute_cov_vec <- function(samps_mat, true_vec) {
  if (is.null(samps_mat) || length(true_vec) == 0) return(NA_real_)
  P <- ncol(samps_mat)
  in_ci <- logical(P)
  for (j in seq_len(P)) {
    ci <- as.numeric(quantile(samps_mat[, j], c(0.025, 0.975), na.rm = TRUE))
    in_ci[j] <- (true_vec[j] >= ci[1]) & (true_vec[j] <= ci[2])
  }
  mean(in_ci)
}
compute_cov_3d <- function(samps_arr, true_mat) {
  if (is.null(samps_arr) || length(true_mat) == 0) return(NA_real_)
  P_ <- dim(samps_arr)[2]; d_ <- dim(samps_arr)[3]
  in_ci <- logical(P_ * d_); cnt <- 0
  for (j in seq_len(P_)) for (k in seq_len(d_)) {
    cnt <- cnt + 1
    ci <- as.numeric(quantile(samps_arr[, j, k], c(0.025, 0.975), na.rm = TRUE))
    in_ci[cnt] <- (true_mat[j, k] >= ci[1]) & (true_mat[j, k] <= ci[2])
  }
  mean(in_ci)
}
cov_scalar <- function(s, true_val, transform = identity) {
  if (is.null(s) || length(s) == 0) return(NA_real_)
  ci <- as.numeric(quantile(transform(s), c(0.025, 0.975), na.rm = TRUE))
  as.numeric((true_val >= ci[1]) & (true_val <= ci[2]))
}
b4_mat_for_cov <- if (!is.null(samps$beta4)) {
  matrix(samps$beta4, nrow = dim(samps$beta4)[1])
} else NULL
b4_true_vec_for_cov <- as.vector(beta4_true)

cov <- list(
  alpha1   = compute_cov_vec(samps$alpha1, alpha1_true),
  alpha2   = compute_cov_vec(samps$alpha2, alpha2_true),
  alpha3   = compute_cov_vec(samps$alpha3, alpha3_true),
  alpha4   = compute_cov_vec(samps$alpha4, alpha4_true),
  beta1    = compute_cov_vec(samps$beta1,  beta1_true),
  beta2    = compute_cov_vec(samps$beta2,  beta2_true),
  beta3    = compute_cov_vec(samps$beta3,  beta3_true),
  beta4_thr = compute_cov_vec(b4_mat_for_cov, b4_true_vec_for_cov),
  a        = compute_cov_3d(samps$a,  A_true),
  b1       = compute_cov_3d(samps$b1, B1_true),
  b2       = compute_cov_3d(samps$b2, B2_true),
  b3       = compute_cov_3d(samps$b3, B3_true),
  b4       = compute_cov_3d(samps$b4, B4_true),
  gamma1   = cov_scalar(samps$log_gamma1, gamma_true, transform = exp),
  gamma2   = cov_scalar(samps$log_gamma2, gamma_true, transform = exp),
  gamma3   = cov_scalar(samps$log_gamma3, gamma_true, transform = exp),
  gamma4   = cov_scalar(samps$log_gamma4, gamma_true, transform = exp),
  sigma0_sq = cov_scalar(samps$sigma0_sq, sigma0_sq_true)
)
cov_df <- data.frame(parameter = names(cov),
                     coverage95 = vapply(cov, function(v)
                       if (is.null(v) || length(v) == 0 || all(is.na(v)))
                         NA_real_ else as.numeric(v), numeric(1)))
write.csv(cov_df, file.path(plot_dir, "lsirm_coverage.csv"),
          row.names = FALSE)

# Aggregate metrics by family
cov_lsirm_overall <- mean(cov_df$coverage95, na.rm = TRUE)
cov_alpha_avg <- mean(c(cov$alpha1, cov$alpha2, cov$alpha3, cov$alpha4),
                      na.rm = TRUE)
cov_beta_avg  <- mean(c(cov$beta1, cov$beta2, cov$beta3, cov$beta4_thr),
                      na.rm = TRUE)
cov_pos_avg   <- mean(c(cov$a, cov$b1, cov$b2, cov$b3, cov$b4),
                      na.rm = TRUE)
cov_gamma_avg <- mean(c(cov$gamma1, cov$gamma2, cov$gamma3, cov$gamma4),
                      na.rm = TRUE)

cat("\n=== LSIRM coverage (95% CI) ===\n")
print(cov_df)
cat(sprintf("  Overall: %.3f, alpha-fam: %.3f, beta-fam: %.3f, pos-fam: %.3f, gamma: %.3f\n",
            cov_lsirm_overall, cov_alpha_avg, cov_beta_avg,
            cov_pos_avg, cov_gamma_avg))

# =========================================================
# 26) result_row.csv : single-row machine-readable summary
# =========================================================
# Used by the sensitivity-analysis aggregator to build the combined
# Markdown table without re-loading the full posterior.
co_within_block <- {
  ord_true <- order(true_item_cluster)
  Cb <- co_cluster[ord_true, ord_true]
  bnd <- cumsum(table(true_item_cluster[ord_true]))
  starts <- c(1, bnd + 1); ends <- c(bnd, nrow(Cb) + 1)
  starts <- starts[1:K_true]; ends <- ends[1:K_true]
  v <- numeric(0)
  for (k in seq_len(K_true)) {
    bl <- Cb[starts[k]:ends[k], starts[k]:ends[k]]
    v <- c(v, bl[upper.tri(bl)])
  }
  mean(v)
}
co_between_block <- {
  ord_true <- order(true_item_cluster)
  Cb <- co_cluster[ord_true, ord_true]
  ut <- upper.tri(Cb)
  same_true <- outer(true_item_cluster[ord_true],
                     true_item_cluster[ord_true], "==")
  mean(Cb[ut & !same_true])
}

result_row <- data.frame(
  label                = exp_label,
  alpha_fix            = alpha_fix,
  tau_fix              = tau_fix,
  update_alpha         = update_alpha,
  update_tau           = update_tau,
  b_epa_coupling       = b_epa_coupling,
  n_split_merge        = n_split_merge,
  n_split_merge_R      = n_split_merge_R,
  n_iter               = common_mcmc$n_iter,
  burnin               = common_mcmc$burnin,
  thin                 = common_mcmc$thin,
  mean_K_plus          = mean(samps$epa_K_plus),
  median_K_plus        = median(samps$epa_K_plus),
  sd_K_plus            = sd(samps$epa_K_plus),
  min_K_plus           = min(samps$epa_K_plus),
  max_K_plus           = max(samps$epa_K_plus),
  K_true               = K_true,
  ari_hclust           = ari_partition,
  ari_dahl             = ari_dahl,
  ari_binder           = ari_binder,
  ari_vi               = ari_vi,
  ari_mode             = ari_mode,
  hclust_K_plus        = median_K_plus,
  dahl_K_plus          = dahl_K_plus,
  binder_K_plus        = binder_K_plus,
  vi_K_plus            = vi_K_plus,
  cor_dist_all         = cor_all,
  R_scale              = R_scale,
  cov_lsirm_overall    = cov_lsirm_overall,
  cov_alpha_avg        = cov_alpha_avg,
  cov_beta_avg         = cov_beta_avg,
  cov_pos_avg          = cov_pos_avg,
  cov_gamma_avg        = cov_gamma_avg,
  cov_a                = cov$a,
  cov_b1               = cov$b1,
  cov_b2               = cov$b2,
  cov_b3               = cov$b3,
  cov_b4               = cov$b4,
  cov_sigma0_sq        = cov$sigma0_sq,
  co_within_mean       = co_within_block,
  co_between_mean      = co_between_block,
  co_contrast          = co_within_block - co_between_block,
  mean_alpha_epa       = mean(samps$epa_alpha),
  mean_tau_epa         = mean(samps$epa_tau),
  split_rate           = diag$split_rate,
  merge_rate           = diag$merge_rate,
  sigma_swap_rate      = diag$sigma_swap_rate,
  alpha_epa_rate       = diag$alpha_epa_rate,
  tau_epa_rate         = diag$tau_epa_rate
)
write.csv(result_row, file.path(plot_dir, "result_row.csv"),
          row.names = FALSE)

cat(sprintf("\n-> v12 simulation plots & artifacts saved to: %s\n", plot_dir))
cat("\n=== Done! ===\n")
