rm(list = ls())

library(Rcpp)
library(vegan)

# =========================================================
# v14 simulation: byte-identical data-generating mechanism to
# simulation_4_layered_v11.R (n=150, P1=P2=P3=P4=30, K_true=4,
# centers (+/-0.5)^2, sigma_meta heterogeneous, meta sizes 12/9/6/3,
# gamma=1.0, sigma0_sq=1.0, kappa=1.0, nu2_true=5, nu2_fit=4).
#
# ONLY the FMC sampler is replaced with v14 (hierarchical mixture-of-
# mixtures + telescoping sampler; Stage 2 default: telescoping_on=TRUE,
# b_variant="B"). All other settings and downstream diagnostics mirror
# the v11 script so the two results are directly comparable.
#
# Outputs (under plot_dir):
#   true_positions.pdf
#   trace_a.pdf, trace_b1..b4.pdf
#   trace_alpha1..4.pdf, trace_beta1..3.pdf, trace_beta4_thr.pdf
#   trace_lsirm_extra.pdf, trace_sigma_alpha.pdf, trace_kappa_per_item.pdf
#   boxplot_groups.pdf
#   positions_true_vs_hat.pdf
#   distance_recovery_scatter.pdf
#   fmc_trace_K_plus.pdf, fmc_trace_eta_K.pdf, fmc_trace_b_0k.pdf,
#   fmc_trace_logdetSigma.pdf, fmc_trace_cluster_sizes.pdf
#   fmc_trace_alpha_telescoping.pdf, fmc_trace_K_telescoping.pdf
#   fmc_co_cluster_true_order.pdf, fmc_co_cluster_partition_order.pdf
#   fmc_b_postmean.pdf
#   biplot_hclust/dahl/binder/vi/true.pdf
#   fmc_item_clusters.csv, fmc_co_cluster.csv, fmc_recovery_summary.csv
# =========================================================

# ----- Path setup (auto-detect host) -----
host_user <- Sys.info()[["user"]]
proj_dir <- if (host_user == "todoo") {
  "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
} else {
  "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"
}
data_dir <- file.path(proj_dir, "data")
setwd(data_dir)

source(file.path(data_dir, "my_LSIRM_FMC_cpp_v14.R"))   # v14 joint LSIRM + MoM telescoping

# Pull plot_trace_scalar / plot_trace_vec / plot_group_box_ci from utils.R.
utils_path <- file.path(proj_dir, "utils.R")
if (!file.exists(utils_path)) utils_path <- file.path(data_dir, "utils.R")
source(utils_path)

set.seed(20260501)

# =========================================================
# 1) Simulation settings (IDENTICAL to v11)
# =========================================================
n  <- 150
P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30
P_total <- P1 + P2 + P3 + P4
d        <- 2L
K_true   <- 4L
K1       <- 5L

gamma_true     <- 1.0
sigma0_sq_true <- 1.0
kappa_true     <- 1.0
nu2_true       <- 5
nu2_fit        <- 4

centers_meta <- rbind(
  c(-0.5,  0.5),
  c( 0.5,  0.5),
  c( 0.5, -0.5),
  c(-0.5, -0.5)
)
stopifnot(nrow(centers_meta) == K_true, ncol(centers_meta) == d)

sigma_meta <- list(
  diag(c(0.15, 0.15)),
  diag(c(0.20, 0.20)),
  diag(c(0.10, 0.10)),
  diag(c(0.18, 0.18))
)
meta_sizes <- c(12L, 9L, 6L, 3L)
stopifnot(length(meta_sizes) == K_true,
          sum(meta_sizes) == P1, sum(meta_sizes) == P2,
          sum(meta_sizes) == P3, sum(meta_sizes) == P4)
assign_meta <- function(sizes) rep(seq_along(sizes), times = sizes)
B1_meta <- assign_meta(meta_sizes)
B2_meta <- assign_meta(meta_sizes)
B3_meta <- assign_meta(meta_sizes)
B4_meta <- assign_meta(meta_sizes)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

centers_resp <- centers_meta * 0.7
sd_cluster_resp <- 0.5

# =========================================================
# 2) Sampling helpers (verbatim v11)
# =========================================================
sample_around_centers <- function(centers, cluster_ids, sd) {
  P <- length(cluster_ids); d_ <- ncol(centers)
  centers[cluster_ids, , drop = FALSE] + matrix(rnorm(P * d_, 0, sd), P, d_)
}
sample_around_centers_sigma <- function(centers, cluster_ids, sigma_list) {
  P <- length(cluster_ids); d_ <- ncol(centers)
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
# 3) TRUE positions (identical to v11)
# =========================================================
resp_cl <- sample.int(nrow(centers_resp), n, replace = TRUE)
A_true  <- sample_around_centers(centers_resp, resp_cl, sd_cluster_resp)
B1_true <- sample_around_centers_sigma(centers_meta, B1_meta, sigma_meta)
B2_true <- sample_around_centers_sigma(centers_meta, B2_meta, sigma_meta)
B3_true <- sample_around_centers_sigma(centers_meta, B3_meta, sigma_meta)
B4_true <- sample_around_centers_sigma(centers_meta, B4_meta, sigma_meta)

# =========================================================
# 4) TRUE intercepts / difficulties / GRM thresholds
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
# 5) Linear predictors / observations (identical to v11)
# =========================================================
D1_true <- dist_mat(A_true, B1_true)
D2_true <- dist_mat(A_true, B2_true)
D3_true <- dist_mat(A_true, B3_true)
D4_true <- dist_mat(A_true, B4_true)

ETA1_true <- outer(alpha1_true, rep(1, P1)) - outer(rep(1, n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha2_true, rep(1, P2)) - outer(rep(1, n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha3_true, rep(1, P3)) - outer(rep(1, n), beta3_true) - gamma_true * D3_true
ETA4_true <- outer(alpha4_true, rep(1, P4))                                 - gamma_true * D4_true

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
# 6) Plot dir + true positions
# =========================================================
K_max          <- 15L
L_sub          <- 4L
alpha_init     <- 1.0
s_alpha        <- 0.8
telescoping_on <- TRUE
b_variant      <- "B"

plot_dir <- file.path(
  proj_dir, "plot",
  sprintf("simulation_4_layered_v14_d%d_Kmax%d_L%d_telesc%s_var%s",
          d, K_max, L_sub,
          ifelse(telescoping_on, "ON", "OFF"), b_variant)
)
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal) {
  if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]
}
draw_ellipse <- function(mu, Sigma, level = 0.95, col = 1, lwd = 1, lty = 2, n = 200) {
  t <- seq(0, 2 * pi, length.out = n)
  circle <- rbind(cos(t), sin(t))
  ev <- eigen(Sigma, symmetric = TRUE)
  scale <- sqrt(qchisq(level, 2))
  pts   <- t(ev$vectors %*% diag(sqrt(pmax(ev$values, 0))) %*% circle) * scale
  pts   <- sweep(pts, 2, mu, "+")
  lines(pts[, 1], pts[, 2], col = col, lwd = lwd, lty = lty)
}
plot_positions <- function(A, B_list, B_meta_list, A_resp_cl,
                           title, rng = NA, draw_true_ellipses = TRUE) {
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
plot_positions(A_true,
               list(B1_true, B2_true, B3_true, B4_true),
               list(B1_meta, B2_meta, B3_meta, B4_meta),
               resp_cl, title = "TRUE latent positions (4-layer, v14)")
dev.off()

# =========================================================
# 7) MCMC hyperparameters / proposal SDs (LSIRM block from v11)
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

common_mcmc <- list(d = d, n_iter = 20000L, burnin = 8000L, thin = 5L)

# v14 fmc_hyper: tuned to the LSIRM b coordinate scale (S_z = 0.25 * I_d,
# equivalent to expected b spread ~ +/- 0.5 per dim). This matches what
# v11 hard-codes via (kappa0 = 1, S0 = 0.434 * I, nu0 = d + 10) under the
# old single-level NIW: E[Sigma_l] = 0.434/9 ~= 0.05 * I.
#
# Under v14's FS Wishart formulation:
#   Sigma_kl^{-1} ~ W_d(c_0, C_0k),  C_0k ~ W_d(g_0, G_0).
# With c_0 = 3 and E[C_0k] = g_0 * G_0^{-1}, choosing G_0 = (c_0 / E_target)^{-1}
# anchors E[Sigma_kl^{-1}] = c_0 / E_target so E[Sigma_kl] ~ E_target.
# Target E[Sigma_kl] = 0.05 * I -> G_0 = (3 / 0.05)^{-1} * I = (1/60) * I.
S_z_target <- 0.25 * diag(d)       # implied per-dim b sd ~ 0.5
v14_fmc_hyper <- list(
  m_0      = rep(0, d),
  M_0      = 10 * S_z_target,       # between-cluster scale (variance ~ 2.5)
  B_0_diag = 0.05 * diag(S_z_target),  # within-cluster, across-l mu spread
  G_0      = (1 / 60) * diag(d),    # E[Sigma_kl] ~ 0.05 * I_d (v11-equivalent)
  g_0      = 1.0,
  c_0      = 3.0,
  nu_gig   = 10.0,
  d_0      = 1.0
)

# =========================================================
# 7b) PCA-based b proxy init (data-driven; mirrors v11/v13/MIDUS).
#     CRITICAL: with lsirm_init = NULL the cpp seeds each b layer from
#     N(0, 0.5), so init_two_level k-means runs on random noise and the
#     chain often collapses to K_+ = 1. We instead build b1..b4 from the
#     first d PCs of the response matrix (transposed so items are rows),
#     giving init_two_level a meaningful partition to start from.
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
# Rescale PCA proxy to a moderate sd (per-dim sd = 1). The raw PCA
# proxy has sd 3-4 in PCA-component units; the LSIRM b's typically
# end up in [-1, 1]. We deliberately decouple the proxy used for init
# from the hyperparameters, and pass explicit fmc_hyper below that
# matches the LSIRM-coordinate scale (v11-equivalent).
b_init_proxy <- scale(b_init_proxy, center = TRUE, scale = TRUE) * 1.0
slice_b <- function(arr, off, P_l) {
  if (P_l == 0) return(matrix(0, 0, d))
  arr[(off + 1):(off + P_l), , drop = FALSE]
}
off <- 0
b1_init <- slice_b(b_init_proxy, off, P1); off <- off + P1
b2_init <- slice_b(b_init_proxy, off, P2); off <- off + P2
b3_init <- slice_b(b_init_proxy, off, P3); off <- off + P3
b4_init <- slice_b(b_init_proxy, off, P4); off <- off + P4
b5_init <- matrix(0, 0, d)
cat(sprintf("[v14 init] PCA-b-proxy sd per dim (after rescale to 0.5): %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))

init_grm_beta <- function(P, K) {
  if (P == 0 || K <= 1) return(matrix(0, nrow = 0, ncol = 0))
  Km1 <- K - 1
  t(sapply(seq_len(P), function(j) sort(rnorm(Km1, 0, 1), decreasing = TRUE)))
}
lsirm_init_smart <- list(
  alpha1 = rnorm(n, 0, 0.1),
  alpha2 = rnorm(n, 0, 0.1),
  alpha3 = rnorm(n, 0, 0.1),
  alpha4 = rnorm(n, 0, 0.1),
  alpha5 = rnorm(n, 0, 0.1),
  beta1  = rnorm(P1, 0, 0.1),
  beta2  = rnorm(P2, 0, 0.1),
  beta3  = rnorm(P3, 0, 0.1),
  a  = matrix(rnorm(n * d, 0, 0.5), n, d),
  b1 = b1_init, b2 = b2_init, b3 = b3_init, b4 = b4_init, b5 = b5_init,
  log_gamma1 = 0, log_gamma2 = 0, log_gamma3 = 0,
  log_gamma4 = 0, log_gamma5 = 0,
  log_kappa = rep(0, P3),
  sigma_alpha1_sq = 1, sigma_alpha2_sq = 1, sigma_alpha3_sq = 1,
  sigma_alpha4_sq = 1, sigma_alpha5_sq = 1,
  tau_beta1_sq = 1, tau_beta2_sq = 1, tau_beta3_sq = 1,
  sigma0_sq = 1,
  beta4 = init_grm_beta(P4, K1),
  beta5 = init_grm_beta(0, 0)
)

cat(sprintf(
  "\n========== simulation_4_layered_v14 (n=%d, P=%d/%d/%d/%d, K_true=%d, K_max=%d, L=%d, telescoping=%s, variant=%s) ==========\n",
  n, P1, P2, P3, P4, K_true, K_max, L_sub,
  ifelse(telescoping_on, "ON", "OFF"), b_variant))

# =========================================================
# 8) Run v14 joint MCMC (Procrustes-aligned via external target)
# =========================================================
t0 <- Sys.time()
result <- lsirm_fmc_v14_cpp(
  Y_bin   = Y_bin,
  Y_con   = Y_con,
  Y_cnt   = Y_cnt,
  Y_ord1  = Y_ord1,
  Y_ord2  = Y_ord2,
  K_max   = K_max,
  L       = L_sub,
  d       = common_mcmc$d,
  n_iter  = common_mcmc$n_iter,
  burnin  = common_mcmc$burnin,
  thin    = common_mcmc$thin,
  nu2     = nu2_fit,
  alpha_const    = alpha_init,
  alpha_init     = alpha_init,
  s_alpha        = s_alpha,
  telescoping_on = telescoping_on,
  b_variant      = b_variant,
  lsirm_hyper    = common_lsirm_hyper,
  fmc_hyper      = v14_fmc_hyper,   # v11-equivalent LSIRM-coordinate scale
  lsirm_prop_sd  = common_lsirm_prop_sd,
  lsirm_init     = lsirm_init_smart,   # PCA-based b proxy -> data-driven init_two_level
  fmc_init       = NULL,               # init_two_level (k-means -> sub-kmeans)
  compute_co_cluster_online = TRUE,
  fmc_warmup     = 100L,
  verbose        = TRUE,
  fix_gamma      = TRUE,   # pin gamma=1 to break LSIRM (b, gamma) scale degeneracy.
                           # v14's MoM does not anchor b scale tightly enough to
                           # prevent the chain from drifting to (b huge, gamma tiny).
  procrustes_target = list(a  = A_true,
                           b1 = B1_true, b2 = B2_true,
                           b3 = B3_true, b4 = B4_true)
)
t1 <- Sys.time()
walltime_sec <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("[v14 sim] wallclock: %.1f s (%.2f min)\n",
            walltime_sec, walltime_sec / 60))

samps <- result
saveRDS(result, file.path(plot_dir, "result.rds"))

# =========================================================
# 9) Acceptance summary + telescoping diagnostics
# =========================================================
acc <- result$accept
cat("\n-- Acceptance --\n")
cat(sprintf("  alpha1..4 mean : %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$alpha1), mean(acc$alpha2), mean(acc$alpha3), mean(acc$alpha4)))
cat(sprintf("  beta1..3 mean  : %.3f / %.3f / %.3f\n",
            mean(acc$beta1), mean(acc$beta2), mean(acc$beta3)))
cat(sprintf("  log_gamma1..4  : %.3f / %.3f / %.3f / %.3f\n",
            acc$log_gamma1, acc$log_gamma2, acc$log_gamma3, acc$log_gamma4))
cat(sprintf("  a / b1..4 mean : %.3f / %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$a), mean(acc$b1), mean(acc$b2), mean(acc$b3), mean(acc$b4)))
cat(sprintf("  log_kappa per-item mean: %.3f\n", mean(acc$log_kappa)))

if (isTRUE(telescoping_on)) {
  cat(sprintf("\n-- v14 telescoping --\n"))
  cat(sprintf("  alpha-MH acceptance: %.3f\n", result$fmc_alpha_mh_accept_rate))
  cat(sprintf("  alpha posterior:    mean=%.3f median=%.3f range=[%.3f, %.3f]\n",
              mean(result$fmc_alpha), median(result$fmc_alpha),
              min(result$fmc_alpha), max(result$fmc_alpha)))
  cat(sprintf("  K (active slots):   mode=%d median=%d max=%d (K_max=%d)\n",
              as.integer(names(sort(table(result$fmc_K), decreasing = TRUE))[1]),
              as.integer(median(result$fmc_K)),
              as.integer(max(result$fmc_K)), K_max))
  cat(sprintf("  K hits K_max rate: %.4f\n",
              mean(result$fmc_K == K_max)))
}

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
# 10) LSIRM traceplots: positions a, b1..b4
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
# 11) LSIRM scalar / vector traces (utils.R helpers)
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
# 12) Box-CI grouping plots
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
# 13) Posterior mean LSIRM positions
# =========================================================
A_hat  <- apply(samps$a,  c(2, 3), mean)
B1_hat <- apply(samps$b1, c(2, 3), mean)
B2_hat <- apply(samps$b2, c(2, 3), mean)
B3_hat <- apply(samps$b3, c(2, 3), mean)
B4_hat <- apply(samps$b4, c(2, 3), mean)

pdf(file.path(plot_dir, "positions_true_vs_hat.pdf"), width = 14, height = 7)
par(mfrow = c(1, 2))
plot_positions(A_true,
               list(B1_true, B2_true, B3_true, B4_true),
               list(B1_meta, B2_meta, B3_meta, B4_meta),
               resp_cl, title = "TRUE positions")
plot_positions(A_hat,
               list(B1_hat, B2_hat, B3_hat, B4_hat),
               list(B1_meta, B2_meta, B3_meta, B4_meta),
               resp_cl, title = "Posterior mean positions (Procrustes-aligned)")
dev.off()

# =========================================================
# 14) Distance recovery
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
# 15) Layer-scale diagnostic R_scale (v11 sec 6)
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
# 16) v14 FMC traceplots
#     (v11 used fmc_rho/fmc_mu/fmc_Sigma; v14 has fmc_eta_K/fmc_b_0k/fmc_C_0k
#      with K_max slots and L=4 subcomponents per slot. We summarise per
#      cluster k by b_0k and the within-cluster mean log|Sigma_kl|.)
# =========================================================
n_save <- length(samps$fmc_K_plus)

pdf(file.path(plot_dir, "fmc_trace_K_plus.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(samps$fmc_K_plus, main = "Occupied cluster count K_+",
        ylab = expression(K["+"]^{(s)}))
abline(h = c(mean(samps$fmc_K_plus), K_true),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2))
legend("topright", legend = c("posterior mean", "K_true"),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

pdf(file.path(plot_dir, "fmc_trace_eta_K.pdf"), width = 9, height = 8)
n_show <- min(K_max, 10L)
par(mfrow = c(ceiling(n_show / 2), 2), mar = c(3, 3, 2, 1))
for (l in 1:n_show) {
  x <- samps$fmc_eta_K[, l]
  ts.plot(x, main = sprintf("eta_K[%d]", l), ylab = "")
  abline(h = c(mean(x), quantile(x, c(.025, .975))),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
}
dev.off()

# b_0k trace (cluster-level mean, K_max x d x n_save).
pdf(file.path(plot_dir, "fmc_trace_b_0k.pdf"), width = 12, height = 10)
n_show <- min(K_max, 10L)
par(mfrow = c(n_show, d), mar = c(2.5, 2.5, 1.5, 1))
for (k in 1:n_show) for (jj in 1:d) {
  x <- samps$fmc_b_0k[k, jj, ]
  ts.plot(x, main = sprintf("b_0k[k=%d, d=%d]", k, jj), ylab = "")
  abline(h = mean(x), col = "darkgreen", lwd = 2)
}
dev.off()

# log|Sigma_kl| trace, averaged across l (per cluster).
pdf(file.path(plot_dir, "fmc_trace_logdetSigma.pdf"), width = 9, height = 8)
n_show <- min(K_max, 10L)
par(mfrow = c(ceiling(n_show / 2), 2), mar = c(3, 3, 2, 1))
for (k in 1:n_show) {
  ld <- vapply(seq_len(n_save), function(s) {
    vals <- numeric(L_sub)
    for (l in seq_len(L_sub)) {
      idx <- (k - 1) * L_sub + l
      ev <- eigen(samps$fmc_Sigma_kl[, , idx, s], symmetric = TRUE,
                  only.values = TRUE)$values
      ev[ev < 1e-12] <- 1e-12
      vals[l] <- sum(log(ev))
    }
    mean(vals)
  }, numeric(1))
  ts.plot(ld, main = sprintf("mean log|Sigma_kl|  (k=%d, over l)", k), ylab = "")
  abline(h = mean(ld), col = "darkgreen", lwd = 2)
}
dev.off()

# Cluster size (n_l) trace from fmc_S (1-based after wrapper).
n_l_trace <- t(apply(samps$fmc_S, 1, function(v) tabulate(v, nbins = K_max)))
pdf(file.path(plot_dir, "fmc_trace_cluster_sizes.pdf"), width = 10, height = 5)
par(mar = c(4, 4, 3, 1))
matplot(n_l_trace, type = "l", lty = 1,
        col = expand_pal(K_max, cluster_pal),
        xlab = "saved iter", ylab = expression(n[k]^{(s)}),
        main = "Item cluster sizes (S)")
legend("topright", legend = paste0("k=", 1:K_max),
       col = expand_pal(K_max, cluster_pal), lty = 1, bty = "n", cex = 0.6)
dev.off()

# v14-specific: alpha + K telescoping traces.
if (isTRUE(telescoping_on)) {
  pdf(file.path(plot_dir, "fmc_trace_alpha_telescoping.pdf"), width = 9, height = 6)
  par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
  ts.plot(samps$fmc_alpha, main = "alpha (telescoping, F(6,3) prior)",
          ylab = "alpha")
  abline(h = mean(samps$fmc_alpha), col = "darkgreen", lwd = 2)
  ts.plot(log(samps$fmc_alpha), main = "log alpha", ylab = "log alpha")
  abline(h = mean(log(samps$fmc_alpha)), col = "darkgreen", lwd = 2)
  dev.off()

  pdf(file.path(plot_dir, "fmc_trace_K_telescoping.pdf"), width = 9, height = 5)
  par(mar = c(4, 4, 3, 1))
  ts.plot(samps$fmc_K, main = "K (active slots, telescoping)",
          ylab = "K", xlab = "saved iter")
  abline(h = c(mean(samps$fmc_K), K_max),
         col = c("darkgreen", "blue"), lwd = 2, lty = c(1, 3))
  legend("topright", legend = c("posterior mean", "K_max"),
         col = c("darkgreen", "blue"), lwd = 2, lty = c(1, 3), bty = "n")
  dev.off()
}

# =========================================================
# 17) FMC clustering recovery (Dahl / Binder / VI / hclust / mode)
# =========================================================
co_cluster <- samps$fmc_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

mode_label <- function(v)
  as.integer(names(sort(table(v), decreasing = TRUE))[1])
item_cluster_mode <- apply(samps$fmc_S, 2, mode_label)

median_K_plus   <- max(2, round(median(samps$fmc_K_plus)))
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
  if (max_idx == expected) return(1)
  (sum_t - expected) / (max_idx - expected)
}

dahl_partition <- function(c_samples, C_post) {
  S <- nrow(c_samples)
  loss <- numeric(S)
  for (s in seq_len(S)) {
    Cs <- outer(c_samples[s, ], c_samples[s, ], FUN = "==") + 0
    loss[s] <- sum((Cs - C_post)^2)
  }
  s_star <- which.min(loss)
  cl_raw <- as.integer(c_samples[s_star, ])
  cl <- as.integer(factor(cl_raw, levels = unique(cl_raw)))
  list(partition = cl, iter = s_star, loss = loss[s_star],
       K_plus = length(unique(cl)))
}
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

dahl <- dahl_partition(samps$fmc_S, co_cluster)
dahl_K_plus        <- dahl$K_plus
dahl_partition_vec <- dahl$partition

binder <- binder_partition(samps$fmc_S, co_cluster)
binder_K_plus        <- binder$K_plus
binder_partition_vec <- binder$partition

cat("\n[VI] Computing pairwise VI distances ...\n")
vi <- vi_partition(samps$fmc_S, vi_max_S = 500L)
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

# Co-cluster heatmaps.
pdf(file.path(plot_dir, "fmc_co_cluster_true_order.pdf"), width = 9, height = 8)
ord_true <- order(true_item_cluster)
par(mar = c(7, 7, 3, 1))
image(seq_len(P_total), seq_len(P_total), co_cluster[ord_true, ord_true],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("FMC co-clustering reordered by TRUE cluster  (ARI=%.3f)", ari_partition))
axis(1, at = seq_len(P_total), labels = item_names_full[ord_true],
     las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_true],
     las = 2, cex.axis = 0.55)
bnd <- cumsum(table(true_item_cluster[ord_true]))
abline(v = bnd + 0.5, col = "red", lty = 2)
abline(h = bnd + 0.5, col = "red", lty = 2)
box()
dev.off()

pdf(file.path(plot_dir, "fmc_co_cluster_partition_order.pdf"), width = 9, height = 8)
ord_fmc <- order(final_partition)
par(mar = c(7, 7, 3, 1))
image(seq_len(P_total), seq_len(P_total), co_cluster[ord_fmc, ord_fmc],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = "FMC co-clustering reordered by FMC partition")
axis(1, at = seq_len(P_total), labels = item_names_full[ord_fmc],
     las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_fmc],
     las = 2, cex.axis = 0.55)
box()
dev.off()

# =========================================================
# 18) b-position scatter colored by true vs mode/Dahl
# =========================================================
B_hat_pm <- rbind(B1_hat, B2_hat, B3_hat, B4_hat)
rownames(B_hat_pm) <- item_names_full

pdf(file.path(plot_dir, "fmc_b_postmean.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
pal_true <- expand_pal(K_true,  cluster_pal)
pal_fmc  <- expand_pal(K_max,   cluster_pal)
plot(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, cex = 1.3,
     col = pal_true[true_item_cluster],
     xlab = "b dim 1", ylab = "b dim 2",
     main = "b posterior mean (coloured by TRUE cluster)")
text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
plot(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, cex = 1.3,
     col = pal_fmc[item_cluster_mode],
     xlab = "b dim 1", ylab = "b dim 2",
     main = "b posterior mean (coloured by FMC mode)")
text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
dev.off()

# =========================================================
# 19) Biplots (one per partition variant)
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
# 20) CSV outputs
# =========================================================
write.csv(data.frame(item              = item_names_full,
                     true_cluster      = true_item_cluster,
                     fmc_mode_cluster  = item_cluster_mode,
                     hclust_partition  = final_partition,
                     dahl_partition    = dahl_partition_vec,
                     binder_partition  = binder_partition_vec,
                     vi_partition      = vi_partition_vec),
          file.path(plot_dir, "fmc_item_clusters.csv"), row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(plot_dir, "fmc_co_cluster.csv"))

write.csv(data.frame(
            mean   = mean(samps$fmc_K_plus),
            median = median(samps$fmc_K_plus),
            sd     = sd(samps$fmc_K_plus),
            min    = min(samps$fmc_K_plus),
            max    = max(samps$fmc_K_plus),
            K_max  = K_max,
            K_true = K_true,
            n_save = n_save,

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

            telescoping_on        = telescoping_on,
            alpha_mh_accept       = ifelse(telescoping_on,
                                            result$fmc_alpha_mh_accept_rate, NA),
            alpha_post_mean       = mean(result$fmc_alpha),
            alpha_post_median     = median(result$fmc_alpha),
            K_telesc_mode         = as.integer(names(sort(table(result$fmc_K),
                                                          decreasing = TRUE))[1]),
            K_hit_Kmax_rate       = mean(result$fmc_K == K_max),
            walltime_sec          = walltime_sec
          ),
          file.path(plot_dir, "fmc_recovery_summary.csv"),
          row.names = FALSE)

cat(sprintf("\n=== v14 FMC summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(samps$fmc_K_plus), median(samps$fmc_K_plus), sd(samps$fmc_K_plus)))
cat(sprintf("  ARI: hclust=%.3f, Dahl=%.3f, Binder=%.3f, VI=%.3f\n",
            ari_partition, ari_dahl, ari_binder, ari_vi))
cat(sprintf("  R_scale = %.3f\n", R_scale))
if (isTRUE(telescoping_on)) {
  cat(sprintf("  alpha-MH accept = %.3f, alpha mean = %.3f, K_max hit rate = %.4f\n",
              result$fmc_alpha_mh_accept_rate, mean(result$fmc_alpha),
              mean(result$fmc_K == K_max)))
}
cat(sprintf("\n-> v14 simulation plots & artifacts saved to: %s\n", plot_dir))
cat("\n=== Done! ===\n")
