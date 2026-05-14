rm(list = ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile / source v10 wrappers
#
#    v10 model (joint LSIRM + PPCA-FMC with NIW + split-merge):
#      - Same 5-layer LSIRM block as v9.
#      - PPCA-style mixture clustering on item-level latent
#        factors eta_j.
#      - V10 changes vs v9:
#          * conjugate NIW prior on (mu_l, Sigma_l)
#          * c_j single-site update integrates out (mu_l, Sigma_l)
#            and rho (collapsed Gibbs with Student-t predictive)
#          * Jain-Neal split-merge moves on the partition
#
#    Simulation:
#      - 4-layer simulation (binary / continuous / count / ordinal),
#        identical data-generating mechanism to simulation_4_layered_v9.R
#        so the only systematic difference is the FMC sampler.
# =========================================================
data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v10.R"))   # joint LSIRM + PPCA-FMC v10
source(file.path(proj_dir, "utils.R"))

set.seed(20260501)

# =========================================================
# 1) Simulation settings  (identical to simulation_4_layered_v9.R)
# =========================================================
n  <- 150
P1 <- 30  # binary
P2 <- 30  # continuous (robust Student-t)
P3 <- 30  # count (NB)
P4 <- 30  # ordinal (GRM, K1=5 categories)
P_total <- P1 + P2 + P3 + P4

d        <- 2L
r_fac    <- 2L     # NEW: reduced from 5 to match LSIRM latent dim (d=2)
                   # Diagnostic from r_fac=5 run: eta posterior PC variance
                   # ratio was 0.85, 0.09, 0.04, 0.01, 0.01 -- effectively 1-D
                   # signal. With r_fac=2 the cluster structure should
                   # concentrate on the two informative directions.
K_star   <- 10L
e0       <- 0.1
K_true   <- 4L

# Row-centering toggle (used both for the FMC sampler and the plot dir name).
# FALSE: feed raw row log-distances to the PPCA layer, letting delta absorb
#        the respondent-mean log-distance via the regular Gibbs update.
row_center_flag <- FALSE

# NIW prior scale on within-cluster covariance Sigma_l. S0 = S0_scale * I_r.
# Tightened from 0.05 (default v10) to 0.01 after observing posterior eta
# within-cluster sd ~ 0.01-0.02.
S0_scale <- 0.01

K1       <- 5L

gamma_true     <- 1.0
sigma0_sq_true <- 1.0
kappa_true     <- 1.0
nu2_true       <- 5
nu2_fit        <- 4

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

assign_meta <- function(P, K) rep_len(seq_len(K), P)

B1_meta <- assign_meta(P1, K_true)
B2_meta <- assign_meta(P2, K_true)
B3_meta <- assign_meta(P3, K_true)
B4_meta <- assign_meta(P4, K_true)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

centers_resp <- centers_meta * 0.7   # was 1.5; scaled to match prior sd=1 (per-coord var ~ 0.49+0.25=0.74)
sd_cluster_resp <- 0.5               # was 0.80

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
# 8) Plot dir + true position visualisation
# =========================================================
plot_dir <- file.path(
  proj_dir, "plot",
  sprintf("simulation_4_layered_v10_rfac%d_%s_S0_%g",
          r_fac,
          if (row_center_flag) "rowctr" else "norowctr",
          S0_scale)
)
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

plot_positions_v10 <- function(A, B_list, B_meta_list, A_resp_cl,
                               title = "TRUE latent positions (4-layer, v10)",
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
plot_positions_v10(A_true,
                   list(B1_true, B2_true, B3_true, B4_true),
                   list(B1_meta, B2_meta, B3_meta, B4_meta),
                   resp_cl)
dev.off()

# =========================================================
# 9) FMC PCA-smart initialisation
# =========================================================
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

eta_init_pca <- init_eta_via_pca(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, r_fac)
km <- kmeans(eta_init_pca, centers = K_star, nstart = 25, iter.max = 50)

fmc_init_smart <- list(
  rho   = rep(1 / K_star, K_star),
  eta   = eta_init_pca,
  c     = km$cluster,
  mu    = km$centers,
  Sigma = array(rep(diag(r_fac), K_star), dim = c(r_fac, r_fac, K_star)),
  Lambda = matrix(rnorm(n * r_fac, 0, 0.5), n, r_fac),
  delta = rep(0, n),
  sigma_eps_sq   = 0.1,
  sigma_delta_sq = 0.1
)
cat(sprintf("[FMC init] PCA-eta sd per dim: %s\n",
            paste(round(apply(eta_init_pca, 2, sd), 3), collapse = ", ")))
cat(sprintf("[FMC init] k-means cluster sizes: %s\n",
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

# V10 NIW hyperparameters.
# After r_fac=2 + row_center=FALSE diagnostic: posterior eta within-cluster
# sd is ~0.01-0.02, while the original S0=0.05 I_r prior implies
# sd(Sigma_l)^{1/2} ~ 0.075 per dim — i.e. the prior thinks clusters are
# ~5-10x looser than they actually are. This inflates the NIW marginal
# likelihood for "loose" clusters, suppressing splits.
# Tighten S0 to 0.01 I_r so E[Sigma_l] = 0.01 / (nu0 - r - 1) = 0.001 (per dim
# sd ~0.033), more compatible with the actual eta scale.
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

common_mcmc <- list(d = d, n_iter = 50000L, burnin = 20000L, thin = 10L)
fmc_warmup_iter <- as.integer(common_mcmc$burnin / 4)
n_split_merge   <- 5L     # number of split-merge proposals per sweep

# =========================================================
# 11) Run v10 joint MCMC
# =========================================================
cat(sprintf(
  "\n========== simulation_4_layered_v10 (n=%d, P=%d/%d/%d/%d, K_true=%d, K*=%d, M_SM=%d, row_center=%s, S0_scale=%g) ==========\n",
  n, P1, P2, P3, P4, K_true, K_star, n_split_merge, row_center_flag, S0_scale))

result <- lsirm_fmc_v10_cpp(
  Y_bin   = Y_bin,
  Y_con   = Y_con,
  Y_cnt   = Y_cnt,
  Y_ord1  = Y_ord1,
  Y_ord2  = Y_ord2,
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
  fmc_init      = fmc_init_smart,
  save_lambda_full = FALSE,
  save_delta_full  = FALSE,
  save_eta_full    = TRUE,
  compute_co_cluster_online = TRUE,
  fmc_warmup    = fmc_warmup_iter,
  n_split_merge = n_split_merge,
  row_center    = row_center_flag,
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

sm <- result$fmc_split_merge
cat("\n-- Split-merge --\n")
cat(sprintf("  split: %d / %d  (rate %.3f)\n",
            sm$split_accepts, sm$split_attempts, sm$split_rate))
cat(sprintf("  merge: %d / %d  (rate %.3f)\n",
            sm$merge_accepts, sm$merge_attempts, sm$merge_rate))

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
plot_positions_v10(A_true,
                   list(B1_true, B2_true, B3_true, B4_true),
                   list(B1_meta, B2_meta, B3_meta, B4_meta),
                   resp_cl, title = "TRUE positions")
plot_positions_v10(A_hat,
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
# 19) FMC traceplots
# =========================================================
n_save <- nrow(samps$fmc_rho)

pdf(file.path(plot_dir, "fmc_trace_K_plus.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(samps$fmc_K_plus, main = "Occupied cluster count K_+",
        ylab = expression(K["+"]^{(s)}))
abline(h = c(mean(samps$fmc_K_plus), K_true),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2))
legend("topright", legend = c("posterior mean", "K_true"),
       col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

pdf(file.path(plot_dir, "fmc_trace_rho.pdf"), width = 9, height = 8)
par(mfrow = c(ceiling(K_star / 2), 2), mar = c(3, 3, 2, 1))
for (l in 1:K_star) {
  x <- samps$fmc_rho[, l]
  ts.plot(x, main = sprintf("rho[%d]", l), ylab = "")
  abline(h = c(mean(x), quantile(x, c(.025, .975))),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
}
dev.off()

pdf(file.path(plot_dir, "fmc_trace_mu.pdf"), width = 12, height = 10)
par(mfrow = c(K_star, r_fac), mar = c(2.5, 2.5, 1.5, 1))
for (l in 1:K_star) for (jj in 1:r_fac) {
  x <- samps$fmc_mu[l, jj, ]
  ts.plot(x, main = sprintf("mu[l=%d, d=%d]", l, jj), ylab = "")
  abline(h = mean(x), col = "darkgreen", lwd = 2)
}
dev.off()

pdf(file.path(plot_dir, "fmc_trace_logdetSigma.pdf"), width = 9, height = 8)
par(mfrow = c(ceiling(K_star / 2), 2), mar = c(3, 3, 2, 1))
for (l in 1:K_star) {
  ld <- vapply(seq_len(n_save), function(s) {
    ev <- eigen(samps$fmc_Sigma[, , l, s], symmetric = TRUE, only.values = TRUE)$values
    ev[ev < 1e-12] <- 1e-12
    sum(log(ev))
  }, numeric(1))
  ts.plot(ld, main = sprintf("log|Sigma_l| (l=%d)", l), ylab = "")
  abline(h = mean(ld), col = "darkgreen", lwd = 2)
}
dev.off()

n_l_trace <- t(apply(samps$fmc_c, 1, function(v) tabulate(v, nbins = K_star)))
pdf(file.path(plot_dir, "fmc_trace_cluster_sizes.pdf"), width = 10, height = 5)
par(mar = c(4, 4, 3, 1))
matplot(n_l_trace, type = "l", lty = 1,
        col = expand_pal(K_star, cluster_pal),
        xlab = "saved iter", ylab = expression(n[l]^{(s)}),
        main = "Item cluster sizes (c)")
legend("topright", legend = paste0("l=", 1:K_star),
       col = expand_pal(K_star, cluster_pal), lty = 1, bty = "n", cex = 0.75)
dev.off()

pdf(file.path(plot_dir, "fmc_trace_sigma_eps_sq.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(as.numeric(samps$fmc_sigma_eps_sq),
        main = "PPCA shared noise variance sigma_eps_sq",
        ylab = expression(sigma[epsilon]^2))
abline(h = c(mean(samps$fmc_sigma_eps_sq),
             quantile(samps$fmc_sigma_eps_sq, c(0.025, 0.975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
dev.off()

pdf(file.path(plot_dir, "fmc_trace_sigma_delta_sq.pdf"), width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
plot_trace_scalar(samps$fmc_sigma_delta_sq, main = "FMC sigma_delta_sq")
dev.off()

# =========================================================
# 20) FMC clustering recovery
# =========================================================
co_cluster <- samps$fmc_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

mode_label <- function(v)
  as.integer(names(sort(table(v), decreasing = TRUE))[1])
item_cluster_mode <- apply(samps$fmc_c, 2, mode_label)

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

ari_partition <- adj_rand_index(final_partition,   true_item_cluster)
ari_mode      <- adj_rand_index(item_cluster_mode, true_item_cluster)

cat("\n=== Clustering recovery ===\n")
cat(sprintf("  median K_+ : %.0f   (true K = %d)\n", median_K_plus, K_true))
cat(sprintf("  ARI(final partition vs true): %.3f\n", ari_partition))
cat(sprintf("  ARI(mode label   vs true):     %.3f  (subject to label switching)\n",
            ari_mode))
cat("\nCross-tab: final_partition vs true_item_cluster\n")
print(table(final_partition, true_item_cluster))
cat("\nCross-tab: item_cluster_mode vs true_item_cluster\n")
print(table(item_cluster_mode, true_item_cluster))

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

eta_pm <- samps$fmc_eta_postmean
pdf(file.path(plot_dir, "fmc_eta_postmean.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
pal_true <- expand_pal(K_true,  cluster_pal)
pal_fmc  <- expand_pal(K_star,  cluster_pal)
d_show   <- min(2, r_fac)
plot(eta_pm[, 1], eta_pm[, 2], pch = 19, cex = 1.3,
     col = pal_true[true_item_cluster],
     xlab = "eta dim 1", ylab = "eta dim 2",
     main = "eta posterior mean (coloured by TRUE cluster)")
text(eta_pm[, 1], eta_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
plot(eta_pm[, 1], eta_pm[, 2], pch = 19, cex = 1.3,
     col = pal_fmc[item_cluster_mode],
     xlab = "eta dim 1", ylab = "eta dim 2",
     main = "eta posterior mean (coloured by FMC mode)")
text(eta_pm[, 1], eta_pm[, 2], labels = item_names_full, pos = 4, cex = 0.55)
dev.off()

# =========================================================
# 21) Biplot
# =========================================================
B_hat_pm <- rbind(B1_hat, B2_hat, B3_hat, B4_hat)
rownames(B_hat_pm) <- item_names_full

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
           sprintf("Biplot — items coloured by FMC partition (ARI=%.3f)", ari_partition),
           file.path(plot_dir, "biplot_fmc_partition.pdf"))
biplot_pdf(A_hat, B_hat_pm, true_item_cluster, item_names_full,
           "Biplot — items coloured by TRUE meta-cluster",
           file.path(plot_dir, "biplot_true_cluster.pdf"))

# =========================================================
# 22) CSV / RDS outputs
# =========================================================
saveRDS(result,
        file.path(plot_dir,
                  sprintf("v10_simulation_result_rfac%d_%s_S0_%g.rds",
                          r_fac,
                          if (row_center_flag) "rowctr" else "norowctr",
                          S0_scale)))

write.csv(data.frame(item             = item_names_full,
                     true_cluster     = true_item_cluster,
                     fmc_mode_cluster = item_cluster_mode,
                     fmc_partition    = final_partition),
          file.path(plot_dir, "fmc_item_clusters.csv"), row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(plot_dir, "fmc_co_cluster.csv"))

write.csv(data.frame(
            mean   = mean(samps$fmc_K_plus),
            median = median(samps$fmc_K_plus),
            sd     = sd(samps$fmc_K_plus),
            min    = min(samps$fmc_K_plus),
            max    = max(samps$fmc_K_plus),
            K_star = K_star,
            K_true = K_true,
            n_save = n_save,
            ari_partition = ari_partition,
            ari_mode      = ari_mode,
            cor_dist_all  = cor_all,
            split_attempts = sm$split_attempts,
            split_accepts  = sm$split_accepts,
            split_rate     = sm$split_rate,
            merge_attempts = sm$merge_attempts,
            merge_accepts  = sm$merge_accepts,
            merge_rate     = sm$merge_rate
          ),
          file.path(plot_dir, "fmc_recovery_summary.csv"),
          row.names = FALSE)

cat(sprintf("\n=== FMC summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(samps$fmc_K_plus), median(samps$fmc_K_plus), sd(samps$fmc_K_plus)))
cat(sprintf("  ARI(partition vs true) = %.3f\n", ari_partition))
cat(sprintf("  split rate = %.3f, merge rate = %.3f\n",
            sm$split_rate, sm$merge_rate))
cat(sprintf("\n-> v10 simulation plots & artifacts saved to: %s\n", plot_dir))
cat("\n=== Done! ===\n")
