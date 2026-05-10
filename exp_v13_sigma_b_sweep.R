rm(list = ls())

library(Rcpp)
library(vegan)

# =========================================================
# exp_v13_sigma_b_sweep.R
#
# Sweep over fixed sigma_b^2 values to find the regime where
# v13 partition recovery works best.  Data is identical to
# simulation_4_layered_v13.R (asymmetric A1+B2 geometry,
# K_true=4, P_total=120).
#
# For each sigma_b^2 in the grid we:
#   1. Run 20k iter / 8k burnin / thin 5  with sigma_b_fixed = TRUE
#      (no IG update -- sigma_b^2 is a regularization knob)
#   2. Save: ARI (hclust/Dahl/Binder/VI), K_+ stats,
#      split/merge rates, distance recovery cor.
#   3. One row per value into exp_v13_sigma_b_sweep_summary.csv
#
# Plot directories per value:
#   plot/exp_v13_sigma_b_sweep/sb_<value>/
# =========================================================

data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v13.R"))
source(file.path(proj_dir, "utils.R"))

set.seed(20260501)

# =========================================================
# 1) Simulation settings (mirror simulation_4_layered_v13.R)
# =========================================================
n  <- 150
P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30
P_total <- P1 + P2 + P3 + P4

d        <- 2L
K_star   <- 10L
e0       <- 0.1
K_true   <- 4L

S0_scale <- 0.434

K1       <- 5L

gamma_true     <- 1.0
sigma0_sq_true <- 1.0
kappa_true     <- 1.0
nu2_true       <- 5
nu2_fit        <- 4

# Grid of fixed sigma_b^2 values
sigma_b_sq_grid <- c(0.05, 0.10, 0.15, 0.20, 0.30)

# =========================================================
# 2) TRUE item meta-clusters  (A1 + B2: 12/9/6/3, centers (+/-0.5)^2)
# =========================================================
centers_meta <- rbind(
  c(-0.5,  0.5),
  c( 0.5,  0.5),
  c( 0.5, -0.5),
  c(-0.5, -0.5)
)

sigma_meta <- list(
  diag(c(0.15, 0.15)),
  diag(c(0.20, 0.20)),
  diag(c(0.10, 0.10)),
  diag(c(0.18, 0.18))
)

meta_sizes <- c(12L, 9L, 6L, 3L)
assign_meta <- function(sizes) rep(seq_along(sizes), times = sizes)
B1_meta <- assign_meta(meta_sizes)
B2_meta <- assign_meta(meta_sizes)
B3_meta <- assign_meta(meta_sizes)
B4_meta <- assign_meta(meta_sizes)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

centers_resp <- centers_meta * 0.7
sd_cluster_resp <- 0.5

# =========================================================
# 3) Sampling helpers
# =========================================================
sample_around_centers <- function(centers, cluster_ids, sd) {
  P  <- length(cluster_ids); d_ <- ncol(centers)
  centers[cluster_ids, , drop = FALSE] +
    matrix(rnorm(P * d_, 0, sd), P, d_)
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
  for (j in 1:P_)
    out[, j] <- sqrt(rowSums((A - matrix(B[j, ], n_, ncol(A), byrow = TRUE))^2))
  out
}
invlogit <- function(x) 1 / (1 + exp(-x))

# =========================================================
# 4) TRUE positions + observations (one shared dataset)
# =========================================================
resp_cl <- sample.int(nrow(centers_resp), n, replace = TRUE)
A_true  <- sample_around_centers(centers_resp, resp_cl, sd_cluster_resp)

B1_true <- sample_around_centers_sigma(centers_meta, B1_meta, sigma_meta)
B2_true <- sample_around_centers_sigma(centers_meta, B2_meta, sigma_meta)
B3_true <- sample_around_centers_sigma(centers_meta, B3_meta, sigma_meta)
B4_true <- sample_around_centers_sigma(centers_meta, B4_meta, sigma_meta)

true_sigma_alpha <- 1.0
true_tau_beta1 <- 1.0; true_tau_beta2 <- 1.0; true_tau_beta3 <- 0.5
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
  n_ <- nrow(ETA); P_ <- ncol(ETA)
  Y <- matrix(NA_integer_, n_, P_)
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

item_names_full <- c(paste0("bin_",  seq_len(P1)),
                     paste0("con_",  seq_len(P2)),
                     paste0("cnt_",  seq_len(P3)),
                     paste0("ord1_", seq_len(P4)))

# =========================================================
# 5) FMC init  (smart: PCA-on-Y -> kmeans for c init)
# =========================================================
init_b_proxy_via_pca <- function(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, d) {
  blocks <- list()
  if (ncol(Y_bin)  > 0) blocks <- c(blocks, list(scale(Y_bin)))
  if (ncol(Y_con)  > 0) blocks <- c(blocks, list(scale(Y_con)))
  if (ncol(Y_cnt)  > 0) blocks <- c(blocks, list(scale(log1p(Y_cnt))))
  if (ncol(Y_ord1) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord1))))
  if (ncol(Y_ord2) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord2))))
  X <- do.call(cbind, blocks); X[is.na(X)] <- 0
  pc <- prcomp(t(X), center = TRUE, scale. = FALSE)
  rk <- min(d, ncol(pc$x))
  out <- matrix(0, nrow = ncol(X), ncol = d)
  out[, seq_len(rk)] <- pc$x[, seq_len(rk), drop = FALSE]
  out
}
b_init_proxy <- init_b_proxy_via_pca(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, d)
km <- kmeans(b_init_proxy, centers = K_star, nstart = 25, iter.max = 50)

# =========================================================
# 6) Common hyperparameters / proposal SDs / MCMC
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
fmc_warmup_iter <- as.integer(common_mcmc$burnin / 4)
n_split_merge   <- 100L

# =========================================================
# 7) Helper: ARI (no extra dependencies)
# =========================================================
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

# =========================================================
# 8) Sweep loop
# =========================================================
sweep_dir <- file.path(proj_dir, "plot", "exp_v13_sigma_b_sweep")
if (!dir.exists(sweep_dir)) dir.create(sweep_dir, recursive = TRUE)

summary_rows <- list()

for (sb_sq in sigma_b_sq_grid) {
  cat(sprintf(
    "\n========== sigma_b^2 = %.3f (FIXED) ==========\n", sb_sq))

  fmc_init_smart <- list(
    rho   = rep(1 / K_star, K_star),
    c     = km$cluster,
    mu    = km$centers,
    Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star)),
    sigma_b_sq = sb_sq
  )

  common_fmc_hyper <- list(
    e0     = e0,
    m0     = rep(0, d),
    kappa0 = 1.0,
    nu0    = d + 10,
    S0     = S0_scale * diag(d),
    sigma_b_isotropic = TRUE,
    sigma_b_fixed     = TRUE,   # NEW: skip IG update, treat as hyperparameter
    a_b = 3.0, b_b = 0.05       # ignored when fixed=TRUE, but still required
  )

  result <- lsirm_fmc_v13_cpp(
    Y_bin   = Y_bin, Y_con   = Y_con, Y_cnt   = Y_cnt,
    Y_ord1  = Y_ord1, Y_ord2 = Y_ord2,
    K_star  = K_star, e0 = e0,
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
    compute_co_cluster_online = TRUE,
    fmc_warmup    = fmc_warmup_iter,
    n_split_merge = n_split_merge,
    verbose       = TRUE,
    fix_gamma     = FALSE,
    procrustes_target = list(a = A_true,
                             b1 = B1_true, b2 = B2_true,
                             b3 = B3_true, b4 = B4_true)
  )

  samps <- result
  co_cluster <- samps$fmc_co_cluster

  median_K_plus <- max(2, round(median(samps$fmc_K_plus)))
  hc_co <- hclust(as.dist(1 - co_cluster), method = "average")
  hclust_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))
  ari_hclust <- adj_rand_index(hclust_partition, true_item_cluster)

  # Dahl partition
  c_samples <- samps$fmc_c
  S <- nrow(c_samples)
  loss <- numeric(S)
  for (s in seq_len(S)) {
    Cs <- outer(c_samples[s, ], c_samples[s, ], FUN = "==") + 0
    loss[s] <- sum((Cs - co_cluster)^2)
  }
  s_dahl <- which.min(loss)
  dahl_partition <- as.integer(factor(c_samples[s_dahl, ],
                                      levels = unique(c_samples[s_dahl, ])))
  dahl_K_plus <- length(unique(dahl_partition))
  ari_dahl <- adj_rand_index(dahl_partition, true_item_cluster)

  # Distance recovery
  A_hat  <- apply(samps$a,  c(2, 3), mean)
  B1_hat <- apply(samps$b1, c(2, 3), mean)
  B2_hat <- apply(samps$b2, c(2, 3), mean)
  B3_hat <- apply(samps$b3, c(2, 3), mean)
  B4_hat <- apply(samps$b4, c(2, 3), mean)
  gamma_post <- c(mean(exp(samps$log_gamma1)),
                  mean(exp(samps$log_gamma2)),
                  mean(exp(samps$log_gamma3)),
                  mean(exp(samps$log_gamma4)))
  D_hat_all <- cbind(dist_mat(A_hat, B1_hat) * gamma_post[1],
                     dist_mat(A_hat, B2_hat) * gamma_post[2],
                     dist_mat(A_hat, B3_hat) * gamma_post[3],
                     dist_mat(A_hat, B4_hat) * gamma_post[4])
  D_true_all <- cbind(D1_true, D2_true, D3_true, D4_true) * gamma_true
  cor_dist_all <- cor(as.vector(D_true_all), as.vector(D_hat_all))

  sm <- result$fmc_split_merge

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    sigma_b_sq    = sb_sq,
    mean_K_plus   = mean(samps$fmc_K_plus),
    median_K_plus = median(samps$fmc_K_plus),
    sd_K_plus     = sd(samps$fmc_K_plus),
    hclust_K_plus = median_K_plus,
    dahl_K_plus   = dahl_K_plus,
    ari_hclust    = ari_hclust,
    ari_dahl      = ari_dahl,
    cor_dist_all  = cor_dist_all,
    split_rate    = sm$split_rate,
    merge_rate    = sm$merge_rate,
    n_save        = nrow(samps$fmc_c)
  )

  # Per-value diagnostic outputs
  per_dir <- file.path(sweep_dir, sprintf("sb_%.3f", sb_sq))
  if (!dir.exists(per_dir)) dir.create(per_dir, recursive = TRUE)

  # K_+ trace
  pdf(file.path(per_dir, "trace_K_plus.pdf"), width = 9, height = 5)
  par(mar = c(4, 4, 3, 1))
  ts.plot(samps$fmc_K_plus,
          main = sprintf("K_+ trace, sigma_b^2 = %.3f (mean=%.2f, sd=%.2f)",
                         sb_sq, mean(samps$fmc_K_plus), sd(samps$fmc_K_plus)),
          ylab = "K_+")
  abline(h = c(mean(samps$fmc_K_plus), K_true),
         col = c("darkgreen", "red"), lwd = 2, lty = c(1, 2))
  dev.off()

  # Co-cluster heatmap (true ordering, with cluster boundaries)
  pdf(file.path(per_dir, "co_cluster_true_order.pdf"), width = 9, height = 8)
  ord_true <- order(true_item_cluster)
  par(mar = c(7, 7, 3, 1))
  image(seq_len(P_total), seq_len(P_total), co_cluster[ord_true, ord_true],
        col  = colorRampPalette(c("white", "steelblue"))(50),
        xlab = "", ylab = "", axes = FALSE,
        main = sprintf("Co-cluster (true-order)  sigma_b^2=%.3f  ARI(hclust)=%.3f",
                       sb_sq, ari_hclust))
  bnd <- cumsum(table(true_item_cluster[ord_true]))
  abline(v = bnd + 0.5, col = "red", lty = 2)
  abline(h = bnd + 0.5, col = "red", lty = 2)
  box()
  dev.off()

  # Biplot: items coloured by Dahl partition vs true
  B_hat_pm <- rbind(B1_hat, B2_hat, B3_hat, B4_hat)
  pal_use  <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                "#999999", "#66C2A5")
  pdf(file.path(per_dir, "biplot_dahl_vs_true.pdf"), width = 14, height = 7)
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  for (lab_set in list(list(lab = true_item_cluster, ttl = "TRUE cluster"),
                       list(lab = dahl_partition,
                            ttl = sprintf("Dahl K_+=%d, ARI=%.3f",
                                          dahl_K_plus, ari_dahl)))) {
    K_lab <- max(lab_set$lab)
    cols  <- if (K_lab > length(pal_use))
               colorRampPalette(pal_use)(K_lab)[lab_set$lab]
             else pal_use[lab_set$lab]
    plot(A_hat[, 1], A_hat[, 2], pch = 21,
         bg = adjustcolor("gray60", alpha.f = 0.30),
         col = "gray40", cex = 0.7,
         xlab = "Dim1", ylab = "Dim2",
         main = sprintf("sigma_b^2=%.3f -- %s", sb_sq, lab_set$ttl),
         xlim = range(A_hat[, 1], B_hat_pm[, 1]) + c(-1, 1) * 0.05,
         ylim = range(A_hat[, 2], B_hat_pm[, 2]) + c(-1, 1) * 0.05)
    points(B_hat_pm[, 1], B_hat_pm[, 2], pch = 22,
           bg = cols, col = "black", cex = 1.7)
  }
  dev.off()

  # Cross-tab CSV (Dahl vs true)
  ct <- table(dahl_partition, true_item_cluster)
  write.csv(as.data.frame.matrix(ct),
            file.path(per_dir, "crosstab_dahl_vs_true.csv"))

  # Per-item assignment CSV (item, true cluster, Dahl, hclust)
  write.csv(data.frame(item             = item_names_full,
                       true_cluster     = true_item_cluster,
                       dahl_partition   = dahl_partition,
                       hclust_partition = hclust_partition),
            file.path(per_dir, "item_assignments.csv"), row.names = FALSE)

  # Compact RDS of MCMC samples (key arrays only, for user inspection)
  saveRDS(list(
            sigma_b_sq_value   = sb_sq,
            fmc_K_plus         = samps$fmc_K_plus,
            fmc_c              = samps$fmc_c,
            fmc_co_cluster     = samps$fmc_co_cluster,
            fmc_sigma_b_sq     = samps$fmc_sigma_b_sq,
            fmc_split_merge    = samps$fmc_split_merge,
            ari_hclust         = ari_hclust,
            ari_dahl           = ari_dahl,
            dahl_partition     = dahl_partition,
            hclust_partition   = hclust_partition,
            cor_dist_all       = cor_dist_all,
            A_hat              = A_hat,
            B_hat_pm           = B_hat_pm,
            true_item_cluster  = true_item_cluster
          ),
          file = file.path(per_dir, "samps_compact.rds"))

  cat(sprintf("  -> per-value artifacts saved to %s\n", per_dir))
}

# =========================================================
# 9) Summary table
# =========================================================
sw_summary <- do.call(rbind, summary_rows)
print(round(sw_summary, 4))

write.csv(sw_summary, file.path(sweep_dir, "exp_v13_sigma_b_sweep_summary.csv"),
          row.names = FALSE)

# =========================================================
# 10) Summary plot: ARI vs sigma_b^2
# =========================================================
pdf(file.path(sweep_dir, "ari_vs_sigma_b_sq.pdf"), width = 10, height = 6)
par(mar = c(4, 4, 3, 1), mfrow = c(1, 2))
plot(sw_summary$sigma_b_sq, sw_summary$ari_hclust,
     type = "b", pch = 19, col = "steelblue", lwd = 2,
     ylim = range(c(0, sw_summary$ari_hclust, sw_summary$ari_dahl), na.rm = TRUE),
     xlab = expression(sigma[b]^2),
     ylab = "ARI",
     main = "ARI vs sigma_b^2")
lines(sw_summary$sigma_b_sq, sw_summary$ari_dahl,
      type = "b", pch = 17, col = "darkorange", lwd = 2)
abline(h = c(0, 1), col = "gray60", lty = 3)
legend("bottomright", legend = c("hclust", "Dahl"),
       col = c("steelblue", "darkorange"), pch = c(19, 17), lty = 1, bty = "n")
plot(sw_summary$sigma_b_sq, sw_summary$mean_K_plus,
     type = "b", pch = 19, col = "darkgreen", lwd = 2,
     xlab = expression(sigma[b]^2),
     ylab = "mean K_+",
     main = "mean K_+ vs sigma_b^2")
abline(h = K_true, col = "red", lty = 2, lwd = 2)
legend("topright", legend = paste0("K_true=", K_true),
       col = "red", lty = 2, lwd = 2, bty = "n")
dev.off()

# Multi-panel comparison: ARI / K_+ / split-merge / distance
pdf(file.path(sweep_dir, "sweep_comparison_4panel.pdf"), width = 12, height = 9)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(sw_summary$sigma_b_sq, sw_summary$ari_hclust,
     type = "b", pch = 19, col = "steelblue", lwd = 2,
     ylim = range(c(0, sw_summary$ari_hclust, sw_summary$ari_dahl), na.rm = TRUE),
     xlab = expression(sigma[b]^2), ylab = "ARI",
     main = "Partition recovery (ARI)")
lines(sw_summary$sigma_b_sq, sw_summary$ari_dahl,
      type = "b", pch = 17, col = "darkorange", lwd = 2)
abline(h = c(0, 1), col = "gray60", lty = 3)
legend("bottomright", legend = c("hclust", "Dahl"),
       col = c("steelblue", "darkorange"), pch = c(19, 17), lty = 1, bty = "n")

plot(sw_summary$sigma_b_sq, sw_summary$mean_K_plus,
     type = "b", pch = 19, col = "darkgreen", lwd = 2,
     ylim = range(c(K_true, sw_summary$mean_K_plus,
                    sw_summary$mean_K_plus + sw_summary$sd_K_plus,
                    sw_summary$mean_K_plus - sw_summary$sd_K_plus)),
     xlab = expression(sigma[b]^2), ylab = "K_+",
     main = "K_+ posterior mean +/- sd")
arrows(sw_summary$sigma_b_sq, sw_summary$mean_K_plus - sw_summary$sd_K_plus,
       sw_summary$sigma_b_sq, sw_summary$mean_K_plus + sw_summary$sd_K_plus,
       angle = 90, code = 3, length = 0.05, col = "darkgreen")
abline(h = K_true, col = "red", lty = 2, lwd = 2)
legend("topright", legend = paste0("K_true=", K_true),
       col = "red", lty = 2, lwd = 2, bty = "n")

plot(sw_summary$sigma_b_sq, sw_summary$split_rate,
     type = "b", pch = 19, col = "purple", lwd = 2,
     ylim = c(0, max(sw_summary$split_rate, sw_summary$merge_rate)),
     xlab = expression(sigma[b]^2), ylab = "rate",
     main = "Split-merge acceptance rates")
lines(sw_summary$sigma_b_sq, sw_summary$merge_rate,
      type = "b", pch = 17, col = "brown", lwd = 2)
legend("topright", legend = c("split", "merge"),
       col = c("purple", "brown"), pch = c(19, 17), lty = 1, bty = "n")

plot(sw_summary$sigma_b_sq, sw_summary$cor_dist_all,
     type = "b", pch = 19, col = "darkblue", lwd = 2,
     ylim = c(0, 1),
     xlab = expression(sigma[b]^2), ylab = "Distance recovery cor",
     main = "LSIRM distance recovery")
abline(h = c(0.9, 1), col = c("gray60", "gray80"), lty = 3)
dev.off()

# Markdown-friendly summary text
sink(file.path(sweep_dir, "exp_v13_sigma_b_sweep_REPORT.txt"))
cat("===== v13 sigma_b^2 sweep report =====\n\n")
cat(sprintf("Data: %d items, K_true=%d, asymmetric 12/9/6/3, centers (+/-0.5)^2\n",
            P_total, K_true))
cat(sprintf("MCMC: %d iter, %d burnin, thin %d  (n_save = %d per fit)\n",
            common_mcmc$n_iter, common_mcmc$burnin, common_mcmc$thin,
            sw_summary$n_save[1]))
cat(sprintf("Sigma_b: ISOTROPIC + FIXED (no Gibbs update)\n\n"))
cat("---- Sweep summary ----\n")
print(round(sw_summary, 4))
cat("\n---- Best ARI(hclust) ----\n")
best_h <- sw_summary[which.max(sw_summary$ari_hclust), ]
cat(sprintf("sigma_b^2 = %.3f  ->  ARI(hclust)=%.3f, ARI(Dahl)=%.3f, mean_K_+=%.2f\n",
            best_h$sigma_b_sq, best_h$ari_hclust, best_h$ari_dahl, best_h$mean_K_plus))
cat("\n---- Best ARI(Dahl) ----\n")
best_d <- sw_summary[which.max(sw_summary$ari_dahl), ]
cat(sprintf("sigma_b^2 = %.3f  ->  ARI(hclust)=%.3f, ARI(Dahl)=%.3f, mean_K_+=%.2f\n",
            best_d$sigma_b_sq, best_d$ari_hclust, best_d$ari_dahl, best_d$mean_K_plus))
sink()

cat("\n========== Sweep complete ==========\n")
cat(sprintf("Summary CSV:  %s\n", file.path(sweep_dir, "exp_v13_sigma_b_sweep_summary.csv")))
cat(sprintf("Report TXT:   %s\n", file.path(sweep_dir, "exp_v13_sigma_b_sweep_REPORT.txt")))
cat(sprintf("4-panel plot: %s\n", file.path(sweep_dir, "sweep_comparison_4panel.pdf")))
cat(sprintf("ARI vs sb^2:  %s\n", file.path(sweep_dir, "ari_vs_sigma_b_sq.pdf")))
cat(sprintf("Per-value:    %s/sb_<value>/\n", sweep_dir))
cat("  - trace_K_plus.pdf, co_cluster_true_order.pdf\n")
cat("  - biplot_dahl_vs_true.pdf, crosstab_dahl_vs_true.csv\n")
cat("  - item_assignments.csv, samps_compact.rds (for interactive R inspection)\n")
