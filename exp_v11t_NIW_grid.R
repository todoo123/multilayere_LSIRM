rm(list = ls())

library(dplyr)
library(purrr)
library(cluster)
library(Rcpp)

################################################################################
# exp_v11t_NIW_grid.R — 0509
#
# Grid search over the v11t NIW prior (S0, nu0, kappa0) on MIDUS data.
# Goal: find a setting that (a) keeps trace multimodality manageable and
# (b) avoids origin-collapse of item positions, by EXPANDING the within-
# and between-cluster variance from the current stab baseline.
#
# Grid (3 x 3 x 3 = 27 cells):
#   S0_init_scale in {1.0, 2.0, 3.0}     -> E[Sigma_l] in {0.11, 0.22, 0.33} per dim
#   nu0           in {12,  8,  6}        -> 12 is current; 8, 6 give looser Sigma_l
#   kappa0        in {1.0, 0.5, 0.1}     -> mu_l prior sd in {0.33, 0.47, 1.05}
#
# Constants:
#   K_star = 6, e0 = 0.05  (stabilization defaults)
#   nu_t = 3, nu2 = 4      (v11t lock-in defaults)
#   n_split_merge = 100
#   n_iter = 15000, burnin = 5000, thin = 5  (preview length, ~2.5 min/cell)
#
# Per-cell outputs (under plot/exp_v11t_NIW_grid/cell_<S0>_<nu0>_<kap>/):
#   - trace_K_plus.pdf
#   - biplot_hclust.pdf
#   - co_cluster_true_order.pdf
#   - samps_compact.rds (key samples for user inspection)
#   - cell_summary.txt (text dump of key metrics)
#
# Master outputs (under plot/exp_v11t_NIW_grid/):
#   - exp_v11t_NIW_grid_summary.csv (27 rows)
#   - exp_v11t_NIW_grid_silhouette_heatmap.pdf
#   - exp_v11t_NIW_grid_Kplus_heatmap.pdf
#   - exp_v11t_NIW_grid_split_merge.pdf
#   - exp_v11t_NIW_grid_REPORT.txt
################################################################################

data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
setwd(data_dir)

################################################################################
# 0-1. Data load (mirror v11t MIDUS pipeline)
################################################################################
cat("\n====== Wave 2 preprocess ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_v4.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all

cat("\n====== Refresher 1 preprocess ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), local = env_r1)
lsirm_all_r1 <- env_r1$lsirm_all

combine_lsirm <- function(l_w2, l_r1) {
  rbind_mat <- function(m1, m2) {
    if (ncol(m1) == 0 && ncol(m2) == 0) return(m1)
    colnames(m2) <- colnames(m1)
    rbind(m1, m2)
  }
  list(
    Y_bin  = rbind_mat(l_w2$Y_bin,  l_r1$Y_bin),
    Y_cnt  = rbind_mat(l_w2$Y_cnt,  l_r1$Y_cnt),
    Y_ord1 = rbind_mat(l_w2$Y_ord1, l_r1$Y_ord1),
    Y_ord2 = rbind_mat(l_w2$Y_ord2, l_r1$Y_ord2),
    Y_con  = rbind_mat(l_w2$Y_con,  l_r1$Y_con),
    col_bin  = l_w2$col_bin,  col_cnt  = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1)

# preprocess scripts may setwd() to refresher folder; restore data_dir
# so the wrapper can sourceCpp(file.path(getwd(), "my_LSIRM_FMC_v11t.cpp")).
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v11t.R"))

mode_label <- function(v) as.integer(names(sort(table(v), decreasing = TRUE))[1])

################################################################################
# 0-2. Subset (case1_all - inflammation + cognition + ord1)
################################################################################
n_all <- nrow(lsirm_all$Y_con)
E  <- matrix(0L, nrow = n_all, ncol = 0)

Y_con_full  <- scale(lsirm_all$Y_con)
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
con_idx <- which(colnames(Y_con_full) %in% inflammation_vars)
Y_con  <- Y_con_full[, con_idx, drop = FALSE]
col_con <- colnames(Y_con_full)[con_idx]

Y_bin   <- lsirm_all$Y_bin
col_bin <- lsirm_all$col_bin
Y_cnt_full  <- lsirm_all$Y_cnt
cnt_cognition_vars <- c(
  "wl_immediate_omit", "wl_immediate_repetition", "wl_immediate_intrusion",
  "wl_delayed_omit",   "wl_delayed_repetition",   "wl_delayed_intrusion",
  "catflu_repetition", "catflu_intrusion",
  "numseries_incorrect", "backcount_error",
  "sgst_normal_incorrect", "sgst_reverse_incorrect",
  "sgst_mixed_nonswitch_incorrect", "sgst_mixed_switch_incorrect",
  "sgst_mixed_all_incorrect"
)
cnt_idx <- which(colnames(Y_cnt_full) %in% cnt_cognition_vars)
Y_cnt   <- Y_cnt_full[, cnt_idx, drop = FALSE]
col_cnt <- colnames(Y_cnt_full)[cnt_idx]

Y_ord1   <- lsirm_all$Y_ord1
col_ord1 <- colnames(Y_ord1)
Y_ord2   <- E
col_ord2 <- character(0)

P_total  <- ncol(Y_bin) + ncol(Y_con) + ncol(Y_cnt) + ncol(Y_ord1)
item_names_full <- c(col_bin, col_con, col_cnt, col_ord1)

cat(sprintf("\n=== MIDUS data: n=%d, P_total=%d (bin %d / con %d / cnt %d / ord1 %d) ===\n",
            n_all, P_total, ncol(Y_bin), ncol(Y_con), ncol(Y_cnt), ncol(Y_ord1)))

################################################################################
# 0-3. Common settings (proposal SDs from v11t lock-in)
################################################################################
d              <- 2L
K_star         <- 6L
e0             <- 0.05
nu_t           <- 3
nu2            <- 4
n_split_merge  <- 100L

common_lsirm_hyper <- list(
  a_sigma = 1, b_sigma = 1,
  a_tau1 = 1, b_tau1 = 1, a_tau2 = 1, b_tau2 = 1, a_tau3 = 1, b_tau3 = 1,
  a_sigma0 = 1, b_sigma0 = 1,
  mu_log_gamma1 = 0, sd_log_gamma1 = 0.4,
  mu_log_gamma2 = 0, sd_log_gamma2 = 0.4,
  mu_log_gamma3 = 0, sd_log_gamma3 = 0.4,
  mu_log_gamma4 = 0, sd_log_gamma4 = 0.4,
  mu_log_gamma5 = 0, sd_log_gamma5 = 0.4,
  mu_log_kappa = 0, sd_log_kappa = 0.1,
  mu_beta4 = 0, sd_beta4 = 2,
  mu_beta5 = 0, sd_beta5 = 2
)

common_lsirm_prop_sd <- list(
  alpha1 = 0.78, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.60, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.15, log_gamma3 = 0.15,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.5,
  beta1 = 0.50, beta2 = 0.18, beta3 = 0.20, beta4 = 0.50, beta5 = 0.30,
  b1 = 0.30, b2 = 0.27, b3 = 0.35, b4 = 0.27, b5 = 0.50,
  log_kappa = 0.30
)

common_mcmc <- list(d = d, n_iter = 15000L, burnin = 5000L, thin = 5L)

################################################################################
# 0-4. Smart FMC init (PCA + kmeans)
################################################################################
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
set.seed(0502)
km <- kmeans(b_init_proxy, centers = K_star, nstart = 25, iter.max = 50)

fmc_init_smart <- list(
  rho   = rep(1 / K_star, K_star),
  c     = km$cluster,
  mu    = km$centers,
  Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star))
)

################################################################################
# 0-5. ARI helper
################################################################################
adj_rand_index <- function(a, b) {
  tab <- table(a, b); n_  <- sum(tab); if (n_ < 2) return(NA_real_)
  sc <- sum(choose(rowSums(tab), 2)); sk <- sum(choose(colSums(tab), 2))
  st <- sum(choose(tab, 2))
  exp_idx <- sc * sk / choose(n_, 2)
  max_idx <- (sc + sk) / 2
  if (max_idx == exp_idx) return(1)
  (st - exp_idx) / (max_idx - exp_idx)
}

dahl_partition_fun <- function(c_samples, C_post) {
  S <- nrow(c_samples)
  loss <- numeric(S)
  for (s in seq_len(S)) {
    Cs <- outer(c_samples[s, ], c_samples[s, ], FUN = "==") + 0
    loss[s] <- sum((Cs - C_post)^2)
  }
  s_star <- which.min(loss)
  cl_raw <- as.integer(c_samples[s_star, ])
  cl <- as.integer(factor(cl_raw, levels = unique(cl_raw)))
  list(partition = cl, K_plus = length(unique(cl)))
}

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal)
  if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]

################################################################################
# 1. Grid definition + output dir setup
################################################################################
S0_grid     <- c(1.0, 2.0, 3.0)
nu0_grid    <- c(12L, 8L, 6L)
kappa0_grid <- c(1.0, 0.5, 0.1)

grid <- expand.grid(
  S0     = S0_grid,
  nu0    = nu0_grid,
  kappa0 = kappa0_grid,
  KEEP.OUT.ATTRS = FALSE
)
n_cells <- nrow(grid)
cat(sprintf("\n=== Grid: %d cells (S0 x nu0 x kappa0 = %dx%dx%d) ===\n",
            n_cells, length(S0_grid), length(nu0_grid), length(kappa0_grid)))

grid_root <- file.path(proj_dir, "plot", "exp_v11t_NIW_grid")
if (!dir.exists(grid_root)) dir.create(grid_root, recursive = TRUE)

################################################################################
# 2. Grid loop
################################################################################
summary_rows <- vector("list", n_cells)
t0_grid <- Sys.time()

for (cell_idx in seq_len(n_cells)) {
  S0_i     <- grid$S0[cell_idx]
  nu0_i    <- grid$nu0[cell_idx]
  kappa0_i <- grid$kappa0[cell_idx]
  cell_tag <- sprintf("S0_%g_nu0_%d_kap_%g", S0_i, nu0_i, kappa0_i)

  cat(sprintf(
    "\n========== [%d/%d] %s   (E[Sigma_l]=%g per dim, mu_l prior sd=%g) ==========\n",
    cell_idx, n_cells, cell_tag,
    S0_i / (nu0_i - d - 1),
    sqrt(S0_i / (kappa0_i * (nu0_i - d - 1)))))

  per_dir <- file.path(grid_root, paste0("cell_", cell_tag))
  if (!dir.exists(per_dir)) dir.create(per_dir, recursive = TRUE)

  fmc_hyper <- list(
    e0     = e0,
    m0     = rep(0, d),
    kappa0 = kappa0_i,
    nu0    = nu0_i,
    S0     = S0_i * diag(d),
    nu_S0  = NA
  )

  t0 <- Sys.time()
  result <- lsirm_fmc_v11t_cpp(
    Y_bin   = Y_bin, Y_con   = round(Y_con, 1), Y_cnt = Y_cnt,
    Y_ord1  = Y_ord1, Y_ord2 = Y_ord2,
    K_star  = K_star, e0 = e0,
    d       = common_mcmc$d,
    n_iter  = common_mcmc$n_iter,
    burnin  = common_mcmc$burnin,
    thin    = common_mcmc$thin,
    nu2     = nu2,
    lsirm_hyper   = common_lsirm_hyper,
    fmc_hyper     = fmc_hyper,
    lsirm_prop_sd = common_lsirm_prop_sd,
    lsirm_init    = NULL,
    fmc_init      = fmc_init_smart,
    compute_co_cluster_online = TRUE,
    fmc_warmup    = max(500L, as.integer(common_mcmc$burnin / 4)),
    n_split_merge = n_split_merge,
    nu_t          = nu_t,
    verbose       = FALSE,
    fix_gamma     = FALSE
  )
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")

  # ---------- Per-cell metrics ----------
  co_cluster <- result$fmc_co_cluster
  rownames(co_cluster) <- colnames(co_cluster) <- item_names_full
  K_plus_chain <- result$fmc_K_plus
  median_K_plus <- max(2, round(median(K_plus_chain)))
  hc_co <- hclust(as.dist(1 - co_cluster), method = "average")
  hclust_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

  dahl <- dahl_partition_fun(result$fmc_c, co_cluster)
  ari_dahl_vs_hclust <- adj_rand_index(dahl$partition, hclust_partition)

  # b posterior mean for silhouette
  A_hat_pm <- apply(result$a, c(2, 3), mean)
  b_arrs <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
  B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
    if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
      return(matrix(0, 0, d))
    apply(arr, c(2, 3), mean)
  }))
  rownames(B_hat_pm) <- item_names_full

  K_p0 <- length(unique(hclust_partition))
  sil_mean <- NA_real_
  if (K_p0 >= 2 && K_p0 < length(hclust_partition)) {
    sil_obj <- cluster::silhouette(hclust_partition, dist(B_hat_pm))
    sil_mean <- mean(sil_obj[, "sil_width"])
  }

  cm_od <- co_cluster[upper.tri(co_cluster)]
  within_pair_n <- sum(choose(table(hclust_partition), 2))
  within_pair_frac <- within_pair_n / choose(P_total, 2)
  pr_high <- mean(cm_od > 0.8)
  pr_amb  <- mean(cm_od >= 0.3 & cm_od <= 0.7)
  within_pair_ratio <- if (within_pair_frac > 0) pr_high / within_pair_frac else NA_real_

  # PEAR (subset for speed)
  set.seed(123)
  n_pear <- 200L
  S_chain <- nrow(result$fmc_c)
  ii <- sample.int(S_chain, n_pear, replace = TRUE)
  jj <- sample.int(S_chain, n_pear, replace = TRUE)
  ok <- ii != jj
  pear_vals <- mapply(function(i, j) {
    adj_rand_index(result$fmc_c[i, ], result$fmc_c[j, ])
  }, ii[ok], jj[ok])
  pear_mean <- mean(pear_vals, na.rm = TRUE)

  # b-postmean origin distance: median sqrt(b1^2 + b2^2) — proxy for geometry compression
  b_origin_dist <- median(sqrt(rowSums(B_hat_pm^2)))

  # Trace multimodality proxy: for 5 random b items, count posterior modes via density valleys (approx)
  # Simpler: report sd of b posterior across saved iters, averaged across items + dims.
  if (!is.null(result$b1) && dim(result$b1)[2] > 0) {
    b1_sd_per_item <- apply(result$b1, c(2, 3), sd)
    b1_sd_mean <- mean(b1_sd_per_item)
  } else { b1_sd_mean <- NA_real_ }

  sm <- result$fmc_split_merge

  summary_rows[[cell_idx]] <- data.frame(
    cell_idx          = cell_idx,
    S0                = S0_i,
    nu0               = nu0_i,
    kappa0            = kappa0_i,
    E_Sigma_l         = S0_i / (nu0_i - d - 1),
    mu_prior_sd       = sqrt(S0_i / (kappa0_i * (nu0_i - d - 1))),
    elapsed_sec       = round(elapsed, 1),
    K_plus_mean       = mean(K_plus_chain),
    K_plus_median     = median(K_plus_chain),
    K_plus_sd         = sd(K_plus_chain),
    K_plus_min        = min(K_plus_chain),
    K_plus_max        = max(K_plus_chain),
    hclust_K          = K_p0,
    dahl_K            = dahl$K_plus,
    ari_dahl_vs_hcl   = ari_dahl_vs_hclust,
    sil_mean          = sil_mean,
    within_pair_ratio = within_pair_ratio,
    pr_co_ambiguous   = pr_amb,
    pear_mean         = pear_mean,
    split_rate        = sm$split_rate,
    merge_rate        = sm$merge_rate,
    b_origin_dist     = b_origin_dist,
    b1_trace_sd       = b1_sd_mean
  )

  # ---------- Per-cell artifacts ----------
  pdf(file.path(per_dir, "trace_K_plus.pdf"), width = 9, height = 5)
  par(mar = c(4, 4, 3, 1))
  ts.plot(K_plus_chain,
          main = sprintf("[%s] K_+ trace (mean=%.2f, sd=%.2f)",
                         cell_tag, mean(K_plus_chain), sd(K_plus_chain)),
          ylab = "K_+")
  abline(h = mean(K_plus_chain), col = "darkgreen", lwd = 2)
  dev.off()

  pdf(file.path(per_dir, "biplot_hclust.pdf"), width = 10, height = 8)
  pal_use <- expand_pal(K_p0, cluster_pal)
  par(mar = c(4, 4, 3, 1))
  xr <- range(A_hat_pm[, 1], B_hat_pm[, 1])
  yr <- range(A_hat_pm[, 2], B_hat_pm[, 2])
  plot(A_hat_pm[, 1], A_hat_pm[, 2], pch = 21,
       bg = adjustcolor("gray60", alpha.f = 0.30), col = "gray40", cex = 0.7,
       xlab = "Dim1", ylab = "Dim2",
       main = sprintf("[%s] Biplot (hclust K=%d, sil=%.3f, ARI(Dahl)=%.3f)",
                      cell_tag, K_p0, sil_mean, ari_dahl_vs_hclust),
       xlim = xr + c(-1, 1) * 0.1 * diff(xr),
       ylim = yr + c(-1, 1) * 0.1 * diff(yr))
  points(B_hat_pm[, 1], B_hat_pm[, 2], pch = 22,
         bg = pal_use[hclust_partition], col = "black", cex = 1.7)
  text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full,
       pos = 4, cex = 0.55)
  dev.off()

  pdf(file.path(per_dir, "co_cluster_hclust_order.pdf"), width = 9, height = 8)
  ord_hc <- hc_co$order
  par(mar = c(7, 7, 3, 1))
  image(seq_len(P_total), seq_len(P_total), co_cluster[ord_hc, ord_hc],
        col  = colorRampPalette(c("white", "steelblue"))(50),
        xlab = "", ylab = "", axes = FALSE,
        main = sprintf("[%s] Co-cluster (hclust order)  K=%d",
                       cell_tag, K_p0))
  axis(1, at = seq_len(P_total), labels = item_names_full[ord_hc],
       las = 2, cex.axis = 0.55)
  axis(2, at = seq_len(P_total), labels = item_names_full[ord_hc],
       las = 2, cex.axis = 0.55)
  box()
  dev.off()

  saveRDS(list(
            cell_tag        = cell_tag,
            S0              = S0_i, nu0 = nu0_i, kappa0 = kappa0_i,
            K_plus_chain    = K_plus_chain,
            fmc_c           = result$fmc_c,
            fmc_co_cluster  = result$fmc_co_cluster,
            fmc_K_plus      = result$fmc_K_plus,
            hclust_partition = hclust_partition,
            dahl_partition   = dahl$partition,
            A_hat_pm         = A_hat_pm,
            B_hat_pm         = B_hat_pm,
            sm               = sm,
            elapsed_sec      = elapsed
          ), file = file.path(per_dir, "samps_compact.rds"))

  # Per-cell text summary (plain dump for quick inspection)
  sink(file.path(per_dir, "cell_summary.txt"))
  cat(sprintf("Cell: %s\n", cell_tag))
  cat(sprintf("  S0=%g  nu0=%d  kappa0=%g\n", S0_i, nu0_i, kappa0_i))
  cat(sprintf("  E[Sigma_l] per dim = %.4f\n", S0_i / (nu0_i - d - 1)))
  cat(sprintf("  mu_l prior sd      = %.4f\n",
              sqrt(S0_i / (kappa0_i * (nu0_i - d - 1)))))
  cat(sprintf("  K_+ mean=%.2f  median=%.0f  sd=%.2f  range=[%d, %d]\n",
              mean(K_plus_chain), median(K_plus_chain), sd(K_plus_chain),
              min(K_plus_chain), max(K_plus_chain)))
  cat(sprintf("  hclust K=%d  Dahl K=%d  ARI(Dahl,hclust)=%.3f\n",
              K_p0, dahl$K_plus, ari_dahl_vs_hclust))
  cat(sprintf("  silhouette (b-based) = %.3f\n", sil_mean))
  cat(sprintf("  within-pair ratio = %.3f, pr_co_ambiguous = %.3f, PEAR = %.3f\n",
              within_pair_ratio, pr_amb, pear_mean))
  cat(sprintf("  split rate = %.3f, merge rate = %.3f\n",
              sm$split_rate, sm$merge_rate))
  cat(sprintf("  b origin distance (median) = %.3f\n", b_origin_dist))
  cat(sprintf("  b1 trace SD (avg per item, dim) = %.4f\n", b1_sd_mean))
  cat(sprintf("  elapsed = %.1f sec\n", elapsed))
  sink()

  cat(sprintf("  -> elapsed %.1f sec, K=%d (Dahl %d), sil=%.3f, sm=(%.3f/%.3f)\n",
              elapsed, K_p0, dahl$K_plus, sil_mean, sm$split_rate, sm$merge_rate))
  rm(result)
  invisible(gc(verbose = FALSE))
}

t1_grid <- Sys.time()
grid_elapsed <- as.numeric(t1_grid - t0_grid, units = "mins")
cat(sprintf("\n=== Grid completed in %.1f minutes ===\n", grid_elapsed))

################################################################################
# 3. Master summary CSV + plots
################################################################################
sw_summary <- do.call(rbind, summary_rows)
write.csv(sw_summary,
          file.path(grid_root, "exp_v11t_NIW_grid_summary.csv"),
          row.names = FALSE)

# ---- Heatmaps: silhouette + K_+ + sm + origin_dist (S0 x nu0, faceted by kappa0) ----
make_heatmap_grid <- function(metric_col, metric_name, file, sweep_dir = grid_root) {
  pdf(file.path(sweep_dir, file), width = 12, height = 4)
  par(mfrow = c(1, length(kappa0_grid)), mar = c(4, 4, 3, 1))
  for (k0 in kappa0_grid) {
    sub <- sw_summary[sw_summary$kappa0 == k0, ]
    Z <- matrix(NA, length(S0_grid), length(nu0_grid),
                dimnames = list(S0_grid, nu0_grid))
    for (i in seq_along(S0_grid))
      for (j in seq_along(nu0_grid)) {
        sel <- sub$S0 == S0_grid[i] & sub$nu0 == nu0_grid[j]
        if (any(sel)) Z[i, j] <- sub[sel, metric_col]
      }
    image(seq_along(S0_grid), seq_along(nu0_grid), Z,
          col  = colorRampPalette(c("white", "steelblue", "darkblue"))(50),
          xlab = "S0_init_scale", ylab = "nu0", axes = FALSE,
          main = sprintf("%s  |  kappa0=%g", metric_name, k0))
    axis(1, at = seq_along(S0_grid), labels = S0_grid)
    axis(2, at = seq_along(nu0_grid), labels = nu0_grid)
    for (i in seq_along(S0_grid))
      for (j in seq_along(nu0_grid))
        if (!is.na(Z[i, j]))
          text(i, j, sprintf("%.3f", Z[i, j]), cex = 0.8)
    box()
  }
  dev.off()
}

make_heatmap_grid("sil_mean",         "Silhouette (b-based)",
                  "exp_v11t_NIW_grid_silhouette_heatmap.pdf")
make_heatmap_grid("K_plus_mean",      "mean K_+",
                  "exp_v11t_NIW_grid_Kplus_heatmap.pdf")
make_heatmap_grid("split_rate",       "split rate",
                  "exp_v11t_NIW_grid_split_rate_heatmap.pdf")
make_heatmap_grid("merge_rate",       "merge rate",
                  "exp_v11t_NIW_grid_merge_rate_heatmap.pdf")
make_heatmap_grid("b_origin_dist",    "median b origin distance",
                  "exp_v11t_NIW_grid_b_origin_dist_heatmap.pdf")
make_heatmap_grid("pear_mean",        "PEAR (posterior expected ARI)",
                  "exp_v11t_NIW_grid_pear_heatmap.pdf")

################################################################################
# 4. Plain-text REPORT for quick inspection
################################################################################
sink(file.path(grid_root, "exp_v11t_NIW_grid_REPORT.txt"))
cat("===== v11t NIW grid search report =====\n\n")
cat(sprintf("Data: MIDUS case1_all  (n=%d, P_total=%d)\n", n_all, P_total))
cat(sprintf("Constants: K_star=%d, e0=%g, nu_t=%d, nu2=%d, n_split_merge=%d\n",
            K_star, e0, nu_t, nu2, n_split_merge))
cat(sprintf("MCMC: %d iter, %d burnin, thin %d (n_save=%d/cell)\n",
            common_mcmc$n_iter, common_mcmc$burnin, common_mcmc$thin,
            (common_mcmc$n_iter - common_mcmc$burnin) / common_mcmc$thin))
cat(sprintf("Grid: S0 in {%s} x nu0 in {%s} x kappa0 in {%s}  (n_cells=%d)\n",
            paste(S0_grid, collapse=","),
            paste(nu0_grid, collapse=","),
            paste(kappa0_grid, collapse=","),
            n_cells))
cat(sprintf("Total elapsed: %.1f minutes\n\n", grid_elapsed))
cat("---- Full summary (sorted by silhouette desc) ----\n")
print(sw_summary[order(-sw_summary$sil_mean), ], row.names = FALSE)

cat("\n---- Best by silhouette ----\n")
best_sil <- sw_summary[which.max(sw_summary$sil_mean), ]
cat(sprintf("S0=%g, nu0=%d, kappa0=%g  ->  sil=%.3f, K=%d (Dahl %d), sm=(%.3f/%.3f), b_origin=%.3f\n",
            best_sil$S0, best_sil$nu0, best_sil$kappa0,
            best_sil$sil_mean, best_sil$hclust_K, best_sil$dahl_K,
            best_sil$split_rate, best_sil$merge_rate, best_sil$b_origin_dist))

cat("\n---- Best by PEAR (posterior partition concentration) ----\n")
best_pear <- sw_summary[which.max(sw_summary$pear_mean), ]
cat(sprintf("S0=%g, nu0=%d, kappa0=%g  ->  PEAR=%.3f, sil=%.3f, K=%d, sm=(%.3f/%.3f)\n",
            best_pear$S0, best_pear$nu0, best_pear$kappa0,
            best_pear$pear_mean, best_pear$sil_mean,
            best_pear$hclust_K, best_pear$split_rate, best_pear$merge_rate))

cat("\n---- Largest b origin distance (least geometry compression) ----\n")
best_geo <- sw_summary[which.max(sw_summary$b_origin_dist), ]
cat(sprintf("S0=%g, nu0=%d, kappa0=%g  ->  b_origin=%.3f, sil=%.3f, K=%d\n",
            best_geo$S0, best_geo$nu0, best_geo$kappa0,
            best_geo$b_origin_dist, best_geo$sil_mean, best_geo$hclust_K))
sink()

cat(sprintf("\n========== Grid sweep complete ==========\n"))
cat(sprintf("Master summary CSV: %s\n",
            file.path(grid_root, "exp_v11t_NIW_grid_summary.csv")))
cat(sprintf("Master REPORT:      %s\n",
            file.path(grid_root, "exp_v11t_NIW_grid_REPORT.txt")))
cat(sprintf("Heatmaps:           %s/exp_v11t_NIW_grid_*_heatmap.pdf\n", grid_root))
cat(sprintf("Per-cell:           %s/cell_<S0>_<nu0>_<kap>/\n", grid_root))
cat("  Per-cell artifacts: trace_K_plus.pdf, biplot_hclust.pdf,\n")
cat("                      co_cluster_hclust_order.pdf, samps_compact.rds, cell_summary.txt\n")
