################################################################################
# exp_v11tm_marginal_b.R
#
# Compare v11tm (marginal_b = TRUE; collapsed-label b update,
# eq.(b-mh-ratio-marginal)) against the v11t_stab lock-in baseline on MIDUS.
#
# Both chains use IDENTICAL hyperparameters / proposal SDs / inits / MCMC
# settings.  Only the b-update kernel differs.
#
# Outputs:
#   - data/plot/.../<run>_result.rds   (the v11tm chain)
#   - exp_v11tm_marginal_b_summary.rds (metric comparison vs baseline)
#   - exp_v11tm_marginal_b.log         (stdout log)
################################################################################
rm(list = ls())
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(cluster)
})

t_global0 <- Sys.time()
data_dir  <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
proj_dir  <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
setwd(data_dir)

################################################################################
# 1. Data
################################################################################
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_v4.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), local = env_r1)
lsirm_all_r1 <- env_r1$lsirm_all

combine_lsirm <- function(l_w2, l_r1) {
  rbind_mat <- function(m1, m2) {
    if (ncol(m1) == 0 && ncol(m2) == 0) return(m1)
    colnames(m2) <- colnames(m1); rbind(m1, m2)
  }
  list(
    Y_bin  = rbind_mat(l_w2$Y_bin,  l_r1$Y_bin),
    Y_cnt  = rbind_mat(l_w2$Y_cnt,  l_r1$Y_cnt),
    Y_ord1 = rbind_mat(l_w2$Y_ord1, l_r1$Y_ord1),
    Y_ord2 = rbind_mat(l_w2$Y_ord2, l_r1$Y_ord2),
    Y_con  = rbind_mat(l_w2$Y_con,  l_r1$Y_con),
    col_bin  = l_w2$col_bin, col_cnt = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1)

source(file.path(data_dir, "my_LSIRM_FMC_cpp_v11tm.R"))

################################################################################
# 2. Variable subsets (identical to v11t_stab)
################################################################################
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
con_all_names <- colnames(lsirm_all$Y_con)
active_idx <- which(con_all_names %in% inflammation_vars)
Y_con_subset   <- scale(lsirm_all$Y_con)[, active_idx, drop = FALSE]
col_con_subset <- con_all_names[active_idx]

cnt_cognition_vars <- c(
  "wl_immediate_omit", "wl_immediate_repetition", "wl_immediate_intrusion",
  "wl_delayed_omit",   "wl_delayed_repetition",   "wl_delayed_intrusion",
  "catflu_repetition", "catflu_intrusion",
  "numseries_incorrect", "backcount_error",
  "sgst_normal_incorrect", "sgst_reverse_incorrect",
  "sgst_mixed_nonswitch_incorrect", "sgst_mixed_switch_incorrect",
  "sgst_mixed_all_incorrect"
)
cnt_idx <- which(colnames(lsirm_all$Y_cnt) %in% cnt_cognition_vars)
Y_cnt_subset   <- lsirm_all$Y_cnt[, cnt_idx, drop = FALSE]
col_cnt_subset <- colnames(lsirm_all$Y_cnt)[cnt_idx]

Y_bin_full   <- lsirm_all$Y_bin
Y_ord1_full  <- lsirm_all$Y_ord1
n_all        <- nrow(Y_con_subset)
E            <- matrix(0L, nrow = n_all, ncol = 0)

################################################################################
# 3. Hyperparameters (identical to v11t_stab lock-in 0509)
################################################################################
d <- 2L; K_star <- 6L; e0 <- 0.05; nu_t <- 3; nu2 <- 20
S0_init_scale <- 1
n_split_merge <- 100L

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
  mu_beta4 = 0, sd_beta4 = 2, mu_beta5 = 0, sd_beta5 = 2
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
common_fmc_hyper <- list(
  e0 = e0, m0 = rep(0, d), kappa0 = 1.0, nu0 = 12,
  S0 = S0_init_scale * diag(d), nu_S0 = NA
)
common_mcmc <- list(d = d, n_iter = 30000, burnin = 10000, thin = 5)

################################################################################
# 4. FMC init (PCA + kmeans, same seed as v11t_stab)
################################################################################
init_b_proxy_via_pca <- function(Y_bin, Y_con, Y_cnt, Y_ord1, d) {
  blocks <- list()
  if (ncol(Y_bin)  > 0) blocks <- c(blocks, list(scale(Y_bin)))
  if (ncol(Y_con)  > 0) blocks <- c(blocks, list(scale(Y_con)))
  if (ncol(Y_cnt)  > 0) blocks <- c(blocks, list(scale(log1p(Y_cnt))))
  if (ncol(Y_ord1) > 0) blocks <- c(blocks, list(scale(as.matrix(Y_ord1))))
  X <- do.call(cbind, blocks)
  X[is.na(X)] <- 0
  pc <- prcomp(t(X), center = TRUE, scale. = FALSE)
  rk <- min(d, ncol(pc$x))
  out <- matrix(0, nrow = ncol(X), ncol = d)
  out[, seq_len(rk)] <- pc$x[, seq_len(rk), drop = FALSE]
  out
}
b_init_proxy <- init_b_proxy_via_pca(Y_bin_full, Y_con_subset, Y_cnt_subset,
                                     Y_ord1_full, d)
set.seed(0502)
km <- kmeans(b_init_proxy, centers = K_star, nstart = 25, iter.max = 50)
fmc_init_smart <- list(
  rho   = rep(1 / K_star, K_star),
  c     = km$cluster,
  mu    = km$centers,
  Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star))
)

cat(sprintf("[v11tm] PCA-b-proxy sd per dim: %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))
cat(sprintf("[v11tm] kmeans cluster sizes: %s\n",
            paste(table(km$cluster), collapse = " ")))

################################################################################
# 5. Run v11tm chain (marginal_b = TRUE)
################################################################################
run_label  <- sprintf("v11tm_MARGB_d%d_K%d_e%g_S0init_%g_nu%d_kap%g_M%d_nut%g_nu2%g",
                     d, K_star, e0, S0_init_scale,
                     common_fmc_hyper$nu0, common_fmc_hyper$kappa0,
                     n_split_merge, nu_t, nu2)
plot_root <- file.path(data_dir, "plot")
case_dir  <- file.path(plot_root, paste0("case1_all_", run_label))
if (!dir.exists(case_dir)) dir.create(case_dir, recursive = TRUE)

cat(sprintf("\n========== v11tm MARGINAL-B [%s] ==========\n", run_label))
t0 <- Sys.time()
result <- lsirm_fmc_v11tm_cpp(
  Y_bin   = Y_bin_full,
  Y_con   = round(Y_con_subset, 1),
  Y_cnt   = Y_cnt_subset,
  Y_ord1  = Y_ord1_full,
  Y_ord2  = E,
  K_star  = K_star, e0 = e0,
  d       = common_mcmc$d,
  n_iter  = common_mcmc$n_iter,
  burnin  = common_mcmc$burnin,
  thin    = common_mcmc$thin,
  nu2     = nu2,
  lsirm_hyper   = common_lsirm_hyper,
  fmc_hyper     = common_fmc_hyper,
  lsirm_prop_sd = common_lsirm_prop_sd,
  lsirm_init    = NULL,
  fmc_init      = fmc_init_smart,
  compute_co_cluster_online = TRUE,
  fmc_warmup    = max(1000L, as.integer(common_mcmc$burnin / 4)),
  n_split_merge = n_split_merge,
  nu_t          = nu_t,
  marginal_b    = TRUE,
  verbose       = TRUE,
  fix_gamma     = FALSE
)
t1 <- Sys.time()
elapsed_min <- as.numeric(t1 - t0, units = "mins")
cat(sprintf("\n[v11tm] MCMC elapsed: %.1f min\n", elapsed_min))

# Persist the chain regardless of any downstream failure
saveRDS(result, file.path(case_dir, paste0("case1_all_", run_label, "_result.rds")))
cat(sprintf("[v11tm] saved RDS to %s\n", case_dir))

################################################################################
# 6. Acceptance + split-merge summary
################################################################################
acc <- result$accept
cat("\n-- Acceptance summary --\n")
cat(sprintf("  alpha1..5 mean : %.3f / %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$alpha1), mean(acc$alpha2), mean(acc$alpha3),
            mean(acc$alpha4), mean(acc$alpha5)))
cat(sprintf("  beta1..3 mean  : %.3f / %.3f / %.3f\n",
            mean(acc$beta1), mean(acc$beta2), mean(acc$beta3)))
cat(sprintf("  log_gamma1..5  : %.3f / %.3f / %.3f / %.3f / %.3f\n",
            acc$log_gamma1, acc$log_gamma2, acc$log_gamma3,
            acc$log_gamma4, acc$log_gamma5))
cat(sprintf("  a / b1..5 mean : %.3f / %.3f, %.3f, %.3f, %.3f, %.3f\n",
            mean(acc$a),  mean(acc$b1), mean(acc$b2),
            mean(acc$b3), mean(acc$b4),
            ifelse(length(acc$b5) > 0, mean(acc$b5), NA_real_)))

sm <- result$fmc_split_merge
cat(sprintf("\n-- Split-merge --\n"))
cat(sprintf("  split: %d / %d  (rate %.3f)\n",
            sm$split_accepts, sm$split_attempts, sm$split_rate))
cat(sprintf("  merge: %d / %d  (rate %.3f)\n",
            sm$merge_accepts, sm$merge_attempts, sm$merge_rate))
cat(sprintf("  net K_+ change (split_acc - merge_acc) = %d\n",
            sm$split_accepts - sm$merge_accepts))

cat(sprintf("\n-- K_+ trace --\n"))
cat(sprintf("  median = %.0f, mean = %.2f, sd = %.2f\n",
            median(result$fmc_K_plus), mean(result$fmc_K_plus),
            sd(result$fmc_K_plus)))

################################################################################
# 7. Posterior summaries — partitions and metrics
################################################################################
co_cluster <- result$fmc_co_cluster
n_save     <- nrow(result$fmc_c)
P_total    <- ncol(result$fmc_c)
item_names_full <- c(colnames(Y_bin_full), col_con_subset, col_cnt_subset,
                     colnames(Y_ord1_full))
item_names_full <- item_names_full[seq_len(P_total)]
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

# Hclust partition (P0)
median_K_plus <- max(2, round(median(result$fmc_K_plus)))
hc_co  <- hclust(as.dist(1 - co_cluster), method = "average")
P0_hc  <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

# Dahl partition
dahl_partition_fun <- function(c_samples, C_post) {
  S <- nrow(c_samples)
  loss <- numeric(S)
  for (s in seq_len(S)) {
    Cs <- outer(c_samples[s, ], c_samples[s, ], FUN = "==") + 0
    loss[s] <- sum((Cs - C_post)^2)
  }
  s_star <- which.min(loss)
  cl_raw <- as.integer(c_samples[s_star, ])
  as.integer(factor(cl_raw, levels = unique(cl_raw)))
}
P5_dahl <- dahl_partition_fun(result$fmc_c, co_cluster)

################################################################################
# 8. Geometry: pairwise b-distances vs LSIRM-distance scatter (S-curve check)
################################################################################
b_pm <- function(arr) apply(arr, c(2, 3), mean)
get_b_pm_full <- function(res) {
  parts <- list()
  if (!is.null(res$b1) && dim(res$b1)[2] > 0) parts <- c(parts, list(b_pm(res$b1)))
  if (!is.null(res$b2) && dim(res$b2)[2] > 0) parts <- c(parts, list(b_pm(res$b2)))
  if (!is.null(res$b3) && dim(res$b3)[2] > 0) parts <- c(parts, list(b_pm(res$b3)))
  if (!is.null(res$b4) && dim(res$b4)[2] > 0) parts <- c(parts, list(b_pm(res$b4)))
  if (!is.null(res$b5) && dim(res$b5)[2] > 0) parts <- c(parts, list(b_pm(res$b5)))
  do.call(rbind, parts)
}
b_pm_v11tm <- get_b_pm_full(result)
within_cluster_sd <- function(b_pm_mat, partition) {
  sds <- vapply(unique(partition), function(k) {
    idx <- which(partition == k)
    if (length(idx) <= 1) return(0)
    sd(as.numeric(dist(b_pm_mat[idx, , drop = FALSE])))
  }, numeric(1))
  weights <- as.integer(table(partition))
  sum(sds * weights) / sum(weights)
}
geom_within_sd_v11tm <- within_cluster_sd(b_pm_v11tm, P5_dahl)

################################################################################
# 9. Compare with v11t_stab baseline
################################################################################
baseline_rds <- file.path(plot_root,
  "case1_all_v11t_STAB_d2_K6_e0.05_S0init_1_nu12_kap1_M100_nut3",
  "case1_all_v11t_STAB_d2_K6_e0.05_S0init_1_nu12_kap1_M100_nut3_result.rds")

cmp <- list(
  v11tm = list(
    elapsed_min  = elapsed_min,
    K_plus_med   = median(result$fmc_K_plus),
    K_plus_mean  = mean(result$fmc_K_plus),
    split_rate   = sm$split_rate,
    merge_rate   = sm$merge_rate,
    acc_b_mean   = c(b1 = mean(acc$b1), b2 = mean(acc$b2),
                     b3 = mean(acc$b3), b4 = mean(acc$b4)),
    n_clusters_dahl  = length(unique(P5_dahl)),
    n_clusters_hc    = length(unique(P0_hc)),
    geom_within_sd   = geom_within_sd_v11tm,
    P0_hc            = P0_hc,
    P5_dahl          = P5_dahl,
    item_names       = item_names_full
  )
)

if (file.exists(baseline_rds)) {
  cat("\n-- Loading v11t_stab baseline for comparison --\n")
  base <- readRDS(baseline_rds)
  base_co <- base$fmc_co_cluster
  base_acc <- base$accept
  base_sm  <- base$fmc_split_merge

  base_K_med <- median(base$fmc_K_plus)
  base_hc    <- hclust(as.dist(1 - base_co), method = "average")
  base_P0    <- cutree(base_hc, k = min(max(2, round(base_K_med)), P_total - 1))
  base_dahl  <- dahl_partition_fun(base$fmc_c, base_co)
  base_b_pm  <- get_b_pm_full(base)
  geom_base  <- within_cluster_sd(base_b_pm, base_dahl)

  # Mean absolute difference of co-cluster matrices (after re-aligning rows
  # since both use the same item ordering this is just elementwise).
  if (all(dim(base_co) == dim(co_cluster))) {
    co_mae <- mean(abs(co_cluster - base_co))
  } else {
    co_mae <- NA_real_
  }

  # Adjusted Rand between v11tm partitions and baseline partitions
  ari <- function(x, y) {
    if (requireNamespace("mclust", quietly = TRUE)) {
      mclust::adjustedRandIndex(x, y)
    } else NA_real_
  }
  ari_dahl <- ari(P5_dahl, base_dahl)
  ari_hc   <- ari(P0_hc, base_P0)

  cmp$v11t_stab <- list(
    K_plus_med    = base_K_med,
    K_plus_mean   = mean(base$fmc_K_plus),
    split_rate    = base_sm$split_rate,
    merge_rate    = base_sm$merge_rate,
    acc_b_mean    = c(b1 = mean(base_acc$b1), b2 = mean(base_acc$b2),
                      b3 = mean(base_acc$b3), b4 = mean(base_acc$b4)),
    n_clusters_dahl = length(unique(base_dahl)),
    n_clusters_hc   = length(unique(base_P0)),
    geom_within_sd  = geom_base,
    P0_hc           = base_P0,
    P5_dahl         = base_dahl
  )
  cmp$comparison <- list(
    co_cluster_MAE = co_mae,
    ARI_Dahl       = ari_dahl,
    ARI_hclust     = ari_hc
  )
  cat(sprintf("co-cluster MAE   : %.4f\n", co_mae))
  cat(sprintf("ARI (Dahl)       : %.4f\n", ari_dahl))
  cat(sprintf("ARI (hclust avg) : %.4f\n", ari_hc))
  cat(sprintf("K_+ median (v11tm / v11t): %.0f / %.0f\n",
              cmp$v11tm$K_plus_med, base_K_med))
  cat(sprintf("Split rate (v11tm / v11t): %.3f / %.3f\n",
              cmp$v11tm$split_rate, base_sm$split_rate))
  cat(sprintf("Geom within-cluster sd (v11tm / v11t): %.4f / %.4f\n",
              geom_within_sd_v11tm, geom_base))
} else {
  cat(sprintf("\n[warn] v11t baseline RDS not found at %s\n", baseline_rds))
}

saveRDS(cmp, file.path(proj_dir, "exp_v11tm_marginal_b_summary.rds"))
cat(sprintf("\n[v11tm] saved comparison summary to %s\n",
            file.path(proj_dir, "exp_v11tm_marginal_b_summary.rds")))

t_global1 <- Sys.time()
cat(sprintf("\n[v11tm] total elapsed: %.1f min\n",
            as.numeric(t_global1 - t_global0, units = "mins")))
cat("[v11tm] DONE\n")
