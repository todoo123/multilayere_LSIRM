rm(list = ls())
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(cluster)
})

t0 <- Sys.time()
data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
setwd(data_dir)

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
    col_bin = l_w2$col_bin, col_cnt = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1)

source(file.path(data_dir, "my_LSIRM_FMC_cpp_v11tm.R"))

inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
con_all_names <- colnames(lsirm_all$Y_con)
active_idx <- which(con_all_names %in% inflammation_vars)
Y_con_subset <- scale(lsirm_all$Y_con[, active_idx, drop = FALSE])

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
Y_cnt_subset <- lsirm_all$Y_cnt[, cnt_idx, drop = FALSE]

n_all <- nrow(Y_con_subset)
E <- matrix(0L, nrow = n_all, ncol = 0)

d <- 2L; K_star <- 6L; e0 <- 0.05; nu_t <- 3; nu2 <- 20

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
  S0 = 1 * diag(d), nu_S0 = NA
)

Y_bin_full <- lsirm_all$Y_bin
Y_ord1_full <- lsirm_all$Y_ord1

set.seed(0502)
P_total_init <- ncol(Y_bin_full) + ncol(Y_con_subset) + ncol(Y_cnt_subset) + ncol(Y_ord1_full) + 0
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
b_init_proxy <- init_b_proxy_via_pca(Y_bin_full, Y_con_subset, Y_cnt_subset, Y_ord1_full, d)
km <- kmeans(b_init_proxy, centers = K_star, nstart = 25, iter.max = 50)
fmc_init_smart <- list(
  rho = rep(1/K_star, K_star),
  c   = km$cluster,
  mu  = km$centers,
  Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star))
)

cat("\n[smoke] running 200-iter v11tm chain (marginal_b=TRUE)...\n")
res <- lsirm_fmc_v11tm_cpp(
  Y_bin = Y_bin_full, Y_con = round(Y_con_subset, 1), Y_cnt = Y_cnt_subset,
  Y_ord1 = Y_ord1_full, Y_ord2 = E,
  K_star = K_star, e0 = e0, d = d,
  n_iter = 200, burnin = 50, thin = 5, nu2 = nu2,
  lsirm_hyper = common_lsirm_hyper, fmc_hyper = common_fmc_hyper,
  lsirm_prop_sd = common_lsirm_prop_sd,
  lsirm_init = NULL, fmc_init = fmc_init_smart,
  compute_co_cluster_online = TRUE, fmc_warmup = 25L,
  n_split_merge = 100L, nu_t = nu_t,
  marginal_b = TRUE,
  verbose = TRUE, fix_gamma = FALSE
)
t1 <- Sys.time()
cat(sprintf("[smoke] elapsed: %.1fs for 200 iter\n", as.numeric(t1 - t0, units="secs")))
cat(sprintf("[smoke] mean acc b1..b5: %.3f / %.3f / %.3f / %.3f / %.3f\n",
            mean(res$accept$b1), mean(res$accept$b2), mean(res$accept$b3),
            mean(res$accept$b4), mean(res$accept$b5)))
sm <- res$fmc_split_merge
cat(sprintf("[smoke] split/merge accept rates: %.3f / %.3f\n",
            sm$split_rate, sm$merge_rate))
cat(sprintf("[smoke] median K_+: %d\n", median(res$fmc_K_plus)))
cat("[smoke] OK\n")
