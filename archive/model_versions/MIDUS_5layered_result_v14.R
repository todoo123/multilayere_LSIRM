rm(list = ls())

library(dplyr)
library(purrr)
library(cluster)   # silhouette()

################################################################################
# v14: Joint multilayered LSIRM + Hierarchical MoM (mixture-of-mixtures)
#      on the global pool {z_q = b_j^(l)} with telescoping sampler.
#
# DIFFERENCES FROM v13:
#   * Drops the auxiliary r_q layer and global Sigma_b. The MoM hierarchy
#     itself (K upper clusters, L lower subcomponents per cluster) plays
#     the role v13 needed Sigma_b for.
#   * No K_star / e0 / split-merge / n_split_merge. Telescoping replaces
#     them: K is sampled from BNB(1,4,3), alpha from F(6,3), with
#     empty-cluster refresh from priors each iteration.
#   * fmc_S / fmc_I (1-based after wrapper) replace fmc_c.
#
# DEFAULTS (Stage 2 telescoping, Variant B b_j MH):
#   K_max = 30, L = 4, alpha_init = 1, s_alpha = 0.8.
#   Variant B (partial collapse over I_q) is the production default;
#   Variant A is available via b_variant = "A" but did not pass the
#   Stage 3 benchmark (ESS not improved, 4-5x wallclock).
################################################################################

################################################################################
# 0. Path setup (auto-detect host)
################################################################################
host_user <- Sys.info()[["user"]]
data_dir <- if (host_user == "todoo") {
  "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
} else {
  "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data"
}
setwd(data_dir)

################################################################################
# 0-1. Data preparation: Wave 2 + Refresher 1 (v4 preprocess) -- v13 verbatim
################################################################################
# Preprocess scripts hardcode /Users/todoo/... paths. Rewrite on the fly
# so the driver runs unchanged on any host with the joint_LSIRM tree.
joint_root <- dirname(data_dir)
source_with_path_rewrite <- function(script_path, env) {
  src <- readLines(script_path)
  src <- gsub("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM",
              joint_root, src, fixed = TRUE)
  eval(parse(text = src), envir = env)
}

cat("\n====== Wave 2 preprocess ======\n")
env_w2 <- new.env(parent = globalenv())
source_with_path_rewrite(file.path(data_dir, "MIDUS_preprocess_v4.R"), env_w2)
lsirm_all_w2 <- env_w2$lsirm_all

cat("\n====== Refresher 1 preprocess ======\n")
env_r1 <- new.env(parent = globalenv())
source_with_path_rewrite(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), env_r1)
lsirm_all_r1 <- env_r1$lsirm_all

combine_lsirm <- function(l_w2, l_r1, label = "") {
  rbind_mat <- function(m1, m2) {
    if (ncol(m1) == 0 && ncol(m2) == 0) return(m1)
    colnames(m2) <- colnames(m1)
    rbind(m1, m2)
  }
  Y_bin  <- rbind_mat(l_w2$Y_bin,  l_r1$Y_bin)
  Y_cnt  <- rbind_mat(l_w2$Y_cnt,  l_r1$Y_cnt)
  Y_ord1 <- rbind_mat(l_w2$Y_ord1, l_r1$Y_ord1)
  Y_ord2 <- rbind_mat(l_w2$Y_ord2, l_r1$Y_ord2)
  Y_con  <- rbind_mat(l_w2$Y_con,  l_r1$Y_con)
  list(
    Y_bin = Y_bin, Y_cnt = Y_cnt,
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2, Y_con = Y_con,
    row_ids = c(l_w2$row_ids, l_r1$row_ids),
    branch  = c(l_w2$branch,  l_r1$branch),
    source  = c(rep("wave2",      length(l_w2$row_ids)),
                rep("refresher1", length(l_r1$row_ids))),
    col_bin  = l_w2$col_bin,  col_cnt  = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1, label = "P1-P3-P4")

################################################################################
# 0-2. Model loading
################################################################################
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v14.R"))   # v14 telescoping MoM

has_valid <- function(x) {
  if (is.null(x)) return(FALSE)
  if (is.matrix(x) && ncol(x) == 0) return(FALSE)
  if (is.array(x) && length(dim(x)) == 3 && dim(x)[2] == 0) return(FALSE)
  if (all(is.na(x))) return(FALSE)
  TRUE
}
mode_label <- function(v) {
  as.integer(names(sort(table(v), decreasing = TRUE))[1])
}
cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal) {
  if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]
}

################################################################################
# 1. Hyperparameter / MCMC settings
################################################################################
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

# LSIRM proposal SDs (inherited from v13 MIDUS retune).
common_lsirm_prop_sd <- list(
  alpha1 = 0.78, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.60, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.15, log_gamma3 = 0.15,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.5,
  beta1 = 0.50, beta2 = 0.18, beta3 = 0.20, beta4 = 0.50, beta5 = 0.30,
  b1 = 0.30, b2 = 0.27, b3 = 0.35, b4 = 0.27, b5 = 0.50,
  log_kappa = 0.30
)

# v14 MoM settings
d              <- 2L
K_max          <- 30L           # generous upper bound; telescoping rarely needs > 10
L_sub          <- 4L            # subcomponents per cluster (variants.md default)
alpha_init     <- 1.0
s_alpha        <- 0.8           # log-alpha RWMH proposal SD
telescoping_on <- TRUE           # Stage 2 default; flip to FALSE for fixed-K Stage 1 baseline
b_variant      <- "B"            # Stage 3 benchmark recommended Variant B as production default

# MCMC schedule. v14 telescoping has lower per-iter overhead than v13
# (no split-merge), so a moderately long chain is feasible.
common_mcmc <- list(d = d, n_iter = 20000L, burnin = 8000L, thin = 5L)
nu2 <- 4

plot_root <- file.path(data_dir, "plot")
if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE)

################################################################################
# 2. Case definition (case1_all: all four heterogeneous layers)
################################################################################
n_all <- nrow(lsirm_all$Y_con)
make_empty <- function(n) matrix(0L, nrow = n, ncol = 0)
E  <- make_empty(n_all)
Y_con_full  <- scale(lsirm_all$Y_con)
Y_bin_full  <- lsirm_all$Y_bin
Y_cnt_full  <- lsirm_all$Y_cnt
Y_ord1_full <- lsirm_all$Y_ord1

cs <- list(
  name  = "case1_all",
  label = "Case 1: All (bin+con+cnt+ord)",
  Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = Y_cnt_full,
  Y_ord1 = Y_ord1_full, Y_ord2 = E,
  col_bin  = lsirm_all$col_bin,
  col_con  = lsirm_all$col_con,
  col_cnt  = lsirm_all$col_cnt,
  col_ord1 = lsirm_all$col_ord1,
  col_ord2 = character(0)
)

################################################################################
# 3. Variable subsets (inflammation + cognition focus; v13 verbatim)
################################################################################
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
con_all_names <- colnames(Y_con_full)
active_idx <- which(con_all_names %in% inflammation_vars)
Y_con_subset   <- Y_con_full[, active_idx, drop = FALSE]
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
cnt_all_names <- colnames(Y_cnt_full)
cnt_idx <- which(cnt_all_names %in% cnt_cognition_vars)
Y_cnt_subset   <- Y_cnt_full[, cnt_idx, drop = FALSE]
col_cnt_subset <- cnt_all_names[cnt_idx]

cs$Y_con   <- Y_con_subset;   cs$col_con   <- col_con_subset
cs$Y_cnt   <- Y_cnt_subset;   cs$col_cnt   <- col_cnt_subset

################################################################################
# 4. PCA-based b proxy init (data-driven, v13 verbatim)
################################################################################
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

P_total_init <- ncol(cs$Y_bin) + ncol(cs$Y_con) + ncol(cs$Y_cnt) +
                ncol(cs$Y_ord1) + ncol(cs$Y_ord2)
n_init       <- nrow(cs$Y_con)

b_init_proxy <- init_b_proxy_via_pca(cs$Y_bin, cs$Y_con, cs$Y_cnt,
                                     cs$Y_ord1, cs$Y_ord2, d)
P1 <- ncol(cs$Y_bin); P2 <- ncol(cs$Y_con); P3 <- ncol(cs$Y_cnt)
P4 <- ncol(cs$Y_ord1); P5 <- ncol(cs$Y_ord2)
slice_b <- function(arr, off, P_l) {
  if (P_l == 0) return(matrix(0, 0, d))
  arr[(off + 1):(off + P_l), , drop = FALSE]
}
off <- 0
b1_init <- slice_b(b_init_proxy, off, P1); off <- off + P1
b2_init <- slice_b(b_init_proxy, off, P2); off <- off + P2
b3_init <- slice_b(b_init_proxy, off, P3); off <- off + P3
b4_init <- slice_b(b_init_proxy, off, P4); off <- off + P4
b5_init <- slice_b(b_init_proxy, off, P5); off <- off + P5

# LSIRM init: use proxy b's + small respondent positions; everything else
# defaults safe.
set.seed(0502)
init_grm_beta <- function(P, K) {
  if (P == 0 || K <= 1) return(matrix(0, nrow = 0, ncol = 0))
  Km1 <- K - 1
  t(sapply(seq_len(P), function(j) sort(rnorm(Km1, 0, 1), decreasing = TRUE)))
}
K1_ord <- if (P4 > 0) max(cs$Y_ord1) else 2L
K2_ord <- if (P5 > 0) max(cs$Y_ord2) else 2L

lsirm_init_smart <- list(
  alpha1 = rnorm(n_init, 0, 0.1),
  alpha2 = rnorm(n_init, 0, 0.1),
  alpha3 = rnorm(n_init, 0, 0.1),
  alpha4 = rnorm(n_init, 0, 0.1),
  alpha5 = rnorm(n_init, 0, 0.1),
  beta1  = rnorm(P1, 0, 0.1),
  beta2  = rnorm(P2, 0, 0.1),
  beta3  = rnorm(P3, 0, 0.1),
  a  = matrix(rnorm(n_init * d, 0, 0.5), n_init, d),
  b1 = b1_init, b2 = b2_init, b3 = b3_init, b4 = b4_init, b5 = b5_init,
  log_gamma1 = 0, log_gamma2 = 0, log_gamma3 = 0,
  log_gamma4 = 0, log_gamma5 = 0,
  log_kappa = rep(0, P3),
  sigma_alpha1_sq = 1, sigma_alpha2_sq = 1, sigma_alpha3_sq = 1,
  sigma_alpha4_sq = 1, sigma_alpha5_sq = 1,
  tau_beta1_sq = 1, tau_beta2_sq = 1, tau_beta3_sq = 1,
  sigma0_sq = 1,
  beta4 = init_grm_beta(P4, K1_ord),
  beta5 = init_grm_beta(P5, K2_ord)
)

cat(sprintf("[v14 init] PCA-b-proxy scale: sd per dim = %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))

################################################################################
# 5. Run chain
################################################################################
run_label <- sprintf(
  "v14_MoM_d%d_Kmax%d_L%d_alphaInit%g_telesc%s_var%s",
  d, K_max, L_sub, alpha_init,
  ifelse(telescoping_on, "ON", "OFF"), b_variant
)
case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", run_label))
if (!dir.exists(case_plot_dir)) dir.create(case_plot_dir, recursive = TRUE)
prefix     <- paste0(cs$name, "_", run_label)
fmc_prefix <- paste0(prefix, "_fmc")

cat(sprintf("\n========== %s [%s] ==========\n", cs$label, run_label))
cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
            nrow(cs$Y_bin),  ncol(cs$Y_bin),
            nrow(cs$Y_con),  ncol(cs$Y_con),
            nrow(cs$Y_cnt),  ncol(cs$Y_cnt),
            nrow(cs$Y_ord1), ncol(cs$Y_ord1),
            nrow(cs$Y_ord2), ncol(cs$Y_ord2)))

t0 <- Sys.time()
result <- lsirm_fmc_v14_cpp(
  Y_bin   = cs$Y_bin,
  Y_con   = round(cs$Y_con, 1),
  Y_cnt   = cs$Y_cnt,
  Y_ord1  = cs$Y_ord1,
  Y_ord2  = cs$Y_ord2,
  K_max   = K_max,
  L       = L_sub,
  d       = common_mcmc$d,
  n_iter  = common_mcmc$n_iter,
  burnin  = common_mcmc$burnin,
  thin    = common_mcmc$thin,
  nu2     = nu2,
  alpha_const    = alpha_init,    # used only if telescoping_on = FALSE
  alpha_init     = alpha_init,
  s_alpha        = s_alpha,
  telescoping_on = telescoping_on,
  b_variant      = b_variant,
  lsirm_hyper    = common_lsirm_hyper,
  fmc_hyper      = NULL,          # use .fmc_default_hyper (data-driven)
  lsirm_prop_sd  = common_lsirm_prop_sd,
  lsirm_init     = lsirm_init_smart,
  fmc_init       = NULL,          # init_two_level (k-means -> sub-kmeans)
  compute_co_cluster_online = TRUE,
  fmc_warmup     = 100L,          # v14 init is data-driven (k-means);
                                  # no long LSIRM-only warmup needed.
  verbose        = TRUE,
  fix_gamma      = FALSE
)
t1 <- Sys.time()
walltime_sec <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("[v14 MIDUS] wallclock: %.1f s (%.2f min)\n",
            walltime_sec, walltime_sec / 60))

# Save chain first for safety BEFORE post-processing.
saveRDS(result, file.path(case_plot_dir, paste0(prefix, "_result.rds")))

################################################################################
# 6. Acceptance + telescoping diagnostics
################################################################################
acc <- result$accept
cat(sprintf("\n-- v14 acceptance --\n"))
cat(sprintf("  alpha1..5: %.3f / %.3f / %.3f / %.3f / %.3f\n",
            mean(acc$alpha1), mean(acc$alpha2), mean(acc$alpha3),
            mean(acc$alpha4), mean(acc$alpha5)))
cat(sprintf("  beta1..3 : %.3f / %.3f / %.3f\n",
            mean(acc$beta1), mean(acc$beta2), mean(acc$beta3)))
cat(sprintf("  log_gamma1..5: %.3f / %.3f / %.3f / %.3f / %.3f\n",
            acc$log_gamma1, acc$log_gamma2, acc$log_gamma3,
            acc$log_gamma4, acc$log_gamma5))
cat(sprintf("  a / b1..5: %.3f / %.3f, %.3f, %.3f, %.3f, %.3f\n",
            mean(acc$a),  mean(acc$b1), mean(acc$b2),
            mean(acc$b3), mean(acc$b4), mean(acc$b5)))
cat(sprintf("  log_kappa per-item mean: %.3f\n", mean(acc$log_kappa)))
if (isTRUE(telescoping_on)) {
  cat(sprintf("\n-- v14 telescoping --\n"))
  cat(sprintf("  alpha-MH acceptance: %.3f\n", result$fmc_alpha_mh_accept_rate))
  cat(sprintf("  alpha posterior:  mean=%.3f  median=%.3f  range=[%.3f, %.3f]\n",
              mean(result$fmc_alpha), median(result$fmc_alpha),
              min(result$fmc_alpha), max(result$fmc_alpha)))
  cat(sprintf("  K (active slots) posterior: mode=%d  median=%d  max=%d (K_max=%d)\n",
              as.integer(names(sort(table(result$fmc_K), decreasing = TRUE))[1]),
              as.integer(median(result$fmc_K)),
              as.integer(max(result$fmc_K)), K_max))
}
cat(sprintf("\n-- v14 K_+ (occupied clusters) --\n"))
Kp_tab <- table(result$fmc_K_plus)
cat(sprintf("  posterior table: %s\n",
            paste(sprintf("%s:%d", names(Kp_tab), as.integer(Kp_tab)),
                  collapse = "  ")))
Kp_mode <- as.integer(names(sort(Kp_tab, decreasing = TRUE))[1])
cat(sprintf("  mode K_+ = %d  (mean %.2f)\n",
            Kp_mode, mean(result$fmc_K_plus)))

################################################################################
# 7. Trace plots: K_+, K, alpha, gamma, sigma_b/anchor analogs
################################################################################
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_traces.pdf")),
    width = 9, height = 7)
par(mfrow = c(2, 2), mar = c(3, 4, 2, 1))
ts.plot(result$fmc_K_plus, main = "K_+ (occupied clusters)",
        ylab = "K_+", xlab = "iter (post-burn, thinned)")
abline(h = Kp_mode, col = "darkgreen", lwd = 2)
if (isTRUE(telescoping_on)) {
  ts.plot(result$fmc_K, main = "K (active slots, telescoping)",
          ylab = "K", xlab = "iter")
  ts.plot(result$fmc_alpha, main = "alpha (telescoping)",
          ylab = "alpha", xlab = "iter")
  ts.plot(log(result$fmc_alpha), main = "log alpha",
          ylab = "log alpha", xlab = "iter")
}
dev.off()

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_K_plus_hist.pdf")),
    width = 7, height = 5)
Kp_range <- range(result$fmc_K_plus)
hist(result$fmc_K_plus,
     breaks = seq(max(0, Kp_range[1] - 0.5), Kp_range[2] + 0.5, by = 1),
     main = "Posterior of K_+ (occupied cluster count)",
     xlab = "K_+", col = "lightblue", border = "white")
abline(v = Kp_mode, col = "darkgreen", lwd = 2)
dev.off()

# LSIRM scalar traces (compact set; gamma + sigma0)
pdf(file.path(case_plot_dir, paste0(prefix, "_trace_scalars.pdf")),
    width = 9, height = 7)
par(mfrow = c(3, 2), mar = c(3, 4, 2, 1))
for (lg in 1:5) {
  nm <- paste0("log_gamma", lg)
  ts.plot(result[[nm]], main = nm, ylab = "log gamma")
}
ts.plot(result$sigma0_sq, main = "sigma0_sq", ylab = "")
dev.off()

################################################################################
# 8. Co-cluster matrix + Dahl point estimate
################################################################################
P_total <- ncol(result$fmc_S)
co_cluster <- result$fmc_co_cluster
stopifnot(!is.null(co_cluster),
          all(dim(co_cluster) == c(P_total, P_total)))

# Build item names (col_bin -> col_con -> col_cnt -> col_ord1 -> col_ord2)
item_names_full <- c(cs$col_bin, cs$col_con, cs$col_cnt,
                     cs$col_ord1, cs$col_ord2)
if (length(item_names_full) == P_total) {
  rownames(co_cluster) <- item_names_full
  colnames(co_cluster) <- item_names_full
}

# Mode partition (per-item posterior mode of S_q).
item_cluster_mode <- apply(result$fmc_S, 2, mode_label)

# Dahl (2006) point estimate.
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
  list(partition = cl, iter = s_star, loss = loss[s_star],
       K_plus = length(unique(cl)))
}
dahl <- dahl_partition_fun(result$fmc_S, co_cluster)
P_dahl <- dahl$partition

cat(sprintf("\n-- v14 Dahl partition --\n"))
cat(sprintf("  iter = %d, loss = %.3f, K_+ = %d\n",
            dahl$iter, dahl$loss, dahl$K_plus))
cat(sprintf("  cluster sizes: %s\n",
            paste(as.integer(table(P_dahl)), collapse = " ")))

# Optional: mcclust / mcclust.ext partitions.
alt_partitions <- list(Dahl = P_dahl, Mode = item_cluster_mode)
if (requireNamespace("mcclust", quietly = TRUE) &&
    requireNamespace("mcclust.ext", quietly = TRUE)) {
  bin_uncon <- tryCatch(mcclust.ext::minbinder.ext(
    co_cluster, cls.draw = result$fmc_S,
    method = "all", include.lg = TRUE, include.greedy = FALSE,
    suppress.comment = TRUE
  ), error = function(e) { cat("[alt] minbinder.ext failed\n"); NULL })
  if (!is.null(bin_uncon)) {
    alt_partitions$minBinder <- as.integer(bin_uncon$cl[which.min(bin_uncon$value), ])
  }
  vi_uncon <- tryCatch(mcclust.ext::minVI(
    co_cluster, cls.draw = result$fmc_S,
    method = "all", include.greedy = FALSE, suppress.comment = TRUE
  ), error = function(e) { cat("[alt] minVI failed\n"); NULL })
  if (!is.null(vi_uncon)) {
    alt_partitions$minVI <- as.integer(vi_uncon$cl[which.min(vi_uncon$value), ])
  }
  cat(sprintf("[alt] partitions computed: %s\n",
              paste(names(alt_partitions), collapse = ", ")))
}

################################################################################
# 9. b posterior mean (LSIRM latent space)
################################################################################
b_arrs <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
  if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
    return(matrix(0, 0, common_mcmc$d))
  apply(arr, c(2, 3), mean)
}))
if (length(item_names_full) == nrow(B_hat_pm)) {
  rownames(B_hat_pm) <- item_names_full
}
A_hat_pm <- apply(result$a, c(2, 3), mean)

################################################################################
# 10. Silhouette for each candidate partition (b-based)
################################################################################
D_b <- dist(B_hat_pm)
sil_table <- list()
for (pname in names(alt_partitions)) {
  part <- alt_partitions[[pname]]
  if (length(unique(part)) < 2 || length(unique(part)) > nrow(B_hat_pm) - 1) {
    sil_table[[pname]] <- list(K = length(unique(part)),
                               sil_mean = NA_real_,
                               frac_neg = NA_real_)
    next
  }
  sil_obj <- tryCatch(cluster::silhouette(part, D_b),
                      error = function(e) NULL)
  if (is.null(sil_obj)) {
    sil_table[[pname]] <- list(K = length(unique(part)),
                               sil_mean = NA_real_,
                               frac_neg = NA_real_)
    next
  }
  sw <- sil_obj[, "sil_width"]
  sil_table[[pname]] <- list(K = length(unique(part)),
                             sil_mean = mean(sw, na.rm = TRUE),
                             frac_neg = mean(sw < 0, na.rm = TRUE))
}
cat(sprintf("\n-- v14 silhouette (b-based, partition x K x sil_mean x frac_neg) --\n"))
for (pname in names(sil_table)) {
  v <- sil_table[[pname]]
  cat(sprintf("  %-12s  K=%d  sil_mean=%.3f  frac<0=%.3f\n",
              pname, v$K, v$sil_mean, v$frac_neg))
}

################################################################################
# 11. Biplot (a + b posterior mean, colored by Dahl partition)
################################################################################
pdf(file.path(case_plot_dir, paste0(prefix, "_biplot_dahl.pdf")),
    width = 9, height = 8)
plot(A_hat_pm[, 1], A_hat_pm[, 2],
     pch = 1, col = "grey60",
     xlab = "dim 1", ylab = "dim 2",
     main = sprintf("v14 LSIRM biplot (Dahl K_+ = %d)", dahl$K_plus),
     xlim = range(c(A_hat_pm[,1], B_hat_pm[,1])) * 1.1,
     ylim = range(c(A_hat_pm[,2], B_hat_pm[,2])) * 1.1)
pal <- expand_pal(max(P_dahl), cluster_pal)
points(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, col = pal[P_dahl], cex = 1.5)
if (!is.null(rownames(B_hat_pm))) {
  text(B_hat_pm[, 1], B_hat_pm[, 2], rownames(B_hat_pm),
       pos = 3, cex = 0.6)
}
legend("topright",
       legend = paste("cluster", seq_along(unique(P_dahl))),
       col = pal[seq_along(unique(P_dahl))], pch = 19, bty = "n", cex = 0.8)
dev.off()

################################################################################
# 12. CSV outputs
################################################################################
write.csv(as.data.frame(co_cluster),
          file.path(case_plot_dir, paste0(fmc_prefix, "_co_cluster.csv")),
          row.names = TRUE)
write.csv(B_hat_pm,
          file.path(case_plot_dir, paste0(fmc_prefix, "_b_postmean.csv")),
          row.names = !is.null(rownames(B_hat_pm)))

alt_df <- data.frame(item = if (is.null(rownames(B_hat_pm))) seq_len(P_total)
                            else rownames(B_hat_pm))
for (pname in names(alt_partitions)) alt_df[[pname]] <- alt_partitions[[pname]]
write.csv(alt_df,
          file.path(case_plot_dir, paste0(fmc_prefix, "_alt_partitions.csv")),
          row.names = FALSE)

summary_df <- data.frame(
  metric = c("walltime_sec", "K_plus_mode", "K_plus_mean", "Dahl_K_plus",
             "alpha_mean_post", "alpha_MH_accept",
             "sil_mean_Dahl"),
  value = c(walltime_sec, Kp_mode, mean(result$fmc_K_plus),
            dahl$K_plus,
            ifelse(telescoping_on, mean(result$fmc_alpha), alpha_init),
            ifelse(telescoping_on, result$fmc_alpha_mh_accept_rate, NA_real_),
            sil_table$Dahl$sil_mean)
)
write.csv(summary_df,
          file.path(case_plot_dir, paste0(prefix, "_summary.csv")),
          row.names = FALSE)

cat(sprintf("\n[v14 MIDUS] outputs saved under %s\n", case_plot_dir))
cat("Final summary:\n")
print(summary_df, row.names = FALSE)
