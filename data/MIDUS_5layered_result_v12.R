rm(list = ls())

library(dplyr)
library(purrr)
library(cluster)   # silhouette()

################################################################################
# v12: Joint multilayered LSIRM + EPA pairwise-partition prior on MIDUS
#
# DIFFERENCES FROM v11
#   - Replaces the finite NIW Gaussian mixture clustering on
#     z_q := b_{j(q)}^{(l(q))} with the EPA pairwise partition prior
#     (Dahl, Day & Tsai 2017) using similarity
#         lambda_qr(z, tau) = exp(-tau ||z_q - z_r||^2)
#     and concentration alpha.  No K_star, e0, mu_l, Sigma_l, S0 — the
#     partition is drawn directly from p_EPA(rho | sigma, z, alpha, tau, delta).
#     b_j prior reverts to isotropic Gaussian b_j ~ N_d(0, sigma_b^2 I_d);
#     the EPA log-prior contributes to the b_j MH ratio under
#     b_epa_coupling = TRUE (paper-canonical fully-joint coupling).
#   - Partition sampler: per-sweep block of
#         (i)   single-item EPA Gibbs on c
#         (ii)  random-swap MH on the permutation sigma (M_perm = P_total)
#         (iii) log-scale MH on (alpha, tau) when update_* is on
#         (iv)  Jain-Neal nonconjugate split-merge with restricted Gibbs
#               (M_SM = 1, R = 5 launch scans)
#   - (alpha, tau) reference setting locked in from
#     simulation_4_layered_v12_2.R sweep:
#         alpha = 1 fixed, tau = 1 fixed.
#     Simulation showed that (i) raising tau monotonically kills the
#     split MH rate so the chain fuses adjacent clusters into K=3, and
#     (ii) free (alpha, tau) collapses K_+ to 2 because alpha drifts to
#     ~0.47 and over-merges.  At (1, 1) the chain concentrates on the
#     true K with hclust(b̂) ARI ~ 0.98--1.00.
#   - Cluster-reporting follows simulation lessons: use median(K_+)
#     avg-link cut on (1 - PSM) and hclust on b posterior mean as the
#     robust point estimates.  Dahl / Binder / VI alternatives are still
#     computed but tend to collapse to K=1 in this regime because the
#     within/between co-cluster contrast is small even when K_+ trace is
#     concentrated (partition-space jitter blurs the PSM).
#
# IMPORTANT NOTES
#   - Cluster labels in epa_c are subject to label switching; use the
#     co-cluster matrix, K_+ trace, and hclust(b̂) for inference.
#   - The EPA partition block is O(P_total^2) per pmf evaluation.
################################################################################

################################################################################
# v11 (legacy header kept for reference):
#       Joint multilayered LSIRM + LATENT-POSITION Gaussian-mixture clustering
#       on MIDUS
#
# DIFFERENCES FROM v10
#   - PPCA measurement layer is REMOVED.  Item clustering is placed
#     directly on the LSIRM item latent positions
#         z_q := b_{j(q)}^{(l(q))} in R^d
#     via a finite NIW Gaussian mixture
#         z_q | c_q = l, mu_l, Sigma_l  ~  N_d(mu_l, Sigma_l).
#     No eta_j, Lambda, delta, sigma_eps_sq, sigma_delta_sq,
#     tau_lambda_sq, row_center.
#   - The b_j MH update INCLUDES the mixture prior log-density, so the
#     clustering and LSIRM are FULLY JOINTLY coupled (as opposed to the
#     v10 one-way LSIRM -> PPCA -> FMC coupling).  See modeling_paper
#     model_v11_latent_position_mixture.tex Sec. 12 for the MH ratio.
#   - Adds a Wishart hyperprior on S0 (Option A in the design log):
#         S0 ~ W(nu_S0, Lambda_S0)
#     so that the cluster-covariance scale is itself learned from data
#     instead of being a manually tuned hyperparameter.  Default
#     nu_S0 = d + 2, Lambda_S0 = S0_init / nu_S0 -> E[S0] = S0_init.
#   - Adds an OPTIONAL "b-prior variance inflation" knob
#     (b_prior_inflation = alpha) used inside the b MH update only:
#         b ~ N_d(mu_{c_q}, alpha * Sigma_{c_q})  (b MH ratio only)
#     The c, mu, Sigma updates use the original Sigma (alpha = 1).
#     alpha = 1   (default): paper-defined fully joint coupling.
#     alpha > 1   : weaker pull on b -> better LSIRM-geometry recovery
#                   without changing cluster identity.  Use alpha = 5
#                   if downstream geometry recovery is the bottleneck;
#                   alpha = 1 is fine for cluster-recovery-focused runs.
#   - Mixture / NIW parameters live in R^d (d = 2 here) -- there is no
#     separate r_fac.  The prior implied per-dim sd of Sigma_l now refers
#     to b's coordinate space, not eta's.
#
# IMPORTANT NOTES
#   - Cluster labels in fmc_c are still subject to label switching.  Use
#     the co-cluster matrix and K_+ trace for inference.  This file
#     reports four point-estimate partitions (P0 = avg-link cut, plus
#     Binder / Dahl / VI restricted-to-MCMC-samples) for comparison.
#   - The Sigma_l update uses ALL items in cluster l (anisotropic Sigma
#     allowed); the Wishart hyperprior on S0 averages cluster shapes
#     across components but keeps Sigma_l shape-driven.
#   - Split-merge proposals are still Jain-Neal restricted Gibbs with
#     NIW-collapsed predictive (algorithmically identical to v10);
#     n_split_merge = 100 in v11 reflects the lower per-attempt
#     acceptance under the joint coupling (b is random-walk MH, slower
#     mixing than v10's eta_j Gibbs).
################################################################################

################################################################################
# 0. Path setup
################################################################################
data_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
setwd(data_dir)

################################################################################
# 0-1. Data preparation: Wave 2 + Refresher 1 (v4 preprocess)
################################################################################
cat("\n====== Wave 2 preprocess ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_v4.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all

cat("\n====== Refresher 1 preprocess ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), local = env_r1)
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
    Y_bin  = Y_bin,  Y_cnt  = Y_cnt,
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2, Y_con = Y_con,
    row_ids  = c(l_w2$row_ids, l_r1$row_ids),
    branch   = c(l_w2$branch,  l_r1$branch),
    source   = c(rep("wave2",      length(l_w2$row_ids)),
                 rep("refresher1", length(l_r1$row_ids))),
    col_bin  = l_w2$col_bin,  col_cnt  = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1, label = "P1-P3-P4")

################################################################################
# 0-2. Model / utility loading
################################################################################
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v12.R"))   # joint LSIRM + EPA partition v12
if (file.exists(file.path(data_dir, "utils.R")))
source(file.path(data_dir, "utils.R"))

has_valid <- function(x) {
  if (is.null(x)) return(FALSE)
  if (is.matrix(x) && ncol(x) == 0) return(FALSE)
  if (is.array(x) && length(dim(x)) == 3 && dim(x)[2] == 0) return(FALSE)
  if (all(is.na(x))) return(FALSE)
  TRUE
}

mode_label <- function(v)
as.integer(names(sort(table(v), decreasing = TRUE))[1])

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal)
if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]

plot_trace_vec <- function(samples_mat, name = "param", mfrow = c(2, 2)) {
  samples_mat <- as.matrix(samples_mat)
  par(mfrow = mfrow)
  for (i in seq_len(ncol(samples_mat))) {
    x <- samples_mat[, i]
    q <- quantile(x, c(.025, .975), na.rm = TRUE)
    ts.plot(x, main = sprintf("%s_%d", name, i))
    abline(h = c(mean(x, na.rm = TRUE), q),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
}
plot_trace_scalar_local <- function(x, true = NA, main = "", transform = identity) {
  y <- transform(x)
  q <- quantile(y, c(.025, .975), na.rm = TRUE)
  ts.plot(y, main = main)
  abline(h = c(mean(y, na.rm = TRUE), q),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
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

# LSIRM proposal SDs are inherited from v10's MIDUS-tuned setting.  These were
# calibrated for a MIDUS chain where b had N(0, I_d) prior; the v11 mixture
# prior on b is structurally tighter, so the same proposal SDs may give
# slightly lower b acceptance.  Re-tune by inspecting result$accept$b{1..5}
# after the run if it drops below 0.15.
common_lsirm_prop_sd <- list(
  alpha1 = 0.78, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.60, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.15, log_gamma3 = 0.15,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.5,
  beta1 = 0.50, beta2 = 0.4, beta3 = 0.20, beta4 = 0.50, beta5 = 0.30,
  b1 = 0.5, b2 = 0.4, b3 = 0.5, b4 = 0.4, b5 = 0.50,
  log_kappa = 0.30
)

# ---- EPA defaults (v12: pairwise partition prior, no NIW mixture) ----
#
# Reference setting from simulation_4_layered_v12_2.R sweep (4-layer
# simulation, P_total = 120, n = 150, K_true = 4, tight clusters):
#   alpha = 1 fixed, tau = 1 fixed  ->  median(K_+) = 4, hclust ARI = 0.98
#   alpha free, tau free            ->  median(K_+) = 2, hclust ARI = 0.34
#   alpha = 1, tau = 2 fixed         ->  median(K_+) = 3, hclust ARI = 0.71
#   alpha = 1, tau = 3 fixed         ->  median(K_+) = 3, hclust ARI = 0.71
# We carry the (1, 1) fixed setting over to MIDUS as the default; sigma_b
# is the b prior std-dev (not variance), kept at 1.0 to match the
# simulation and to leave the b-prior scale consistent with v10/v11.

d        <- 2L
sigma_b  <- 1.0

# (alpha, tau): EPA mass parameter and similarity temperature.
# Env-var overridable so a tau-sweep can be run without editing the file.
alpha_fix    <- as.numeric(Sys.getenv("V12_ALPHA_FIX",    "1.0"))
tau_fix      <- as.numeric(Sys.getenv("V12_TAU_FIX",      "1.0"))
update_alpha <- as.logical(Sys.getenv("V12_UPDATE_ALPHA", "FALSE"))
update_tau   <- as.logical(Sys.getenv("V12_UPDATE_TAU",   "FALSE"))

# K_init: starting partition count for the EPA Gibbs init via k-means on
# the PCA-b-proxy.  The EPA sampler is open-ended so this only seeds
# c^{(0)}; the chain explores K_+ freely from there.
K_init <- 10L

# EPA hyperpriors.  alpha ~ Gamma(a_alpha, b_alpha), tau ~ Gamma(a_tau, b_tau);
# delta = 0 (DDT 2017 default).  These hyperpriors are only used when the
# corresponding update_* flag is TRUE; with alpha_fix and tau_fix supplied
# they double as proposal-init values.
common_epa_hyper <- list(
  a_alpha = 1.0, b_alpha = 1.0,
  a_tau   = 1.0, b_tau   = 1.0,
  delta   = 0.0
)

# Log-scale random-walk MH proposal SDs for (alpha, tau) when updating.
common_epa_prop_sd <- list(log_alpha = 0.5, log_tau = 0.5)

# Per-sweep partition sampler ops.
n_split_merge   <- 1L          # M_SM   (Jain-Neal split-merge attempts)
n_split_merge_R <- 5L          # R      (restricted-Gibbs launch scans)
n_perm_swaps    <- NULL        # M_perm (default = P_total in wrapper)
b_epa_coupling  <- TRUE        # paper-canonical fully-joint coupling
epa_warmup      <- 0L          # paper has no warmup; all updates from iter 0

# Long chain: EPA partition block is O(P_total^2) per pmf eval and the
# partition trace mixes more slowly than the LSIRM block.  100k / 20k / 10
# matches the v12_2 simulation sweep that gave a clean K_+ posterior.
# Override via env vars V12_N_ITER / V12_BURNIN / V12_THIN.
common_mcmc <- list(
  d      = d,
  n_iter = as.integer(Sys.getenv("V12_N_ITER", "100000")),
  burnin = as.integer(Sys.getenv("V12_BURNIN", "20000")),
  thin   = as.integer(Sys.getenv("V12_THIN",   "10"))
)
nu2 <- 4

plot_root <- file.path(data_dir, "plot")
if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE)

################################################################################
# 2. Case definition (lsirm_all 기준, Y_ord2 항상 제외; case1_all만 사용)
################################################################################
n_all <- nrow(lsirm_all$Y_con)
make_empty <- function(n) matrix(0L, nrow = n, ncol = 0)
E  <- make_empty(n_all)
Eo <- matrix(0L, nrow = n_all, ncol = 0)

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
# 3. Variable subsets (v6 / v7 / v9 / v10와 동일 switch)
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

Y_ord1_subset   <- Y_ord1_full
col_ord1_subset <- colnames(Y_ord1_full)

cs$Y_con   <- Y_con_subset;   cs$col_con   <- col_con_subset
cs$Y_cnt   <- Y_cnt_subset;   cs$col_cnt   <- col_cnt_subset
cs$Y_ord1  <- Y_ord1_subset;  cs$col_ord1  <- col_ord1_subset

################################################################################
# 3-bis. Smart EPA initialisation
#
# v12 EPA pmf is non-exchangeable in (sigma, c) so the initial partition
# only weakly biases the chain; we still seed with a non-degenerate
# k-means partition on the PCA-b-proxy.  The proxy = first d PCs of the
# centred/log-transformed response matrix transposed so items are rows.
# alpha and tau init to their fixed values (or prior mean if they were
# being updated).
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
set.seed(0502)  # reproducibility for k-means init AND MCMC chain
km <- kmeans(b_init_proxy,
             centers = min(K_init, P_total_init - 1L),
             nstart = 25, iter.max = 50)
c_init_km <- km$cluster

epa_init_smart <- list(
  c     = as.integer(c_init_km),                # 1-based; wrapper -> 0-based
  sigma = sample.int(P_total_init, P_total_init), # uniform Perm(P_total) draw
  alpha = alpha_fix,
  tau   = tau_fix
)

cat(sprintf("[EPA init] PCA-b-proxy scale: sd per dim = %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))
cat(sprintf("[EPA init] kmeans cluster sizes (K_init=%d): %s\n",
            K_init, paste(table(c_init_km), collapse = " ")))

################################################################################
# 4. Single chain run
################################################################################
run_label     <- sprintf("v12_epa_d%d_sigb_%g_a%g_t%g_uA%s_uT%s_M%d_R%d_iter%dk",
                         d, sigma_b, alpha_fix, tau_fix,
                         if (update_alpha) "T" else "F",
                         if (update_tau)   "T" else "F",
                         n_split_merge, n_split_merge_R,
                         common_mcmc$n_iter %/% 1000L)
case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", run_label))
if (!dir.exists(case_plot_dir)) dir.create(case_plot_dir, recursive = TRUE)
prefix     <- paste0(cs$name, "_", run_label)
epa_prefix <- paste0(prefix, "_epa")

cat(sprintf("\n========== %s [%s] ==========\n", cs$label, run_label))
cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
            nrow(cs$Y_bin),  ncol(cs$Y_bin),
            nrow(cs$Y_con),  ncol(cs$Y_con),
            nrow(cs$Y_cnt),  ncol(cs$Y_cnt),
            nrow(cs$Y_ord1), ncol(cs$Y_ord1),
            nrow(cs$Y_ord2), ncol(cs$Y_ord2)))

##############################################################################
# 4-A. v12 joint MCMC (LSIRM + EPA partition)
##############################################################################
result <- lsirm_epa_v12_cpp(
  Y_bin   = cs$Y_bin,
  Y_con   = round(cs$Y_con, 1),
  Y_cnt   = cs$Y_cnt,
  Y_ord1  = cs$Y_ord1,
  Y_ord2  = cs$Y_ord2,
  d       = common_mcmc$d,
  n_iter  = common_mcmc$n_iter,
  burnin  = common_mcmc$burnin,
  thin    = common_mcmc$thin,
  nu2     = nu2,
  lsirm_hyper   = common_lsirm_hyper,
  epa_hyper     = common_epa_hyper,
  lsirm_prop_sd = common_lsirm_prop_sd,
  epa_prop_sd   = common_epa_prop_sd,
  lsirm_init    = NULL,
  epa_init      = epa_init_smart,
  sigma_b       = sigma_b,
  compute_co_cluster_online = TRUE,
  epa_warmup      = epa_warmup,
  n_split_merge   = n_split_merge,
  n_split_merge_R = n_split_merge_R,
  n_perm_swaps    = n_perm_swaps,
  b_epa_coupling  = b_epa_coupling,
  update_alpha    = update_alpha,
  update_tau      = update_tau,
  verbose       = TRUE,
  fix_gamma     = FALSE
)

##############################################################################
# 4-B. Acceptance summary + split-merge diagnostics
##############################################################################
acc <- result$accept
cat(sprintf("\n-- %s Acceptance --\n", run_label))
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
            mean(acc$b3), mean(acc$b4), mean(acc$b5)))
cat(sprintf("  LSIRM log_kappa per-item mean: %.3f\n", mean(acc$log_kappa)))

sm <- result$epa_diagnostics
cat(sprintf("\n-- %s EPA partition diagnostics --\n", run_label))
cat(sprintf("  split: %d / %d  (rate %.3f)\n",
            sm$split_accepts, sm$split_attempts, sm$split_rate))
cat(sprintf("  merge: %d / %d  (rate %.3f)\n",
            sm$merge_accepts, sm$merge_attempts, sm$merge_rate))
cat(sprintf("  sigma swaps: %.0f / %.0f  (rate %.3f)\n",
            sm$sigma_swap_accepts, sm$sigma_swap_attempts, sm$sigma_swap_rate))
cat(sprintf("  alpha MH: %d / %d  (rate %.3f)\n",
            sm$alpha_epa_accepts, sm$alpha_epa_attempts, sm$alpha_epa_rate))
cat(sprintf("  tau   MH: %d / %d  (rate %.3f)\n",
            sm$tau_epa_accepts, sm$tau_epa_attempts, sm$tau_epa_rate))
cat(sprintf("  net K_+ change (split_acc - merge_acc) = %d\n",
            sm$split_accepts - sm$merge_accepts))

# Tuning mode: skip all post-MCMC processing if env var is set.
if (Sys.getenv("MIDUS_TUNE_ONLY", "0") == "1") {
  saveRDS(result$accept,
          file.path(data_dir, "midus_tune_acc.rds"))
  cat("\n[MIDUS_TUNE_ONLY] saved acceptance rds, exiting.\n")
  quit(save = "no", status = 0)
}
case_plot_dir
##############################################################################
# 4-C. LSIRM traceplots (identical structure to v10)
##############################################################################
if (has_valid(result$a)) {
  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_a.pdf")),
      width = 8, height = 12)
  par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
  for (i in 1:dim(result$a)[2])
    for (j in 1:dim(result$a)[3])
      ts.plot(result$a[, i, j], main = paste0("a: ", i, "_", j))
  dev.off()
}

for (k in 1:5) {
  bname <- paste0("b", k)
  bmat <- result[[bname]]
  if (!has_valid(bmat) || dim(bmat)[2] == 0) next
  col_layer <- switch(bname,
                      b1 = cs$col_bin, b2 = cs$col_con, b3 = cs$col_cnt,
                      b4 = cs$col_ord1, b5 = cs$col_ord2)
  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", bname, ".pdf")),
      width = 8, height = 12)
  par(mfrow = c(2, 2), mar = c(3, 3, 2, 1))
  for (i in 1:dim(bmat)[2])
    for (j in 1:dim(bmat)[3])
      ts.plot(bmat[, i, j],
              main = paste0(bname, ": ",
                            (if (length(col_layer) >= i) col_layer[i] else i),
                            "_d", j))
  dev.off()
}

for (al in 1:5) {
  aname <- paste0("alpha", al)
  if (has_valid(result[[aname]]) && ncol(result[[aname]]) > 0) {
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", aname, ".pdf")),
        width = 8, height = 12)
    plot_trace_vec(result[[aname]], name = aname, mfrow = c(3, 2))
    dev.off()
  }
}

for (bn in c("beta1", "beta2", "beta3")) {
  if (has_valid(result[[bn]]) && ncol(result[[bn]]) > 0) {
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", bn, ".pdf")),
        width = 8, height = 12)
    plot_trace_vec(result[[bn]], name = bn, mfrow = c(3, 2))
    dev.off()
  }
}

if (has_valid(result$beta4)) {
  b4s <- result$beta4
  P4d <- dim(b4s)[2]; Km1 <- dim(b4s)[3]
  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_beta4_thr.pdf")),
      width = 8, height = 12)
  par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
  for (j in 1:P4d) for (k in 1:Km1) {
    x <- b4s[, j, k]
    ts.plot(x,
            main = sprintf("beta4[%s, k=%d]",
                           ifelse(length(cs$col_ord1) >= j,
                                  cs$col_ord1[j], j), k))
    abline(h = c(mean(x), quantile(x, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()
}
if (has_valid(result$beta5)) {
  b5s <- result$beta5
  P5d <- dim(b5s)[2]; Km1 <- dim(b5s)[3]
  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_beta5_thr.pdf")),
      width = 8, height = 12)
  par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
  for (j in 1:P5d) for (k in 1:Km1) {
    x <- b5s[, j, k]
    ts.plot(x,
            main = sprintf("beta5[%s, k=%d]",
                           ifelse(length(cs$col_ord2) >= j,
                                  cs$col_ord2[j], j), k))
    abline(h = c(mean(x), quantile(x, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()
}

if (has_valid(result$log_kappa) && is.matrix(result$log_kappa) &&
    ncol(result$log_kappa) > 0) {
  lk <- result$log_kappa
  P3d <- ncol(lk)
  cn <- if (length(cs$col_cnt) >= P3d) cs$col_cnt else paste0("cnt_j", 1:P3d)
  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_lsirm_kappa_per_item.pdf")),
      width = 10, height = 12)
  par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
  for (j in 1:P3d) {
    kx <- exp(lk[, j])
    ts.plot(kx, main = sprintf("LSIRM kappa[%s]", cn[j]))
    abline(h = c(mean(kx), quantile(kx, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()
}

pdf(file.path(case_plot_dir, paste0(prefix, "_trace_lsirm_extra.pdf")),
    width = 8, height = 18)
par(mfrow = c(7, 2), mar = c(3, 3, 2, 1))
plot_trace_scalar_local(result$sigma0_sq,  main = "sigma0_sq")
plot_trace_scalar_local(result$log_gamma1, main = "gamma1 (Bin)",  transform = exp)
plot_trace_scalar_local(result$log_gamma2, main = "gamma2 (Con)",  transform = exp)
plot_trace_scalar_local(result$log_gamma3, main = "gamma3 (Cnt)",  transform = exp)
plot_trace_scalar_local(result$log_gamma4, main = "gamma4 (Ord1)", transform = exp)
plot_trace_scalar_local(result$log_gamma5, main = "gamma5 (Ord2)", transform = exp)
for (al in 1:5) {
  sname <- paste0("sigma_alpha", al, "_sq")
  if (has_valid(result[[sname]])) plot_trace_scalar_local(result[[sname]], main = sname)
}
if (has_valid(result$lambda2_mean))
  plot_trace_scalar_local(result$lambda2_mean, main = "lambda2_mean")
dev.off()

##############################################################################
# 4-D. EPA parameter traceplots
#
# v12: K_+, alpha (if updated), tau (if updated), log p_EPA, cluster sizes.
# No rho / mu_l / Sigma_l / S0 — EPA partition has no NIW component params.
##############################################################################
n_save  <- length(result$epa_K_plus)
P_total <- ncol(result$epa_c)
item_names_full <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1, cs$col_ord2)
item_names_full <- item_names_full[seq_len(P_total)]

# K_+ trace
pdf(file.path(case_plot_dir, paste0(epa_prefix, "_trace_K_plus.pdf")),
    width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(result$epa_K_plus, main = "Occupied cluster count K_+",
        ylab = expression(K["+"]^{(s)}))
abline(h = c(mean(result$epa_K_plus),
             quantile(result$epa_K_plus, c(.025, .975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
dev.off()

# (alpha, tau) traces
pdf(file.path(case_plot_dir, paste0(epa_prefix, "_trace_alpha_tau.pdf")),
    width = 9, height = 8)
par(mfrow = c(2, 1), mar = c(3, 3, 2, 1))
ts.plot(result$epa_alpha,
        main = sprintf("EPA alpha  (update=%s)", update_alpha))
abline(h = c(mean(result$epa_alpha),
             quantile(result$epa_alpha, c(.025, .975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
ts.plot(result$epa_tau,
        main = sprintf("EPA tau  (update=%s)", update_tau))
abline(h = c(mean(result$epa_tau),
             quantile(result$epa_tau, c(.025, .975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
dev.off()

# log p_EPA
if (!is.null(result$epa_log_pmf)) {
  pdf(file.path(case_plot_dir, paste0(epa_prefix, "_trace_log_pmf.pdf")),
      width = 9, height = 5)
  par(mar = c(4, 4, 3, 1))
  ts.plot(result$epa_log_pmf,
          main = "log p_EPA(rho | sigma, z, alpha, tau)")
  abline(h = mean(result$epa_log_pmf), col = "darkgreen", lwd = 2)
  dev.off()
}

# Cluster sizes (open-ended; cap visualised slots at max occupied K)
n_slots_trace <- max(result$epa_K_plus)
n_slots_show  <- min(n_slots_trace, length(cluster_pal) * 2L)
n_l_trace <- t(apply(result$epa_c, 1,
                     function(v) tabulate(v, nbins = n_slots_show)))
pdf(file.path(case_plot_dir, paste0(epa_prefix, "_trace_cluster_sizes.pdf")),
    width = 10, height = 5)
par(mar = c(4, 4, 3, 1))
matplot(n_l_trace, type = "l", lty = 1,
        col = expand_pal(n_slots_show, cluster_pal),
        xlab = "saved iter", ylab = expression(n[l]^{(s)}),
        main = sprintf("Item cluster sizes (epa_c, max K_+ = %d)",
                       n_slots_trace))
legend("topright", legend = paste0("l=", 1:n_slots_show),
       col = expand_pal(n_slots_show, cluster_pal), lty = 1,
       bty = "n", cex = 0.75)
dev.off()

##############################################################################
# 4-E. Posterior summaries — co-cluster + final partition + biplot
##############################################################################
co_cluster <- result$epa_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

item_cluster_mode <- apply(result$epa_c, 2, mode_label)

median_K_plus <- max(2, round(median(result$epa_K_plus)))
hc_co <- hclust(as.dist(1 - co_cluster), method = "average")
final_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

ord_hc <- hc_co$order

n_slots_heat <- max(result$epa_c)
pdf(file.path(case_plot_dir, paste0(epa_prefix, "_membership_heatmap_c.pdf")),
    width = 10, height = 7)
par(mar = c(7, 4, 3, 1))
image(seq_len(n_save), seq_len(P_total),
      result$epa_c[, ord_hc, drop = FALSE],
      col = expand_pal(n_slots_heat, cluster_pal),
      xlab = "MCMC iteration (saved)", ylab = "",
      main = "Item cluster membership trace  (epa_c over iterations; ordered by hclust on 1-PSM; subject to label switching)",
      axes = FALSE)
axis(1)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.55)
box()
legend("topright", legend = paste0("l=", 1:n_slots_heat),
       fill = expand_pal(n_slots_heat, cluster_pal),
       bty = "n", cex = 0.75)
dev.off()

##############################################################################
# 4-E-bis. Alternative partitions via Binder / Dahl / VI loss minimisation
#
# We keep `final_partition` (P0 above) as the default for downstream
# silhouette / PEAR / biplot, but compute four principled alternatives:
#   P1: minBinder (a=b=1), unconstrained (mcclust.ext::minbinder.ext)
#   P3: minBinder (a=b=1), draws-only (mcclust::minbinder, method="draws")
#   P4: minVI, unconstrained (mcclust.ext::minVI)
#   P5: Dahl (2006) -- pick MCMC sample minimising Frobenius ||C^(s) - PSM||
#       (self-contained, no mcclust dependency)
##############################################################################

# Self-contained Dahl point estimate (used regardless of mcclust availability)
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
dahl <- dahl_partition_fun(result$epa_c, co_cluster)
P5_dahl <- dahl$partition

if (requireNamespace("mcclust", quietly = TRUE) &&
    requireNamespace("mcclust.ext", quietly = TRUE)) {

  loss_binder <- function(c_, psm) mcclust::binder(c_, psm)
  loss_vi     <- function(c_, psm) mcclust.ext::VI.lb(matrix(c_, nrow = 1), psm)[1]
  k_unique    <- function(c_) length(unique(c_))

  bin_uncon <- mcclust.ext::minbinder.ext(
    co_cluster, cls.draw = result$epa_c,
    method = "all", include.lg = TRUE, include.greedy = TRUE,
    suppress.comment = TRUE
  )
  P1 <- as.integer(bin_uncon$cl[which.min(bin_uncon$value), ])
  P1_winner <- rownames(bin_uncon$cl)[which.min(bin_uncon$value)]

  bin_draws <- mcclust::minbinder(
    co_cluster, cls.draw = result$epa_c, method = "draws"
  )
  P3 <- as.integer(if (is.matrix(bin_draws$cl)) bin_draws$cl[1, ] else bin_draws$cl)

  vi_uncon <- mcclust.ext::minVI(
    co_cluster, cls.draw = result$epa_c,
    method = "all", include.greedy = TRUE, suppress.comment = TRUE
  )
  P4 <- as.integer(vi_uncon$cl[which.min(vi_uncon$value), ])
  P4_winner <- rownames(vi_uncon$cl)[which.min(vi_uncon$value)]

  alt_list <- list(
    P0_prev_avgcut             = list(c = as.integer(final_partition), winner = "median-K_+ avg-link"),
    P1_minBinder_unconstrained = list(c = P1, winner = P1_winner),
    P3_minBinder_drawsOnly     = list(c = P3, winner = "draws"),
    P4_minVI_unconstrained     = list(c = P4, winner = P4_winner),
    P5_Dahl_drawsOnly          = list(c = P5_dahl, winner = sprintf("iter=%d", dahl$iter))
  )

  # b posterior mean used for silhouette in v11 (the LSIRM latent space
  # IS the mixture's coordinate space; no separate eta).
  A_hat_pm <- apply(result$a, c(2, 3), mean)
  b_arrs <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
  B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
    if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
      return(matrix(0, 0, common_mcmc$d))
    apply(arr, c(2, 3), mean)
  }))
  rownames(B_hat_pm) <- item_names_full

  d_b_for_sil <- dist(B_hat_pm)
  sil_one <- function(c_) {
    if (k_unique(c_) < 2) return(NA_real_)
    s <- cluster::silhouette(c_, d_b_for_sil)
    mean(s[, "sil_width"])
  }

  alt_summary <- data.frame(
    partition = names(alt_list),
    winner    = vapply(alt_list, `[[`, "", "winner"),
    K         = vapply(alt_list, function(x) k_unique(x$c), 0L),
    binder    = vapply(alt_list, function(x) loss_binder(x$c, co_cluster), 0),
    vi_lb     = vapply(alt_list, function(x) loss_vi(x$c, co_cluster),     0),
    b_sil     = vapply(alt_list, function(x) sil_one(x$c),                 0)
  )
  ari_vs_P0 <- function(c_) {
    a <- alt_list$P0_prev_avgcut$c
    tab <- table(a, c_); n_ <- sum(tab); if (n_ < 2) return(NA_real_)
    sc <- sum(choose(rowSums(tab), 2)); sk <- sum(choose(colSums(tab), 2))
    st <- sum(choose(tab, 2))
    exp_idx <- sc * sk / choose(n_, 2)
    max_idx <- (sc + sk) / 2
    if (max_idx == exp_idx) return(1)
    (st - exp_idx) / (max_idx - exp_idx)
  }
  alt_summary$ari_vs_P0 <- vapply(alt_list, function(x) ari_vs_P0(x$c), 0)
  alt_summary$binder    <- round(alt_summary$binder, 3)
  alt_summary$vi_lb     <- round(alt_summary$vi_lb, 4)
  alt_summary$b_sil     <- round(alt_summary$b_sil, 3)
  alt_summary$ari_vs_P0 <- round(alt_summary$ari_vs_P0, 3)

  cat("\n-- v12 alternative partitions (P0 still used downstream) --\n")
  cat(sprintf("  Binder convention: sum_{j<k} |I{c_j=c_k} - C_jk|, range [0, %d]\n",
              choose(P_total, 2)))
  print(alt_summary, row.names = FALSE)

  alt_per_item <- data.frame(
    item                       = item_names_full,
    P0_prev_avgcut             = alt_list$P0_prev_avgcut$c,
    P1_minBinder_unconstrained = alt_list$P1_minBinder_unconstrained$c,
    P3_minBinder_drawsOnly     = alt_list$P3_minBinder_drawsOnly$c,
    P4_minVI_unconstrained     = alt_list$P4_minVI_unconstrained$c,
    P5_Dahl_drawsOnly          = alt_list$P5_Dahl_drawsOnly$c
  )
  write.csv(alt_per_item,
            file.path(case_plot_dir,
                      paste0(epa_prefix, "_alt_partitions.csv")),
            row.names = FALSE)
  write.csv(alt_summary,
            file.path(case_plot_dir,
                      paste0(epa_prefix, "_alt_partition_summary.csv")),
            row.names = FALSE)

} else {
  cat("\n[alt partitions skipped] mcclust and/or mcclust.ext not installed.\n")
  cat("  install via:\n")
  cat("    install.packages('mcclust')\n")
  cat("    remotes::install_github('sarawade/mcclust.ext')\n")
  cat("  Falling back to P0 (avg-link) and P5 (Dahl) only.\n")

  A_hat_pm <- apply(result$a, c(2, 3), mean)
  b_arrs <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
  B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
    if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
      return(matrix(0, 0, common_mcmc$d))
    apply(arr, c(2, 3), mean)
  }))
  rownames(B_hat_pm) <- item_names_full

  alt_per_item <- data.frame(
    item                = item_names_full,
    P0_prev_avgcut      = as.integer(final_partition),
    P5_Dahl_drawsOnly   = P5_dahl
  )
  write.csv(alt_per_item,
            file.path(case_plot_dir,
                      paste0(epa_prefix, "_alt_partitions.csv")),
            row.names = FALSE)
}

pdf(file.path(case_plot_dir, paste0(epa_prefix, "_co_cluster.pdf")),
    width = 9, height = 8)
par(mar = c(7, 7, 3, 1))
image(seq_along(item_names_full), seq_along(item_names_full),
      co_cluster[ord_hc, ord_hc],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("EPA item co-clustering  (d=%d, sigma_b=%g, alpha=%g, tau=%g; ordered by hclust on 1-PSM)",
                     d, sigma_b, alpha_fix, tau_fix))
axis(1, at = seq_along(item_names_full), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.6)
axis(2, at = seq_along(item_names_full), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.6)
box()
dev.off()

# v11: b posterior-mean scatter (replaces v10's eta posterior-mean plot)
pdf(file.path(case_plot_dir, paste0(epa_prefix, "_b_postmean.pdf")),
    width = 9, height = 8)
par(mar = c(4, 4, 3, 1))
pal_use <- expand_pal(max(item_cluster_mode), cluster_pal)
plot(B_hat_pm[, 1], B_hat_pm[, 2], pch = 19, cex = 1.4,
     col = pal_use[item_cluster_mode],
     xlab = "b dim 1", ylab = "b dim 2",
     main = "b posterior mean  (coloured by mode cluster)")
text(B_hat_pm[, 1], B_hat_pm[, 2], labels = item_names_full,
     pos = 4, cex = 0.55)
dev.off()

##############################################################################
# 4-E (cont). Biplot
##############################################################################
make_epa_biplot <- function(A_hat, B_hat, item_partition, item_names,
                            title, filename, plot_dir, pal) {
  k_max <- max(item_partition)
  pal_use <- if (k_max > length(pal))
    colorRampPalette(pal)(k_max) else pal[seq_len(k_max)]
  pdf(file.path(plot_dir, filename), width = 10, height = 8)
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
  uq_part <- sort(unique(item_partition))
  legend("topright",
         legend = c("Respondents (a_i)",
                    sprintf("Item partition %d", uq_part)),
         pch    = c(21, rep(22, length(uq_part))),
         pt.bg  = c(adjustcolor("gray60", alpha.f = 0.30),
                    pal_use[uq_part]),
         col    = c("gray40", rep("black", length(uq_part))),
         bty = "n", cex = 0.75)
  dev.off()
}

if (common_mcmc$d == 2 && nrow(B_hat_pm) >= 1) {
  make_epa_biplot(
    A_hat = A_hat_pm, B_hat = B_hat_pm,
    item_partition = final_partition,
    item_names     = item_names_full,
    title = sprintf("Joint LSIRM + EPA v12  (d=%d, sigma_b=%g, alpha=%g, tau=%g, mean K_+=%.1f, split=%.2f)  |  %s",
                    d, sigma_b, alpha_fix, tau_fix,
                    mean(result$epa_K_plus),
                    sm$split_rate,
                    cs$label),
    filename = paste0(epa_prefix, "_biplot.pdf"),
    plot_dir = case_plot_dir,
    pal      = cluster_pal
  )
} else {
  cat(sprintf("[biplot skipped] d = %d (biplot is implemented for d == 2 only)\n",
              common_mcmc$d))
}

##############################################################################
# 4-F. CSV outputs and saved RDS
##############################################################################
saveRDS(result, file.path(case_plot_dir, paste0(prefix, "_result.rds")))

write.csv(data.frame(item             = item_names_full,
                     epa_mode_cluster = item_cluster_mode,
                     epa_partition    = final_partition,
                     epa_dahl         = P5_dahl),
          file.path(case_plot_dir, paste0(epa_prefix, "_item_clusters.csv")),
          row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(case_plot_dir, paste0(epa_prefix, "_co_cluster.csv")))

# K_+ summary CSV (incl. silhouette) is written in section 4-I.

write.csv(round(B_hat_pm, 4),
          file.path(case_plot_dir, paste0(epa_prefix, "_b_postmean.csv")))

##############################################################################
# 4-G. v12 cluster-scale diagnostic (b_pm-based)
#
# v12 has no NIW/S0 component, so we just report the empirical within-cluster
# sd of b_pm per dim using modal labels, vs the b prior std (sigma_b).  This
# is informational only; large within-cluster sd relative to sigma_b means
# the LSIRM b's are spread well beyond the prior shrinkage scale.
##############################################################################
empirical_within_sd <- vapply(seq_len(d), function(q) {
  resids <- numeric(0)
  for (l in unique(item_cluster_mode)) {
    idx <- which(item_cluster_mode == l)
    if (length(idx) < 2) next
    resids <- c(resids, B_hat_pm[idx, q] - mean(B_hat_pm[idx, q]))
  }
  if (length(resids) < 2) return(NA_real_)
  sqrt(mean(resids^2))
}, numeric(1))

cat(sprintf("\n-- v12 cluster-scale diagnostic (b_pm based) --\n"))
cat(sprintf("  b prior std (sigma_b): %.4f\n", sigma_b))
cat("  empirical within-cluster sd of b_pm per dim:\n")
print(round(empirical_within_sd, 4))

write.csv(data.frame(
            dim                   = seq_len(d),
            sigma_b               = sigma_b,
            empirical_within_sd_b = empirical_within_sd
          ),
          file.path(case_plot_dir,
                    paste0(epa_prefix, "_b_scale_diagnostic.csv")),
          row.names = FALSE)

##############################################################################
# 4-H. Silhouette diagnostic (b-based, NOT co-cluster-based)
#
# v11: clustering target is z_q = b_j; silhouette uses Euclidean distance
# between item posterior-mean b-positions in R^d.  Same rationale as v10's
# eta-based silhouette (avoid circularity of using (1 - co_cluster) as
# distance).
##############################################################################
sil_obj <- cluster::silhouette(final_partition, dist(B_hat_pm))
sil_mat <- if (!is.null(sil_obj)) as.data.frame(sil_obj[, , drop = FALSE]) else NULL

if (!is.null(sil_mat)) {
  sil_mat$item <- item_names_full
  sil_overall_mean   <- mean(sil_mat$sil_width)
  sil_overall_median <- median(sil_mat$sil_width)
  sil_frac_neg       <- mean(sil_mat$sil_width < 0)

  per_cluster_sil <- aggregate(sil_width ~ cluster, data = sil_mat,
                               FUN = function(x) c(n = length(x),
                                                   mean = mean(x),
                                                   median = median(x),
                                                   min = min(x)))

  cat(sprintf("\n-- v12 silhouette diagnostic (b-based) --\n"))
  cat(sprintf("  overall mean = %.3f, median = %.3f, frac<0 = %.3f\n",
              sil_overall_mean, sil_overall_median, sil_frac_neg))
  cat("  per-cluster:\n")
  print(per_cluster_sil)

  neg_idx <- which(sil_mat$sil_width < 0)
  if (length(neg_idx) > 0) {
    cat(sprintf("\n  Items with negative silhouette (%d, candidates for re-assignment):\n",
                length(neg_idx)))
    print(sil_mat[neg_idx, c("item", "cluster", "sil_width", "neighbor")])
  }

  sil_csv <- sil_mat[, c("item", "cluster", "neighbor", "sil_width")]
  sil_csv$sil_width <- round(sil_csv$sil_width, 4)
  write.csv(sil_csv,
            file.path(case_plot_dir,
                      paste0(epa_prefix, "_silhouette_per_item.csv")),
            row.names = FALSE)

  pdf(file.path(case_plot_dir, paste0(epa_prefix, "_silhouette.pdf")),
      width = 9, height = 6)
  par(mar = c(4, 4, 3, 1))
  plot(sil_obj, main = sprintf("Silhouette (b-based)  k=%d, mean=%.3f",
                                length(unique(final_partition)),
                                sil_overall_mean),
       col = expand_pal(length(unique(final_partition)), cluster_pal),
       border = NA)
  dev.off()
} else {
  sil_overall_mean   <- NA_real_
  sil_overall_median <- NA_real_
  sil_frac_neg       <- NA_real_
  cat("\n[silhouette] insufficient data (need >= 2 clusters with >= 1 item each)\n")
}

##############################################################################
# 4-H-bis. Co-clustering quality diagnostics (same as v10)
##############################################################################
.adj_rand <- function(a, b) {
  tab <- table(a, b); n_ <- sum(tab); if (n_ < 2) return(NA_real_)
  sc <- sum(choose(rowSums(tab), 2)); sk <- sum(choose(colSums(tab), 2))
  st <- sum(choose(tab, 2))
  exp_idx <- sc * sk / choose(n_, 2)
  max_idx <- (sc + sk) / 2
  if (max_idx == exp_idx) return(1)
  (st - exp_idx) / (max_idx - exp_idx)
}

# (1) within-pair confidence ratio
cm_od <- co_cluster[upper.tri(co_cluster)]
within_pair_n <- sum(choose(table(final_partition), 2))
within_pair_frac <- within_pair_n / choose(P_total, 2)
pr_high   <- mean(cm_od > 0.8)
pr_low    <- mean(cm_od < 0.2)
pr_amb    <- mean(cm_od >= 0.3 & cm_od <= 0.7)
within_pair_ratio <- pr_high / within_pair_frac

baseline_mean <- sum((table(final_partition) / P_total)^2) - 1 / P_total
od_mean       <- mean(cm_od)

cat(sprintf("\n-- v12 co-clustering quality --\n"))
cat(sprintf("  off-diag mean = %.3f  (uniform-K baseline = %.3f)\n",
            od_mean, baseline_mean))
cat(sprintf("  Pr(co > 0.8) = %.3f,  within-pair frac = %.3f,  ratio = %.3f\n",
            pr_high, within_pair_frac, within_pair_ratio))
cat(sprintf("  Pr(co < 0.2) = %.3f\n", pr_low))
cat(sprintf("  Pr(co [0.3,0.7]) = %.3f  (ambiguous zone; >30%% = weak)\n",
            pr_amb))
ratio_quality <- (
  if (within_pair_ratio > 0.7) "strong"
  else if (within_pair_ratio > 0.3) "moderate"
  else "weak"
)
cat(sprintf("  -> within-pair ratio is %s\n", ratio_quality))

# (2) PEAR
n_pear_pairs <- 500L
set.seed(123)
n_save_local <- nrow(result$epa_c)
ii <- sample.int(n_save_local, n_pear_pairs, replace = TRUE)
jj <- sample.int(n_save_local, n_pear_pairs, replace = TRUE)
ok <- ii != jj
pear_vals <- mapply(function(i, j) {
  .adj_rand(result$epa_c[i, ], result$epa_c[j, ])
}, ii[ok], jj[ok])
pear_mean <- mean(pear_vals, na.rm = TRUE)
pear_sd   <- sd(pear_vals, na.rm = TRUE)

cat(sprintf("\n-- v12 PEAR (posterior expected adjusted Rand) --\n"))
cat(sprintf("  n_pairs sampled = %d\n", length(pear_vals)))
cat(sprintf("  mean PEAR = %.3f, sd = %.3f\n", pear_mean, pear_sd))
cat(sprintf("  range [%.3f, %.3f]\n",
            min(pear_vals, na.rm = TRUE), max(pear_vals, na.rm = TRUE)))
pear_quality <- (
  if (pear_mean > 0.6) "strong (posterior concentrated)"
  else if (pear_mean > 0.3) "moderate"
  else "weak (posterior diffuse over partition space)"
)
cat(sprintf("  -> %s\n", pear_quality))

write.csv(data.frame(
            off_diag_mean       = od_mean,
            uniform_K_baseline  = baseline_mean,
            pr_co_gt_0.8        = pr_high,
            pr_co_lt_0.2        = pr_low,
            pr_co_ambiguous     = pr_amb,
            within_pair_frac    = within_pair_frac,
            within_pair_ratio   = within_pair_ratio,
            ratio_quality       = ratio_quality,
            pear_mean           = pear_mean,
            pear_sd             = pear_sd,
            pear_quality        = pear_quality
          ),
          file.path(case_plot_dir,
                    paste0(epa_prefix, "_coclustering_quality.csv")),
          row.names = FALSE)

##############################################################################
# 4-I. Final summary (K_+ summary CSV with silhouette + console print)
##############################################################################
write.csv(data.frame(
            mean   = mean(result$epa_K_plus),
            median = median(result$epa_K_plus),
            sd     = sd(result$epa_K_plus),
            min    = min(result$epa_K_plus),
            max    = max(result$epa_K_plus),
            K_init = K_init,
            n_save = n_save,
            split_attempts = sm$split_attempts,
            split_accepts  = sm$split_accepts,
            split_rate     = sm$split_rate,
            merge_attempts = sm$merge_attempts,
            merge_accepts  = sm$merge_accepts,
            merge_rate     = sm$merge_rate,
            sigma_swap_rate = sm$sigma_swap_rate,
            alpha_epa_rate  = sm$alpha_epa_rate,
            tau_epa_rate    = sm$tau_epa_rate,
            mean_alpha_epa  = mean(result$epa_alpha),
            mean_tau_epa    = mean(result$epa_tau),
            sil_mean          = sil_overall_mean,
            sil_median        = sil_overall_median,
            sil_frac_neg      = sil_frac_neg,
            within_pair_ratio = within_pair_ratio,
            pr_co_ambiguous   = pr_amb,
            pear_mean         = pear_mean,
            sigma_b           = sigma_b,
            alpha_fix         = alpha_fix,
            tau_fix           = tau_fix,
            update_alpha      = update_alpha,
            update_tau        = update_tau,
            b_epa_coupling    = b_epa_coupling
          ),
          file.path(case_plot_dir, paste0(epa_prefix, "_K_plus_summary.csv")),
          row.names = FALSE)

cat(sprintf("\n=== v12 EPA summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(result$epa_K_plus), median(result$epa_K_plus),
            sd(result$epa_K_plus)))
cat(sprintf("  mean(alpha) = %.3f, mean(tau) = %.3f\n",
            mean(result$epa_alpha), mean(result$epa_tau)))
cat(sprintf("  silhouette (b-based): mean=%.3f, frac<0=%.3f\n",
            sil_overall_mean, sil_frac_neg))
cat(sprintf("  co-clustering: within-pair ratio=%.3f (%s), ambig=%.3f, PEAR=%.3f\n",
            within_pair_ratio, ratio_quality, pr_amb, pear_mean))
cat(sprintf("  split rate = %.3f, merge rate = %.3f, sigma swap rate = %.3f\n",
            sm$split_rate, sm$merge_rate, sm$sigma_swap_rate))
cat(sprintf("  alpha_fix = %g (update=%s), tau_fix = %g (update=%s), sigma_b = %g\n",
            alpha_fix, update_alpha, tau_fix, update_tau, sigma_b))

cat(sprintf("\n  Final partition (cutree on 1 - PSM, k=%d):\n",
            length(unique(final_partition))))
print(table(final_partition))
cat(sprintf("\n  Mode-cluster table (subject to label switching):\n"))
print(table(item_cluster_mode))
cat(sprintf("\n  Dahl partition (k=%d):\n", length(unique(P5_dahl))))
print(table(P5_dahl))
cat(sprintf("\n-> v12 plots & artifacts saved to: %s\n", case_plot_dir))



result$accept$log_gamma1
result$accept$log_gamma2
result$accept$log_gamma3
result$accept$log_gamma4
result$accept$a
result$accept$b1
result$accept$b2
result$accept$b3
result$accept$b4
result$accept$beta4_thr
