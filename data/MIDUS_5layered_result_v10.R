rm(list = ls())

library(dplyr)
library(purrr)
library(cluster)   # silhouette()

################################################################################
# v10: Joint multilayered LSIRM + PPCA-style mixture clustering on MIDUS
#
# DIFFERENCES FROM v9
#   - Mixture component prior is conjugate NIW (m0, kappa0, nu0, S0)
#     replacing v9's independent N(m0, V0) and IW(nu0, S0).
#   - c_j single-site update is COLLAPSED: integrates out (mu_l, Sigma_l)
#     and rho, giving the (n_{l,-j}+e0) * Student-t predictive form.
#   - Adds Jain-Neal restricted-Gibbs split-merge moves on the partition
#     (n_split_merge proposals per sweep) to break label stickiness and
#     allow K_+ to explore solutions with K_+ >= 3.
#   - S0 default tightened from 0.5*I to 0.01*I (per modeling paper
#     Sec. 5.1 elicitation note: prior implied per-dim sd of ~0.033
#     matches the empirical eta within-cluster sd observed in the
#     four-layer simulation; loose S0 was the bottleneck that kept
#     v10 simulation runs at K_+ <= 3 with ARI ~ 0.46 -- tightening to
#     0.01 lifted ARI to 0.89 with K_+ = 5).
#   - Returns fmc_split_merge with split/merge attempt and acceptance
#     counts. Use the split rate (>= ~10%) and the K_+ trace (sd > 0)
#     as primary diagnostics for sampler health.
#
# IMPORTANT NOTES (carried over from v9)
#   - Coupling is strict ONE-WAY (LSIRM -> FMC). The LSIRM block in
#     my_LSIRM_FMC_v10.cpp is byte-identical to v9; clustering does NOT
#     enter the LSIRM acceptance ratios.
#   - Cluster labels in fmc_c are subject to label switching across
#     iterations. Per-iteration labels are NOT directly interpretable.
#     Use the co-cluster matrix (fmc_co_cluster) and K_+ trace
#     (fmc_K_plus) for inference. The final partition below is built
#     from the posterior similarity matrix via average-linkage hclust.
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
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v10.R"))   # joint LSIRM + PPCA-FMC v10
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
  alpha1 = 0.6, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.05, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.30,
  beta1 = 0.50, beta2 = 0.1, beta3 = 0.20, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.35, b2 = 0.20, b3 = 0.20, b4 = 0.10, b5 = 0.50,
  log_kappa = 0.30
)

# ---- FMC defaults (v10) ----
# Overfitted-prior strategy (Rousseau & Mengersen 2011): with K_star >> K_true,
# small e_0 < d_l/2 makes the posterior empty redundant clusters. The v10
# collapsed Gibbs + split-merge sampler additionally lets the chain discover
# K_+ >= 3 without having to pass through low-probability single-site steps.
#
# Diagnostic NOTE (carried from v9 + v10 simulation):
#   - PCA initialisation (section 3-bis) breaks the (Lambda=0, eta=0) trap.
#   - Empty components remain reachable thanks to the (n_{l,-j}+e0) factor in
#     the NIW-collapsed update, so e_0 in {0.05, 0.10} is the recommended range.
#   - r_fac = 5 retained from v9 to allow comparison; the v10 simulation showed
#     that smaller r_fac (e.g. 2) concentrates signal but also caps available
#     dimensions. For real data with unknown structure, r_fac = 5 is a safer
#     default; reduce only if eta posterior PC variance ratio shows a clear
#     low-rank structure.
r_fac  <- 5L
K_star <- 10L
e0     <- 0.1

# Number of Jain-Neal split-merge proposals per Gibbs sweep.
# 5 worked well in the four-layer simulation (split rate 14.9% on the
# tight-S0 setting). Smaller MIDUS chains (n=221, P~50) may get away with 3.
n_split_merge <- 5L

common_fmc_hyper <- list(
  e0            = e0,
  m0            = rep(0, r_fac),
  # NIW prior precision for mu_l: kappa0 = 1e-3 keeps Sigma_l/kappa0 wide
  # (per-dim sd ~ sqrt(S0_scale / ((nu0 - r - 1) * kappa0)) ~ 1.05 with
  # S0_scale = 0.01, nu0 = r + 10 = 15), matching v9's V0 = 9*I_r in
  # marginal magnitude.
  kappa0        = 1e-3,
  nu0           = r_fac + 10,                 # informative IW prior
  # S0 = 5e-3 I_r (sweet spot between 0.01 and 0.001 runs).
  # Comparison so far on MIDUS:
  #   S0=0.01  -> K_+=3 (median), Pr(co>0.8)=3.9%, Pr(co [0.3,0.7])=50%,
  #               silhouette 0.40, within-co 0.55-0.65 (clean partition,
  #               soft individual co-clustering).
  #   S0=0.001 -> K_+=6 (median), Pr(co>0.8)=0.5%, Pr(co [0.3,0.7])=27%,
  #               silhouette 0.22, within-co 0.36-0.51 (over-fragmented;
  #               tighter inter-cluster separation but weaker intra-cluster
  #               agreement and a singleton emerges).
  # 5e-3 keeps the prior in roughly the right magnitude (per-dim sd ~ 0.024,
  # ratio ~ 9-14x with empirical 0.0017-0.0026) without over-fragmenting.
  # Re-check the elicitation diagnostic after the run.
  S0            = 5e-3 * diag(r_fac),
  tau_lambda_sq = 4.0,                         # carried from v9 second pass
  a_eps         = 5, b_eps = 0.1,              # PPCA shared noise variance
  a_delta       = 2, b_delta = 1
)

# v10 keeps v9 second-pass MCMC settings (long chain due to LSIRM heavy
# tails). Increase burnin if K_+ trace shows transition behavior.
common_mcmc <- list(d = 2, n_iter = 120000, burnin = 40000, thin = 10)
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
# 3. Variable subsets (v6 / v7 / v9와 동일 switch)
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
# 3-bis. Smart FMC initialization (carried from v9; same rationale)
################################################################################
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
P_total_init <- ncol(cs$Y_bin) + ncol(cs$Y_con) + ncol(cs$Y_cnt) +
                ncol(cs$Y_ord1) + ncol(cs$Y_ord2)
n_init       <- nrow(cs$Y_con)

eta_init_pca <- init_eta_via_pca(cs$Y_bin, cs$Y_con, cs$Y_cnt,
                                 cs$Y_ord1, cs$Y_ord2, r_fac)
km <- kmeans(eta_init_pca, centers = K_star, nstart = 25, iter.max = 50)
c_init_km <- km$cluster

fmc_init_smart <- list(
  rho     = rep(1 / K_star, K_star),
  eta     = eta_init_pca,
  c       = c_init_km,
  mu      = km$centers,
  Sigma   = array(rep(diag(r_fac), K_star),
                  dim = c(r_fac, r_fac, K_star)),
  Lambda  = matrix(rnorm(n_init * r_fac, 0, 0.5), n_init, r_fac),
  delta   = rep(0, n_init),
  sigma_eps_sq   = 0.1,
  sigma_delta_sq = 0.1
)

cat(sprintf("[FMC init] PCA-eta scale: sd per dim = %s\n",
            paste(round(apply(eta_init_pca, 2, sd), 3), collapse = ", ")))
cat(sprintf("[FMC init] kmeans cluster sizes: %s\n",
            paste(table(c_init_km), collapse = " ")))

################################################################################
# 4. Single chain run
################################################################################
run_label     <- sprintf("v10_fmc_r%d_K%d_e%g_S0_%g_M%d",
                         r_fac, K_star, e0,
                         common_fmc_hyper$S0[1, 1], n_split_merge)
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

##############################################################################
# 4-A. v10 joint MCMC
##############################################################################
result <- lsirm_fmc_v10_cpp(
  Y_bin   = cs$Y_bin,
  Y_con   = round(cs$Y_con, 1),
  Y_cnt   = cs$Y_cnt,
  Y_ord1  = cs$Y_ord1,
  Y_ord2  = cs$Y_ord2,
  r_fac   = r_fac, K_star = K_star, e0 = e0,
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
  save_lambda_full = FALSE,
  save_delta_full  = FALSE,
  save_eta_full    = TRUE,
  compute_co_cluster_online = TRUE,
  fmc_warmup    = max(1000L, as.integer(common_mcmc$burnin / 4)),
  n_split_merge = n_split_merge,
  row_center    = TRUE,
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

sm <- result$fmc_split_merge
cat(sprintf("\n-- %s Split-merge --\n", run_label))
cat(sprintf("  split: %d / %d  (rate %.3f)\n",
            sm$split_accepts, sm$split_attempts, sm$split_rate))
cat(sprintf("  merge: %d / %d  (rate %.3f)\n",
            sm$merge_accepts, sm$merge_attempts, sm$merge_rate))
cat(sprintf("  net K_+ change (split_acc - merge_acc) = %d\n",
            sm$split_accepts - sm$merge_accepts))

##############################################################################
# 4-C. LSIRM traceplots (identical structure to v9)
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
# 4-D. FMC parameter traceplots
##############################################################################
n_save  <- nrow(result$fmc_rho)
P_total <- ncol(result$fmc_c)
item_names_full <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1, cs$col_ord2)
item_names_full <- item_names_full[seq_len(P_total)]

# K_+ trace
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_K_plus.pdf")),
    width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(result$fmc_K_plus, main = "Occupied cluster count K_+",
        ylab = expression(K["+"]^{(s)}))
abline(h = mean(result$fmc_K_plus), col = "darkgreen", lwd = 2)
dev.off()

# rho trace
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_rho.pdf")),
    width = 9, height = 8)
par(mfrow = c(ceiling(K_star / 2), 2), mar = c(3, 3, 2, 1))
for (l in 1:K_star) {
  x <- result$fmc_rho[, l]
  ts.plot(x, main = sprintf("rho[%d]", l), ylab = "")
  abline(h = c(mean(x), quantile(x, c(.025, .975))),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
}
dev.off()

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_sigma_delta_sq.pdf")),
    width = 8, height = 5)
par(mar = c(4, 4, 3, 1))
plot_trace_scalar_local(result$fmc_sigma_delta_sq, main = "FMC sigma_delta_sq")
dev.off()

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_mu.pdf")),
    width = 10, height = 12)
par(mfrow = c(K_star, r_fac), mar = c(3, 3, 2, 1))
for (l in 1:K_star) for (jj in 1:r_fac) {
  x <- result$fmc_mu[l, jj, ]
  ts.plot(x, main = sprintf("mu[l=%d, d=%d]", l, jj), ylab = "")
  abline(h = mean(x), col = "darkgreen", lwd = 2)
}
dev.off()

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_logdetSigma.pdf")),
    width = 9, height = 8)
par(mfrow = c(ceiling(K_star / 2), 2), mar = c(3, 3, 2, 1))
for (l in 1:K_star) {
  ld <- vapply(seq_len(n_save),
               function(s) {
                 ev <- eigen(result$fmc_Sigma[, , l, s], symmetric = TRUE,
                             only.values = TRUE)$values
                 ev[ev < 1e-12] <- 1e-12
                 sum(log(ev))
               }, numeric(1))
  ts.plot(ld, main = sprintf("log|Sigma_l| (l=%d)", l), ylab = "")
  abline(h = mean(ld), col = "darkgreen", lwd = 2)
}
dev.off()

n_l_trace <- t(apply(result$fmc_c, 1, function(v) tabulate(v, nbins = K_star)))
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_cluster_sizes.pdf")),
    width = 10, height = 5)
par(mar = c(4, 4, 3, 1))
matplot(n_l_trace, type = "l", lty = 1,
        col = expand_pal(K_star, cluster_pal),
        xlab = "saved iter", ylab = expression(n[l]^{(s)}),
        main = "Item cluster sizes (c)")
legend("topright", legend = paste0("l=", 1:K_star),
       col = expand_pal(K_star, cluster_pal), lty = 1, bty = "n", cex = 0.75)
dev.off()

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_membership_heatmap_c.pdf")),
    width = 10, height = 7)
par(mar = c(7, 4, 3, 1))
ord_w <- order(item_names_full)
image(seq_len(n_save), seq_len(P_total),
      result$fmc_c[, ord_w, drop = FALSE],
      col = expand_pal(K_star, cluster_pal),
      xlab = "MCMC iteration (saved)", ylab = "",
      main = "Item cluster membership trace  (c_j over iterations; subject to label switching)",
      axes = FALSE)
axis(1)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_w],
     las = 2, cex.axis = 0.55)
box()
legend("topright", legend = paste0("l=", 1:K_star),
       fill = expand_pal(K_star, cluster_pal), bty = "n", cex = 0.75)
dev.off()

##############################################################################
# 4-E. Posterior summaries — co-cluster + final partition + biplot
##############################################################################
co_cluster <- result$fmc_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

item_cluster_mode <- apply(result$fmc_c, 2, mode_label)

median_K_plus <- max(2, round(median(result$fmc_K_plus)))
hc_co <- hclust(as.dist(1 - co_cluster), method = "average")
final_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_co_cluster.pdf")),
    width = 9, height = 8)
ord_c <- order(item_cluster_mode)
par(mar = c(7, 7, 3, 1))
image(seq_along(item_names_full), seq_along(item_names_full),
      co_cluster[ord_c, ord_c],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("FMC item co-clustering  (r=%d, K*=%d, e0=%g)",
                     r_fac, K_star, e0))
axis(1, at = seq_along(item_names_full), labels = item_names_full[ord_c],
     las = 2, cex.axis = 0.6)
axis(2, at = seq_along(item_names_full), labels = item_names_full[ord_c],
     las = 2, cex.axis = 0.6)
box()
dev.off()

eta_pm <- result$fmc_eta_postmean
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_eta_postmean.pdf")),
    width = 9, height = 8)
par(mar = c(4, 4, 3, 1))
if (r_fac == 2) {
  pal_use <- expand_pal(K_star, cluster_pal)
  plot(eta_pm[, 1], eta_pm[, 2], pch = 19, cex = 1.4,
       col = pal_use[item_cluster_mode],
       xlab = "eta dim 1", ylab = "eta dim 2",
       main = sprintf("FMC eta posterior mean  (coloured by mode cluster)"))
  text(eta_pm[, 1], eta_pm[, 2], labels = item_names_full,
       pos = 4, cex = 0.55)
} else {
  d_show <- min(3, r_fac)
  pairs(eta_pm[, 1:d_show],
        col = expand_pal(K_star, cluster_pal)[item_cluster_mode],
        pch = 19, cex = 0.9,
        labels = paste0("eta", 1:d_show),
        main = sprintf("FMC eta posterior mean (first %d dims)", d_show))
}
dev.off()

##############################################################################
# 4-E (cont). Biplot
##############################################################################
A_hat_pm <- apply(result$a, c(2, 3), mean)
b_arrs <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
  if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
    return(matrix(0, 0, common_mcmc$d))
  apply(arr, c(2, 3), mean)
}))
rownames(B_hat_pm) <- item_names_full

make_fmc_biplot <- function(A_hat, B_hat, item_partition, item_names,
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
  make_fmc_biplot(
    A_hat = A_hat_pm, B_hat = B_hat_pm,
    item_partition = final_partition,
    item_names     = item_names_full,
    title = sprintf("Joint LSIRM+PPCA-FMC v10  (r=%d, K*=%d, e0=%g, mean K_+=%.1f, split=%.2f)  |  %s",
                    r_fac, K_star, e0,
                    mean(result$fmc_K_plus),
                    sm$split_rate,
                    cs$label),
    filename = paste0(fmc_prefix, "_biplot.pdf"),
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

write.csv(data.frame(item            = item_names_full,
                     fmc_mode_cluster = item_cluster_mode,
                     fmc_partition    = final_partition),
          file.path(case_plot_dir, paste0(fmc_prefix, "_item_clusters.csv")),
          row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(case_plot_dir, paste0(fmc_prefix, "_co_cluster.csv")))

# NOTE: K_+ summary CSV (incl. silhouette) is written in section 4-I,
# after silhouette is computed in section 4-H.

write.csv(data.frame(respondent     = seq_along(result$fmc_delta_postmean),
                     delta_postmean = result$fmc_delta_postmean),
          file.path(case_plot_dir, paste0(fmc_prefix, "_resp_postmean.csv")),
          row.names = FALSE)

sigma_eps_sq_trace <- as.numeric(result$fmc_sigma_eps_sq)
write.csv(data.frame(
            mean   = mean(sigma_eps_sq_trace),
            median = median(sigma_eps_sq_trace),
            sd     = sd(sigma_eps_sq_trace),
            min    = min(sigma_eps_sq_trace),
            max    = max(sigma_eps_sq_trace),
            postmean = result$fmc_sigma_eps_sq_postmean
          ),
          file.path(case_plot_dir, paste0(fmc_prefix, "_sigma_eps_sq_summary.csv")),
          row.names = FALSE)
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_sigma_eps_sq.pdf")),
    width = 9, height = 5)
par(mar = c(4, 4, 3, 1))
ts.plot(sigma_eps_sq_trace, main = "PPCA shared noise variance sigma_eps_sq",
        ylab = expression(sigma[epsilon]^2))
abline(h = c(mean(sigma_eps_sq_trace),
             quantile(sigma_eps_sq_trace, c(0.025, 0.975))),
       col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
dev.off()

write.csv(round(result$fmc_eta_postmean, 4),
          file.path(case_plot_dir, paste0(fmc_prefix, "_eta_postmean.csv")))

write.csv(round(result$fmc_Lambda_postmean, 4),
          file.path(case_plot_dir, paste0(fmc_prefix, "_Lambda_postmean.csv")))

##############################################################################
# 4-G. v10-specific: S0 elicitation diagnostic
#
# Compute empirical within-cluster sd of eta_pm using modal labels and
# compare to the prior-implied sd. If the prior is much looser than the
# empirical signal, suggest tightening S0 in a follow-up run.
##############################################################################
empirical_within_sd <- vapply(seq_len(r_fac), function(q) {
  resids <- numeric(0)
  for (l in unique(item_cluster_mode)) {
    idx <- which(item_cluster_mode == l)
    if (length(idx) < 2) next
    resids <- c(resids, eta_pm[idx, q] - mean(eta_pm[idx, q]))
  }
  if (length(resids) < 2) return(NA_real_)
  sqrt(mean(resids^2))
}, numeric(1))

S0_scale_used <- common_fmc_hyper$S0[1, 1]
prior_implied_sd <- sqrt(S0_scale_used / (common_fmc_hyper$nu0 - r_fac - 1))
ratio <- prior_implied_sd / empirical_within_sd

cat(sprintf("\n-- v10 S0 elicitation diagnostic --\n"))
cat(sprintf("  S0 scale used: %.4f\n", S0_scale_used))
cat(sprintf("  prior implied per-dim sd = sqrt(S0/(nu0-r-1)) = %.4f\n",
            prior_implied_sd))
cat(sprintf("  empirical within-cluster sd of eta_pm per dim:\n"))
print(round(empirical_within_sd, 4))
cat(sprintf("  ratio (prior implied / empirical) per dim:\n"))
print(round(ratio, 2))
if (any(ratio > 5, na.rm = TRUE)) {
  cat("  WARNING: prior >> empirical (ratio > 5). Consider tightening S0:\n")
  suggested <- max(empirical_within_sd, na.rm = TRUE)^2 * (common_fmc_hyper$nu0 - r_fac - 1)
  cat(sprintf("    suggested S0_scale ~ max(emp_sd)^2 * (nu0-r-1) = %.4f\n", suggested))
}

write.csv(data.frame(
            dim                = seq_len(r_fac),
            empirical_within_sd = empirical_within_sd,
            prior_implied_sd    = prior_implied_sd,
            ratio               = ratio
          ),
          file.path(case_plot_dir, paste0(fmc_prefix, "_S0_elicitation.csv")),
          row.names = FALSE)

##############################################################################
# 4-H. Silhouette diagnostic (eta-based, NOT co-cluster-based)
#
# Standard silhouette uses Euclidean distance between item factor scores
# (eta_pm in R^r) -- this is the natural representation space of items.
# We avoid the common pitfall of computing silhouette on (1 - co_cluster):
# that conflates "do items end up co-clustered?" with "are they close in
# representation space?", which is circular for a clustering posterior.
#
# Output:
#   - per-cluster silhouette mean / median / min
#   - overall mean silhouette and fraction with negative width
#   - per-item width (CSV) so individual mis-clustered items can be flagged
#
# Interpretation guide (Kaufman & Rousseeuw 1990 thresholds):
#   sil > 0.5  : strong structure in this cluster
#   0.25-0.5   : reasonable structure
#   < 0.25     : weak; cluster may be artifactual
#   < 0        : the item is closer to a different cluster than its own
##############################################################################
sil_obj <- cluster::silhouette(final_partition, dist(eta_pm))
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

  cat(sprintf("\n-- v10 silhouette diagnostic (eta-based) --\n"))
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

  # Per-item silhouette CSV
  sil_csv <- sil_mat[, c("item", "cluster", "neighbor", "sil_width")]
  sil_csv$sil_width <- round(sil_csv$sil_width, 4)
  write.csv(sil_csv,
            file.path(case_plot_dir,
                      paste0(fmc_prefix, "_silhouette_per_item.csv")),
            row.names = FALSE)

  # Per-cluster silhouette PDF
  pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_silhouette.pdf")),
      width = 9, height = 6)
  par(mar = c(4, 4, 3, 1))
  plot(sil_obj, main = sprintf("Silhouette (eta-based)  k=%d, mean=%.3f",
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
# 4-H-bis. Co-clustering quality diagnostics
#
# Two complementary measures:
#   (1) within-pair confidence ratio:
#         Pr(C_jk > 0.8) / (frac. within-cluster pairs in final partition)
#       Compares the fraction of pairs that the posterior puts in the same
#       cluster with high confidence to the fraction that the imposed
#       partition puts together. ratio < 0.3 = weak; > 0.7 = strong.
#
#   (2) PEAR (Posterior Expected Adjusted Rand index, Fritsch & Ickstadt
#       2009): mean ARI between sampled partitions across MCMC iterations.
#       High PEAR (> 0.5) means the posterior is concentrated on a single
#       partition shape, so even moderately fuzzy individual co-cluster
#       values can still translate into a stable partition.
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

# uniform-K baseline: expected co-cluster mean if pairs were placed at
# random into clusters of the imposed sizes.
baseline_mean <- sum((table(final_partition) / P_total)^2) - 1 / P_total
od_mean       <- mean(cm_od)

cat(sprintf("\n-- v10 co-clustering quality --\n"))
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
n_save_local <- nrow(result$fmc_c)
ii <- sample.int(n_save_local, n_pear_pairs, replace = TRUE)
jj <- sample.int(n_save_local, n_pear_pairs, replace = TRUE)
ok <- ii != jj
pear_vals <- mapply(function(i, j) {
  .adj_rand(result$fmc_c[i, ], result$fmc_c[j, ])
}, ii[ok], jj[ok])
pear_mean <- mean(pear_vals, na.rm = TRUE)
pear_sd   <- sd(pear_vals, na.rm = TRUE)

cat(sprintf("\n-- v10 PEAR (posterior expected adjusted Rand) --\n"))
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

# Combined CSV
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
                    paste0(fmc_prefix, "_coclustering_quality.csv")),
          row.names = FALSE)

##############################################################################
# 4-I. Final summary (K_+ summary CSV with silhouette + console print)
##############################################################################
write.csv(data.frame(
            mean   = mean(result$fmc_K_plus),
            median = median(result$fmc_K_plus),
            sd     = sd(result$fmc_K_plus),
            min    = min(result$fmc_K_plus),
            max    = max(result$fmc_K_plus),
            K_star = K_star,
            n_save = n_save,
            split_attempts = sm$split_attempts,
            split_accepts  = sm$split_accepts,
            split_rate     = sm$split_rate,
            merge_attempts = sm$merge_attempts,
            merge_accepts  = sm$merge_accepts,
            merge_rate     = sm$merge_rate,
            sil_mean          = sil_overall_mean,
            sil_median        = sil_overall_median,
            sil_frac_neg      = sil_frac_neg,
            within_pair_ratio = within_pair_ratio,
            pr_co_ambiguous   = pr_amb,
            pear_mean         = pear_mean
          ),
          file.path(case_plot_dir, paste0(fmc_prefix, "_K_plus_summary.csv")),
          row.names = FALSE)

cat(sprintf("\n=== v10 FMC summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(result$fmc_K_plus), median(result$fmc_K_plus),
            sd(result$fmc_K_plus)))
cat(sprintf("  silhouette (eta-based): mean=%.3f, frac<0=%.3f\n",
            sil_overall_mean, sil_frac_neg))
cat(sprintf("  co-clustering: within-pair ratio=%.3f (%s), ambig=%.3f, PEAR=%.3f\n",
            within_pair_ratio, ratio_quality, pr_amb, pear_mean))
cat(sprintf("  split rate = %.3f, merge rate = %.3f\n",
            sm$split_rate, sm$merge_rate))

cat(sprintf("\n  Final partition (cutree on 1 - PSM, k=%d):\n",
            length(unique(final_partition))))
print(table(final_partition))
cat(sprintf("\n  Mode-cluster table (subject to label switching):\n"))
print(table(item_cluster_mode))
cat(sprintf("\n-> v10 plots & artifacts saved to: %s\n", case_plot_dir))
