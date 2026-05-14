rm(list = ls())

library(dplyr)
library(purrr)
library(cluster)   # silhouette()

################################################################################
# v12_skewt: v11t_stab + SDB (Sahu-Dey-Branco) skew-t component family
#
#   Same hyperparameters as MIDUS_5layered_result_v11t_stab.R / v11tm
#   (K_star=6, kappa0=1.0, nu0=12, S0_init_scale=1, nu_t=3, nu2=20,
#   n_iter=30000, burnin=10000, thin=5).
#
#   MODEL CHANGE: each cluster's component goes from elliptical Student-t
#       b_q | c=l ~ t_nu(mu_l, Sigma_l)            (v11t/v11tm)
#   to SDB skew-t
#       b_q | c=l ~ SkewT_nu(mu_l, Sigma_l, Delta_l)   (v12_skewt)
#   represented hierarchically as
#       w_q ~ Gamma(nu_t/2, nu_t/2),
#       U_q | w_q ~ N_+(0, 1/w_q),
#       b_q | c=l, w_q, U_q ~ N_d(mu_l + Delta_l U_q, Sigma_l / w_q).
#
#   Delta_l in R^d is the per-cluster SKEW DIRECTION.  Delta_l = 0
#   recovers v11t exactly.  Prior: Delta_l ~ N_d(0, tau_Delta^2 I);
#   tau_Delta = 1.0 default.
#
#   Conditional-b kernel only (no marginal_b in v12).  Compare to
#   v11t/v11tm to assess whether allowing cluster shape skew reduces
#   LSIRM-geometry compression and improves cluster recovery in
#   asymmetric layers (cognition counts, inflammation markers).
################################################################################
# v11t: Joint multilayered LSIRM + LATENT-POSITION STUDENT-T MIXTURE
#       clustering on MIDUS
#
# DIFFERENCES FROM v11_looseSigma (the lock-in 0506 reference):
#   - Replaces the Gaussian within-cluster prior
#       b_q | c_q = l ~ N_d(mu_l, Sigma_l)
#     with a Student-t prior (Gaussian scale-mixture form, Lange-Little-
#     Taylor 1989):
#       w_q ~ Gamma(nu_t/2, nu_t/2)
#       b_q | c_q = l, w_q ~ N_d(mu_l, Sigma_l / w_q)
#     so that marginal b_q ~ t_{nu_t}(mu_l, Sigma_l).  nu_t = 4 fixed.
#   - Heavy tails tolerate boundary items without strong pull toward
#     cluster mean -- LSIRM biplot geometry is less compressed than
#     under v11 looseSigma (which had b items pulled toward mu_c by
#     ~6x stronger precision than LSIRM-only prior).
#   - Removes b_prior_inflation knob (sampler-level non-coherent hack
#     in v11) -- v11t replaces it with a model-coherent Student-t.
#   - NIW posterior uses WEIGHTED sufficient statistics: kappa_n
#     includes sum w_q, weighted scatter S_w in IW posterior, etc.
#     (See `niw_posterior_from_indices_weighted` in v11t cpp.)
#   - Per-sweep block adds w_q | b, c, mu, Sigma ~ Gamma full-conditional.
#
# All other settings inherited from v11_looseSigma:
#   - Hyperparameters: S0_init_scale = 0.434, nu0 = 10, kappa0 = 0.1
#     (target E[Sigma_l] = 0.062 * I preserved, data-driven IW)
#   - LSIRM proposal SDs unchanged (0503 v10 lock-in)
#   - n_split_merge = 100, MCMC 80k / 20k / 10
#
# Validation (sim 4-layer): ARI hclust 0.903 (vs v11 0.860), Dahl/Binder/VI
# 0.872 (vs v11 0.878).  Split rate 0.042 (vs v11 ~0.008, 5x improvement).
# w_q diagnostic: mean E[w_q] = 1.014, 40% downweighted boundary items.
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
data_dir <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data"
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
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v12_skewt.R"))  # joint LSIRM + SDB skew-t mixture v12
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

# LSIRM proposal SDs.
# 0507 v11t MIDUS retune: previous SDs were inherited from v10's lock-in,
# calibrated for a chain where b had a fully independent N(0, I_d) prior.
# Both v11 looseSigma (Gaussian mixture) and v11t (Student-t mixture) gave
# acceptance ~0.13-0.23 for b1..b4 on MIDUS (n=221, P=68) -- proposal step
# size is 2-3x oversized given the tighter cluster-mixture posterior.
#   beta2: 0.40 -> 0.18  (acceptance 0.099 -> target ~0.30)
#   b1   : 0.50 -> 0.30  (acceptance 0.147 -> target ~0.32)
#   b2   : 0.40 -> 0.27  (acceptance 0.205 -> target ~0.32)
#   b3   : 0.50 -> 0.35  (acceptance 0.230 -> target ~0.33)
#   b4   : 0.40 -> 0.27  (acceptance 0.137 -> target ~0.32)
# Other SDs unchanged (acceptance already in 0.30-0.55 healthy range, see
# v11t MIDUS run 0507 acceptance table).  Verify by re-inspecting acceptance
# after this run; nudge further if still off-target.
common_lsirm_prop_sd <- list(
  alpha1 = 0.78, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.60, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.15, log_gamma3 = 0.15,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.5,
  beta1 = 0.50, beta2 = 0.18, beta3 = 0.20, beta4 = 0.50, beta5 = 0.30,
  b1 = 0.30, b2 = 0.27, b3 = 0.35, b4 = 0.27, b5 = 0.50,
  log_kappa = 0.30
)

# ---- FMC defaults (v11: NIW in R^d, no PPCA layer) ----
#
# Mixture parameters now live in R^d (d = 2), the LSIRM latent space.  The
# NIW prior on (mu_l, Sigma_l) is augmented with a Wishart hyperprior on
# S0 so that the cluster-covariance scale is data-driven rather than
# manually calibrated.
#
# Defaults rationale:
#   - K_star = 10 with overfitted-prior strategy (Rousseau & Mengersen
#     2011): e0 < d_l/2 with d_l = d + d(d+1)/2 = 5 for d=2, so any
#     e0 < 2.5 asymptotically empties redundant clusters.  e0 = 0.05 is
#     well within the safe range.
#   - kappa0 = 1 (0505 update from 1e-3): the previous very-weak-prior
#     setting let empty-slot mu_l draws explode (mu_l|empty ~ N(0, S0/kappa0)
#     with kappa0=1e-3 -> sd ~ sqrt(1000) per dim).  Simulation showed mu
#     trace ±60 spikes that disappeared at kappa0=1, while ARI rose from
#     0.82 to 0.88 and split/merge accept rates jumped 4-5x.
#   - nu0 = d + 10 = 12 gives moderately informative IW prior.  Lower
#     nu0 (e.g. d + 5) allows more anisotropic Sigma_l but biases the
#     marginal scale; the simulation showed that lowering nu0 alone is
#     not the right knob, see modeling paper Sec. 11.
#   - S0_init = 0.5 * diag(d): starting point for the Wishart hyperprior.
#     The hyperprior makes the EXACT value less critical because S0 is
#     resampled each iteration -- this just sets the prior expectation
#     E[S0] = S0_init under the default Lambda_S0 = S0_init / nu_S0.
#   - b_prior_inflation = 1: full joint coupling.  Set to 5 if a follow-up
#     run shows that b posterior is over-shrunk to cluster centers (visible
#     as a marked S-curve in the distance-recovery scatter or as items
#     visually piled on cluster centers in the biplot).
d      <- 2L
K_star <- 10L                       # 0509 stabilization: was 10; reduces label-switching mode count from K_star! ~ 3.6M to 720
# v11tm 0511 e0 sweep: 0.05 단독 marginal_b 가 MIDUS 에서 single-cluster collapse
# (Dahl K=1, ARI vs v11t = 0).  e0 ↑ 로 빈 cluster 가 P(c_q=l) ∝ (n_{l,-q}+e0)
# weight 를 더 받아 dominant cluster 쪽 positive feedback 약화.  Rousseau-
# Mengersen safe range (e0 < d_l/2 = 2.5) 안.  0.1 로 약하게 풀기.
e0     <- 0.1

# Number of Jain-Neal split-merge proposals per Gibbs sweep.  Higher than
# the v10 default of 5 because v11's per-attempt acceptance is lower under
# the joint coupling (b moves via random-walk MH, not Gibbs).  100 is the
# Option-A brute-force boost that gave ~6x more accepted moves per sweep
# on the four-layer simulation.
n_split_merge <- 100L

# 0506 LOOSE-SIGMA variant (option A in history/0506_history.md):
#   Hold the prior MEAN of Sigma_l at the previous lock-in target
#   (E[Sigma_l] = 0.062 * I, per-dim sd ~ 0.25, derived from v10 biplot's
#    visible cluster radius ~0.5 = 2-sigma) but RELAX the IW concentration
#   so that the data is allowed to drive Sigma_l shape and scale.
#
#   nu0 = 10 (loose; was 100 -- nu0=100 effectively pinned Sigma_l).
#   S0  = 0.062 * (nu0 - d - 1) * I = 0.062 * 7 * I ~= 0.434 * I
#         (so that S0 / (nu0 - d - 1) is unchanged at 0.062 * I).
#
# Risk being checked by this run: the runaway shrinkage feedback that
# motivated nu0=100 in the first place.  Mechanism:
#     prior pulls b -> mu_c => empirical within-cluster var shrinks
#     -> NIW posterior Sigma_l shrinks (S_n / (nu_n - d - 1))
#     -> b prior precision grows
#     -> tighter pull, runaway, items pile on mu_c.
# Diagnostic to inspect post-run:
#   - trace of mean(diag(Sigma_l)) per occupied l : should NOT collapse to ~0
#   - b posterior-mean scatter            : items should NOT pile on mu_c
#   - within-cluster sd of b_pm           : should be a meaningful fraction of mu_l span
# kappa0 unchanged at 0.1 (mu_l|Sigma_l ~ N(0, Sigma_l/0.1) prior allows
# cluster-center span ~ +-1.5 at the target Sigma_l).
S0_init_scale <- 0.5
common_fmc_hyper <- list(
  e0     = e0,
  m0     = rep(0, d),
  kappa0 = 0.1,                          # 0509 stabilization: was 0.1 (loose); tightened to reduce mu_l wandering
  nu0    = 10,                           # 100 -> 10 (loose, data-driven Sigma_l)
  S0     = S0_init_scale * diag(d),      # E[Sigma_l] = 0.434/7 ~= 0.062 * I (preserved)
  nu_S0  = NA                            # Wishart hyperprior on S0 still off
)

# v11t Student-t mixture df.  Default 4 (matches LSIRM continuous-layer
# robust likelihood).  Heavy tail allows LSIRM biplot items at cluster
# boundary to escape strong pull toward mu_c without giving up cluster
# identification at the center.  This REPLACES the v11 b_prior_inflation
# knob (which was a non-coherent sampler-level hack); v11t is a fully
# Bayesian model change.
# Validated range on 4-layer simulation: nu_t in [3, 10] safe.  Lower
# nu_t -> heavier tails / weaker pull / softer clusters.
nu_t <- 4

# v12 SDB skew-t: prior sd for cluster skew direction Delta_l ~ N_d(0, tau_Delta^2 I).
# tau_Delta = 1.0 -> moderate skew prior (matches typical LSIRM latent scale).
# tau_Delta -> 0  -> collapses to v11t symmetric t mixture.
# Recommend keeping tau_Delta in [0.5, 2.0] for first MIDUS runs.
tau_Delta <- 1.0

# Long chain because the LSIRM block has heavy tails; the v11 joint
# coupling inherits the same regime.
common_mcmc <- list(d = d, n_iter = 100000, burnin = 20000, thin = 10)   # 0509 stabilization preview: was 120k/30k/10
nu2 <- 4      # 0509 stabilization: was 4 (heavy-tail t for outlier robustness); raised to make continuous-layer likelihood near-Gaussian -> tighter b2 posterior, less trace multimodality (trade-off: less robust to inflammation outliers)

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
# 3-bis. Smart FMC initialisation
#
# In v11 the mixture lives directly in R^d (d = 2), the LSIRM latent space.
# The cluster init only needs P_total labels in {1,..,K_star}.  As a cheap
# proxy for b_j we use the first d PCs of the centered/log-transformed
# response matrix (transposed so items are rows); k-means on that proxy
# gives a non-degenerate starting partition.  The actual b_j positions
# are sampled inside the LSIRM block.
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
set.seed(0502)  # reproducibility for FMC k-means init AND MCMC chain
km <- kmeans(b_init_proxy, centers = K_star, nstart = 25, iter.max = 50)
c_init_km <- km$cluster

# Sigma init = identity, matching v10's diag(r_fac).  A loose / non-
# informative init lets the NIW posterior draws determine cluster shape
# from data, rather than starting from an artificially tight covariance
# that biases the early-iteration b MH ratio.
fmc_init_smart <- list(
  rho   = rep(1 / K_star, K_star),
  c     = c_init_km,
  mu    = km$centers,
  Sigma = array(rep(diag(d), K_star), dim = c(d, d, K_star))
)

cat(sprintf("[FMC init] PCA-b-proxy scale: sd per dim = %s\n",
            paste(round(apply(b_init_proxy, 2, sd), 3), collapse = ", ")))
cat(sprintf("[FMC init] kmeans cluster sizes: %s\n",
            paste(table(c_init_km), collapse = " ")))

################################################################################
# 4. Single chain run
################################################################################
run_label     <- sprintf("v12_skewt_d%d_K%d_e%g_S0init_%g_nu%d_kap%g_M%d_nut%g_tauD%g_nu2%g",
                         d, K_star, e0,
                         S0_init_scale,
                         common_fmc_hyper$nu0,
                         common_fmc_hyper$kappa0,
                         n_split_merge,
                         nu_t, tau_Delta, nu2)
case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", run_label))
if (!dir.exists(case_plot_dir)) {
  ok <- dir.create(case_plot_dir, recursive = TRUE, showWarnings = TRUE)
  if (!isTRUE(ok) || !dir.exists(case_plot_dir))
    stop("Failed to create case_plot_dir: ", case_plot_dir)
}
cat(sprintf("[case_plot_dir] %s  (exists=%s)\n",
            case_plot_dir, dir.exists(case_plot_dir)))
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
# 4-A. v12 joint MCMC (SDB skew-t mixture)
##############################################################################
result <- lsirm_fmc_v12_skewt_cpp(
  Y_bin   = cs$Y_bin,
  Y_con   = round(cs$Y_con, 1),
  Y_cnt   = cs$Y_cnt,
  Y_ord1  = cs$Y_ord1,
  Y_ord2  = cs$Y_ord2,
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
  tau_Delta     = tau_Delta,
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
# 4-D. FMC parameter traceplots
#
# v11: only mu_l, Sigma_l (via log|Sigma|), rho, K_+, S0 (if hyperprior on),
# cluster sizes.  No eta, Lambda, delta, sigma_eps_sq, sigma_delta_sq.
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

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_mu.pdf")),
    width = 10, height = 12)
par(mfrow = c(K_star, d), mar = c(3, 3, 2, 1))
for (l in 1:K_star) for (jj in 1:d) {
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

# v11: S0 hyperprior trace (cluster scale evolves data-driven)
if (!is.null(result$fmc_S0)) {
  pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_trace_S0.pdf")),
      width = 10, height = 6)
  par(mfrow = c(d, d), mar = c(3, 3, 2, 1))
  for (i in 1:d) for (j in 1:d) {
    x <- result$fmc_S0[i, j, ]
    ts.plot(x, main = sprintf("S0[%d,%d]", i, j), ylab = "")
    abline(h = mean(x), col = "darkgreen", lwd = 2)
  }
  dev.off()
}

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

##############################################################################
# 4-E. Posterior summaries — co-cluster + final partition + biplot
##############################################################################
co_cluster <- result$fmc_co_cluster
rownames(co_cluster) <- colnames(co_cluster) <- item_names_full

item_cluster_mode <- apply(result$fmc_c, 2, mode_label)

median_K_plus <- max(2, round(median(result$fmc_K_plus)))
hc_co <- hclust(as.dist(1 - co_cluster), method = "average")
final_partition <- cutree(hc_co, k = min(median_K_plus, P_total - 1))

ord_hc <- hc_co$order

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_membership_heatmap_c.pdf")),
    width = 10, height = 7)
par(mar = c(7, 4, 3, 1))
image(seq_len(n_save), seq_len(P_total),
      result$fmc_c[, ord_hc, drop = FALSE],
      col = expand_pal(K_star, cluster_pal),
      xlab = "MCMC iteration (saved)", ylab = "",
      main = "Item cluster membership trace  (c_q over iterations; ordered by hclust on 1-PSM; subject to label switching)",
      axes = FALSE)
axis(1)
axis(2, at = seq_len(P_total), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.55)
box()
legend("topright", legend = paste0("l=", 1:K_star),
       fill = expand_pal(K_star, cluster_pal), bty = "n", cex = 0.75)
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
dahl <- dahl_partition_fun(result$fmc_c, co_cluster)
P5_dahl <- dahl$partition

# v11tm 0511: 사용자 요청으로 RDS save 비활성화했지만, post-processing 중 e0=1
# 에서 silhouette degenerate edge case 로 죽어서 chain 을 잃는 문제 발생.
# Defensive 차원에서 saveRDS 복구 — 사용자가 manual 로 삭제 가능.
saveRDS(result, file.path(case_plot_dir, paste0(prefix, "_result.rds")))

if (requireNamespace("mcclust", quietly = TRUE) &&
    requireNamespace("mcclust.ext", quietly = TRUE)) {

  loss_binder <- function(c_, psm) mcclust::binder(c_, psm)
  loss_vi     <- function(c_, psm) mcclust.ext::VI.lb(matrix(c_, nrow = 1), psm)[1]
  k_unique    <- function(c_) length(unique(c_))

  # 0506: include.greedy = FALSE to bypass mcclust.ext's local_explore
  # subscript-out-of-bounds bug under degenerate co-cluster matrices.
  # tryCatch ensures any other failure mode also degrades gracefully:
  # the offending partition falls back to P0 (avg-link cut).
  bin_uncon <- tryCatch(
    mcclust.ext::minbinder.ext(
      co_cluster, cls.draw = result$fmc_c,
      method = "all", include.lg = TRUE, include.greedy = FALSE,
      suppress.comment = TRUE
    ),
    error = function(e) { cat(sprintf("[alt] minbinder.ext failed: %s\n", conditionMessage(e))); NULL }
  )
  if (!is.null(bin_uncon)) {
    P1 <- as.integer(bin_uncon$cl[which.min(bin_uncon$value), ])
    P1_winner <- rownames(bin_uncon$cl)[which.min(bin_uncon$value)]
  } else {
    P1 <- as.integer(final_partition); P1_winner <- "FALLBACK_P0"
  }

  bin_draws <- tryCatch(
    mcclust::minbinder(
      co_cluster, cls.draw = result$fmc_c, method = "draws"
    ),
    error = function(e) { cat(sprintf("[alt] minbinder(draws) failed: %s\n", conditionMessage(e))); NULL }
  )
  if (!is.null(bin_draws)) {
    P3 <- as.integer(if (is.matrix(bin_draws$cl)) bin_draws$cl[1, ] else bin_draws$cl)
  } else {
    P3 <- as.integer(final_partition)
  }

  vi_uncon <- tryCatch(
    mcclust.ext::minVI(
      co_cluster, cls.draw = result$fmc_c,
      method = "all", include.greedy = FALSE, suppress.comment = TRUE
    ),
    error = function(e) { cat(sprintf("[alt] minVI failed: %s\n", conditionMessage(e))); NULL }
  )
  if (!is.null(vi_uncon)) {
    P4 <- as.integer(vi_uncon$cl[which.min(vi_uncon$value), ])
    P4_winner <- rownames(vi_uncon$cl)[which.min(vi_uncon$value)]
  } else {
    P4 <- as.integer(final_partition); P4_winner <- "FALLBACK_P0"
  }

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
    s <- tryCatch(cluster::silhouette(c_, d_b_for_sil),
                  error = function(e) NA)
    # silhouette() returns NA (scalar) in degenerate edge cases
    # (e.g., when all clusters except one are singletons).
    if (!is.matrix(s)) return(NA_real_)
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

  cat("\n-- v11 alternative partitions (P0 still used downstream) --\n")
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
                      paste0(fmc_prefix, "_alt_partitions.csv")),
            row.names = FALSE)
  write.csv(alt_summary,
            file.path(case_plot_dir,
                      paste0(fmc_prefix, "_alt_partition_summary.csv")),
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
                      paste0(fmc_prefix, "_alt_partitions.csv")),
            row.names = FALSE)
}

pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_co_cluster.pdf")),
    width = 9, height = 8)
par(mar = c(7, 7, 3, 1))
image(seq_along(item_names_full), seq_along(item_names_full),
      co_cluster[ord_hc, ord_hc],
      col  = colorRampPalette(c("white", "steelblue"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("FMC item co-clustering  (d=%d, K*=%d, e0=%g; ordered by hclust on 1-PSM)",
                     d, K_star, e0))
axis(1, at = seq_along(item_names_full), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.6)
axis(2, at = seq_along(item_names_full), labels = item_names_full[ord_hc],
     las = 2, cex.axis = 0.6)
box()
dev.off()

# v11: b posterior-mean scatter (replaces v10's eta posterior-mean plot)
pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_b_postmean.pdf")),
    width = 9, height = 8)
par(mar = c(4, 4, 3, 1))
pal_use <- expand_pal(K_star, cluster_pal)
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
    title = sprintf("Joint LSIRM + Student-t mixture v11t  (d=%d, K*=%d, e0=%g, mean K_+=%.1f, split=%.2f, nu_t=%g)  |  %s",
                    d, K_star, e0,
                    mean(result$fmc_K_plus),
                    sm$split_rate,
                    nu_t,
                    cs$label),
    filename = paste0(fmc_prefix, "_biplot.pdf"),
    plot_dir = case_plot_dir,
    pal      = cluster_pal
  )

  # ---- v11tm 0511: per-partition biplots (P0/P1/P3/P4/P5) ----
  # In-memory generation right after the default biplot, so no RDS round-trip
  # is needed.  alt_list is built only when mcclust + mcclust.ext succeed; we
  # fall back to {P0, P5_Dahl} otherwise.
  alt_biplot_list <- if (exists("alt_list") && is.list(alt_list)) {
    lapply(alt_list, function(x) as.integer(x$c))
  } else {
    list(P0_prev_avgcut    = as.integer(final_partition),
         P5_Dahl_drawsOnly = as.integer(P5_dahl))
  }
  short_lbl <- c(
    P0_prev_avgcut             = "P0 avg-link cut",
    P1_minBinder_unconstrained = "P1 minBinder (unconstrained)",
    P3_minBinder_drawsOnly     = "P3 minBinder (draws only)",
    P4_minVI_unconstrained     = "P4 minVI (unconstrained)",
    P5_Dahl_drawsOnly          = "P5 Dahl (draws only)"
  )
  for (pc in names(alt_biplot_list)) {
    part <- alt_biplot_list[[pc]]
    pretty <- if (pc %in% names(short_lbl)) short_lbl[[pc]] else pc
    fname  <- paste0(fmc_prefix, "_biplot_", pc, ".pdf")
    title  <- sprintf("v11tm biplot — %s  (K=%d; d=%d, K*=%d, e0=%g, nu_t=%g, mean K_+=%.1f, split=%.2f)  |  %s",
                      pretty, length(unique(part)),
                      d, K_star, e0, nu_t,
                      mean(result$fmc_K_plus), sm$split_rate, cs$label)
    make_fmc_biplot(
      A_hat = A_hat_pm, B_hat = B_hat_pm,
      item_partition = part,
      item_names     = item_names_full,
      title          = title,
      filename       = fname,
      plot_dir       = case_plot_dir,
      pal            = cluster_pal
    )
    cat(sprintf("  alt biplot -> %s  (K=%d)\n", fname, length(unique(part))))
  }
} else {
  cat(sprintf("[biplot skipped] d = %d (biplot is implemented for d == 2 only)\n",
              common_mcmc$d))
}

##############################################################################
# 4-F. CSV outputs (RDS already saved above before alt-partition block)
##############################################################################
write.csv(data.frame(item            = item_names_full,
                     fmc_mode_cluster = item_cluster_mode,
                     fmc_partition    = final_partition,
                     fmc_dahl         = P5_dahl),
          file.path(case_plot_dir, paste0(fmc_prefix, "_item_clusters.csv")),
          row.names = FALSE)

write.csv(round(co_cluster, 3),
          file.path(case_plot_dir, paste0(fmc_prefix, "_co_cluster.csv")))

# K_+ summary CSV (incl. silhouette) is written in section 4-I.

write.csv(round(B_hat_pm, 4),
          file.path(case_plot_dir, paste0(fmc_prefix, "_b_postmean.csv")))

##############################################################################
# 4-G. v11-specific: S0 hyperprior and cluster-scale diagnostic
#
# Compare the prior implied per-dim sd of Sigma_l with the empirical
# within-cluster sd of b_pm using modal labels.  With the Wishart
# hyperprior on S0, we ALSO report the posterior-mean S0 to see how the
# data adjusted the cluster-scale.
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

S0_init_used <- common_fmc_hyper$S0[1, 1]
prior_implied_sd_init <- sqrt(S0_init_used / (common_fmc_hyper$nu0 - d - 1))

S0_post_mean <- if (!is.null(result$fmc_S0))
  apply(result$fmc_S0, c(1, 2), mean) else common_fmc_hyper$S0
prior_implied_sd_post <- sqrt(diag(S0_post_mean) / (common_fmc_hyper$nu0 - d - 1))

cat(sprintf("\n-- v11 cluster-scale diagnostic (b_pm based) --\n"))
cat(sprintf("  S0 init scale (diag): %.4f\n", S0_init_used))
cat(sprintf("  prior-implied per-dim sd at INIT  = sqrt(S0_init/(nu0-d-1)) = %.4f\n",
            prior_implied_sd_init))
cat("  S0 posterior mean:\n")
print(round(S0_post_mean, 4))
cat("  prior-implied per-dim sd at POSTERIOR S0:\n")
print(round(prior_implied_sd_post, 4))
cat("  empirical within-cluster sd of b_pm per dim:\n")
print(round(empirical_within_sd, 4))

ratio_post <- prior_implied_sd_post / empirical_within_sd
cat("  ratio (post-S0 implied / empirical) per dim:\n")
print(round(ratio_post, 2))
if (any(ratio_post > 5, na.rm = TRUE)) {
  cat("  WARNING: post-S0 prior >> empirical (ratio > 5).  Either (i) the\n")
  cat("    Wishart hyperprior is too loose, or (ii) the modal labels are\n")
  cat("    over-merging (so empirical within-cluster sd is artificially small).\n")
}

write.csv(data.frame(
            dim                       = seq_len(d),
            S0_init_diag              = diag(common_fmc_hyper$S0),
            S0_post_mean_diag         = diag(S0_post_mean),
            prior_implied_sd_init     = prior_implied_sd_init,
            prior_implied_sd_post     = prior_implied_sd_post,
            empirical_within_sd_b     = empirical_within_sd,
            ratio_post                = ratio_post
          ),
          file.path(case_plot_dir, paste0(fmc_prefix, "_S0_diagnostic.csv")),
          row.names = FALSE)

##############################################################################
# 4-H. Silhouette diagnostic (b-based, NOT co-cluster-based)
#
# v11: clustering target is z_q = b_j; silhouette uses Euclidean distance
# between item posterior-mean b-positions in R^d.  Same rationale as v10's
# eta-based silhouette (avoid circularity of using (1 - co_cluster) as
# distance).
##############################################################################
sil_obj <- if (length(unique(final_partition)) >= 2) {
  tryCatch(cluster::silhouette(final_partition, dist(B_hat_pm)),
           error = function(e) NA)
} else NA
sil_mat <- if (is.matrix(sil_obj)) {
  as.data.frame(sil_obj[, , drop = FALSE])
} else NULL

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

  cat(sprintf("\n-- v11 silhouette diagnostic (b-based) --\n"))
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
                      paste0(fmc_prefix, "_silhouette_per_item.csv")),
            row.names = FALSE)

  pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_silhouette.pdf")),
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

cat(sprintf("\n-- v11 co-clustering quality --\n"))
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

cat(sprintf("\n-- v11 PEAR (posterior expected adjusted Rand) --\n"))
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
                    paste0(fmc_prefix, "_coclustering_quality.csv")),
          row.names = FALSE)

##############################################################################
# 4-H-ter. v11t-specific: Student-t weight (w_q) diagnostics
#
# For each item q we have a posterior trace of w_q over saved iterations.
# Items at cluster boundary should have w_q distributed below 1 (Gamma mass
# shifted left), items near cluster center should sit around 1+ slightly.
# This block:
#   - plots posterior E[w_q] sorted, coloured by P0 cluster, with 95% CI
#   - histogram of E[w_q] across items
#   - per-item CSV (item, P0 cluster, posterior mean & 95% CI of w_q)
##############################################################################
if (!is.null(result$fmc_ws)) {
  ws_mat   <- result$fmc_ws                          # n_save x P_total
  ws_pmean <- colMeans(ws_mat)
  ws_q025  <- apply(ws_mat, 2, quantile, 0.025)
  ws_q975  <- apply(ws_mat, 2, quantile, 0.975)

  pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_v11t_ws_post_mean.pdf")),
      width = 12, height = 6)
  par(mar = c(8, 4, 3, 1))
  ord_w  <- order(ws_pmean)
  K_part <- length(unique(final_partition))
  pal_w  <- expand_pal(K_part, cluster_pal)
  cols_w <- pal_w[final_partition][ord_w]
  plot(seq_along(ws_pmean), ws_pmean[ord_w], pch = 19, cex = 0.8,
       col = cols_w,
       ylim = range(c(ws_q025, ws_q975)),
       xlab = "", ylab = "posterior E[w_q]",
       main = sprintf("v11t Student-t weight posterior (sorted; nu_t=%g)", nu_t),
       xaxt = "n")
  segments(seq_along(ws_pmean), ws_q025[ord_w],
           seq_along(ws_pmean), ws_q975[ord_w],
           col = cols_w, lwd = 0.8)
  abline(h = 1, lty = 2, col = "gray40")
  axis(1, at = seq_along(ws_pmean),
       labels = item_names_full[ord_w], las = 2, cex.axis = 0.50)
  legend("topleft",
         legend = c(sprintf("P0 cluster %d", seq_len(K_part)),
                    "w_q = 1 (Gaussian-equiv)"),
         col    = c(pal_w, "gray40"),
         pch    = c(rep(19, K_part), NA),
         lty    = c(rep(NA, K_part), 2),
         bty = "n", cex = 0.7)
  dev.off()

  pdf(file.path(case_plot_dir, paste0(fmc_prefix, "_v11t_ws_hist.pdf")),
      width = 7, height = 5)
  par(mar = c(4, 4, 3, 1))
  hist(ws_pmean, breaks = 30, col = "steelblue", border = "white",
       main = sprintf("Posterior E[w_q] across %d items (nu_t=%g)",
                      length(ws_pmean), nu_t),
       xlab = "E[w_q]")
  abline(v = 1, lty = 2, col = "red", lwd = 2)
  dev.off()

  write.csv(data.frame(
              item               = item_names_full,
              P0_cluster         = as.integer(final_partition),
              ws_post_mean       = round(ws_pmean, 4),
              ws_post_q025       = round(ws_q025, 4),
              ws_post_q975       = round(ws_q975, 4)
            ),
            file.path(case_plot_dir, paste0(fmc_prefix, "_v11t_ws_per_item.csv")),
            row.names = FALSE)

  cat(sprintf("\n-- v11t Student-t weight summary --\n"))
  cat(sprintf("  mean(E[w_q]) across items: %.3f\n", mean(ws_pmean)))
  cat(sprintf("  range of E[w_q]:           [%.3f, %.3f]\n",
              min(ws_pmean), max(ws_pmean)))
  cat(sprintf("  fraction of items with E[w_q] < 1 (downweighted):    %.3f\n",
              mean(ws_pmean < 1)))
  cat(sprintf("  fraction of items with E[w_q] < 0.5 (strong outliers): %.3f\n",
              mean(ws_pmean < 0.5)))

  # Show the 10 most downweighted items (likely cluster-boundary / outlier
  # items) for paper-text use.
  ord_low <- order(ws_pmean)[seq_len(min(10, length(ws_pmean)))]
  cat("\n  10 most-downweighted items (lowest E[w_q]):\n")
  print(data.frame(
    item        = item_names_full[ord_low],
    P0_cluster  = as.integer(final_partition[ord_low]),
    ws_post_mean = round(ws_pmean[ord_low], 4)
  ), row.names = FALSE)
}

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
            pear_mean         = pear_mean,
            nu_t              = nu_t,
            S0_hyperprior     = isTRUE(result$fmc_S0_hyperprior),
            ws_post_mean_overall = if (!is.null(result$fmc_ws))
                                    mean(colMeans(result$fmc_ws)) else NA_real_,
            ws_frac_lt_1      = if (!is.null(result$fmc_ws))
                                    mean(colMeans(result$fmc_ws) < 1) else NA_real_,
            ws_frac_lt_0p5    = if (!is.null(result$fmc_ws))
                                    mean(colMeans(result$fmc_ws) < 0.5) else NA_real_
          ),
          file.path(case_plot_dir, paste0(fmc_prefix, "_K_plus_summary.csv")),
          row.names = FALSE)

cat(sprintf("\n=== v11t FMC summary ===\n"))
cat(sprintf("  mean(K_+) = %.2f, median(K_+) = %.0f, sd(K_+) = %.2f\n",
            mean(result$fmc_K_plus), median(result$fmc_K_plus),
            sd(result$fmc_K_plus)))
cat(sprintf("  silhouette (b-based): mean=%.3f, frac<0=%.3f\n",
            sil_overall_mean, sil_frac_neg))
cat(sprintf("  co-clustering: within-pair ratio=%.3f (%s), ambig=%.3f, PEAR=%.3f\n",
            within_pair_ratio, ratio_quality, pr_amb, pear_mean))
cat(sprintf("  split rate = %.3f, merge rate = %.3f\n",
            sm$split_rate, sm$merge_rate))
cat(sprintf("  nu_t (Student-t df, fixed) = %g\n", nu_t))

cat(sprintf("\n  Final partition (cutree on 1 - PSM, k=%d):\n",
            length(unique(final_partition))))
print(table(final_partition))
cat(sprintf("\n  Mode-cluster table (subject to label switching):\n"))
print(table(item_cluster_mode))
cat(sprintf("\n  Dahl partition (k=%d):\n", length(unique(P5_dahl))))
print(table(P5_dahl))
cat(sprintf("\n-> v11t plots & artifacts saved to: %s\n", case_plot_dir))



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
