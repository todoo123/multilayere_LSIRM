rm(list=ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile Rcpp (v6: 5-layered hierarchical, GRM ordinal, robust continuous)
#    NOTE: Using 5-layer model with empty 5th layer for 4-layer simulation
# =========================================================
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data")
source("my_LSIRM_5layered_hierarchical_cpp_v6.R")
source("utils.R")

# =========================================================
# 1) Simulation settings
# =========================================================
n  <- 150
P1 <- 10  # binary
P2 <- 18  # continuous (robust)
P3 <- 10  # count (NB)
P4 <- 10  # ordinal (GRM, K1=5 categories)

d <- 2
gamma_true <- 1.0

K <- 3  # number of clusters

K1 <- 5   # ordinal categories

# Hierarchical position parameters
sigma1_sq_true <- 0.3  # global-local coupling variance

# Robust continuous: Student-t df
nu2_true <- 5    # true df for generating Y_con
nu2_fit  <- 4    # df used in fitting

sigma0_sq_true <- 1.0  # continuous layer residual variance
kappa_true <- 1.0      # NB overdispersion

# =========================================================
# 2) Cluster centers for positions
# =========================================================
centers_resp <- rbind(
  c(-1.5,  1.5),
  c( 1.5,  1.5),
  c( 0.0, -1.5)
)

centers_bin <- rbind(
  c(-1,  2),
  c( 2,  2),
  c( 0,  0)
)

centers_con <- rbind(
  c( 2, -1),
  c( 3,  3),
  c(-1, -1)
)

centers_cnt <- rbind(
  c(-1,  2),
  c( 2,  2),
  c( 0,  0)
)

centers_ord <- rbind(
  c(-2,  0),
  c( 1,  2),
  c( 1, -1)
)

sd_cluster_resp <- 0.80
sd_cluster_item <- 0.30

# =========================================================
# 3) Helper: sample positions with K clusters
# =========================================================
sample_positions_3clusters <- function(N, centers, sd=0.4, prob=NULL){
  K <- nrow(centers)
  if(is.null(prob)) prob <- rep(1/K, K)
  cl <- sample(1:K, size=N, replace=TRUE, prob=prob)
  X  <- centers[cl, , drop=FALSE] + matrix(rnorm(N*ncol(centers), 0, sd), N, ncol(centers))
  list(X=X, cl=cl)
}

# =========================================================
# 4) True positions: global z, local a1-a4, items b1-b4
# =========================================================
# Global respondent position z
resp <- sample_positions_3clusters(n, centers_resp, sd_cluster_resp)
Z_true <- resp$X     # n x d
Z_cl   <- resp$cl

# Local positions: a_l | z ~ N(z, sigma1_sq * I)
generate_local <- function(Z, sigma1_sq) {
  Z + matrix(rnorm(nrow(Z)*ncol(Z), 0, sqrt(sigma1_sq)), nrow(Z), ncol(Z))
}
set.seed(42)
A1_true <- generate_local(Z_true, sigma1_sq_true)
A2_true <- generate_local(Z_true, sigma1_sq_true)
A3_true <- generate_local(Z_true, sigma1_sq_true)
A4_true <- generate_local(Z_true, sigma1_sq_true)

# Item positions (per layer)
item_bin <- sample_positions_3clusters(P1, centers_bin, sd_cluster_item)
item_con <- sample_positions_3clusters(P2, centers_con, sd_cluster_item)
item_cnt <- sample_positions_3clusters(P3, centers_cnt, sd_cluster_item)
item_ord <- sample_positions_3clusters(P4, centers_ord, sd_cluster_item)

B1_true <- item_bin$X;  B1_cl <- item_bin$cl
B2_true <- item_con$X;  B2_cl <- item_con$cl
B3_true <- item_cnt$X;  B3_cl <- item_cnt$cl
B4_true <- item_ord$X;  B4_cl <- item_ord$cl

# =========================================================
# 5) True intercept / difficulty parameters
# =========================================================
true_sigma_alpha <- 2.0
true_tau_beta1   <- 1.5
true_tau_beta2   <- 1.0
true_tau_beta3   <- 0.5

# Layer-specific alphas
alpha1_true <- rnorm(n, 0, true_sigma_alpha)
alpha2_true <- rnorm(n, 0, true_sigma_alpha)
alpha3_true <- rnorm(n, 0, true_sigma_alpha)
alpha4_true <- rnorm(n, 0, true_sigma_alpha)

beta1_true <- rnorm(P1, 0, true_tau_beta1)
beta2_true <- rnorm(P2, 0, true_tau_beta2)
beta3_true <- rnorm(P3, 0, true_tau_beta3)

# =========================================================
# 6) GRM ordinal thresholds (descending order)
# =========================================================
generate_grm_thresholds <- function(P, K) {
  Km1 <- K - 1
  beta_mat <- matrix(NA, P, Km1)
  for (j in 1:P) {
    beta_mat[j, ] <- sort(rnorm(Km1, 0, 1.5), decreasing = TRUE)
  }
  beta_mat
}

beta4_true <- generate_grm_thresholds(P4, K1)  # P4 x (K1-1)

cat("Beta4 thresholds (first 3 items):\n")
print(round(beta4_true[1:3, ], 2))

# =========================================================
# 7) Distances and linear predictors (layer-specific positions)
# =========================================================
dist_mat <- function(A, B){
  n <- nrow(A); P <- nrow(B)
  out <- matrix(0, n, P)
  for(j in 1:P){
    out[,j] <- sqrt(rowSums((A - matrix(B[j,], n, ncol(A), byrow=TRUE))^2))
  }
  out
}

# Each layer uses its own local position a_l
D1_true <- dist_mat(A1_true, B1_true)
D2_true <- dist_mat(A2_true, B2_true)
D3_true <- dist_mat(A3_true, B3_true)
D4_true <- dist_mat(A4_true, B4_true)

ETA1_true <- outer(alpha1_true, rep(1,P1)) - outer(rep(1,n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha2_true, rep(1,P2)) - outer(rep(1,n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha3_true, rep(1,P3)) - outer(rep(1,n), beta3_true) - gamma_true * D3_true
ETA4_true <- outer(alpha4_true, rep(1,P4)) - gamma_true * D4_true  # GRM: no beta intercept, thresholds instead

# =========================================================
# 8) Generate survey data
# =========================================================
invlogit <- function(x) 1/(1+exp(-x))

# --- Layer 1: Binary ---
P_bern <- invlogit(ETA1_true)
Y_bin <- matrix(rbinom(n*P1, 1, as.vector(P_bern)), n, P1)

# --- Layer 2: Continuous (Student-t robust) ---
lambda_true <- matrix(rgamma(n*P2, shape = nu2_true/2, rate = nu2_true/2), n, P2)
Y_con <- ETA2_true + matrix(rnorm(n*P2, 0, sqrt(sigma0_sq_true)), n, P2) / sqrt(lambda_true)
storage.mode(Y_con) <- "numeric"

# --- Layer 3: Count (NB) ---
MU_cnt <- exp(ETA3_true)
size_nb <- 1 / kappa_true
Y_cnt <- matrix(rnbinom(n*P3, size=size_nb, mu=as.vector(MU_cnt)), n, P3)

# --- Layer 4: Ordinal (GRM, K1 categories) ---
generate_grm_data <- function(ETA, beta_thr, K_cat) {
  n_obs <- nrow(ETA)
  P_obs <- ncol(ETA)
  Y <- matrix(NA_integer_, n_obs, P_obs)

  for (j in 1:P_obs) {
    for (i in 1:n_obs) {
      eta_ij <- ETA[i, j]
      p_ge <- invlogit(eta_ij + beta_thr[j, ])

      p <- numeric(K_cat)
      p[1] <- 1 - p_ge[1]
      for (k in 2:(K_cat - 1)) {
        p[k] <- p_ge[k - 1] - p_ge[k]
      }
      p[K_cat] <- p_ge[K_cat - 1]

      p[p < 0] <- 0
      ps <- sum(p)
      if (ps <= 0) {
        p <- rep(0, K_cat); p[round(K_cat/2)] <- 1
      } else {
        p <- p / ps
      }
      Y[i, j] <- sample.int(K_cat, size = 1, prob = p)
    }
  }
  storage.mode(Y) <- "integer"
  Y
}

Y_ord1 <- generate_grm_data(ETA4_true, beta4_true, K1)

# Empty 5th layer (model is 5-layer but we use only 4)
Y_ord2 <- matrix(1L, nrow = n, ncol = 0)

cat("\n=== Data Summary ===\n")
cat(sprintf("Y_bin:  %d x %d (binary)\n",  nrow(Y_bin),  ncol(Y_bin)))
cat(sprintf("Y_con:  %d x %d (continuous)\n", nrow(Y_con),  ncol(Y_con)))
cat(sprintf("Y_cnt:  %d x %d (count)\n",   nrow(Y_cnt),  ncol(Y_cnt)))
cat(sprintf("Y_ord1: %d x %d (ordinal, K=%d)\n", nrow(Y_ord1), ncol(Y_ord1), K1))
cat(sprintf("\nOrdinal response table:\n")); print(table(Y_ord1))

# =========================================================
# 9) Plot TRUE positions
# =========================================================
plot_positions_v6 <- function(Z, A_list, B_list, Z_cl, B_cl_list,
                              layer_names = NULL, rng = NA, title = NA) {
  if (is.null(layer_names))
    layer_names <- c("Binary", "Continuous", "Count", "Ordinal")

  all_pts <- rbind(Z, do.call(rbind, A_list), do.call(rbind, B_list))
  if (any(is.na(rng))) {
    rng <- range(all_pts, na.rm = TRUE)
  }
  if (is.na(title)) {
    title <- "TRUE latent positions (V6 hierarchical, 4-layer)"
  }

  item_cols <- c("forestgreen", "orange", "cyan4", "purple")
  local_cols <- c("palegreen", "lightyellow", "lightskyblue", "plum")

  base::plot(Z, pch=21, bg="gray80", col="black",
       xlab="Dim1", ylab="Dim2", main=title,
       xlim=rng, ylim=rng, cex=0.9)
  text(Z, labels=Z_cl, cex=0.5, pos=3, col="gray30")

  for (l in seq_along(B_list)) {
    points(B_list[[l]], pch=21, bg=item_cols[l], col=item_cols[l], cex=1.2)
    text(B_list[[l]], labels=B_cl_list[[l]], cex=0.7, pos=4, col=item_cols[l])
  }

  legend("topleft",
         legend=c("Respondents (z)", paste0(layer_names, " items")),
         pch=21,
         pt.bg=c("gray80", item_cols[1:length(B_list)]),
         col=c("black", item_cols[1:length(B_list)]),
         bty="n", cex=0.8)
}

if (!dir.exists("plot")) dir.create("plot", recursive = TRUE)

pdf("plot/4layered_v6_true_positions.pdf", width = 10, height = 8)
par(mfrow=c(1,1))
plot_positions_v6(Z_true,
                  list(A1_true, A2_true, A3_true, A4_true),
                  list(B1_true, B2_true, B3_true, B4_true),
                  Z_cl, list(B1_cl, B2_cl, B3_cl, B4_cl))
dev.off()

# --- Plot global vs local positions ---
pdf("plot/4layered_v6_true_global_vs_local.pdf", width = 12, height = 10)
par(mfrow=c(2,3), mar=c(4,4,3,1))
layer_names <- c("Global (z)", "L1: Binary (a1)", "L2: Continuous (a2)",
                 "L3: Count (a3)", "L4: Ordinal (a4)")
pos_list <- list(Z_true, A1_true, A2_true, A3_true, A4_true)
pos_cols <- c("black", "forestgreen", "orange", "cyan4", "purple")

all_pos <- do.call(rbind, pos_list)
rng <- range(all_pos)

for (l in seq_along(pos_list)) {
  base::plot(pos_list[[l]], pch=21, bg=ifelse(l==1, "gray80", "white"),
       col=pos_cols[l], cex=0.8,
       xlab="Dim1", ylab="Dim2", main=layer_names[l],
       xlim=rng, ylim=rng)
  text(pos_list[[l]], labels=1:n, cex=0.4, pos=3, col=pos_cols[l])
}
dev.off()


# =========================================================
# 10) Run Model (V6: hierarchical, using 4 of 5 layers)
# =========================================================
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data")

result <- lsirm_hierarchical_layer5_grm_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
  d = 2,
  n_iter = 100000,
  burnin = 10000,
  thin = 10,
  nu2 = nu2_fit,
  hyper = list(
    a_sigma=1, b_sigma=0.1,
    a_tau1=1, b_tau1=0.1, a_tau2=1, b_tau2=0.1, a_tau3=1, b_tau3=0.1,
    a_sigma0=1, b_sigma0=1,
    mu_log_gamma1=0, sd_log_gamma1=1,
    mu_log_gamma2=0, sd_log_gamma2=1,
    mu_log_gamma3=0, sd_log_gamma3=1,
    mu_log_gamma4=0, sd_log_gamma4=1,
    mu_log_gamma5=0, sd_log_gamma5=1,
    mu_log_kappa=0, sd_log_kappa=1,
    mu_beta4=0, sd_beta4=2,
    mu_beta5=0, sd_beta5=2,
    a_sigma_global=1, b_sigma_global=0.5
  ),
  prop_sd = list(
    alpha1=0.1, alpha2=0.1, alpha3=0.1, alpha4=0.1, alpha5=0.1,
    log_gamma1=0.05, log_gamma2=0.05, log_gamma3=0.05, log_gamma4=0.05, log_gamma5=0.05,
    a1=0.1, a2=0.1, a3=0.1, a4=0.1, a5=0.1,
    beta1=0.1, beta2=0.1, beta3=0.1, beta4=0.3, beta5=0.3,
    b1=0.1, b2=0.1, b3=0.1, b4=0.1, b5=0.1,
    log_kappa=0.05
  ),
  init = NULL,
  verbose = TRUE,
  fix_gamma = TRUE
)

# =========================================================
# 11) Acceptance rates
# =========================================================
cat("\n=== Acceptance Rates ===\n")
result$accept
fit <- result

if (!dir.exists("plot")) dir.create("plot", recursive = TRUE)

if (is.null(fit$samples)) {
  fit$samples <- fit
}
samps <- fit$samples

# =========================================================
# 12) Traceplots: latent positions z, a1-a4, b1-b4
# =========================================================

# --- z (global) ---
pdf("plot/4layered_v6_trace_z.pdf", width = 8, height = 12)
par(mfrow = c(4,2), mar=c(3,3,2,1))
for(i in 1:min(8, dim(samps$z)[2])){
  for(j in 1:dim(samps$z)[3]){
    ts.plot(samps$z[,i,j], main = paste0('z: ', i, '_', j))
    abline(h = Z_true[i,j], col = "red", lty = 2)
  }
}
dev.off()

# --- a1-a4 (local) ---
for (l in 1:4) {
  al_name <- paste0("a", l)
  al_samps <- samps[[al_name]]
  al_true <- list(A1_true, A2_true, A3_true, A4_true)[[l]]

  pdf(paste0("plot/4layered_v6_trace_", al_name, ".pdf"), width = 8, height = 12)
  par(mfrow = c(4,2), mar=c(3,3,2,1))
  for(i in 1:min(8, dim(al_samps)[2])){
    for(j in 1:dim(al_samps)[3]){
      ts.plot(al_samps[,i,j], main = paste0(al_name, ': ', i, '_', j))
      abline(h = al_true[i,j], col = "red", lty = 2)
    }
  }
  dev.off()
}

# --- b1-b4 (items) ---
for (l in 1:4) {
  bl_name <- paste0("b", l)
  bl_samps <- samps[[bl_name]]

  pdf(paste0("plot/4layered_v6_trace_", bl_name, ".pdf"), width = 8, height = 12)
  par(mfrow = c(2,2), mar=c(3,3,2,1))
  for(i in 1:dim(bl_samps)[2]){
    for(j in 1:dim(bl_samps)[3]){
      ts.plot(bl_samps[,i,j], main = paste0(bl_name, ': ', i, '_', j))
    }
  }
  dev.off()
}

# =========================================================
# 13) Traceplots: alpha1-alpha4, beta1-beta3
# =========================================================
leg <- list(
  x      = "topright",
  legend = c("Posterior mean", "95% credible interval", "True value"),
  col    = c("darkgreen", "blue", "red"),
  lwd    = 2,
  lty    = c(1, 3, 2),
  bty    = "n",
  cex    = 0.8
)

alpha_trues <- list(alpha1_true, alpha2_true, alpha3_true, alpha4_true)
for (l in 1:4) {
  al_name <- paste0("alpha", l)
  pdf(paste0("plot/4layered_v6_trace_", al_name, ".pdf"), width = 8, height = 12)
  plot_trace_vec(samps[[al_name]], alpha_trues[[l]], al_name, mfrow = c(3,2), leg = leg)
  dev.off()
}

pdf("plot/4layered_v6_trace_beta1.pdf", width = 8, height = 12)
plot_trace_vec(samps$beta1, beta1_true, "beta1", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_v6_trace_beta2.pdf", width = 8, height = 12)
plot_trace_vec(samps$beta2, beta2_true, "beta2", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_v6_trace_beta3.pdf", width = 8, height = 12)
plot_trace_vec(samps$beta3, beta3_true, "beta3", mfrow = c(3,2), leg = leg)
dev.off()

# =========================================================
# 14) GRM threshold traceplots (beta4)
# =========================================================
if (!is.null(samps$beta4)) {
  beta4_samps <- samps$beta4  # (n_save, P4, K1-1)
  n_save <- dim(beta4_samps)[1]
  beta4_mat <- matrix(beta4_samps, nrow = n_save, ncol = P4 * (K1 - 1))
  beta4_true_vec <- as.vector(beta4_true)

  pdf("plot/4layered_v6_trace_beta4_thr.pdf", width = 8, height = 12)
  plot_trace_vec(beta4_mat, beta4_true_vec, "beta4 (GRM thresholds, ord)", mfrow = c(3,2), leg = leg)
  dev.off()
}

# =========================================================
# 15) Extra scalar parameters
# =========================================================
pdf("plot/4layered_v6_extra_parameter.pdf", width = 8, height = 14)
par(mfrow = c(4,2), mar=c(3,3,2,1))

plot_trace_scalar(samps$sigma0_sq, true = sigma0_sq_true, main = "sigma0_sq")

plot_trace_scalar(samps$log_gamma1, true = gamma_true, main = "gamma1 (Bin)", transform = exp)
plot_trace_scalar(samps$log_gamma2, true = gamma_true, main = "gamma2 (Con)", transform = exp)
plot_trace_scalar(samps$log_gamma3, true = gamma_true, main = "gamma3 (Cnt)", transform = exp)
plot_trace_scalar(samps$log_gamma4, true = gamma_true, main = "gamma4 (Ord)", transform = exp)

plot_trace_scalar(samps$log_kappa, true = kappa_true, main = "kappa", transform = exp)

# sigma1_sq (hierarchical coupling)
plot_trace_scalar(samps$sigma1_sq, true = sigma1_sq_true, main = "sigma1_sq (global-local)")

# lambda2_mean (robust weight)
if (!is.null(samps$lambda2_mean)) {
  plot_trace_scalar(samps$lambda2_mean, true = 1.0, main = "lambda2_mean (robust weight)")
}

dev.off()

# =========================================================
# 15b) Layer-specific sigma_alpha traceplots
# =========================================================
pdf("plot/4layered_v6_trace_sigma_alpha.pdf", width = 8, height = 8)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for (l in 1:4) {
  sa_name <- paste0("sigma_alpha", l, "_sq")
  if (!is.null(samps[[sa_name]])) {
    plot_trace_scalar(samps[[sa_name]], true = true_sigma_alpha^2,
                      main = paste0("sigma_alpha", l, "_sq"))
  }
}
dev.off()

# =========================================================
# 16) Lambda2 diagnostics
# =========================================================
if (!is.null(samps$lambda2) && length(dim(samps$lambda2)) == 3) {
  lam <- samps$lambda2  # (n_save, n, P2)

  n_edge_show <- min(12, n * P2)
  set.seed(42)
  all_edges <- expand.grid(i = 1:n, j = 1:P2)
  edge_idx  <- all_edges[sort(sample(nrow(all_edges), n_edge_show)), ]

  pdf("plot/4layered_v6_trace_lambda2_edges.pdf", width = 10, height = 12)
  par(mfrow = c(4, 3), mar = c(3, 3, 2, 1))
  for (r in seq_len(nrow(edge_idx))) {
    ii <- edge_idx$i[r]; jj <- edge_idx$j[r]
    x <- lam[, ii, jj]
    ts.plot(x, main = bquote(lambda[2] ~ "(" * .(ii) * "," * .(jj) * ")"))
    abline(h = mean(x), col = "darkgreen", lwd = 2)
    abline(h = 1, col = "red", lty = 2)
  }
  dev.off()
}

if (!is.null(samps$lambda2_postmean)) {
  lam_pm <- samps$lambda2_postmean

  pdf("plot/4layered_v6_lambda2_postmean_heatmap.pdf", width = 10, height = 8)
  col_pal <- colorRampPalette(c("red", "white", "steelblue"))(100)
  lam_clipped <- pmin(lam_pm, quantile(lam_pm, 0.99))
  image(1:nrow(lam_pm), 1:ncol(lam_pm), lam_clipped,
        col = col_pal, xlab = "Respondent (i)", ylab = "Item (j)",
        main = expression("Posterior Mean of " * lambda[ij]^{(2)}),
        axes = FALSE)
  axis(1, at = seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm))),
       labels = round(seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm)))))
  axis(2, at = 1:ncol(lam_pm), labels = paste0("j", 1:ncol(lam_pm)), las = 2, cex.axis = 0.7)
  box()
  dev.off()

  pdf("plot/4layered_v6_lambda2_postmean_boxplot.pdf", width = 10, height = 6)
  colnames(lam_pm) <- paste0("j", 1:ncol(lam_pm))
  boxplot(as.data.frame(lam_pm), las = 2, cex.axis = 0.7,
          main = expression("Posterior Mean " * lambda[ij]^{(2)} * " by Item"),
          ylab = expression(lambda[ij]^{(2)}), col = "lightyellow", outline = TRUE)
  abline(h = 1, col = "red", lty = 2)
  dev.off()

  if (exists("lambda_true")) {
    pdf("plot/4layered_v6_lambda2_true_vs_hat.pdf", width = 7, height = 7)
    plot(as.vector(lambda_true), as.vector(lam_pm),
         pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.5,
         xlab = "True lambda", ylab = "Posterior mean lambda",
         main = "Lambda2 recovery: True vs Estimated")
    abline(a = 0, b = 1, col = "red", lwd = 2)
    dev.off()
    cat(sprintf("Lambda2 recovery (cor): %.3f\n", cor(as.vector(lambda_true), as.vector(lam_pm))))
  }

  cat("\n-- Lambda2 Posterior Mean Summary --\n")
  cat(sprintf("  Overall mean: %.3f\n", mean(lam_pm)))
  cat(sprintf("  Min: %.3f | Q1: %.3f | Median: %.3f | Q3: %.3f | Max: %.3f\n",
              min(lam_pm), quantile(lam_pm, 0.25), median(lam_pm),
              quantile(lam_pm, 0.75), max(lam_pm)))
  n_outlier <- sum(lam_pm < 0.5)
  cat(sprintf("  Edges with lambda < 0.5 (potential outliers): %d / %d (%.1f%%)\n",
              n_outlier, length(lam_pm), 100 * n_outlier / length(lam_pm)))
}

# =========================================================
# 17) Boxplots: alpha, beta, thresholds
# =========================================================
pdf("plot/4layered_v6_boxplot_groups.pdf", width = 10, height = 8)

for (l in 1:4) {
  al_name <- paste0("alpha", l)
  plot_group_box_ci(samps[[al_name]], alpha_trues[[l]], al_name, per_page = 24)
}

plot_group_box_ci(samps$beta1, beta1_true, "beta1", per_page = 24)
plot_group_box_ci(samps$beta2, beta2_true, "beta2", per_page = 24)
plot_group_box_ci(samps$beta3, beta3_true, "beta3", per_page = 24)

# GRM thresholds boxplot
if (!is.null(samps$beta4)) {
  beta4_samps <- samps$beta4
  n_save <- dim(beta4_samps)[1]
  beta4_mat <- matrix(beta4_samps, nrow = n_save, ncol = P4 * (K1 - 1))
  beta4_true_vec <- as.vector(beta4_true)
  plot_group_box_ci(beta4_mat, beta4_true_vec, "beta4 (GRM thresholds, ord)", per_page = 24)
}

dev.off()

# =========================================================
# 18) Posterior mean positions
# =========================================================
Z_hat  <- apply(samps$z,  c(2,3), mean)
A1_hat <- apply(samps$a1, c(2,3), mean)
A2_hat <- apply(samps$a2, c(2,3), mean)
A3_hat <- apply(samps$a3, c(2,3), mean)
A4_hat <- apply(samps$a4, c(2,3), mean)
B1_hat <- apply(samps$b1, c(2,3), mean)
B2_hat <- apply(samps$b2, c(2,3), mean)
B3_hat <- apply(samps$b3, c(2,3), mean)
B4_hat <- apply(samps$b4, c(2,3), mean)

# =========================================================
# 19) Plot TRUE vs estimated positions
# =========================================================
pdf("plot/4layered_v6_positions_true_vs_hat.pdf", width = 14, height = 6)
par(mfrow = c(1,2))
plot_positions_v6(Z_true,
                  list(A1_true, A2_true, A3_true, A4_true),
                  list(B1_true, B2_true, B3_true, B4_true),
                  Z_cl, list(B1_cl, B2_cl, B3_cl, B4_cl),
                  title = "TRUE positions (4-layer hierarchical)")
plot_positions_v6(Z_hat,
                  list(A1_hat, A2_hat, A3_hat, A4_hat),
                  list(B1_hat, B2_hat, B3_hat, B4_hat),
                  Z_cl, list(B1_cl, B2_cl, B3_cl, B4_cl),
                  title = "Posterior mean positions (V6 hierarchical)")
dev.off()

# --- Global vs local: TRUE vs estimated (5-panel each) ---
pdf("plot/4layered_v6_global_local_true_vs_hat.pdf", width = 12, height = 20)
par(mfrow=c(5,2), mar=c(4,4,3,1))
layer_names <- c("Global (z)", "L1: Binary (a1)", "L2: Continuous (a2)",
                 "L3: Count (a3)", "L4: Ordinal (a4)")
true_list <- list(Z_true, A1_true, A2_true, A3_true, A4_true)
hat_list  <- list(Z_hat,  A1_hat,  A2_hat,  A3_hat,  A4_hat)
pos_cols <- c("black", "forestgreen", "orange", "cyan4", "purple")

all_pos <- do.call(rbind, c(true_list, hat_list))
rng <- range(all_pos)

for (l in seq_along(true_list)) {
  base::plot(true_list[[l]], pch=21, bg=ifelse(l==1,"gray80","white"),
       col=pos_cols[l], cex=0.8,
       xlab="Dim1", ylab="Dim2", main=paste0(layer_names[l], " - TRUE"),
       xlim=rng, ylim=rng)
  text(true_list[[l]], labels=1:n, cex=0.4, pos=3, col=pos_cols[l])

  base::plot(hat_list[[l]], pch=21, bg=ifelse(l==1,"gray80","white"),
       col=pos_cols[l], cex=0.8,
       xlab="Dim1", ylab="Dim2", main=paste0(layer_names[l], " - Estimated"),
       xlim=rng, ylim=rng)
  text(hat_list[[l]], labels=1:n, cex=0.4, pos=3, col=pos_cols[l])
}
dev.off()

# =========================================================
# 20) Distance recovery (per-layer, using local positions)
# =========================================================
gamma1_post <- mean(exp(samps$log_gamma1))
gamma2_post <- mean(exp(samps$log_gamma2))
gamma3_post <- mean(exp(samps$log_gamma3))
gamma4_post <- mean(exp(samps$log_gamma4))

D1_hat <- dist_mat(A1_hat, B1_hat) * gamma1_post
D2_hat <- dist_mat(A2_hat, B2_hat) * gamma2_post
D3_hat <- dist_mat(A3_hat, B3_hat) * gamma3_post
D4_hat <- dist_mat(A4_hat, B4_hat) * gamma4_post

D_true_all <- cbind(D1_true*gamma_true, D2_true*gamma_true, D3_true*gamma_true,
                    D4_true*gamma_true)
D_hat_all  <- cbind(D1_hat, D2_hat, D3_hat, D4_hat)

vec <- function(M) as.vector(M)

cor_all <- cor(vec(D_true_all), vec(D_hat_all))
cor_1 <- cor(vec(D1_true*gamma_true), vec(D1_hat))
cor_2 <- cor(vec(D2_true*gamma_true), vec(D2_hat))
cor_3 <- cor(vec(D3_true*gamma_true), vec(D3_hat))
cor_4 <- cor(vec(D4_true*gamma_true), vec(D4_hat))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("All layers combined:     %.3f\n", cor_all))
cat(sprintf("L1 Binary    (n x P1):   %.3f\n", cor_1))
cat(sprintf("L2 Continuous(n x P2):   %.3f\n", cor_2))
cat(sprintf("L3 Count     (n x P3):   %.3f\n", cor_3))
cat(sprintf("L4 Ordinal   (n x P4):   %.3f\n", cor_4))

pdf("plot/4layered_v6_distance_recovery_scatter.pdf", width = 7, height = 7)
plot(
  vec(D_true_all), vec(D_hat_all),
  pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.3,
  xlab = "True distances x gamma_true",
  ylab = "Estimated distances x gamma_post",
  main = sprintf("Distance recovery: 4-layer hierarchical (r=%.3f)", cor_all)
)
abline(a = 0, b = 1, col = "red", lwd = 2)
dev.off()

pdf("plot/4layered_v6_distance_recovery_by_layer.pdf", width = 12, height = 6)
par(mfrow = c(2,2), mar=c(3,3,2,1))

layer_labels <- c("Binary", "Continuous", "Count", "Ordinal")
D_true_list <- list(D1_true, D2_true, D3_true, D4_true)
D_hat_list  <- list(D1_hat,  D2_hat,  D3_hat,  D4_hat)
cor_list    <- c(cor_1, cor_2, cor_3, cor_4)

for (l in 1:4) {
  plot(vec(D_true_list[[l]]*gamma_true), vec(D_hat_list[[l]]),
       pch=16, col=rgb(0,0,0,0.4), cex=0.5,
       main=sprintf("%s (r=%.3f)", layer_labels[l], cor_list[l]),
       xlab="True", ylab="Hat")
  abline(a=0,b=1,col="red",lwd=2)
}
dev.off()

# =========================================================
# 21) Hierarchical diagnostics: global-local distance
# =========================================================
euc_dist_rows <- function(A, B) sqrt(rowSums((A - B)^2))

# TRUE global-local distances
dist_zl_true <- data.frame(
  L1 = euc_dist_rows(Z_true, A1_true),
  L2 = euc_dist_rows(Z_true, A2_true),
  L3 = euc_dist_rows(Z_true, A3_true),
  L4 = euc_dist_rows(Z_true, A4_true)
)

# ESTIMATED global-local distances
dist_zl_hat <- data.frame(
  L1 = euc_dist_rows(Z_hat, A1_hat),
  L2 = euc_dist_rows(Z_hat, A2_hat),
  L3 = euc_dist_rows(Z_hat, A3_hat),
  L4 = euc_dist_rows(Z_hat, A4_hat)
)

pdf("plot/4layered_v6_global_local_distance_hist.pdf", width = 12, height = 8)
par(mfrow = c(2, 4), mar=c(4,4,3,1))

layer_cols <- c("forestgreen", "orange", "cyan4", "purple")
x_max <- max(unlist(dist_zl_true), unlist(dist_zl_hat), na.rm = TRUE) * 1.05
breaks_common <- seq(0, ceiling(x_max * 10) / 10, length.out = 25)

for (l in 1:4) {
  hist(dist_zl_true[[l]], breaks = breaks_common, col = layer_cols[l], border = "white",
       main = paste0(layer_labels[l], " (TRUE)"),
       xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
       ylab = "Frequency", xlim = c(0, x_max))
  abline(v = mean(dist_zl_true[[l]]), col = "red", lwd = 2, lty = 2)
  legend("topright",
         legend = sprintf("mean=%.3f", mean(dist_zl_true[[l]])),
         bty = "n", cex = 0.8)
}

for (l in 1:4) {
  hist(dist_zl_hat[[l]], breaks = breaks_common, col = layer_cols[l], border = "white",
       main = paste0(layer_labels[l], " (Estimated)"),
       xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
       ylab = "Frequency", xlim = c(0, x_max))
  abline(v = mean(dist_zl_hat[[l]]), col = "red", lwd = 2, lty = 2)
  legend("topright",
         legend = sprintf("mean=%.3f", mean(dist_zl_hat[[l]])),
         bty = "n", cex = 0.8)
}
dev.off()

# --- Density overlay ---
pdf("plot/4layered_v6_global_local_distance_density.pdf", width = 12, height = 5)
par(mfrow = c(1,2), mar=c(4,4,3,1))

dens_true <- lapply(dist_zl_true, density)
dens_hat  <- lapply(dist_zl_hat,  density)

y_max_t <- max(sapply(dens_true, function(d) max(d$y)))
base::plot(NULL, xlim = c(0, x_max), ylim = c(0, y_max_t * 1.1),
     xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
     ylab = "Density", main = "Global-Local Distance (TRUE)")
for (l in 1:4) lines(dens_true[[l]], col = layer_cols[l], lwd = 2)
legend("topright", legend = layer_labels, col = layer_cols, lwd = 2, bty = "n", cex = 0.9)
abline(v = sqrt(2 * sigma1_sq_true), col = "gray50", lty = 3, lwd = 1.5)
text(sqrt(2 * sigma1_sq_true), y_max_t * 0.9,
     labels = sprintf("E[||.||]=%.2f", sqrt(2 * sigma1_sq_true)),
     pos = 4, cex = 0.7, col = "gray30")

y_max_h <- max(sapply(dens_hat, function(d) max(d$y)))
base::plot(NULL, xlim = c(0, x_max), ylim = c(0, y_max_h * 1.1),
     xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
     ylab = "Density", main = "Global-Local Distance (Estimated)")
for (l in 1:4) lines(dens_hat[[l]], col = layer_cols[l], lwd = 2)
legend("topright", legend = layer_labels, col = layer_cols, lwd = 2, bty = "n", cex = 0.9)

sigma1_sq_hat <- mean(samps$sigma1_sq)
abline(v = sqrt(2 * sigma1_sq_hat), col = "gray50", lty = 3, lwd = 1.5)
text(sqrt(2 * sigma1_sq_hat), y_max_h * 0.9,
     labels = sprintf("E[||.||]=%.2f", sqrt(2 * sigma1_sq_hat)),
     pos = 4, cex = 0.7, col = "gray30")

dev.off()

# =========================================================
# 22) Position recovery: z, a1-a4 correlation
# =========================================================
cat("\n=== Position recovery (Procrustes-aligned correlation) ===\n")
cat(sprintf("  z  (global):  Dim1 r=%.3f, Dim2 r=%.3f\n",
            cor(Z_true[,1], Z_hat[,1]), cor(Z_true[,2], Z_hat[,2])))

A_true_list <- list(A1_true, A2_true, A3_true, A4_true)
A_hat_list  <- list(A1_hat,  A2_hat,  A3_hat,  A4_hat)
for (l in 1:4) {
  cat(sprintf("  a%d (local L%d): Dim1 r=%.3f, Dim2 r=%.3f\n", l, l,
              cor(A_true_list[[l]][,1], A_hat_list[[l]][,1]),
              cor(A_true_list[[l]][,2], A_hat_list[[l]][,2])))
}

# =========================================================
# 23) sigma1_sq recovery summary
# =========================================================
cat("\n=== sigma1_sq recovery ===\n")
cat(sprintf("  True:          %.3f\n", sigma1_sq_true))
cat(sprintf("  Posterior mean: %.3f\n", mean(samps$sigma1_sq)))
cat(sprintf("  Posterior sd:   %.3f\n", sd(samps$sigma1_sq)))
cat(sprintf("  95%% CI:        [%.3f, %.3f]\n",
            quantile(samps$sigma1_sq, 0.025), quantile(samps$sigma1_sq, 0.975)))

cat("\n=== Done! ===\n")
