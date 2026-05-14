rm(list=ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile Rcpp (v5: robust continuous layer)
# =========================================================
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM")
source("my_LSIRM_4layered_nonhierarchical_cpp_v5.R")

# =========================================================
# 1) Simulation settings
# =========================================================
n  <- 150
P2 <- 18  # continuous
P1 <- 10  # binary
P3 <- 10  # count
P4 <- 10  # ordinal

d <- 2
gamma_true <- 1.0

n_smart <- 45
n_stupid <- 45

K <- 3

centers_resp <- rbind(
  c(1, 1),
  c(1, 1),
  c(1, 1)
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

centers_likert <- rbind(
  c(-1,  2),
  c( 2,  2),
  c( 0,  0)
)

sd_cluster_resp <- 1.00
sd_cluster_item <- 0.30

sigma_likert_latent <- 1.0
kappa_true <- 1

# Robust simulation: degrees of freedom for continuous layer errors
nu2_true <- 5    # Student-t df for generating Y_con (heavy-tailed)
nu2_fit  <- 4    # df used in fitting

# =========================================================
# 2) Helper: sample latent positions with 3 clear clusters
# =========================================================
sample_positions_3clusters <- function(N, centers, sd=0.4, prob=NULL){
  K <- nrow(centers)
  if(is.null(prob)) prob <- rep(1/K, K)
  cl <- sample(1:K, size=N, replace=TRUE, prob=prob)
  X  <- centers[cl, , drop=FALSE] + matrix(rnorm(N*ncol(centers), 0, sd), N, ncol(centers))
  list(X=X, cl=cl)
}

# =========================================================
# 3) True respondent positions a_i and item positions b_j^(layer)
# =========================================================
resp <- sample_positions_3clusters(n, centers_resp, sd_cluster_resp)
A_true <- resp$X
A_cl   <- resp$cl

item_bin <- sample_positions_3clusters(P1, centers_bin, sd_cluster_item)
B1_true  <- item_bin$X
B1_cl    <- item_bin$cl

item_lik <- sample_positions_3clusters(P2, centers_con, sd_cluster_item)
B2_true  <- item_lik$X
B2_cl    <- item_lik$cl

item_cnt <- sample_positions_3clusters(P3, centers_cnt, sd_cluster_item)
B3_true  <- item_cnt$X
B3_cl    <- item_cnt$cl

item_lsg <- sample_positions_3clusters(P4, centers_likert, sd_cluster_item)
B4_true  <- item_lsg$X
B4_cl    <- item_lsg$cl

true_sigma_alpha <- 2.0
true_tau_beta1   <- 1.5
true_tau_beta2   <- 1.0
true_tau_beta3   <- 0.5

alpha_true <- rnorm(n, mean = 0, sd = true_sigma_alpha)

ord_a <- order(alpha_true, decreasing = TRUE)
ability_group <- rep("stupid", n)
ability_group[ord_a[1:n_smart]] <- "smart"

beta1_true <- rnorm(P1, mean = 0, sd = true_tau_beta1)
beta2_true <- rnorm(P2, mean = 0, sd = true_tau_beta2)
beta3_true <- rnorm(P3, mean = 0, sd = true_tau_beta3)

# =========================================================
# 4) Ordinal thresholds
# =========================================================
softplus <- function(x) log1p(exp(-abs(x))) + pmax(x, 0)

build_thresholds_fixed0 <- function(delta_vec) {
  Kminus1 <- 1 + length(delta_vec)
  b <- numeric(Kminus1)
  b[1] <- 0.0
  if (Kminus1 >= 2) {
    for (k in 2:Kminus1) {
      b[k] <- b[k - 1] + softplus(delta_vec[k - 1])
    }
  }
  b
}

true_mu_u     <- 0
true_sd_u     <- 2
true_mu_delta <- 0
true_sd_delta <- 1

u_true <- rep(0.0, P4)

K_ord <- 5
delta_true <- matrix(rnorm(P4 * (K_ord - 2), true_mu_delta, true_sd_delta),
                     nrow = P4, ncol = (K_ord - 2))

beta4_list_true <- vector("list", P4)
for (j in 1:P4) {
  beta4_list_true[[j]] <- build_thresholds_fixed0(delta_true[j, ])
}

# =========================================================
# 5) Distances and linear predictors
# =========================================================
dist_mat <- function(A, B){
  n <- nrow(A); P <- nrow(B)
  out <- matrix(0, n, P)
  for(j in 1:P){
    out[,j] <- sqrt(rowSums((A - matrix(B[j,], n, ncol(A), byrow=TRUE))^2))
  }
  out
}

D1_true <- dist_mat(A_true, B1_true)
D2_true <- dist_mat(A_true, B2_true)
D3_true <- dist_mat(A_true, B3_true)
D4_true <- dist_mat(A_true, B4_true)

ETA1_true <- outer(alpha_true, rep(1,P1)) - outer(rep(1,n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha_true, rep(1,P2)) - outer(rep(1,n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha_true, rep(1,P3)) - outer(rep(1,n), beta3_true) - gamma_true * D3_true
ETA4_true <- outer(alpha_true, rep(1, P4)) - gamma_true * D4_true

# =========================================================
# 6) Generate survey data
# =========================================================
invlogit <- function(x) 1/(1+exp(-x))

# binary
P_bern <- invlogit(ETA1_true)
Y_bin <- matrix(rbinom(n*P1, 1, as.vector(P_bern)), n, P1)

# continuous: Student-t errors (robust scenario)
# t(nu) = Normal(0,1) / sqrt(chi2(nu)/nu)
# equivalently: Normal(0, sigma^2) with sigma^2 scaled by 1/lambda, lambda ~ Gamma(nu/2, nu/2)
lambda_true <- matrix(rgamma(n*P2, shape = nu2_true/2, rate = nu2_true/2), n, P2)
Y_con <- ETA2_true + matrix(rnorm(n*P2, 0, sigma_likert_latent), n, P2) / sqrt(lambda_true)
storage.mode(Y_con) <- "numeric"

# count (NB2)
MU_cnt <- exp(ETA3_true)
size_nb <- 1 / kappa_true
Y_cnt <- matrix(rnbinom(n*P3, size=size_nb, mu=as.vector(MU_cnt)), n, P3)

# ordinal
Y_ord <- matrix(NA_integer_, n, P4)

for (j in 1:P4) {
  b_vec <- beta4_list_true[[j]]
  for (i in 1:n) {
    eta_ij <- ETA4_true[i, j]
    Cvec <- invlogit(eta_ij - b_vec)

    p <- numeric(K_ord)
    p[1] <- 1 - Cvec[1]
    p[K_ord] <- Cvec[K_ord - 1]
    if (K_ord > 2) {
      for (y in 2:(K_ord - 1)) {
        p[y] <- Cvec[y - 1] - Cvec[y]
      }
    }

    p[p < 0] <- 0
    ps <- sum(p)
    if (ps <= 0) {
      p <- rep(0, K_ord); p[round(K_ord/2)] <- 1
    } else {
      p <- p / ps
    }

    Y_ord[i, j] <- sample.int(K_ord, size = 1, prob = p)
  }
}
storage.mode(Y_ord) <- "integer"


# =========================================================
# 7) Plot TRUE positions
# =========================================================
plot_positions <- function(A, B1, B2, B3, B4,
                           A_cl, B1_cl, B2_cl, B3_cl, B4_cl,
                           rng = NA, title = NA){
  if (sum(is.na(rng)) == TRUE){
    rng <- range(A[,1],A[,2],
                 B1[,1],B1[,2],
                 B2[,1],B2[,2],
                 B3[,1],B3[,2],
                 B4[,1],B4[,2])
  }
  if (is.na(title) == TRUE){
    title <- "TRUE latent positions: Respondents(gray), bin(red), con(orange), cnt(brown), ord(purple)"
  }

  base::plot(A, pch=21, bg="gray80", col="black",
       xlab="Dim1", ylab="Dim2", main=title,
       xlim=rng, ylim=rng)
  text(A, labels=A_cl, cex=0.6, pos=3, col="gray30")

  points(B1, pch=21, bg="red",    col="red");    text(B1, labels=B1_cl, cex=0.7, pos=4, col="red4")
  points(B2, pch=21, bg="orange", col="orange"); text(B2, labels=B2_cl, cex=0.7, pos=4, col="orange4")
  points(B3, pch=21, bg="brown",  col="brown");  text(B3, labels=B3_cl, cex=0.7, pos=4, col="brown4")
  points(B4, pch=21, bg="purple", col="purple"); text(B4, labels=B4_cl, cex=0.7, pos=4, col="purple4")

  legend("topleft",
         legend=c("Respondents","Binary items","Continuous items","Count items","Ordinal items"),
         pch=21,
         pt.bg=c("gray80","red","orange","brown","purple"),
         col=c("black","red","orange","brown","purple"),
         bty="n")
}

par(mfrow=c(1,1))
plot_positions(A_true, B1_true, B2_true, B3_true, B4_true,
               A_cl, B1_cl, B2_cl, B3_cl, B4_cl)


# =========================================================
# 8) Running Model (v5: robust)
# =========================================================
setwd('/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM')
source("my_LSIRM_4layered_nonhierarchical_cpp_v5.R")
result <- lsirm_sharedpos_layer4_robust_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord,
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
    mu_log_kappa=0, sd_log_kappa=1,
    mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
  ),
  prop_sd = list(
    alpha=0.1, log_gamma1=0.05, log_gamma2=0.05, log_gamma3=0.05, log_gamma4=0.05, a=0.1,
    beta1=0.1, beta2=0.1, beta3=0.1,
    b1=0.1, b2=0.1, b3=0.1, b4=0.1,
    log_kappa=0.05, u=0.1, delta=0.1
  ),
  init = NULL,
  verbose = TRUE,
  fix_gamma = TRUE
)

# =========================================================
# 0) Acceptance rates
# =========================================================
result$accept
fit4 <- result

if (!dir.exists("plot")) dir.create("plot", recursive = TRUE)

if (is.null(fit4$samples)) {
  fit4$samples <- fit4
}

# =========================================================
# 1) Traceplots: latent positions a, b1..b4
# =========================================================

pdf("plot/4layered_v5_trace_a.pdf", width = 8, height = 12)
par(mfrow = c(4,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$a)[2]){
  for(j in 1:dim(fit4$samples$a)[3]){
    ts.plot(fit4$samples$a[,i,j], main = paste0('a: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_v5_trace_b1.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b1)[2]){
  for(j in 1:dim(fit4$samples$b1)[3]){
    ts.plot(fit4$samples$b1[,i,j], main = paste0('b1: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_v5_trace_b2.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b2)[2]){
  for(j in 1:dim(fit4$samples$b2)[3]){
    ts.plot(fit4$samples$b2[,i,j], main = paste0('b2: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_v5_trace_b3.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b3)[2]){
  for(j in 1:dim(fit4$samples$b3)[3]){
    ts.plot(fit4$samples$b3[,i,j], main = paste0('b3: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_v5_trace_b4.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b4)[2]){
  for(j in 1:dim(fit4$samples$b4)[3]){
    ts.plot(fit4$samples$b4[,i,j], main = paste0('b4: ', i, '_', j))
  }
}
dev.off()

#-------------------------------------------------------------------------------
source("utils.R")

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
# 2) Traceplots: alpha, beta1..beta3
# =========================================================
pdf("plot/4layered_v5_trace_alpha.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$alpha, alpha_true, "alpha", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_v5_trace_beta1.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta1, beta1_true, "beta1", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_v5_trace_beta2.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta2, beta2_true, "beta2", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_v5_trace_beta3.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta3, beta3_true, "beta3", mfrow = c(3,2), leg = leg)
dev.off()

# =========================================================
# 3) Ordinal threshold params trace (u, delta, thr)
# =========================================================
if (exists("beta4_list_true") && !exists("thr_true_mat")) {
  P4 <- length(beta4_list_true)
  Kminus1_true <- length(beta4_list_true[[1]])
  thr_true_mat <- matrix(NA, P4, Kminus1_true)
  for (j in 1:P4) thr_true_mat[j, ] <- beta4_list_true[[j]]
}

if (exists("u_true")) {
  pdf("plot/4layered_v5_trace_u.pdf", width = 8, height = 12)
  plot_trace_vec(fit4$samples$u, u_true, "u (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
} else {
  pdf("plot/4layered_v5_trace_u.pdf", width = 8, height = 12)
  plot_trace_vec(fit4$samples$u, true = NA, "u (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
}

if (!is.null(fit4$samples$delta)) {
  delta_samps <- fit4$samples$delta
  n_save <- dim(delta_samps)[1]
  P4     <- dim(delta_samps)[2]
  Km2    <- dim(delta_samps)[3]
  delta_mat <- matrix(delta_samps, nrow = n_save, ncol = P4*Km2)

  delta_true_vec <- if (exists("delta_true")) as.vector(delta_true) else rep(NA, P4*Km2)

  pdf("plot/4layered_v5_trace_delta.pdf", width = 8, height = 12)
  plot_trace_vec(delta_mat, delta_true_vec, "delta (ordinal gaps)", mfrow = c(3,2), leg = leg)
  dev.off()
}

if (!is.null(fit4$samples$thr)) {
  thr_samps <- fit4$samples$thr
  n_save <- dim(thr_samps)[1]
  P4 <- dim(thr_samps)[2]
  Kminus1 <- dim(thr_samps)[3]
  thr_mat <- matrix(thr_samps, nrow = n_save, ncol = P4*Kminus1)

  thr_true_vec <- if (exists("thr_true_mat")) as.vector(thr_true_mat) else rep(NA, P4*Kminus1)

  pdf("plot/4layered_v5_trace_thr.pdf", width = 8, height = 12)
  plot_trace_vec(thr_mat, thr_true_vec, "thresholds thr (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
}

# =========================================================
# 4) Extra scalar parameters (per-layer gamma)
# =========================================================
pdf("plot/4layered_v5_extra_parameter.pdf", width = 8, height = 14)
par(mfrow = c(4,2), mar=c(3,3,2,1))

plot_trace_scalar(
  fit4$samples$sigma0_sq,
  true = if (exists("sigma0_true")) sigma0_true else NA,
  main = "sigma0_sq"
)

plot_trace_scalar(fit4$samples$log_gamma1, true = gamma_true, main = "gamma1 (Bin)", transform = exp)
plot_trace_scalar(fit4$samples$log_gamma2, true = gamma_true, main = "gamma2 (Con)", transform = exp)
plot_trace_scalar(fit4$samples$log_gamma3, true = gamma_true, main = "gamma3 (Cnt)", transform = exp)
plot_trace_scalar(fit4$samples$log_gamma4, true = gamma_true, main = "gamma4 (Ord)", transform = exp)

plot_trace_scalar(
  fit4$samples$log_kappa,
  true = if (exists("kappa_true")) kappa_true else NA,
  main = "kappa",
  transform = exp
)

plot_trace_scalar(fit4$samples$sigma_alpha_sq, true = true_sigma_alpha^2, main = "sigma_alpha_sq")

# lambda2_mean traceplot
if (!is.null(fit4$samples$lambda2_mean)) {
  plot_trace_scalar(fit4$samples$lambda2_mean, true = 1.0, main = "lambda2_mean (robust weight)")
}

dev.off()

# =========================================================
# 4b) Lambda2 diagnostics
# =========================================================
if (!is.null(fit4$samples$lambda2) && length(dim(fit4$samples$lambda2)) == 3) {
  lam <- fit4$samples$lambda2  # (n_save, n, P2)

  # Per-edge traceplot (12 random edges)
  n_edge_show <- min(12, n * P2)
  set.seed(42)
  all_edges <- expand.grid(i = 1:n, j = 1:P2)
  edge_idx  <- all_edges[sort(sample(nrow(all_edges), n_edge_show)), ]

  pdf("plot/4layered_v5_trace_lambda2_edges.pdf", width = 10, height = 12)
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

if (!is.null(fit4$samples$lambda2_postmean)) {
  lam_pm <- fit4$samples$lambda2_postmean

  # Heatmap
  pdf("plot/4layered_v5_lambda2_postmean_heatmap.pdf", width = 10, height = 8)
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

  # Boxplot
  pdf("plot/4layered_v5_lambda2_postmean_boxplot.pdf", width = 10, height = 6)
  colnames(lam_pm) <- paste0("j", 1:ncol(lam_pm))
  boxplot(as.data.frame(lam_pm), las = 2, cex.axis = 0.7,
          main = expression("Posterior Mean " * lambda[ij]^{(2)} * " by Item"),
          ylab = expression(lambda[ij]^{(2)}), col = "lightyellow", outline = TRUE)
  abline(h = 1, col = "red", lty = 2)
  dev.off()

  # Compare with true lambda
  if (exists("lambda_true")) {
    pdf("plot/4layered_v5_lambda2_true_vs_hat.pdf", width = 7, height = 7)
    plot(as.vector(lambda_true), as.vector(lam_pm),
         pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.5,
         xlab = "True lambda", ylab = "Posterior mean lambda",
         main = "Lambda2 recovery: True vs Estimated")
    abline(a = 0, b = 1, col = "red", lwd = 2)
    dev.off()
    cat(sprintf("Lambda2 recovery (cor): %.3f\n", cor(as.vector(lambda_true), as.vector(lam_pm))))
  }

  cat("\n── Lambda2 Posterior Mean Summary ──\n")
  cat(sprintf("  Overall mean: %.3f\n", mean(lam_pm)))
  cat(sprintf("  Min: %.3f | Q1: %.3f | Median: %.3f | Q3: %.3f | Max: %.3f\n",
              min(lam_pm), quantile(lam_pm, 0.25), median(lam_pm),
              quantile(lam_pm, 0.75), max(lam_pm)))
  n_outlier <- sum(lam_pm < 0.5)
  cat(sprintf("  Edges with lambda < 0.5 (potential outliers): %d / %d (%.1f%%)\n",
              n_outlier, length(lam_pm), 100 * n_outlier / length(lam_pm)))
}

# =========================================================
# 5) Boxplots
# =========================================================
pdf("plot/4layered_v5_boxplot_groups.pdf", width = 10, height = 8)

plot_group_box_ci(fit4$samples$alpha, alpha_true, "alpha", per_page = 24)
plot_group_box_ci(fit4$samples$beta1, beta1_true, "beta1", per_page = 24)
plot_group_box_ci(fit4$samples$beta2, beta2_true, "beta2", per_page = 24)
plot_group_box_ci(fit4$samples$beta3, beta3_true, "beta3", per_page = 24)

plot_group_box_ci(fit4$samples$u, if (exists("u_true")) u_true else NA, "u (ordinal)", per_page = 24)

if (!is.null(fit4$samples$thr)) {
  thr_samps <- fit4$samples$thr
  n_save <- dim(thr_samps)[1]
  P4 <- dim(thr_samps)[2]
  Kminus1 <- dim(thr_samps)[3]
  thr_mat <- matrix(thr_samps, nrow = n_save, ncol = P4*Kminus1)

  thr_true_vec <- if (exists("thr_true_mat")) as.vector(thr_true_mat) else NA
  plot_group_box_ci(thr_mat, thr_true_vec, "thr (ordinal thresholds)", per_page = 24)
}

if (!is.null(fit4$samples$delta)) {
  delta_samps <- fit4$samples$delta
  n_save <- dim(delta_samps)[1]
  P4 <- dim(delta_samps)[2]
  Km2 <- dim(delta_samps)[3]
  delta_mat <- matrix(delta_samps, nrow = n_save, ncol = P4*Km2)
  delta_true_vec <- if (exists("delta_true")) as.vector(delta_true) else NA
  plot_group_box_ci(delta_mat, delta_true_vec, "delta (ordinal gaps)", per_page = 24)
}

dev.off()

# =========================================================
# 6) Posterior mean positions
# =========================================================
samps <- fit4$samples

A_hat  <- apply(samps$a,  c(2,3), mean)
B1_hat <- apply(samps$b1, c(2,3), mean)
B2_hat <- apply(samps$b2, c(2,3), mean)
B3_hat <- apply(samps$b3, c(2,3), mean)
B4_hat <- apply(samps$b4, c(2,3), mean)

# =========================================================
# 7) Plot TRUE vs estimated positions
# =========================================================
pdf("plot/4layered_v5_positions_true_vs_hat.pdf", width = 10, height = 5)
par(mfrow = c(1,2))
plot_positions(A_true, B1_true, B2_true, B3_true, B4_true,
               A_cl, B1_cl, B2_cl, B3_cl, B4_cl,
               title = "TRUE positions (4-layer)")
plot_positions(A_hat,  B1_hat,  B2_hat,  B3_hat,  B4_hat,
               A_cl, B1_cl, B2_cl, B3_cl, B4_cl,
               title = "Posterior mean positions (4-layer robust)")
dev.off()

# =========================================================
# 8) Distance recovery (per-layer gamma)
# =========================================================
gamma1_post <- mean(exp(fit4$samples$log_gamma1))
gamma2_post <- mean(exp(fit4$samples$log_gamma2))
gamma3_post <- mean(exp(fit4$samples$log_gamma3))
gamma4_post <- mean(exp(fit4$samples$log_gamma4))

D1_hat <- dist_mat(A_hat, B1_hat) * gamma1_post
D2_hat <- dist_mat(A_hat, B2_hat) * gamma2_post
D3_hat <- dist_mat(A_hat, B3_hat) * gamma3_post
D4_hat <- dist_mat(A_hat, B4_hat) * gamma4_post

D_true_all <- cbind(D1_true*gamma_true, D2_true*gamma_true, D3_true*gamma_true, D4_true*gamma_true)
D_hat_all  <- cbind(D1_hat,  D2_hat,  D3_hat,  D4_hat)

vec <- function(M) as.vector(M)

cor_all <- cor(vec(D_true_all), vec(D_hat_all))
cor_1 <- cor(vec(D1_true*gamma_true), vec(D1_hat))
cor_2 <- cor(vec(D2_true*gamma_true), vec(D2_hat))
cor_3 <- cor(vec(D3_true*gamma_true), vec(D3_hat))
cor_4 <- cor(vec(D4_true*gamma_true), vec(D4_hat))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("All layers (n x (P1+P2+P3+P4)): %.3f\n", cor_all))
cat(sprintf("Binary layer    (n x P1): %.3f\n", cor_1))
cat(sprintf("Continuous layer(n x P2): %.3f\n", cor_2))
cat(sprintf("Count layer     (n x P3): %.3f\n", cor_3))
cat(sprintf("Ordinal layer   (n x P4): %.3f\n", cor_4))

pdf("plot/4layered_v5_distance_recovery_scatter.pdf", width = 7, height = 7)
plot(
  vec(D_true_all), vec(D_hat_all),
  pch = 16, col = rgb(0, 0, 0, 0.5), cex = 0.5,
  xlab = "True distances x gamma_true",
  ylab = "Estimated distances x gamma_post",
  main = "Distance recovery: 4-layer robust LSIRM"
)
abline(a = 0, b = 1, col = "red", lwd = 2)
dev.off()

pdf("plot/4layered_v5_distance_recovery_by_layer.pdf", width = 10, height = 8)
par(mfrow = c(2,2), mar=c(3,3,2,1))

plot(vec(D1_true*gamma_true), vec(D1_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Binary", xlab="True", ylab="Hat")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D2_true*gamma_true), vec(D2_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Continuous (robust)", xlab="True", ylab="Hat")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D3_true*gamma_true), vec(D3_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Count", xlab="True", ylab="Hat")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D4_true*gamma_true), vec(D4_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Ordinal", xlab="True", ylab="Hat")
abline(a=0,b=1,col="red",lwd=2)

dev.off()
