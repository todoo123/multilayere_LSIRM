rm(list=ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile Rcpp (provided file)
# =========================================================
# If compilation fails, you may need to re-upload the .cpp file
# setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM")
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSM")
source("my_LSIRM_4layered_nonhierarchical.R")
# =========================================================
# 1) Simulation settings (as requested)
# =========================================================
n  <- 150
P2 <- 20  # continue
P1 <- 20  # binary
P3 <- 20  # count
P4 <- 20  # likert

d <- 2
gamma_true <- 0.7

# difficulty: 5 easy / 5 hard per item-type
# ability: 45 smart / 45 stupid
n_smart <- 45
n_stupid <- 45

# latent clusters (3) for respondents; items sampled around same centers
K <- 3

centers_resp <- rbind(
  c(-2,  -2),
  c( 0,  0),
  c( 2,  2)
)

centers_bin <- rbind(
  c(-2,  -2),
  c( 0,  0),
  c( 2,  2)
)

centers_con <- rbind(
  c(-2,  -2),
  c( 0,  0),
  c( 2,  2)
)

centers_cnt <- rbind(
  c(-2,  -2),
  c( 0,  0),
  c( 2,  2)
)

centers_likert <- rbind(
  c(-2,  -2),
  c( 0,  0),
  c( 2,  2)
)


sd_cluster_resp <- 0.40
sd_cluster_item <- 0.40

# noise/scale for each layer
sigma_likert_latent <- 1.0   # latent y* noise for ordinal
# for count NB2: Var = mu + kappa * mu^2  => size = 1/kappa
kappa_true <- 0.6

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

# Assign each item to a cluster center then sample around it
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

# True scales (choose values you want)
true_sigma_alpha <- 1.0   # sd of alpha (respondent main effect)
true_tau_beta1   <- 1.5   # sd of beta for binary items
true_tau_beta2   <- 1.0   # sd of beta for likert items
true_tau_beta3   <- 0.5   # sd of beta for count items

# --- ability (respondent): alpha_true ---
alpha_true <- rnorm(n, mean = 0, sd = true_sigma_alpha)

# (optional) keep your “smart/stupid 45/45” label purely for visualization
# NOTE: this is NOT forced; it's just labeling by ranks
ord_a <- order(alpha_true, decreasing = TRUE)
ability_group <- rep("stupid", n)
ability_group[ord_a[1:n_smart]] <- "smart"   # top 45 as "smart" label

# --- difficulty (item): beta1/2/3_true ---
beta1_true <- rnorm(P1, mean = 0, sd = true_tau_beta1)
beta2_true <- rnorm(P2, mean = 0, sd = true_tau_beta2)
beta3_true <- rnorm(P3, mean = 0, sd = true_tau_beta3)

# =========================================================
# NEW) True ordinal thresholds (u, delta) and beta4_list_true
# - MUST match your build_thresholds() in the MCMC code
# =========================================================
# build_thresholds <- function(u, delta_vec) {
#   # delta_vec length = (K-2) corresponding to k=2..K-1
#   Kminus1 <- 1 + length(delta_vec)
#   b <- numeric(Kminus1)
#   b[1] <- u
#   if (Kminus1 >= 2) {
#     for (k in 2:Kminus1) {
#       b[k] <- b[k - 1] + exp(delta_vec[k - 1])
#     }
#   }
#   b
# }
# softplus version
# numerically stable softplus
softplus <- function(x) {
  ifelse(x > 0,
         x + log1p(exp(-x)),
         log1p(exp(x)))
}

build_thresholds <- function(u, delta_vec) {
  # delta_vec length = (K-2) corresponding to k=2..K-1
  Kminus1 <- 1 + length(delta_vec)
  b <- numeric(Kminus1)
  b[1] <- u
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

u_true <- rnorm(P4, true_mu_u, true_sd_u)

# K=5 => K-2=3
K_ord <- 5
delta_true <- matrix(rnorm(P4 * (K_ord - 2), true_mu_delta, true_sd_delta),
                     nrow = P4, ncol = (K_ord - 2))

beta4_list_true <- vector("list", P4)
for (j in 1:P4) {
  beta4_list_true[[j]] <- build_thresholds(u_true[j], delta_true[j, ])
  # length = K-1 = 4
}
# 
# 
# # (optional) keep your “easy/hard 5/5 per layer” label for visualization
# # Here: "easy" = higher beta (easier item -> higher propensity)
# make_easyhard_label <- function(beta){
#   P <- length(beta)
#   ord <- order(beta, decreasing = TRUE)
#   grp <- rep("hard", P)
#   grp[ord[1:(P/2)]] <- "easy"
#   grp
# }
# diff1 <- make_easyhard_label(beta1_true)
# diff2 <- make_easyhard_label(beta2_true)
# diff3 <- make_easyhard_label(beta3_true)

# =========================================================
# 5) Distances and linear predictors
# =========================================================
dist_mat <- function(A, B){
  # returns n x P Euclidean distance
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
D4_true <- dist_mat(A_true, B4_true)   # n x P4
# test- beta 를 difficulty 로 분류해서 + 에서 - 로
ETA1_true <- outer(alpha_true, rep(1,P1)) - outer(rep(1,n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha_true, rep(1,P2)) - outer(rep(1,n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha_true, rep(1,P3)) - outer(rep(1,n), beta3_true) - gamma_true * D3_true
ETA4_true <- outer(alpha_true, rep(1, P4)) - gamma_true * D4_true
# =========================================================
# 6) Generate survey data:
#    - binary: Bernoulli(logit^{-1}(ETA1))
#    - likert(1..5): ordinal probit via latent y* = ETA2 + e, cutpoints
#    - count: NB2 with mu=exp(ETA3) and kappa_true
# =========================================================
invlogit <- function(x) 1/(1+exp(-x))

# binary
P_bern <- invlogit(ETA1_true)
Y_bin <- matrix(rbinom(n*P1, 1, as.vector(P_bern)), n, P1)

# continuous
cuts <- c(-1.5, -0.5, 0.5, 1.5)
Y_con <- ETA2_true + matrix(rnorm(n*P2, 0, sigma_likert_latent), n, P2)
storage.mode(Y_con) <- "numeric"  # your cpp takes numeric matrix

# count (NB2): Var=mu + kappa*mu^2  => size = 1/kappa
MU_cnt <- exp(ETA3_true)
size_nb <- 1 / kappa_true
Y_cnt <- matrix(rnbinom(n*P3, size=size_nb, mu=as.vector(MU_cnt)), n, P3)

# ordinal
Y_ord <- matrix(NA_integer_, n, P4)

for (j in 1:P4) {
  b_vec <- beta4_list_true[[j]]   # length K-1
  for (i in 1:n) {
    eta_ij <- ETA4_true[i, j]
    Cvec <- invlogit(eta_ij - b_vec)  # length K-1
    
    # category probabilities (length K)
    p <- numeric(K_ord)
    p[1] <- 1 - Cvec[1]
    p[K_ord] <- Cvec[K_ord - 1]
    if (K_ord > 2) {
      for (y in 2:(K_ord - 1)) {
        p[y] <- Cvec[y - 1] - Cvec[y]
      }
    }
    
    # numerical safety
    p[p < 0] <- 0
    ps <- sum(p)
    if (ps <= 0) {
      # fallback (should be rare): put mass on nearest category
      p <- rep(0, K_ord); p[round(K_ord/2)] <- 1
    } else {
      p <- p / ps
    }
    
    Y_ord[i, j] <- sample.int(K_ord, size = 1, prob = p)
  }
}
storage.mode(Y_ord) <- "integer"


# =========================================================
# 7) Plot TRUE positions (respondents + items by layer)
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
  
  plot(A, pch=21, bg="gray80", col="black",
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
# 8) Running Models
# =========================================================
# 
# # 1. R code
# fit4 <- lsirm_sharedpos_layer4_lsgrm(
#   Y_bin = Y_bin,
#   Y_con = Y_con,
#   Y_cnt = Y_cnt,
#   Y_ord = Y_ord,
#   d = 2, n_iter = 100000, burnin = 10000, thin = 5,
#   prop_sd = list(
#     # shared
#     alpha      = 0.50,
#     log_gamma  = 0.05,
#     a          = 0.50,
#     
#     # layer1..3
#     beta1      = 0.50,
#     beta2      = 0.30,
#     beta3      = 0.30,
#     b1         = 0.50,
#     b2         = 0.20,
#     b3         = 0.30,
#     log_kappa  = 0.20,
#     
#     # layer4 (ordinal)
#     b4         = 0.40,
#     u          = 0.30,
#     delta      = 0.30
#   ),
#   verbose = TRUE
# )

# 2. Rcpp code
source("my_LSIRM_4layered_nonhierarchical_cpp.R")
result <- lsirm_sharedpos_layer4_lsgrm_cpp(
    Y_bin, Y_con, Y_cnt, Y_ord,
    d = 2,
    n_iter = 10000,
    burnin = 1000,
    thin = 5,
    hyper = list(
      a_sigma=1, b_sigma=0.1,
      a_tau1=1, b_tau1=0.1, a_tau2=1, b_tau2=0.3, a_tau3=1, b_tau3=0.1,
      a_sigma0=1, b_sigma0=0.5,
      mu_log_gamma=0, sd_log_gamma=1,
      mu_log_kappa=0, sd_log_kappa=1,
      mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
    ),
    prop_sd = list(
      alpha=0.5, log_gamma=0.05, a=0.2,
      beta1=0.5, beta2=0.2, beta3=0.3,
      b1=0.5, b2=0.2, b3=0.3, b4=0.7,
      log_kappa=0.20, u=0.3, delta=0.3
    ),
    init = NULL,
    verbose = TRUE
)
# =========================================================
# 0) Acceptance rates
# =========================================================
result$accept
# fit4$accept
fit4<-result

# (optional) make sure plot dir exists
if (!dir.exists("plot")) dir.create("plot", recursive = TRUE)

# 결과가 'samples'라는 이름의 리스트 안에 들어있지 않고 바로 top-level에 있다면:
if (is.null(fit4$samples)) {
  # fit4 자체가 샘플 리스트라고 가정하고 복사
  fit4$samples <- fit4
}

#=========================================================
# 1) Traceplots: latent positions a, b1..b4
# =========================================================

pdf("plot/4layered_trace_a.pdf", width = 8, height = 12)
par(mfrow = c(4,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$a)[2]){
  for(j in 1:dim(fit4$samples$a)[3]){
    ts.plot(fit4$samples$a[,i,j], main = paste0('a: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_trace_b1.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b1)[2]){
  for(j in 1:dim(fit4$samples$b1)[3]){
    ts.plot(fit4$samples$b1[,i,j], main = paste0('b1: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_trace_b2.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b2)[2]){
  for(j in 1:dim(fit4$samples$b2)[3]){
    ts.plot(fit4$samples$b2[,i,j], main = paste0('b2: ', i, '_', j))
  }
}
dev.off()

pdf("plot/4layered_trace_b3.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b3)[2]){
  for(j in 1:dim(fit4$samples$b3)[3]){
    ts.plot(fit4$samples$b3[,i,j], main = paste0('b3: ', i, '_', j))
  }
}
dev.off()

# NEW: b4 trace
pdf("plot/4layered_trace_b4.pdf", width = 8, height = 12)
par(mfrow = c(2,2), mar=c(3,3,2,1))
for(i in 1:dim(fit4$samples$b4)[2]){
  for(j in 1:dim(fit4$samples$b4)[3]){
    ts.plot(fit4$samples$b4[,i,j], main = paste0('b4: ', i, '_', j))
  }
}
dev.off()

#-------------------------------------------------------------------------------
source("utils.R")

# shared legend
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
# 2) Traceplots: alpha, beta1..beta3 (same as before)
# =========================================================
pdf("plot/4layered_trace_alpha.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$alpha, alpha_true, "alpha", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_trace_beta1.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta1, beta1_true, "beta1", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_trace_beta2.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta2, beta2_true, "beta2", mfrow = c(3,2), leg = leg)
dev.off()

pdf("plot/4layered_trace_beta3.pdf", width = 8, height = 12)
plot_trace_vec(fit4$samples$beta3, beta3_true, "beta3", mfrow = c(3,2), leg = leg)
dev.off()

# =========================================================
# 3) NEW: Ordinal threshold params trace (u, delta, thr)
# =========================================================
# true thresholds matrix (P4 x (K-1)) if you have beta4_list_true
# - If you already have thr_true_mat, skip this block.
if (exists("beta4_list_true") && !exists("thr_true_mat")) {
  P4 <- length(beta4_list_true)
  Kminus1_true <- length(beta4_list_true[[1]])
  thr_true_mat <- matrix(NA, P4, Kminus1_true)
  for (j in 1:P4) thr_true_mat[j, ] <- beta4_list_true[[j]]
}

# u
if (exists("u_true")) {
  pdf("plot/4layered_trace_u.pdf", width = 8, height = 12)
  plot_trace_vec(fit4$samples$u, u_true, "u (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
} else {
  pdf("plot/4layered_trace_u.pdf", width = 8, height = 12)
  plot_trace_vec(fit4$samples$u, true = NA, "u (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
}

# delta (only if K>2 and you simulated delta_true)
if (!is.null(fit4$samples$delta)) {
  # flatten: (n_save x (P4*(K-2)))
  delta_samps <- fit4$samples$delta
  n_save <- dim(delta_samps)[1]
  P4     <- dim(delta_samps)[2]
  Km2    <- dim(delta_samps)[3]
  delta_mat <- matrix(delta_samps, nrow = n_save, ncol = P4*Km2)
  
  delta_true_vec <- if (exists("delta_true")) as.vector(delta_true) else rep(NA, P4*Km2)
  
  pdf("plot/4layered_trace_delta.pdf", width = 8, height = 12)
  plot_trace_vec(delta_mat, delta_true_vec, "delta (ordinal gaps)", mfrow = c(3,2), leg = leg)
  dev.off()
}

# thr = built thresholds b_vec (P4 x (K-1))
if (!is.null(fit4$samples$thr)) {
  thr_samps <- fit4$samples$thr
  n_save <- dim(thr_samps)[1]
  P4 <- dim(thr_samps)[2]
  Kminus1 <- dim(thr_samps)[3]
  thr_mat <- matrix(thr_samps, nrow = n_save, ncol = P4*Kminus1)
  
  thr_true_vec <- if (exists("thr_true_mat")) as.vector(thr_true_mat) else rep(NA, P4*Kminus1)
  
  pdf("plot/4layered_trace_thr.pdf", width = 8, height = 12)
  plot_trace_vec(thr_mat, thr_true_vec, "thresholds thr (ordinal)", mfrow = c(3,2), leg = leg)
  dev.off()
}

# =========================================================
# 4) Extra scalar parameters (add log_kappa too)
# =========================================================
pdf("plot/4layered_extra_parameter.pdf", width = 8, height = 12)
par(mfrow = c(3,2), mar=c(3,3,2,1))

# NOTE: In your current model, sigma0_sq is the continuous-layer residual variance.
# "true" should be the true residual variance you used to simulate Y_con (NOT sigma_likert_latent).
plot_trace_scalar(
  fit4$samples$sigma0_sq,
  true = if (exists("sigma0_true")) sigma0_true else NA,
  main = "sigma0_sq"
)

plot_trace_scalar(
  fit4$samples$log_gamma,
  true = gamma_true,
  main = "gamma",
  transform = exp
)

plot_trace_scalar(
  fit4$samples$log_kappa,
  true = if (exists("kappa_true")) kappa_true else NA,
  main = "kappa",
  transform = exp
)

plot_trace_scalar(
  fit4$samples$tau_beta1_sq,
  true = true_tau_beta1,
  main = "tau_beta1_sq (log)",
  transform = log
)

plot_trace_scalar(
  fit4$samples$tau_beta2_sq,
  true = true_tau_beta2,
  main = "tau_beta2_sq (log)",
  transform = log
)

plot_trace_scalar(
  fit4$samples$tau_beta3_sq,
  true = true_tau_beta3,
  main = "tau_beta3_sq (log)",
  transform = log
)
dev.off()

# =========================================================
# 5) Boxplots (add u + thr; delta optional)
# =========================================================
pdf("plot/4layered_boxplot_groups.pdf", width = 10, height = 8)

plot_group_box_ci(fit4$samples$alpha, alpha_true, "alpha", per_page = 24)
plot_group_box_ci(fit4$samples$beta1, beta1_true, "beta1", per_page = 24)
plot_group_box_ci(fit4$samples$beta2, beta2_true, "beta2", per_page = 24)
plot_group_box_ci(fit4$samples$beta3, beta3_true, "beta3", per_page = 24)

# NEW: ordinal u
plot_group_box_ci(fit4$samples$u, if (exists("u_true")) u_true else NA, "u (ordinal)", per_page = 24)

# NEW: ordinal thresholds thr (flattened)
if (!is.null(fit4$samples$thr)) {
  thr_samps <- fit4$samples$thr
  n_save <- dim(thr_samps)[1]
  P4 <- dim(thr_samps)[2]
  Kminus1 <- dim(thr_samps)[3]
  thr_mat <- matrix(thr_samps, nrow = n_save, ncol = P4*Kminus1)
  
  thr_true_vec <- if (exists("thr_true_mat")) as.vector(thr_true_mat) else NA
  plot_group_box_ci(thr_mat, thr_true_vec, "thr (ordinal thresholds)", per_page = 24)
}

# optional: delta boxplots
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
# 6) Posterior mean positions (after Procrustes alignment)
# =========================================================
samps <- fit4$samples

A_hat  <- apply(samps$a,  c(2,3), mean)
B1_hat <- apply(samps$b1, c(2,3), mean)
B2_hat <- apply(samps$b2, c(2,3), mean)
B3_hat <- apply(samps$b3, c(2,3), mean)
B4_hat <- apply(samps$b4, c(2,3), mean)

# =========================================================
# 7) Plot posterior mean positions (TRUE vs estimated)
# =========================================================
pdf("plot/4layered_positions_true_vs_hat.pdf", width = 10, height = 5)
par(mfrow = c(1,2))
plot_positions(A_true, B1_true, B2_true, B3_true, B4_true,
               A_cl, B1_cl, B2_cl, B3_cl, B4_cl,
               title = "TRUE positions (4-layer)")
plot_positions(A_hat,  B1_hat,  B2_hat,  B3_hat,  B4_hat,
               A_cl, B1_cl, B2_cl, B3_cl, B4_cl,
               title = "Posterior mean positions (4-layer)")
dev.off()

# =========================================================
# 8) Compare TRUE vs posterior mean distances (now 4 layers)
# =========================================================
gamma_post <- mean(exp(fit4$samples$log_gamma))

D1_hat <- dist_mat(A_hat, B1_hat) * gamma_post
D2_hat <- dist_mat(A_hat, B2_hat) * gamma_post
D3_hat <- dist_mat(A_hat, B3_hat) * gamma_post
D4_hat <- dist_mat(A_hat, B4_hat) * gamma_post

D_true_all <- cbind(D1_true, D2_true, D3_true, D4_true) * gamma_true
D_hat_all  <- cbind(D1_hat,  D2_hat,  D3_hat,  D4_hat)

vec <- function(M) as.vector(M)

cor_all <- cor(vec(D_true_all), vec(D_hat_all))
cor_1 <- cor(vec(D1_true), vec(D1_hat))
cor_2 <- cor(vec(D2_true), vec(D2_hat))
cor_3 <- cor(vec(D3_true), vec(D3_hat))
cor_4 <- cor(vec(D4_true), vec(D4_hat))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("All layers (n x (P1+P2+P3+P4)): %.3f\n", cor_all))
cat(sprintf("Binary layer    (n x P1): %.3f\n", cor_1))
cat(sprintf("Continuous layer(n x P2): %.3f\n", cor_2))
cat(sprintf("Count layer     (n x P3): %.3f\n", cor_3))
cat(sprintf("Ordinal layer   (n x P4): %.3f\n", cor_4))

pdf("plot/4layered_distance_recovery_scatter.pdf", width = 7, height = 7)
plot(
  vec(D_true_all), vec(D_hat_all),
  pch = 16, col = rgb(0, 0, 0, 0.5), cex = 0.5,
  xlab = "True distances × gamma_true",
  ylab = "Estimated distances × gamma_post",
  main = "Distance recovery: 4-layer LSIRM + LSGRM"
)
abline(a = 0, b = 1, col = "red", lwd = 2)
dev.off()

# Optional: per-layer scatter (4 panels)

pdf("plot/4layered_distance_recovery_by_layer.pdf", width = 10, height = 8)
par(mfrow = c(2,2), mar=c(3,3,2,1))

plot(vec(D1_true*gamma_true), vec(D1_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Binary", xlab="True (×gamma_true)", ylab="Hat (×gamma_post)")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D2_true*gamma_true), vec(D2_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Continuous", xlab="True (×gamma_true)", ylab="Hat (×gamma_post)")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D3_true*gamma_true), vec(D3_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Count", xlab="True (×gamma_true)", ylab="Hat (×gamma_post)")
abline(a=0,b=1,col="red",lwd=2)

plot(vec(D4_true*gamma_true), vec(D4_hat),
     pch=16, col=rgb(0,0,0,0.4), cex=0.5,
     main="Ordinal", xlab="True (×gamma_true)", ylab="Hat (×gamma_post)")
abline(a=0,b=1,col="red",lwd=2)

dev.off()

