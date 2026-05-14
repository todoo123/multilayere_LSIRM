rm(list=ls())

library(Rcpp)
library(vegan)

# =========================================================
# 0) Compile Rcpp (provided file)
# =========================================================
# If compilation fails, you may need to re-upload the .cpp file
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM")

sourceCpp("my_LSIRM_3layered_nonhierarchical.cpp")

# =========================================================
# 1) Simulation settings (as requested)
# =========================================================
n  <- 120
P2 <- 20  # continue
P1 <- 20  # binary
P3 <- 20  # count

d <- 2
gamma_true <- 0.7

# difficulty: 5 easy / 5 hard per item-type
# ability: 45 smart / 45 stupid
n_smart <- 45
n_stupid <- 45

# latent clusters (3) for respondents; items sampled around same centers
K <- 3

centers_resp <- rbind(
  c(-1,  -1),
  c( 0,  0),
  c( 1,  1)
)

centers_bin <- rbind(
  c(-1,  -1),
  c( 0,  0),
  c( 1,  1)
)

centers_con <- rbind(
  c(-1,  -1),
  c( 0,  0),
  c( 1,  1)
)

centers_cnt <- rbind(
  c(-1,  -1),
  c( 0,  0),
  c( 1,  1)
)

centers_likert <- rbind(
  c(-1,  -1),
  c( 0,  0),
  c( 1,  1)
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

# (optional) keep your “easy/hard 5/5 per layer” label for visualization
# Here: "easy" = higher beta (easier item -> higher propensity)
make_easyhard_label <- function(beta){
  P <- length(beta)
  ord <- order(beta, decreasing = TRUE)
  grp <- rep("hard", P)
  grp[ord[1:(P/2)]] <- "easy"
  grp
}
diff1 <- make_easyhard_label(beta1_true)
diff2 <- make_easyhard_label(beta2_true)
diff3 <- make_easyhard_label(beta3_true)

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

ETA1_true <- outer(alpha_true, rep(1,P1)) + outer(rep(1,n), beta1_true) - gamma_true * D1_true
ETA2_true <- outer(alpha_true, rep(1,P2)) + outer(rep(1,n), beta2_true) - gamma_true * D2_true
ETA3_true <- outer(alpha_true, rep(1,P3)) + outer(rep(1,n), beta3_true) - gamma_true * D3_true

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

# 5-point Likert (ordinal probit construction)
# latent y* = ETA2 + Normal(0, sigma^2), cutpoints -> 1..5
cuts <- c(-1.5, -0.5, 0.5, 1.5)
Y2_star <- ETA2_true + matrix(rnorm(n*P2, 0, sigma_likert_latent), n, P2)

Y_con <- matrix(NA, n, P2)
Y_con[Y2_star <= cuts[1]] <- 1
Y_con[Y2_star > cuts[1] & Y2_star <= cuts[2]] <- 2
Y_con[Y2_star > cuts[2] & Y2_star <= cuts[3]] <- 3
Y_con[Y2_star > cuts[3] & Y2_star <= cuts[4]] <- 4
Y_con[Y2_star > cuts[4]] <- 5
storage.mode(Y_con) <- "numeric"  # your cpp takes numeric matrix

# count (NB2): Var=mu + kappa*mu^2  => size = 1/kappa
MU_cnt <- exp(ETA3_true)
size_nb <- 1 / kappa_true
Y_cnt <- matrix(rnbinom(n*P3, size=size_nb, mu=as.vector(MU_cnt)), n, P3)
range(1,2,3,4)
# =========================================================
# 7) Plot TRUE positions (respondents + items by layer)
# =========================================================
plot_positions <- function(A, B1, B2, B3, A_cl, B1_cl, B2_cl, B3_cl, rng = NA, title = NA){
  if(sum(is.na(rng)) == TRUE){
    rng <- range(A[,1],A[,2],B1[,1],B1[,2],B2[,1],B2[,2],B3[,1],B3[,2])
  }else{
    rng <- rng
  }
  if(is.na(title) == TRUE){
    title = "TRUE latent positions: Respondents (gray), Items (bin:red, lik:orange, cnt:brown)"
  }else{
    title = title
  }
  base::plot(A, pch=21, bg="gray80", col="black",
       xlab="Dim1", ylab="Dim2",
       main=title,
       xlim=rng, ylim=rng)
  text(A, labels=A_cl, cex=0.6, pos=3, col="gray30")
  
  points(B1, pch=21, bg="red",    col="red")
  points(B2, pch=21, bg="orange", col="orange")
  points(B3, pch=21, bg="brown",  col="brown")
  
  text(B1, labels=B1_cl, cex=0.7, pos=4, col="red4")
  text(B2, labels=B2_cl, cex=0.7, pos=4, col="orange4")
  text(B3, labels=B3_cl, cex=0.7, pos=4, col="brown4")
  legend("topleft",
         legend=c("Respondents","Binary items","Likert items","Count items"),
         pch=21, pt.bg=c("gray80","red","orange","brown"), col=c("black","red","orange","brown"),
         bty="n")
}
par(mfrow = c(1,1))
plot_positions(A_true, B1_true, B2_true, B3_true, A_cl, B1_cl, B2_cl, B3_cl)

# =========================================================
# 8) Fit multilayer LSIRM (your wrapper, using Rcpp sampler)
# =========================================================
# NOTE: You provided lsirm_sharedpos_layer3() wrapper.
# Paste your wrapper function here (unchanged) then run fit.
source("my_LSIRM_3layered_nonhierarchical_cpp.R")
fit <- lsirm_sharedpos_layer3(
  Y_bin = Y_bin,
  Y_con = Y2_star,
  Y_cnt = Y_cnt,  
  # 5점 likert 척도로 바꾸다 보니 parameter 왜곡이 생겨서 수월한 simulation 을 위하여 일단 continuous 값으로 수행하였음
  d = d,
  n_iter = 100000,   # start small; increase later (e.g. 5000)
  burnin = 10000,
  thin = 5,
  verbose = TRUE,
  prop_sd = list(
    alpha      = 0.40,
    beta1      = 1.50,
    beta2      = 0.30,
    beta3      = 0.30,
    log_gamma  = 0.03,
    log_kappa  = 0.50,
    a          = 0.40,
    b1         = 0.50,
    b2         = 0.10,
    b3         = 0.30
  )
)

fit$accept

pdf("plot/multilayered_trace_a.pdf", width = 8, height = 12)
par(mfrow = c(4,2))
for(i in 1:dim(fit$samples$a)[2]){
  for(j in 1:dim(fit$samples$a)[3]){
    ts.plot(fit$samples$a[,i,j], main = paste0('a: ', i, '_',j))
  }
}
dev.off()

pdf("plot/multilayered_trace_b1.pdf", width = 8, height = 12)
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b1)[2]){
  for(j in 1:dim(fit$samples$b1)[3]){
    ts.plot(fit$samples$b1[,i,j], main = paste0('b1: ', i, '_',j))
  }
}
dev.off()

pdf("plot/multilayered_trace_b2.pdf", width = 8, height = 12)
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b2)[2]){
  for(j in 1:dim(fit$samples$b2)[3]){
    ts.plot(fit$samples$b2[,i,j], main = paste0('b2: ', i, '_',j))
  }
}
dev.off()

pdf("plot/multilayered_trace_b3.pdf", width = 8, height = 12)
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b3)[2]){
  for(j in 1:dim(fit$samples$b3)[3]){
    ts.plot(fit$samples$b3[,i,j], main = paste0('b3: ', i, '_',j))
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
pdf("plot/multilayered_trace_alpha.pdf", width = 8, height = 12)
plot_trace_vec(fit$samples$alpha, alpha_true, "alpha", mfrow = c(3,2), leg = leg)

pdf("plot/multilayered_trace_beta1.pdf", width = 8, height = 12)
plot_trace_vec(fit$samples$beta1, beta1_true, "beta1", mfrow = c(3,2), leg = leg)

# - beta2: difficulty of continuous item
# 이 부분이 true 값에서 가장 많이 벗어나 있음 - sampling 잘못되었을 가능성?
# likert 5 점 척도 data 지만, normal 분포에서 얻어지는 likert 5 점 척도임. 현재로서는.
# 그래서 5점 척도 없애고 일단 continuous data 그대로 사용하기로 했음 - 어느 정도 parameter 복원되는 것 같음
pdf("plot/multilayered_trace_beta2.pdf", width = 8, height = 12)
plot_trace_vec(fit$samples$beta2, beta2_true, "beta2", mfrow = c(3,2), leg = leg)

pdf("plot/multilayered_trace_beta3.pdf", width = 8, height = 12)
plot_trace_vec(fit$samples$beta3, beta3_true, "beta3", mfrow = c(3,2), leg = leg)
dev.off()

#-------------------------------------------------------------------------------
pdf("plot/extra_parameter.pdf", width = 8, height = 12)
par(mfrow = c(3,2))
plot_trace_scalar(
  fit$samples$sigma0_sq,
  true = sigma_likert_latent,
  main = "sigma0_sq"
)
# gamma 는 복원 안 되어도 상관 없음
plot_trace_scalar(
  fit$samples$log_gamma,
  true = gamma_true,
  main = "gamma",
  transform = exp
)

plot_trace_scalar(
  fit$samples$tau_beta1_sq,
  true = true_tau_beta1,
  main = "tau_beta1_sq (log)",
  transform = log
)

plot_trace_scalar(
  fit$samples$tau_beta2_sq,
  true = true_tau_beta2,
  main = "tau_beta2_sq (log)",
  transform = log
)

plot_trace_scalar(
  fit$samples$tau_beta3_sq,
  true = true_tau_beta3,
  main = "tau_beta3_sq (log)",
  transform = log
)
dev.off()

# boxplot
# 한 페이지에 24개씩 (너무 많으면 자동으로 page가 늘어남)
pdf("plot/multilayered_boxplot_groups.pdf", width = 10, height = 8)

plot_group_box_ci(fit$samples$alpha, alpha_true, "alpha", per_page = 24)
plot_group_box_ci(fit$samples$beta1, beta1_true, "beta1", per_page = 24)
plot_group_box_ci(fit$samples$beta2, beta2_true, "beta2", per_page = 24)
plot_group_box_ci(fit$samples$beta3, beta3_true, "beta3", per_page = 24)

dev.off()

# =========================================================
# 9) Posterior mean positions (after Procrustes alignment)
# =========================================================
samps <- fit$samples

A_hat  <- apply(samps$a,  c(2,3), mean)
B1_hat <- apply(samps$b1, c(2,3), mean)
B2_hat <- apply(samps$b2, c(2,3), mean)
B3_hat <- apply(samps$b3, c(2,3), mean)

# =========================================================
# 10) Plot posterior mean positions: 
# - geometry 가 아예 복원되지 않는 것은 아니다. 
# - 또한 전체적으로 scale down 된 것을 gamma 가 true 값보다 크게 되어 보상하는 것으로 보인다.
# =========================================================
par(mfrow = c(1,1))
plot_positions(A_true, B1_true, B2_true, B3_true, A_cl, B1_cl, B2_cl, B3_cl)
plot_positions(A_hat, B1_hat, B2_hat, B3_hat, A_cl, B1_cl, B2_cl, B3_cl)


# =========================================================
# 11) Compare TRUE vs posterior mean distances (n x P)
#     (This is the key for later "binary-only vs multilayer" comparison)
# =========================================================
gamma_true
gamma_post <- mean(exp(fit$samples$log_gamma))
D1_hat <- dist_mat(A_hat, B1_hat)*gamma_post
D2_hat <- dist_mat(A_hat, B2_hat)*gamma_post
D3_hat <- dist_mat(A_hat, B3_hat)*gamma_post

# concatenate (n x (P1+P2+P3))
D_true_all <- cbind(D1_true, D2_true, D3_true)*gamma_true
D_hat_all  <- cbind(D1_hat,  D2_hat,  D3_hat)

vec <- function(M) as.vector(M)

cor_all <- cor(vec(D_true_all), vec(D_hat_all))
cor_1 <- cor(vec(D1_true), vec(D1_hat))
cor_2 <- cor(vec(D2_true), vec(D2_hat))
cor_3 <- cor(vec(D3_true), vec(D3_hat))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("All layers (n x 30): %.3f\n", cor_all))
cat(sprintf("Binary layer   (n x 10): %.3f\n", cor_1))
cat(sprintf("Likert layer   (n x 10): %.3f\n", cor_2))
cat(sprintf("Count layer    (n x 10): %.3f\n", cor_3))

base::plot(
  vec(D_true_all), vec(D_hat_all),
  pch = 16, col = rgb(0, 0, 0, 0.5), cex = 0.5,
  xlab = "True distances",
  ylab = "Estimated distances",
  main = "Distance recovery: multilayered LSIRM",
  xlim = c(0,4),
  ylim = c(0,4)
)
abline(a = 0, b = 1, col = "red", lwd = 2)

# 이 부분은 그렇게 LSIRM 과 multilayered LSIRM 의 차이를 잘 구분해 내지 못해서 제외함
# # =========================================================
# # 12) 더 직관적으로 item - response 간의 geometry 가 얼마나 비슷한지 확인
# # svd 를 이용해서 각 item / respondent 의 상대적인 거리 rank 기반 관계가 얼마나 유사한지를
# # cosine similarity 를 이용하여 확인할 수 있음
# # =========================================================
# source("utils.R")
# 
# out <- rank_svd_compare_refactored(D_true_all, D_hat_all,
#                                    k = 4,
#                                    plotting = TRUE,
#                                    center = TRUE,
#                                    rank_mode = "row")
# 
# out$similarity$subspace_U
# respondent_cosine_aligned <- abs(out$similarity$respondent_cosine)
# summary(respondent_cosine_aligned)
# 
# # 
# # out <- rank_svd_compare_refactored(D_true_all, D_hat_all,
# #                                    k = 4,
# #                                    plotting = TRUE,
# #                                    center = TRUE,
# #                                    rank_mode = "col")
# # 
# # out$similarity$subspace_V
# # item_cosine_aligned <- abs(out$similarity$item_cosine)
# # summary(item_cosine_aligned)
# # 


# =========================================================
# 13-1) 이제는 모든 data 를 binary 화 해서 추정 돌리고 Parameter 잘 복원하는지, 거리 관계 복원하는지 비교할 것
# =========================================================
source("my_LSIRM_cpp.R")

binarize_by_colmean <- function(X, strict = TRUE){
  X <- as.matrix(X)
  cm <- colMeans(X, na.rm = TRUE)
  
  if(strict){
    X_bin <- sweep(X, 2, cm, FUN = ">")   # TRUE/FALSE
  } else {
    X_bin <- sweep(X, 2, cm, FUN = ">=")  # 동률 포함
  }
  
  storage.mode(X_bin) <- "integer"
  list(X_bin = X_bin, colmean = cm)
}

# -----------------------------
# 2) Y_cnt, Y2_star 이진화
# -----------------------------
out_cnt  <- binarize_by_colmean(Y_cnt, strict = TRUE)
Y_cnt_bin <- out_cnt$X_bin

out_y2   <- binarize_by_colmean(Y2_star, strict = TRUE)
Y2_star_bin <- out_y2$X_bin

# -----------------------------
# 3) Y_bin 과 column-wise 결합
# -----------------------------
Y_bin_all <- cbind(Y_bin, Y2_star_bin, Y_cnt_bin)

fit_2 <- lsirm_basic(
  Y_bin = Y_bin_all,
  d = d,
  n_iter = 100000,
  burnin = 10000,
  thin = 5,
  prop_sd = list(
    alpha = 0.70,
    beta  = 0.50,
    log_gamma = 0.05,
    a = 0.50,
    b = 0.30
  ),
  verbose = TRUE
)

# =========================================================
# 13-2) 결과 확인
# =========================================================
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

pdf("plot/unilayered_trace_alpha.pdf", width = 8, height = 12)
plot_trace_vec(
  samples_mat = fit_2$samples$alpha,
  true_vec    = alpha_true,
  name        = "alpha",
  mfrow       = c(3,2),
  leg         = leg
)
dev.off()

pdf("plot/unilayered_trace_beta.pdf", width = 8, height = 12)
beta_true <- c(beta1_true, beta2_true, beta3_true)
plot_trace_vec(
  samples_mat = fit_2$samples$beta,
  true_vec    = beta_true,
  name        = "beta",
  mfrow       = c(3,2),
  leg         = leg
)
dev.off()

pdf("plot/unilayered_trace_extra_parameter.pdf", width = 8, height = 12)
par(mfrow = c(3,2))
plot_trace_scalar(
  x    = fit_2$samples$log_gamma,
  true = log(gamma_true),
  main = "log_gamma"
)

plot_trace_scalar(
  x         = fit_2$samples$tau_beta_sq,
  true      = true_tau_beta1,
  main      = "tau_beta_sq (log)",
  transform = log
)

plot_trace_scalar(
  x         = fit_2$samples$sigma_alpha_sq,
  true      = true_sigma_alpha,
  main      = "sigma_alpha_sq (log)",
  transform = log
)
dev.off()




pdf("plot/unilayered_boxplot_groups.pdf", width = 10, height = 8)

plot_group_box_ci(fit_2$samples$alpha, alpha_true, "alpha", per_page = 24)

plot_group_box_ci(fit_2$samples$beta[,1:20], beta1_true, "beta", per_page = 24)
plot_group_box_ci(fit_2$samples$beta[,21:40], beta2_true, "beta", per_page = 24)
plot_group_box_ci(fit_2$samples$beta[,41:60], beta3_true, "beta", per_page = 24)

scalar_samples <- cbind(
  log_gamma       = fit_2$samples$log_gamma,
  log_tau_beta    = log(fit_2$samples$tau_beta_sq),
  log_sigma_alpha = log(fit_2$samples$sigma_alpha_sq)
)
scalar_true <- c(
  log_gamma       = log(gamma_true),
  log_tau_beta    = log(true_tau_beta1),
  log_sigma_alpha = log(true_sigma_alpha)
)

plot_group_box_ci(scalar_samples, scalar_true, "scalars(log)", per_page = 24)

dev.off()






samps_2 <- fit_2$samples
S <- dim(samps_2$a)[1]

A_hat_2  <- apply(samps_2$a,  c(2,3), mean)
B_hat_2 <- apply(samps_2$b, c(2,3), mean)
dim(B_hat_2)
B1_hat_2 <- B_hat_2[1:20,]
B2_hat_2 <- B_hat_2[21:40,]
B3_hat_2 <- B_hat_2[41:60,]

par(mfrow = c(1,1))
plot_positions(A_true, B1_true, B2_true, B3_true, A_cl, B1_cl, B2_cl, B3_cl, rng = c(-2.1,2.1), title = "true position")
plot_positions(A_hat, B1_hat, B2_hat, B3_hat, A_cl, B1_cl, B2_cl, B3_cl, rng = c(-2.1,2.1), title = "multilayered LSIRM")
plot_positions(A_hat_2, B1_hat_2, B2_hat_2, B3_hat_2, A_cl, B1_cl, B2_cl, B3_cl, rng = c(-2.1,2.1), title = "LSIRM")
# =======================================================
# geometry 복원 여부
# =======================================================

gamma_post_2 <- mean(exp(fit_2$samples$log_gamma))
D1_hat_2 <- dist_mat(A_hat_2, B1_hat_2)*gamma_post_2
D2_hat_2 <- dist_mat(A_hat_2, B2_hat_2)*gamma_post_2
D3_hat_2 <- dist_mat(A_hat_2, B3_hat_2)*gamma_post_2

# concatenate (n x (P1+P2+P3))
D_true_all <- cbind(D1_true, D2_true, D3_true)*gamma_true
D_hat_all_2  <- cbind(D1_hat_2,  D2_hat_2,  D3_hat_2)

vec <- function(M) as.vector(M)

cor_all_2 <- cor(vec(D_true_all), vec(D_hat_all_2))
cor_1_2 <- cor(vec(D1_true), vec(D1_hat_2))
cor_2_2 <- cor(vec(D2_true), vec(D2_hat_2))
cor_3_2 <- cor(vec(D3_true), vec(D3_hat_2))

cat("\n=== Distance recovery (correlation) ===\n")
cat(sprintf("All layers (n x 30): %.3f\n", cor_all_2))
cat(sprintf("Binary layer   (n x 10): %.3f\n", cor_1_2))
cat(sprintf("Likert layer   (n x 10): %.3f\n", cor_2_2))
cat(sprintf("Count layer    (n x 10): %.3f\n", cor_3_2))

base::plot(
  vec(D_true_all), vec(D_hat_all_2),
  pch = 16, col = rgb(0, 0, 0, 0.5), cex = 0.5,
  xlab = "True distances",
  ylab = "Estimated distances",
  main = "Distance recovery: LSIRM",
  xlim = c(0,4),
  ylim = c(0,4)
)
abline(a = 0, b = 1, col = "red", lwd = 2)

out_2 <- rank_svd_compare_refactored(D_true_all, D_hat_all_2,
                                   k = 4,
                                   plotting = TRUE,
                                   center = TRUE,
                                   rank_mode = "row")

out_2$similarity$subspace_U
respondent_cosine_aligned <- abs(out_2$similarity$respondent_cosine)
summary(respondent_cosine_aligned)
# 
# out_2 <- rank_svd_compare_refactored(D_true_all, D_hat_all_2,
#                                    k = 10,
#                                    plotting = TRUE,
#                                    center = TRUE,
#                                    rank_mode = "col")
# 
# out_2$similarity$subspace_V
# item_cosine_aligned <- abs(out_2$similarity$item_cosine)
# summary(item_cosine_aligned)
# 


# =================================================
# real data analysis - KSHA
# =================================================
source("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM/KSAH_dataprocessing.R")
fit <- lsirm_sharedpos_layer3(
  Y_bin = Y_bin,
  Y_con = Y_likert_5,
  Y_cnt = Y_cnt,  
  # 5점 likert 척도로 바꾸다 보니 parameter 왜곡이 생겨서 수월한 simulation 을 위하여 일단 continuous 값으로 수행하였음
  d = d,
  n_iter = 100000,   # start small; increase later (e.g. 5000)
  burnin = 10000,
  thin = 5,
  verbose = TRUE,
  prop_sd = list(
    alpha      = 0.40,
    beta1      = 1.50,
    beta2      = 0.30,
    beta3      = 0.30,
    log_gamma  = 0.03,
    log_kappa  = 0.50,
    a          = 0.40,
    b1         = 0.50,
    b2         = 0.10,
    b3         = 0.30
  )
)


out_cnt  <- binarize_by_colmean(Y_cnt, strict = TRUE)
Y_cnt_bin <- out_cnt$X_bin

out_y2   <- binarize_by_colmean(Y_likert_5, strict = TRUE)
Y_likert_5_bin <- out_y2$X_bin

Y_bin_all <- cbind(Y_bin, Y_likert_5_bin, Y_cnt_bin)

fit_2 <- lsirm_basic(
  Y_bin = Y_bin_all,
  d = d,
  n_iter = 100000,
  burnin = 10000,
  thin = 5,
  prop_sd = list(
    alpha = 0.70,
    beta  = 0.50,
    log_gamma = 0.05,
    a = 0.50,
    b = 0.30
  ),
  verbose = TRUE
)


samps <- fit$samples

A  <- apply(samps$a,  c(2,3), mean)
B1 <- apply(samps$b1, c(2,3), mean)
B2 <- apply(samps$b2, c(2,3), mean)
B3 <- apply(samps$b3, c(2,3), mean)
rng <- range(A[,1],A[,2],B1[,1],B1[,2],B2[,1],B2[,2],B3[,1],B3[,2])
par(mfrow = c(1,1))
base::plot(A, pch=21, bg="gray80", col="black",
           xlab="Dim1", ylab="Dim2",
           main="multilayered LSIRM",
           xlim=rng, ylim=rng)
text(A, labels=1:dim(A)[1], cex=0.6, pos=3, col="gray30")

points(B1, pch=21, bg="red",    col="red")
points(B2, pch=21, bg="orange", col="orange")
points(B3, pch=21, bg="brown",  col="brown")

text(B1, labels=1:dim(B1)[1], cex=0.7, pos=4, col="red4")
text(B2, labels=1:dim(B2)[1], cex=0.7, pos=4, col="orange4")
text(B3, labels=1:dim(B3)[1], cex=0.7, pos=4, col="brown4")
legend("topleft",
       legend=c("Respondents","Binary items","Likert items","Count items"),
       pch=21, pt.bg=c("gray80","red","orange","brown"), col=c("black","red","orange","brown"),
       bty="n")

# count 1,2 는 사교육/주중 수면시간
# binary 는 최근 3개월 간 과식 여부/완하제 복용 여부

samps_2 <- fit_2$samples
A_hat_2  <- apply(samps_2$a,  c(2,3), mean)
B_hat_2 <- apply(samps_2$b, c(2,3), mean)


rng <- range(A_hat_2[,1],A_hat_2[,2],B_hat_2[,1], B_hat_2[,2])
A <-A_hat_2
B1<-B_hat_2[1:2,]
B2<-B_hat_2[3:(3+38),]
B3<-B_hat_2[(3+38+1):(3+38+1+1),]

par(mfrow = c(1,1))
base::plot(A, pch=21, bg="gray80", col="black",
           xlab="Dim1", ylab="Dim2",
           main="LSIRM",
           xlim=rng, ylim=rng)
text(A, labels=1:dim(A)[1], cex=0.6, pos=3, col="gray30")

points(B1, pch=21, bg="red",    col="red")
points(B2, pch=21, bg="orange", col="orange")
points(B3, pch=21, bg="brown",  col="brown")

text(B1, labels=1:dim(B1)[1], cex=0.7, pos=4, col="red4")
text(B2, labels=1:dim(B2)[1], cex=0.7, pos=4, col="orange4")
text(B3, labels=1:dim(B3)[1], cex=0.7, pos=4, col="brown4")
legend("topleft",
       legend=c("Respondents","Binary items","Likert items","Count items"),
       pch=21, pt.bg=c("gray80","red","orange","brown"), col=c("black","red","orange","brown"),
       bty="n")
