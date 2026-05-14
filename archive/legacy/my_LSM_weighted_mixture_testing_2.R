
# ------------------------------------------------------------------
# R 구현: Mixture Latent Space Model (Bayesian MCMC, 정리 버전)
# ------------------------------------------------------------------
# 모델 요약:
# Y_ij = log(W_ij)
#
# Likelihood (Mixture):
# Y_ij | i_ij = 1 ~ N(eta_ij, sigma_0^2)
# Y_ij | i_ij = 0 ~ N(eta_shrink, sigma_shrink^2)
#
# eta_ij = theta_i + theta_j - beta(=1) * d(z_i, z_j)
#
# Priors:
# i_ij ~ Bernoulli(p)
# p ~ Beta(a_p, b_p)
# eta_shrink ~ N(mu_shrink, var_shrink)
# sigma_shrink^2 ~ IG(a_shrink, b_shrink)
# sigma_0^2 ~ IG(a_0, b_0)
# theta_i ~ N(0, var_theta)
# beta ~ Gamma(a_beta, b_beta)
# z_i ~ N(0, sigma_1^2 * I)
# sigma_1^2 ~ IG(a_sigma1, b_sigma1)
# ------------------------------------------------------------------


suppressPackageStartupMessages({
  library(MASS)  # mvrnorm
  library(stats) # dist
  library(vegan) # procrustes
  library(igraph)
  library(brainGraph)
})

# ....................................
# 헬퍼 함수
# ....................................

# Inverse Gamma 샘플링
rinvgamma <- function(n, shape, rate) {
  1 / rgamma(n, shape = shape, rate = rate)
}

# 로그-우도 계산 (i_ij = 1인 edge에 대해서만)
calc_log_lik_1 <- function(Y, eta, sigma_0_sq, I_idx_1) {
  if (length(I_idx_1) == 0) return(0)
  sum(dnorm(Y[I_idx_1], eta[I_idx_1], sqrt(sigma_0_sq), log = TRUE))
}

# L2 거리 행렬 (Rcpp 버전 L2_dist_cpp와 인터페이스 맞추기용)
L2_dist <- function(Z) {
  as.matrix(dist(Z))
}

# eta_ij 행렬 계산
calculate_eta_matrix <- function(theta, beta, dist_mat) {
  N <- length(theta)
  # eta_ij = theta_i + theta_j - beta * d_ij
  theta_sum <- outer(theta, theta, "+")
  theta_sum - beta * dist_mat
}

# z variable 의 전체 posterior likelihood 계산 (likelihood + prior)
calc_log_post_z <- function(Z, Y_vec, upper_tri_indices, I_idx_1,
                            theta, beta, sigma_0_sq, sigma_1_sq) {
  N <- nrow(Z)
  
  # 현재 eta 계산
  dist_mat <- L2_dist(Z)
  eta_mat <- calculate_eta_matrix(theta, beta, dist_mat)
  eta_vec <- eta_mat[upper_tri_indices]
  
  # log-likelihood (i_ij = 1인 edge에 대해서만)
  log_lik <- calc_log_lik_1(Y_vec, eta_vec, sigma_0_sq, I_idx_1)
  
  # prior(Z_i)
  log_prior_z <- sum(dnorm(Z, 0, sqrt(sigma_1_sq), log = TRUE))
  
  return(log_lik + log_prior_z)
}

# ------------------------------------------------------------------
# 메인 MCMC 함수
# ------------------------------------------------------------------

run_mixture_lsm_mcmc <- function(W, 
                                 K = 2,           # 잠재 공간 차원
                                 n_iter = 10000, 
                                 burn_in = 5000, 
                                 thin = 5,
                                 # 하이퍼파라미터
                                 priors = list(
                                   a_p = 1, b_p = 1,
                                   mu_shrink = 0, var_shrink = 10,
                                   a_shrink = 0.05, b_shrink = 0.05,
                                   a_0 = 0.05, b_0 = 0.05,
                                   # mu_alpha = 0, var_alpha = 1,
                                   var_theta = 1,
                                   a_beta = 1, b_beta = 1,     # Gamma(a_beta, b_beta)
                                   a_sigma1 = 0.05, b_sigma1 = 0.05,
                                   prop_sd_beta = 0.1,
                                   prop_sd_z = 0.1
                                 )) {
  
  cat("MCMC 시작...\n")
  
  # -------------------------
  # 데이터 준비
  # -------------------------
  # log(0) 방지용 작은 epsilon
  eps <- 1e-8
  Y <- log(pmax(W, eps))
  N <- nrow(Y)
  
  # 상삼각 인덱스 (i < j만 사용: undirected network 가정)
  upper_tri_indices <- which(upper.tri(Y), arr.ind = FALSE)
  n_pairs <- length(upper_tri_indices)
  
  Y_vec <- Y[upper_tri_indices]
  
  # (i, j) 쌍 인덱스
  pair_indices <- which(upper.tri(Y), arr.ind = TRUE)
  row_indices <- pair_indices[, 1]
  col_indices <- pair_indices[, 2]
  
  # -------------------------
  # 샘플 저장 공간
  # -------------------------
  n_save <- floor((n_iter - burn_in) / thin)
  if (n_save <= 0) stop("n_iter가 burn_in/thin 보다 커야 합니다.")
  
  samples <- list(
    p              = numeric(n_save),
    eta_shrink     = numeric(n_save),
    sigma_shrink_sq= numeric(n_save),
    sigma_0_sq     = numeric(n_save),
    # alpha          = numeric(n_save),
    theta          = array(0, dim = c(N, n_save)),
    beta           = numeric(n_save),
    Z              = array(0, dim = c(N, K, n_save)),
    sigma_1_sq     = numeric(n_save),
    I_mat          = array(0, dim = c(N, N, n_save)),  # P(i_ij=1 | Y)의 사후 평균
    ll_z           = numeric(n_save)
  )
  
  # -------------------------
  # 초기값 설정
  # -------------------------
  # 인디케이터 I (upper triangle만)
  I_vec <- rbinom(n_pairs, 1, 0.5)
  
  # 잠재 좌표 Z
  Z <- matrix(rnorm(N * K, 0, 1), N, K)
  
  # Parameter P 초기값
  P_curr <- 0.5
  
  # shrink 컴포넌트 초기값
  if (any(I_vec == 0)) {
    eta_shrink_curr <- mean(Y_vec[I_vec == 0])
    sigma_shrink_sq_curr <- var(Y_vec[I_vec == 0])
  } else {
    eta_shrink_curr <- 0
    sigma_shrink_sq_curr <- 1
  }
  if (is.nan(eta_shrink_curr)) eta_shrink_curr <- 0
  if (is.na(sigma_shrink_sq_curr) || sigma_shrink_sq_curr <= 0) sigma_shrink_sq_curr <- 1
  
  # LSM 컴포넌트 초기값
  if (any(I_vec == 1)) {
    sigma_0_sq_curr <- var(Y_vec[I_vec == 1])
    # alpha_curr <- mean(Y_vec[I_vec == 1])
  } else {
    sigma_0_sq_curr <- 1
    # alpha_curr <- 0
  }
  if (is.na(sigma_0_sq_curr) || sigma_0_sq_curr <= 0) sigma_0_sq_curr <- 1
  # if (is.nan(alpha_curr)) alpha_curr <- 0
  
  theta_curr <- rnorm(N, 0, 0.1)    # centering 제거 (베이즈 타겟 그대로)
  beta_curr  <- 1.0
  sigma_1_sq_curr <- 1.0
  
  # MH 수락 횟수 카운트
  accept_beta <- 0
  accept_z <- numeric(N)
  
  # -------------------------
  # MCMC 루프
  # -------------------------
  save_idx <- 1
  
  for (iter in 1:n_iter) {
    
    if (iter %% 100 == 0) {
      cat(sprintf("반복 %d/%d...\n", iter, n_iter))
    }
    
    # 현재 I=1, I=0 인덱스
    I_idx_1 <- which(I_vec == 1)
    I_idx_0 <- which(I_vec == 0)
    n_1 <- length(I_idx_1)
    n_0 <- length(I_idx_0)
    
    # -------------------------
    # 1. p 업데이트 (Beta)
    # -------------------------
    post_a_p <- priors$a_p + n_1
    post_b_p <- priors$b_p + n_0
    P_curr <- rbeta(1, post_a_p, post_b_p)
    
    # -------------------------
    # 2. eta_shrink 업데이트 (Normal)
    # -------------------------
    post_var_shrink <- 1 / (n_0 / sigma_shrink_sq_curr + 1 / priors$var_shrink)
    sum_Y_0 <- if (n_0 > 0) sum(Y_vec[I_idx_0]) else 0
    post_mean_shrink <- post_var_shrink * 
      (sum_Y_0 / sigma_shrink_sq_curr + priors$mu_shrink / priors$var_shrink)
    eta_shrink_curr <- rnorm(1, post_mean_shrink, sqrt(post_var_shrink))
    eta_shrink_curr <- 1
    # -------------------------
    # 3. sigma_shrink^2 업데이트 (IG)
    # -------------------------
    post_a_shrink <- priors$a_shrink + n_0 / 2
    sum_sq_err_0 <- if (n_0 > 0) sum((Y_vec[I_idx_0] - eta_shrink_curr)^2) else 0
    post_b_shrink <- priors$b_shrink + 0.5 * sum_sq_err_0
    sigma_shrink_sq_curr <- rinvgamma(1, post_a_shrink, post_b_shrink)
    sigma_shrink_sq_curr <- 0.1
    # -------------------------
    # 4. sigma_0^2 업데이트 (IG)
    # -------------------------
    # 현재 Z 기준 거리/eta 계산
    dist_mat <- L2_dist(Z)
    eta_mat <- calculate_eta_matrix(theta_curr, beta_curr, dist_mat)
    eta_vec <- eta_mat[upper_tri_indices]
    
    post_a_0 <- priors$a_0 + n_1 / 2
    sum_sq_err_1 <- if (n_1 > 0) sum((Y_vec[I_idx_1] - eta_vec[I_idx_1])^2) else 0
    post_b_0 <- priors$b_0 + 0.5 * sum_sq_err_1
    sigma_0_sq_curr <- rinvgamma(1, post_a_0, post_b_0)
    
    # # -------------------------
    # # 5. alpha 업데이트 (Normal)
    # # -------------------------
    # post_var_alpha <- 1 / (n_1 / sigma_0_sq_curr + 1 / priors$var_alpha)
    # 
    # if (n_1 > 0) {
    #   theta_i_vec <- theta_curr[row_indices[I_idx_1]]
    #   theta_j_vec <- theta_curr[col_indices[I_idx_1]]
    #   dist_vec    <- dist_mat[upper_tri_indices][I_idx_1]
    #   
    #   residuals <- Y_vec[I_idx_1] - (theta_i_vec + theta_j_vec + beta_curr * dist_vec)
    #   sum_res   <- sum(residuals)
    # } else {
    #   sum_res <- 0
    # }
    # 
    # post_mean_alpha <- post_var_alpha * 
    #   (sum_res / sigma_0_sq_curr + priors$mu_alpha / priors$var_alpha)
    # alpha_curr <- rnorm(1, post_mean_alpha, sqrt(post_var_alpha))
    
    # -------------------------
    # 6. theta_i 업데이트 (Normal, node-wise)
    # -------------------------
    for (i in 1:N) {
      involved_idx_1 <- which((row_indices == i | col_indices == i) & I_vec == 1)
      n_i <- length(involved_idx_1)
      
      post_var_theta_i <- 1 / (n_i / sigma_0_sq_curr + 1 / priors$var_theta)
      
      if (n_i > 0) {
        js <- ifelse(row_indices[involved_idx_1] == i,
                     col_indices[involved_idx_1],
                     row_indices[involved_idx_1])
        
        res_i <- Y_vec[involved_idx_1] - 
          (theta_curr[js] - beta_curr * dist_mat[i, js])
        sum_res_i <- sum(res_i)
      } else {
        sum_res_i <- 0
      }
      
      post_mean_theta_i <- post_var_theta_i * (sum_res_i / sigma_0_sq_curr)
      theta_curr[i] <- rnorm(1, post_mean_theta_i, sqrt(post_var_theta_i))
    }
    # (중요) 더 이상 theta를 centering 하지 않음 (베이즈 타겟 분포 그대로 유지)
    
    # -------------------------
    # 7. beta 업데이트 (MH, log-normal proposal)
    # -------------------------
    # 현재 eta, log-likelihood
    eta_mat <- calculate_eta_matrix(theta_curr, beta_curr, dist_mat)
    eta_vec <- eta_mat[upper_tri_indices]
    log_lik_curr <- calc_log_lik_1(Y_vec, eta_vec, sigma_0_sq_curr, I_idx_1)
    
    # proposal: log(beta') ~ N(log(beta), prop_sd_beta^2)
    log_beta_prop <- rnorm(1, log(beta_curr), priors$prop_sd_beta)
    beta_prop <- exp(log_beta_prop)
    
    # 제안값에서의 eta, log-likelihood
    eta_mat_prop <- calculate_eta_matrix(theta_curr, beta_prop, dist_mat)
    eta_vec_prop <- eta_mat_prop[upper_tri_indices]
    log_lik_prop <- calc_log_lik_1(Y_vec, eta_vec_prop, sigma_0_sq_curr, I_idx_1)
    
    # prior (Gamma)
    log_prior_curr <- dgamma(beta_curr, priors$a_beta, priors$b_beta, log = TRUE)
    log_prior_prop <- dgamma(beta_prop, priors$a_beta, priors$b_beta, log = TRUE)
    
    # Jacobian: log-normal proposal (W=log beta)
    log_jacobian <- log(beta_prop) - log(beta_curr)
    
    # log MH 비
    log_r <- (log_lik_prop + log_prior_prop) - (log_lik_curr + log_prior_curr) + log_jacobian
    
    if (log(runif(1)) < log_r) {
      beta_curr <- beta_prop
      accept_beta <- accept_beta + 1
    }
    # test: beta 값 고정
    beta_curr <- 1
    
    # -------------------------
    # 8. Z_i 업데이트 (MH, node-wise)
    # -------------------------
    for (i in 1:N) {
      
      # 현재 전체 log-likelihood
      dist_mat <- L2_dist(Z)
      eta_mat <- calculate_eta_matrix(theta_curr, beta_curr, dist_mat)
      eta_vec <- eta_mat[upper_tri_indices]
      log_lik_curr_z <- calc_log_lik_1(Y_vec, eta_vec, sigma_0_sq_curr, I_idx_1)
      
      # proposal: Z_i' ~ N(Z_i, prop_sd_z^2 I_K)
      Z_prop <- Z
      Z_prop[i, ] <- mvrnorm(1, Z[i, ], diag(priors$prop_sd_z^2, K))
      
      dist_mat_prop <- L2_dist(Z_prop)
      eta_mat_prop <- calculate_eta_matrix(theta_curr, beta_curr, dist_mat_prop)
      eta_vec_prop <- eta_mat_prop[upper_tri_indices]
      log_lik_prop_z <- calc_log_lik_1(Y_vec, eta_vec_prop, sigma_0_sq_curr, I_idx_1)
      
      # prior(Z_i)
      log_prior_curr_z <- sum(dnorm(Z[i, ], 0, sqrt(sigma_1_sq_curr), log = TRUE))
      log_prior_prop_z <- sum(dnorm(Z_prop[i, ], 0, sqrt(sigma_1_sq_curr), log = TRUE))
      
      # MH 비
      log_r_z <- (log_lik_prop_z + log_prior_prop_z) - (log_lik_curr_z + log_prior_curr_z)
      
      if (log(runif(1)) < log_r_z) {
        Z <- Z_prop
        dist_mat <- dist_mat_prop
        accept_z[i] <- accept_z[i] + 1
      }
    }
    
    # -------------------------
    # 9. sigma_1^2 업데이트 (IG)
    # -------------------------
    post_a_sigma1 <- priors$a_sigma1 + (N * K) / 2
    post_b_sigma1 <- priors$b_sigma1 + 0.5 * sum(Z^2)
    sigma_1_sq_curr <- rinvgamma(1, post_a_sigma1, post_b_sigma1)
    
    # -------------------------
    # 10. I_ij 업데이트 (Bernoulli, edge-wise)
    # -------------------------
    dist_mat <- L2_dist(Z)
    eta_mat <- calculate_eta_matrix(theta_curr, beta_curr, dist_mat)
    eta_vec <- eta_mat[upper_tri_indices]
    
    log_p1_vec <- dnorm(Y_vec, eta_vec,            sqrt(sigma_0_sq_curr),      log = TRUE) + log(P_curr)
    log_p0_vec <- dnorm(Y_vec, eta_shrink_curr,    sqrt(sigma_shrink_sq_curr), log = TRUE) + log(1 - P_curr)
    
    # log-sum-exp trick
    max_log_p <- pmax(log_p1_vec, log_p0_vec)
    prob_1 <- exp(log_p1_vec - max_log_p)
    prob_0 <- exp(log_p0_vec - max_log_p)
    
    p_ij_1 <- prob_1 / (prob_1 + prob_0)
    
    I_vec <- rbinom(n_pairs, 1, p_ij_1)
    
    # -------------------------
    # 샘플 저장
    # -------------------------
    if (iter > burn_in && (iter - burn_in) %% thin == 0) {
      samples$p[save_idx]              <- P_curr
      samples$eta_shrink[save_idx]     <- eta_shrink_curr
      samples$sigma_shrink_sq[save_idx]<- sigma_shrink_sq_curr
      samples$sigma_0_sq[save_idx]     <- sigma_0_sq_curr
      # samples$alpha[save_idx]          <- alpha_curr
      samples$theta[, save_idx]        <- theta_curr
      samples$beta[save_idx]           <- beta_curr
      samples$Z[, , save_idx]          <- Z
      samples$sigma_1_sq[save_idx]     <- sigma_1_sq_curr
      
      # I_ij의 사후 평균 (P(i_ij=1 | Y))
      I_mat_p <- matrix(0, N, N)
      I_mat_p[upper_tri_indices] <- p_ij_1
      I_mat_p <- I_mat_p + t(I_mat_p)
      samples$I_mat[, , save_idx] <- I_mat_p
      
      # z 의 posterior density 구하기 - procrustes matching 을 위함
      samples$ll_z[save_idx] <- calc_log_post_z(Z, Y_vec, upper_tri_indices, I_idx_1,
                                                theta_curr, beta_curr,
                                                sigma_0_sq_curr, sigma_1_sq_curr) 
      
      save_idx <- save_idx + 1
    }
  }
  
  # -------------------------
  # 수락률 및 평균 계산
  # -------------------------
  cat("MCMC 완료.\n")
  cat(sprintf("Beta 수락률: %.2f%%\n", (accept_beta / n_iter) * 100))
  cat(sprintf("Z 수락률 (평균): %.2f%%\n", mean(accept_z / n_iter) * 100))
  
  # -------------------------
  # procrustes matching 후 z 재정렬
  # -------------------------
  map_idx <- which.max(samples$ll_z)
  Z_ref <- samples$Z[,,map_idx]
  if (requireNamespace("vegan", quietly = TRUE)) {
    for (s in seq_len(n_save)) {
      if (s == map_idx) next
      Z_s <- samples$Z[,,s]
      fit <- procrustes(Z_ref, Z_s, scale = FALSE)
      samples$Z[, ,s] <- predict(fit)
    }
  }
  
  # -------------------------
  # output warpper
  # -------------------------
  res <- list(
    samples = samples,
    mcmc = list(
      accept_beta = accept_beta / n_iter,
      accept_z    = accept_z    / n_iter
    ),
    priors = priors
  )
  
  return(res)
}


# ------------------------------------------------------------------
# 메인 실행 예제
# ------------------------------------------------------------------
# data sampling
# karate
g <- make_graph("Zachary")
R_eff <- communicability(g)

# sbm 으로 graph sampling
source("Utils.R")
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.30, 0.10, 0.00,
                     0.10, 0.60, 0.10,
                     0.00, 0.10, 0.90), nrow = 3, ncol = 3)
res<-sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)


cluster <- c(10, 10)
prob_mat <- matrix(c(0.90, 0.10,
                     0.10, 0.90), nrow = 2, ncol = 2)
res<-sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)
R_eff <- exp(res$proximity)
g <- res$g
res <- run_mixture_lsm_mcmc(R_eff, 
                           K = 2,           # 잠재 공간 차원
                           n_iter = 10000, 
                           burn_in = 1000, 
                           thin = 1,
                           # 하이퍼파라미터
                           priors = list(
                             a_p = 1, b_p = 1,
                             mu_shrink = 0, var_shrink = 1,
                             a_shrink = 0.05, b_shrink = 0.05,
                             a_0 = 0.05, b_0 = 0.05,
                             # mu_alpha = 0, var_alpha = 1,
                             var_theta = 1,
                             a_beta = 1, b_beta = 1,     # Gamma(a_beta, b_beta)
                             a_sigma1 = 0.05, b_sigma1 = 0.05,
                             prop_sd_beta = 0.1,
                             prop_sd_z = 0.03
                           ))

plot(g)
log(R_eff[17,])
log(R_eff[1,])
log(R_eff[33,])
log(R_eff[34,])
# additive terms
par(mfrow = c(2,2))
for(i in 1:dim(res$samples$theta)[1]){
  ts.plot(res$samples$theta[i,], main = paste0("theta_", i))
}

# traceplot of summation of additive terms
par(mfrow = c(4,4))
for(i in 1:dim(res$samples$theta)[1]){
  for(j in 1:dim(res$samples$theta)[1])
    if(i == j){
      next
    }else{
      ts.plot(res$samples$theta[i,] + res$samples$theta[j,], main = paste0("sum_theta_", i, "_", j))
    }
}

plot(g)
# plotting
par(mfrow = c(1,2))
node_size<-rowMeans(res$samples$theta)
# 위의 node_size 를 이용해서 plot(g) 에서 g 의 각 node 의 크기를 조정하는 코드 구현
# posterior additive term 을 적용한 것. (communicability 로 fitting 한 모델에서 얻음)
plot(g, vertex.size = 5 + 10 * (node_size - min(node_size)) / (max(node_size) - min(node_size)), main = 'additive term')

# g 의 각 node 별 degree 를 이용해서 g 의 각 node 의 크기를 조정하는 plotting 코드 구현
deg <- degree(g)
plot(g, vertex.size = 5 + 10 * (deg - min(deg)) / (max(deg) - min(deg)), main = 'degree effect')

# 위의 두 plot 결과 보면, additive term 이 degree effect 와 유사하지만 hub 를 이어 주는 node 들: 2,3,9 에
# additive effect 가 더 크게 보정을 해 주고 있는 것을 알 수 있다.

# latent positions
par(mfrow = c(2,2))
for(i in 1:dim(res$samples$Z)[1]){
  for(j in 1:dim(res$samples$Z)[2])
    ts.plot(res$samples$Z[i,j,], main = paste0("Z_", i, "_", j))
}

latent_map <- colMeans(aperm(res$samples$Z, c(3,1,2), dims = 1))
plot(latent_map)
# latent_map 에 label 추가
text(latent_map, labels = 1:dim(latent_map)[1], pos = 3)

plot(g, labels = 1:dim(latent_map)[1])

par(mfrow = c(1,1))
ts.plot(res$samples$sigma_0_sq)
ts.plot(res$samples$sigma_shrink_sq)
ts.plot(res$samples$eta_shrink)
ts.plot(res$samples$p)
ts.plot(res$samples$sigma_1_sq)
hist(log(R_eff), breaks = 30, main = "log(communicability)")
I<-aperm(res$samples$I_mat, c(3,1,2))

for(i in 1:dim(res$samples$I_mat)[2]){
  par(mfrow = c(5,5))
  for(j in 1:dim(res$samples$I_mat)[2]){
    ts.plot(I[,i,j], main = paste0(i, '_', j))
  }
}

par(mfrow = c(1,1))
plot(g)
hist(R_eff)
R_eff[26,25]
R_eff[17, 6]
R_eff[17, 7]
R_eff[14, 9]
R_eff[9 ,31]
R_eff[9 ,14]
R_eff[7 ,17]
R_eff[6 ,17]

source('utils.R')
result_assess(res, R_eff, model = 'LSM_conti_mixture_add_only', comparison_plot = TRUE)
hist(log(R_eff))
