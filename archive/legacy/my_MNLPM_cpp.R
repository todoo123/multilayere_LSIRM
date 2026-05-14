library(Rcpp)
library(RcppArmadillo)
library(vegan)
# 1. C++ 코드 컴파일 및 로드
Rcpp::sourceCpp("my_MNLPM.cpp")

# 2. R 래퍼 함수 정의
# 이 함수는 기존 R 코드와 동일한 인터페이스를 가지지만,
# 내부적으로 C++ 함수를 호출하고 R에서 Procrustes 정렬을 수행합니다.

MNLPM_2layer_Rcpp <- function(Y_bin, Y_con,
                              iter = 5000, burnin = 1000, thinning = 5,
                              d = 2,
                              # Priors
                              sigma2_z = 1.0,
                              sigma2_theta_bin = 1.0, sigma2_theta_con = 0.1,
                              a1 = 2.0, b1 = 1.0,
                              a0_bin = 0.5, b0_bin = 0.5,
                              a0_con = 0.5, b0_con = 0.5,
                              xi1_prior = 0, psi1_prior = 1.0,
                              xi2_prior = 0, psi2_prior = 1.0,
                              m_kappa = 0, s_kappa = 1,
                              # proposals
                              prop_z = 0.1,
                              prop_u_bin = 0.1, prop_u_con = 0.1,
                              prop_theta_bin = 0.1, prop_theta_con = 0.1,
                              prop_alpha_bin = 0.05, prop_beta_bin = 0.05,
                              prop_alpha_con = 0.05, prop_beta_con = 0.05, 
                              prop_kappa = 0.2,
                              # inits
                              z_init = NULL,
                              alpha1_init = 0, beta1_init = 1,
                              alpha2_init = 0, beta2_init = 1, kappa2_init = 1,
                              verbose = TRUE,
                              seed = 1) {
  
  # R 레벨에서 시드 설정
  set.seed(seed)
  n <- nrow(Y_bin)
  
  # -----------------------------------------------------------
  # (1) Rcpp 함수 호출 (핵심 MCMC 샘플링)
  # -----------------------------------------------------------
  # (verbose=TRUE로 설정하면 C++에서 직접 콘솔에 출력합니다)
  
  # 원본 R 코드의 하이퍼파라미터 이름을 C++ 함수 인자에 맞게 전달
  # (주의: R 코드는 beta ~ Gamma(a, b) [rate=b] 였으나,
  #  C++ 코드는 R::dgamma(shape, rate)를 사용하므로 b0가 rate가 됩니다.)
  cpp_results <- MNLPM_2layer_cpp(
    Y_bin = Y_bin, Y_con = Y_con,
    iter = iter, burnin = burnin, thinning = thinning, d = d,
    sigma2_z = sigma2_z,
    sigma2_theta_bin = sigma2_theta_bin, sigma2_theta_con = sigma2_theta_con,
    a1 = a1, b1 = b1,
    a0_bin = a0_bin, b0_bin = b0_bin, # R 코드와 일관되게 b0가 rate
    a0_con = a0_con, b0_con = b0_con, # R 코드와 일관되게 b0가 rate
    xi1_prior = xi1_prior, psi1_prior = psi1_prior,
    xi2_prior = xi2_prior, psi2_prior = psi2_prior,
    m_kappa = m_kappa, s_kappa = s_kappa,
    prop_z = prop_z,
    prop_u_bin = prop_u_bin, prop_u_con = prop_u_con,
    prop_theta_bin = prop_theta_bin, prop_theta_con = prop_theta_con,
    prop_alpha_bin = prop_alpha_bin, prop_beta_bin = prop_beta_bin,
    prop_alpha_con = prop_alpha_con, prop_beta_con = prop_beta_con, 
    prop_kappa = prop_kappa,
    z_init = z_init,
    alpha1_init = alpha1_init, beta1_init = beta1_init,
    alpha2_init = alpha2_init, beta2_init = beta2_init, kappa2_init = kappa2_init,
    verbose = verbose,
    seed = seed
  )
  
  # C++에서 반환된 샘플 추출
  samples <- cpp_results$samples
  z_save <- aperm(samples$z, c(3, 1, 2))
  u1_save <- aperm(samples$u1, c(3, 1, 2))
  u2_save <- aperm(samples$u2, c(3, 1, 2))
  ns <- dim(z_save)[3] # (ns, n, d)
  

  # -----------------------------------------------------------
  # (2) Procrustes 정렬 (R 레벨에서 수행)
  # -----------------------------------------------------------

  if (requireNamespace("vegan", quietly = TRUE)) {

    # MAP (Maximum A Posteriori) 인덱스 찾기
    # C++에서 이미 계산된 로그 우도 사용 - 근데 ll_con 의 절대크기가 너무 커서;;
    # 공평한 기준인지 모르겠음.
    map_idx <- which.max(samples$ll_bin + samples$ll_con)
    Z_map <- matrix(z_save[map_idx, , ], n, d)

    cat(sprintf("Performing Procrustes alignment (using MAP at index %d)...\n", map_idx))

    for (ss in seq_len(ns)) {
      if (ss == map_idx) next

      # C++의 arma::cube는 R에서 (n, d, ns)가 아닌 (ns, n, d)로 반환됩니다.
      Z_s <- matrix(z_save[ss, , ], n, d)
      U1_s <- matrix(u1_save[ss, , ], n, d)
      U2_s <- matrix(u2_save[ss, , ], n, d)

      # z를 기준으로 정렬
      fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)

      # z, u1, u2에 동일한 회전/변환 적용
      z_save[ss, , ] <- predict(fit)
      u1_save[ss, , ] <- predict(fit, newdata = U1_s)
      u2_save[ss, , ] <- predict(fit, newdata = U2_s)
    }

    # 정렬된 샘플로 교체
    cpp_results$samples$z <- z_save
    cpp_results$samples$u1 <- u1_save
    cpp_results$samples$u2 <- u2_save

  } else {
    warning("Package 'vegan' not found. Skipping Procrustes alignment.")
  }

  # # -----------------------------------------------------------
  # # (3) 원본 R 함수와 동일한 형식으로 최종 결과 반환
  # # -----------------------------------------------------------
  # 
  # # (config 리스트는 C++에서 만들지 않았으므로 R에서 추가)
  # final_list <- list(
  #   samples = cpp_results$samples,
  #   accept = cpp_results$accept,
  #   config = list(
  #     iter = iter, burnin = burnin, thinning = thinning,
  #     d = d, 
  #     prop_sd = list(z = prop_z,
  #                    u_bin = prop_u_bin, theta_bin = prop_theta_bin,
  #                    alpha_bin = prop_alpha_bin, beta_bin = prop_beta_bin,
  #                    u_con = prop_u_con, theta_con = prop_theta_con,
  #                    alpha_con = prop_alpha_con, beta_con = prop_beta_con, 
  #                    kappa = prop_kappa),
  #     priors = list(
  #       sigma2_z = sigma2_z,
  #       sigma2_theta = sigma2_theta,
  #       sigma2_l = list(a1 = a1, b1 = b1),
  #       binary = list(a0 = a0_bin, b0 = b0_bin, xi = xi1_prior, psi2 = psi1_prior),
  #       conti  = list(a0 = a0_con, b0 = b0_con, m_kappa = m_kappa, s_kappa = s_kappa,
  #                     xi = xi2_prior, psi2 = psi2_prior)
  #     )
  #   )
  # )
  # 
  # return(final_list)
  return(cpp_results)
}


# -----------------------------------------------------------------------------
# (사용 예시)
# -----------------------------------------------------------------------------

# # (1) 원본 R 함수 로드 (비교용)
# source("my_MNLPM.R") 
# 
# # (2) 가상 데이터 생성
# N <- 20 # 노드 수
# d <- 2  # 잠재 차원
# set.seed(123)
# 
# Z <- matrix(rnorm(N * d), N, d)
# D_Z <- L2_dist(Z)
# 
# # Binary (Layer 1)
# alpha1 <- 0.5
# beta1 <- 1.0
# theta1 <- rnorm(N, 0, 0.5)
# eta1 <- alpha1 + outer(theta1, theta1, "+") - beta1 * D_Z
# pi1 <- plogis(eta1)
# Y_bin <- matrix(0, N, N)
# Y_bin[upper.tri(Y_bin)] <- rbinom(N*(N-1)/2, 1, pi1[upper.tri(pi1)])
# Y_bin <- Y_bin + t(Y_bin)
# 
# # Continuous (Layer 2)
# alpha2 <- 1.0
# beta2 <- 0.5
# kappa2 <- 0.8
# theta2 <- rnorm(N, 0, 0.5)
# eta2 <- alpha2 + outer(theta2, theta2, "+") - beta2 * D_Z
# mu_log <- eta2 - kappa2/2
# Y_con <- matrix(0, N, N)
# Y_con[upper.tri(Y_con)] <- rlnorm(N*(N-1)/2, meanlog = mu_log[upper.tri(mu_log)], sdlog = sqrt(kappa2))
# Y_con <- Y_con + t(Y_con)
# 
# # (3) Rcpp 버전 실행 (매우 빠름)
# cat("--- Rcpp 버전 실행 ---\n")
# system.time({
#   fit_rcpp <- MNLPM_2layer_Rcpp(Y_bin, Y_con, 
#                                iter = 5000, burnin = 1000, 
#                                d = d, verbose = TRUE, seed = 123)
# })
# 
# # (4) R 버전 실행 (매우 느림)
# cat("\n--- R 버전 실행 (비교용) ---\n")
# system.time({
#   fit_R <- MNLPM_2layer(Y_bin, Y_con, 
#                         iter = 5000, burnin = 1000, 
#                         d = d, verbose = TRUE, seed = 123)
# })
# 
# # (5) 결과 비교 (Procrustes 정렬 후)
# z_est_rcpp <- apply(fit_rcpp$samples$z, c(2,3), mean)
# z_est_R <- apply(fit_R$samples$z, c(2,3), mean)
# 
# # Rcpp 결과와 R 결과는 Procrustes 정렬로 인해 거의 동일해야 합니다.
# plot(z_est_rcpp, main = "Rcpp (Procrustes)")
# plot(z_est_R, main = "R (Procrustes)")
# 
# # 수용률 비교
# summary(fit_rcpp$accept$z)
# summary(fit_R$accept$z)