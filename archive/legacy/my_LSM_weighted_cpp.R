library(vegan) # Procrustes 분석을 위해
library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("my_LSM_weighted.cpp")


# Rcpp 함수를 호출하고 Procrustes 분석을 수행하는 R 래퍼 함수
LSM_conti <- function(Y, iter = 5000, burnin = 1000, thinning = 5, d = 2,
                      sigma2 = 1, a0 = 0.5, b0 = 0.5, m_kappa = 0, s_kappa = 1,
                      prop_z = 0.05, prop_alpha = 0.05, prop_beta = 0.05, prop_kappa = 0.2,
                      alpha_init = 0, beta_init = 1, xi_init = 0, kappa_init = 1,
                      psi2_init = 1, z_init = NULL, verbose = T){
  
  # 1. Rcpp 함수 호출하여 MCMC 샘플링 수행
  mcmc_results <- LSM_conti_cpp(
    Y = Y, iter = iter, burnin = burnin, thinning = thinning, d = d,
    sigma2 = sigma2, a0 = a0, b0 = b0, m_kappa = m_kappa, s_kappa = s_kappa,
    prop_z = prop_z, prop_alpha = prop_alpha, prop_beta = prop_beta, prop_kappa = prop_kappa,
    alpha_init = alpha_init, beta_init = beta_init, xi_init = xi_init, 
    kappa_init = kappa_init, psi2_init = psi2_init, 
    z_init = z_init, verbose = verbose
  )

  # 2. Procrustes 분석 (원본 R 코드와 동일한 로직)
  z_save <- mcmc_results$samples$z
  ll_save <- mcmc_results$samples$ll

  # Rcpp 결과물 (n, d, ns) -> R (ns, n, d)로 차원 변경
  z_save_permuted <- aperm(z_save, c(3, 1, 2))

  n <- nrow(Y)
  ns <- dim(z_save_permuted)[1] # 저장된 샘플 수

  if (ns > 0 && sum(is.na(ll_save)) == 0) { # ll_save에 NA가 없는지 확인
    # 1) MAP 인덱스 선택
    map_idx <- which.max(ll_save)

    if (length(map_idx) == 0) map_idx <- 1 # 혹시 모를 방어 코드

    Z_map <- z_save_permuted[map_idx, , , drop = FALSE]
    Z_map <- matrix(Z_map[1, , ], nrow = n, ncol = d)

    # 2) 모든 저장 샘플을 Z_map에 정렬
    for (s in seq_len(ns)) {
      Z_s <- matrix(z_save_permuted[s, , ], nrow = n, ncol = d)
      if (s != map_idx && !any(is.na(Z_s))) { # NA가 없는 샘플만 정렬
          fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
          Z_aligned <- predict(fit)
          z_save_permuted[s, , ] <- Z_aligned
          }
    }
  }
  # Procrustes 정렬된 z 샘플로 교체
  mcmc_results$samples$z <- z_save_permuted

  return(mcmc_results)
}