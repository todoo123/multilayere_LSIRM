library(vegan) # Procrustes 분석을 위해
library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("my_LSJM.cpp")

# Rcpp 함수를 호출하고 Procrustes 분석을 수행하는 R 래퍼 함수
LSJM_2layer <- function(Y_bin, Y_con,
                        iter = 5000, burnin = 1000, thinning = 5,
                        d = 2,
                        # shared z prior variance
                        sigma2 = 1.0,
                        # Binary priors/hypers
                        a0_bin = 0.5, b0_bin = 0.5,
                        m0_bin = 0, kappa0_bin = 1/2, alpha0_bin = 3, beta0_bin = 1/2,
                        # Continuous priors/hypers
                        a0_con = 0.5, b0_con = 0.5,
                        m_kappa = 0, s_kappa = 1,
                        m0_con = 0, kappa0_con = 1/2, alpha0_con = 3, beta0_con = 1/2,
                        # proposals
                        prop_z = 0.1,
                        prop_alpha_bin = 0.05, prop_beta_bin = 0.05,
                        prop_alpha_con = 0.05, prop_beta_con = 0.05, prop_kappa = 0.2,
                        # inits
                        z_init = NULL,
                        alpha1_init = 0, beta1_init = 1, xi1_init = 0, psi1_init = 1,
                        alpha2_init = 0, beta2_init = 1, xi2_init = 0, psi2_init = 1, kappa2_init = 1,
                        verbose = TRUE,
                        seed = 1) {
  
  n <- nrow(Y_bin)
  
  # 1. Rcpp 함수 호출하여 MCMC 샘플링 수행
  mcmc_results <- LSJM_2layer_cpp(
    Y_bin = Y_bin, Y_con = Y_con,
    iter = iter, burnin = burnin, thinning = thinning, d = d,
    sigma2 = sigma2,
    a0_bin = a0_bin, b0_bin = b0_bin, m0_bin = m0_bin, kappa0_bin = kappa0_bin, 
    alpha0_bin = alpha0_bin, beta0_bin = beta0_bin,
    a0_con = a0_con, b0_con = b0_con, m_kappa = m_kappa, s_kappa = s_kappa,
    m0_con = m0_con, kappa0_con = kappa0_con, alpha0_con = alpha0_con, beta0_con = beta0_con,
    prop_z = prop_z,
    prop_alpha_bin = prop_alpha_bin, prop_beta_bin = prop_beta_bin,
    prop_alpha_con = prop_alpha_con, prop_beta_con = prop_beta_con, prop_kappa = prop_kappa,
    z_init = z_init,
    alpha1_init = alpha1_init, beta1_init = beta1_init, xi1_init = xi1_init, psi1_init = psi1_init,
    alpha2_init = alpha2_init, beta2_init = beta2_init, xi2_init = xi2_init, 
    psi2_init = psi2_init, kappa2_init = kappa2_init,
    verbose = verbose,
    seed = seed
  )
  
  # 2. Procrustes 분석 (원본 R 코드와 동일한 로직)
  z_save <- mcmc_results$samples$z
  ll_joint_save <- mcmc_results$samples$ll_joint
  
  # Rcpp (n, d, ns) -> R (ns, n, d)로 차원 변경
  z_save_permuted <- aperm(z_save, c(3, 1, 2)) 
  
  ns <- dim(z_save_permuted)[1] # 저장된 샘플 수
  
  if (ns > 0 && sum(is.na(ll_joint_save)) == 0) {
    map_idx <- which.max(ll_joint_save)
    if (length(map_idx) == 0) map_idx <- 1
    
    Z_map <- z_save_permuted[map_idx, , , drop = FALSE]
    Z_map <- matrix(Z_map[1, , ], nrow = n, ncol = d)
    
    for (s in seq_len(ns)) {
      if (s == map_idx) next
      Z_s <- matrix(z_save_permuted[s, , ], nrow = n, ncol = d)
      if (!any(is.na(Z_s))) {
        tryCatch({
          fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
          z_save_permuted[s, , ] <- predict(fit)
        }, error = function(e) { 
          warning(paste("Procrustes failed for sample", s))
        })
      }
    }
  }
  
  mcmc_results$samples$z <- z_save_permuted
  
  # R 원본 코드와 동일한 config 구조 반환 (C++에서 생략했던 부분)
  mcmc_results$config = list(
    iter = iter, burnin = burnin, thinning = thinning,
    d = d, sigma2 = sigma2,
    prop_sd = list(z = prop_z,
                   alpha_bin = prop_alpha_bin, beta_bin = prop_beta_bin,
                   alpha_con = prop_alpha_con, beta_con = prop_beta_con, kappa = prop_kappa),
    priors = list(
      binary = list(a0 = a0_bin, b0 = b0_bin, m0 = m0_bin, kappa0 = kappa0_bin,
                    alpha0 = alpha0_bin, beta0 = beta0_bin),
      conti  = list(a0 = a0_con, b0 = b0_con, m_kappa = m_kappa, s_kappa = s_kappa,
                    m0 = m0_con, kappa0 = kappa0_con, alpha0 = alpha0_con, beta0 = beta0_con)
    )
  )
  
  return(mcmc_results)
}