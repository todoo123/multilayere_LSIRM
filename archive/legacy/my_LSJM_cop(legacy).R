rm(list = ls())

## -------------------------- (0) Utility ------------------------
L2_dist <- function(z) {
  n <- nrow(z)
  D <- matrix(0.0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      # distance metric
      d <- sqrt(sum((z[i,] - z[j,])^2))
      D[i,j] <- d
      D[j,i] <- d
    }
  } 
  return(D)
}

dot_dist <- function(z) {
  n <- nrow(z)
  # 각 행(노드)의 norm 계산
  norms <- sqrt(rowSums(z^2))
  
  # dot product 행렬
  G <- z %*% t(z)
  
  # 코사인 유사도 (dot product / norm 표준화)
  sim <- G / (outer(norms, norms))
  
  # distance 로 변환 (1 - similarity)
  D <- 1 - sim
  
  return(D)
}

## ---------- (1) loglike, prior, update rule ----------
################################################################
## continuous network
################################################################

# loglik_conti_network <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
#   D <- L2_dist(z)
#   eta <- alpha - beta*D
#   upper <- which(upper.tri(Y), arr.ind = TRUE)
#   d <- D[upper]
#   y <- pmax(Y[upper], eps)
#   rate <- kappa * exp(-eta)
#   ll <- sum(stats::dgamma(y, shape = kappa, rate = rate, log = T))
#   return(ll)
# }


loglik_conti_network <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  D <- L2_dist(z)
  # D <- dot_dist(z)
  # communicability distance: eta <- alpha + beta*D 를 사용해야 함. communicability 와 그 의미가 반대이므로.
  eta <- alpha - beta*D
  upper <- which(upper.tri(Y), arr.ind = TRUE)
  y <- pmax(Y[upper], eps)
  mu_log <- eta[upper] - kappa/2        # 평균 exp(eta)로 맞춰주기 위함
  ll <- sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
  return(ll)
}

# z_i ~ normal(0, sigma2)
log_prior_z_i <- function(z_i, sigma2) {
  return(sum(stats::dnorm(z_i, mean = 0, sd = sqrt(sigma2), log = T)))
}

# alpha ~ normal(xi, psi2)
log_prior_alpha <- function(alpha, xi, psi2) {
  return(stats::dnorm(alpha, mean = xi, sd = sqrt(psi2), log = T))
}

# beta ~ Gamma(a0, b0)
log_prior_beta <- function(beta, a, b) {
  return(stats::dgamma(beta, shape = a, rate = b, log = T))
}

# log(kappa) ~ normal(a, b)
log_prior_kappa <- function(kappa, a, b) {
  return(stats::dnorm(log(kappa), a, sqrt(b), log = T)-log(kappa))
}

################################################################
## binary network
################################################################
loglik_binary_network <- function(Y, z, alpha, beta) {
  D <- L2_dist(z)
  # D <- dot_dist(z)
  eta <- alpha - beta*D
  pi <- stats::plogis(eta)
  n <- nrow(Y)
  ll <- 0.0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      y <- Y[i,j]
      p <- pi[i,j]
      if(y == 1){
        ll <- ll + log(p + 1e-11)
      }else{
        ll <- ll + log(1-p + 1e-11)
      }
    }
  }
  return(ll)
}
# prior distribution function 공유
# z_i ~ normal(0, sigma2)
# alpha ~ normal(xi, psi2)
# beta ~ Gamma(a0, b0)


## ---------- (2) 레이어별 파라미터 업데이트 래퍼 ----------
# 네 기존 함수들과 이름이 겹치지 않도록 래퍼로 감쌈

# Binary layer updates (MH + Gibbs for xi1, psi1)
# MH
bin_update_alpha <- function(alpha, z, Y, xi, psi2, prop_alpha, beta) {
  ll_curr <- loglik_binary_network(Y, z, alpha, beta) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_binary_network(Y, z, alpha_prop, beta) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

bin_update_beta <- function(beta, z, Y, alpha, prop_beta, a, b){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_binary_network(Y, z, alpha, beta) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_binary_network(Y, z, alpha, beta_prop) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

# test:
# phi: log(beta) 상태변수로 업데이트 (대칭 제안 → Hastings 보정 불필요)
update_phi_beta <- function(phi, z, Y, alpha, loglik_fn, prop_phi,
                            mu_phi = 0, s2_phi = 0.5^2,  # prior: N(mu_phi, s2_phi)
                            extra_args = list()) {
  beta      <- exp(phi)
  curr <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta), extra_args)) +
    dnorm(phi, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  
  phi_prop  <- phi + rnorm(1, 0, prop_phi)    # 대칭 제안
  beta_prop <- exp(phi_prop)
  prop <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta_prop), extra_args)) +
    dnorm(phi_prop, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  
  if (log(runif(1)) < (prop - curr)) {
    list(phi = phi_prop, accepted = TRUE)
  } else {
    list(phi = phi, accepted = FALSE)
  }
}



# gibbs
bin_update_xi_psi2 <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
  # xi ~ normal(m0, kappa0)
  # psi2 ~ IG(alpha0, beta0)
  n <- length(alpha)
  abar <- mean(alpha)
  
  # 1) xi | psi, alpha
  kappa_n <- (1/psi2 + 1/kappa0)^(-1)
  m_n <- kappa_n * (m0/kappa0 + alpha/psi2)
  xi_new <- rnorm(1, mean = m_n, sd = sqrt(kappa_n))
  
  # 2) psi | xi, alpha      # Inv-Gamma(shape, scale)
  shape <- alpha0 + n/2
  scale <- beta0 + 0.5 * sum((alpha - xi_new)^2)
  psi2_new <- 1 / rgamma(1, shape = shape, rate = scale)
  
  list(xi = xi_new, psi2 = psi2_new)
}

# Continuous layer updates (MH + Gibbs for xi2, psi2)
# MH
con_update_alpha <- function(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa) {
  ll_curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_conti_network(Y, z, alpha_prop, beta, kappa) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

con_update_beta <- function(beta, z, Y, alpha, prop_beta, a, b, kappa){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_conti_network(Y, z, alpha, beta_prop, kappa) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

con_update_kappa <- function(kappa, z, Y, alpha, beta, prop_kappa, m_kappa = 0, s_kappa = 1){
  # log(kappa) ~ normal(m_kappa, s_kappa)
  curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_kappa(kappa, m_kappa, s_kappa)
  kappa_prop <- exp(log(kappa) + rnorm(1, 0, prop_kappa))
  
  prop <- loglik_conti_network(Y, z, alpha, beta, kappa_prop) + log_prior_kappa(kappa_prop, m_kappa, s_kappa)
  # log-정규 제안 보정 항 추가:
  acc <- (prop - curr) + (log(kappa_prop) - log(kappa))
  if (log(runif(1)) < acc) {
    return(list(kappa=kappa_prop, accepted=TRUE))
  } else {
    return(list(kappa=kappa, accepted=FALSE))
  }
}

# Gibbs
con_update_xi_psi2 <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
  # xi ~ normal(m0, kappa0)
  # psi2 ~ IG(alpha0, beta0)
  n <- length(alpha)
  abar <- mean(alpha)
  
  # 1) xi | psi, alpha
  kappa_n <- (1/psi2 + 1/kappa0)^(-1)
  m_n <- kappa_n * (m0/kappa0 + alpha/psi2)
  xi_new <- rnorm(1, mean = m_n, sd = sqrt(kappa_n))
  
  # 2) psi | xi, alpha      # Inv-Gamma(shape, scale)
  shape <- alpha0 + n/2
  scale <- beta0 + 0.5 * sum((alpha - xi_new)^2)
  psi2_new <- 1 / rgamma(1, shape = shape, rate = scale)
  
  list(xi = xi_new, psi2 = psi2_new)
}


## ---------- (3) 공통: z 업데이트 (두 레이어 합동) ----------
# 주의: 기존 binary 코드에 있던 z 제안분산 버그를 고쳤음.
# - proposal sd는 prop_z를 사용 (기존 코드 일부가 sqrt(sigma2)로 제안하던 문제 수정)
lsjm_update_z_single <- function(j, z, Y_bin, Y_con,
                                 alpha1, beta1,
                                 alpha2, beta2, kappa2,
                                 sigma2, prop_z, d) {
  zi_old <- z[j, ]
  z_prop <- z
  zi_prop <- zi_old + rnorm(d, 0, prop_z)  # <-- 제안분산은 prop_z
  z_prop[j, ] <- zi_prop
  
  # 두 레이어 로그우도 합 + 공통 prior(z_j | 0, sigma2 I)
  ll_old <- loglik_binary_network(Y_bin, z, alpha1, beta1) +
    loglik_conti_network(Y_con, z, alpha2, beta2, kappa2) +
    log_prior_z_i(zi_old, sigma2)
  ll_new <- loglik_binary_network(Y_bin, z_prop, alpha1, beta1) +
    loglik_conti_network(Y_con, z_prop, alpha2, beta2, kappa2) +
    log_prior_z_i(zi_prop, sigma2)
  
  if (log(runif(1)) < (ll_new - ll_old)) {
    list(z = z_prop, accepted = TRUE)
  } else {
    list(z = z, accepted = FALSE)
  }
}
## ====================== (A) Copula utilities ======================
clamp <- function(x, lo=1e-12, hi=1-1e-12) pmin(pmax(x, lo), hi)
safe_qnorm <- function(u) qnorm(clamp(u))
safe_pnorm <- function(x) pnorm(x)

# rho(d) = tanh(k0 + k1 * d), 안정화를 위해 clip
rho_from_kappa <- function(D, k0, k1, clip=0.98) {
  rho <- tanh(k0 + k1 * D)
  pmin(pmax(rho, -clip), clip)
}

# (연속층: Lognormal 마진) PIT → Z2
# 주의: 기존 loglik_conti_network와 동일한 파라매터화:
#   mu_log = eta - kappa/2, sdlog = sqrt(kappa)
pit_Z2_lognormal <- function(Y_con, z, alpha2, beta2, kappa2, eps=1e-12) {
  D <- L2_dist(z)
  # D <- dot_dist(z)
  eta2 <- alpha2 - beta2 * D
  mu_log <- eta2 - kappa2/2
  upper <- which(upper.tri(Y_con), arr.ind = TRUE)
  y <- pmax(Y_con[upper], eps)
  u2 <- plnorm(y, meanlog = mu_log[upper], sdlog = sqrt(kappa2))
  z2 <- safe_qnorm(u2)
  list(z2 = z2, upper = upper, D = D, eta2 = eta2)
}

# (이진층: logit 마진) 컷포인트 q = Phi^{-1}(1 - p)
cutpoint_q_from_eta1 <- function(eta1) {
  p <- plogis(eta1)
  q <- qnorm(1 - p)  # == -qnorm(p)
  list(p = p, q = q)
}

## ============= (B) Copula-결합 베르누이 조건부 로그우도 =============
# log P(Y_bin | Y_con, params) with Gaussian copula
loglik_binary_cond_copula <- function(Y_bin, Z2, Q, RHO, upper_idx) {
  # Y_bin: n x n, 0/1
  y <- Y_bin[upper_idx]
  q <- Q[upper_idx]
  rho <- RHO[upper_idx]
  z2 <- Z2
  denom_sd <- sqrt(1 - rho^2)
  # cond prob for Y=1
  prob1 <- 1 - safe_pnorm( (q - rho * z2) / denom_sd )
  prob1 <- clamp(prob1)
  ll <- ifelse(y == 1, log(prob1), log(1 - prob1))
  sum(ll)
}

## ============= (C) Copula-결합 "공동" 로그우도 (연속 + 조건부 베르누이) =============
loglik_joint_copula <- function(Y_bin, Y_con,
                                z,
                                alpha1, beta1,
                                alpha2, beta2, kappa2,
                                k0, k1,
                                eps = 1e-12) {
  # --- 연속층: 기존 lognormal 로그우도 그대로 ---
  ll_con <- loglik_conti_network(Y_con, z, alpha2, beta2, kappa2, eps)
  
  # --- 연속층으로부터 Z2 (결정적) ---
  pit <- pit_Z2_lognormal(Y_con, z, alpha2, beta2, kappa2, eps)
  z2 <- pit$z2; upper <- pit$upper; D <- pit$D
  
  # --- 이진층: eta1, q 계산 ---
  # D <- L2_dist(z)  # 원래 거리 쓸 거면 이 줄로 교체
  eta1 <- alpha1 - beta1 * D
  cq   <- cutpoint_q_from_eta1(eta1)
  qmat <- cq$q
  
  # --- rho(d) ---
  RHO <- rho_from_kappa(D, k0, k1, clip=0.98)
  
  # --- 베르누이 조건부 로그우도 ---
  ll_bin_cond <- loglik_binary_cond_copula(Y_bin, Z2 = z2, Q = qmat, RHO = RHO, upper_idx = upper)
  
  ll_con + ll_bin_cond
}


## ====================== (D) Copula kappa prior & MH ======================
# Normal prior for kappa = (k0, k1)
log_prior_kappa_vec <- function(k0, k1, mu = c(0,0), Sigma_inv = diag(2)) {
  kappa <- c(k0, k1)
  as.numeric(-0.5 * crossprod(kappa - mu, Sigma_inv %*% (kappa - mu)))
}

# RW-MH update for kappa = (k0, k1)
update_kappa_copula <- function(k0, k1,
                                Y_bin, Y_con, z,
                                alpha1, beta1,
                                alpha2, beta2, kappa2,
                                prop_Sigma,        # 2x2 proposal covariance
                                prior_mu = c(0,0), prior_Sigma_inv = diag(2),
                                eps = 1e-12) {
  curr_ll <- loglik_joint_copula(Y_bin, Y_con, z,
                                 alpha1, beta1, alpha2, beta2, kappa2,
                                 k0, k1, eps)
  curr_lp <- log_prior_kappa_vec(k0, k1, prior_mu, prior_Sigma_inv)
  
  kappa_prop <- as.numeric(MASS::mvrnorm(1, mu = c(k0, k1), Sigma = prop_Sigma))
  k0p <- kappa_prop[1]; k1p <- kappa_prop[2]
  
  prop_ll <- loglik_joint_copula(Y_bin, Y_con, z,
                                 alpha1, beta1, alpha2, beta2, kappa2,
                                 k0p, k1p, eps)
  prop_lp <- log_prior_kappa_vec(k0p, k1p, prior_mu, prior_Sigma_inv)
  
  acc <- (prop_ll + prop_lp) - (curr_ll + curr_lp)
  if (log(runif(1)) < acc) {
    list(k0 = k0p, k1 = k1p, accepted = TRUE, ll_new = prop_ll)
  } else {
    list(k0 = k0,  k1 = k1,  accepted = FALSE, ll_new = curr_ll)
  }
}

## ====================== (E) z 업데이트: copula 포함 버전 ======================
lsjm_update_z_single_copula <- function(j, z, Y_bin, Y_con,
                                        alpha1, beta1,
                                        alpha2, beta2, kappa2,
                                        k0, k1,      # copula params
                                        sigma2, prop_z, d) {
  zi_old <- z[j, ]
  z_prop <- z
  zi_prop <- zi_old + rnorm(d, 0, prop_z)
  z_prop[j, ] <- zi_prop
  
  # 공동 로그우도 + prior
  ll_old <- loglik_joint_copula(Y_bin, Y_con, z,
                                alpha1, beta1, alpha2, beta2, kappa2,
                                k0, k1) + log_prior_z_i(zi_old, sigma2)
  ll_new <- loglik_joint_copula(Y_bin, Y_con, z_prop,
                                alpha1, beta1, alpha2, beta2, kappa2,
                                k0, k1) + log_prior_z_i(zi_prop, sigma2)
  
  if (log(runif(1)) < (ll_new - ll_old)) {
    list(z = z_prop, accepted = TRUE)
  } else {
    list(z = z, accepted = FALSE)
  }
}


## ====================== (F) 메인 드라이버: Copula 확장 ======================
LSJM_2layer_copula <- function(Y_bin, Y_con,
                               iter = 5000, burnin = 1000, thinning = 5,
                               d = 2,
                               sigma2 = 1.0,
                               # Binary priors/hypers
                               a0_bin = 0.5, b0_bin = 0.5,
                               m0_bin = 0, kappa0_bin = 1/2, alpha0_bin = 3, beta0_bin = 1/2,
                               # Continuous priors/hypers (lognormal layer)
                               a0_con = 0.5, b0_con = 0.5,
                               m_kappa = 0, s_kappa = 1,
                               m0_con = 0, kappa0_con = 1/2, alpha0_con = 3, beta0_con = 1/2,
                               # Copula kappa prior (Normal)
                               kappa_mu = c(0,0), kappa_Sigma = diag(2),
                               # proposals
                               prop_z = 0.1,
                               prop_alpha_bin = 0.05, prop_beta_bin = 0.05,
                               prop_alpha_con = 0.05, prop_beta_con = 0.05, prop_kappa = 0.2,
                               # copula kappa proposal (2x2)
                               prop_kappa_Sigma = diag(c(0.05, 0.05)^2),
                               # inits
                               z_init = NULL,
                               alpha1_init = 0, beta1_init = 1, xi1_init = 0, psi1_init = 1,
                               alpha2_init = 0, beta2_init = 1, xi2_init = 0, psi2_init = 1, kappa2_init = 1,
                               k0_init = 0, k1_init = 0,   # copula 상관 초기값
                               verbose = TRUE,
                               seed = 1) {
  set.seed(seed)
  stopifnot(is.matrix(Y_bin), is.matrix(Y_con))
  n <- nrow(Y_bin); stopifnot(ncol(Y_bin) == n, nrow(Y_con) == n, ncol(Y_con) == n)
  if (max(abs(Y_bin - t(Y_bin))) > 1e-8) stop("Y_bin must be symmetric.")
  if (max(abs(Y_con - t(Y_con))) > 1e-8) stop("Y_con must be symmetric.")
  
  # init z
  if (is.null(z_init)) {
    z <- matrix(rnorm(n * d, 0, sqrt(sigma2)), n, d)
  } else {
    stopifnot(is.matrix(z_init), nrow(z_init) == n, ncol(z_init) == d)
    z <- z_init
  }
  
  # layer params
  alpha1 <- alpha1_init; beta1 <- beta1_init; xi1 <- xi1_init; psi1 <- psi1_init
  alpha2 <- alpha2_init; beta2 <- beta2_init; xi2 <- xi2_init; psi2 <- psi2_init; kappa2 <- kappa2_init
  # copula params
  k0 <- k0_init; k1 <- k1_init
  
  # lognormal(beta) = phi sampling 유지
  phi1 <- log(beta1_init); phi2 <- log(beta2_init)
  
  # storage
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning)
  ns <- length(keep_idx)
  
  z_save <- array(NA_real_, dim = c(ns, n, d))
  alpha1_save <- beta1_save <- xi1_save <- psi1_save <- numeric(ns)
  alpha2_save <- beta2_save <- xi2_save <- psi2_save <- kappa2_save <- numeric(ns)
  k0_save <- k1_save <- numeric(ns)
  
  ll_con_save <- ll_bincond_save <- ll_joint_save <- numeric(ns)
  
  acc_z <- integer(n)
  acc_alpha1 <- acc_beta1 <- acc_alpha2 <- acc_beta2 <- acc_kappa2 <- 0L
  acc_kappa_cop <- 0L
  
  s <- 0L
  kappa_Sigma_inv <- solve(kappa_Sigma)
  
  for (it in 1:iter) {
    # (1) z single-site MH (joint with copula)
    for (j in 1:n) {
      up <- lsjm_update_z_single_copula(j, z, Y_bin, Y_con,
                                        alpha1, beta1,
                                        alpha2, beta2, kappa2,
                                        k0, k1,
                                        sigma2, prop_z, d)
      z <- up$z
      if (up$accepted) acc_z[j] <- acc_z[j] + 1L
    }
    
    # (2) Binary layer params (원래와 동일)
    up_a1 <- bin_update_alpha(alpha1, z, Y_bin, xi1, psi1, prop_alpha_bin, beta1)
    alpha1 <- up_a1$alpha; if (up_a1$accepted) acc_alpha1 <- acc_alpha1 + 1L
    
    up_xi1psi1 <- bin_update_xi_psi2(alpha1, xi1, psi1, m0 = m0_bin, kappa0 = kappa0_bin,
                                     alpha0 = alpha0_bin, beta0 = beta0_bin)
    xi1 <- up_xi1psi1$xi; psi1 <- up_xi1psi1$psi2
    
    # (3) Continuous layer params (원래와 동일)
    up_a2 <- con_update_alpha(alpha2, z, Y_con, xi2, psi2, prop_alpha_con, beta2, kappa2)
    alpha2 <- up_a2$alpha; if (up_a2$accepted) acc_alpha2 <- acc_alpha2 + 1L
    
    up_k2 <- con_update_kappa(kappa2, z, Y_con, alpha2, beta2, prop_kappa, m_kappa, s_kappa)
    kappa2 <- up_k2$kappa; if (up_k2$accepted) acc_kappa2 <- acc_kappa2 + 1L
    
    up_xi2psi2 <- con_update_xi_psi2(alpha2, xi2, psi2, m0 = m0_con, kappa0 = kappa0_con,
                                     alpha0 = alpha0_con, beta0 = beta0_con)
    xi2 <- up_xi2psi2$xi; psi2 <- up_xi2psi2$psi2
    
    # lognormal(beta) = phi sampling 유지
    up_phi1 <- update_phi_beta(phi1, z, Y_bin, alpha1, loglik_binary_network,
                               prop_phi = prop_beta_bin, mu_phi = 5, s2_phi = 1^2)
    phi1 <- up_phi1$phi; beta1 <- exp(phi1); if (up_phi1$accepted) acc_beta1 <- acc_beta1 + 1L
    
    up_phi2 <- update_phi_beta(phi2, z, Y_con, alpha2, loglik_conti_network,
                               prop_phi = prop_beta_con, mu_phi = 0, s2_phi = 1^2,
                               extra_args = list(kappa = kappa2))
    phi2 <- up_phi2$phi; beta2 <- exp(phi2); if (up_phi2$accepted) acc_beta2 <- acc_beta2 + 1L
    
    # test
    beta1 <- 1
    beta2 <- 1
    
    # (4) Copula kappa = (k0, k1) 업데이트
    up_kcop <- update_kappa_copula(k0, k1,
                                   Y_bin, Y_con, z,
                                   alpha1, beta1,
                                   alpha2, beta2, kappa2,
                                   prop_Sigma = prop_kappa_Sigma,
                                   prior_mu = kappa_mu, prior_Sigma_inv = kappa_Sigma_inv)
    if (up_kcop$accepted) acc_kappa_cop <- acc_kappa_cop + 1L
    k0 <- up_kcop$k0; k1 <- up_kcop$k1
    
    # (5) save (copula 로그우도 기준)
    if (it %in% keep_idx) {
      s <- s + 1L
      z_save[s, , ] <- z
      
      alpha1_save[s] <- alpha1; beta1_save[s] <- beta1; xi1_save[s] <- xi1; psi1_save[s] <- psi1
      alpha2_save[s] <- alpha2; beta2_save[s] <- beta2; xi2_save[s] <- xi2; psi2_save[s] <- psi2; kappa2_save[s] <- kappa2
      k0_save[s] <- k0; k1_save[s] <- k1
      
      # 분해 저장: 연속층 ll + 베르누이(조건부) ll
      ll_con <- loglik_conti_network(Y_con, z, alpha2, beta2, kappa2)
      pit <- pit_Z2_lognormal(Y_con, z, alpha2, beta2, kappa2)
      z2 <- pit$z2; upper <- pit$upper; D <- pit$D
      eta1 <- alpha1 - beta1 * D
      qmat <- cutpoint_q_from_eta1(eta1)$q
      RHO <- rho_from_kappa(D, k0, k1)
      ll_bincond <- loglik_binary_cond_copula(Y_bin, z2, qmat, RHO, upper)
      ll_con_save[s] <- ll_con
      ll_bincond_save[s] <- ll_bincond
      ll_joint_save[s] <- ll_con + ll_bincond - 0.5 / sigma2 * sum(z^2)  # joint up to const
    }
    
    if (verbose && it %% 100 == 0) {
      # 현재 파라미터로 ll_con, ll_bincond를 즉석 계산
      ll_con_curr <- loglik_conti_network(Y_con, z, alpha2, beta2, kappa2)
      pit_curr <- pit_Z2_lognormal(Y_con, z, alpha2, beta2, kappa2)
      z2_curr <- pit_curr$z2; upper_curr <- pit_curr$upper; D_curr <- pit_curr$D
      eta1_curr <- alpha1 - beta1 * D_curr
      qmat_curr <- cutpoint_q_from_eta1(eta1_curr)$q
      RHO_curr  <- rho_from_kappa(D_curr, k0, k1)
      ll_bincond_curr <- loglik_binary_cond_copula(Y_bin, z2_curr, qmat_curr, RHO_curr, upper_curr)
      
      cat(sprintf(
        "[iter %d] a1=%.3f b1=%.3f | a2=%.3f b2=%.3f k2=%.3f | k0=%.3f k1=%.3f | ll_con=%.1f ll_bin|con=%.1f\n",
        it, alpha1, beta1, alpha2, beta2, kappa2, k0, k1, ll_con_curr, ll_bincond_curr
      ))
    }
  }
  
  # Procrustes 정렬 (기존 유지)
  map_idx <- which.max(ll_joint_save)
  Z_map <- matrix(z_save[map_idx, , ], n, d)
  if (requireNamespace("vegan", quietly = TRUE)) {
    for (ss in seq_len(ns)) {
      if (ss == map_idx) next
      Z_s <- matrix(z_save[ss, , ], n, d)
      fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
      z_save[ss, , ] <- predict(fit)
    }
  }
  
  list(
    samples = list(
      z = z_save,
      # binary
      alpha1 = alpha1_save, beta1 = beta1_save, xi1 = xi1_save, psi1 = psi1_save,
      # continuous
      alpha2 = alpha2_save, beta2 = beta2_save, kappa2 = kappa2_save, xi2 = xi2_save, psi2 = psi2_save,
      # copula
      k0 = k0_save, k1 = k1_save,
      # traces
      ll_con = ll_con_save, ll_bin_cond = ll_bincond_save, ll_joint = ll_joint_save
    ),
    accept = list(
      z = acc_z / iter,
      alpha1 = acc_alpha1 / iter, beta1 = acc_beta1 / iter,
      alpha2 = acc_alpha2 / iter, beta2 = acc_beta2 / iter, kappa2 = acc_kappa2 / iter,
      kappa_copula = acc_kappa_cop / iter
    ),
    config = list(
      iter = iter, burnin = burnin, thinning = thinning,
      d = d, sigma2 = sigma2,
      prop_sd = list(z = prop_z,
                     alpha_bin = prop_alpha_bin, beta_bin = prop_beta_bin,
                     alpha_con = prop_alpha_con, beta_con = prop_beta_con,
                     kappa2 = prop_kappa,  # lognormal layer var
                     kappa_copula = prop_kappa_Sigma),
      priors = list(
        binary = list(a0 = a0_bin, b0 = b0_bin, m0 = m0_bin, kappa0 = kappa0_bin,
                      alpha0 = alpha0_bin, beta0 = beta0_bin),
        conti  = list(a0 = a0_con, b0 = b0_con, m_kappa = m_kappa, s_kappa = s_kappa,
                      m0 = m0_con, kappa0 = kappa0_con, alpha0 = alpha0_con, beta0 = beta0_con),
        copula = list(mu = kappa_mu, Sigma = kappa_Sigma)
      )
    )
  )
}


library(network)
library(igraph)
library(brainGraph)
g <- make_graph("Zachary")
Y = as_adjacency_matrix(g, sparse = F)
comm <- communicability(g)
comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)


fit_cop <- LSJM_2layer_copula(
  Y_bin = Y,
  Y_con = katz_mat,          # 연속층 (여기서는 lognormal 마진으로 적합)
  iter = 7000, burnin = 1000, thinning = 3,
  d = 2, sigma2 = 1,
  # proposals
  prop_z = 0.4,
  prop_alpha_bin = 0.5, prop_beta_bin = 0.3,
  prop_alpha_con = 0.1, prop_beta_con = 0.1, prop_kappa = 0.1,
  prop_kappa_Sigma = diag(c(0.2, 0.2)^2),  # copula (k0,k1) 제안 분산
  # inits for copula
  k0_init = 0, k1_init = 0,
  verbose = TRUE
)

fit_cop$accept
par(mfrow = c(3,dim(fit_cop$samples$z)[3]))

for(i in 1:dim(fit_cop$samples$z)[2]){
  for(j in 1:dim(fit_cop$samples$z)[3]){
    ts.plot(fit_cop$samples$z[,i,j])
  }
}

par(mfrow = c(2,1))
ts.plot(fit_cop$samples$alpha1)
ts.plot(fit_cop$samples$alpha2)
ts.plot(fit_cop$samples$beta1)
ts.plot(fit_cop$samples$beta2)

ts.plot(fit_cop$samples$xi1)
ts.plot(fit_cop$samples$xi2)
ts.plot(fit_cop$samples$psi1)
ts.plot(fit_cop$samples$psi2)
ts.plot(fit_cop$samples$ll_bin)
ts.plot(fit_cop$samples$ll_con)

ts.plot(fit_cop$samples$k1)
ts.plot(fit_cop$samples$k0)
par(mfrow = c(1,1))
# likelihood
ts.plot(fit_cop$samples$ll_con)
ts.plot(fit_cop$samples$ll_bin_cond)
ts.plot(fit_cop$samples$ll_joint)
ts.plot(fit_cop$samples$kappa2)

par(mfrow = c(1,1))
latentmap <- colMeans(fit_cop$samples$z, dims = 1)
plot(latentmap,main = "colMeans(result$samples$z, dims = 1)",
     xlab = "x-axis",
     ylab = "y-axis",
     col = "grey", # 기본 점 색상
     pch = 19) 



# 특정 인덱스의 점에 빨간색 칠하기
# 3번째와 7번째 점
red <- c(1, 33, 34)

blue <- c(12)
# points() 함수를 사용하여 해당 점만 덧그림
points(x = latentmap[red,1],
       y = latentmap[red,2],
       col = "red",
       pch = 19, # 기본 플롯과 동일하게 설정
       cex = 1.5) # 점 크기를 조금 더 키워서 강조

points(x = latentmap[blue,1],
       y = latentmap[blue,2],
       col = "blue",
       pch = 19, # 기본 플롯과 동일하게 설정
       cex = 1.5) # 점 크기를 조금 더 키워서 강조
# ---- 숫자 라벨 추가 ----
n <- nrow(latentmap)
lab_col <- rep("grey20", n)
lab_col[red]  <- "red"
lab_col[blue] <- "blue"

# 각 점 위(pos=3)에 번호 붙이기 (겹침 줄이려면 offset 늘리기)
text(latentmap[,1], latentmap[,2],
     labels = 1:n, col = lab_col,
     cex = 0.8, pos = 3, offset = 0.3)


# in 3D
# install.packages("plotly")
library(plotly)

# latentmap: n×2 또는 n×3
X <- as.matrix(latentmap)
stopifnot(ncol(X) >= 2)
if (ncol(X) == 2) X <- cbind(X, 0)
colnames(X) <- c("x","y","z")

red  <- c(1, 33, 34)
blue <- c(12)

n <- nrow(X)
base_col  <- rep("grey", n)
base_col[red]  <- "red"
base_col[blue] <- "blue"
base_size <- rep(6, n); base_size[c(red, blue)] <- 10

labels <- paste0("idx: ", seq_len(n))

p <- plot_ly(type = "scatter3d", mode = "markers") |>
  add_markers(
    x = X[,1], y = X[,2], z = X[,3],
    marker = list(size = base_size),
    text = labels, hoverinfo = "text",
    color = I(base_col),
    showlegend = FALSE
  )

# ---- 항상 보이는 텍스트 라벨 (하이라이트 인덱스만) ----
hl <- unique(c(red, blue))
p <- p |>
  add_text(
    x = X[hl,1], y = X[hl,2], z = X[hl,3],
    text = hl,                     # 숫자 라벨
    textposition = "top center",
    textfont = list(size = 12, color = base_col[hl]),
    showlegend = FALSE,
    inherit = FALSE
  )

# ---- (옵션) 모든 점에 라벨을 항상 표시하려면 아래 주석 해제 ----
# p <- p |>
#   add_text(
#     x = X[,1], y = X[,2], z = X[,3],
#     text = seq_len(n),
#     textposition = "top center",
#     textfont = list(size = 10, color = "rgba(50,50,50,0.8)"),
#     showlegend = FALSE,
#     inherit = FALSE
#   )

p <- p |>
  layout(
    title = "colMeans(result$samples$z, dims = 1)",
    scene = list(
      xaxis = list(title = "x-axis"),
      yaxis = list(title = "y-axis"),
      zaxis = list(title = if (ncol(latentmap) >= 3) "z-axis" else "z=0 plane"),
      aspectmode = "cube"
    )
  )

p

motif_GOF_test(fit_cop, Y, B = 200, k_grid = 1:5, S_grid = 1:5, arg = 'simple_motif', custom_text = NULL, model = 'LSJM', L2_dist)
motif_GOF_test(fit_cop, Y, B = 200, k_grid = 1:5, S_grid = 1:5, arg = 'kstars', custom_text = NULL, model = 'LSJM', L2_dist)
motif_GOF_test(fit_cop, Y, B = 200, k_grid = 1:5, S_grid = 1:5, arg = 'dsp', custom_text = NULL, model = 'LSJM', L2_dist)
motif_GOF_test(fit_cop, Y, B = 200, k_grid = 1:5, S_grid = 1:5, arg = 'esp', custom_text = NULL, model = 'LSJM', L2_dist)

