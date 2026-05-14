# rm(list = ls())
################################################################################
# Hierarchical Joint LSM (Binary + Continuous)
# - PDF (1111_meeting.pdf) 기반
# - z_i: global latent position
# - u_i^(l): layer-specific latent position, u_i^(l) ~ N(z_i, sigma2_l * I)
# - eta_ij^(l): alpha^(l) + theta_i^(l) + theta_j^(l) - beta^(l) * d(u_i^(l), u_j^(l))
################################################################################
library(network)
library(igraph)
library(brainGraph)
# install.packages("invgamma") # sigma2 업데이트를 위해 필요
library(invgamma) 

## -------------------------- (0) Utility ------------------------
# L2 거리 계산 (기존 코드와 동일)
L2_dist <- function(z) {
  n <- nrow(z)
  D <- matrix(0.0, n, n)
  if (n == 0) return(D)
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

## ---------- (1) loglike, prior, update rule (Hierarchical) ----------

################################################################
## (NEW) 계층적 모델용 로그 우도 함수
################################################################

# Binary network (Hierarchical)
loglik_binary_network_hier <- function(Y, u, alpha, theta, beta) {
  D <- L2_dist(u)
  # eta_ij = alpha + theta_i + theta_j - beta * d_ij
  theta_sum <- outer(theta, theta, "+")
  eta <- alpha + theta_sum - beta * D
  
  pi <- stats::plogis(eta)
  n <- nrow(Y)
  ll <- 0.0
  if (n == 0) return(ll)
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

# Continuous network (Hierarchical)
loglik_conti_network_hier <- function(Y, u, alpha, theta, beta, kappa, eps = 1e-12) {
  D <- L2_dist(u)
  # eta_ij = alpha + theta_i + theta_j - beta * d_ij
  theta_sum <- outer(theta, theta, "+")
  eta <- alpha + theta_sum - beta * D
  
  upper <- which(upper.tri(Y), arr.ind = TRUE)
  y <- pmax(Y[upper], eps)
  mu_log <- eta[upper] - kappa/2  # 평균 exp(eta)로 맞춰주기 위함
  ll <- sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
  return(ll)
}

################################################################
## (NEW) 계층적 모델용 사전분포 함수
################################################################

# z_i ~ normal(0, sigma2_z * I) 
log_prior_z_i <- function(z_i, sigma2_z = 1.0) {
  return(sum(stats::dnorm(z_i, mean = 0, sd = sqrt(sigma2_z), log = T)))
}

# u_i^(l) | z_i, sigma2_l ~ N(z_i, sigma2_l * I) 
log_prior_u_i <- function(u_i, z_i, sigma2_l) {
  return(sum(stats::dnorm(u_i, mean = z_i, sd = sqrt(sigma2_l), log = T)))
}

# theta_i^(l) ~ normal(0, sigma2_theta * I) [cite: 13]
log_prior_theta_i <- function(theta_i, sigma2_theta = 1.0) {
  return(stats::dnorm(theta_i, mean = 0, sd = sqrt(sigma2_theta), log = T))
}

# alpha ~ normal(xi, psi2) (기존 코드와 동일)
log_prior_alpha <- function(alpha, xi, psi2) {
  return(stats::dnorm(alpha, mean = xi, sd = sqrt(psi2), log = T))
}

# beta ~ Gamma(a0, b0) (기존 코드와 동일) [cite: 16]
log_prior_beta <- function(beta, a, b) {
  return(stats::dgamma(beta, shape = a, rate = b, log = T))
}

# log(kappa) ~ normal(a, b) (기존 코드와 동일)
log_prior_kappa <- function(kappa, a, b) {
  return(stats::dnorm(log(kappa), a, sqrt(b), log = T)-log(kappa))
}

################################################################
## (NEW) 계층적 모델용 업데이트 함수
################################################################

# --- Binary Layer (l=1) ---

bin_update_alpha_hier <- function(alpha, u, theta, Y, xi, psi2, prop_alpha, beta) {
  ll_curr <- loglik_binary_network_hier(Y, u, alpha, theta, beta) + 
    log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_binary_network_hier(Y, u, alpha_prop, theta, beta) + 
    log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

bin_update_beta_hier <- function(beta, u, theta, Y, alpha, prop_beta, a, b){
  # beta ~ Gamma(a,b)
  # proposal: log(beta) ~ N(log(beta_curr), prop_beta^2)
  # Jacobian: log(beta_prop) [cite: 41, 42]
  curr <- loglik_binary_network_hier(Y, u, alpha, theta, beta) + 
    log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_binary_network_hier(Y, u, alpha, theta, beta_prop) + 
    log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

bin_update_theta_i_hier <- function(i, theta, Y, u, alpha, beta, prop_theta, sigma2_theta) {
  theta_curr <- theta
  theta_i_curr <- theta[i]
  # likelihood + prior [cite: 35]
  ll_curr <- loglik_binary_network_hier(Y, u, alpha, theta_curr, beta) +
    log_prior_theta_i(theta_i_curr, sigma2_theta)
  
  theta_prop <- theta
  theta_i_prop <- theta_i_curr + rnorm(1, 0, prop_theta)
  theta_prop[i] <- theta_i_prop
  
  ll_prop <- loglik_binary_network_hier(Y, u, alpha, theta_prop, beta) +
    log_prior_theta_i(theta_i_prop, sigma2_theta)
  
  if (log(runif(1)) < (ll_prop - ll_curr)) {
    return(list(theta = theta_prop, accepted = TRUE))
  } else {
    return(list(theta = theta_curr, accepted = FALSE))
  }
}

bin_update_u_i_hier <- function(i, u, Y, alpha, theta, beta, z_i, sigma2_l, prop_u, d) {
  u_curr <- u
  u_i_curr <- u[i, ]
  # likelihood + prior P(u_i | z_i) [cite: 46]
  ll_curr <- loglik_binary_network_hier(Y, u_curr, alpha, theta, beta) +
    log_prior_u_i(u_i_curr, z_i, sigma2_l)
  
  u_prop <- u
  u_i_prop <- u_i_curr + rnorm(d, 0, prop_u)
  u_prop[i, ] <- u_i_prop
  
  ll_prop <- loglik_binary_network_hier(Y, u_prop, alpha, theta, beta) +
    log_prior_u_i(u_i_prop, z_i, sigma2_l)
  
  if (log(runif(1)) < (ll_prop - ll_curr)) {
    return(list(u = u_prop, accepted = TRUE))
  } else {
    return(list(u = u_curr, accepted = FALSE))
  }
}

# --- Continuous Layer (l=2) ---

con_update_alpha_hier <- function(alpha, u, theta, Y, xi, psi2, prop_alpha, beta, kappa) {
  ll_curr <- loglik_conti_network_hier(Y, u, alpha, theta, beta, kappa) + 
    log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_conti_network_hier(Y, u, alpha_prop, theta, beta, kappa) + 
    log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

con_update_beta_hier <- function(beta, u, theta, Y, alpha, prop_beta, a, b, kappa){
  curr <- loglik_conti_network_hier(Y, u, alpha, theta, beta, kappa) + 
    log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_conti_network_hier(Y, u, alpha, theta, beta_prop, kappa) + 
    log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

con_update_kappa_hier <- function(kappa, u, theta, Y, alpha, beta, prop_kappa, m_kappa, s_kappa){
  # log(kappa) ~ normal(m_kappa, s_kappa)
  curr <- loglik_conti_network_hier(Y, u, alpha, theta, beta, kappa) + 
    log_prior_kappa(kappa, m_kappa, s_kappa)
  kappa_prop <- exp(log(kappa) + rnorm(1, 0, prop_kappa))
  
  prop <- loglik_conti_network_hier(Y, u, alpha, theta, beta, kappa_prop) + 
    log_prior_kappa(kappa_prop, m_kappa, s_kappa)
  # log-정규 제안 보정 항 추가:
  acc <- (prop - curr) + (log(kappa_prop) - log(kappa))
  if (log(runif(1)) < acc) {
    return(list(kappa=kappa_prop, accepted=TRUE))
  } else {
    return(list(kappa=kappa, accepted=FALSE))
  }
}

con_update_theta_i_hier <- function(i, theta, Y, u, alpha, beta, kappa, prop_theta, sigma2_theta) {
  theta_curr <- theta
  theta_i_curr <- theta[i]
  ll_curr <- loglik_conti_network_hier(Y, u, alpha, theta_curr, beta, kappa) +
    log_prior_theta_i(theta_i_curr, sigma2_theta)
  
  theta_prop <- theta
  theta_i_prop <- theta_i_curr + rnorm(1, 0, prop_theta)
  theta_prop[i] <- theta_i_prop
  
  ll_prop <- loglik_conti_network_hier(Y, u, alpha, theta_prop, beta, kappa) +
    log_prior_theta_i(theta_i_prop, sigma2_theta)
  
  if (log(runif(1)) < (ll_prop - ll_curr)) {
    return(list(theta = theta_prop, accepted = TRUE))
  } else {
    return(list(theta = theta_curr, accepted = FALSE))
  }
}

con_update_u_i_hier <- function(i, u, Y, alpha, theta, beta, kappa, z_i, sigma2_l, prop_u, d) {
  u_curr <- u
  u_i_curr <- u[i, ]
  ll_curr <- loglik_conti_network_hier(Y, u_curr, alpha, theta, beta, kappa) +
    log_prior_u_i(u_i_curr, z_i, sigma2_l)
  
  u_prop <- u
  u_i_prop <- u_i_curr + rnorm(d, 0, prop_u)
  u_prop[i, ] <- u_i_prop
  
  ll_prop <- loglik_conti_network_hier(Y, u_prop, alpha, theta, beta, kappa) +
    log_prior_u_i(u_i_prop, z_i, sigma2_l)
  
  if (log(runif(1)) < (ll_prop - ll_curr)) {
    return(list(u = u_prop, accepted = TRUE))
  } else {
    return(list(u = u_curr, accepted = FALSE))
  }
}


# --- Gibbs Samplers (Shared logic for l=1, 2) ---

# Hyperparameters (xi, psi2) for alpha (기존 코드와 동일)
update_xi_psi2_gibbs <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
  n <- length(alpha) # alpha가 하나뿐이므로 n=1. 계층구조를 쓰지 않음.
  # PDF [cite: 14, 15] 에서는 계층 구조를 썼다고 하지만,
  # alpha^(l) ~ N(0,1), theta_i^(l) ~ N(0,1) 로 고정했으므로 [cite: 12, 13]
  # 이 함수는 사실상 현재 모델(PDF)에서는 사용되지 않아야 함.
  # 하지만 기존 코드와의 호환성을 위해 남겨두고, xi, psi2를 고정값으로 사용.
  # 만약 alpha^(l) ~ N(xi, psi2) 를 추정한다면 이 함수를 사용.
  
  # 1) xi | psi, alpha
  kappa_n <- (1/psi2 + 1/kappa0)^(-1)
  m_n <- kappa_n * (m0/kappa0 + alpha/psi2)
  xi_new <- rnorm(1, mean = m_n, sd = sqrt(kappa_n))
  
  # 2) psi | xi, alpha
  shape <- alpha0 + n/2
  scale <- beta0 + 0.5 * sum((alpha - xi_new)^2)
  psi2_new <- 1 / rgamma(1, shape = shape, rate = scale)
  
  list(xi = xi_new, psi2 = psi2_new)
}

# (NEW) sigma_l^2 | u^(l), z ~ IG(a_post, b_post)
# Prior: sigma_l^2 ~ IG(a1, b1) 
update_sigma2_l_gibbs <- function(u_l, z, a1, b1) {
  n <- nrow(u_l)
  d <- ncol(u_l)
  
  a_post <- a1 + (n * d) / 2
  
  ss <- sum((u_l - z)^2)
  b_post <- b1 + ss / 2
  
  # dinvgamma(shape = a, scale = b)
  # rgamma(shape = a, rate = b) -> 1/rgamma(shape=a, rate=b)가 IG(a,b)
  sigma2_new <- 1 / rgamma(1, shape = a_post, rate = b_post)
  
  return(sigma2_new)
}

# --- Global Parameter (z_i) ---

# (NEW) z_i | u_i^(1), u_i^(2), sigma2_1, sigma2_2
update_z_i_hier <- function(i, z, u1, u2, sigma2_1, sigma2_2, prop_z, d, sigma2_z) {
  z_curr <- z
  z_i_curr <- z[i, ]
  
  # Target: P(u_i^(1) | z_i) * P(u_i^(2) | z_i) * P(z_i) [cite: 50, 53]
  ll_curr <- log_prior_u_i(u1[i, ], z_i_curr, sigma2_1) +
    log_prior_u_i(u2[i, ], z_i_curr, sigma2_2) +
    log_prior_z_i(z_i_curr, sigma2_z)
  
  z_prop <- z
  z_i_prop <- z_i_curr + rnorm(d, 0, prop_z)
  z_prop[i, ] <- z_i_prop
  
  ll_prop <- log_prior_u_i(u1[i, ], z_i_prop, sigma2_1) +
    log_prior_u_i(u2[i, ], z_i_prop, sigma2_2) +
    log_prior_z_i(z_i_prop, sigma2_z)
  
  if (log(runif(1)) < (ll_prop - ll_curr)) {
    return(list(z = z_prop, accepted = TRUE))
  } else {
    return(list(z = z_curr, accepted = FALSE))
  }
}


## ---------- (4) 메인 드라이버 (Hierarchical) ----------
MNLPM_2layer <- function(Y_bin, Y_con,
                            iter = 5000, burnin = 1000, thinning = 5,
                            d = 2,
                            # Priors
                            # z_i ~ N(0, sigma2_z * I) 
                            sigma2_z = 1.0,
                            # theta_i ~ N(0, sigma2_theta * I) [cite: 13]
                            sigma2_theta = 1.0, 
                            # sigma_l^2 ~ IG(a1, b1) 
                            a1 = 2.0, b1 = 1.0,
                            # beta_l ~ Gamma(a0, b0) [cite: 16]
                            a0_bin = 0.5, b0_bin = 0.5,
                            a0_con = 0.5, b0_con = 0.5,
                            # alpha_l ~ N(xi_l, psi2_l) [cite: 12]
                            # PDF에서는 (0,1)로 고정 [cite: 12] -> xi=0, psi2=1
                            xi1_prior = 0, psi1_prior = 1.0,
                            xi2_prior = 0, psi2_prior = 1.0,
                            # (참고: 기존 코드의 hyperprior는 사용 안 함)
                            # m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2,
                            # kappa (log-normal)
                            m_kappa = 0, s_kappa = 1,
                            # proposals
                            prop_z = 0.1,
                            prop_u_bin = 0.1, prop_u_con = 0.1,
                            prop_theta_bin = 0.1, prop_theta_con = 0.1,
                            prop_alpha_bin = 0.05, prop_beta_bin = 0.05,
                            prop_alpha_con = 0.05, prop_beta_con = 0.05, prop_kappa = 0.2,
                            # inits
                            z_init = NULL,
                            alpha1_init = 0, beta1_init = 1,
                            alpha2_init = 0, beta2_init = 1, kappa2_init = 1,
                            verbose = TRUE,
                            seed = 1) {
  
  set.seed(seed)
  stopifnot(is.matrix(Y_bin), is.matrix(Y_con))
  n <- nrow(Y_bin); stopifnot(ncol(Y_bin) == n, nrow(Y_con) == n, ncol(Y_con) == n)
  if (max(abs(Y_bin - t(Y_bin))) > 1e-8) stop("Y_bin must be symmetric.")
  if (max(abs(Y_con - t(Y_con))) > 1e-8) stop("Y_con must be symmetric.")
  
  # --- init z (global) ---
  if (is.null(z_init)) {
    z <- matrix(rnorm(n * d, 0, sqrt(sigma2_z)), n, d)
  } else {
    stopifnot(is.matrix(z_init), nrow(z_init) == n, ncol(z_init) == d)
    z <- z_init
  }
  
  # --- init layer-specific params ---
  # l=1 (binary)
  sigma2_1 <- 1.0
  u1 <- z + matrix(rnorm(n * d, 0, sqrt(sigma2_1)), n, d)
  theta1 <- rnorm(n, 0, sqrt(sigma2_theta))
  alpha1 <- alpha1_init; beta1 <- beta1_init
  
  # l=2 (continuous)
  sigma2_2 <- 1.0
  u2 <- z + matrix(rnorm(n * d, 0, sqrt(sigma2_2)), n, d)
  theta2 <- rnorm(n, 0, sqrt(sigma2_theta))
  alpha2 <- alpha2_init; beta2 <- beta2_init; kappa2 <- kappa2_init
  
  # hyperparams (PDF [cite: 12]에 따라 고정)
  xi1 <- xi1_prior; psi1 <- psi1_prior
  xi2 <- xi2_prior; psi2 <- psi2_prior
  
  # --- storage ---
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning)
  ns <- length(keep_idx)
  
  z_save <- array(NA_real_, dim = c(ns, n, d))
  # binary layer (l=1)
  u1_save <- array(NA_real_, dim = c(ns, n, d))
  theta1_save <- array(NA_real_, dim = c(ns, n))
  alpha1_save <- beta1_save <- sigma2_1_save <- numeric(ns)
  # conti layer (l=2)
  u2_save <- array(NA_real_, dim = c(ns, n, d))
  theta2_save <- array(NA_real_, dim = c(ns, n))
  alpha2_save <- beta2_save <- kappa2_save <- sigma2_2_save <- numeric(ns)
  
  # traces
  ll_bin_save <- ll_con_save <- numeric(ns)
  
  # acceptances
  acc_z <- integer(n)
  acc_u1 <- integer(n); acc_theta1 <- integer(n)
  acc_alpha1 <- 0L; acc_beta1 <- 0L
  acc_u2 <- integer(n); acc_theta2 <- integer(n)
  acc_alpha2 <- 0L; acc_beta2 <- 0L; acc_kappa2 <- 0L
  
  s <- 0L
  
  # --- sampler ---
  for (it in 1:iter) {
    
    # (1) Update global z (node-by-node)
    for (j in 1:n) {
      up <- update_z_i_hier(j, z, u1, u2, sigma2_1, sigma2_2, prop_z, d, sigma2_z)
      z <- up$z
      if (up$accepted) acc_z[j] <- acc_z[j] + 1L
    }
    
    # (2) Update Binary layer (l=1) params
    # u_i^(1)
    for (j in 1:n) {
      up_u1 <- bin_update_u_i_hier(j, u1, Y_bin, alpha1, theta1, beta1, 
                                   z[j, ], sigma2_1, prop_u_bin, d)
      u1 <- up_u1$u
      if (up_u1$accepted) acc_u1[j] <- acc_u1[j] + 1L
    }
    # theta_i^(1)
    for (j in 1:n) {
      up_t1 <- bin_update_theta_i_hier(j, theta1, Y_bin, u1, alpha1, beta1,
                                       prop_theta_bin, sigma2_theta)
      theta1 <- up_t1$theta
      if (up_t1$accepted) acc_theta1[j] <- acc_theta1[j] + 1L
    }
    # alpha^(1)
    up_a1 <- bin_update_alpha_hier(alpha1, u1, theta1, Y_bin, xi1, psi1, prop_alpha_bin, beta1)
    alpha1 <- up_a1$alpha; if (up_a1$accepted) acc_alpha1 <- acc_alpha1 + 1L
    # beta^(1)
    up_b1 <- bin_update_beta_hier(beta1, u1, theta1, Y_bin, alpha1, prop_beta_bin, a0_bin, b0_bin)
    beta1 <- up_b1$beta;   if (up_b1$accepted) acc_beta1 <- acc_beta1 + 1L
    # sigma2^(1)
    sigma2_1 <- update_sigma2_l_gibbs(u1, z, a1, b1)
    
    
    # (3) Update Continuous layer (l=2) params
    # u_i^(2)
    for (j in 1:n) {
      up_u2 <- con_update_u_i_hier(j, u2, Y_con, alpha2, theta2, beta2, kappa2,
                                   z[j, ], sigma2_2, prop_u_con, d)
      u2 <- up_u2$u
      if (up_u2$accepted) acc_u2[j] <- acc_u2[j] + 1L
    }
    # theta_i^(2)
    for (j in 1:n) {
      up_t2 <- con_update_theta_i_hier(j, theta2, Y_con, u2, alpha2, beta2, kappa2,
                                       prop_theta_con, sigma2_theta)
      theta2 <- up_t2$theta
      if (up_t2$accepted) acc_theta2[j] <- acc_theta2[j] + 1L
    }
    # alpha^(2)
    up_a2 <- con_update_alpha_hier(alpha2, u2, theta2, Y_con, xi2, psi2, prop_alpha_con, beta2, kappa2)
    alpha2 <- up_a2$alpha; if (up_a2$accepted) acc_alpha2 <- acc_alpha2 + 1L
    # beta^(2)
    up_b2 <- con_update_beta_hier(beta2, u2, theta2, Y_con, alpha2, prop_beta_con, a0_con, b0_con, kappa2)
    beta2 <- up_b2$beta;   if (up_b2$accepted) acc_beta2 <- acc_beta2 + 1L
    # kappa^(2)
    up_k2 <- con_update_kappa_hier(kappa2, u2, theta2, Y_con, alpha2, beta2, prop_kappa, m_kappa, s_kappa)
    kappa2 <- up_k2$kappa; if (up_k2$accepted) acc_kappa2 <- acc_kappa2 + 1L
    # sigma2^(2)
    sigma2_2 <- update_sigma2_l_gibbs(u2, z, a1, b1)
    
    
    # (4) save
    if (it %in% keep_idx) {
      s <- s + 1L
      z_save[s, , ] <- z
      
      u1_save[s, , ] <- u1
      theta1_save[s, ] <- theta1
      alpha1_save[s] <- alpha1
      beta1_save[s] <- beta1
      sigma2_1_save[s] <- sigma2_1
      
      u2_save[s, , ] <- u2
      theta2_save[s, ] <- theta2
      alpha2_save[s] <- alpha2
      beta2_save[s] <- beta2
      kappa2_save[s] <- kappa2
      sigma2_2_save[s] <- sigma2_2
      
      ll_bin_save[s] <- loglik_binary_network_hier(Y_bin, u1, alpha1, theta1, beta1)
      ll_con_save[s] <- loglik_conti_network_hier(Y_con, u2, alpha2, theta2, beta2, kappa2)
    }
    
    if (verbose && it %% 100 == 0) {
      cat(sprintf("[iter %d] a1=%.2f b1=%.2f | a2=%.2f b2=%.2f k2=%.2f | s2_1=%.2f s2_2=%.2f | ll_bin=%.1f ll_con=%.1f\n",
                  it, alpha1, beta1, alpha2, beta2, kappa2, sigma2_1, sigma2_2,
                  loglik_binary_network_hier(Y_bin, u1, alpha1, theta1, beta1),
                  loglik_conti_network_hier(Y_con, u2, alpha2, theta2, beta2, kappa2)))
    }
  }
  
  # --- Procrustes alignment to MAP (by ll_bin + ll_con) ---
  map_idx <- which.max(ll_bin_save + ll_con_save)
  Z_map <- matrix(z_save[map_idx, , ], n, d)
  
  if (requireNamespace("vegan", quietly = TRUE)) {
    for (ss in seq_len(ns)) {
      if (ss == map_idx) next
      
      Z_s <- matrix(z_save[ss, , ], n, d)
      U1_s <- matrix(u1_save[ss, , ], n, d)
      U2_s <- matrix(u2_save[ss, , ], n, d)
      
      # z를 기준으로 정렬
      fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
      
      # z, u1, u2에 동일한 회전/변환 적용 - 이 부분이 문제가 있는 것 아닐까? 왜 동일한 회전/변환 적용하는 것이 정당화되는가?
      # 왜냐면 z, u1, u2 가 joint 하게 추정되므로.
      z_save[ss, , ] <- predict(fit)
      u1_save[ss, , ] <- predict(fit, newdata = U1_s)
      u2_save[ss, , ] <- predict(fit, newdata = U2_s)
    }
  } else {
    warning("Package 'vegan' not found. Skipping Procrustes alignment.")
  }
  
  list(
    samples = list(
      z = z_save,
      # binary
      u1 = u1_save,
      theta1 = theta1_save,
      alpha1 = alpha1_save, 
      beta1 = beta1_save, 
      sigma2_1 = sigma2_1_save,
      # continuous
      u2 = u2_save,
      theta2 = theta2_save,
      alpha2 = alpha2_save, 
      beta2 = beta2_save, 
      kappa2 = kappa2_save, 
      sigma2_2 = sigma2_2_save,
      # traces
      ll_bin = ll_bin_save, 
      ll_con = ll_con_save
    ),
    accept = list(
      z = acc_z / iter,
      u1 = acc_u1 / iter,
      theta1 = acc_theta1 / iter,
      alpha1 = acc_alpha1 / iter, 
      beta1 = acc_beta1 / iter,
      u2 = acc_u2 / iter,
      theta2 = acc_theta2 / iter,
      alpha2 = acc_alpha2 / iter, 
      beta2 = acc_beta2 / iter, 
      kappa2 = acc_kappa2 / iter
    ),
    config = list(
      iter = iter, burnin = burnin, thinning = thinning,
      d = d, 
      prop_sd = list(z = prop_z,
                     u_bin = prop_u_bin, theta_bin = prop_theta_bin,
                     alpha_bin = prop_alpha_bin, beta_bin = prop_beta_bin,
                     u_con = prop_u_con, theta_con = prop_theta_con,
                     alpha_con = prop_alpha_con, beta_con = prop_beta_con, 
                     kappa = prop_kappa),
      priors = list(
        sigma2_z = sigma2_z,
        sigma2_theta = sigma2_theta,
        sigma2_l = list(a1 = a1, b1 = b1),
        binary = list(a0 = a0_bin, b0 = b0_bin, xi = xi1_prior, psi2 = psi1_prior),
        conti  = list(a0 = a0_con, b0 = b0_con, m_kappa = m_kappa, s_kappa = s_kappa,
                      xi = xi2_prior, psi2 = psi2_prior)
      )
    )
  )
}