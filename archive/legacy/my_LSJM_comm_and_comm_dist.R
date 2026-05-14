rm(list = ls())
################################################################################
# LSJM (Continuous + Continuous) with shared latent positions z and shared sigma2
################################################################################
library(network)
library(igraph)
library(brainGraph)
g <- make_graph("Zachary")
Y = as_adjacency_matrix(g, sparse = F)
comm <- communicability(g)
comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)

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

# test: lognormal update for beta
update_phi_beta <- function(phi, z, Y, alpha, loglik_fn, prop_phi, kappa,
                            mu_phi = 0, s2_phi = 0.5^2,  # prior: N(mu_phi, s2_phi)
                            extra_args = list()) {
  beta      <- exp(phi)
  curr <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta, kappa = kappa))) +
    dnorm(phi, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  
  phi_prop  <- phi + rnorm(1, 0, prop_phi)    # 대칭 제안
  beta_prop <- exp(phi_prop)
  prop <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta_prop, kappa = kappa))) +
    dnorm(phi_prop, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  
  if (log(runif(1)) < (prop - curr)) {
    list(phi = phi_prop, accepted = TRUE)
  } else {
    list(phi = phi, accepted = FALSE)
  }
}


loglik_conti_network_comm <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  # D <- L2_dist(z)
  D <- dot_dist(z)
  # communicability distance: eta <- alpha + beta*D 를 사용해야 함. communicability 와 그 의미가 반대이므로.
  eta <- alpha - beta*D
  upper <- which(upper.tri(Y), arr.ind = TRUE)
  y <- pmax(Y[upper], eps)
  mu_log <- eta[upper] - kappa/2        # 평균 exp(eta)로 맞춰주기 위함
  ll <- sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
  return(ll)
}

loglik_conti_network_comm_dist <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  # D <- L2_dist(z)
  D <- dot_dist(z)
  # communicability distance: eta <- alpha + beta*D 를 사용해야 함. communicability 와 그 의미가 반대이므로.
  eta <- alpha + beta*D
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
  return(stats::dnorm(log(kappa), a, sqrt(b), log = T))
}

## ---------- (2) 레이어별 파라미터 업데이트 래퍼 ----------

# Continuous layer updates (MH + Gibbs for xi2, psi2)
# communicability
# MH
con_update_alpha_comm <- function(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa) {
  ll_curr <- loglik_conti_network_comm(Y, z, alpha, beta, kappa) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_conti_network_comm(Y, z, alpha_prop, beta, kappa) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

con_update_beta_comm <- function(beta, z, Y, alpha, prop_beta, a, b, kappa){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_conti_network_comm(Y, z, alpha, beta, kappa) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_conti_network_comm(Y, z, alpha, beta_prop, kappa) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

con_update_kappa_comm <- function(kappa, z, Y, alpha, beta, prop_kappa, m_kappa = 0, s_kappa = 1){
  # log(kappa) ~ normal(m_kappa, s_kappa)
  curr <- loglik_conti_network_comm(Y, z, alpha, beta, kappa) + log_prior_kappa(kappa, m_kappa, s_kappa)
  kappa_prop <- exp(log(kappa) + rnorm(1, 0, prop_kappa))
  
  prop <- loglik_conti_network_comm(Y, z, alpha, beta, kappa_prop) + log_prior_kappa(kappa_prop, m_kappa, s_kappa)
  if (log(runif(1)) < (prop - curr)) {
    return(list(kappa=kappa_prop, accepted=TRUE))
  } else {
    return(list(kappa=kappa, accepted=FALSE))
  }
}

# Gibbs
con_update_xi_psi2_comm <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
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

# communicability distance
# MH
con_update_alpha_comm_dist <- function(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa) {
  ll_curr <- loglik_conti_network_comm_dist(Y, z, alpha, beta, kappa) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_conti_network_comm_dist(Y, z, alpha_prop, beta, kappa) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

con_update_beta_comm_dist <- function(beta, z, Y, alpha, prop_beta, a, b, kappa){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_conti_network_comm_dist(Y, z, alpha, beta, kappa) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_conti_network_comm_dist(Y, z, alpha, beta_prop, kappa) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

con_update_kappa_comm_dist <- function(kappa, z, Y, alpha, beta, prop_kappa, m_kappa = 0, s_kappa = 1){
  # log(kappa) ~ normal(m_kappa, s_kappa)
  curr <- loglik_conti_network_comm_dist(Y, z, alpha, beta, kappa) + log_prior_kappa(kappa, m_kappa, s_kappa)
  kappa_prop <- exp(log(kappa) + rnorm(1, 0, prop_kappa))
  
  prop <- loglik_conti_network_comm_dist(Y, z, alpha, beta, kappa_prop) + log_prior_kappa(kappa_prop, m_kappa, s_kappa)
  if (log(runif(1)) < (prop - curr)) {
    return(list(kappa=kappa_prop, accepted=TRUE))
  } else {
    return(list(kappa=kappa, accepted=FALSE))
  }
}

# Gibbs
con_update_xi_psi2_comm_dist <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
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
lsjm_update_z_single <- function(j, z, Y_comm, Y_comm_dist,
                                 alpha1, beta1, kappa2_1,
                                 alpha2, beta2, kappa2_2,
                                 sigma2, prop_z, d) {
  zi_old <- z[j, ]
  z_prop <- z
  zi_prop <- zi_old + rnorm(d, 0, prop_z)  # <-- 제안분산은 prop_z
  z_prop[j, ] <- zi_prop
  
  # 두 레이어 로그우도 합 + 공통 prior(z_j | 0, sigma2 I)
  ll_old <- loglik_conti_network_comm(Y_comm, z, alpha1, beta1, kappa2_1) +
    loglik_conti_network_comm_dist(Y_comm_dist, z, alpha2, beta2, kappa2_2) +
    log_prior_z_i(zi_old, sigma2)
  ll_new <- loglik_conti_network_comm(Y_comm, z_prop, alpha1, beta1, kappa2_1) +
    loglik_conti_network_comm_dist(Y_comm_dist, z_prop, alpha2, beta2, kappa2_2) +
    log_prior_z_i(zi_prop, sigma2)
  
  if (log(runif(1)) < (ll_new - ll_old)) {
    list(z = z_prop, accepted = TRUE)
  } else {
    list(z = z, accepted = FALSE)
  }
}
Y_comm<-comm; Y_comm_dist<-comm_dist
iter = 5000; burnin = 1000; thinning = 5;
d = 2;
# shared z prior variance
sigma2 = 1.0;
# comm priors/hypers
a0_1 = 0.5; b0_1 = 0.5;
m_kappa_1 = 0; s_kappa_1 = 1;
m0_1 = 0; kappa0_1 = 1/2; alpha0_1 = 3; beta0_1 = 1/2;
# comm_dist priors/hypers
a0_2 = 0.5; b0_2 = 0.5;
m_kappa_2 = 0; s_kappa_2 = 1;
m0_2 = 0; kappa0_2 = 1/2; alpha0_2 = 3; beta0_2 = 1/2;
# proposals
prop_z = 0.1;
prop_alpha_1 = 0.05; prop_beta_1 = 0.05; prop_kappa_1 = 0.2;
prop_alpha_2 = 0.05; prop_beta_2 = 0.05; prop_kappa_2 = 0.2;
# inits
z_init = NULL;
alpha1_init = 0; beta1_init = 1; xi1_init = 0; psi2_1_init = 1; kappa2_1_init = 1;
alpha2_init = 0; beta2_init = 1; xi2_init = 0; psi2_2_init = 1; kappa2_2_init = 1;
verbose = TRUE;
seed = 42
## ---------- (4) 메인 드라이버 ----------
LSJM_2layer <- function(Y_comm, Y_comm_dist,
                        iter = 5000, burnin = 1000, thinning = 5,
                        d = 2,
                        # shared z prior variance
                        sigma2 = 1.0,
                        # comm priors/hypers
                        a0_1 = 0.5, b0_1 = 0.5,
                        m_kappa_1 = 0, s_kappa_1 = 1,
                        m0_1 = 0, kappa0_1 = 1/2, alpha0_1 = 3, beta0_1 = 1/2,
                        # comm_dist priors/hypers
                        a0_2 = 0.5, b0_2 = 0.5,
                        m_kappa_2 = 0, s_kappa_2 = 1,
                        m0_2 = 0, kappa0_2 = 1/2, alpha0_2 = 3, beta0_2 = 1/2,
                        # proposals
                        prop_z = 0.1,
                        prop_alpha_1 = 0.05, prop_beta_1 = 0.05, prop_kappa_1 = 0.2,
                        prop_alpha_2 = 0.05, prop_beta_2 = 0.05, prop_kappa_2 = 0.2,
                        # inits
                        z_init = NULL,
                        alpha1_init = 0, beta1_init = 1, xi1_init = 0, psi2_1_init = 1, kappa2_1_init = 1,
                        alpha2_init = 0, beta2_init = 1, xi2_init = 0, psi2_2_init = 1, kappa2_2_init = 1,
                        verbose = TRUE,
                        seed = 1) {
  
  set.seed(seed)
  stopifnot(is.matrix(Y_comm_dist), is.matrix(Y_comm_dist))
  n <- nrow(Y_comm); stopifnot(ncol(Y_comm) == n, nrow(Y_comm_dist) == n, ncol(Y_comm_dist) == n)
  if (max(abs(Y_comm - t(Y_comm))) > 1e-8) stop("Y_bin must be symmetric.")
  if (max(abs(Y_comm_dist - t(Y_comm_dist))) > 1e-8) stop("Y_con must be symmetric.")
  
  # --- init z (공통) ---
  if (is.null(z_init)) {
    z <- matrix(rnorm(n * d, 0, sqrt(sigma2)), n, d)
  } else {
    stopifnot(is.matrix(z_init), nrow(z_init) == n, ncol(z_init) == d)
    z <- z_init
  }
  
  # --- layer params ---
  alpha1 <- alpha1_init; beta1 <- beta1_init; xi1 <- xi1_init; psi2_1 <- psi2_1_init; kappa2_1 <- kappa2_1_init
  alpha2 <- alpha2_init; beta2 <- beta2_init; xi2 <- xi2_init; psi2_2 <- psi2_2_init; kappa2_2 <- kappa2_2_init
  # test: lognormal beta init
  phi1 <- log(beta1_init); phi2 <- log(beta2_init)
  
  # --- storage ---
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning)
  ns <- length(keep_idx)
  
  z_save <- array(NA_real_, dim = c(ns, n, d))
  # binary layer
  alpha1_save <- beta1_save <- xi1_save <- psi2_1_save <- kappa2_1_save <- numeric(ns)
  # conti layer
  alpha2_save <- beta2_save <- xi2_save <- psi2_2_save <- kappa2_2_save <- numeric(ns)
  # traces
  ll_comm_save <- ll_comm_dist_save <- ll_joint_save <- numeric(ns)
  
  # acceptances
  acc_z <- integer(n)
  acc_alpha1 <- 0L; acc_beta1 <- 0L; acc_kappa2_1 <- 0L
  acc_alpha2 <- 0L; acc_beta2 <- 0L; acc_kappa2_2 <- 0L
  
  s <- 0L
  
  # --- sampler ---
  for (it in 1:iter) {
    # (1) z single-site MH (joint)
    for (j in 1:n) {
      up <- lsjm_update_z_single(j, z, Y_comm, Y_comm_dist,
                                 alpha1, beta1, kappa2_1,
                                 alpha2, beta2, kappa2_2,
                                 sigma2, prop_z, d)
      z <- up$z
      if (up$accepted) acc_z[j] <- acc_z[j] + 1L
    }
    

    # (2) Continuous layer params (comm)
    up_a1 <- con_update_alpha_comm(alpha1, z, Y_comm, xi1, psi2_1, prop_alpha_1, beta1, kappa2_1)
    alpha1 <- up_a1$alpha; if (up_a1$accepted) acc_alpha1 <- acc_alpha1 + 1L
    
    # up_b1 <- con_update_beta_comm(beta1, z, Y_comm, alpha1, prop_beta_1, a0_1, b0_1, kappa2_1)
    # beta1 <- up_b1$beta;   if (up_b1$accepted) acc_beta1 <- acc_beta1 + 1L
    
    up_k2_1 <- con_update_kappa_comm(kappa2_1, z, Y_comm, alpha1, beta1, prop_kappa_1, m_kappa_1, s_kappa_1)
    kappa2_1 <- up_k2_1$kappa; if (up_k2_1$accepted) acc_kappa2_1 <- acc_kappa2_1 + 1L
    
    up_xi1psi2_1 <- con_update_xi_psi2_comm(alpha1, xi1, psi2_1, m0 = m0_1, kappa0 = kappa0_1,
                                     alpha0 = alpha0_1, beta0 = beta0_1)
    xi1 <- up_xi1psi2_1$xi; psi2_1 <- up_xi1psi2_1$psi2
    
    # (3) Continuous layer params (comm_dist)
    up_a2 <- con_update_alpha_comm_dist(alpha2, z, Y_comm_dist, xi2, psi2_2, prop_alpha_2, beta2, kappa2_2)
    alpha2 <- up_a2$alpha; if (up_a2$accepted) acc_alpha2 <- acc_alpha2 + 1L
    
    # up_b2 <- con_update_beta_comm_dist(beta2, z, Y_comm_dist, alpha2, prop_beta_2, a0_2, b0_2, kappa2_2)
    # beta2 <- up_b2$beta;   if (up_b2$accepted) acc_beta2 <- acc_beta2 + 1L
    
    up_k2_2 <- con_update_kappa_comm_dist(kappa2_2, z, Y_comm_dist, alpha2, beta2, prop_kappa_2, m_kappa_2, s_kappa_2)
    kappa2_2 <- up_k2_2$kappa; if (up_k2_2$accepted) acc_kappa2_2 <- acc_kappa2_2 + 1L
    
    up_xi2psi2_2 <- con_update_xi_psi2_comm_dist(alpha2, xi2, psi2_2, m0 = m0_2, kappa0 = kappa0_2,
                                     alpha0 = alpha0_2, beta0 = beta0_2)
    xi2 <- up_xi2psi2_2$xi; psi2_2 <- up_xi2psi2_2$psi2
    
    # test: lognormal(beta) = phi sampling
    up_phi1 <- update_phi_beta(phi1, z, Y_comm, alpha1, loglik_conti_network_comm,
                               prop_phi = prop_beta_1, mu_phi = 0, s2_phi = 1^2,
                               kappa = kappa2_1)
    phi1 <- up_phi1$phi; beta1 <- exp(phi1); if (up_phi1$accepted) acc_beta1 <- acc_beta1 + 1L
    
    up_phi2 <- update_phi_beta(phi2, z, Y_comm_dist, alpha2, loglik_conti_network_comm_dist,
                               prop_phi = prop_beta_2, mu_phi = 0, s2_phi = 1^2,
                               kappa = kappa2_2)
    phi2 <- up_phi2$phi; beta2 <- exp(phi2); if (up_phi2$accepted) acc_beta2 <- acc_beta2 + 1L
    
    
    # (4) save
    if (it %in% keep_idx) {
      s <- s + 1L
      z_save[s, , ] <- z
      
      alpha1_save[s] <- alpha1; beta1_save[s] <- beta1; xi1_save[s] <- xi1; psi2_1_save[s] <- psi2_1; kappa2_1_save[s] <- kappa2_1
      alpha2_save[s] <- alpha2; beta2_save[s] <- beta2; xi2_save[s] <- xi2; psi2_2_save[s] <- psi2_2; kappa2_2_save[s] <- kappa2_2
      
      ll_comm <- loglik_conti_network_comm(Y_comm, z, alpha1, beta1, kappa2_1)
      ll_comm_dist <- loglik_conti_network_comm_dist(Y_comm_dist, z, alpha2, beta2, kappa2_2)
      ll_comm_save[s] <- ll_comm
      ll_comm_dist_save[s] <- ll_comm_dist
      ll_joint_save[s] <- ll_comm + ll_comm_dist - 0.5 / sigma2 * sum(z^2)  # joint up to const
    }
    
    if (verbose && it %% 100 == 0) {
      cat(sprintf("[iter %d] a1=%.3f b1=%.3f k1=%.3f | a2=%.3f b2=%.3f k2=%.3f | ll_bin=%.1f ll_con=%.1f\n",
                  it, alpha1, beta1, kappa2_1, alpha2, beta2, kappa2_2,
                  loglik_conti_network_comm(Y_comm, z, alpha1, beta1, kappa2_1),
                  loglik_conti_network_comm_dist(Y_comm_dist, z, alpha2, beta2, kappa2_2)))
    }
  }
  
  # --- Procrustes alignment to MAP (by joint ll as proxy) ---
  # vegan가 설치되어 있으면 정렬, 아니면 pass
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
      # comm
      alpha1 = alpha1_save, beta1 = beta1_save, kappa2_1 = kappa2_1_save, xi1 = xi1_save, psi2_1 = psi2_1_save,
      # comm_dist
      alpha2 = alpha2_save, beta2 = beta2_save, kappa2_2 = kappa2_2_save, xi2 = xi2_save, psi2_2 = psi2_2_save,
      # traces
      ll_comm = ll_comm_save, ll_comm_dist = ll_comm_dist_save, ll_joint = ll_joint_save
    ),
    accept = list(
      z = acc_z / iter,
      alpha1 = acc_alpha1 / iter, beta1 = acc_beta1 / iter, kappa2_1 = acc_kappa2_1 / iter,
      alpha2 = acc_alpha2 / iter, beta2 = acc_beta2 / iter, kappa2_2 = acc_kappa2_2 / iter
    ),
    config = list(
      iter = iter, burnin = burnin, thinning = thinning,
      d = d, sigma2 = sigma2,
      prop_sd = list(z = prop_z,
                     alpha_1 = prop_alpha_1, beta_1 = prop_beta_1, kappa_1 = prop_kappa_1,
                     alpha_2 = prop_alpha_2, beta_2 = prop_beta_2, kappa_2 = prop_kappa_2),
      priors = list(
        comm = list(a0 = a0_1, b0 = b0_1, m_kappa_1 = m_kappa_1, s_kappa_1 = s_kappa_1,
                      alpha0_1 = alpha0_1, beta0_1 = beta0_1),
        comm_dist  = list(a0 = a0_2, b0 = b0_2, m_kappa_2 = m_kappa_2, s_kappa_2 = s_kappa_2,
                      alpha0_2 = alpha0_2, beta0_2 = beta0_2)
      )
    )
  )
}

temp_comm <- comm
diag(temp_comm) <- 0
plot(log(as.numeric(temp_comm)), as.numeric(comm_dist))
cov(log(as.numeric(temp_comm)+ 1e-8), as.numeric(comm_dist))

fit <- LSJM_2layer(Y_comm = katz_mat,
                   Y_comm_dist = comm_dist,
                   iter = 5000, burnin = 1000, thinning = 1,
                   d = 2, sigma2 = 1,
                   # proposals
                   prop_z = 0.05,
                   prop_alpha_1 = 0.5, prop_beta_1 = 0.5, prop_kappa_1 = 0.1,
                   prop_alpha_2 = 0.05, prop_beta_2 = 0.01, prop_kappa_2 = 0.1,
                   verbose = TRUE)

# comm 의 경우: prop_z = 0.03, prop_alpha = 0.05, prop_beta = 0.03, prop_kappa = 0.1
# comm_dist 의 경우: prop_z = 0.01, prop_alpha = 0.05, prop_beta = 0.03, prop_kappa = 0.1
# 기본적으로 likelihood 가 comm_dist 가 더 높게 나옴. -> comm_dist 만 돌렸을 때 형태가 그대로 나옴
# comm_dist 를 이길 만 한 정보를 주는 Proximity 가 머가 있을까 - clustering 경향을 나타낼 수 있어야 함.
# comm_dist 가 degree 불균등성에 대해서 penalty 를 주기 때문에 clustering 구조만 좀 더 잘 나타내면 돼.
# 

fit$accept
par(mfrow = c(4,2))

for(i in 1:dim(fit$samples$z)[2]){
  for(j in 1:2){
    ts.plot(fit$samples$z[,i,j])
  }
}

par(mfrow = c(2,1))
ts.plot(fit$samples$alpha1)
ts.plot(fit$samples$alpha2)
# beta1,2 가 수렴을 잘 안 한다
ts.plot(fit$samples$beta1)
ts.plot(fit$samples$beta2)

ts.plot(fit$samples$xi1)
ts.plot(fit$samples$xi2)
ts.plot(fit$samples$psi2_1)
ts.plot(fit$samples$psi2_2)
ts.plot(fit$samples$ll_comm)
ts.plot(fit$samples$ll_comm_dist)

par(mfrow = c(1,1))
ts.plot(fit$samples$ll_joint)

latentmap <- colMeans(fit$samples$z, dims = 1)
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

# motif 복원 테스트
motif_GOF_test(fit, Y, B = 500, custom_text = 'LSJM with karate network', arg = 'simple_motif', model = 'LSJM')
motif_GOF_test(fit, Y, B = 500, custom_text = 'LSJM with karate network', arg = 'kstars', model = 'LSJM')
motif_GOF_test(fit, Y, B = 500, custom_text = 'LSJM with karate network', arg = 'dsp', model = 'LSJM')
motif_GOF_test(fit, Y, B = 500, custom_text = 'LSJM with karate network', arg = 'esp', model = 'LSJM')
plot(g)
deg <- degree(g)  # 각 노드의 degree
