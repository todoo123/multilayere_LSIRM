rm(list = ls())
################################################################################
# LSJM (Binary + Communicability + Communicability-Distance)
# - shared latent positions z (R^d)
# - shared prior variance sigma2 for each coordinate of z_i
# - Binary: logistic; Continuous: Lognormal with E[Y]=exp(eta)
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
      d <- sqrt(sum((z[i,] - z[j,])^2))
      D[i,j] <- d
      D[j,i] <- d
    }
  }
  D
}

## ---------- (1) loglike, prior ----------
# Binary (logistic)
loglik_binary_network <- function(Y, z, alpha, beta) {
  D <- L2_dist(z)
  eta <- alpha - beta*D
  p   <- stats::plogis(eta)
  n <- nrow(Y)
  ll <- 0.0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      y <- Y[i,j]
      pij <- p[i,j]
      ll <- ll + if (y==1) log(pij + 1e-11) else log(1 - pij + 1e-11)
    }
  }
  ll
}

# Continuous (lognormal): E[Y] = exp(eta)
# Comm (similarity): eta = alpha - beta*D
loglik_conti_comm <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  D <- L2_dist(z)
  eta <- alpha - beta*D
  up  <- which(upper.tri(Y), arr.ind=TRUE)
  y   <- pmax(Y[up], eps)
  mu_log <- eta[up] - kappa/2
  sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
}

# Comm_dist (distance): eta = alpha + beta*D
loglik_conti_comm_dist <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  D <- L2_dist(z)
  eta <- alpha + beta*D
  up  <- which(upper.tri(Y), arr.ind=TRUE)
  y   <- pmax(Y[up], eps)
  mu_log <- eta[up] - kappa/2
  sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
}

# Priors
log_prior_z_i <- function(z_i, sigma2) {
  sum(dnorm(z_i, mean = 0, sd = sqrt(sigma2), log = TRUE))
}
log_prior_alpha <- function(alpha, xi, psi2) {
  dnorm(alpha, mean = xi, sd = sqrt(psi2), log = TRUE)
}
log_prior_beta_gamma <- function(beta, a, b) {
  dgamma(beta, shape = a, rate = b, log = TRUE)
}
# log(kappa) ~ N(m, s2)  =>  p(kappa) = (1/kappa) N(log kappa | m, s2)
log_prior_kappa <- function(kappa, m, s2) {
  dnorm(log(kappa), mean = m, sd = sqrt(s2), log = TRUE) - log(kappa)
}

## ---------- (2) generic log-scale MH updaters ----------
# For beta (log-normal proposal, normal prior on phi=log beta)
update_phi_beta <- function(phi, z, Y, alpha, loglik_fn, prop_phi,
                            mu_phi = 0, s2_phi = 1, extra_args = list()) {
  beta     <- exp(phi)
  curr_ll  <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta), extra_args))
  curr     <- curr_ll + dnorm(phi, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  phi_prop <- phi + rnorm(1, 0, prop_phi)
  beta_p   <- exp(phi_prop)
  prop_ll  <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta_p), extra_args))
  prop     <- prop_ll + dnorm(phi_prop, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  if (log(runif(1)) < (prop - curr)) list(phi=phi_prop, accepted=TRUE) else list(phi=phi, accepted=FALSE)
}

# For kappa (phi_k = log kappa)
update_phi_kappa <- function(phi_k, z, Y, alpha, beta, loglik_fn, prop_phi,
                             mu_phi = 0, s2_phi = 1) {
  kappa    <- exp(phi_k)
  curr_ll  <- loglik_fn(Y=Y, z=z, alpha=alpha, beta=beta, kappa=kappa)
  curr     <- curr_ll + dnorm(phi_k, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)  # prior on phi
  phi_prop <- phi_k + rnorm(1, 0, prop_phi)
  kappa_p  <- exp(phi_prop)
  prop_ll  <- loglik_fn(Y=Y, z=z, alpha=alpha, beta=beta, kappa=kappa_p)
  prop     <- prop_ll + dnorm(phi_prop, mean=mu_phi, sd=sqrt(s2_phi), log=TRUE)
  if (log(runif(1)) < (prop - curr)) list(phi=phi_prop, accepted=TRUE) else list(phi=phi_k, accepted=FALSE)
}

## ---------- (3) one-step MH for scalar alphas (common for all layers) ----------
update_alpha_scalar <- function(alpha, z, Y, xi, psi2, prop_alpha, beta, loglik_fn, extra_args=list()) {
  curr <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=alpha, beta=beta), extra_args)) +
    log_prior_alpha(alpha, xi, psi2)
  a_prop <- alpha + rnorm(1, 0, prop_alpha)
  prop  <- do.call(loglik_fn, c(list(Y=Y, z=z, alpha=a_prop, beta=beta), extra_args)) +
    log_prior_alpha(a_prop, xi, psi2)
  if (log(runif(1)) < (prop - curr)) list(alpha=a_prop, accepted=TRUE) else list(alpha=alpha, accepted=FALSE)
}

## ---------- (4) Normal–InvGamma hyper updates for (xi, psi2) ----------
# alpha | xi, psi2 ~ Normal(xi, psi2); xi ~ Normal(m0, kappa0); psi2 ~ InvGamma(alpha0, beta0)
gibbs_xi_psi2 <- function(alpha, xi, psi2, m0=0, kappa0=1/2, alpha0=3, beta0=1/2) {
  # xi | psi2, alpha
  kappa_n <- (1/psi2 + 1/kappa0)^(-1)
  m_n     <- kappa_n * (m0/kappa0 + alpha/psi2)
  xi_new  <- rnorm(1, mean = m_n, sd = sqrt(kappa_n))
  # psi2 | xi, alpha
  shape <- alpha0 + 1/2
  scale <- beta0 + 0.5*(alpha - xi_new)^2
  psi2_new <- 1 / rgamma(1, shape = shape, rate = scale)
  list(xi=xi_new, psi2=psi2_new)
}

## ---------- (5) single-site z update (joint over 3 layers) ----------
lsjm_update_z_single <- function(j, z,
                                 Y_bin,   alpha_b, beta_b,
                                 Y_comm,  alpha_c, beta_c, kappa_c,
                                 Y_cdist, alpha_d, beta_d, kappa_d,
                                 sigma2, prop_z, d) {
  zi_old <- z[j, ]
  z_prop <- z
  zi_prop <- zi_old + rnorm(d, 0, prop_z)
  z_prop[j, ] <- zi_prop
  
  # full likelihoods + prior contribution of z_j
  ll_old <- loglik_binary_network(Y_bin, z,      alpha_b, beta_b) +
    loglik_conti_comm     (Y_comm, z,    alpha_c, beta_c, kappa_c) +
    loglik_conti_comm_dist(Y_cdist, z,   alpha_d, beta_d, kappa_d) +
    log_prior_z_i(zi_old, sigma2)
  ll_new <- loglik_binary_network(Y_bin, z_prop, alpha_b, beta_b) +
    loglik_conti_comm     (Y_comm, z_prop,    alpha_c, beta_c, kappa_c) +
    loglik_conti_comm_dist(Y_cdist, z_prop,   alpha_d, beta_d, kappa_d) +
    log_prior_z_i(zi_prop, sigma2)
  
  if (log(runif(1)) < (ll_new - ll_old)) list(z=z_prop, accepted=TRUE) else list(z=z, accepted=FALSE)
}

## ---------- (6) MAIN DRIVER ----------
LSJM_3layer <- function(Y_bin, Y_comm, Y_comm_dist,
                        iter = 5000, burnin = 1000, thinning = 5,
                        d = 2,
                        # shared z prior variance
                        sigma2 = 1.0,
                        # Binary priors/hypers
                        a0_bin = 0.5, b0_bin = 0.5,              # (optional) if gamma prior is used instead of phi
                        m0_bin = 0, kappa0_bin = 1/2, alpha0_bin = 3, beta0_bin = 1/2,
                        # Comm priors/hypers
                        m_phi_beta_c = 0, s2_phi_beta_c = 1,     # log-beta prior for comm
                        m_phi_kap_c  = 0, s2_phi_kap_c  = 1,     # log-kappa prior for comm
                        m0_c = 0, kappa0_c = 1/2, alpha0_c = 3, beta0_c = 1/2,
                        # Comm_dist priors/hypers
                        m_phi_beta_d = 0, s2_phi_beta_d = 1,     # log-beta prior for comm_dist
                        m_phi_kap_d  = 0, s2_phi_kap_d  = 1,     # log-kappa prior for comm_dist
                        m0_d = 0, kappa0_d = 1/2, alpha0_d = 3, beta0_d = 1/2,
                        # proposals
                        prop_z = 0.1,
                        prop_alpha_bin = 0.05, prop_phi_beta_bin = 0.05,   # if using phi-beta for binary
                        prop_alpha_c = 0.05, prop_phi_beta_c = 0.05, prop_phi_kappa_c = 0.2,
                        prop_alpha_d = 0.05, prop_phi_beta_d = 0.05, prop_phi_kappa_d = 0.2,
                        # inits
                        z_init = NULL,
                        alpha_b_init = 0, beta_b_init = 1, xi_b_init = 0, psi2_b_init = 1,
                        alpha_c_init = 0, beta_c_init = 1, xi_c_init = 0, psi2_c_init = 1, kappa_c_init = 1,
                        alpha_d_init = 0, beta_d_init = 1, xi_d_init = 0, psi2_d_init = 1, kappa_d_init = 1,
                        verbose = TRUE, seed = 1) {
  
  set.seed(seed)
  stopifnot(is.matrix(Y_bin), is.matrix(Y_comm), is.matrix(Y_comm_dist))
  n <- nrow(Y_bin); stopifnot(ncol(Y_bin)==n, nrow(Y_comm)==n, ncol(Y_comm)==n, nrow(Y_comm_dist)==n, ncol(Y_comm_dist)==n)
  if (max(abs(Y_bin       - t(Y_bin)))       > 1e-8) stop("Y_bin must be symmetric.")
  if (max(abs(Y_comm      - t(Y_comm)))      > 1e-8) stop("Y_comm must be symmetric.")
  if (max(abs(Y_comm_dist - t(Y_comm_dist))) > 1e-8) stop("Y_comm_dist must be symmetric.")
  
  # init z
  if (is.null(z_init)) z <- matrix(rnorm(n*d, 0, sqrt(sigma2)), n, d) else {
    stopifnot(is.matrix(z_init), nrow(z_init)==n, ncol(z_init)==d); z <- z_init
  }
  
  # layer params
  alpha_b <- alpha_b_init; beta_b <- beta_b_init; xi_b <- xi_b_init; psi2_b <- psi2_b_init
  alpha_c <- alpha_c_init; beta_c <- beta_c_init; xi_c <- xi_c_init; psi2_c <- psi2_c_init; kappa_c <- kappa_c_init
  alpha_d <- alpha_d_init; beta_d <- beta_d_init; xi_d <- xi_d_init; psi2_d <- psi2_d_init; kappa_d <- kappa_d_init
  
  # log-scales
  phi_beta_b <- log(beta_b)
  phi_beta_c <- log(beta_c); phi_kap_c <- log(kappa_c)
  phi_beta_d <- log(beta_d); phi_kap_d <- log(kappa_d)
  
  # storage
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning); ns <- length(keep_idx)
  z_save <- array(NA_real_, dim = c(ns, n, d))
  alpha_b_save <- beta_b_save <- xi_b_save <- psi2_b_save <- numeric(ns)
  alpha_c_save <- beta_c_save <- xi_c_save <- psi2_c_save <- kappa_c_save <- numeric(ns)
  alpha_d_save <- beta_d_save <- xi_d_save <- psi2_d_save <- kappa_d_save <- numeric(ns)
  ll_bin_save <- ll_comm_save <- ll_cdist_save <- ll_joint_save <- numeric(ns)
  
  # accept counters
  acc_z <- integer(n)
  acc_alpha_b <- acc_phi_beta_b <- 0L
  acc_alpha_c <- acc_phi_beta_c <- acc_phi_kap_c <- 0L
  acc_alpha_d <- acc_phi_beta_d <- acc_phi_kap_d <- 0L
  
  s <- 0L
  
  for (it in 1:iter) {
    # (1) z updates
    for (j in 1:n) {
      up <- lsjm_update_z_single(j, z,
                                 Y_bin,   alpha_b, beta_b,
                                 Y_comm,  alpha_c, beta_c, kappa_c,
                                 Y_comm_dist, alpha_d, beta_d, kappa_d,
                                 sigma2, prop_z, d)
      z <- up$z
      if (up$accepted) acc_z[j] <- acc_z[j] + 1L
    }
    
    # (2) Binary layer
    up_ab <- update_alpha_scalar(alpha_b, z, Y_bin, xi_b, psi2_b, prop_alpha_bin, beta_b,
                                 loglik_fn = loglik_binary_network)
    alpha_b <- up_ab$alpha; if (up_ab$accepted) acc_alpha_b <- acc_alpha_b + 1L
    
    up_phibb <- update_phi_beta(phi_beta_b, z, Y_bin, alpha_b, loglik_fn = loglik_binary_network,
                                prop_phi = prop_phi_beta_bin, mu_phi = 0, s2_phi = 1)
    phi_beta_b <- up_phibb$phi; beta_b <- exp(phi_beta_b); if (up_phibb$accepted) acc_phi_beta_b <- acc_phi_beta_b + 1L
    
    up_xi_psi_b <- gibbs_xi_psi2(alpha_b, xi_b, psi2_b, m0 = m0_bin, kappa0 = kappa0_bin,
                                 alpha0 = alpha0_bin, beta0 = beta0_bin)
    xi_b <- up_xi_psi_b$xi; psi2_b <- up_xi_psi_b$psi2
    
    # (3) Comm layer (continuous)
    up_ac <- update_alpha_scalar(alpha_c, z, Y_comm, xi_c, psi2_c, prop_alpha_c, beta_c,
                                 loglik_fn = loglik_conti_comm, extra_args = list(kappa = kappa_c))
    alpha_c <- up_ac$alpha; if (up_ac$accepted) acc_alpha_c <- acc_alpha_c + 1L
    
    up_phibc <- update_phi_beta(phi_beta_c, z, Y_comm, alpha_c, loglik_fn = loglik_conti_comm,
                                prop_phi = prop_phi_beta_c, mu_phi = m_phi_beta_c, s2_phi = s2_phi_beta_c,
                                extra_args = list(kappa = kappa_c))
    phi_beta_c <- up_phibc$phi; beta_c <- exp(phi_beta_c); if (up_phibc$accepted) acc_phi_beta_c <- acc_phi_beta_c + 1L
    
    up_phikc <- update_phi_kappa(phi_k = phi_kap_c, z = z, Y = Y_comm, alpha = alpha_c, beta = beta_c,
                                 loglik_fn = loglik_conti_comm, prop_phi = prop_phi_kappa_c,
                                 mu_phi = m_phi_kap_c, s2_phi = s2_phi_kap_c)
    phi_kap_c <- up_phikc$phi; kappa_c <- exp(phi_kap_c); if (up_phikc$accepted) acc_phi_kap_c <- acc_phi_kap_c + 1L
    
    up_xi_psi_c <- gibbs_xi_psi2(alpha_c, xi_c, psi2_c, m0 = m0_c, kappa0 = kappa0_c,
                                 alpha0 = alpha0_c, beta0 = beta0_c)
    xi_c <- up_xi_psi_c$xi; psi2_c <- up_xi_psi_c$psi2
    
    # (4) Comm_dist layer (continuous)
    up_ad <- update_alpha_scalar(alpha_d, z, Y_comm_dist, xi_d, psi2_d, prop_alpha_d, beta_d,
                                 loglik_fn = loglik_conti_comm_dist, extra_args = list(kappa = kappa_d))
    alpha_d <- up_ad$alpha; if (up_ad$accepted) acc_alpha_d <- acc_alpha_d + 1L
    
    up_phibd <- update_phi_beta(phi_beta_d, z, Y_comm_dist, alpha_d, loglik_fn = loglik_conti_comm_dist,
                                prop_phi = prop_phi_beta_d, mu_phi = m_phi_beta_d, s2_phi = s2_phi_beta_d,
                                extra_args = list(kappa = kappa_d))
    phi_beta_d <- up_phibd$phi; beta_d <- exp(phi_beta_d); if (up_phibd$accepted) acc_phi_beta_d <- acc_phi_beta_d + 1L
    
    up_phikd <- update_phi_kappa(phi_k = phi_kap_d, z = z, Y = Y_comm_dist, alpha = alpha_d, beta = beta_d,
                                 loglik_fn = loglik_conti_comm_dist, prop_phi = prop_phi_kappa_d,
                                 mu_phi = m_phi_kap_d, s2_phi = s2_phi_kap_d)
    phi_kap_d <- up_phikd$phi; kappa_d <- exp(phi_kap_d); if (up_phikd$accepted) acc_phi_kap_d <- acc_phi_kap_d + 1L
    
    up_xi_psi_d <- gibbs_xi_psi2(alpha_d, xi_d, psi2_d, m0 = m0_d, kappa0 = kappa0_d,
                                 alpha0 = alpha0_d, beta0 = beta0_d)
    xi_d <- up_xi_psi_d$xi; psi2_d <- up_xi_psi_d$psi2
    
    # (5) save
    if (it %in% keep_idx) {
      s <- s + 1L
      z_save[s,,] <- z
      alpha_b_save[s] <- alpha_b; beta_b_save[s] <- beta_b; xi_b_save[s] <- xi_b; psi2_b_save[s] <- psi2_b
      alpha_c_save[s] <- alpha_c; beta_c_save[s] <- beta_c; xi_c_save[s] <- xi_c; psi2_c_save[s] <- psi2_c; kappa_c_save[s] <- kappa_c
      alpha_d_save[s] <- alpha_d; beta_d_save[s] <- beta_d; xi_d_save[s] <- xi_d; psi2_d_save[s] <- psi2_d; kappa_d_save[s] <- kappa_d
      llb  <- loglik_binary_network    (Y_bin,       z, alpha_b, beta_b)
      llc  <- loglik_conti_comm        (Y_comm,      z, alpha_c, beta_c, kappa_c)
      lld  <- loglik_conti_comm_dist   (Y_comm_dist, z, alpha_d, beta_d, kappa_d)
      ll_bin_save[s]   <- llb
      ll_comm_save[s]  <- llc
      ll_cdist_save[s] <- lld
      ll_joint_save[s] <- llb + llc + lld - 0.5/sigma2 * sum(z^2)   # up to const
    }
    
    if (verbose && it %% 100 == 0) {
      cat(sprintf("[iter %d]  Bin: a=%.3f b=%.3f | Comm: a=%.3f b=%.3f k=%.3f | Cdist: a=%.3f b=%.3f k=%.3f | ll(b,c,d)=(%.1f, %.1f, %.1f)\n",
                  it, alpha_b, beta_b, alpha_c, beta_c, kappa_c, alpha_d, beta_d, kappa_d,
                  loglik_binary_network(Y_bin, z, alpha_b, beta_b),
                  loglik_conti_comm(Y_comm, z, alpha_c, beta_c, kappa_c),
                  loglik_conti_comm_dist(Y_comm_dist, z, alpha_d, beta_d, kappa_d)))
    }
  }
  
  # --- Procrustes alignment to MAP (by joint ll as proxy) ---
  map_idx <- which.max(ll_joint_save)
  Z_map <- matrix(z_save[map_idx,,], n, d)
  if (requireNamespace("vegan", quietly = TRUE)) {
    for (ss in seq_len(ns)) {
      if (ss == map_idx) next
      Z_s <- matrix(z_save[ss,,], n, d)
      fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
      z_save[ss,,] <- predict(fit)
    }
  }
  
  list(
    samples = list(
      z = z_save,
      # binary
      alpha_bin = alpha_b_save, beta_bin = beta_b_save, xi_bin = xi_b_save, psi2_bin = psi2_b_save,
      # comm
      alpha_comm = alpha_c_save, beta_comm = beta_c_save, kappa_comm = kappa_c_save, xi_comm = xi_c_save, psi2_comm = psi2_c_save,
      # comm_dist
      alpha_cdist = alpha_d_save, beta_cdist = beta_d_save, kappa_cdist = kappa_d_save, xi_cdist = xi_d_save, psi2_cdist = psi2_d_save,
      # traces
      ll_bin = ll_bin_save, ll_comm = ll_comm_save, ll_comm_dist = ll_cdist_save, ll_joint = ll_joint_save
    ),
    accept = list(
      z = acc_z / iter,
      alpha_bin = acc_alpha_b / iter, beta_bin = acc_phi_beta_b / iter,
      alpha_comm = acc_alpha_c / iter, beta_comm = acc_phi_beta_c / iter, kappa_comm = acc_phi_kap_c / iter,
      alpha_cdist = acc_alpha_d / iter, beta_cdist = acc_phi_beta_d / iter, kappa_cdist = acc_phi_kap_d / iter
    ),
    config = list(
      iter=iter, burnin=burnin, thinning=thinning, d=d, sigma2=sigma2,
      prop_sd = list(z=prop_z,
                     alpha_bin=prop_alpha_bin, phi_beta_bin=prop_phi_beta_bin,
                     alpha_comm=prop_alpha_c, phi_beta_comm=prop_phi_beta_c, phi_kappa_comm=prop_phi_kappa_c,
                     alpha_cdist=prop_alpha_d, phi_beta_cdist=prop_phi_beta_d, phi_kappa_cdist=prop_phi_kappa_d),
      priors = list(
        bin = list(m0=m0_bin, kappa0=kappa0_bin, alpha0=alpha0_bin, beta0=beta0_bin),
        comm = list(m_phi_beta=m_phi_beta_c, s2_phi_beta=s2_phi_beta_c, m_phi_kap=m_phi_kap_c, s2_phi_kap=s2_phi_kap_c,
                    m0=m0_c, kappa0=kappa0_c, alpha0=alpha0_c, beta0=beta0_c),
        comm_dist = list(m_phi_beta=m_phi_beta_d, s2_phi_beta=s2_phi_beta_d, m_phi_kap=m_phi_kap_d, s2_phi_kap=s2_phi_kap_d,
                         m0=m0_d, kappa0=kappa0_d, alpha0=alpha0_d, beta0=beta0_d)
      )
    )
  )
}

################################################################################
# 사용 예시 (Zachary로 간단 테스트; 실제론 각 레이어에 맞는 행렬 넣기)
################################################################################
library(igraph); g <- make_graph("Zachary")

fit <- LSJM_3layer(Y_bin = Y, Y_comm = comm, Y_comm_dist = comm_dist,
                   iter = 5000, burnin = 1000, thinning = 1, 
                   # Binary priors/hypers
                   a0_bin = 0.5, b0_bin = 0.5,              # (optional) if gamma prior is used instead of phi
                   m0_bin = 0, kappa0_bin = 1/2, alpha0_bin = 3, beta0_bin = 1/2,
                   # Comm priors/hypers
                   m_phi_beta_c = 5, s2_phi_beta_c = 1,     # log-beta prior for comm
                   m_phi_kap_c  = 0, s2_phi_kap_c  = 1,     # log-kappa prior for comm
                   m0_c = 0, kappa0_c = 1/2, alpha0_c = 3, beta0_c = 1/2,
                   # Comm_dist priors/hypers
                   m_phi_beta_d = 0, s2_phi_beta_d = 1,     # log-beta prior for comm_dist
                   m_phi_kap_d  = 0, s2_phi_kap_d  = 1,     # log-kappa prior for comm_dist
                   m0_d = 0, kappa0_d = 1/2, alpha0_d = 3, beta0_d = 1/2,
                   # proposals
                   prop_z = 0.05,
                   prop_alpha_bin = 0.2, prop_phi_beta_bin = 0.5,   # if using phi-beta for binary
                   prop_alpha_c = 0.2, prop_phi_beta_c = 0.5, prop_phi_kappa_c = 0.2,
                   prop_alpha_d = 0.03, prop_phi_beta_d = 0.03, prop_phi_kappa_d = 0.2,
                   verbose = TRUE, seed = 1)

fit$accept
par(mfrow = c(3,2))

for(i in 1:dim(fit$samples$z)[2]){
  for(j in 1:2){
    ts.plot(fit$samples$z[,i,j])
  }
}

par(mfrow = c(3,1))
# alpha
ts.plot(fit$samples$alpha_bin)
ts.plot(fit$samples$alpha_comm)
# alpha_cdist acceptance rate 낮음
ts.plot(fit$samples$alpha_cdist)
# beta
ts.plot(fit$samples$beta_bin)
ts.plot(fit$samples$beta_comm)
# communicability distance 의 beta 값이 갈피를 잡지 못하고 있음
# 그러나 값은 가장 높음
ts.plot(fit$samples$beta_cdist)
# xi
ts.plot(fit$samples$xi_bin)
ts.plot(fit$samples$xi_comm)
ts.plot(fit$samples$xi_cdist)
# psi
ts.plot(fit$samples$psi2_bin)
ts.plot(fit$samples$psi2_comm)
ts.plot(fit$samples$psi2_cdist)
# kappa
par(mfrow = c(2,1))
ts.plot(fit$samples$kappa_comm)
ts.plot(fit$samples$kappa_cdist)

# likelihood
par(mfrow = c(4,1))
ts.plot(fit$samples$ll_bin)
ts.plot(fit$samples$ll_comm)
ts.plot(fit$samples$ll_comm_dist)
ts.plot(fit$samples$ll_joint)


par(mfrow = c(1,1))
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


motif_GOF_test(fit, Y, B = 500, custom_text = paste0("3 layered LSJM"), arg = "simple_motif", model = 'LSJM_3')
