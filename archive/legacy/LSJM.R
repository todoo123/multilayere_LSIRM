## ===== LSJM (Binary + Continuous distance) : Base R MCMC with IG prior on delta^2 =====
rm(list = ls())
## -------- Utilities --------
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
  return(D)
}


loglik_binary_layer <- function(Y1, z, alpha1) {
  D <- L2_dist(z)
  eta <- alpha1 - D
  pi <- plogis(eta)
  n <- nrow(Y1)
  ll <- 0.0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      y <- Y1[i,j]
      p <- pi[i,j]
      ll <- ll + if (y == 1) log(p + 1e-12) else log1p(-p + 1e-12)
    }
  }
  return(ll)
}

loglik_cont_layer <- function(Y2, z, alpha2, delta2) {
  D <- L2_dist(z)
  # 여기서는 거리 효과가 + 로 들어감 - communicabiltiy distance 가 클수록 연결이 약화되어야 하기 때문이다. 
  # edge 연결 효과와는 반대로 추정해야 한다.
  mu <- alpha2 + D
  n <- nrow(Y2)
  ll <- 0.0
  const <- -0.5 * log(2*pi*delta2)
  inv2 <- 1.0 / (2*delta2)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      r <- Y2[i,j] - mu[i,j]
      ll <- ll + const - inv2 * r*r
    }
  }
  return(ll)
}

log_prior_z_i <- function(z_i, sigma2) {
  return(- sum(z_i^2) / (2*sigma2) - log(2*pi*sigma2))
}

log_prior_alpha <- function(alpha, xi, psi2) {
  return(-0.5*log(2*pi*psi2) - 0.5*(alpha - xi)^2/psi2)
}

## -- local conditional for z_i
log_cond_z_i <- function(i, z, Y1, Y2, alpha1, alpha2, delta2, sigma2) {
  n <- nrow(z)
  zi <- z[i,]
  ll1 <- 0.0
  ll2 <- 0.0
  for (j in 1:n) {
    if (j == i) next
    d <- sqrt(sum((zi - z[j,])^2))
    # Layer 1
    y1 <- Y1[min(i,j), max(i,j)]
    p  <- plogis(alpha1 - d)
    if (y1 == 1) ll1 <- ll1 + log(p + 1e-12) else ll1 <- ll1 + log1p(-p + 1e-12)
    # Layer 2
    y2 <- Y2[min(i,j), max(i,j)]
    # distance 효과 역으로 추정
    r  <- y2 - (alpha2 + d)
    ll2 <- ll2 + (-0.5*log(2*pi*delta2) - (r*r)/(2*delta2))
  }
  lp <- log_prior_z_i(zi, sigma2)
  ll1 + ll2 + lp
}

## -------- Single-site MH updates --------
update_z_i <- function(i, z, Y1, Y2, alpha1, alpha2, delta2, sigma2, prop_sd, d) {
  current_logpost <- log_cond_z_i(i, z, Y1, Y2, alpha1, alpha2, delta2, sigma2)
  zi_prop <- z[i,] + rnorm(d, 0, prop_sd)
  z_prop <- z
  z_prop[i,] <- zi_prop
  prop_logpost <- log_cond_z_i(i, z_prop, Y1, Y2, alpha1, alpha2, delta2, sigma2)
  acc <- prop_logpost - current_logpost
  if (log(runif(1)) < acc) list(z = z_prop, accepted = TRUE)
  else list(z = z, accepted = FALSE)
}

update_alpha1 <- function(alpha1, z, Y1, xi1, psi1_2, prop_sd) {
  ll_curr <- loglik_binary_layer(Y1, z, alpha1) + log_prior_alpha(alpha1, xi1, psi1_2)
  a_prop <- alpha1 + rnorm(1, 0, prop_sd)
  ll_prop <- loglik_binary_layer(Y1, z, a_prop) + log_prior_alpha(a_prop, xi1, psi1_2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) list(alpha = a_prop, accepted = TRUE)
  else list(alpha = alpha1, accepted = FALSE)
}

update_alpha2 <- function(alpha2, z, Y2, delta2, xi2, psi2_2, prop_sd) {
  ll_curr <- loglik_cont_layer(Y2, z, alpha2, delta2) + log_prior_alpha(alpha2, xi2, psi2_2)
  a_prop <- alpha2 + rnorm(1, 0, prop_sd)
  ll_prop <- loglik_cont_layer(Y2, z, a_prop, delta2) + log_prior_alpha(a_prop, xi2, psi2_2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) list(alpha = a_prop, accepted = TRUE)
  else list(alpha = alpha2, accepted = FALSE)
}

## -------- Conjugate hyper updates (Gibbs) --------
# Priors: xi_k ~ N(0, 1),  psi_k^2 ~ IG(a_psi, b_psi)
sample_xi_given_alpha_psi <- function(alpha, psi2) {
  v <- 1.0 / (1.0 + 1.0/psi2)
  m <- v * (alpha/psi2)
  rnorm(1, mean = m, sd = sqrt(v))
}

sample_psi2_given_alpha_xi <- function(alpha, xi, a_psi, b_psi) {
  shape <- a_psi + 0.5
  rate  <- b_psi + 0.5*(alpha - xi)^2
  x <- rgamma(1, shape = shape, rate = rate)  # Gamma(shape, rate)
  1.0 / x                                      # -> IG(shape, rate)
}

## -------- NEW: Gibbs update for delta^2 with IG prior --------
# Prior: delta2 ~ IG(a_delta, b_delta)
sample_delta2_given <- function(Y2, z, alpha2, a_delta, b_delta) {
  D <- L2_dist(z)
  mu <- alpha2 - D
  n <- nrow(Y2)
  rss <- 0.0
  for (i in 1:(n-1)) for (j in (i+1):n) {
    r <- Y2[i,j] - mu[i,j]
    rss <- rss + r*r
  }
  m <- n*(n-1)/2
  shape <- a_delta + 0.5*m
  rate  <- b_delta + 0.5*rss
  x <- rgamma(1, shape = shape, rate = rate)  # Gamma
  1.0 / x                                      # -> IG
}

## -------- Driver --------
lsjm_mcmc <- function(Y1, Y2,
                      n_iter = 5000, burn = 1000, thin = 10, d = 2,
                      # latent prior
                      sigma2 = 1.0,
                      # delta^2 prior (Inverse-Gamma)
                      a_delta = 2.0, b_delta = 2.0,
                      # RW proposal sds
                      prop_sd_z = 0.05, prop_sd_a1 = 0.05, prop_sd_a2 = 0.05,
                      # hyperpriors for psi_k^2
                      a_psi = 2.0, b_psi = 2.0,
                      # initial values
                      z_init = NULL, alpha1_init = 0.0, alpha2_init = 0.0,
                      delta2_init = 0.1,
                      xi1_init = 0.0, xi2_init = 0.0,
                      psi1_2_init = 1.0, psi2_2_init = 1.0,
                      verbose = TRUE) {
  
  stopifnot(is.matrix(Y1), is.matrix(Y2))
  n <- nrow(Y1)
  stopifnot(ncol(Y1) == n, nrow(Y2) == n, ncol(Y2) == n)
  if (max(abs(Y1 - t(Y1))) > 1e-8) stop("Y1 must be symmetric.")
  if (max(abs(Y2 - t(Y2))) > 1e-8) stop("Y2 must be symmetric.")
  
  # init
  if (is.null(z_init)) {
    z <- matrix(rnorm(n*d, 0, sqrt(sigma2)), n, d)
  } else {
    stopifnot(is.matrix(z_init) && nrow(z_init) == n && ncol(z_init) == d)
    z <- z_init
  }
  alpha1 <- alpha1_init
  alpha2 <- alpha2_init
  delta2 <- delta2_init
  
  xi1 <- xi1_init
  xi2 <- xi2_init
  psi1_2 <- psi1_2_init
  psi2_2 <- psi2_2_init
  
  # storage
  keep_idx <- seq(from = burn + 1, to = n_iter, by = thin)
  ns <- length(keep_idx)
  z_save <- array(NA_real_, dim = c(ns, n, d))
  a1_save <- a2_save <- xi1_save <- xi2_save <- psi1_save <- psi2_save <- delta2_save <- numeric(ns)
  ll1_save <- ll2_save <- numeric(ns)
  
  acc_z <- rep(0L, n)
  acc_a1 <- acc_a2 <- 0L
  
  s <- 0L
  for (it in 1:n_iter) {
    # --- update z_i
    for (i in 1:n) {
      up <- update_z_i(i, z, Y1, Y2, alpha1, alpha2, delta2, sigma2, prop_sd_z, d)
      z <- up$z
      if (up$accepted) acc_z[i] <- acc_z[i] + 1L
    }
    
    # --- update alpha1, alpha2 (MH)
    up1 <- update_alpha1(alpha1, z, Y1, xi1, psi1_2, prop_sd_a1)
    alpha1 <- up1$alpha; if (up1$accepted) acc_a1 <- acc_a1 + 1L
    
    up2 <- update_alpha2(alpha2, z, Y2, delta2, xi2, psi2_2, prop_sd_a2)
    alpha2 <- up2$alpha; if (up2$accepted) acc_a2 <- acc_a2 + 1L
    
    # --- Gibbs for (xi_k, psi_k^2)
    xi1  <- sample_xi_given_alpha_psi(alpha1, psi1_2)
    xi2  <- sample_xi_given_alpha_psi(alpha2, psi2_2)
    psi1_2 <- sample_psi2_given_alpha_xi(alpha1, xi1, a_psi, b_psi)
    psi2_2 <- sample_psi2_given_alpha_xi(alpha2, xi2, a_psi, b_psi)
    
    # --- NEW: Gibbs for delta^2 (IG prior)
    delta2 <- sample_delta2_given(Y2, z, alpha2, a_delta, b_delta)
    
    # --- save
    if (it %in% keep_idx) {
      s <- s + 1L
      z_save[s,,] <- z
      a1_save[s]  <- alpha1
      a2_save[s]  <- alpha2
      xi1_save[s] <- xi1
      xi2_save[s] <- xi2
      psi1_save[s] <- psi1_2
      psi2_save[s] <- psi2_2
      delta2_save[s] <- delta2
      ll1_save[s] <- loglik_binary_layer(Y1, z, alpha1)
      ll2_save[s] <- loglik_cont_layer(Y2, z, alpha2, delta2)
    }
    
    if (verbose && (it %% 100 == 0)) {
      print(paste0('iteration: ', it, ' alpha1: ', alpha1, ' alpha2: ', alpha2, ' delta2: ', delta2,
             ' likelihood 1: ', loglik_binary_layer(Y1, z, alpha1), 
             ' likelihood 2: ', loglik_cont_layer(Y2, z, alpha2, delta2)))
    }
  }
  
  list(
    samples = list(
      z = z_save,
      alpha1 = a1_save, alpha2 = a2_save,
      xi1 = xi1_save, xi2 = xi2_save,
      psi1_2 = psi1_save, psi2_2 = psi2_save,
      delta2 = delta2_save,
      ll1 = ll1_save, ll2 = ll2_save
    ),
    accept = list(
      z = acc_z / n_iter,
      alpha1 = acc_a1 / n_iter,
      alpha2 = acc_a2 / n_iter
    ),
    config = list(
      n_iter = n_iter, burn = burn, thin = thin,
      sigma2 = sigma2,
      a_delta = a_delta, b_delta = b_delta,
      prop_sd = list(z = prop_sd_z, a1 = prop_sd_a1, a2 = prop_sd_a2),
      a_psi = a_psi, b_psi = b_psi
    )
  )
}

# ============================================================================
# practice from network of 'test_function.R' file
# ============================================================================
fit <- list()
for(i in 1:length(nets)){
  fit[[i]] <- lsjm_mcmc(as.matrix(as_adjacency_matrix(intergraph::asIgraph(nets[[i]]))),
                   nets[[i]]$comm_dist.scaled,
                   n_iter = 4000, burn = 2000, thin = 10,
                   sigma2 = 1.0,
                   a_delta = 2, b_delta = 2,     # IG prior for delta^2
                   prop_sd_z = 0.1, prop_sd_a1 = 0.05, prop_sd_a2 = 0.03,
                   a_psi = 2, b_psi = 2,
                   verbose = TRUE)
}

for(i in 1:length(nets)){
  motif_GOF_test(fit[[i]],
                 as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
                 B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
                 arg = 'simple_motif', model = 'LSJM')
  
  motif_GOF_test(fit[[i]],
                 as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
                 B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
                 arg = 'kstars', model = 'LSJM')
  
  motif_GOF_test(fit[[i]], 
                 as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
                 B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'),
                 arg = 'esp', model = 'LSJM')
  
  motif_GOF_test(fit[[i]], 
                 as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
                 B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'),
                 arg = 'dsp', model = 'LSJM')
  
  motif_GOF_test(fit[[i]], 
                 as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
                 B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
                 arg = 'closure_rate', model = 'LSJM')
  
}

i<-20

fit[[i]] <- lsjm_mcmc(as.matrix(as_adjacency_matrix(intergraph::asIgraph(nets[[i]]))),
                      nets[[i]]$comm_dist.scaled,
                      n_iter = 4000, burn = 2000, thin = 10,
                      sigma2 = 1.0,
                      a_delta = 2, b_delta = 2,     # IG prior for delta^2
                      prop_sd_z = 0.1, prop_sd_a1 = 0.05, prop_sd_a2 = 0.03,
                      a_psi = 2, b_psi = 2,
                      verbose = TRUE)

motif_GOF_test(fit[[i]],
               as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
               B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
               arg = 'simple_motif', model = 'LSJM')

motif_GOF_test(fit[[i]],
               as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
               B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
               arg = 'kstars', model = 'LSJM')

motif_GOF_test(fit[[i]], 
               as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
               B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'),
               arg = 'esp', model = 'LSJM')

motif_GOF_test(fit[[i]], 
               as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
               B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'),
               arg = 'dsp', model = 'LSJM')

motif_GOF_test(fit[[i]], 
               as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
               B = 500, custom_text = paste0('decay value: ',decay_values[i], ' LSJM'), 
               arg = 'closure_rate', model = 'LSJM')

