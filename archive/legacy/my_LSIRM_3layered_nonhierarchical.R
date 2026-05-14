rm(list = ls())
library(vegan)

# -------------------------------------------------------
# Shared-Position 3-layer LSIRM (0105_meeting version)
# - shared respondent position a_i across layers
# - shared respondent main effect alpha_i across layers
# - shared distance effect gamma across layers
# - layer-specific item effects beta^(l)_j and item positions b^(l)_j
# -------------------------------------------------------

lsirm_sharedpos_layer3 <- function(
    Y_bin,              # n x P1, 0/1
    Y_con,              # n x P2, continuous (likert ok)
    Y_cnt,              # n x P3, 0,1,2,... (count)
    d       = 2,
    n_iter  = 5000,
    burnin  = 1000,
    thin    = 5,
    hyper   = list(
      # alpha variance prior (shared)
      a_sigma = 1, b_sigma = 0.1,
      
      # beta variance priors (layer-specific)
      a_tau1  = 1, b_tau1  = 0.1,
      a_tau2  = 1, b_tau2  = 0.1,
      a_tau3  = 1, b_tau3  = 0.1,
      
      # residual variance prior (continuous layer)
      a_sigma0 = 1, b_sigma0 = 1,
      
      # log-gamma prior (shared gamma)
      mu_log_gamma = 0, sd_log_gamma = 1,
      
      # dispersion (count layer) prior: log kappa ~ N(mu, sd^2)
      mu_log_kappa = 0, sd_log_kappa = 1
    ),
    prop_sd = list(
      alpha      = 0.10,  # shared respondent main effect
      beta1      = 0.10,
      beta2      = 0.10,
      beta3      = 0.10,
      log_gamma  = 0.05,  # shared gamma
      log_kappa  = 0.05,  # NB dispersion
      a          = 0.10,  # shared respondent position
      b1         = 0.10,
      b2         = 0.10,
      b3         = 0.10
    ),
    init    = NULL,
    verbose = TRUE
) {
  
  # -------- helper: stable log(1+exp(x)) ----------
  log1pexp <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
  
  # -------- data ----------
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  Y_cnt <- as.matrix(Y_cnt)
  
  if (sum(is.na(Y_bin)) + sum(is.na(Y_con)) + sum(is.na(Y_cnt)) > 0) {
    stop("Missing values (NA) are not allowed.")
  }
  
  n  <- nrow(Y_bin)
  P1 <- ncol(Y_bin)
  P2 <- ncol(Y_con)
  P3 <- ncol(Y_cnt)
  
  Z_con <- Y_con  # 필요하면 변환을 여기서
  
  # -------- init ----------
  if (is.null(init)) {
    alpha <- rnorm(n, 0, 0.1)
    
    beta1 <- rnorm(P1, 0, 0.1)
    beta2 <- rnorm(P2, 0, 0.1)
    beta3 <- rnorm(P3, 0, 0.1)
    
    a  <- matrix(rnorm(n * d, 0, 0.5), n, d)
    b1 <- matrix(rnorm(P1 * d, 0, 0.5), P1, d)
    b2 <- matrix(rnorm(P2 * d, 0, 0.5), P2, d)
    b3 <- matrix(rnorm(P3 * d, 0, 0.5), P3, d)
    
    log_gamma <- 0
    gamma     <- exp(log_gamma)
    
    log_kappa <- 0
    kappa     <- exp(log_kappa)
    
    sigma_alpha_sq <- 1
    tau_beta1_sq   <- 1
    tau_beta2_sq   <- 1
    tau_beta3_sq   <- 1
    sigma0_sq      <- 1
  } else {
    alpha <- init$alpha
    beta1 <- init$beta1
    beta2 <- init$beta2
    beta3 <- init$beta3
    
    a  <- init$a
    b1 <- init$b1
    b2 <- init$b2
    b3 <- init$b3
    
    log_gamma <- init$log_gamma
    gamma     <- exp(log_gamma)
    
    log_kappa <- init$log_kappa
    kappa     <- exp(log_kappa)
    
    sigma_alpha_sq <- init$sigma_alpha_sq
    tau_beta1_sq   <- init$tau_beta1_sq
    tau_beta2_sq   <- init$tau_beta2_sq
    tau_beta3_sq   <- init$tau_beta3_sq
    sigma0_sq      <- init$sigma0_sq
  }
  
  # -------- storage (keep compatibility fields z,a1,a2,a3 = a) ----------
  n_save <- floor((n_iter - burnin) / thin)
  
  samples <- list(
    alpha = array(NA, c(n_save, n)),
    beta1 = array(NA, c(n_save, P1)),
    beta2 = array(NA, c(n_save, P2)),
    beta3 = array(NA, c(n_save, P3)),
    
    log_gamma = numeric(n_save),
    log_kappa = numeric(n_save),
    
    sigma_alpha_sq = numeric(n_save),
    tau_beta1_sq   = numeric(n_save),
    tau_beta2_sq   = numeric(n_save),
    tau_beta3_sq   = numeric(n_save),
    sigma0_sq      = numeric(n_save),
    
    # shared positions
    a  = array(NA, c(n_save, n, d)),
    b1 = array(NA, c(n_save, P1, d)),
    b2 = array(NA, c(n_save, P2, d)),
    b3 = array(NA, c(n_save, P3, d)),
    
    # compatibility outputs (same shape as old code)
    z  = array(NA, c(n_save, n, d)),
    a1 = array(NA, c(n_save, n, d)),
    a2 = array(NA, c(n_save, n, d)),
    a3 = array(NA, c(n_save, n, d)),
    
    loglik = numeric(n_save)
  )
  
  accept <- list(
    alpha     = rep(0, n),
    beta1     = rep(0, P1),
    beta2     = rep(0, P2),
    beta3     = rep(0, P3),
    log_gamma = 0,
    log_kappa = 0,
    a         = rep(0, n),
    b1        = rep(0, P1),
    b2        = rep(0, P2),
    b3        = rep(0, P3)
  )
  
  save_idx <- 0
  
  # precompute distance matrices (update locally during MH)
  dist_mat <- function(A, B) {
    # returns nrow(A) x nrow(B)
    nA <- nrow(A); nB <- nrow(B)
    D <- matrix(0, nA, nB)
    for (i in 1:nA) {
      for (j in 1:nB) {
        diff_ij <- A[i, ] - B[j, ]
        D[i, j] <- sqrt(sum(diff_ij^2))
      }
    }
    D
  }
  
  D1 <- dist_mat(a, b1)
  D2 <- dist_mat(a, b2)
  D3 <- dist_mat(a, b3)
  
  # -------- MCMC ----------
  for (iter in 1:n_iter) {
    
    if (verbose && iter %% 500 == 0) cat("Iter:", iter, "/", n_iter, "\n")
    
    # ==========================
    # 1) alpha_i (shared) : MH using all layers
    # ==========================
    for (i in 1:n) {
      # current
      # layer1 Bernoulli loglik
      eta1_i <- alpha[i] + beta1 - gamma * D1[i, ]
      ll1 <- sum(Y_bin[i, ] * eta1_i - log1pexp(eta1_i))
      
      # layer2 Normal loglik
      mu2_i <- alpha[i] + beta2 - gamma * D2[i, ]
      r2 <- Z_con[i, ] - mu2_i
      ll2 <- -0.5 / sigma0_sq * sum(r2^2)
      
      # layer3 NB loglik
      mu3_i <- exp(alpha[i] + beta3 - gamma * D3[i, ])
      ll3 <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_i, log = TRUE))
      
      lp_cur <- -0.5 * alpha[i]^2 / sigma_alpha_sq
      logpost_cur <- ll1 + ll2 + ll3 + lp_cur
      
      # proposal
      a_prop <- rnorm(1, alpha[i], prop_sd$alpha)
      
      eta1_p <- a_prop + beta1 - gamma * D1[i, ]
      ll1_p <- sum(Y_bin[i, ] * eta1_p - log1pexp(eta1_p))
      
      mu2_p <- a_prop + beta2 - gamma * D2[i, ]
      r2p <- Z_con[i, ] - mu2_p
      ll2_p <- -0.5 / sigma0_sq * sum(r2p^2)
      
      mu3_p <- exp(a_prop + beta3 - gamma * D3[i, ])
      ll3_p <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_p, log = TRUE))
      
      lp_p <- -0.5 * a_prop^2 / sigma_alpha_sq
      logpost_p <- ll1_p + ll2_p + ll3_p + lp_p
      
      log_acc <- logpost_p - logpost_cur
      if (log(runif(1)) < log_acc) {
        alpha[i] <- a_prop
        accept$alpha[i] <- accept$alpha[i] + 1
      }
    }
    
    # ==========================
    # 2) beta^(l) : MH (layer-specific, conditional on shared alpha,a,gamma)
    # ==========================
    # beta1
    for (j in 1:P1) {
      eta_cur <- alpha + beta1[j] - gamma * D1[, j]
      ll_cur <- sum(Y_bin[, j] * eta_cur - log1pexp(eta_cur))
      lp_cur <- -0.5 * beta1[j]^2 / tau_beta1_sq
      
      b_prop <- rnorm(1, beta1[j], prop_sd$beta1)
      eta_p <- alpha + b_prop - gamma * D1[, j]
      ll_p <- sum(Y_bin[, j] * eta_p - log1pexp(eta_p))
      lp_p <- -0.5 * b_prop^2 / tau_beta1_sq
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        beta1[j] <- b_prop
        accept$beta1[j] <- accept$beta1[j] + 1
      }
    }
    
    # beta2
    for (j in 1:P2) {
      mu_cur <- alpha + beta2[j] - gamma * D2[, j]
      r_cur <- Z_con[, j] - mu_cur
      ll_cur <- -0.5 / sigma0_sq * sum(r_cur^2)
      lp_cur <- -0.5 * beta2[j]^2 / tau_beta2_sq
      
      b_prop <- rnorm(1, beta2[j], prop_sd$beta2)
      mu_p <- alpha + b_prop - gamma * D2[, j]
      r_p <- Z_con[, j] - mu_p
      ll_p <- -0.5 / sigma0_sq * sum(r_p^2)
      lp_p <- -0.5 * b_prop^2 / tau_beta2_sq
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        beta2[j] <- b_prop
        accept$beta2[j] <- accept$beta2[j] + 1
      }
    }
    
    # beta3
    for (j in 1:P3) {
      mu_cur <- exp(alpha + beta3[j] - gamma * D3[, j])
      ll_cur <- sum(dnbinom(Y_cnt[, j], size = 1 / kappa, mu = mu_cur, log = TRUE))
      lp_cur <- -0.5 * beta3[j]^2 / tau_beta3_sq
      
      b_prop <- rnorm(1, beta3[j], prop_sd$beta3)
      mu_p <- exp(alpha + b_prop - gamma * D3[, j])
      ll_p <- sum(dnbinom(Y_cnt[, j], size = 1 / kappa, mu = mu_p, log = TRUE))
      lp_p <- -0.5 * b_prop^2 / tau_beta3_sq
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        beta3[j] <- b_prop
        accept$beta3[j] <- accept$beta3[j] + 1
      }
    }
    
    # ==========================
    # 3) log_gamma (shared) : MH using all layers
    # ==========================
    # current
    eta1 <- outer(alpha, beta1, "+") - gamma * D1
    ll1 <- sum(Y_bin * eta1 - log1pexp(eta1))
    
    mu2 <- outer(alpha, beta2, "+") - gamma * D2
    r2 <- Z_con - mu2
    ll2 <- -0.5 / sigma0_sq * sum(r2^2)
    
    mu3 <- exp(outer(alpha, beta3, "+") - gamma * D3)
    ll3 <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3, log = TRUE))
    
    lp_cur <- dnorm(log_gamma, mean = hyper$mu_log_gamma, sd = hyper$sd_log_gamma, log = TRUE)
    logpost_cur <- ll1 + ll2 + ll3 + lp_cur
    
    # prop
    log_gamma_prop <- rnorm(1, log_gamma, prop_sd$log_gamma)
    gamma_prop <- exp(log_gamma_prop)
    
    eta1p <- outer(alpha, beta1, "+") - gamma_prop * D1
    ll1p <- sum(Y_bin * eta1p - log1pexp(eta1p))
    
    mu2p <- outer(alpha, beta2, "+") - gamma_prop * D2
    r2p <- Z_con - mu2p
    ll2p <- -0.5 / sigma0_sq * sum(r2p^2)
    
    mu3p <- exp(outer(alpha, beta3, "+") - gamma_prop * D3)
    ll3p <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3p, log = TRUE))
    
    lp_p <- dnorm(log_gamma_prop, mean = hyper$mu_log_gamma, sd = hyper$sd_log_gamma, log = TRUE)
    logpost_p <- ll1p + ll2p + ll3p + lp_p
    
    log_acc <- logpost_p - logpost_cur
    if (log(runif(1)) < log_acc) {
      log_gamma <- log_gamma_prop
      gamma <- gamma_prop
      accept$log_gamma <- accept$log_gamma + 1
    }
    
    # ==========================
    # 4) shared respondent position a_i : MH using all layers + N(0,I) prior
    # ==========================
    for (i in 1:n) {
      a_cur <- a[i, ]
      
      # current ll
      eta1_i <- alpha[i] + beta1 - gamma * D1[i, ]
      ll1 <- sum(Y_bin[i, ] * eta1_i - log1pexp(eta1_i))
      
      mu2_i <- alpha[i] + beta2 - gamma * D2[i, ]
      r2 <- Z_con[i, ] - mu2_i
      ll2 <- -0.5 / sigma0_sq * sum(r2^2)
      
      mu3_i <- exp(alpha[i] + beta3 - gamma * D3[i, ])
      ll3 <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_i, log = TRUE))
      
      lp_cur <- -0.5 * sum(a_cur^2)  # N(0, I)
      logpost_cur <- ll1 + ll2 + ll3 + lp_cur
      
      # proposal
      a_prop <- a_cur + rnorm(d, 0, prop_sd$a)
      
      # update row distances quickly
      d1p <- numeric(P1)
      for (j in 1:P1) d1p[j] <- sqrt(sum((a_prop - b1[j, ])^2))
      d2p <- numeric(P2)
      for (j in 1:P2) d2p[j] <- sqrt(sum((a_prop - b2[j, ])^2))
      d3p <- numeric(P3)
      for (j in 1:P3) d3p[j] <- sqrt(sum((a_prop - b3[j, ])^2))
      
      eta1_p <- alpha[i] + beta1 - gamma * d1p
      ll1p <- sum(Y_bin[i, ] * eta1_p - log1pexp(eta1_p))
      
      mu2_p <- alpha[i] + beta2 - gamma * d2p
      r2p <- Z_con[i, ] - mu2_p
      ll2p <- -0.5 / sigma0_sq * sum(r2p^2)
      
      mu3_p <- exp(alpha[i] + beta3 - gamma * d3p)
      ll3p <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_p, log = TRUE))
      
      lp_p <- -0.5 * sum(a_prop^2)
      logpost_p <- ll1p + ll2p + ll3p + lp_p
      
      log_acc <- logpost_p - logpost_cur
      if (log(runif(1)) < log_acc) {
        a[i, ] <- a_prop
        D1[i, ] <- d1p
        D2[i, ] <- d2p
        D3[i, ] <- d3p
        accept$a[i] <- accept$a[i] + 1
      }
    }
    
    # ==========================
    # 5) item positions b^(l)_j : MH (per layer), prior N(0,I)
    # ==========================
    # b1
    for (j in 1:P1) {
      b_cur <- b1[j, ]
      eta_cur <- alpha + beta1[j] - gamma * D1[, j]
      ll_cur <- sum(Y_bin[, j] * eta_cur - log1pexp(eta_cur))
      lp_cur <- -0.5 * sum(b_cur^2)
      
      b_prop <- b_cur + rnorm(d, 0, prop_sd$b1)
      
      d1p <- numeric(n)
      for (i in 1:n) d1p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
      eta_p <- alpha + beta1[j] - gamma * d1p
      ll_p <- sum(Y_bin[, j] * eta_p - log1pexp(eta_p))
      lp_p <- -0.5 * sum(b_prop^2)
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        b1[j, ] <- b_prop
        D1[, j] <- d1p
        accept$b1[j] <- accept$b1[j] + 1
      }
    }
    
    # b2
    for (j in 1:P2) {
      b_cur <- b2[j, ]
      mu_cur <- alpha + beta2[j] - gamma * D2[, j]
      r_cur <- Z_con[, j] - mu_cur
      ll_cur <- -0.5 / sigma0_sq * sum(r_cur^2)
      lp_cur <- -0.5 * sum(b_cur^2)
      
      b_prop <- b_cur + rnorm(d, 0, prop_sd$b2)
      
      d2p <- numeric(n)
      for (i in 1:n) d2p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
      mu_p <- alpha + beta2[j] - gamma * d2p
      r_p <- Z_con[, j] - mu_p
      ll_p <- -0.5 / sigma0_sq * sum(r_p^2)
      lp_p <- -0.5 * sum(b_prop^2)
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        b2[j, ] <- b_prop
        D2[, j] <- d2p
        accept$b2[j] <- accept$b2[j] + 1
      }
    }
    
    # b3
    for (j in 1:P3) {
      b_cur <- b3[j, ]
      mu_cur <- exp(alpha + beta3[j] - gamma * D3[, j])
      ll_cur <- sum(dnbinom(Y_cnt[, j], size = 1 / kappa, mu = mu_cur, log = TRUE))
      lp_cur <- -0.5 * sum(b_cur^2)
      
      b_prop <- b_cur + rnorm(d, 0, prop_sd$b3)
      
      d3p <- numeric(n)
      for (i in 1:n) d3p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
      mu_p <- exp(alpha + beta3[j] - gamma * d3p)
      ll_p <- sum(dnbinom(Y_cnt[, j], size = 1 / kappa, mu = mu_p, log = TRUE))
      lp_p <- -0.5 * sum(b_prop^2)
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        b3[j, ] <- b_prop
        D3[, j] <- d3p
        accept$b3[j] <- accept$b3[j] + 1
      }
    }
    
    # ==========================
    # 6) log_kappa (dispersion) : MH (count layer only)
    # ==========================
    mu3 <- exp(outer(alpha, beta3, "+") - gamma * D3)
    ll_cur <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3, log = TRUE))
    lp_cur <- dnorm(log_kappa, mean = hyper$mu_log_kappa, sd = hyper$sd_log_kappa, log = TRUE)
    logpost_cur <- ll_cur + lp_cur
    
    log_kappa_prop <- rnorm(1, log_kappa, prop_sd$log_kappa)
    kappa_prop <- exp(log_kappa_prop)
    ll_p <- sum(dnbinom(Y_cnt, size = 1 / kappa_prop, mu = mu3, log = TRUE))
    lp_p <- dnorm(log_kappa_prop, mean = hyper$mu_log_kappa, sd = hyper$sd_log_kappa, log = TRUE)
    logpost_p <- ll_p + lp_p
    
    log_acc <- logpost_p - logpost_cur
    if (log(runif(1)) < log_acc) {
      log_kappa <- log_kappa_prop
      kappa <- kappa_prop
      accept$log_kappa <- accept$log_kappa + 1
    }
    
    # ==========================
    # 7) Gibbs updates: sigma0_sq, sigma_alpha_sq, tau_beta*_sq
    # ==========================
    # sigma0_sq
    {
      mu2 <- outer(alpha, beta2, "+") - gamma * D2
      r2 <- Z_con - mu2
      SSE <- sum(r2^2)
      shape <- hyper$a_sigma0 + (n * P2) / 2
      rate  <- hyper$b_sigma0 + 0.5 * SSE
      sigma0_sq <- 1 / rgamma(1, shape = shape, rate = rate)
    }
    
    # sigma_alpha_sq
    {
      shape <- hyper$a_sigma + n / 2
      rate  <- hyper$b_sigma + 0.5 * sum(alpha^2)
      sigma_alpha_sq <- 1 / rgamma(1, shape = shape, rate = rate)
    }
    
    # tau_beta1_sq
    {
      shape <- hyper$a_tau1 + P1 / 2
      rate  <- hyper$b_tau1 + 0.5 * sum(beta1^2)
      tau_beta1_sq <- 1 / rgamma(1, shape = shape, rate = rate)
    }
    
    # tau_beta2_sq
    {
      shape <- hyper$a_tau2 + P2 / 2
      rate  <- hyper$b_tau2 + 0.5 * sum(beta2^2)
      tau_beta2_sq <- 1 / rgamma(1, shape = shape, rate = rate)
    }
    
    # tau_beta3_sq
    {
      shape <- hyper$a_tau3 + P3 / 2
      rate  <- hyper$b_tau3 + 0.5 * sum(beta3^2)
      tau_beta3_sq <- 1 / rgamma(1, shape = shape, rate = rate)
    }
    
    # ==========================
    # 8) save + (optional) procrustes later
    # ==========================
    if (iter > burnin && (iter - burnin) %% thin == 0) {
      save_idx <- save_idx + 1
      
      samples$alpha[save_idx, ] <- alpha
      samples$beta1[save_idx, ] <- beta1
      samples$beta2[save_idx, ] <- beta2
      samples$beta3[save_idx, ] <- beta3
      
      samples$log_gamma[save_idx] <- log_gamma
      samples$log_kappa[save_idx] <- log_kappa
      
      samples$sigma_alpha_sq[save_idx] <- sigma_alpha_sq
      samples$tau_beta1_sq[save_idx]   <- tau_beta1_sq
      samples$tau_beta2_sq[save_idx]   <- tau_beta2_sq
      samples$tau_beta3_sq[save_idx]   <- tau_beta3_sq
      samples$sigma0_sq[save_idx]      <- sigma0_sq
      
      samples$a[save_idx, , ]  <- a
      samples$b1[save_idx, , ] <- b1
      samples$b2[save_idx, , ] <- b2
      samples$b3[save_idx, , ] <- b3
      
      # compatibility copies
      samples$z[save_idx, , ]  <- a
      samples$a1[save_idx, , ] <- a
      samples$a2[save_idx, , ] <- a
      samples$a3[save_idx, , ] <- a
      
      # loglik
      eta1 <- outer(alpha, beta1, "+") - gamma * D1
      ll1 <- sum(Y_bin * eta1 - log1pexp(eta1))
      
      mu2 <- outer(alpha, beta2, "+") - gamma * D2
      r2  <- Z_con - mu2
      ll2 <- -0.5 * sum(r2^2) / sigma0_sq - 0.5 * n * P2 * log(2 * pi * sigma0_sq)
      
      mu3 <- exp(outer(alpha, beta3, "+") - gamma * D3)
      ll3 <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3, log = TRUE))
      
      samples$loglik[save_idx] <- ll1 + ll2 + ll3
    }
  }
  
  # ==========================
  # Procrustes matching (align a, b1, b2, b3 together)
  # ==========================
  if (requireNamespace("vegan", quietly = TRUE)) {
    
    loglik_save <- samples$loglik
    map_idx <- which.max(loglik_save)
    
    A_map  <- matrix(samples$a[map_idx, , ],  n,  d)
    B1_map <- matrix(samples$b1[map_idx, , ], P1, d)
    B2_map <- matrix(samples$b2[map_idx, , ], P2, d)
    B3_map <- matrix(samples$b3[map_idx, , ], P3, d)
    
    X_map <- rbind(A_map, B1_map, B2_map, B3_map)
    idx_A  <- 1:n
    idx_B1 <- (n + 1):(n + P1)
    idx_B2 <- (n + P1 + 1):(n + P1 + P2)
    idx_B3 <- (n + P1 + P2 + 1):(n + P1 + P2 + P3)
    
    for (ss in seq_len(n_save)) {
      if (ss == map_idx) next
      
      A_s  <- matrix(samples$a[ss, , ],  n,  d)
      B1_s <- matrix(samples$b1[ss, , ], P1, d)
      B2_s <- matrix(samples$b2[ss, , ], P2, d)
      B3_s <- matrix(samples$b3[ss, , ], P3, d)
      
      Y_s <- rbind(A_s, B1_s, B2_s, B3_s)
      
      fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
      Y_aligned <- predict(fit_all)
      
      samples$a[ss, , ]  <- Y_aligned[idx_A,  , drop = FALSE]
      samples$b1[ss, , ] <- Y_aligned[idx_B1, , drop = FALSE]
      samples$b2[ss, , ] <- Y_aligned[idx_B2, , drop = FALSE]
      samples$b3[ss, , ] <- Y_aligned[idx_B3, , drop = FALSE]
      
      # compatibility copies
      samples$z[ss, , ]  <- samples$a[ss, , ]
      samples$a1[ss, , ] <- samples$a[ss, , ]
      samples$a2[ss, , ] <- samples$a[ss, , ]
      samples$a3[ss, , ] <- samples$a[ss, , ]
    }
  } else {
    warning("Package 'vegan' is not installed. Skipping Procrustes matching.")
  }
  
  # acceptance ratio
  accept$alpha     <- accept$alpha / n_iter
  accept$beta1     <- accept$beta1 / n_iter
  accept$beta2     <- accept$beta2 / n_iter
  accept$beta3     <- accept$beta3 / n_iter
  accept$log_gamma <- accept$log_gamma / n_iter
  accept$log_kappa <- accept$log_kappa / n_iter
  accept$a         <- accept$a / n_iter
  accept$b1        <- accept$b1 / n_iter
  accept$b2        <- accept$b2 / n_iter
  accept$b3        <- accept$b3 / n_iter
  
  return(list(
    samples = samples,
    accept  = accept,
    hyper   = hyper,
    prop_sd = prop_sd
  ))
}


# source(".../KSAH_dataprocessing.R")
source("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM/KSAH_dataprocessing.R")

fit <- lsirm_sharedpos_layer3(
  Y_bin = Y_bin,
  Y_con = Y_likert_5,
  Y_cnt = Y_cnt,
  d     = 2,
  n_iter = 100000,
  burnin = 50000,
  thin   = 5,
  prop_sd = list(
    alpha      = 0.40,
    beta1      = 1.00,
    beta2      = 0.30,
    beta3      = 0.30,
    log_gamma  = 0.05,
    log_kappa  = 0.20,
    a          = 0.70,
    b1         = 0.50,
    b2         = 0.50,
    b3         = 0.20
  )
)

fit$accept

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a)[2]){
  for(j in 1:dim(fit$samples$a)[3]){
    ts.plot(fit$samples$a[,i,j], main = paste0('a: ', i, '_',j))
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b1)[2]){
  for(j in 1:dim(fit$samples$b1)[3]){
    ts.plot(fit$samples$b1[,i,j], main = paste0('b1: ', i, '_',j))
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b2)[2]){
  for(j in 1:dim(fit$samples$b2)[3]){
    ts.plot(fit$samples$b2[,i,j], main = paste0('b2: ', i, '_',j))
  }
}

par(mfrow = c(2,2))
# identifiability 생겨버림.
for(i in 1:dim(fit$samples$b3)[2]){
  for(j in 1:dim(fit$samples$b3)[3]){
    ts.plot(fit$samples$b3[,i,j], main = paste0('b3: ', i, '_',j))
  }
}

#-------------------------------------------------------------------------------
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha)[2]){
  ts.plot(fit$samples$alpha[,i], main = paste0('alpha_',i))
}

# - beta2: difficulty of binary item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta1)[2]){
  ts.plot(fit$samples$beta1[,i], main = paste0('beta1_',i))
}

# - beta2: difficulty of continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta2)[2]){
  ts.plot(fit$samples$beta2[,i], main = paste0('beta2_',i))
}

# - beta3: difficulty of count item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta3)[2]){
  ts.plot(fit$samples$beta3[,i], main = paste0('beta3_',i))
}

#-------------------------------------------------------------------------------
ts.plot(fit$samples$sigma0_sq, main = 'sigma0_sq')

ts.plot(exp(fit$samples$log_gamma), main = 'gamma')

ts.plot(log(fit$samples$tau_beta1_sq), main = 'tau_beta1_sq log transformed')

ts.plot(log(fit$samples$tau_beta2_sq), main = 'tau_beta2_sq log transformed')

ts.plot(log(fit$samples$tau_beta3_sq), main = 'tau_beta3_sq log transformed')

################################################################################
# latent position visualization
################################################################################
# response
a_mean <- colMeans(fit$samples$a, dims = 1)

base::plot(
  a_mean,
  pch = 21, col = "black", bg = "gray80",
  main = "Latent positions: a",
  ylim = c(-3,3),
  xlim = c(-3,3)
)

# --- 숫자 라벨 추가 ---

text(a_mean,  labels = 1:dim(a_mean)[1],  pos = 4, cex = 0.7)

legend(
  "topright",
  legend = c("a: respondent"),
  pch = 21,
  bty = "n"
)

# item
b3_mean <- colMeans(fit$samples$b3, dims = 1)
b2_mean <- colMeans(fit$samples$b2, dims = 1)
b1_mean <- colMeans(fit$samples$b1, dims = 1)

par(mfrow = c(1,1))
plot(
  a_mean,
  pch = 21, col = "red", bg = "red",
  main = "Item positions: b2 (blue), b1 (skyblue)"
)
points(b2_mean, pch = 21, col = "navy", bg = "blue")
points(b1_mean, pch = 21, col = "steelblue4", bg = "skyblue")
points(b3_mean, pch = 21, col = "grey", bg = "grey")
# --- 숫자 라벨 추가 ---
text(a_mean, labels = 1:dim(a_mean)[1], pos = 4, cex = 0.7, col = "red")
text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")
text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "steelblue4")
text(b3_mean, labels = 1:dim(b3_mean)[1], pos = 4, cex = 0.7, col = "grey")

legend(
  "topright",
  legend = c("a: respondent", "b2: continuous item", "b1: binary item", "b3: count item"),
  pch = 21,
  pt.bg = c("red", "blue", "skyblue","grey"),
  col = c("red","navy", "steelblue4","grey"),
  bty = "n"
)
