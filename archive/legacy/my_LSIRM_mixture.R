rm(list = ls())
# ----------------------------
# Global-Local LSIRM (Option 1)
# ----------------------------
library(vegan)
# Y_bin = Y_bin
# Y_con = Y_likert
# d     = 2
# n_iter = 5000
# burnin = 1000
# thin   = 5
# hyper   = list(
#   # alpha variance priors
#   a_sigma1 = 1, b_sigma1 = 1,
#   a_sigma2 = 1, b_sigma2 = 1,
#   # beta variance priors
#   a_tau1   = 1, b_tau1   = 1,
#   a_tau2   = 1, b_tau2   = 1,
#   # residual variance prior (continuous layer)
#   a_sigma0 = 1, b_sigma0 = 1,
#   # log-gamma priors
#   mu_log_gamma1 = 0, sd_log_gamma1 = 1,
#   mu_log_gamma2 = 0, sd_log_gamma2 = 1
# )
# prop_sd = list(
#   alpha1     = 0.10,
#   beta1      = 0.10,
#   alpha2     = 0.10,
#   beta2      = 0.10,
#   log_gamma1 = 0.05,
#   log_gamma2 = 0.05,
#   a1         = 0.10,  # local respondent pos, layer1
#   a2         = 0.10,  # local respondent pos, layer2
#   b1         = 0.10,  # item pos binary
#   b2         = 0.10   # item pos continuous
# )
# init    = NULL
# verbose = TRUE

lsirm_global_local_option1 <- function(
    Y_bin,              # n x P1, 0/1
    Y_con,              # n x P2, >0 (continuous)
    d       = 2,        # latent dimension
    n_iter  = 5000,
    burnin  = 1000,
    thin    = 5,
    hyper   = list(
      # alpha variance priors
      a_sigma1 = 1, b_sigma1 = 0.1,
      a_sigma2 = 1, b_sigma2 = 0.1,
      # beta variance priors
      a_tau1   = 1, b_tau1   = 0.1,
      a_tau2   = 1, b_tau2   = 0.1,
      # residual variance prior (continuous layer)
      a_sigma0 = 1, b_sigma0 = 1,
      # Global-Local linkage variance (sigma_1^2)
      a_sigma_global = 1, b_sigma_global = 1,
      # log-gamma priors
      mu_log_gamma1 = 0, sd_log_gamma1 = 1,
      mu_log_gamma2 = 0, sd_log_gamma2 = 1
    ),
    prop_sd = list(
      alpha1     = 0.10, beta1      = 0.10,
      alpha2     = 0.10, beta2      = 0.10,
      log_gamma1 = 0.05, log_gamma2 = 0.05,
      a1         = 0.10,  # local respondent pos, layer1
      a2         = 0.10,  # local respondent pos, layer2
      b1         = 0.10,  # item pos binary
      b2         = 0.10   # item pos continuous
    ),
    init    = NULL,
    verbose = TRUE
) {
  # -------------------------
  # Helper: stable log(1+exp(x))
  # -------------------------
  log1pexp <- function(x) {
    ifelse(x > 0,
           x + log1p(exp(-x)),
           log1p(exp(x)))
  }
  
  # -------------------------
  # 기본 설정 및 데이터 전처리
  # -------------------------
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  
  n  <- nrow(Y_bin)            # respondents
  P1 <- ncol(Y_bin)            # binary items
  P2 <- ncol(Y_con)            # continuous items
  
  # log-transform for continuous layer
  
  # non-negative 조건 없어도 될 것 같다.
  # Z_con <- log(Y_con)
  Z_con <- Y_con
  
  # missing indicator (NA 허용하지 않음)
  if(sum(is.na(Y_bin)) + sum(is.na(Z_con))){
    stop("Missing values are not allowed in the data.")
  }
  
  # -------------------------
  # 초기값 세팅
  # -------------------------
  if (is.null(init)) {
    alpha1 <- rnorm(n, 0, 0.1)
    beta1  <- rnorm(P1, 0, 0.1)
    alpha2 <- rnorm(n, 0, 0.1)
    beta2  <- rnorm(P2, 0, 0.1)
    
    log_gamma1 <- 0
    log_gamma2 <- 0
    
    # local positions
    a1 <- matrix(rnorm(n * d, 0, 0.5), n, d)
    a2 <- matrix(rnorm(n * d, 0, 0.5), n, d)
    b1 <- matrix(rnorm(P1 * d, 0, 0.5), P1, d)
    b2 <- matrix(rnorm(P2 * d, 0, 0.5), P2, d)
    
    # global positions
    # z  <- matrix(0, n, d)
    z  <- matrix(rnorm(n * d, 0, 1), n, d)
    
    # variance params
    sigma_alpha1_sq <- 1
    sigma_alpha2_sq <- 1
    tau_beta1_sq    <- 1
    tau_beta2_sq    <- 1
    sigma0_sq       <- 1
    sigma1_sq       <- 0.5
  } else {
    alpha1 <- init$alpha1
    beta1  <- init$beta1
    alpha2 <- init$alpha2
    beta2  <- init$beta2
    log_gamma1 <- init$log_gamma1
    log_gamma2 <- init$log_gamma2
    a1 <- init$a1; a2 <- init$a2
    b1 <- init$b1; b2 <- init$b2
    z  <- init$z
    sigma_alpha1_sq <- init$sigma_alpha1_sq
    sigma_alpha2_sq <- init$sigma_alpha2_sq
    tau_beta1_sq    <- init$tau_beta1_sq
    tau_beta2_sq    <- init$tau_beta2_sq
    sigma0_sq       <- init$sigma0_sq
    sigma1_sq       <- init$sigma1_sq
  }
  
  gamma1 <- exp(log_gamma1)
  gamma2 <- exp(log_gamma2)
  
  # -------------------------
  # 저장 객체 만들기
  # -------------------------
  n_save <- floor((n_iter - burnin) / thin)
  
  samples <- list(
    alpha1      = array(NA, c(n_save, n)),
    beta1       = array(NA, c(n_save, P1)),
    alpha2      = array(NA, c(n_save, n)),
    beta2       = array(NA, c(n_save, P2)),
    log_gamma1  = numeric(n_save),
    log_gamma2  = numeric(n_save),
    sigma_alpha1_sq = numeric(n_save),
    sigma_alpha2_sq = numeric(n_save),
    tau_beta1_sq    = numeric(n_save),
    tau_beta2_sq    = numeric(n_save),
    sigma0_sq       = numeric(n_save),
    sigma1_sq       = numeric(n_save),
    a1          = array(NA, c(n_save, n, d)),
    a2          = array(NA, c(n_save, n, d)),
    b1          = array(NA, c(n_save, P1, d)),
    b2          = array(NA, c(n_save, P2, d)),
    z           = array(NA, c(n_save, n, d)),
    loglik      = numeric(n_save)
  )
  
  accept <- list(
    alpha1     = rep(0, n),
    beta1      = rep(0, P1),
    alpha2     = rep(0, n),
    beta2      = rep(0, P2),
    log_gamma1 = 0,
    log_gamma2 = 0,
    a1         = rep(0, n),
    a2         = rep(0, n),
    b1         = rep(0, P1),
    b2         = rep(0, P2)
  )
  save_idx <- 0
  
  # -------------------------
  # MCMC 시작
  # -------------------------
  for (iter in 1:n_iter) {
    
    if (verbose && iter %% 500 == 0) {
      cat("Iter:", iter, "/", n_iter, "\n")
    }
    
    # ---------------------------------
    # 1. Binary layer (ℓ = 1)
    # ---------------------------------
    # computing distance matrix
    D1 <- matrix(0, n, P1)
    for (i in 1:n) {
      for (j in 1:P1) {
        diff_ij <- a1[i, ] - b1[j, ]
        D1[i, j] <- sqrt(sum(diff_ij^2))
      }
    }
    
    # (1) alpha1_i 업데이트 (MH)
    for (i in 1:n) {
      
      # 현재 log-likelihood + prior
      eta_i    <- alpha1[i] + beta1 - gamma1 * D1[i,]
      y_i   <- Y_bin[i,]
      
      loglik_current <- sum(y_i * eta_i - log1p(exp(eta_i)))
      logprior_current <- -0.5 * alpha1[i]^2 / sigma_alpha1_sq
      
      # 제안
      alpha_prop <- rnorm(1, mean = alpha1[i], sd = prop_sd$alpha1)
      eta_i_prop   <- alpha_prop + beta1 - gamma1 * D1[i,]
      loglik_prop <- sum(y_i * eta_i_prop - log1p(exp(eta_i_prop)))
      logprior_prop <- -0.5 * alpha_prop^2 / sigma_alpha1_sq
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        alpha1[i] <- alpha_prop
        accept$alpha1[i] <- accept$alpha1[i] + 1
      }
    }
    
    # (2) beta1_j 업데이트 (MH)
    for (j in 1:P1) {
      
      # 현재 log-likelihood + prior
      eta_j  <- alpha1 + beta1[j] - gamma1 * D1[,j]
      y_j <- Y_bin[,j]
      loglik_current <- sum(y_j * eta_j - log1p(exp(eta_j)))
      logprior_current <- -0.5 * beta1[j]^2 / tau_beta1_sq
      
      beta_prop <- rnorm(1, mean = beta1[j], sd = prop_sd$beta1)
      eta_prop  <- alpha1 + beta_prop - gamma1 * D1[,j]
      loglik_prop <- sum(y_j * eta_prop - log1p(exp(eta_prop)))
      logprior_prop <- -0.5 * beta_prop^2 / tau_beta1_sq
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        beta1[j] <- beta_prop
        accept$beta1[j] <- accept$beta1[j] + 1
      }
    }
    
    # (3) log_gamma1 업데이트 (MH) - global
    # 현재 log-likelihood + prior
    alphabeta <- outer(alpha1, beta1, "+")
    eta     <- alphabeta - gamma1 * D1
    loglik_current <- sum(Y_bin * eta - log1p(exp(eta)))
    logprior_current <- dnorm(log_gamma1,
                              mean = hyper$mu_log_gamma1,
                              sd   = hyper$sd_log_gamma1,
                              log  = TRUE)
    
    log_gamma_prop <- rnorm(1, mean = log_gamma1, sd = prop_sd$log_gamma1)
    gamma_prop     <- exp(log_gamma_prop)
    eta_prop       <- alphabeta - gamma_prop * D1
    loglik_prop    <- sum(Y_bin * eta_prop - log1p(exp(eta_prop)))
    logprior_prop  <- dnorm(log_gamma_prop,
                            mean = hyper$mu_log_gamma1,
                            sd   = hyper$sd_log_gamma1,
                            log  = TRUE)
    
    # MH using jacobian
    log_acc <- (loglik_prop + logprior_prop) -
      (loglik_current + logprior_current) + 
      (log_gamma1 - log_gamma_prop)
    
    
    if (log(runif(1)) < log_acc) {
      log_gamma1 <- log_gamma_prop
      gamma1     <- gamma_prop
      accept$log_gamma1 <- accept$log_gamma1 + 1
    }
    
    # (4) a1_i (local respondent position, layer 1) 업데이트 (MH)
    for (i in 1:n) {
      a_current <- a1[i, ]
      
      # likelihood part
      d_i <- D1[i,]
      eta  <- alpha1[i] + beta1 - gamma1 * d_i
      y_i <- Y_bin[i,]
      loglik_current <- sum(y_i * eta - log1p(exp(eta)))
      
      # prior: a1_i | z_i ~ N(z_i, I_d)
      diff_current <- a_current - z[i, ]
      logprior_current <- -0.5 / sigma1_sq * sum(diff_current^2)
      
      # proposal
      a_prop <- a_current + rnorm(d, 0, prop_sd$a1)
      
      # update distances only for row i
      d_prop <- numeric(length(d_i))
      for (k in 1:length(d_i)) {
        d_prop[k] <- sqrt(sum((a_prop - b1[k, ])^2))
      }
      
      eta_prop <- alpha1[i] + beta1 - gamma1 * d_prop
      loglik_prop <- sum(y_i * eta_prop - log1p(exp(eta_prop)))
      
      diff_prop <- a_prop - z[i, ]
      logprior_prop <- -0.5 / sigma1_sq * sum(diff_prop^2)
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        a1[i, ] <- a_prop
        D1[i,] <- d_prop
        accept$a1[i] <- accept$a1[i] + 1
      }
    }
    
    # (5) b1_j (item position, layer 1) 업데이트 (MH)
    for (j in 1:P1) {
      b_current <- b1[j, ]
      
      # likelihood
      d_j <- D1[, j]
      eta  <- alpha1 + beta1[j] - gamma1 * d_j
      y_j <- Y_bin[, j]
      loglik_current <- sum(y_j * eta - log1p(exp(eta)))
      
      # prior: N(0, I_d)
      logprior_current <- -0.5 * sum(b_current^2)
      
      # proposal
      b_prop <- b_current + rnorm(d, 0, prop_sd$b1)
      
      # update distances only for column j
      d_prop <- numeric(length(d_j))
      for (k in 1:length(d_j)) {
        d_prop[k] <- sqrt(sum((b_prop - a1[k, ])^2))
      }
      
      eta_prop <- alpha1 + beta1[j] - gamma1 * d_prop
      loglik_prop <- sum(y_j * eta_prop - log1p(exp(eta_prop)))
      
      logprior_prop <- -0.5 * sum(b_prop^2)
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        b1[j, ] <- b_prop
        D1[,j] <- d_prop
        accept$b1[j] <- accept$b1[j] + 1
      }
    }
    
    # (6) variance params (binary layer) – Gibbs
    # sigma_alpha1^2 | alpha1 ~ IG(a_sigma1 + n/2, b_sigma1 + 0.5 * sum(alpha1^2))
    sigma_alpha1_shape <- hyper$a_sigma1 + n / 2
    sigma_alpha1_rate  <- hyper$b_sigma1 + 0.5 * sum(alpha1^2)
    sigma_alpha1_sq    <- 1 / rgamma(1,
                                     shape = sigma_alpha1_shape,
                                     rate  = sigma_alpha1_rate)
    
    # tau_beta1^2 | beta1 ~ IG(a_tau1 + P1/2, b_tau1 + 0.5 * sum(beta1^2))
    tau_beta1_shape <- hyper$a_tau1 + P1 / 2
    tau_beta1_rate  <- hyper$b_tau1 + 0.5 * sum(beta1^2)
    tau_beta1_sq    <- 1 / rgamma(1,
                                  shape = tau_beta1_shape,
                                  rate  = tau_beta1_rate)
    
    # ---------------------------------
    # 2. Continuous layer (ℓ = 2)
    # ---------------------------------
    # Distance matrix D2: n x P2
    D2 <- matrix(0, n, P2)
    for (i in 1:n) {
      for (j in 1:P2) {
        diff_ij <- a2[i, ] - b2[j, ]
        D2[i, j] <- sqrt(sum(diff_ij^2))
      }
    }
    
    # (1) alpha2_i 업데이트 (MH, Normal likelihood)
    for (i in 1:n) {
      d_i <- D2[i,]
      y_i <- Z_con[i,]
      mu_current <- alpha2[i] + beta2 - gamma2 * d_i
      resid_current <- y_i - mu_current
      loglik_current <- -0.5 / sigma0_sq * sum(resid_current^2)
      logprior_current <- -0.5 * alpha2[i]^2 / sigma_alpha2_sq
      
      alpha_prop <- rnorm(1, mean = alpha2[i], sd = prop_sd$alpha2)
      mu_prop <- alpha_prop + beta2 - gamma2 * d_i
      resid_prop <- y_i - mu_prop
      loglik_prop <- -0.5 / sigma0_sq * sum(resid_prop^2)
      logprior_prop <- -0.5 * alpha_prop^2 / sigma_alpha2_sq
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        alpha2[i] <- alpha_prop
        accept$alpha2[i] <- accept$alpha2[i] + 1
      }
    }
    
    # (2) beta2_j 업데이트 (MH)
    for (j in 1:P2) {
      y_j <- Z_con[, j]
      d_j <- D2[, j]
      
      mu_current <- alpha2 + beta2[j] - gamma2 * d_j
      resid_current <- y_j - mu_current
      loglik_current <- -0.5 / sigma0_sq * sum(resid_current^2)
      logprior_current <- -0.5 * beta2[j]^2 / tau_beta2_sq
      
      beta_prop <- rnorm(1, mean = beta2[j], sd = prop_sd$beta2)
      mu_prop <- alpha2 + beta_prop - gamma2 * d_j
      resid_prop <- y_j - mu_prop
      loglik_prop <- -0.5 / sigma0_sq * sum(resid_prop^2)
      logprior_prop <- -0.5 * beta_prop^2 / tau_beta2_sq
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        beta2[j] <- beta_prop
        accept$beta2[j] <- accept$beta2[j] + 1
      }
    }
    
    # (3) log_gamma2 업데이트 (MH)
    # current
    alphabeta <- outer(alpha2, beta2, "+")
    mu_current <- alphabeta - gamma2 * D2
    resid_current <- Z_con - mu_current
    loglik_current <- -0.5 / sigma0_sq * sum(resid_current^2)
    logprior_current <- dnorm(log_gamma2,
                              mean = hyper$mu_log_gamma2,
                              sd   = hyper$sd_log_gamma2,
                              log  = TRUE)
    # prop
    log_gamma_prop <- rnorm(1, mean = log_gamma2, sd = prop_sd$log_gamma2)
    gamma_prop     <- exp(log_gamma_prop)
    mu_prop <- alphabeta - gamma_prop * D2
    resid_prop <- Z_con - mu_prop
    loglik_prop <- -0.5 / sigma0_sq * sum(resid_prop^2)
    logprior_prop <- dnorm(log_gamma_prop,
                           mean = hyper$mu_log_gamma2,
                           sd   = hyper$sd_log_gamma2,
                           log  = TRUE)
    
    log_acc <- (loglik_prop + logprior_prop) -
      (loglik_current + logprior_current)
    
    if (log(runif(1)) < log_acc) {
      log_gamma2 <- log_gamma_prop
      gamma2     <- gamma_prop
      accept$log_gamma2 <- accept$log_gamma2 + 1
    }
    
    # (4) a2_i (local respondent pos, layer 2) 업데이트 (MH)
    for (i in 1:n) {
      a_current <- a2[i, ]
      
      y_i <- Z_con[i, ]
      d_i <- D2[i, ]
      mu_current <- alpha2[i] + beta2 - gamma2 * d_i
      resid_current <- y_i - mu_current
      loglik_current <- -0.5 / sigma0_sq * sum(resid_current^2)
      
      # prior: a2_i | z_i ~ N(z_i, I_d)
      diff_current <- a_current - z[i, ]
      logprior_current <- -0.5 / sigma1_sq * sum(diff_current^2)
      
      # proposal
      a_prop <- a_current + rnorm(d, 0, prop_sd$a2)
      
      # update distances only for row i
      d_prop <- numeric(length(d_i))
      for (k in 1:length(d_i)) {
        d_prop[k] <- sqrt(sum((a_prop - b2[k, ])^2))
      }
      
      mu_prop <- alpha2[i] + beta2 - gamma2 * d_prop
      resid_prop <- y_i - mu_prop
      loglik_prop <- -0.5 / sigma0_sq * sum(resid_prop^2)
      
      diff_prop <- a_prop - z[i, ]
      logprior_prop <- -0.5 / sigma1_sq * sum(diff_prop^2)
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        a2[i, ] <- a_prop
        D2[i,] <- d_prop
        accept$a2[i] <- accept$a2[i] + 1
      }
    }
    
    # (5) b2_j (item pos, layer 2) 업데이트 (MH)
    for (j in 1:P2) {
      b_current <- b2[j, ]
      
      y_j <- Z_con[, j]
      d_j <- D2[, j]
      mu_current <- alpha2 + beta2[j] - gamma2 * d_j
      resid_current <- y_j - mu_current
      loglik_current <- -0.5 / sigma0_sq * sum(resid_current^2)
      
      # prior: N(0, I_d)
      logprior_current <- -0.5 * sum(b_current^2)
      
      # proposal
      b_prop <- b_current + rnorm(d, 0, prop_sd$b2)
      
      d_prop <- numeric(length(d_j))
      for (k in 1:length(d_j)) {
        d_prop[k] <- sqrt(sum((a2[k, ] - b_prop)^2))
      }
      
      mu_prop <- alpha2 + beta2[j] - gamma2 * d_prop
      resid_prop <- y_j - mu_prop
      loglik_prop <- -0.5 / sigma0_sq * sum(resid_prop^2)
      
      # prior: N(0, I_d)
      logprior_prop <- -0.5 * sum(b_prop^2)
      
      log_acc <- (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current)
      
      if (log(runif(1)) < log_acc) {
        b2[j, ] <- b_prop
        D2[,j] <- d_prop
        accept$b2[j] <- accept$b2[j] + 1
      }
    }
    
    # (6) sigma0^2 (residual variance, continuous) – Gibbs
    {
      mu_vec <- alphabeta - gamma2 * D2
      resid_vec <- Z_con - mu_vec
      SSE <- sum(resid_vec^2)
      
      shape_sigma0 <- hyper$a_sigma0 + n*P2 / 2
      rate_sigma0  <- hyper$b_sigma0 + 0.5 * SSE
      sigma0_sq    <- 1 / rgamma(1,
                                 shape = shape_sigma0,
                                 rate  = rate_sigma0)
    }
    
    # (7) variance params (continuous layer) – Gibbs
    # sigma_alpha2^2
    sigma_alpha2_shape <- hyper$a_sigma2 + n / 2
    sigma_alpha2_rate  <- hyper$b_sigma2 + 0.5 * sum(alpha2^2)
    sigma_alpha2_sq    <- 1 / rgamma(1,
                                     shape = sigma_alpha2_shape,
                                     rate  = sigma_alpha2_rate)
    
    # tau_beta2^2
    tau_beta2_shape <- hyper$a_tau2 + P2 / 2
    tau_beta2_rate  <- hyper$b_tau2 + 0.5 * sum(beta2^2)
    tau_beta2_sq    <- 1 / rgamma(1,
                                  shape = tau_beta2_shape,
                                  rate  = tau_beta2_rate)
    
    # ---------------------------------
    # 3. Global position z_i – Gibbs
    # ---------------------------------
    # (3.1) Update z_i (Global Position) (Gibbs)
    # Model: z_i ~ N(0, I), a_i^(l) ~ N(z_i, sigma1^2 I)
    # Likelihood from a1, a2.
    # Posterior Precision = 1 (prior) + L/sigma1^2 (from layers). Here L=2.
    var_z_post <- 1/(1 + 2/sigma1_sq)
    sd_z_post  <- sqrt(var_z_post)
    
    for (i in 1:n) {
      # Posterior Mean = (0 + a1[i,] + a2[i,]) / 3
      mean_z_post <- var_z_post * (a1[i, ] + a2[i, ]) / sigma1_sq
      
      for (k in 1:d){
        z[i, k] <- rnorm(1, mean = mean_z_post[k], sd = sd_z_post)
      }
    }
    
    # (3.2) Update sigma1^2 (Global-Local Linkage Variance) (Gibbs)
    # a_i^(l) ~ N(z_i, sigma1^2 I)
    # Total data points: n * 2(layers) * d(dimensions)
    # SSE = sum ||a1_i - z_i||^2 + sum ||a2_i - z_i||^2
    sse_z <- sum((a1 - z)^2) + sum((a2 - z)^2)
    
    # Note: The LaTeX formula says shape = a1 + n/2.
    # Statistically, for d-dimensions, it is usually n*L*d/2.
    # Implementing based on dimension d for robustness.
    # L = 2 layers.
    total_obs <- n * 2 * d
    sigma1_sq <- 1 / rgamma(1, 
                            shape = hyper$a_sigma_global + total_obs/2, 
                            rate = hyper$b_sigma_global + sse_z/2)
    
    # ---------------------------------
    # 4. 저장
    # ---------------------------------
    if (iter > burnin && (iter - burnin) %% thin == 0) {
      save_idx <- save_idx + 1
      
      samples$alpha1[save_idx, ] <- alpha1
      samples$beta1[save_idx, ]  <- beta1
      samples$alpha2[save_idx, ] <- alpha2
      samples$beta2[save_idx, ]  <- beta2
      
      samples$log_gamma1[save_idx] <- log_gamma1
      samples$log_gamma2[save_idx] <- log_gamma2
      
      samples$sigma_alpha1_sq[save_idx] <- sigma_alpha1_sq
      samples$sigma_alpha2_sq[save_idx] <- sigma_alpha2_sq
      samples$tau_beta1_sq[save_idx]    <- tau_beta1_sq
      samples$tau_beta2_sq[save_idx]    <- tau_beta2_sq
      samples$sigma0_sq[save_idx]       <- sigma0_sq
      samples$sigma1_sq[save_idx]       <- sigma1_sq
      
      samples$a1[save_idx, , ] <- a1
      samples$a2[save_idx, , ] <- a2
      samples$b1[save_idx, , ] <- b1
      samples$b2[save_idx, , ] <- b2
      samples$z[save_idx, , ]  <- z
      
      # compute likelihood for procrustes matching
      ## 1) Binary layer log-likelihood
      ## eta1_ij = alpha1_i + beta1_j - gamma1 * D1_ij
      eta1 <- outer(alpha1, beta1, "+") - gamma1 * D1  # n x P1
      loglik_bin <- sum(Y_bin * eta1 - log1p(exp(eta1)))
      
      ## 2) Continuous layer log-likelihood
      ## mu2_ij = alpha2_i + beta2_j - gamma2 * D2_ij
      mu2 <- outer(alpha2, beta2, "+") - gamma2 * D2   # n x P2
      resid2 <- Z_con - mu2
      loglik_con <-
        -0.5 * sum(resid2^2) / sigma0_sq -
        0.5 * n * P2 * log(2 * pi * sigma0_sq)
      
      ## 3) 전체 log-likelihood
      samples$loglik[save_idx] <- loglik_bin + loglik_con
    }
  } # end for(iter)
  
  z_save <- samples$z
  a1_save <- samples$a1
  a2_save <- samples$a2
  b1_save <- samples$b1
  b2_save <- samples$b2
  
  loglik_save <- samples$loglik
  map_idx <- which.max(loglik_save)
  
  Z_map   <- matrix(z_save[map_idx, , ], n, d)
  a1_map <- matrix(a1_save[map_idx, , ], n, d)
  a2_map <- matrix(a2_save[map_idx, , ], n, d)
  b1_map <- matrix(b1_save[map_idx, , ], dim(b1_save)[2], d)
  b2_map <- matrix(b2_save[map_idx, , ], dim(b2_save)[2], d)
  X_map <- rbind(Z_map, a1_map, a2_map, b1_map, b2_map)
  # # procrustes matching
  # if (requireNamespace("vegan", quietly = TRUE)) {
  #   for (ss in seq_len(n_save)) {
  #     if (ss == map_idx) next
  #     Z_s <- matrix(z_save[ss, , ], n, d)
  #     a1_s <- matrix(a1_save[ss, , ], n, d)
  #     a2_s <- matrix(a2_save[ss, , ], n, d)
  #     b1_s <- matrix(b1_save[ss, , ], dim(b1_save)[2], d)
  #     b2_s <- matrix(b2_save[ss, , ], dim(b2_save)[2], d)
  #     
  #     fit_z <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
  #     fit_a1 <- vegan::procrustes(X = a1_map, Y = a1_s, scale = FALSE)
  #     fit_a2 <- vegan::procrustes(X = a2_map, Y = a2_s, scale = FALSE)
  #     fit_b1 <- vegan::procrustes(X = b1_map, Y = b1_s, scale = FALSE)
  #     fit_b2 <- vegan::procrustes(X = b2_map, Y = b2_s, scale = FALSE)
  #     
  #     z_save[ss, , ] <- predict(fit_z)
  #     a1_save[ss, , ] <- predict(fit_a1)
  #     a2_save[ss, , ] <- predict(fit_a2)
  #     b1_save[ss, , ] <- predict(fit_b1)
  #     b2_save[ss, , ] <- predict(fit_b2)
  #   }
  # } else {
  #   warning("Package 'vegan' is not installed. Skipping Procrustes matching.")
  # }
  # 
  # samples$z <- z_save
  # samples$a1 <- a1_save
  # samples$a2 <- a2_save
  # samples$b1 <- b1_save
  # samples$b2 <- b2_save
  
  if (requireNamespace("vegan", quietly = TRUE)) {
    
    # 블록별 행 개수 (일반화)
    n_Z  <- nrow(Z_map)
    n_a1 <- nrow(a1_map)
    n_a2 <- nrow(a2_map)
    n_b1 <- nrow(b1_map)
    n_b2 <- nrow(b2_map)
    
    block_sizes <- c(n_Z, n_a1, n_a2, n_b1, n_b2)
    cs <- cumsum(block_sizes)
    
    idx_Z  <- 1:cs[1]
    idx_a1 <- (cs[1] + 1):cs[2]
    idx_a2 <- (cs[2] + 1):cs[3]
    idx_b1 <- (cs[3] + 1):cs[4]
    idx_b2 <- (cs[4] + 1):cs[5]
    
    ## ----- 1. 각 저장된 샘플에 대해 Procrustes 정렬 -----
    for (ss in seq_len(n_save)) {
      if (ss == map_idx) next
      
      # 사람 latent
      Z_s  <- matrix(z_save[ss, , ],  n_Z,  d)
      a1_s <- matrix(a1_save[ss, , ], n_a1, d)
      a2_s <- matrix(a2_save[ss, , ], n_a2, d)
      
      # 문항 latent
      b1_s <- matrix(b1_save[ss, , ], n_b1, d)
      b2_s <- matrix(b2_save[ss, , ], n_b2, d)
      
      # 1) concatenate
      Y_s <- rbind(Z_s, a1_s, a2_s, b1_s, b2_s)
      
      # 2) Single Procrustes
      fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
      Y_s_aligned <- predict(fit_all)
      
      # 3) Split back (행 개수에만 의존하므로 완전 general)
      z_save[ss, , ]  <- Y_s_aligned[idx_Z,  , drop = FALSE]
      a1_save[ss, , ] <- Y_s_aligned[idx_a1, , drop = FALSE]
      a2_save[ss, , ] <- Y_s_aligned[idx_a2, , drop = FALSE]
      b1_save[ss, , ] <- Y_s_aligned[idx_b1, , drop = FALSE]
      b2_save[ss, , ] <- Y_s_aligned[idx_b2, , drop = FALSE]
    }
    
  } else {
    warning("Package 'vegan' is not installed. Skipping Procrustes matching.")
  }
  
  
  samples$z  <- z_save
  samples$a1 <- a1_save
  samples$a2 <- a2_save
  samples$b1 <- b1_save
  samples$b2 <- b2_save
  
  
  # acceptance ratio
  accept$alpha1     <- accept$alpha1 / n_iter
  accept$beta1      <- accept$beta1 / n_iter
  accept$alpha2     <- accept$alpha2 / n_iter
  accept$beta2      <- accept$beta2 / n_iter
  accept$log_gamma1 <- accept$log_gamma1 / n_iter
  accept$log_gamma2 <- accept$log_gamma2 / n_iter
  accept$a1         <- accept$a1 / n_iter
  accept$a2         <- accept$a2 / n_iter
  accept$b1         <- accept$b1 / n_iter
  accept$b2         <- accept$b2 / n_iter
  
  list(
    samples = samples,
    accept  = accept,
    hyper   = hyper,
    prop_sd = prop_sd
  )
}
################################################################################
# simulation
################################################################################
# simulation setting:

# latent position scheme: 
### 1. respondent latent position is same for layer 1,2
### 2. item latent position has two cluster for each layer 1,2

# 1. ability, difficulty sampling for layer1, 2
res_num = 40
bin_num = 20
con_num = 20
sigma_alpha1 <- 2
alpha_1  <- rnorm(res_num, mean = 0, sd = sigma_alpha1)
sigma_alpha2 <- 1
alpha_2  <- rnorm(res_num, mean = 2, sd = sigma_alpha2)
tau_beta1 <- 2
beta_1  <- rnorm(bin_num, mean = 0, sd = tau_beta1)
tau_beta2 <- 1
beta_2  <- rnorm(con_num, mean = 3, sd = tau_beta2)

# 2. z, layer1, layer2 latent position sampling
respondents <- matrix(ncol = 2, nrow = res_num)
bin_layer_item <- matrix(ncol = 2, nrow = bin_num)
con_layer_item <- matrix(ncol = 2, nrow = con_num)

# using same centroid for item, respondent
centroid_1 <- c(1,1)
centroid_2 <- c(-1,-1)
cluster_num_res = c(20,20)
cluster_num_item_bin = c(10,10)
cluster_num_item_con = c(10,10)

# 클러스터 생성 함수
make_cluster <- function(n, centroid, sd = 0.1) {
  cbind(
    rnorm(n, mean = centroid[1], sd = sd),
    rnorm(n, mean = centroid[2], sd = sd)
  )
}

make_res_position <- function(res, layernum = 2, sd = 0.1){
  n <- dim(res)[1]
  d <- dim(res)[2]
  
  layer <- array(NA, dim = c(layernum, n, d))
  for(i in 1:layernum){
    for(j in 1:d){
      layer[i,,j] <- res[,j] + rnorm(n, mean = 0, sd = 0.1)
    }
  }
  return(layer)
}


# Respondent latent positions
sigma_res_global = 0.1
res_cluster1 <- make_cluster(cluster_num_res[1], centroid_1, sd = sigma_res_global)
res_cluster2 <- make_cluster(cluster_num_res[2], centroid_2, sd = sigma_res_global)
res_global <- rbind(res_cluster1, res_cluster2)
sigma_res_1_2 = 0.1
res_layered <- make_res_position(res_global, layernum = 2, sd = sigma_res_1_2)

# Binary items latent positions
sigma_item_bin = 0.1
item_bin_cluster1 <- make_cluster(cluster_num_item_bin[1], centroid_1, sd = sigma_item_bin)
item_bin_cluster2 <- make_cluster(cluster_num_item_bin[2], centroid_2, sd = sigma_item_bin)
bin_item <- rbind(item_bin_cluster1, item_bin_cluster2)

# Continuous items latent positions
sigma_item_con = 0.1
item_con_cluster1 <- make_cluster(cluster_num_item_con[1], centroid_1, sd = sigma_item_con)
item_con_cluster2 <- make_cluster(cluster_num_item_con[2], centroid_2, sd = sigma_item_con)
con_item <- rbind(item_con_cluster1, item_con_cluster2)

# plotting
par(mfrow = c(1,2))
plot(bin_item)
plot(con_item)
par(mfrow = c(2,2))
plot(res_global)
plot(res_layered[1,,])
plot(res_layered[2,,])

# 4. sample responses for each layer
l2_dist_mat <- function(A, B) {
  # A: n × d, B: m × d
  # output: n × m matrix, D[i,j] = ||A_i - B_j||
  A2 <- rowSums(A^2)
  B2 <- rowSums(B^2)
  D2 <- outer(A2, B2, "+") - 2 * (A %*% t(B))
  sqrt(D2)
}

dist_bin_res <- l2_dist_mat(res_layered[1,,], bin_item)
dist_con_res <- l2_dist_mat(res_layered[2,,], con_item)

gamma1_true <- 0.5
eta_1 <- outer(alpha_1, beta_1, '+') - gamma1_true * dist_bin_res
Y_bin <- matrix(rbinom(length(plogis(eta_1)), size = 1, prob = as.vector(plogis(eta_1))), 
                nrow = nrow(eta_1), ncol = ncol(eta_1))

gamma2_true <- 1
sigma0_true <- 0.3
eta_2 <- outer(alpha_2, beta_2, '+') - gamma2_true * dist_con_res
Y_con <- eta_2 + rnorm(dim(eta_2)[1]*dim(eta_2)[2], mean = 0, sd = sigma0_true)
Y_con
convert_matrix <- function(mat) {
  breaks <- c(-Inf, 2, 3, 4, 5, Inf)
  labels <- c(1, 2, 3, 4, 5)
  
  matrix(
    as.numeric(cut(mat, breaks = breaks, labels = labels, right = FALSE)),
    nrow = nrow(mat),
    ncol = ncol(mat)
  )
}

Y_likert <- convert_matrix(Y_con)
Y_likert

true_var <- list(
  sigma_alpha1 = sigma_alpha1,
  alpha_1 = alpha_1,
  sigma_alpha2 = sigma_alpha2,
  alpha_2 = alpha_2, 
  tau_beta1 = tau_beta1,
  beta_1 = beta_1,
  tau_beta2 = tau_beta2,
  beta_2 = beta_2,
  sigma_res_global = sigma_res_global,
  res_global = res_global,
  sigma_res_1_2 = sigma_res_1_2,
  res_layered = res_layered, 
  sigma_item_bin = sigma_item_bin, 
  bin_item = bin_item,
  sigma_item_con = sigma_item_con,
  con_item = con_item,
  gamma1_true = gamma1_true,
  gamma2_true = gamma2_true,
  sigma0_true = sigma0_true
)


fit <- lsirm_global_local_option1(
  Y_bin = Y_bin,
  Y_con = Y_likert,
  d     = 2,
  n_iter = 10000,
  burnin = 5000,
  thin   = 2,
  prop_sd = list(
    alpha1     = 0.80,
    beta1      = 0.65,
    alpha2     = 0.20,
    beta2      = 0.20,
    log_gamma1 = 0.30,
    log_gamma2 = 0.01,
    a1         = 0.70,
    a2         = 0.1,
    b1         = 0.70,
    b2         = 0.1
  )
)

fit$accept


# result reporting
################################################################################
# traceplot
################################################################################
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$z)[2]){
  for(j in 1:dim(fit$samples$z)[3]){
    ts.plot(fit$samples$z[,i,j], main = paste0('z: ', i, '_',j))
    # abline(h = true_var$res_global[i,j], col = "red", lwd = 2)
  }
}

for(i in 1:dim(fit$samples$a1)[2]){
  for(j in 1:dim(fit$samples$a1)[3]){
    ts.plot(fit$samples$a1[,i,j], main = paste0('a1: ', i, '_',j))
    # abline(h = true_var$res_layered[1,i,j], col = "red", lwd = 2)
  }
}

for(i in 1:dim(fit$samples$a2)[2]){
  for(j in 1:dim(fit$samples$a2)[3]){
    ts.plot(fit$samples$a2[,i,j], main = paste0('a2: ', i, '_',j))
    # abline(h = true_var$res_layered[2,i,j], col = "red", lwd = 2)
  }
}

for(i in 1:dim(fit$samples$b1)[2]){
  for(j in 1:dim(fit$samples$b1)[3]){
    ts.plot(fit$samples$b1[,i,j], main = paste0('b1: ', i, '_',j))
    # abline(h = true_var$bin_item[i,j], col = "red", lwd = 2)
  }
}

for(i in 1:dim(fit$samples$b2)[2]){
  for(j in 1:dim(fit$samples$b2)[3]){
    ts.plot(fit$samples$b2[,i,j], main = paste0('b2: ', i, '_',j))
    # abline(h = true_var$con_item[i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta1)[2]){
  ts.plot(fit$samples$beta1[,i], main = paste0('beta1_',i))
  abline(h = true_var$beta_1[i], col = "red", lwd = 2)
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha1)[2]){
  ts.plot(fit$samples$alpha1[,i], main = paste0('alpha1_',i))
  abline(h = true_var$alpha_1[i], col = "red", lwd = 2)
}


#-------------------------------------------------------------------------------
# 지금 전체적으로 layer 2 의 additive term 이 과대추정되고 있는 상황임

# 불안정함/수렴에 시간이 오래 걸리는 것 같다. - true 값 복원 잘 못함
# - beta2: difficulty of continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta2)[2]){
  ts.plot(fit$samples$beta2[,i], main = paste0('beta2_',i))
  abline(h = true_var$beta_2[i], col = "red", lwd = 2)
}
true_var$beta_2
# 불안정함/수렴에 시간이 오래 걸리는 것 같다. - true 값 복원 잘 못함
# - alpha2: ability of resonpondent - continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha2)[2]){
  ts.plot(fit$samples$alpha2[,i], main = paste0('alpha2_',i))
  abline(h = true_var$alpha_2[i], col = "red", lwd = 2)
}
true_var$alpha_2

# 혹시 alpha2 와 beta2 간의 identifiability 가 있는 것은 아닐까?
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta2)[2]){
  ts.plot(fit$samples$beta2[,i] + fit$samples$alpha2[,i], main = paste0('alpha2 + beta2_',i))
  abline(h = true_var$beta_2[i] + true_var$alpha_2[i], col = "red", lwd = 2)
}

#-------------------------------------------------------------------------------
# 분산의 경우 과소추정됨
# 0.5 그러나 likert 척도로 바꾸는 과정에서 분산이 과소추정되었을 수 있다
ts.plot(fit$samples$sigma0_sq, main = 'sigma0_sq')
abline(h = true_var$sigma0_true, col = "red", lwd = 2)

# gamma1 은 과소추정
ts.plot(exp(fit$samples$log_gamma1), main = 'gamma1')
abline(h = true_var$gamma1_true, col = "red", lwd = 2)

# gamma2 는 추정값이 유사하지만 불안정
ts.plot(exp(fit$samples$log_gamma2), main = 'gamma2')
abline(h = true_var$gamma2_true, col = "red", lwd = 2)

ts.plot(fit$samples$sigma_alpha1_sq, main = 'sigma_alpha1_sq')
abline(h = true_var$sigma_alpha1, col = "red", lwd = 2)
ts.plot(log(fit$samples$sigma_alpha1_sq), main = 'sigma_alpha1_sq log transformed')
abline(h = log(true_var$sigma_alpha1), col = "red", lwd = 2)

# 불안정 - 과소추정
ts.plot(fit$samples$sigma_alpha2_sq, main = 'sigma_alpha2_sq')
abline(h = true_var$sigma_alpha2, col = "red", lwd = 2)
ts.plot(log(fit$samples$sigma_alpha2_sq), main = 'sigma_alpha2_sq log transformed')
abline(h = log(true_var$sigma_alpha2), col = "red", lwd = 2)

# 불안정
ts.plot(fit$samples$tau_beta1_sq, main = 'tau_beta1_sq')
abline(h = true_var$tau_beta1, col = "red", lwd = 2)
ts.plot(log(fit$samples$tau_beta1_sq), main = 'tau_beta1_sq log transformed')
abline(h = log(true_var$tau_beta1), col = "red", lwd = 2)

# 과대추정 / 추정 불안정
ts.plot(fit$samples$tau_beta2_sq, main = 'tau_beta2_sq')
abline(h = true_var$tau_beta2, col = "red", lwd = 2)
ts.plot(log(fit$samples$tau_beta2_sq), main = 'tau_beta2_sq log transformed')
abline(h = log(true_var$tau_beta2), col = "red", lwd = 2)


################################################################################
# latent position visualization
################################################################################
dim(fit$samples$z)
par(mfrow =c(2,2))

plot(colMeans(fit$samples$z, dims = 1))
plot(colMeans(fit$samples$a1, dims = 1), col = "red")
plot(colMeans(fit$samples$b1, dims = 1), col = "blue")
plot(colMeans(fit$samples$a2, dims = 1), col = "red")
plot(colMeans(fit$samples$b2, dims = 1), col = "blue")


# simulation 결과
# sampling setting
# n_iter = 100000,
# burnin = 10000,
# thin   = 5,



################################################################################
# real data
################################################################################
source("/Users/todoo/Desktop/학교/대학원/Research/joint_LSM/KSAH_dataprocessing.R")
test_PHQ<-read.csv("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM/data/EFA_data_Inhan/PHQ/data.csv")
test_PHQ
Y_bin <- as.matrix(test_PHQ[1:30, 9:17])
Y_likert <- test_PHQ[1:30, 36:47]
Y_bin <- ifelse(Y_bin <= 1, 0, 1)

fit <- lsirm_global_local_option1(
  Y_bin = Y_bin,
  Y_con = Y_likert_5,
  d     = 2,
  n_iter = 10000,
  burnin = 5000,
  thin   = 5,
  prop_sd = list(
    alpha1     = 0.70,
    beta1      = 0.90,
    alpha2     = 0.40,
    beta2      = 0.30,
    log_gamma1 = 0.30,
    log_gamma2 = 0.10,
    a1         = 0.70,
    a2         = 0.50,
    b1         = 0.70,
    b2         = 0.50
  )
)

fit$accept


# result reporting
################################################################################
# traceplot
################################################################################
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$z)[2]){
  for(j in 1:dim(fit$samples$z)[3]){
    ts.plot(fit$samples$z[,i,j], main = paste0('z: ', i, '_',j))
    # abline(h = true_var$res_global[i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a1)[2]){
  for(j in 1:dim(fit$samples$a1)[3]){
    ts.plot(fit$samples$a1[,i,j], main = paste0('a1: ', i, '_',j))
    # abline(h = true_var$res_layered[1,i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a2)[2]){
  for(j in 1:dim(fit$samples$a2)[3]){
    ts.plot(fit$samples$a2[,i,j], main = paste0('a2: ', i, '_',j))
    # abline(h = true_var$res_layered[2,i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b1)[2]){
  for(j in 1:dim(fit$samples$b1)[3]){
    ts.plot(fit$samples$b1[,i,j], main = paste0('b1: ', i, '_',j))
    # abline(h = true_var$bin_item[i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b2)[2]){
  for(j in 1:dim(fit$samples$b2)[3]){
    ts.plot(fit$samples$b2[,i,j], main = paste0('b2: ', i, '_',j))
    # abline(h = true_var$con_item[i,j], col = "red", lwd = 2)
  }
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta1)[2]){
  ts.plot(fit$samples$beta1[,i], main = paste0('beta1_',i))
  # abline(h = true_var$beta_1[i], col = "red", lwd = 2)
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha1)[2]){
  ts.plot(fit$samples$alpha1[,i], main = paste0('alpha1_',i))
  # abline(h = true_var$alpha_1[i], col = "red", lwd = 2)
}
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta1)[2]){
#   for(j in 1:dim(fit$samples$alpha1)[2]){
#     ts.plot(fit$samples$beta1[,i] + fit$samples$alpha1[,j], main = paste0('alpha1 + beta1_',i, '_', j))
#   }
# }

#-------------------------------------------------------------------------------
# 지금 전체적으로 layer 2 의 additive term 이 과대추정되고 있는 상황임

# 불안정함/수렴에 시간이 오래 걸리는 것 같다. - true 값 복원 잘 못함
# - beta2: difficulty of continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta2)[2]){
  ts.plot(fit$samples$beta2[,i], main = paste0('beta2_',i))
  # abline(h = true_var$beta_2[i], col = "red", lwd = 2)
}

# 불안정함/수렴에 시간이 오래 걸리는 것 같다. - true 값 복원 잘 못함
# - alpha2: ability of resonpondent - continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha2)[2]){
  ts.plot(fit$samples$alpha2[,i], main = paste0('alpha2_',i))
  # abline(h = true_var$alpha_2[i], col = "red", lwd = 2)
}


# # 혹시 alpha2 와 beta2 간의 identifiability 가 있는 것은 아닐까?
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta2)[2]){
#   for(j in 1:dim(fit$samples$alpha2)[2]){
#     ts.plot(fit$samples$beta2[,i] + fit$samples$alpha2[,j], main = paste0('alpha2 + beta2_',i, '_', j))
#   }
# }

#-------------------------------------------------------------------------------
# 분산의 경우 과소추정됨
# 0.5 그러나 likert 척도로 바꾸는 과정에서 분산이 과소추정되었을 수 있다
ts.plot(fit$samples$sigma0_sq, main = 'sigma0_sq')
# abline(h = true_var$sigma0_true, col = "red", lwd = 2)

ts.plot(fit$samples$sigma1_sq, main = 'sigma1_sq')

# gamma1 은 과소추정
ts.plot(exp(fit$samples$log_gamma1), main = 'gamma1')
# abline(h = true_var$gamma1_true, col = "red", lwd = 2)

# gamma2 는 추정값이 유사하지만 불안정
ts.plot(exp(fit$samples$log_gamma2), main = 'gamma2')
# abline(h = true_var$gamma2_true, col = "red", lwd = 2)

ts.plot(fit$samples$sigma_alpha1_sq, main = 'sigma_alpha1_sq')
# abline(h = true_var$sigma_alpha1, col = "red", lwd = 2)
ts.plot(log(fit$samples$sigma_alpha1_sq), main = 'sigma_alpha1_sq log transformed')
# abline(h = log(true_var$sigma_alpha1), col = "red", lwd = 2)

# 불안정 - 과소추정
ts.plot(fit$samples$sigma_alpha2_sq, main = 'sigma_alpha2_sq')
# abline(h = true_var$sigma_alpha2, col = "red", lwd = 2)
ts.plot(log(fit$samples$sigma_alpha2_sq), main = 'sigma_alpha2_sq log transformed')
# abline(h = log(true_var$sigma_alpha2), col = "red", lwd = 2)

# 불안정
ts.plot(fit$samples$tau_beta1_sq, main = 'tau_beta1_sq')
# abline(h = true_var$tau_beta1, col = "red", lwd = 2)
ts.plot(log(fit$samples$tau_beta1_sq), main = 'tau_beta1_sq log transformed')
# abline(h = log(true_var$tau_beta1), col = "red", lwd = 2)

# 과대추정 / 추정 불안정
ts.plot(fit$samples$tau_beta2_sq, main = 'tau_beta2_sq')
# abline(h = true_var$tau_beta2, col = "red", lwd = 2)
ts.plot(log(fit$samples$tau_beta2_sq), main = 'tau_beta2_sq log transformed')
# abline(h = log(true_var$tau_beta2), col = "red", lwd = 2)


################################################################################
# latent position visualization
################################################################################
# response
z_mean  <- colMeans(fit$samples$z,  dims = 1)[1:30,]
a1_mean <- colMeans(fit$samples$a1, dims = 1)[1:30,]
a2_mean <- colMeans(fit$samples$a2, dims = 1)[1:30,]

par(mfrow = c(1,1))
base::plot(
  z_mean,
  pch = 21, col = "black", bg = "gray80",
  main = "Latent positions: z (gray), a1 (red), a2 (orange)",
  ylim = c(-2,2),
  xlim = c(-2,2)
)

points(a1_mean, pch = 21, col = "red4", bg = "red")
points(a2_mean, pch = 21, col = "darkorange4", bg = "orange")

# --- 숫자 라벨 추가 ---

text(z_mean,  labels = 1:dim(z_mean)[1],  pos = 4, cex = 0.7)
text(a1_mean, labels = 1:dim(a1_mean)[1], pos = 4, cex = 0.7, col = "red4")
text(a2_mean, labels = 1:dim(a2_mean)[1], pos = 4, cex = 0.7, col = "darkorange4")

legend(
  "topright",
  legend = c("z: global respondent", "a1: binary respondent", "a2: continuous respondent"),
  pch = 21,
  pt.bg = c("gray80", "red", "orange"),
  col = c("black", "red4", "darkorange4"),
  bty = "n"
)

# item
b2_mean <- colMeans(fit$samples$b2, dims = 1)
b1_mean <- colMeans(fit$samples$b1, dims = 1)

base::plot(
  b2_mean,
  pch = 21, col = "navy", bg = "blue",
  main = "Item positions: b2 (blue), b1 (skyblue)"
)

points(b1_mean, pch = 21, col = "steelblue4", bg = "skyblue")

# --- 숫자 라벨 추가 ---
text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")
text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "steelblue4")

legend(
  "topright",
  legend = c("b2: continuous item", "b1: binary item"),
  pch = 21,
  pt.bg = c("blue", "skyblue"),
  col = c("navy", "steelblue4"),
  bty = "n"
)

# binary: item - respondent
base::plot(
  a1_mean,
  pch = 21, col = "red4", bg = "red",
  main = "binary: item - respondent"
)

points(b1_mean, pch = 21, col = "steelblue4", bg = "skyblue")

text(a1_mean, labels = 1:dim(a1_mean)[1], pos = 4, cex = 0.7, col = "red4")
text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "steelblue4")

legend(
  "topright",
  legend = c("a1: binary respondent", "b1: binary item"),
  pch = 21,
  pt.bg = c("red", "skyblue"),
  col = c("red4", "steelblue4"),
  bty = "n"
)

# continuous: item - respondent
base::plot(
  a2_mean,
  pch = 21, col = "darkorange4", bg = "orange",
  main = "continuous: item - respondent"
)

points(b2_mean, pch = 21, col = "navy", bg = "blue")

text(a2_mean, labels = 1:dim(a2_mean)[1], pos = 4, cex = 0.7, col = "darkorange4")
text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")

legend(
  "topright",
  legend = c("a2: continuous respondent", "b2: continuous item"),
  pch = 21,
  pt.bg = c("orange", "blue"),
  col = c("darkorange4", "navy"),
  bty = "n"
)

