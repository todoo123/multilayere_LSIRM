library(vegan)

# =========================================================
# Helpers for ordinal (LSGRM layer)
# =========================================================
inv_logit <- function(x) 1 / (1 + exp(-x))

build_thresholds <- function(u, delta_vec) {
  # delta_vec length = (K-2) corresponding to k=2..K-1
  Kminus1 <- 1 + length(delta_vec)
  b <- numeric(Kminus1)
  b[1] <- u
  if (Kminus1 >= 2) {
    for (k in 2:Kminus1) {
      b[k] <- b[k - 1] + exp(delta_vec[k - 1])
    }
  }
  b
}

log_p_ordinal_single <- function(y, eta, b_vec) {
  # eta = alpha_j - gamma * d_ji   (여기서는 "beta 없음" 버전)
  # b_vec length = K-1, thresholds for P(Y >= k+1)
  Kminus1 <- length(b_vec)
  K <- Kminus1 + 1L
  C <- inv_logit(eta - b_vec)
  
  if (y == 1L) {
    p <- 1 - C[1]
  } else if (y == K) {
    p <- C[Kminus1]
  } else {
    p <- C[y - 1L] - C[y]
  }
  if (is.na(p) || p <= 0) return(-Inf)
  log(p)
}

loglik_item_i_ord <- function(y_col, alpha, dist_col, b_vec, gamma) {
  # y_col length n
  eta <- alpha - gamma * dist_col
  ll <- 0
  for (j in seq_along(y_col)) {
    ll <- ll + log_p_ordinal_single(y = y_col[j], eta = eta[j], b_vec = b_vec)
  }
  ll
}

loglik_resp_j_ord <- function(y_row, alpha_j, dist_row, b_list, gamma) {
  # y_row length P4
  eta <- alpha_j - gamma * dist_row
  ll <- 0
  for (i in seq_along(y_row)) {
    ll <- ll + log_p_ordinal_single(
      y     = y_row[i],
      eta   = eta[i],
      b_vec = b_list[[i]]
    )
  }
  ll
}

# =========================================================
# Shared-Position 4-layer LSIRM + LSGRM ordering layer
# - shared: alpha_i, gamma, a_i
# - layer1 (bin): beta1, b1
# - layer2 (con): beta2, b2, sigma0_sq
# - layer3 (cnt): beta3, b3, kappa
# - layer4 (ord): b4 (item pos), thresholds (u, delta) => b_list
# =========================================================
lsirm_sharedpos_layer4_lsgrm <- function(
    Y_bin,              # n x P1, 0/1
    Y_con,              # n x P2, continuous
    Y_cnt,              # n x P3, 0,1,2,...
    Y_ord,              # n x P4, integer in 1..K
    d       = 2,
    n_iter  = 5000,
    burnin  = 1000,
    thin    = 5,
    
    hyper   = list(
      # shared alpha prior variance
      a_sigma = 1, b_sigma = 0.1,
      
      # beta variance priors (layers 1..3)
      a_tau1  = 1, b_tau1  = 0.1,
      a_tau2  = 1, b_tau2  = 0.1,
      a_tau3  = 1, b_tau3  = 0.1,
      
      # continuous layer residual variance
      a_sigma0 = 1, b_sigma0 = 1,
      
      # shared log-gamma prior
      mu_log_gamma = 0, sd_log_gamma = 1,
      
      # count layer dispersion prior
      mu_log_kappa = 0, sd_log_kappa = 1,
      
      # ordinal thresholds priors
      mu_u = 0, sd_u = 1,
      mu_delta = 0, sd_delta = 1
    ),
    
    prop_sd = list(
      # shared
      alpha      = 0.10,
      log_gamma  = 0.05,
      a          = 0.10,
      
      # layer1..3
      beta1      = 0.10,
      beta2      = 0.10,
      beta3      = 0.10,
      b1         = 0.10,
      b2         = 0.10,
      b3         = 0.10,
      log_kappa  = 0.05,
      
      # layer4 (ordinal)
      b4         = 0.10,
      u          = 0.10,
      delta      = 0.10
    ),
    
    init    = NULL,
    verbose = TRUE
) {
  
  # ----- stable log(1+exp(x)) -----
  log1pexp <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
  
  # ----- data -----
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  Y_cnt <- as.matrix(Y_cnt)
  
  Y_ord <- as.matrix(Y_ord)
  storage.mode(Y_ord) <- "integer"
  
  if (sum(is.na(Y_bin)) + sum(is.na(Y_con)) + sum(is.na(Y_cnt)) + sum(is.na(Y_ord)) > 0) {
    stop("Missing values (NA) are not allowed.")
  }
  
  n  <- nrow(Y_bin)
  P1 <- ncol(Y_bin)
  P2 <- ncol(Y_con)
  P3 <- ncol(Y_cnt)
  P4 <- ncol(Y_ord)
  
  if (nrow(Y_con) != n || nrow(Y_cnt) != n || nrow(Y_ord) != n) stop("All layers must have same n.")
  
  # ordinal K
  K <- max(Y_ord)
  if (min(Y_ord) < 1 || K < 2) stop("Y_ord must be integers in 1..K with K>=2.")
  Kminus1 <- K - 1L
  
  Z_con <- Y_con
  
  # ----- init -----
  if (is.null(init)) {
    alpha <- rnorm(n, 0, 0.1)
    
    beta1 <- rnorm(P1, 0, 0.1)
    beta2 <- rnorm(P2, 0, 0.1)
    beta3 <- rnorm(P3, 0, 0.1)
    
    a  <- matrix(rnorm(n * d, 0, 0.5), n, d)
    b1 <- matrix(rnorm(P1 * d, 0, 0.5), P1, d)
    b2 <- matrix(rnorm(P2 * d, 0, 0.5), P2, d)
    b3 <- matrix(rnorm(P3 * d, 0, 0.5), P3, d)
    b4 <- matrix(rnorm(P4 * d, 0, 0.5), P4, d)
    
    log_gamma <- hyper$mu_log_gamma
    gamma     <- exp(log_gamma)
    
    log_kappa <- hyper$mu_log_kappa
    kappa     <- exp(log_kappa)
    
    sigma_alpha_sq <- 1
    tau_beta1_sq   <- 1
    tau_beta2_sq   <- 1
    tau_beta3_sq   <- 1
    sigma0_sq      <- 1
    
    # ordinal thresholds
    u <- rnorm(P4, hyper$mu_u, hyper$sd_u)
    delta <- if (K > 2) matrix(rnorm(P4 * (K-2), hyper$mu_delta, hyper$sd_delta), P4, (K-2)) else matrix(numeric(P4*0), P4, 0)
    
  } else {
    alpha <- init$alpha
    
    beta1 <- init$beta1
    beta2 <- init$beta2
    beta3 <- init$beta3
    
    a  <- init$a
    b1 <- init$b1
    b2 <- init$b2
    b3 <- init$b3
    b4 <- init$b4
    
    log_gamma <- init$log_gamma
    gamma     <- exp(log_gamma)
    
    log_kappa <- init$log_kappa
    kappa     <- exp(log_kappa)
    
    sigma_alpha_sq <- init$sigma_alpha_sq
    tau_beta1_sq   <- init$tau_beta1_sq
    tau_beta2_sq   <- init$tau_beta2_sq
    tau_beta3_sq   <- init$tau_beta3_sq
    sigma0_sq      <- init$sigma0_sq
    
    u <- init$u
    delta <- init$delta
  }
  
  # build ordinal thresholds list
  b_list <- vector("list", P4)
  for (j in 1:P4) {
    b_list[[j]] <- build_thresholds(u[j], if (K > 2) delta[j, ] else numeric(0))
  }
  
  # ----- distance helper -----
  dist_mat <- function(A, B) {
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
  D4 <- dist_mat(a, b4)
  
  # ----- storage -----
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
    
    a  = array(NA, c(n_save, n, d)),
    b1 = array(NA, c(n_save, P1, d)),
    b2 = array(NA, c(n_save, P2, d)),
    b3 = array(NA, c(n_save, P3, d)),
    b4 = array(NA, c(n_save, P4, d)),
    
    # ordinal thresholds
    u     = array(NA, c(n_save, P4)),
    delta = if (K > 2) array(NA, c(n_save, P4, K-2)) else NULL,
    thr   = array(NA, c(n_save, P4, K-1)),
    
    # compatibility outputs (keep old fields)
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
    b3        = rep(0, P3),
    b4        = rep(0, P4),
    thr       = rep(0, P4)
  )
  
  save_idx <- 0
  
  # =========================================================
  # MCMC
  # =========================================================
  for (iter in 1:n_iter) {
    
    if (verbose && iter %% 500 == 0) cat("Iter:", iter, "/", n_iter, "\n")
    
    # -------------------------------------------------------
    # 1) alpha_i (shared) : MH using ALL 4 layers
    # -------------------------------------------------------
    for (i in 1:n) {
      # current ll
      eta1_i <- alpha[i] + beta1 - gamma * D1[i, ]
      ll1 <- sum(Y_bin[i, ] * eta1_i - log1pexp(eta1_i))
      
      mu2_i <- alpha[i] + beta2 - gamma * D2[i, ]
      r2 <- Z_con[i, ] - mu2_i
      ll2 <- -0.5 / sigma0_sq * sum(r2^2)
      
      mu3_i <- exp(alpha[i] + beta3 - gamma * D3[i, ])
      ll3 <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_i, log = TRUE))
      
      # ordinal layer (no beta): alpha_i - gamma * D4[i,]
      ll4 <- loglik_resp_j_ord(Y_ord[i, ], alpha[i], D4[i, ], b_list, gamma)
      
      lp_cur <- -0.5 * alpha[i]^2 / sigma_alpha_sq
      logpost_cur <- ll1 + ll2 + ll3 + ll4 + lp_cur
      
      # proposal
      a_prop <- rnorm(1, alpha[i], prop_sd$alpha)
      
      eta1_p <- a_prop + beta1 - gamma * D1[i, ]
      ll1_p <- sum(Y_bin[i, ] * eta1_p - log1pexp(eta1_p))
      
      mu2_p <- a_prop + beta2 - gamma * D2[i, ]
      r2p <- Z_con[i, ] - mu2_p
      ll2_p <- -0.5 / sigma0_sq * sum(r2p^2)
      
      mu3_p <- exp(a_prop + beta3 - gamma * D3[i, ])
      ll3_p <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_p, log = TRUE))
      
      ll4_p <- loglik_resp_j_ord(Y_ord[i, ], a_prop, D4[i, ], b_list, gamma)
      
      lp_p <- -0.5 * a_prop^2 / sigma_alpha_sq
      logpost_p <- ll1_p + ll2_p + ll3_p + ll4_p + lp_p
      
      log_acc <- logpost_p - logpost_cur
      if (log(runif(1)) < log_acc) {
        alpha[i] <- a_prop
        accept$alpha[i] <- accept$alpha[i] + 1
      }
    }
    
    # -------------------------------------------------------
    # 2) beta^(l) (layers 1..3) : MH (layer-specific)
    # -------------------------------------------------------
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
    
    # -------------------------------------------------------
    # 3) log_gamma (shared) : MH using ALL 4 layers
    # -------------------------------------------------------
    eta1 <- outer(alpha, beta1, "+") - gamma * D1
    ll1 <- sum(Y_bin * eta1 - log1pexp(eta1))
    
    mu2 <- outer(alpha, beta2, "+") - gamma * D2
    r2 <- Z_con - mu2
    ll2 <- -0.5 / sigma0_sq * sum(r2^2)
    
    mu3 <- exp(outer(alpha, beta3, "+") - gamma * D3)
    ll3 <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3, log = TRUE))
    
    # ordinal (sum over items using column-wise function)
    ll4 <- 0
    for (j in 1:P4) {
      ll4 <- ll4 + loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_list[[j]], gamma)
    }
    
    lp_cur <- dnorm(log_gamma, mean = hyper$mu_log_gamma, sd = hyper$sd_log_gamma, log = TRUE)
    logpost_cur <- ll1 + ll2 + ll3 + ll4 + lp_cur
    
    log_gamma_prop <- rnorm(1, log_gamma, prop_sd$log_gamma)
    gamma_prop <- exp(log_gamma_prop)
    
    eta1p <- outer(alpha, beta1, "+") - gamma_prop * D1
    ll1p <- sum(Y_bin * eta1p - log1pexp(eta1p))
    
    mu2p <- outer(alpha, beta2, "+") - gamma_prop * D2
    r2p <- Z_con - mu2p
    ll2p <- -0.5 / sigma0_sq * sum(r2p^2)
    
    mu3p <- exp(outer(alpha, beta3, "+") - gamma_prop * D3)
    ll3p <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3p, log = TRUE))
    
    ll4p <- 0
    for (j in 1:P4) {
      ll4p <- ll4p + loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_list[[j]], gamma_prop)
    }
    
    lp_p <- dnorm(log_gamma_prop, mean = hyper$mu_log_gamma, sd = hyper$sd_log_gamma, log = TRUE)
    logpost_p <- ll1p + ll2p + ll3p + ll4p + lp_p
    
    log_acc <- logpost_p - logpost_cur
    if (log(runif(1)) < log_acc) {
      log_gamma <- log_gamma_prop
      gamma <- gamma_prop
      accept$log_gamma <- accept$log_gamma + 1
    }
    
    # -------------------------------------------------------
    # 4) shared respondent positions a_i : MH using ALL 4 layers
    # -------------------------------------------------------
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
      
      ll4 <- loglik_resp_j_ord(Y_ord[i, ], alpha[i], D4[i, ], b_list, gamma)
      
      lp_cur <- -0.5 * sum(a_cur^2)
      logpost_cur <- ll1 + ll2 + ll3 + ll4 + lp_cur
      
      # proposal
      a_prop <- a_cur + rnorm(d, 0, prop_sd$a)
      
      # recompute only row distances
      d1p <- numeric(P1); for (j in 1:P1) d1p[j] <- sqrt(sum((a_prop - b1[j, ])^2))
      d2p <- numeric(P2); for (j in 1:P2) d2p[j] <- sqrt(sum((a_prop - b2[j, ])^2))
      d3p <- numeric(P3); for (j in 1:P3) d3p[j] <- sqrt(sum((a_prop - b3[j, ])^2))
      d4p <- numeric(P4); for (j in 1:P4) d4p[j] <- sqrt(sum((a_prop - b4[j, ])^2))
      
      eta1_p <- alpha[i] + beta1 - gamma * d1p
      ll1p <- sum(Y_bin[i, ] * eta1_p - log1pexp(eta1_p))
      
      mu2_p <- alpha[i] + beta2 - gamma * d2p
      r2p <- Z_con[i, ] - mu2_p
      ll2p <- -0.5 / sigma0_sq * sum(r2p^2)
      
      mu3_p <- exp(alpha[i] + beta3 - gamma * d3p)
      ll3p <- sum(dnbinom(Y_cnt[i, ], size = 1 / kappa, mu = mu3_p, log = TRUE))
      
      ll4p <- loglik_resp_j_ord(Y_ord[i, ], alpha[i], d4p, b_list, gamma)
      
      lp_p <- -0.5 * sum(a_prop^2)
      logpost_p <- ll1p + ll2p + ll3p + ll4p + lp_p
      
      log_acc <- logpost_p - logpost_cur
      if (log(runif(1)) < log_acc) {
        a[i, ] <- a_prop
        D1[i, ] <- d1p
        D2[i, ] <- d2p
        D3[i, ] <- d3p
        D4[i, ] <- d4p
        accept$a[i] <- accept$a[i] + 1
      }
    }
    
    # -------------------------------------------------------
    # 5) item positions b^(l)_j : MH (per layer), prior N(0,I)
    # -------------------------------------------------------
    # b1
    for (j in 1:P1) {
      b_cur <- b1[j, ]
      eta_cur <- alpha + beta1[j] - gamma * D1[, j]
      ll_cur <- sum(Y_bin[, j] * eta_cur - log1pexp(eta_cur))
      lp_cur <- -0.5 * sum(b_cur^2)
      
      b_prop <- b_cur + rnorm(d, 0, prop_sd$b1)
      d1p <- numeric(n); for (i in 1:n) d1p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
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
      d2p <- numeric(n); for (i in 1:n) d2p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
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
      d3p <- numeric(n); for (i in 1:n) d3p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
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
    
    # b4 (ordinal)
    for (j in 1:P4) {
      b_cur <- b4[j, ]
      ll_cur <- loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_list[[j]], gamma)
      lp_cur <- -0.5 * sum(b_cur^2)
      
      b_prop <- b_cur + rnorm(d, 0, prop_sd$b4)
      d4p <- numeric(n); for (i in 1:n) d4p[i] <- sqrt(sum((a[i, ] - b_prop)^2))
      ll_p <- loglik_item_i_ord(Y_ord[, j], alpha, d4p, b_list[[j]], gamma)
      lp_p <- -0.5 * sum(b_prop^2)
      
      log_acc <- (ll_p + lp_p) - (ll_cur + lp_cur)
      if (log(runif(1)) < log_acc) {
        b4[j, ] <- b_prop
        D4[, j] <- d4p
        accept$b4[j] <- accept$b4[j] + 1
      }
    }
    
    # -------------------------------------------------------
    # 6) ordinal thresholds (u, delta) item-wise MH + Jacobian
    # -------------------------------------------------------
    for (j in 1:P4) {
      u_prop_j <- u[j] + rnorm(1, 0, prop_sd$u)
      if (K > 2) {
        delta_prop_j <- delta[j, ] + rnorm(K-2, 0, prop_sd$delta)
      } else {
        delta_prop_j <- numeric(0)
      }
      
      b_prop_j <- build_thresholds(u_prop_j, delta_prop_j)
      
      # likelihood change for column j only
      ll_cur <- loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_list[[j]], gamma)
      ll_p   <- loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_prop_j, gamma)
      
      # priors
      lp_cur <- dnorm(u[j], hyper$mu_u, hyper$sd_u, log = TRUE)
      lp_p   <- dnorm(u_prop_j, hyper$mu_u, hyper$sd_u, log = TRUE)
      
      if (K > 2) {
        lp_cur <- lp_cur + sum(dnorm(delta[j, ], hyper$mu_delta, hyper$sd_delta, log = TRUE))
        lp_p   <- lp_p   + sum(dnorm(delta_prop_j, hyper$mu_delta, hyper$sd_delta, log = TRUE))
      }
      
      # Jacobian for exp-increments: sum(delta)
      lj_cur <- if (K > 2) sum(delta[j, ]) else 0
      lj_p   <- if (K > 2) sum(delta_prop_j) else 0
      
      log_acc <- (ll_p - ll_cur) + (lp_p - lp_cur) + (lj_p - lj_cur)
      
      if (log(runif(1)) < log_acc) {
        u[j] <- u_prop_j
        if (K > 2) delta[j, ] <- delta_prop_j
        b_list[[j]] <- b_prop_j
        accept$thr[j] <- accept$thr[j] + 1
      }
    }
    
    # -------------------------------------------------------
    # 7) log_kappa (count layer only)
    # -------------------------------------------------------
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
    
    # -------------------------------------------------------
    # 8) Gibbs: sigma0_sq, sigma_alpha_sq, tau_beta*_sq
    # -------------------------------------------------------
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
    
    # -------------------------------------------------------
    # 9) save
    # -------------------------------------------------------
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
      samples$b4[save_idx, , ] <- b4
      
      samples$u[save_idx, ] <- u
      if (K > 2) samples$delta[save_idx, , ] <- delta
      
      thr_mat <- matrix(NA, P4, K-1)
      for (j in 1:P4) thr_mat[j, ] <- b_list[[j]]
      samples$thr[save_idx, , ] <- thr_mat
      
      # compatibility copies
      samples$z[save_idx, , ]  <- a
      samples$a1[save_idx, , ] <- a
      samples$a2[save_idx, , ] <- a
      samples$a3[save_idx, , ] <- a
      
      # loglik (sum of 4 layers)
      eta1 <- outer(alpha, beta1, "+") - gamma * D1
      ll1 <- sum(Y_bin * eta1 - log1pexp(eta1))
      
      mu2 <- outer(alpha, beta2, "+") - gamma * D2
      r2  <- Z_con - mu2
      ll2 <- -0.5 * sum(r2^2) / sigma0_sq - 0.5 * n * P2 * log(2 * pi * sigma0_sq)
      
      mu3 <- exp(outer(alpha, beta3, "+") - gamma * D3)
      ll3 <- sum(dnbinom(Y_cnt, size = 1 / kappa, mu = mu3, log = TRUE))
      
      ll4 <- 0
      for (j in 1:P4) {
        ll4 <- ll4 + loglik_item_i_ord(Y_ord[, j], alpha, D4[, j], b_list[[j]], gamma)
      }
      
      samples$loglik[save_idx] <- ll1 + ll2 + ll3 + ll4
    }
  }
  
  # =========================================================
  # Procrustes matching (align a, b1, b2, b3, b4 together)
  # =========================================================
  if (requireNamespace("vegan", quietly = TRUE)) {
    
    loglik_save <- samples$loglik
    map_idx <- which.max(loglik_save)
    
    A_map  <- matrix(samples$a[map_idx, , ],  n,  d)
    B1_map <- matrix(samples$b1[map_idx, , ], P1, d)
    B2_map <- matrix(samples$b2[map_idx, , ], P2, d)
    B3_map <- matrix(samples$b3[map_idx, , ], P3, d)
    B4_map <- matrix(samples$b4[map_idx, , ], P4, d)
    
    X_map <- rbind(A_map, B1_map, B2_map, B3_map, B4_map)
    
    idx_A  <- 1:n
    idx_B1 <- (n + 1):(n + P1)
    idx_B2 <- (n + P1 + 1):(n + P1 + P2)
    idx_B3 <- (n + P1 + P2 + 1):(n + P1 + P2 + P3)
    idx_B4 <- (n + P1 + P2 + P3 + 1):(n + P1 + P2 + P3 + P4)
    
    for (ss in seq_len(n_save)) {
      if (ss == map_idx) next
      
      A_s  <- matrix(samples$a[ss, , ],  n,  d)
      B1_s <- matrix(samples$b1[ss, , ], P1, d)
      B2_s <- matrix(samples$b2[ss, , ], P2, d)
      B3_s <- matrix(samples$b3[ss, , ], P3, d)
      B4_s <- matrix(samples$b4[ss, , ], P4, d)
      
      Y_s <- rbind(A_s, B1_s, B2_s, B3_s, B4_s)
      
      fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
      Y_aligned <- predict(fit_all)
      
      samples$a[ss, , ]  <- Y_aligned[idx_A,  , drop = FALSE]
      samples$b1[ss, , ] <- Y_aligned[idx_B1, , drop = FALSE]
      samples$b2[ss, , ] <- Y_aligned[idx_B2, , drop = FALSE]
      samples$b3[ss, , ] <- Y_aligned[idx_B3, , drop = FALSE]
      samples$b4[ss, , ] <- Y_aligned[idx_B4, , drop = FALSE]
      
      samples$z[ss, , ]  <- samples$a[ss, , ]
      samples$a1[ss, , ] <- samples$a[ss, , ]
      samples$a2[ss, , ] <- samples$a[ss, , ]
      samples$a3[ss, , ] <- samples$a[ss, , ]
    }
    
  } else {
    warning("Package 'vegan' is not installed. Skipping Procrustes matching.")
  }
  
  # ----- acceptance rates -----
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
  accept$b4        <- accept$b4 / n_iter
  accept$thr       <- accept$thr / n_iter
  
  return(list(
    samples = samples,
    accept  = accept,
    hyper   = hyper,
    prop_sd = prop_sd,
    info    = list(n=n, P1=P1, P2=P2, P3=P3, P4=P4, K=K, d=d, burnin=burnin, thin=thin)
  ))
}


# 
# 
# 
# library(vegan)
# 
# # ---------------------------
# # 1) toy dimensions
# # ---------------------------
# n  <- 40
# P1 <- 6
# P2 <- 6
# P3 <- 6
# P4 <- 6
# d  <- 2
# K  <- 5          # ordinal categories: 1..K
# kappa_true <- 0.7
# 
# # ---------------------------
# # 2) true latent parameters
# # ---------------------------
# alpha_true <- rnorm(n, 0, 0.7)
# gamma_true <- 1.0
# 
# a_true  <- matrix(rnorm(n*d, 0, 1), n, d)
# b1_true <- matrix(rnorm(P1*d, 0, 1), P1, d)
# b2_true <- matrix(rnorm(P2*d, 0, 1), P2, d)
# b3_true <- matrix(rnorm(P3*d, 0, 1), P3, d)
# b4_true <- matrix(rnorm(P4*d, 0, 1), P4, d)
# 
# beta1_true <- rnorm(P1, 0, 0.5)
# beta2_true <- rnorm(P2, 0, 0.5)
# beta3_true <- rnorm(P3, 0, 0.5)
# 
# sigma0_true <- 0.7
# 
# # ordinal thresholds: u + cumulative exp(delta)
# u_true <- rnorm(P4, 0, 0.7)
# delta_true <- matrix(rnorm(P4*(K-2), 0, 0.4), P4, (K-2))
# 
# build_thresholds <- function(u, delta_vec) {
#   Kminus1 <- 1 + length(delta_vec)
#   b <- numeric(Kminus1)
#   b[1] <- u
#   if (Kminus1 >= 2) {
#     for (k in 2:Kminus1) b[k] <- b[k-1] + exp(delta_vec[k-1])
#   }
#   b
# }
# b_list_true <- lapply(1:P4, function(j) build_thresholds(u_true[j], delta_true[j, ]))
# 
# # distance matrices
# dist_mat <- function(A,B){
#   nA <- nrow(A); nB <- nrow(B)
#   D <- matrix(0, nA, nB)
#   for(i in 1:nA){
#     for(j in 1:nB){
#       D[i,j] <- sqrt(sum((A[i,]-B[j,])^2))
#     }
#   }
#   D
# }
# D1 <- dist_mat(a_true, b1_true)
# D2 <- dist_mat(a_true, b2_true)
# D3 <- dist_mat(a_true, b3_true)
# D4 <- dist_mat(a_true, b4_true)
# 
# inv_logit <- function(x) 1/(1+exp(-x))
# 
# # ---------------------------
# # 3) simulate each layer
# # ---------------------------
# 
# # layer1: Bernoulli
# eta1 <- outer(alpha_true, beta1_true, "+") - gamma_true * D1
# p1 <- inv_logit(eta1)
# Y_bin <- matrix(rbinom(n*P1, size=1, prob=as.vector(p1)), n, P1)
# 
# # layer2: Gaussian
# mu2 <- outer(alpha_true, beta2_true, "+") - gamma_true * D2
# Y_con <- mu2 + matrix(rnorm(n*P2, 0, sigma0_true), n, P2)
# 
# # layer3: NegBin (NB2: size=1/kappa, mu=exp(eta))
# eta3 <- outer(alpha_true, beta3_true, "+") - gamma_true * D3
# mu3 <- exp(eta3)
# Y_cnt <- matrix(rnbinom(n*P3, size=1/kappa_true, mu=as.vector(mu3)), n, P3)
# 
# # layer4: Ordinal (LSGRM)
# # eta4_ij = alpha_i - gamma*d_ij  (beta4 없음)
# log_p_ordinal_single <- function(y, eta, b_vec) {
#   Kminus1 <- length(b_vec)
#   K <- Kminus1 + 1L
#   C <- inv_logit(eta - b_vec)
#   if (y == 1L) p <- 1 - C[1]
#   else if (y == K) p <- C[Kminus1]
#   else p <- C[y-1L] - C[y]
#   if (is.na(p) || p <= 0) return(-Inf)
#   log(p)
# }
# draw_ordinal <- function(eta, b_vec){
#   # return y in 1..K
#   Kminus1 <- length(b_vec)
#   K <- Kminus1 + 1L
#   C <- inv_logit(eta - b_vec)  # length K-1, C[k]=P(Y>=k+1)
#   probs <- numeric(K)
#   probs[1] <- 1 - C[1]
#   if (K > 2) {
#     for (y in 2:(K-1)) probs[y] <- C[y-1] - C[y]
#   }
#   probs[K] <- C[K-1]
#   # safety
#   probs <- pmax(probs, 0); probs <- probs / sum(probs)
#   sample.int(K, size=1, prob=probs)
# }
# 
# Y_ord <- matrix(NA_integer_, n, P4)
# for(i in 1:n){
#   for(j in 1:P4){
#     eta4_ij <- alpha_true[i] - gamma_true * D4[i,j]
#     Y_ord[i,j] <- draw_ordinal(eta4_ij, b_list_true[[j]])
#   }
# }
# 
# # ---------------------------
# # 4) fit (small iter just to test it runs)
# # ---------------------------
# fit <- lsirm_sharedpos_layer4_lsgrm(
#   Y_bin = Y_bin,
#   Y_con = Y_con,
#   Y_cnt = Y_cnt,
#   Y_ord = Y_ord,
#   d = d,
#   n_iter = 400,
#   burnin = 100,
#   thin = 5,
#   verbose = TRUE
# )
# fit$accept
# # quick sanity checks
# str(fit$accept)
# cat("Saved draws:", dim(fit$samples$a)[1], "\n")
# cat("Mean gamma (post):", mean(exp(fit$samples$log_gamma)), "\n")
# 
