library(vegan)

# =========================================================
# Helpers (log-likelihood for graded response with latent space)
# =========================================================

inv_logit <- function(x) 1 / (1 + exp(-x))

# Build thresholds b_{i,1:(K-1)} from (u_i, delta_i2..delta_i(K-1))
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

# log p(Y=y | alpha, b_vec, gamma*dist) for a single observation
log_p_ordinal_single <- function(y, eta, b_vec) {
  # eta = alpha_j - gamma * d_ji
  # b_vec length = K-1, thresholds for P(Y >= k+1)
  # C_k = P(Y >= k+1) for k=1..K-1
  Kminus1 <- length(b_vec)
  K <- Kminus1 + 1L
  
  # cumulative probs
  C <- inv_logit(eta - b_vec)
  
  if (y == 1L) {
    p <- 1 - C[1]
  } else if (y == K) {
    p <- C[Kminus1]
  } else {
    # y in 2..K-1
    p <- C[y - 1L] - C[y]
  }
  
  # numerical safety
  if (is.na(p) || p <= 0) return(-Inf)
  log(p)
}

# log-likelihood for one item column i (uses all respondents)
loglik_item_i <- function(y_col, alpha, dist_col, b_vec, gamma) {
  # y_col length N, dist_col length N
  eta <- alpha - gamma * dist_col
  ll <- 0
  for (j in seq_along(y_col)) {
    ll <- ll + log_p_ordinal_single(y = y_col[j], eta = eta[j], b_vec = b_vec)
  }
  ll
}
# log-likelihood for one respondent row j (uses all items)
loglik_resp_j <- function(y_row, alpha_j, dist_row, b_list, gamma) {
  # y_row length P (respondent j's responses across items)
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

# full log-likelihood
loglik_full <- function(Y, alpha, A, B, b_list, gamma) {
  N <- nrow(Y); P <- ncol(Y)
  # precompute distances NxP
  # dist_{ji} = ||A_j - B_i||
  ll <- 0
  for (i in 1:P) {
    dist_col <- sqrt(rowSums((A - matrix(B[i, ], N, ncol(A), byrow = TRUE))^2))
    ll <- ll + loglik_item_i(Y[, i], alpha, dist_col, b_list[[i]], gamma)
  }
  ll
}

# =========================================================
# Main: unilayer LSGRM (graded response) - MH within Gibbs
# =========================================================
lsgrm_basic <- function(
    Y_ord,                      # n x P, integer in 1..K
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    
    hyper = list(
      # alpha variance prior
      a_alpha = 1, b_alpha = 0.1,   # Inv-Gamma for sigma_alpha_sq
      
      # threshold priors (u, delta)
      mu_u = 0, sd_u = 1,
      mu_delta = 0, sd_delta = 1,
      
      # log-gamma prior
      mu_log_gamma = 0, sd_log_gamma = 1
    ),
    
    # proposals: diagonal RWMH
    prop_sd = list(
      alpha = 0.15,
      a     = 0.10,
      b     = 0.10,
      log_gamma = 0.05,
      
      # threshold block diag scales
      u     = 0.10,
      delta = 0.10
    ),
    
    init = NULL,
    verbose = TRUE
){
  Y <- as.matrix(Y_ord)
  storage.mode(Y) <- "integer"
  
  N <- nrow(Y)
  P <- ncol(Y)
  
  # infer K from data
  K <- max(Y, na.rm = TRUE)
  if (min(Y, na.rm = TRUE) < 1 || K < 2) stop("Y_ord must be integers in 1..K with K>=2")
  
  # dimensions
  Kminus1 <- K - 1L
  if (Kminus1 < 1) stop("K must be >= 2")
  
  # --------------------------
  # Initialize parameters
  # --------------------------
  if (is.null(init)) init <- list()
  
  alpha <- if (!is.null(init$alpha)) as.numeric(init$alpha) else rnorm(N, 0, 0.5)
  sigma_alpha_sq <- if (!is.null(init$sigma_alpha_sq)) init$sigma_alpha_sq else 1.0
  
  A <- if (!is.null(init$a)) matrix(init$a, N, d) else matrix(rnorm(N*d, 0, 1), N, d)
  B <- if (!is.null(init$b)) matrix(init$b, P, d) else matrix(rnorm(P*d, 0, 1), P, d)
  
  log_gamma <- if (!is.null(init$log_gamma)) init$log_gamma else hyper$mu_log_gamma
  gamma <- exp(log_gamma)
  
  # thresholds per item: u_i and delta_i (length K-2)
  u <- if (!is.null(init$u)) as.numeric(init$u) else rnorm(P, hyper$mu_u, hyper$sd_u)
  
  delta <- if (!is.null(init$delta)) {
    # expect P x (K-2)
    as.matrix(init$delta)
  } else {
    if (K > 2) matrix(rnorm(P*(K-2), hyper$mu_delta, hyper$sd_delta), P, (K-2)) else matrix(numeric(P*0), P, 0)
  }
  
  # build b thresholds list
  b_list <- vector("list", P)
  for (i in 1:P) {
    b_list[[i]] <- build_thresholds(u[i], if (K > 2) delta[i, ] else numeric(0))
  }
  
  # --------------------------
  # Storage
  # --------------------------
  n_save <- floor((n_iter - burnin) / thin)
  samples <- list(
    # keep naming compatible with your lsirm_basic
    alpha = matrix(NA, n_save, N),         # use alpha == alpha for compatibility
    alpha = matrix(NA, n_save, N),
    
    a = array(NA, dim = c(n_save, N, d)),
    b = array(NA, dim = c(n_save, P, d)),
    
    # thresholds (store u, delta, and b thresholds)
    u = matrix(NA, n_save, P),
    delta = if (K > 2) array(NA, dim = c(n_save, P, K-2)) else NULL,
    thr = array(NA, dim = c(n_save, P, K-1)),   # b_{ik}
    
    log_gamma = rep(NA, n_save),
    sigma_alpha_sq = rep(NA, n_save),
    
    loglik = rep(NA, n_save)
  )
  
  accept <- list(
    alpha = rep(0, N),
    a     = rep(0, N),
    b     = rep(0, P),
    log_gamma = 0,
    thr   = rep(0, P)
  )
  
  # # current log-likelihood
  # loglik_cur <- loglik_full(Y, alpha, A, B, b_list, gamma)
  
  # progress helper
  if (verbose) cat("Start MH-within-Gibbs for LSGRM...\n")
  
  save_idx <- 0
  
  # =========================================================
  # MCMC loop
  # =========================================================
  for (it in 1:n_iter) {
    
    # --------------------------
    # (A) Update alpha_j (RWMH)
    # --------------------------
    for (j in 1:N) {
      # 현재/제안 alpha
      alpha_cur <- alpha[j]
      alpha_prp <- alpha_cur + rnorm(1, 0, prop_sd$alpha)
      # respondent j의 응답행
      y_row <- Y[j, ]
      # respondent j의 dist_row (각 item까지 거리)
      dist_row <- sqrt(rowSums((B - matrix(A[j, ], nrow(B), ncol(B), byrow = TRUE))^2))
      # 또는 (A,B가 반대 정의면 그에 맞게)
      
      ll_cur_j <- loglik_resp_j(y_row, alpha_cur, dist_row, b_list, gamma)
      ll_prp_j <- loglik_resp_j(y_row, alpha_prp, dist_row, b_list, gamma)
      logprior_cur <- dnorm(alpha_cur, 0, sqrt(sigma_alpha_sq), log = TRUE)
      logprior_prp <- dnorm(alpha_prp, 0, sqrt(sigma_alpha_sq), log = TRUE)
      
      log_acc <- (ll_prp_j - ll_cur_j) + (logprior_prp - logprior_cur)
      if (log(runif(1)) < log_acc) {
        alpha[j] <- alpha_prp
        accept$alpha[j] <- accept$alpha[j] + 1
      }
    }
    
    # --------------------------
    # (B) Update sigma_alpha_sq (Gibbs: Inv-Gamma)
    # --------------------------
    a_post <- hyper$a_alpha + N/2
    b_post <- hyper$b_alpha + 0.5 * sum(alpha^2)
    # sample Inv-Gamma via 1/Gamma
    sigma_alpha_sq <- 1 / rgamma(1, shape = a_post, rate = b_post)
    
    # --------------------------
    # (C) Update A (respondent positions) (RWMH)
    # prior: N(0, I)
    # --------------------------
    for (j in 1:N) {
      A_prop <- A
      A_prop[j, ] <- A[j, ] + rnorm(d, 0, prop_sd$a)
      
      # respondent j의 응답행
      y_row <- Y[j, ]
      # respondent j의 dist_row (각 item까지 거리)
      dist_row_cur <- sqrt(rowSums((B - matrix(A[j, ], nrow(B), ncol(B), byrow = TRUE))^2))
      dist_row_prp <- sqrt(rowSums((B - matrix(A_prop[j, ], nrow(B), ncol(B), byrow = TRUE))^2))
      # 또는 (A,B가 반대 정의면 그에 맞게)
      
      ll_cur_j <- loglik_resp_j(y_row, alpha[j], dist_row_cur, b_list, gamma)
      ll_prp_j <- loglik_resp_j(y_row, alpha[j], dist_row_prp, b_list, gamma)
      
      logprior_cur <- -0.5 * sum(A[j, ]^2)
      logprior_prp <- -0.5 * sum(A_prop[j, ]^2)
      
      log_acc <- (ll_prp_j - ll_cur_j) + (logprior_prp - logprior_cur)
      if (log(runif(1)) < log_acc) {
        A <- A_prop
        accept$a[j] <- accept$a[j] + 1
      }
    }
    
    # --------------------------
    # (D) Update B (item positions) (RWMH)
    # prior: N(0, I)
    # --------------------------
    for (i in 1:P) {
      B_prop <- B
      B_prop[i, ] <- B[i, ] + rnorm(d, 0, prop_sd$b)
      
      # item i의 응답열
      y_col <- Y[, i]
      # item i의 dist_col (모든 respondent까지 거리)
      dist_col_cur <- sqrt(rowSums((A - matrix(B[i, ], nrow(A), ncol(A), byrow = TRUE))^2))
      dist_col_prp <- sqrt(rowSums((A - matrix(B_prop[i, ], nrow(A), ncol(A), byrow = TRUE))^2))
      
      # item i에 대한 loglik만 비교
      ll_cur_i <- loglik_item_i(y_col, alpha, dist_col_cur, b_list[[i]], gamma)
      ll_prp_i <- loglik_item_i(y_col, alpha, dist_col_prp, b_list[[i]], gamma)
      
      logprior_cur <- -0.5 * sum(B[i, ]^2)
      logprior_prp <- -0.5 * sum(B_prop[i, ]^2)
      
      log_acc <- (ll_prp_i - ll_cur_i) + (logprior_prp - logprior_cur)
      
      if (log(runif(1)) < log_acc) {
        B <- B_prop
        accept$b[i] <- accept$b[i] + 1
      }
    }
    
    
    # --------------------------
    # (E) Update log_gamma (RWMH on log-scale)
    # --------------------------
    log_gamma_prop <- log_gamma + rnorm(1, 0, prop_sd$log_gamma)
    gamma_prop <- exp(log_gamma_prop)
    gamma <- exp(log_gamma)
    loglik_cur <- loglik_full(Y, alpha, A, B, b_list, gamma)
    loglik_prop <- loglik_full(Y, alpha, A, B, b_list, gamma_prop)
    
    logprior_cur <- dnorm(log_gamma, hyper$mu_log_gamma, hyper$sd_log_gamma, log = TRUE)
    logprior_prp <- dnorm(log_gamma_prop, hyper$mu_log_gamma, hyper$sd_log_gamma, log = TRUE)
    
    log_acc <- (loglik_prop - loglik_cur) + (logprior_prp - logprior_cur)
    if (log(runif(1)) < log_acc) {
      log_gamma <- log_gamma_prop
      gamma <- gamma_prop
      loglik_cur <- loglik_prop
      accept$log_gamma <- accept$log_gamma + 1
    }
    
    # --------------------------
    # (F) Update thresholds item-wise (block with diagonal RW)
    # Params: u_i, delta_i2..delta_i(K-1)
    # Add Jacobian change sum(delta)
    # --------------------------
    for (i in 1:P) {
      u_prop_i <- u[i] + rnorm(1, 0, prop_sd$u)
      
      if (K > 2) {
        delta_prop_i <- delta[i, ] + rnorm(K-2, 0, prop_sd$delta)  # diagonal RW
      } else {
        delta_prop_i <- numeric(0)
      }
      
      b_prop_i <- build_thresholds(u_prop_i, delta_prop_i)
      
      # compute likelihood change for column i only (faster than full)
      dist_col <- sqrt(rowSums((A - matrix(B[i, ], N, d, byrow = TRUE))^2))
      ll_cur_i <- loglik_item_i(Y[, i], alpha, dist_col, b_list[[i]], gamma)
      ll_prp_i <- loglik_item_i(Y[, i], alpha, dist_col, b_prop_i, gamma)
      
      # priors
      lp_cur <- dnorm(u[i], hyper$mu_u, hyper$sd_u, log = TRUE)
      lp_prp <- dnorm(u_prop_i, hyper$mu_u, hyper$sd_u, log = TRUE)
      
      if (K > 2) {
        lp_cur <- lp_cur + sum(dnorm(delta[i, ], hyper$mu_delta, hyper$sd_delta, log = TRUE))
        lp_prp <- lp_prp + sum(dnorm(delta_prop_i, hyper$mu_delta, hyper$sd_delta, log = TRUE))
      }
      
      # Jacobian change: sum(delta) for k=2..K-1 (i.e., delta indices)
      lj_cur <- if (K > 2) sum(delta[i, ]) else 0
      lj_prp <- if (K > 2) sum(delta_prop_i) else 0
      
      log_acc <- (ll_prp_i - ll_cur_i) + (lp_prp - lp_cur) + (lj_prp - lj_cur)
      
      if (log(runif(1)) < log_acc) {
        u[i] <- u_prop_i
        if (K > 2) delta[i, ] <- delta_prop_i
        b_list[[i]] <- b_prop_i
        
        # update full loglik (only changed column i)
        loglik_cur <- loglik_cur + (ll_prp_i - ll_cur_i)
        
        accept$thr[i] <- accept$thr[i] + 1
      }
    }
    
    # --------------------------
    # Save
    # --------------------------
    if (it > burnin && ((it - burnin) %% thin == 0)) {
      save_idx <- save_idx + 1
      
      samples$alpha[save_idx, ] <- alpha
      samples$alpha[save_idx, ] <- alpha
      
      samples$a[save_idx, , ] <- A
      samples$b[save_idx, , ] <- B
      
      samples$u[save_idx, ] <- u
      if (K > 2) samples$delta[save_idx, , ] <- delta
      
      # store thresholds matrix P x (K-1)
      thr_mat <- matrix(NA, P, K-1)
      for (i in 1:P) thr_mat[i, ] <- b_list[[i]]
      samples$thr[save_idx, , ] <- thr_mat
      
      samples$log_gamma[save_idx] <- log_gamma
      samples$sigma_alpha_sq[save_idx] <- sigma_alpha_sq
      
      samples$loglik[save_idx] <- loglik_cur
    }
    
    if (verbose && (it %% 100 == 0)) {
      cat(sprintf("iter %d / %d | loglik=%.2f | gamma=%.3f\n", it, n_iter, loglik_cur, gamma))
    }
  }
  
  # acceptance rates
  accept_rate <- list(
    alpha = accept$alpha / n_iter,
    a = accept$a / n_iter,
    b = accept$b / n_iter,
    log_gamma = accept$log_gamma / n_iter,
    thr = accept$thr / n_iter
  )
  
  fit <- list(
    samples = samples,
    accept = accept_rate,
    info = list(N = N, P = P, K = K, d = d, burnin = burnin, thin = thin,
                hyper = hyper, prop_sd = prop_sd)
  )
  
  # =========================================================
  # Procrustes matching (same as your lsirm_basic)
  # =========================================================
  loglik_save <- samples$loglik
  map_idx <- which.max(loglik_save)
  
  A_map <- matrix(samples$a[map_idx, , ], N, d)
  B_map <- matrix(samples$b[map_idx, , ], P, d)
  X_map <- rbind(A_map, B_map)
  idx_A <- 1:N
  idx_B <- (N + 1):(N + P)
  
  n_save <- dim(samples$a)[1]
  for (ss in seq_len(n_save)) {
    if (ss == map_idx) next
    
    A_s <- matrix(samples$a[ss, , ], N, d)
    B_s <- matrix(samples$b[ss, , ], P, d)
    Y_s <- rbind(A_s, B_s)
    
    fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
    Y_aligned <- predict(fit_all)
    
    samples$a[ss, , ] <- Y_aligned[idx_A, , drop = FALSE]
    samples$b[ss, , ] <- Y_aligned[idx_B, , drop = FALSE]
  }
  
  fit$samples <- samples
  fit
}

# --------------------------
# Example usage
# --------------------------
set.seed(1)
n <- 40; P <- 12; K <- 5; d <- 2
# fake ordinal data (just for testing shape; not from the model)
Y <- matrix(sample(1:K, n*P, replace=TRUE), n, P)

fit <- lsgrm_basic(
  Y_ord = Y,
  d = d,
  n_iter = 10000,
  burnin = 1000,
  thin = 5,
  prop_sd = list(alpha=0.5, a=0.8, b=0.8, log_gamma=0.8, u=0.3, delta=0.3),
  verbose = TRUE
)
fit$accept

# trace gamma
ts.plot(exp(fit$samples$log_gamma), main="gamma")

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$thr)[2]){
  for(j in 1:dim(fit$samples$thr)[3]){
    ts.plot(fit$samples$thr[,i,j], main = paste0(i, "_", j))
  }
}



# posterior mean positions
a_mean <- colMeans(fit$samples$a, dims=1)
b_mean <- colMeans(fit$samples$b, dims=1)
plot(a_mean, pch=21, col="red", bg="brown"); points(b_mean, pch=21, col="navy", bg="skyblue")


