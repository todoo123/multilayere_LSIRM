library(Rcpp)
library(RcppArmadillo)
library(vegan)

# 컴파일
sourceCpp("my_LSIRM.cpp")

lsirm_basic <- function(
    Y_bin,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    hyper = list(
      a_sigma = 1, b_sigma = 0.1,
      a_tau   = 1, b_tau   = 0.1,
      mu_log_gamma = 0, sd_log_gamma = 1
    ),
    prop_sd = list(
      alpha = 0.10,
      beta  = 0.10,
      log_gamma = 0.05,
      a = 0.10,
      b = 0.10
    ),
    init = NULL,
    verbose = TRUE,
    fix_gamma = FALSE
){
  Y_bin <- as.matrix(Y_bin)
  
  fit <- basic_lsirm_cpp(
    Y_bin = Y_bin,
    d = d,
    n_iter = n_iter,
    burnin = burnin,
    thin = thin,
    hyper = hyper,
    prop_sd = prop_sd,
    init = init,
    verbose = verbose,
    fix_gamma = fix_gamma
  )
  
  samples <- fit$samples
  
  # --------------------------
  # Procrustes matching (R)
  # --------------------------
  loglik_save <- samples$loglik
  map_idx <- which.max(loglik_save)
  
  n <- nrow(Y_bin)
  P <- ncol(Y_bin)
  
  A_map <- matrix(samples$a[map_idx, , ], n, d)
  B_map <- matrix(samples$b[map_idx, , ], P, d)
  
  X_map <- rbind(A_map, B_map)
  idx_A <- 1:n
  idx_B <- (n + 1):(n + P)
  
  n_save <- dim(samples$a)[1]
  for(ss in seq_len(n_save)){
    if(ss == map_idx) next
    
    A_s <- matrix(samples$a[ss, , ], n, d)
    B_s <- matrix(samples$b[ss, , ], P, d)
    Y_s <- rbind(A_s, B_s)
    
    fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
    Y_aligned <- predict(fit_all)
    
    samples$a[ss, , ] <- Y_aligned[idx_A, , drop = FALSE]
    samples$b[ss, , ] <- Y_aligned[idx_B, , drop = FALSE]
    
    # compatibility 유지
    samples$z[ss, , ]  <- samples$a[ss, , ]
    samples$a1[ss, , ] <- samples$a[ss, , ]
    samples$a2[ss, , ] <- samples$a[ss, , ]
    samples$a3[ss, , ] <- samples$a[ss, , ]
  }
  
  fit$samples <- samples
  fit
}

# 
# set.seed(1)
# n <- 40; P <- 15; d <- 2
# Y <- matrix(rbinom(n*P, 1, 0.3), n, P)
# 
# fit <- lsirm_basic(
#   Y_bin = Y_bin_all,
#   d = d,
#   n_iter = 10000,
#   burnin = 1000,
#   thin = 5,
#   prop_sd = list(
#     alpha = 0.70,
#     beta  = 0.50,
#     log_gamma = 0.05,
#     a = 0.50,
#     b = 0.30
#   ),
#   verbose = TRUE
# )
# 
# fit$accept
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$a)[2]){
#   for(j in 1:dim(fit$samples$a)[3]){
#     ts.plot(fit$samples$a[, i, j], main = paste0("a: ", i, "_", j))
#   }
# }
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$b)[2]){
#   for(j in 1:dim(fit$samples$b)[3]){
#     ts.plot(fit$samples$b[, i, j], main = paste0("b: ", i, "_", j))
#   }
# }
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$alpha)[2]){
#   ts.plot(fit$samples$alpha[, i], main = paste0("alpha_", i))
# }
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta)[2]){
#   ts.plot(fit$samples$beta[, i], main = paste0("beta_", i))
# }
# 
# 
# par(mfrow = c(2,2))
# 
# ts.plot(exp(fit$samples$log_gamma), main = "gamma = exp(log_gamma)")
# ts.plot(fit$samples$log_gamma,       main = "log_gamma")
# 
# ts.plot(fit$samples$sigma_alpha_sq,  main = "sigma_alpha_sq")
# ts.plot(log(fit$samples$sigma_alpha_sq), main = "log(sigma_alpha_sq)")
# 
# ts.plot(fit$samples$tau_beta_sq,     main = "tau_beta_sq")
# ts.plot(log(fit$samples$tau_beta_sq), main = "log(tau_beta_sq)")
# 
# ts.plot(fit$samples$loglik, main = "log-likelihood")
# 
# # posterior mean latent positions
# a_mean <- colMeans(fit$samples$a, dims = 1)  # n x d
# b_mean <- colMeans(fit$samples$b, dims = 1)  # P x d
# 
# par(mfrow = c(1,1))
# base::plot(
#   a_mean,
#   pch = 21, col = "red", bg = "brown",
#   main = "Latent positions: a (respondents) + b (items)"
# )
# 
# points(b_mean, pch = 21, col = "navy", bg = "skyblue")
# 
# # labels
# text(a_mean, labels = 1:nrow(a_mean), pos = 4, cex = 0.7, col = "red")
# text(b_mean, labels = 1:nrow(b_mean), pos = 4, cex = 0.7, col = "navy")
# 
# legend(
#   "topright",
#   legend = c("a: respondent", "b: item"),
#   pch = 21,
#   pt.bg = c("brown", "skyblue"),
#   col = c("red", "navy"),
#   bty = "n"
# )
# 
# 
# 
# 
# 
# 
# 
# 
