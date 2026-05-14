library(Rcpp)
library(RcppArmadillo)
library(vegan)

# 1) 컴파일
sourceCpp("my_LSIRM_3layered_nonhierarchical.cpp")

lsirm_sharedpos_layer3 <- function(
    Y_bin, Y_con, Y_cnt,
    d = 2,
    n_iter = 5000,
    burnin = 1000,
    thin = 5,
    hyper = list(
      a_sigma = 1, b_sigma = 0.1,
      a_tau1  = 1, b_tau1  = 0.1,
      a_tau2  = 1, b_tau2  = 0.1,
      a_tau3  = 1, b_tau3  = 0.05,
      a_sigma0 = 1, b_sigma0 = 1,
      mu_log_gamma = 0, sd_log_gamma = 1,
      mu_log_kappa = 0, sd_log_kappa = 1
    ),
    prop_sd = list(
      alpha = 0.10, beta1 = 0.10, beta2 = 0.10, beta3 = 0.10,
      log_gamma = 0.05, log_kappa = 0.05,
      a = 0.10, b1 = 0.10, b2 = 0.10, b3 = 0.10
    ),
    init = NULL,
    verbose = TRUE
){
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  Y_cnt <- as.matrix(Y_cnt)
  
  fit <- lsirm_sharedpos_layer3_cpp(
    Y_bin = Y_bin,
    Y_con = Y_con,
    Y_cnt = Y_cnt,
    d = d,
    n_iter = n_iter,
    burnin = burnin,
    thin = thin,
    hyper = hyper,
    prop_sd = prop_sd,
    init = init,
    verbose = verbose
  )
  
  samples <- fit$samples
  
  # --------------------------
  # Procrustes matching (R)
  # --------------------------
  loglik_save <- samples$loglik
  map_idx <- which.max(loglik_save)
  
  n  <- nrow(Y_bin)
  P1 <- ncol(Y_bin)
  P2 <- ncol(Y_con)
  P3 <- ncol(Y_cnt)
  
  A_map  <- matrix(samples$a[map_idx, , ],  n,  d)
  B1_map <- matrix(samples$b1[map_idx, , ], P1, d)
  B2_map <- matrix(samples$b2[map_idx, , ], P2, d)
  B3_map <- matrix(samples$b3[map_idx, , ], P3, d)
  
  X_map <- rbind(A_map, B1_map, B2_map, B3_map)
  idx_A  <- 1:n
  idx_B1 <- (n + 1):(n + P1)
  idx_B2 <- (n + P1 + 1):(n + P1 + P2)
  idx_B3 <- (n + P1 + P2 + 1):(n + P1 + P2 + P3)
  
  n_save <- dim(samples$a)[1]
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
    
    samples$z[ss, , ]  <- samples$a[ss, , ]
    samples$a1[ss, , ] <- samples$a[ss, , ]
    samples$a2[ss, , ] <- samples$a[ss, , ]
    samples$a3[ss, , ] <- samples$a[ss, , ]
  }
  
  fit$samples <- samples
  fit
}
# 
# source("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM/KSAH_dataprocessing.R")
# 
# 
# fit <- lsirm_sharedpos_layer3(
#   Y_bin = Y_bin,
#   Y_con = Y_likert_5,
#   Y_cnt = Y_cnt,
#   d     = 2,
#   n_iter = 100000,
#   burnin = 50000,
#   thin   = 5,
#   prop_sd = list(
#     alpha      = 0.40,
#     beta1      = 1.50,
#     beta2      = 0.30,
#     beta3      = 0.10,
#     log_gamma  = 0.03,
#     log_kappa  = 0.50,
#     a          = 0.70,
#     b1         = 0.50,
#     b2         = 0.50,
#     b3         = 0.20
#   )
# )
# 
# 
# 
# fit$accept
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$a)[2]){
#   for(j in 1:dim(fit$samples$a)[3]){
#     ts.plot(fit$samples$a[,i,j], main = paste0('a: ', i, '_',j))
#   }
# }
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$b1)[2]){
#   for(j in 1:dim(fit$samples$b1)[3]){
#     ts.plot(fit$samples$b1[,i,j], main = paste0('b1: ', i, '_',j))
#   }
# }
# 
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$b2)[2]){
#   for(j in 1:dim(fit$samples$b2)[3]){
#     ts.plot(fit$samples$b2[,i,j], main = paste0('b2: ', i, '_',j))
#   }
# }
# 
# par(mfrow = c(2,2))
# # identifiability 생겨버림.
# for(i in 1:dim(fit$samples$b3)[2]){
#   for(j in 1:dim(fit$samples$b3)[3]){
#     ts.plot(fit$samples$b3[,i,j], main = paste0('b3: ', i, '_',j))
#   }
# }
# 
# #-------------------------------------------------------------------------------
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$alpha)[2]){
#   ts.plot(fit$samples$alpha[,i], main = paste0('alpha_',i))
# }
# 
# # - beta1: difficulty of binary item
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta1)[2]){
#   ts.plot(fit$samples$beta1[,i], main = paste0('beta1_',i))
# }
# 
# # - beta2: difficulty of continuous item
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta2)[2]){
#   ts.plot(fit$samples$beta2[,i], main = paste0('beta2_',i))
# }
# 
# # - beta3: difficulty of count item
# par(mfrow = c(2,2))
# for(i in 1:dim(fit$samples$beta3)[2]){
#   ts.plot(fit$samples$beta3[,i], main = paste0('beta3_',i))
# }
# 
# #-------------------------------------------------------------------------------
# ts.plot(fit$samples$sigma0_sq, main = 'sigma0_sq')
# 
# ts.plot(exp(fit$samples$log_gamma), main = 'gamma')
# 
# ts.plot(log(fit$samples$tau_beta1_sq), main = 'tau_beta1_sq log transformed')
# 
# ts.plot(log(fit$samples$tau_beta2_sq), main = 'tau_beta2_sq log transformed')
# 
# ts.plot(log(fit$samples$tau_beta3_sq), main = 'tau_beta3_sq log transformed')
# 
# ################################################################################
# # latent position visualization
# ################################################################################
# # response
# a_mean <- colMeans(fit$samples$a, dims = 1)
# # par(mfrow = c(1,1))
# # base::plot(
# #   a_mean,
# #   pch = 21, col = "black", bg = "gray80",
# #   main = "Latent positions: a",
# #   ylim = c(-3,3),
# #   xlim = c(-3,3)
# # )
# # 
# # # --- 숫자 라벨 추가 ---
# # 
# # text(a_mean,  labels = 1:dim(a_mean)[1],  pos = 4, cex = 0.7)
# # 
# # legend(
# #   "topright",
# #   legend = c("a: respondent"),
# #   pch = 21,
# #   bty = "n",
# #   col = "gray80",
# #   pt.bg = "gray80"
# # )
# 
# # item
# b3_mean <- colMeans(fit$samples$b3, dims = 1)
# b2_mean <- colMeans(fit$samples$b2, dims = 1)
# b1_mean <- colMeans(fit$samples$b1, dims = 1)
# 
# par(mfrow = c(1,1))
# plot(
#   a_mean,
#   pch = 21, col = "red", bg = "brown",
#   main = "Item positions: b2 (blue), b1 (skyblue)"
# )
# points(b2_mean, pch = 21, col = "navy", bg = "blue")
# points(b1_mean, pch = 21, col = "steelblue4", bg = "skyblue")
# points(b3_mean, pch = 21, col = "grey", bg = "grey")
# # --- 숫자 라벨 추가 ---
# text(a_mean, labels = 1:dim(a_mean)[1], pos = 4, cex = 0.7, col = "red")
# text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")
# text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "steelblue4")
# text(b3_mean, labels = 1:dim(b3_mean)[1], pos = 4, cex = 0.7, col = "grey")
# 
# legend(
#   "topright",
#   legend = c("a: respondent", "b2: continuous item", "b1: binary item", "b3: count item"),
#   pch = 21,
#   pt.bg = c("brown", "blue", "skyblue","grey"),
#   col = c("red","navy", "steelblue4","grey"),
#   bty = "n"
# )
# 
# 
# 
# 
