library(Rcpp)
library(coda)
library(RcppArmadillo)
sourceCpp("my_LSIRM_mixture_3layered.cpp")

lsirm_global_local_layer3 <- function(
    Y_bin,              # n x P1, 0/1
    Y_con,              # n x P2, >0 (continuous)
    Y_cnt,              # n x P3, 0,1,2,... (count)
    d       = 2,        # latent dimension
    n_iter  = 5000,
    burnin  = 1000,
    thin    = 5,
    hyper   = list(
      # alpha variance priors
      a_sigma1 = 1, b_sigma1 = 0.1,
      a_sigma2 = 1, b_sigma2 = 0.1,
      a_sigma3 = 1, b_sigma3 = 0.1,
      # beta variance priors
      a_tau1   = 1, b_tau1   = 0.1,
      a_tau2   = 1, b_tau2   = 0.1,
      a_tau3   = 1, b_tau3   = 0.1,
      # residual variance prior (continuous layer)
      a_sigma0 = 1, b_sigma0 = 1,
      # dispersion parameter prior (count layer)
      mu_log_alpha = 0, sd_log_alpha = 1,
      # Global-Local linkage variance (sigma_1^2)
      a_sigma_global = 0.005, b_sigma_global = 0.005,
      # log-gamma priors
      mu_log_gamma1 = 0, sd_log_gamma1 = 1,
      mu_log_gamma2 = 0, sd_log_gamma2 = 1,
      mu_log_gamma3 = 0, sd_log_gamma3 = 1
    ),
    prop_sd = list(
      alpha1     = 0.10, beta1      = 0.10,
      alpha2     = 0.10, beta2      = 0.10,
      alpha3     = 0.10, beta3      = 0.10,
      log_gamma1 = 0.05, log_gamma2 = 0.05, log_gamma3 = 0.05, 
      log_alpha  = 0.05,  # dispersion parameter, layer3
      a1         = 0.10,  # local respondent pos, layer1
      a2         = 0.10,  # local respondent pos, layer2
      a3         = 0.10,  # local respondent pos, layer3
      b1         = 0.10,  # item pos binary
      b2         = 0.10,  # item pos continuous
      b3         = 0.10   # item pos count
    ),
    init    = NULL,
    verbose = TRUE
) {
  # -------------------------
  # 기본 전처리 + NA 체크
  # -------------------------
  Y_bin <- as.matrix(Y_bin)
  Y_con <- as.matrix(Y_con)
  Y_cnt <- as.matrix(Y_cnt)
  
  if (sum(is.na(Y_bin)) + sum(is.na(Y_con)) + sum(is.na(Y_cnt)) > 0L) {
    stop("Missing values are not allowed in the data.")
  }
  
  # -------------------------
  # C++ MCMC 실행
  # -------------------------
  res_cpp <- lsirm_global_local_layer3_cpp(
    Y_bin = Y_bin,
    Y_con = Y_con,
    Y_cnt = Y_cnt,
    d     = d,
    n_iter = n_iter,
    burnin = burnin,
    thin   = thin,
    hyper  = hyper,
    prop_sd = prop_sd,
    init    = init,
    verbose = verbose
  )
  
  samples <- res_cpp$samples
  accept  <- res_cpp$accept
  
  # -------------------------
  # Procrustes matching (R, vegan)
  # -------------------------
  z_save  <- samples$z   # dim: (n_save, n, d)
  a1_save <- samples$a1  # (n_save, n, d)
  a2_save <- samples$a2  # (n_save, n, d)
  a3_save <- samples$a3  # (n_save, n, d)
  b1_save <- samples$b1  # (n_save, P1, d)
  b2_save <- samples$b2  # (n_save, P2, d)
  b3_save <- samples$b3  # (n_save, P3, d)
  loglik_save <- samples$loglik
  
  n_save <- dim(z_save)[1]
  n      <- dim(z_save)[2]
  d      <- dim(z_save)[3]
  P1     <- dim(b1_save)[2]
  P2     <- dim(b2_save)[2]
  P3     <- dim(b3_save)[2]
  
  # MAP iteration 기준으로 기준 좌표 설정
  map_idx <- which.max(loglik_save)
  
  Z_map  <- matrix(z_save[map_idx, , ],  nrow = n,   ncol = d)
  a1_map <- matrix(a1_save[map_idx, , ], nrow = n,   ncol = d)
  a2_map <- matrix(a2_save[map_idx, , ], nrow = n,   ncol = d)
  a3_map <- matrix(a3_save[map_idx, , ], nrow = n,   ncol = d)
  b1_map <- matrix(b1_save[map_idx, , ], nrow = P1,  ncol = d)
  b2_map <- matrix(b2_save[map_idx, , ], nrow = P2,  ncol = d)
  b3_map <- matrix(b3_save[map_idx, , ], nrow = P3,  ncol = d)
  
  # 기준 concat
  X_map <- rbind(Z_map, a1_map, a2_map, a3_map, b1_map, b2_map, b3_map)
  
  if (requireNamespace("vegan", quietly = TRUE)) {
    
    # 블록별 행 개수
    n_Z  <- nrow(Z_map)
    n_a1 <- nrow(a1_map)
    n_a2 <- nrow(a2_map)
    n_a3 <- nrow(a3_map)
    n_b1 <- nrow(b1_map)
    n_b2 <- nrow(b2_map)
    n_b3 <- nrow(b3_map)
    
    block_sizes <- c(n_Z, n_a1, n_a2, n_a3, n_b1, n_b2, n_b3)
    cs <- cumsum(block_sizes)
    
    idx_Z  <- 1:cs[1]
    idx_a1 <- (cs[1] + 1):cs[2]
    idx_a2 <- (cs[2] + 1):cs[3]
    idx_a3 <- (cs[3] + 1):cs[4]
    idx_b1 <- (cs[4] + 1):cs[5]
    idx_b2 <- (cs[5] + 1):cs[6]
    idx_b3 <- (cs[6] + 1):cs[7]
    
    # 각 저장된 샘플에 대해 Procrustes 정렬
    for (ss in seq_len(n_save)) {
      if (ss == map_idx) next
      
      Z_s  <- matrix(z_save[ss, , ],  nrow = n_Z,  ncol = d)
      a1_s <- matrix(a1_save[ss, , ], nrow = n_a1, ncol = d)
      a2_s <- matrix(a2_save[ss, , ], nrow = n_a2, ncol = d)
      a3_s <- matrix(a3_save[ss, , ], nrow = n_a3, ncol = d)
      b1_s <- matrix(b1_save[ss, , ], nrow = n_b1, ncol = d)
      b2_s <- matrix(b2_save[ss, , ], nrow = n_b2, ncol = d)
      b3_s <- matrix(b3_save[ss, , ], nrow = n_b3, ncol = d)
      
      Y_s <- rbind(Z_s, a1_s, a2_s, a3_s, b1_s, b2_s, b3_s)
      
      fit_all <- vegan::procrustes(X = X_map, Y = Y_s, scale = FALSE)
      Y_s_aligned <- stats::predict(fit_all)
      
      z_save[ss, , ]  <- Y_s_aligned[idx_Z,  , drop = FALSE]
      a1_save[ss, , ] <- Y_s_aligned[idx_a1, , drop = FALSE]
      a2_save[ss, , ] <- Y_s_aligned[idx_a2, , drop = FALSE]
      a3_save[ss, , ] <- Y_s_aligned[idx_a3, , drop = FALSE]
      b1_save[ss, , ] <- Y_s_aligned[idx_b1, , drop = FALSE]
      b2_save[ss, , ] <- Y_s_aligned[idx_b2, , drop = FALSE]
      b3_save[ss, , ] <- Y_s_aligned[idx_b3, , drop = FALSE]
    }
    
  } else {
    warning("Package 'vegan' is not installed. Skipping Procrustes matching.")
  }
  
  # Procrustes 정렬된 결과를 samples에 다시 저장
  samples$z  <- z_save
  samples$a1 <- a1_save
  samples$a2 <- a2_save
  samples$a3 <- a3_save
  samples$b1 <- b1_save
  samples$b2 <- b2_save
  samples$b3 <- b3_save
  
  # 최종 output 구조: 기존 함수와 동일
  list(
    samples = samples,
    accept  = accept,
    hyper   = hyper,
    prop_sd = prop_sd
  )
}
################################################################################
# real data
################################################################################
# KSAH 데이터 전처리 코드 실행
source("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM/KSAH_dataprocessing.R")

fit <- lsirm_global_local_layer3(
  Y_bin = Y_bin,
  Y_con = Y_likert_5,
  Y_cnt = Y_cnt,
  d     = 2,
  n_iter = 300000,
  burnin = 100000,
  thin   = 10,
  hyper   = list(
    # alpha variance priors
    a_sigma1 = 1, b_sigma1 = 0.1,
    a_sigma2 = 1, b_sigma2 = 0.1,
    a_sigma3 = 1, b_sigma3 = 0.1,
    # beta variance priors
    a_tau1   = 0.1, b_tau1   = 0.1,
    a_tau2   = 0.1, b_tau2   = 0.1,
    a_tau3   = 0.005, b_tau3   = 0.005,
    # residual variance prior (continuous layer)
    a_sigma0 = 1, b_sigma0 = 1,
    # dispersion parameter prior (count layer)
    mu_log_alpha = 0, sd_log_alpha = 1,
    # Global-Local linkage variance (sigma_1^2)
    a_sigma_global = 0.001, b_sigma_global = 0.001,
    # log-gamma priors
    mu_log_gamma1 = 0, sd_log_gamma1 = 1,
    mu_log_gamma2 = 0, sd_log_gamma2 = 1,
    mu_log_gamma3 = 0, sd_log_gamma3 = 1
  ),
  prop_sd = list(
    alpha1     = 0.70,
    beta1      = 1.00,
    alpha2     = 0.40,
    beta2      = 0.30,
    alpha3     = 0.20,
    beta3      = 0.10,
    log_gamma1 = 0.1,
    log_gamma2 = 0.02,
    log_gamma3 = 0.1,
    log_alpha  = 0.30,
    a1         = 0.70,
    a2         = 0.50,
    a3         = 1.00,
    b1         = 0.50,
    b2         = 0.50,
    b3         = 0.30
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
  }
}
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a1)[2]){
  for(j in 1:dim(fit$samples$a1)[3]){
    ts.plot(fit$samples$a1[,i,j], main = paste0('a1: ', i, '_',j))
  }
}
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a2)[2]){
  for(j in 1:dim(fit$samples$a2)[3]){
    ts.plot(fit$samples$a2[,i,j], main = paste0('a2: ', i, '_',j))
  }
}
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$a3)[2]){
  for(j in 1:dim(fit$samples$a3)[3]){
    ts.plot(fit$samples$a3[,i,j], main = paste0('a3: ', i, '_',j))
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
# identifiability 생겨버림- 마지막 문항 뺴고 iter 늘리고, beta3 분산의 하이퍼파라미터 0쪽으로 미니까 좀 나아짐 
# 아마 beta3 의 추정이 안정화되면서 같이 안정화 된 것 같음
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$b3)[2]){
  for(j in 1:dim(fit$samples$b3)[3]){
    ts.plot(fit$samples$b3[,i,j], main = paste0('b3: ', i, '_',j))
  }
}

#-------------------------------------------------------------------------------
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta1)[2]){
  ts.plot(fit$samples$beta1[,i], main = paste0('beta1_',i))
}

par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha1)[2]){
  ts.plot(fit$samples$alpha1[,i], main = paste0('alpha1_',i))
}

# - beta2: difficulty of continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta2)[2]){
  ts.plot(fit$samples$beta2[,i], main = paste0('beta2_',i))
}

# - alpha2: ability of resonpondent - continuous item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha2)[2]){
  ts.plot(fit$samples$alpha2[,i], main = paste0('alpha2_',i))
}

# - beta3: difficulty of count item - ess 얼마나 나올라나
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$beta3)[2]){
  ts.plot(fit$samples$beta3[,i], main = paste0('beta3_',i))
}

# - alpha3: ability of resonpondent - count item
par(mfrow = c(2,2))
for(i in 1:dim(fit$samples$alpha3)[2]){
  ts.plot(fit$samples$alpha3[,i], main = paste0('alpha3_',i))
}

#-------------------------------------------------------------------------------
ts.plot(fit$samples$sigma0_sq, main = 'sigma0_sq')

ts.plot(fit$samples$sigma1_sq, main = 'sigma1_sq')

ts.plot(exp(fit$samples$log_gamma1), main = 'gamma1')

ts.plot(exp(fit$samples$log_gamma2), main = 'gamma2')

ts.plot(exp(fit$samples$log_gamma3), main = 'gamma3')

# ts.plot(fit$samples$sigma_alpha1_sq, main = 'sigma_alpha1_sq')
ts.plot(log(fit$samples$sigma_alpha1_sq), main = 'sigma_alpha1_sq log transformed')

# ts.plot(fit$samples$sigma_alpha2_sq, main = 'sigma_alpha2_sq')
ts.plot(log(fit$samples$sigma_alpha2_sq), main = 'sigma_alpha2_sq log transformed')

# ts.plot(fit$samples$sigma_alpha3_sq, main = 'sigma_alpha3_sq')
ts.plot(log(fit$samples$sigma_alpha3_sq), main = 'sigma_alpha3_sq log transformed')

# ts.plot(fit$samples$tau_beta1_sq, main = 'tau_beta1_sq')
ts.plot(log(fit$samples$tau_beta1_sq), main = 'tau_beta1_sq log transformed')

# ts.plot(fit$samples$tau_beta2_sq, main = 'tau_beta2_sq')
ts.plot(log(fit$samples$tau_beta2_sq), main = 'tau_beta2_sq log transformed')

# ts.plot(fit$samples$tau_beta3_sq, main = 'tau_beta3_sq')
ts.plot(log(fit$samples$tau_beta3_sq), main = 'tau_beta3_sq log transformed')

ts.plot(exp(fit$samples$log_alpha), main = 'log_alpha')

# effective sample size
effectiveSize(fit$samples$z[,,1])
effectiveSize(fit$samples$z[,,2])
effectiveSize(fit$samples$b3[,,1])
effectiveSize(fit$samples$b3[,,2])
effectiveSize(fit$samples$b2[,,1])
effectiveSize(fit$samples$b2[,,2])
effectiveSize(fit$samples$b1[,,1])
effectiveSize(fit$samples$b1[,,2])
effectiveSize(fit$samples$a3[,,1])
effectiveSize(fit$samples$a3[,,2])
effectiveSize(fit$samples$a2[,,1])
effectiveSize(fit$samples$a2[,,2])
effectiveSize(fit$samples$a1[,,1])
effectiveSize(fit$samples$a1[,,2])
effectiveSize(fit$samples$log_alpha)
effectiveSize(fit$samples$sigma1_sq)
effectiveSize(fit$samples$sigma0_sq)
effectiveSize(fit$samples$tau_beta3_sq)
effectiveSize(fit$samples$tau_beta2_sq)
effectiveSize(fit$samples$tau_beta1_sq)
effectiveSize(fit$samples$sigma_alpha3_sq)
effectiveSize(fit$samples$sigma_alpha2_sq)
effectiveSize(fit$samples$sigma_alpha1_sq)
effectiveSize(fit$samples$log_gamma3)
effectiveSize(fit$samples$log_gamma2)
effectiveSize(fit$samples$log_gamma1)
effectiveSize(fit$samples$beta3)
effectiveSize(fit$samples$beta2)
effectiveSize(fit$samples$beta1)
effectiveSize(fit$samples$alpha3)
effectiveSize(fit$samples$alpha2)
effectiveSize(fit$samples$alpha1)


################################################################################
# latent position visualization
################################################################################
# response
z_mean  <- colMeans(fit$samples$z,  dims = 1)
a1_mean <- colMeans(fit$samples$a1, dims = 1)
a2_mean <- colMeans(fit$samples$a2, dims = 1)
a3_mean <- colMeans(fit$samples$a3, dims = 1)

par(mfrow = c(1,1))
base::plot(
  z_mean,
  pch = 21, col = "black", bg = "gray80",
  main = "Latent positions: z (gray), a1 (red), a2 (orange)",
  ylim = c(-4,4),
  xlim = c(-4,4)
)

points(a1_mean, pch = 21, col = "red", bg = "red")
points(a2_mean, pch = 21, col = "orange", bg = "orange")
points(a3_mean, pch = 21, col = "brown", bg = "brown")

# --- 숫자 라벨 추가 ---
text(z_mean,  labels = 1:dim(z_mean)[1],  pos = 4, cex = 0.7)
text(a1_mean, labels = 1:dim(a1_mean)[1], pos = 4, cex = 0.7, col = "red")
text(a2_mean, labels = 1:dim(a2_mean)[1], pos = 4, cex = 0.7, col = "orange")
text(a3_mean, labels = 1:dim(a3_mean)[1], pos = 4, cex = 0.7, col = "brown")

legend(
  "topright",
  legend = c("z: global respondent", "a1: binary respondent", "a2: continuous respondent", "a3: count respondent"),
  pch = 21,
  pt.bg = c("gray80", "red", "orange", "brown"),
  col = c("black", "red", "orange","brown"),
  bty = "n"
)

# item
b3_mean <- colMeans(fit$samples$b3, dims = 1)
b2_mean <- colMeans(fit$samples$b2, dims = 1)
b1_mean <- colMeans(fit$samples$b1, dims = 1)

base::plot(
  b2_mean,
  pch = 21, col = "navy", bg = "navy",
  main = "Item positions: b2 (blue), b1 (skyblue)",
  ylim = c(-4,4),
  xlim = c(-4,4)
)

points(b1_mean, pch = 21, col = "skyblue", bg = "skyblue")
points(b3_mean, pch = 21, col = "blue", bg = "blue")
# --- 숫자 라벨 추가 ---
text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")
text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "skyblue")
text(b3_mean, labels = 1:dim(b3_mean)[1], pos = 4, cex = 0.7, col = "blue")

legend(
  "topright",
  legend = c("b2: continuous item", "b1: binary item", "b3: count item"),
  pch = 21,
  pt.bg = c("navy", "skyblue","blue"),
  col = c("navy", "skyblue","blue"),
  bty = "n"
)

# binary: item - respondent
base::plot(
  a1_mean,
  pch = 21, col = "red", bg = "red",
  main = "binary: item - respondent",
  ylim = c(-3,3), 
  xlim = c(-3,3)
)

points(b1_mean, pch = 21, col = "skyblue", bg = "skyblue")

text(a1_mean, labels = 1:dim(a1_mean)[1], pos = 4, cex = 0.7, col = "red")
text(b1_mean, labels = 1:dim(b1_mean)[1], pos = 4, cex = 0.7, col = "skyblue")

legend(
  "topright",
  legend = c("a1: binary respondent", "b1: binary item"),
  pch = 21,
  pt.bg = c("red", "skyblue"),
  col = c("red", "skyblue"),
  bty = "n"
)
points(a1_mean[c(68,40,103),], pch = 21, col = "yellow", bg = "yellow")
points(a1_mean[c(111,64),], pch = 21, col = "green", bg = "green")
# continuous: item - respondent
base::plot(
  a2_mean,
  pch = 21, col = "orange", bg = "orange",
  main = "continuous: item - respondent",
  ylim = c(-3,3), 
  xlim = c(-3,3)
)

points(b2_mean, pch = 21, col = "navy", bg = "navy")
text(a2_mean, labels = 1:dim(a2_mean)[1], pos = 4, cex = 0.7, col = "orange")
text(b2_mean, labels = 1:dim(b2_mean)[1], pos = 4, cex = 0.7, col = "navy")
a2_mean[69,]


legend(
  "topright",
  legend = c("a2: continuous respondent", "b2: continuous item"),
  pch = 21,
  pt.bg = c("orange", "navy"),
  col = c("orange", "navy"),
  bty = "n"
)


# count: item - respondent
base::plot(
  a3_mean,
  pch = 21, col = "brown", bg = "brown",
  main = "count: item - respondent",
  ylim = c(-2,2),
  xlim = c(-2,2)
)

points(b3_mean, pch = 21, col = "blue", bg = "blue")
ind<-which(Y_cnt[,1] > 20)
points(a3_mean[ind[1:length(ind)],], pch = 21, col = "green", bg = "green")
text(a3_mean, labels = 1:dim(a3_mean)[1], pos = 4, cex = 0.7, col = "brown")
text(b3_mean, labels = 1:dim(b3_mean)[1], pos = 4, cex = 0.7, col = "blue")

legend(
  "topright",
  legend = c("a3: count respondent", "b3: count item"),
  pch = 21,
  pt.bg = c("brown", "blue"),
  col = c("brown", "blue"),
  bty = "n"
)





## --- 전체 좌표 범위 계산 ---
all_pos <- rbind(z_mean, b1_mean, b2_mean, b3_mean)

# 약간 padding 을 줘서 보기 좋게
pad_x <- diff(range(all_pos[,1])) * 0.05
pad_y <- diff(range(all_pos[,2])) * 0.05

xlim <- range(all_pos[,1]) + c(-pad_x, pad_x)
ylim <- range(all_pos[,2]) + c(-pad_y, pad_y)

par(mfrow = c(1, 1))

## --- 전체 latent space plot ---
base::plot(
  z_mean,
  pch  = 21,
  col  = "black",
  bg   = "black",
  xlim = xlim,
  ylim = ylim,
  main = "Global latent space: z (gray), b1 (skyblue), b2 (navy), b3 (blue)",
  xlab = "Latent dimension 1",
  ylab = "Latent dimension 2"
)

## --- item 위치 추가 ---
points(b1_mean, pch = 21, col = "skyblue", bg = "skyblue")
points(b2_mean, pch = 21, col = "green",      bg = "green")
points(b3_mean, pch = 21, col = "blue",     bg = "blue")

## --- 숫자 라벨 ---
text(z_mean,
     labels = 1:dim(z_mean)[1],
     pos    = 4, cex = 0.7, col = "black")

text(b1_mean,
     labels = 1:dim(b1_mean)[1],
     pos    = 4, cex = 0.7, col = "skyblue")

text(b2_mean,
     labels = 1:dim(b2_mean)[1],
     pos    = 4, cex = 0.7, col = "green")

text(b3_mean,
     labels = 1:dim(b3_mean)[1],
     pos    = 4, cex = 0.7, col = "blue")

## --- 범례 ---
legend(
  "topright",
  legend = c("z: global respondent",
             "b1: binary item",
             "b2: continuous item",
             "b3: count item"),
  pch   = 21,
  pt.bg = c("black", "skyblue", "green", "blue"),
  col   = c("black", "skyblue", "navy", "blue"),
  bty   = "n"
)

