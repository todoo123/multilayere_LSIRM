################################################################################
# LSM for binary edges
################################################################################
# rm(list = ls())
## utilities
L2_dist <- function(z) {
  n <- nrow(z)
  D <- matrix(0.0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      # distance metric
      d <- sqrt(sum((z[i,] - z[j,])^2))
      # d <- (sum((z[i,] - z[j,])^2))
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

loglik_binary_network <- function(Y, z, alpha, beta) {
  D <- L2_dist(z)
  # D <- dot_dist(z)
  eta <- alpha - beta*D
  pi <- stats::plogis(eta)
  n <- nrow(Y)
  ll <- 0.0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      y <- Y[i,j]
      p <- pi[i,j]
      if(y == 1){
        ll <- ll + log(p + 1e-11)
      }else{
        ll <- ll + log(1-p + 1e-11)
      }
    }
  }
  return(ll)
}


## compute density

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

## update function

# MH
update_z_j <- function(j, z, Y, alpha, sigma2, prop_z, beta, d){
  z_j <- z[j,]
  z_prop <- z
  prop_z_j <- z_j + rnorm(d, mean = 0, sd = sqrt(prop_z))
  z_prop[j, ] <- prop_z_j
  
  current_logpost <- loglik_binary_network(Y, z, alpha, beta) + log_prior_z_i(z_j, sigma2)
  prop_logpost <-  loglik_binary_network(Y, z_prop, alpha, beta) + log_prior_z_i(prop_z_j, sigma2)
  
  acc <- prop_logpost - current_logpost
  if (log(runif(1)) < acc) {
    return(list(z = z_prop, accepted = TRUE))
  } else {
    return(list(z = z, accepted = FALSE))
  }
}

update_alpha <- function(alpha, z, Y, xi, psi2, prop_alpha, beta) {
  ll_curr <- loglik_binary_network(Y, z, alpha, beta) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_binary_network(Y, z, alpha_prop, beta) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

update_beta <- function(beta, z, Y, alpha, prop_beta, a, b){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_binary_network(Y, z, alpha, beta) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_binary_network(Y, z, alpha, beta_prop) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

# gibbs
update_xi_psi2 <- function(alpha, xi, psi2, m0 = 0, kappa0 = 1/2, alpha0 = 3, beta0 = 1/2) {
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

# LSM 구조
# formular: 
# P(Y_{ij} = 1 | z, \alpha) = logit^{-1}(\alpha - \beta d(z_i, z_j))
# beta ~ IG(a0, b0)
# z ~ N(0, sigma2)
# alpha ~ N(xi, psi2)
# xi ~ N(m0, kappa0)
# psi2 ~ IG(alpha0, beta0)


LSM <- function(Y, iter = 5000, burnin = 1000, thinning = 5,
                sigma2 = 1, a0 = 0.5, b0 = 0.5, d = 2,
                prop_z = 0.05, prop_alpha = 0.05, prop_beta = 0.05,
                alpha_init = 0, beta_init = 1, xi_init = 0,
                psi2_init = 1, z_init = NULL, verbose = T){
  # data validation
  stopifnot(is.matrix(Y))
  n <- nrow(Y)
  stopifnot(ncol(Y) == n)
  if (max(abs(Y - t(Y))) > 1e-8) stop("Y1 must be symmetric.")
  
  # init
  if (is.null(z_init)) {
    z <- matrix(rnorm(n*d, 0, sqrt(sigma2)), nrow = n, ncol = d)
  } else {
    stopifnot(is.matrix(z_init) && nrow(z_init) == n && ncol(z_init) == d)
    z <- z_init
  }
  alpha <- alpha_init
  xi <- xi_init
  psi2 <- psi2_init
  beta <- beta_init
  
  # storage
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning)
  ns <- length(keep_idx)
  z_save <- array(NA_real_, dim = c(ns, n, d))
  alpha_save <- xi_save <- psi2_save <- beta_save <- numeric(ns)
  ll_save <- numeric(ns)
  
  # acceptance rate
  acc_z <- rep(0L, n)
  acc_alpha  <- 0L
  acc_beta <- 0L
  
  # storage index
  s <- 0L
  
  # sampler
  for(i in 1:iter){
    # update z_i (MH)
    for (j in 1:n) {
      up <- update_z_j(j, z, Y, alpha, sigma2, prop_z, beta, d)
      z <- up$z
      if (up$accepted) {
        acc_z[j] <- acc_z[j] + 1
      }
    }
    
    # update alpha (MH)
    up_alpha <- update_alpha(alpha, z, Y, xi, psi2, prop_alpha, beta)
    alpha <- up_alpha$alpha
    if (up_alpha$accepted) {
      acc_alpha <- acc_alpha + 1
    }
    
    # update beta (MH)
    up_beta <- update_beta(beta, z, Y, alpha, prop_beta, a = a0, b = b0)
    beta <- up_beta$beta
    if (up_beta$accepted) {
      acc_beta <- acc_beta + 1
    }
    
    # test: remove beta
    beta <- 1
    acc_beta <- acc_beta + 1L

    # Gibbs for (xi, psi2)
    up_xi_psi2 <- update_xi_psi2(alpha, xi, psi2)
    xi <- up_xi_psi2$xi
    psi2 <- up_xi_psi2$psi2
    
    # save
    if (i %in% keep_idx) {
      s <- s + 1
      z_save[s,,] <- z
      alpha_save[s]  <- alpha
      xi_save[s] <- xi
      psi2_save[s] <- psi2
      beta_save[s] <- beta
      ll_save[s] <- loglik_binary_network(Y, z, alpha, beta)
    }
    
    if (verbose && (i %% 100 == 0)) {
      print(paste0('iteration: ', i, ' alpha: ', alpha, ' beta: ', beta))
      print(paste0(' likelihood 1: ', loglik_binary_network(Y, z, alpha, beta)))
    }
  }
  
  # procrustes matching for latent variable coordinate
  # 1) MAP 인덱스 선택 (여기선 ll_save 최대를 MAP proxy로 사용)
  map_idx <- which.max(ll_save)
  Z_map   <- z_save[map_idx, , , drop = FALSE]  # 1 x n x d
  Z_map   <- matrix(Z_map[1, , ], nrow = n, ncol = d)
  
  # 2) 모든 저장 샘플을 Z_map에 정렬 (회전+이동만; scale=FALSE)
  #    vegan::procrustes(X=타깃, Y=정렬대상)
  for (s in seq_len(ns)) {
    Z_s <- matrix(z_save[s, , ], nrow = n, ncol = d)
    if (s == map_idx) {
      # MAP 샘플은 그대로 두되, 혹시 모를 수치오차 방지를 위해 행렬로 재저장
      z_save[s, , ] <- Z_s
    } else {
      fit <- vegan::procrustes(X = Z_map, Y = Z_s, scale = FALSE)
      Z_aligned <- predict(fit)           # n x 2
      z_save[s, , ] <- Z_aligned
    }
  }
  
  return(
    list(
      samples = list(
        z = z_save,
        alpha = alpha_save,
        xi = xi_save,
        psi2 = psi2_save,
        beta = beta_save,
        ll = ll_save
      ),
      accept = list(
        z = acc_z / iter,
        alpha = acc_alpha / iter,
        beta = acc_beta / iter
      ),
      config = list(
        iter = iter, burnin = burnin, thinning = thinning,
        sigma2 = sigma2,
        prop_sd = list(z = prop_z, alpha = prop_alpha, beta = prop_beta),
        a0 = a0,
        b0 = b0
      )
    ))
}

# # usage
# result<-LSM(Y, iter = 5000, burnin = 1000, thinning = 2,
#             prop_z = 0.5, prop_alpha = 0.3, prop_beta = 0.1, d = 2)
# 
# mean(result$samples$alpha)
# 
# result<-LSM(Y, iter = 5000, burnin = 1000, thinning = 2,
#             prop_z = 0.5, prop_alpha = 0.3, prop_beta = 0.1, d = 3)
# # result report
# result$accept
# par(mfrow = c(3,2))
# d <- 2
# for(i in 1:dim(result$samples$z)[2]){
#   for(j in 1:d){
#     ts.plot(result$samples$z[,i,j])
#   }
# }
# 
# par(mfrow = c(1,1))
# ts.plot(result$samples$alpha)
# ts.plot(result$samples$xi)
# ts.plot(result$samples$psi2)
# ts.plot(result$samples$beta)
# ts.plot(result$samples$ll)
# 
# 
# motif_GOF_test(result, Y, B = 500, custom_text = 'ergm with karate network', arg = 'simple_motif', model = 'LSM', dist_fun = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'ergm with karate network', arg = 'kstars', model = 'LSM', dist_fun = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'ergm with karate network', arg = 'dsp', model = 'LSM', dist_fun = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'ergm with karate network', arg = 'esp', model = 'LSM', dist_fun = L2_dist)
# 
# 
# # embedding plotting
# # original network plotting
# library(network)
# par(mfrow = c(1,1))
# plot(network(Y, directed = F), displaylabels = T)
# rowSums(Y)
# # result embedding plotting
# # 1, 33, 34 어디있는지
# latentmap <- colMeans(result$samples$z, dims = 1)
# plot(latentmap,main = "colMeans(result$samples$z, dims = 1)",
#      xlab = "x-axis",
#      ylab = "y-axis",
#      col = "grey", # 기본 점 색상
#      pch = 19,
#      ylim = c(-1,1),
#      xlim = c(-1.5,1.5))
# 
# # 특정 인덱스의 점에 빨간색 칠하기
# # 3번째와 7번째 점
# red <- c(1, 33, 34)
# blue <- c(12)
# # points() 함수를 사용하여 해당 점만 덧그림
# points(x = latentmap[red,1],
#        y = latentmap[red,2],
#        col = "red",
#        pch = 19, # 기본 플롯과 동일하게 설정
#        cex = 1.5) # 점 크기를 조금 더 키워서 강조
# 
# points(x = latentmap[blue,1],
#        y = latentmap[blue,2],
#        col = "blue",
#        pch = 19, # 기본 플롯과 동일하게 설정
#        cex = 1.5) # 점 크기를 조금 더 키워서 강조
# # ---- 숫자 라벨 추가 ----
# n <- nrow(latentmap)
# lab_col <- rep("grey20", n)
# lab_col[red]  <- "red"
# lab_col[blue] <- "blue"
# 
# # 각 점 위(pos=3)에 번호 붙이기 (겹침 줄이려면 offset 늘리기)
# text(latentmap[,1], latentmap[,2],
#      labels = 1:n, col = lab_col,
#      cex = 0.8, pos = 3, offset = 0.3)
# 
# # in 3D
# # install.packages("plotly")
# library(plotly)
# 
# # latentmap: n×2 또는 n×3
# X <- as.matrix(latentmap)
# stopifnot(ncol(X) >= 2)
# if (ncol(X) == 2) X <- cbind(X, 0)
# colnames(X) <- c("x","y","z")
# 
# red  <- c(1, 33, 34)
# blue <- c(12)
# 
# n <- nrow(X)
# base_col  <- rep("grey", n)
# base_col[red]  <- "red"
# base_col[blue] <- "blue"
# base_size <- rep(6, n); base_size[c(red, blue)] <- 10
# 
# labels <- paste0("idx: ", seq_len(n))
# 
# p <- plot_ly(type = "scatter3d", mode = "markers") |>
#   add_markers(
#     x = X[,1], y = X[,2], z = X[,3],
#     marker = list(size = base_size),
#     text = labels, hoverinfo = "text",
#     color = I(base_col),
#     showlegend = FALSE
#   )
# 
# # ---- 항상 보이는 텍스트 라벨 (하이라이트 인덱스만) ----
# hl <- unique(c(red, blue))
# p <- p |>
#   add_text(
#     x = X[hl,1], y = X[hl,2], z = X[hl,3],
#     text = hl,                     # 숫자 라벨
#     textposition = "top center",
#     textfont = list(size = 12, color = base_col[hl]),
#     showlegend = FALSE,
#     inherit = FALSE
#   )
# 
# # ---- (옵션) 모든 점에 라벨을 항상 표시하려면 아래 주석 해제 ----
# # p <- p |>
# #   add_text(
# #     x = X[,1], y = X[,2], z = X[,3],
# #     text = seq_len(n),
# #     textposition = "top center",
# #     textfont = list(size = 10, color = "rgba(50,50,50,0.8)"),
# #     showlegend = FALSE,
# #     inherit = FALSE
# #   )
# 
# p <- p |>
#   layout(
#     title = "colMeans(result$samples$z, dims = 1)",
#     scene = list(
#       xaxis = list(title = "x-axis"),
#       yaxis = list(title = "y-axis"),
#       zaxis = list(title = if (ncol(latentmap) >= 3) "z-axis" else "z=0 plane"),
#       aspectmode = "cube"
#     )
#   )
# 
# p
# posterior_mean_prob_LSM(result)
# 
# 
# motif_GOF_test(result_2, Y, B = 500, custom_text = 'LSM_karate', arg = 'simple_motif', model = 'LSM')
# count_edges(Y)
# P_hat <- posterior_mean_prob_LSM(result_2)
# count_edges(simulate_from_P(P_hat))
# count_triangles(simulate_from_P(P_hat))
# count_triangles(Y)
# 
# 
# library(latentnet)
# test_ergm <- ergmm(network(Y, directed = F) ~ euclidean(d = 2),
#                    control = ergmm.control(burnin = 5000,
#                                            sample.size = 100000,
#                                            interval = 5))
# 
# test_ergm_dotprd <- ergmm(network(Y, directed = F) ~ bilinear(d = 2),
#                 control = ergmm.control(burnin = 5000,
#                                         sample.size = 100000,
#                                         interval = 5))
# 
# plot(test_ergm, labels = T)
# plot(test_ergm_dotprd, labels = T)
# 
# # ergmm mode 코드 에러 있음
# motif_GOF_test(test_ergm, Y, B = 500, custom_text = 'ergm with karate network', arg = 'simple_motif', model = 'ergm')
# motif_GOF_test(test_ergm, Y, B = 500, custom_text = 'ergm with karate network', arg = 'kstars', model = 'ergm')
# motif_GOF_test(test_ergm, Y, B = 500, custom_text = 'ergm with karate network', arg = 'dsp', model = 'ergm')
# motif_GOF_test(test_ergm, Y, B = 500, custom_text = 'ergm with karate network', arg = 'esp', model = 'ergm')
# 
# motif_GOF_test(test_ergm_dotprd, Y, B = 500, custom_text = 'ergm with karate network', arg = 'simple_motif', model = 'ergm', dot_dist)
# motif_GOF_test(test_ergm_dotprd, Y, B = 500, custom_text = 'ergm with karate network', arg = 'kstars', model = 'ergm')
# motif_GOF_test(test_ergm_dotprd, Y, B = 500, custom_text = 'ergm with karate network', arg = 'dsp', model = 'ergm')
# motif_GOF_test(test_ergm_dotprd, Y, B = 500, custom_text = 'ergm with karate network', arg = 'esp', model = 'ergm')
# 
# 
# 
# 
# # appendix
# # theoretical - empirical LSM result 비교
# 
# logit <- function(x){
#   return(log(x/(1-x)))
# }
# inv_logit <- function(x){
#   return(exp(x)/(exp(x)+1))
# }
# 
# eps <- 1e-5
# temp_mat <- Y
# temp_mat[temp_mat == 1] <- 1-eps
# temp_mat[temp_mat == 0] <- eps
# 
# delta <- logit(temp_mat)
# isSymmetric(delta)
# alpha<-mean(result$sample$alpha)
# alpha<-max(delta)
# target_mat <- alpha - delta
# diag(target_mat) <- 0
# isSymmetric(target_mat)
# S<-target_mat
# 
# n <- nrow(Y)
# J   <- diag(n) - matrix(1, n, n)/n
# B <- -0.5 * J %*% S^2 %*% J
# egS <- eigen(B, symmetric = TRUE)
# lam <- egS$values
# U   <- egS$vectors
# theo_err <- sqrt(sum(lam[3:34]^2))
# Zs <- colMeans(result$sample$z, dim =1)
# B_emp <- Zs %*% t(Zs)
# emp_err <- norm(B-B_emp ,type = "F")
# 
# c(theo_err, emp_err)
# B - (U %*% diag(lam) %*% t(U))
# 
# # --------------------------------------
# # theoretical result 와 empirical result 비교:
# # --------------------------------------
# par(mfrow = c(1, 2))
# 
# # --- (1) empirical result ---
# plot(Zs, pch = 19, col = "grey40",
#      main = "Empirical (Zs)", xlab = "Z1", ylab = "Z2")
# text(Zs[,1], Zs[,2],
#      labels = 1:nrow(Zs),
#      pos = 3, cex = 0.7, col = "black")
# 
# # 강조할 index
# highlight_idx <- c(1, 33, 34)
# points(Zs[highlight_idx, 1], Zs[highlight_idx, 2],
#        col = "red", pch = 19, cex = 1.3)
# 
# lam
# # --- (2) theoretical result ---
# theo_pt <- U[, 1:2] %*% diag(sqrt(lam[1:2]))
# plot(theo_pt, pch = 19, col = "grey40",
#      main = "Theoretical (MDS)", xlab = "Dim 1", ylab = "Dim 2")
# text(theo_pt[,1], theo_pt[,2],
#      labels = 1:nrow(theo_pt),
#      pos = 3, cex = 0.7, col = "black")
# 
# points(theo_pt[highlight_idx, 1], theo_pt[highlight_idx, 2],
#        col = "red", pch = 19, cex = 1.3)
# 
# par(mfrow = c(1,1))
# plot(g)
# # 구조 복원 성능 평가
# # theoretical result 는 점추정이므로, 공평한 비교는 아니다. 
# # 하지만 bias 존재 여부 정도는 판단할 수 있을 것이라고 생각함. 
# 
# theo_result <- list()
# theo_result$samples$z<-theo_pt
# dim(theo_result$samples$z) <- c(1, 34, 2)
# theo_result$samples$beta <- 1
# theo_result$samples$alpha <- alpha
# 
# motif_GOF_test(theo_result, Y, B = 500, custom_text = 'theoretical', arg = 'simple_motif', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'empirical', arg = 'simple_motif', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(theo_result, Y, B = 500, custom_text = 'theoretical', arg = 'kstars', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'empirical', arg = 'kstars', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(theo_result, Y, B = 500, custom_text = 'theoretical', arg = 'dsp', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'empirical', arg = 'dsp', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(theo_result, Y, B = 500, custom_text = 'theoretical', arg = 'esp', model = 'LSM', dist_func = L2_dist)
# motif_GOF_test(result, Y, B = 500, custom_text = 'empirical', arg = 'esp', model = 'LSM', dist_func = L2_dist)
# 
# 
# samp<-theo_result$samples
# theo_P<-inv_logit(alpha - L2_dist(theo_pt))
# Y[33,1]
# theo_P[33,1]
# theo_P[theo_P > 0.9] <- 1
# theo_P[theo_P <= 0.9] <- 0
# plot(network(theo_P))
# Y[29,21]
# theo_P[29,21]
# (Y - theo_P)[33,34]
