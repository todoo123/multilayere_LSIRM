################################################################################
# LSM for weighted edges
################################################################################
# rm(list = ls())

library(network)
library(igraph)
library(brainGraph)
# g <- make_graph("Zachary")
# Y = as_adjacency_matrix(g, sparse = F)
# comm <- communicability(g)
# comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
# hist(log(comm))
# hist(log(comm_dist))

## utilities
L2_dist <- function(z) {
  n <- nrow(z)
  D <- matrix(0.0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      # distance metric
      d <- sqrt(sum((z[i,] - z[j,])^2))
      D[i,j] <- d
      D[j,i] <- d
    }
  } 
  return(D)
}

# gamma 에서 log normal 로 likelihood 바꿔서 해 보겠음.
# 해석상에서도 더 좋고, log 를 씌운 데이터 분포가 normal 에 가까워 보이기 때문이다.
# 
# loglik_conti_network <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
#   D <- L2_dist(z)
#   eta <- alpha - beta*D
#   upper <- which(upper.tri(Y), arr.ind = TRUE)
#   d <- D[upper]
#   y <- pmax(Y[upper], eps)
#   rate <- kappa * exp(-eta)
#   ll <- sum(stats::dgamma(y, shape = kappa, rate = rate, log = T))
#   return(ll)
# }


loglik_conti_network <- function(Y, z, alpha, beta, kappa, eps = 1e-12) {
  D <- L2_dist(z)
  # communicability distance: eta <- alpha + beta*D 를 사용해야 함. communicability 와 그 의미가 반대이므로.
  eta <- alpha - beta*D
  upper <- which(upper.tri(Y), arr.ind = TRUE)
  y <- pmax(Y[upper], eps)
  mu_log <- eta[upper] - kappa/2        # 평균 exp(eta)로 맞춰주기 위함
  ll <- sum(dlnorm(y, meanlog = mu_log, sdlog = sqrt(kappa), log = TRUE))
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

# log(kappa) ~ normal(a, b)
log_prior_kappa <- function(kappa, a, b) {
  return(stats::dnorm(log(kappa), a, sqrt(b), log = T))
}

## update function
update_z_j <- function(j, z, Y, alpha, sigma2, prop_z, beta, kappa, d){
  z_j <- z[j,]
  z_prop <- z
  prop_z_j <- z_j + rnorm(d, mean = 0, sd = sqrt(prop_z))
  z_prop[j, ] <- prop_z_j
  
  current_logpost <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_z_i(z_j, sigma2)
  prop_logpost <-  loglik_conti_network(Y, z_prop, alpha, beta, kappa) + log_prior_z_i(prop_z_j, sigma2)
  
  acc <- prop_logpost - current_logpost
  if (log(runif(1)) < acc) {
    return(list(z = z_prop, accepted = TRUE))
  } else {
    return(list(z = z, accepted = FALSE))
  }
}

update_alpha <- function(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa) {
  ll_curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_alpha(alpha, xi, psi2)
  alpha_prop <- alpha + rnorm(1, 0, prop_alpha)
  ll_prop <- loglik_conti_network(Y, z, alpha_prop, beta, kappa) + log_prior_alpha(alpha_prop, xi, psi2)
  acc <- ll_prop - ll_curr
  if (log(runif(1)) < acc) {
    return(list(alpha = alpha_prop, accepted = TRUE))
  } else {
    return(list(alpha = alpha, accepted = FALSE))
  }
}

update_beta <- function(beta, z, Y, alpha, prop_beta, a, b, kappa){
  # beta ~ Gamma(a,b)
  # proposal distribution - normal: need jacobian
  
  curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_beta(beta, a, b) + log(beta)
  beta_prop <- exp(log(beta) + rnorm(1, 0, prop_beta))
  
  prop <- loglik_conti_network(Y, z, alpha, beta_prop, kappa) + log_prior_beta(beta_prop, a, b) + log(beta_prop)
  if (log(runif(1)) < (prop - curr)) {
    return(list(beta=beta_prop, accepted=TRUE))
  } else {
    return(list(beta=beta, accepted=FALSE))
  }
}

update_kappa <- function(kappa, z, Y, alpha, beta, prop_kappa, m_kappa = 0, s_kappa = 1){
  # log(kappa) ~ normal(m_kappa, s_kappa)
  curr <- loglik_conti_network(Y, z, alpha, beta, kappa) + log_prior_kappa(kappa, m_kappa, s_kappa)
  kappa_prop <- exp(log(kappa) + rnorm(1, 0, prop_kappa))
  
  prop <- loglik_conti_network(Y, z, alpha, beta, kappa_prop) + log_prior_kappa(kappa_prop, m_kappa, s_kappa)
  if (log(runif(1)) < (prop - curr)) {
    return(list(kappa=kappa_prop, accepted=TRUE))
  } else {
    return(list(kappa=kappa, accepted=FALSE))
  }
}


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

LSM_conti <- function(Y, iter = 5000, burnin = 1000, thinning = 5, d = 2,
                sigma2 = 1, a0 = 0.5, b0 = 0.5, m_kappa = 0, s_kappa = 1,
                prop_z = 0.05, prop_alpha = 0.05, prop_beta = 0.05, prop_kappa = 0.2,
                alpha_init = 0, beta_init = 1, xi_init = 0, kappa_init = 1,
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
  kappa <- kappa_init
  
  # storage
  keep_idx <- seq(from = burnin + 1, to = iter, by = thinning)
  ns <- length(keep_idx)
  z_save <- array(NA_real_, dim = c(ns, n, d))
  alpha_save <- xi_save <- psi2_save <- beta_save <- kappa_save <- numeric(ns)
  ll_save <- numeric(ns)
  
  # acceptance rate
  acc_z <- rep(0L, n)
  acc_alpha  <- 0L
  acc_beta <- 0L
  acc_kappa <- 0L
  
  # storage index
  s <- 0L
  
  # sampler
  for(i in 1:iter){
    # update z_i (MH)
    for (j in 1:n) {
      up <- update_z_j(j, z, Y, alpha, sigma2, prop_z, beta, kappa, d)
      z <- up$z
      if (up$accepted) {
        acc_z[j] <- acc_z[j] + 1
      }
    }
    
    # update alpha (MH)
    up_alpha <- update_alpha(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa)
    alpha <- up_alpha$alpha
    if (up_alpha$accepted) {
      acc_alpha <- acc_alpha + 1
    }
    
    # update beta (MH)
    up_beta <- update_beta(beta, z, Y, alpha, prop_beta, a = a0, b = b0, kappa)
    beta <- up_beta$beta
    if (up_beta$accepted) {
      acc_beta <- acc_beta + 1
    }
    
    # test: remove beta
    beta <- 1
    # acc_beta <- acc_beta + 1L
    
    # update kappa (MH)
    up_kappa <- update_kappa(kappa, z, Y, alpha, beta, prop_kappa, m_kappa, s_kappa)
    kappa <- up_kappa$kappa
    if (up_kappa$accepted) {
      acc_kappa <- acc_kappa + 1
    }
    
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
      kappa_save[s] <- kappa
      ll_save[s] <- loglik_conti_network(Y, z, alpha, beta, kappa)
    }
    
    if (verbose && (i %% 100 == 0)) {
      print(paste0('iteration: ', i, ' alpha: ', alpha, ' beta: ', beta))
      print(paste0(' likelihood 1: ', loglik_conti_network(Y, z, alpha, beta, kappa)))
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
        kappa = kappa_save,
        ll = ll_save
      ),
      accept = list(
        z = acc_z / iter,
        alpha = acc_alpha / iter,
        beta = acc_beta / iter,
        kappa = acc_kappa / iter
      ),
      config = list(
        iter = iter, burnin = burnin, thinning = thinning,
        sigma2 = sigma2,
        prop_sd = list(z = prop_z, alpha = prop_alpha, beta = prop_beta, kappa = prop_kappa),
        a0 = a0,
        b0 = b0,
        m_kappa = m_kappa,
        s_kappa = s_kappa
      )
    ))
}

# # usage
# 
# comm_dist[1,]
# comm_dist[33,]
# comm_dist[34,]
# comm_dist[20,]
# comm_dist[16,]
# 
# 
# # comm 의 경우: prop_z = 0.03, prop_alpha = 0.05, prop_beta = 0.03, prop_kappa = 0.1
# # comm_dist 의 경우: prop_z = 0.01, prop_alpha = 0.05, prop_beta = 0.03, prop_kappa = 0.1
# result<-LSM_conti(comm, iter = 7000, burnin = 1000, thinning = 3, d = 2,
#             prop_z = 0.2, prop_alpha = 0.05, prop_beta = 0.1, prop_kappa = 0.1)
# 
# # result report
# result$accept
# par(mfrow = c(3,2))
# 
# for(i in 1:dim(result$samples$z)[2]){
#   for(j in 1:2){
#     ts.plot(result$samples$z[,i,j])
#   }
# }
# 
# par(mfrow = c(1,1))
# ts.plot(result$samples$alpha)
# ts.plot(result$samples$xi)
# ts.plot(result$samples$psi2)
# # beta 가 영 갈피를 못 잡는다
# ts.plot(result$samples$beta)
# ts.plot(result$samples$kappa)
# ts.plot(result$samples$ll)
# 
# # embedding plotting
# hist(comm)
# comm_trunc<-comm
# comm_trunc[comm_trunc<10] <- 0
# 
# # original network plotting
# plot(network(comm_trunc, directed = F), displaylabels = T)
# rowSums(comm)
# rowSums(comm_trunc)[4]
# 
# # result embedding plotting
# # 1, 33, 34 어디있는지
# latentmap <- colMeans(result$samples$z, dims = 1)
# plot(latentmap,main = "colMeans(result$samples$z, dims = 1)",
#      xlab = "x-axis",
#      ylab = "y-axis",
#      col = "grey", # 기본 점 색상
#      pch = 19) 
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
# # 시각화 결과와 실제 distance 결과가 비슷하다. 신기...
# # hub 인 점들이 오히려 멀리 배치되는 것은 degree 차이가 많이 나는 경우,
# # 그 차이로 인해서 distance 가 멀리 계산되기 때문인 것 같다.
# # communicability distance 가 가장 가깝게 계산되는 상황은 자기가 가진 degree 가
# # 모두 특정 node 와 공유되는 경우이다. 따라서 degree 와는 반대의 상황이 관찰된다.
# 
# sum(comm_dist[33,])
# sum(comm_dist[1,])
# sum(comm_dist[27,])
# sum(comm_dist[12,])
# sum(comm_dist[29,])
# sum(comm_dist[3,])
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
# 
# # 결과를 이용하여 communicability 다시 복원: distribution 상으로는 거의 같음. 잘 복원한 것을 확인하였음.
# # fit: LSM_conti()의 반환값
# # B : 생성할 복제 네트워크 개수
# # indices: 사용할 사후 샘플 인덱스 길이 B (미지정 시 무작위로 B개 뽑음)
# # mask: (선택) TRUE/FALSE 행렬. FALSE 위치는 0으로 강제(간선 금지). 기본 NULL(제한 없음)
# # diag_zero: 대각선 0으로 만들지 여부 (무방향 단순그래프라면 TRUE가 일반적)
# posterior_sample_weighted_networks <- function(fit, B = 1, d = 2, indices = NULL,
#                                                mask = NULL, diag_zero = TRUE) {
#   zs    <- fit$samples$z      # [S, n, d]
#   alpha <- fit$samples$alpha  # [S]
#   beta  <- fit$samples$beta   # [S]
#   kappa <- fit$samples$kappa  # [S]
#   S     <- dim(zs)[1]
#   n     <- dim(zs)[2]
#   
#   if (is.null(indices)) {
#     indices <- sample.int(S, B, replace = TRUE)
#   } else {
#     stopifnot(length(indices) == B, all(indices >= 1), all(indices <= S))
#   }
#   if (!is.null(mask)) {
#     stopifnot(is.matrix(mask), all(dim(mask) == c(n, n)))
#   }
#   
#   Yreps <- array(NA_real_, dim = c(B, n, n))
#   U <- upper.tri(matrix(0, n, n), diag = FALSE)
#   
#   for (b in seq_len(B)) {
#     s  <- indices[b]
#     Zs <- matrix(zs[s,,], nrow = n, ncol = d)
#     
#     # 거리행렬
#     D <- as.matrix(dist(Zs, method = "euclidean"))
#     
#     # 선형예측자 및 평균
#     # eta = alpha - beta * d
#     eta <- alpha[s] - beta[s]*D
#     mu  <- exp(eta)                       # E[Y | theta_s]
#     
#     # 감마 모수(rate = kappa / mu) — 네 코드의 우도와 동일 (rate = kappa*exp(-eta))
#     rate_mat <- kappa[s] / mu
#     
#     # 상삼각에서만 샘플링 후 대칭 복사
#     vals <- rgamma(sum(U), shape = kappa[s], rate = rate_mat[U])
#     Y <- matrix(0.0, n, n)
#     Y[U] <- vals
#     Y <- Y + t(Y)
#     if (diag_zero) diag(Y) <- 0
#     
#     # (선택) 마스크 적용: 금지 위치는 0으로
#     if (!is.null(mask)) {
#       Y[!mask] <- 0
#       # 대칭성 유지
#       if (!isSymmetric(mask)) {
#         Y <- (Y + t(Y))/2
#       }
#       if (diag_zero) diag(Y) <- 0
#     }
#     
#     Yreps[b,,] <- Y
#   }
#   Yreps
# }
# # fit <- LSM_conti(Y, ...)
# 
# # 1) 복제 네트워크 10개 생성 (사후샘플 무작위로 10개 뽑아서)
# Yreps <- posterior_sample_weighted_networks(result, B = 10, d = 3)
# hist(Yreps, freq = F)
# hist(comm, freq = F)
