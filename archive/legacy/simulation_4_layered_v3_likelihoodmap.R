# ---------- numerically stable helpers ----------
log1pexp <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
inv_logit <- function(x) 1/(1+exp(-x))
softplus  <- function(x) log1pexp(x)  # C++의 softplus=log1pexp와 동일

# u=0 고정 thresholds: b1=0, b_{k}=b_{k-1}+softplus(delta_{k-1})
build_thresholds_fixed0 <- function(delta_vec){
  Kminus1 <- 1 + length(delta_vec)
  b <- numeric(Kminus1)
  b[1] <- 0.0
  if(Kminus1 >= 2){
    for(k in 2:Kminus1){
      b[k] <- b[k-1] + softplus(delta_vec[k-1])
    }
  }
  b
}

# ordinal single log-prob (C++ 로직과 동일)
log_p_ordinal_single <- function(y_val, eta, b_vec){
  # y_val in 1..K, b_vec length K-1
  Kminus1 <- length(b_vec)
  K <- Kminus1 + 1
  Cvec <- inv_logit(eta - b_vec)  # length K-1
  
  if(y_val == 1){
    p <- 1 - Cvec[1]
  } else if(y_val == K){
    p <- Cvec[Kminus1]
  } else {
    p <- Cvec[y_val-1] - Cvec[y_val]
  }
  if(p <= 1e-16) return(-1e16)
  log(p)
}

# Euclidean distances from a single point (1x2) to a set of points (P x 2)
dist_to_points <- function(pos, B){
  sqrt(rowSums((t(t(B) - as.numeric(pos)))^2))
}


true_par <- list(
  A = A_true,
  B1 = B1_true,
  B2 = B2_true,
  B3 = B3_true,
  B4 = B4_true,
  alpha = alpha_true,
  beta1 = beta1_true,
  beta2 = beta2_true,
  beta3 = beta3_true,
  gamma = gamma_true,
  kappa = kappa_true,
  sigma0_sq = sigma_likert_latent^2,  # 연속형 레이어의 관측분산
  delta = delta_true,                 # P4 x (K_ord-2)
  K_ord = K_ord
)

loglik_a_i <- function(pos, i, Y_bin, Y_con, Y_cnt, Y_ord, par){
  A <- par$A; B1 <- par$B1; B2 <- par$B2; B3 <- par$B3; B4 <- par$B4
  alpha <- par$alpha; beta1 <- par$beta1; beta2 <- par$beta2; beta3 <- par$beta3
  gamma <- par$gamma; kappa <- par$kappa; sigma0_sq <- par$sigma0_sq
  delta <- par$delta; K_ord <- par$K_ord
  
  ll <- 0
  # layer1: binary
  if(ncol(Y_bin) > 0){
    d1 <- dist_to_points(pos, B1)
    eta <- alpha[i] - beta1 - gamma*d1
    ll <- ll + sum(Y_bin[i,]*eta - log1pexp(eta))
  }
  
  # layer2: continuous (Normal)
  if(ncol(Y_con) > 0){
    d2 <- dist_to_points(pos, B2)
    mu <- alpha[i] - beta2 - gamma*d2
    ll <- ll + sum(dnorm(Y_con[i,], mean=mu, sd=sqrt(sigma0_sq), log=TRUE))
  }
  
  # layer3: count (NegBin, size=1/kappa, prob=size/(size+mu))
  if(ncol(Y_cnt) > 0){
    d3 <- dist_to_points(pos, B3)
    mu_cnt <- exp(alpha[i] - beta3 - gamma*d3)
    size <- 1/kappa
    prob <- size/(size + mu_cnt)
    ll <- ll + sum(dnbinom(Y_cnt[i,], size=size, prob=prob, log=TRUE))
  }
  
  # layer4: ordinal
  if(ncol(Y_ord) > 0){
    for(j in 1:ncol(Y_ord)){
      d4 <- sqrt(sum((pos - B4[j,])^2))
      eta4 <- alpha[i] - gamma*d4
      thr <- build_thresholds_fixed0(delta[j,])
      ll <- ll + log_p_ordinal_single(Y_ord[i,j], eta4, thr)
    }
  }
  
  ll
}

loglik_b1_j <- function(pos, j, Y_bin, par){
  A <- par$A; alpha <- par$alpha; beta1 <- par$beta1; gamma <- par$gamma
  d <- sqrt(rowSums((A - matrix(pos, nrow(A), 2, byrow=TRUE))^2))
  eta <- alpha - beta1[j] - gamma*d
  sum(Y_bin[,j]*eta - log1pexp(eta))
}

loglik_b2_j <- function(pos, j, Y_con, par){
  A <- par$A; alpha <- par$alpha; beta2 <- par$beta2; gamma <- par$gamma; sigma0_sq <- par$sigma0_sq
  d <- sqrt(rowSums((A - matrix(pos, nrow(A), 2, byrow=TRUE))^2))
  mu <- alpha - beta2[j] - gamma*d
  sum(dnorm(Y_con[,j], mean=mu, sd=sqrt(sigma0_sq), log=TRUE))
}

loglik_b3_j <- function(pos, j, Y_cnt, par){
  A <- par$A; alpha <- par$alpha; beta3 <- par$beta3; gamma <- par$gamma; kappa <- par$kappa
  d <- sqrt(rowSums((A - matrix(pos, nrow(A), 2, byrow=TRUE))^2))
  mu_cnt <- exp(alpha - beta3[j] - gamma*d)
  size <- 1/kappa
  prob <- size/(size + mu_cnt)
  sum(dnbinom(Y_cnt[,j], size=size, prob=prob, log=TRUE))
}

loglik_b4_j <- function(pos, j, Y_ord, par){
  A <- par$A; alpha <- par$alpha; gamma <- par$gamma
  delta <- par$delta
  thr <- build_thresholds_fixed0(delta[j,])
  
  d <- sqrt(rowSums((A - matrix(pos, nrow(A), 2, byrow=TRUE))^2))
  eta <- alpha - gamma*d
  
  ll <- 0
  for(i in 1:nrow(A)){
    ll <- ll + log_p_ordinal_single(Y_ord[i,j], eta[i], thr)
  }
  ll
}

likelihood_map_2d <- function(ll_fun, center, span=3, grid_n=120, ...){
  xs <- seq(center[1]-span, center[1]+span, length.out=grid_n)
  ys <- seq(center[2]-span, center[2]+span, length.out=grid_n)
  
  Z <- outer(xs, ys, Vectorize(function(x,y) ll_fun(c(x,y), ...)))
  list(xs=xs, ys=ys, Z=Z, center=center)
}

plot_likelihood_map <- function(map, main="log-likelihood map", add_contour=TRUE){
  image(map$xs, map$ys, map$Z, xlab="Dim1", ylab="Dim2", main=main)
  if(add_contour) contour(map$xs, map$ys, map$Z, add=TRUE)
  points(map$center[1], map$center[2], pch=19)
}

plot_lik_scatter <- function(map, n_points = 8000, main="loglik scatter",
                             n_legend = 5, legend_pos = "topright"){
  # grid에서 일부만 샘플링
  xs <- map$xs; ys <- map$ys
  Z <- map$Z
  gx <- rep(xs, times=length(ys))
  gy <- rep(ys, each=length(xs))
  gz <- as.vector(Z)
  
  set.seed(1)
  idx <- sample.int(length(gz), size=min(n_points, length(gz)))
  gx <- gx[idx]; gy <- gy[idx]; gz <- gz[idx]
  
  # 색: gz를 quantile로 나눠서 팔레트
  br <- quantile(gz, probs=seq(0,1,length.out=100), na.rm=TRUE)
  col <- colorRampPalette(c("navy","cyan","yellow","red"))(99)
  zbin <- cut(gz, breaks=br, include.lowest=TRUE, labels=FALSE)
  
  plot(gx, gy, pch=16, cex=0.6, col=col[zbin],
       xlab="Dim1", ylab="Dim2", main=main)
  points(map$center[1], map$center[2], pch=19, cex=1.2)
  
  # ---- legend (quantile bins -> representative levels) ----
  # 범례는 너무 촘촘하면 보기 힘들어서 n_legend개 정도만 표시
  probs_legend <- seq(0, 1, length.out = n_legend)
  qvals <- quantile(gz, probs = probs_legend, na.rm = TRUE)
  
  # 각 qval이 어느 색 bin에 들어가는지 찾기 (br 구간 기준)
  # findInterval: returns in [1, length(br)-1]
  bins <- findInterval(qvals, br, all.inside = TRUE)
  
  # bins가 1..99 범위로 나오도록 정리
  bins <- pmax(1, pmin(99, bins))
  
  legend_labels <- paste0(round(probs_legend*100), "%: ", formatC(qvals, digits=3, format="f"))
  
  legend(legend_pos,
         legend = legend_labels,
         pch = 16,
         col = col[bins],
         pt.cex = 1.0,
         bty = "n",
         title = "log-lik (quantiles)")
}



# 실제 likelihood map
# a
par(mfrow =c(2,2))
for(index in 1:dim(true_par$A)[1]){
  i <- index
  center <- true_par$A[i,]
  map_ai <- likelihood_map_2d(
    ll_fun = loglik_a_i,
    center = center,
    span = 3, grid_n = 150,
    i = i,
    Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt, Y_ord = Y_ord,
    par = true_par
  )
  plot_lik_scatter(map_ai, main=sprintf("scatter loglik over a[%d,]", i))
}

# b1

for(index in 1:dim(true_par$B1)[1]){
  j <- index
  center <- true_par$B1[j,]
  
  map_b1j <- likelihood_map_2d(
    ll_fun = loglik_b1_j,
    center = center,
    span = 3, grid_n = 150,
    j = j, Y_bin = Y_bin, par = true_par
  )
  
  plot_lik_scatter(map_b1j, main=sprintf("loglik over b1[%d,]", j))
}


# b2
for(index in 1:dim(true_par$B2)[1]){
  j <- index
  center <- true_par$B2[j,]
  
  map_b2j <- likelihood_map_2d(
    ll_fun = loglik_b2_j,
    center = center,
    span = 3, grid_n = 150,
    j = j, Y_con = Y_con, par = true_par
  )
  
  plot_lik_scatter(map_b2j, main=sprintf("loglik over b2[%d,]", j))
}


# b3
for(index in 1:dim(true_par$B2)[1]){
  j <- index
  center <- true_par$B3[j,]
  
  map_b3j <- likelihood_map_2d(
    ll_fun = loglik_b3_j,
    center = center,
    span = 3, grid_n = 150,
    j = j, Y_cnt = Y_cnt, par = true_par
  )
  
  plot_lik_scatter(map_b3j, main=sprintf("loglik over b3[%d,]", j))
}


# b4
for(index in 1:dim(true_par$B2)[1]){
  j <- index
  center <- true_par$B4[j,]
  
  map_b4j <- likelihood_map_2d(
    ll_fun = loglik_b4_j,
    center = center,
    span = 3, grid_n = 150,
    j = j, Y_ord = Y_ord, par = true_par
  )
  
  plot_lik_scatter(map_b4j, main=sprintf("loglik over b4[%d,]", j))
}
