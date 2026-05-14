rm(list = ls())
## =========================================================
## High-order transitivity PPC for ergmm (latentnet)
## =========================================================
library(Matrix)     # sparse 연산 & forceSymmetric
library(ggplot2)    # 박스플롯
library(network)    # 관찰망을 network 객체로 갖고 있다면 사용
library(ergm)
library(intergraph)
library(igraph)
library(latentnet)
library(dplyr)
# ---------- 유틸 ----------
as_sym_adj <- function(A){
  A <- Matrix((A > 0) * 1, sparse = TRUE)
  diag(A) <- 0
  A <- forceSymmetric(A, uplo = "U")
  return(A)
}

common_neighbors_mat <- function(A){
  # 각 쌍의 공동이웃 수: A^2의 (i,j)
  C <- A %*% A
  diag(C) <- 0
  return(C)
}

# ---------- (1) 공동이웃 기반 폐합 곡선 ----------
# P(Y_ij = 1 | CN_ij >= s), s in S_grid
closure_by_common_neighbors <- function(A, S_grid = 1:5){
  # A <- as_adjacency_matrix(intergraph::asIgraph(nets[[1]]))
  A <- as_sym_adj(A)
  C <- common_neighbors_mat(A)
  a <- A[upper.tri(A, diag = FALSE)]
  c <- C[upper.tri(C, diag = FALSE)]
  

  S_grid <- S_grid[S_grid >= 1]
  if(length(c)) {
    S_grid <- S_grid[S_grid <= max(c)]
    if(length(S_grid) == 0) S_grid <- 1
  } else {
    S_grid <- 1
  }
  
  res_list <- list()
  for (i in seq_along(S_grid)) {
    s <- S_grid[i]
    idx <- which(c >= s)
    den <- length(idx)
    if (den > 0) {
      num <- sum(a[idx])
      rate <- num / den
    } else {
      num <- NA_real_
      rate <- NA_real_
    }
    res_list[[i]] <- data.frame(
      metric = "CN",
      level = s,
      rate = rate,
      num = num,
      den = den
    )
  }
  
  res <- do.call(rbind, res_list)
  return(res)
}

# ---------- (2) l-스텝 경로 폐합률 ----------
# l-스텝 경로가 1개 이상 존재하는 쌍 중 엣지로 "닫힌" 비율
lwalk_closure <- function(A, L_grid = 3:4){
  stopifnot(all(L_grid >= 2))
  A <- as_sym_adj(A)
  a_ut <- A[upper.tri(A, diag = FALSE)]
  M <- A
  res <- list()
  Lmax <- max(L_grid)
  for(l in 2:Lmax){
    M <- M %*% A              # 길이 l 경로 수
    M@x <- as.numeric(M@x > 0)  # booleanize(한 개 이상 있으면 1)
    diag(M) <- 0
    if(l %in% L_grid){
      m_ut <- M[upper.tri(M, diag = FALSE)]
      idx <- which(m_ut > 0)
      den <- length(idx)
      num <- if(den) sum(a_ut[idx]) else NA_real_
      rate <- if(den) num/den else NA_real_
      res[[as.character(l)]] <- data.frame(metric = "TL", level = l, rate = rate, num = num, den = den)
    }
  }
  res <- do.call(rbind, res)
  return(res)
}

# ---------- ergmm에서 tie probability 얻기 ----------
# posterior mean tie probability 행렬 (대칭/무대각 보정)
prob_from_ergmm <- function(fit){
  P <- t(predict(fit, type = "post"))  # latentnet::ergmm predict
  P[lower.tri(P)] <- t(P)[lower.tri(P)] # keep symmetricity - ignore slight error
  diag(P) <- 0
  P
}

# ---------- 확률행렬로부터 네트워크 샘플 ----------
simulate_from_P <- function(P){
  n <- nrow(P)
  U <- matrix(runif(n*n), n, n)
  Y <- (U < P) * 1
  Y[lower.tri(Y)] <- t(Y)[lower.tri(Y)]
  diag(Y) <- 0
  as_sym_adj(Y)
}

# ---------- 관찰망 & 복제망에서 두 지표를 모두 계산 ----------
two_metrics <- function(A, S_grid = 1:5, L_grid = 3:4){
  A <- as_sym_adj(A)
  return(rbind(
    closure_by_common_neighbors(A, S_grid),
    lwalk_closure(A, L_grid)
  ))
}


# ---------- PPC: 여러 번 샘플해 박스플롯용 long 데이터 만들기 ----------
ppc_two_metrics_box <- function(A_obs, fit, B = 200, S_grid = 1:5, L_grid = 3:4){
  # 관찰망 지표
  obs_long <- two_metrics(A_obs, S_grid, L_grid)
  obs_long$what <- "obs"
  
  # posterior mean P에서 복제망 샘플
  P_hat <- prob_from_ergmm(fit)
  
  reps <- list()
  for(i in 1:B){
    A_b <- simulate_from_P(P_hat)
    df <- two_metrics(A_b, sequences, sequences)
    df$rep <- i
    df$what <- "rep"
    reps[[i]] <- df[df$metric == 'CN',]
  }
  
  # reps <- lapply(1:B, function(b){
  #   A_b <- simulate_from_P(P_hat)
  #   df <- two_metrics(A_b, S_grid, L_grid)
  #   df$rep <- b
  #   df$what <- "rep"
  #   df
  # })
  
  rep_long <- do.call(rbind, reps)
  
  list(obs = obs_long, reps = rep_long)
}


# ---------- 박스플롯 + 관찰값 오버레이 ----------
plot_two_metrics_box <- function(ppc_out, custom_text = NULL){
  obs <- ppc_out$obs[ppc_out$obs$metric =="CN",]
  reps <- ppc_out$reps[ppc_out$reps$metric =="CN",]
  merge_data <- merge(reps, obs, by = "level", all.x = TRUE)
  
  # rmse per order
  rmse_result <- merge_data %>%
    mutate(diff_sq = (rate.x - rate.y)^2) %>%
    group_by(level) %>%
    summarize(rmse = sqrt(mean(diff_sq, na.rm = TRUE)), .groups = "drop") %>%
    arrange(level)
  
  print(rmse_result)
  
  # x축 라벨: "level\nRMSE=0.123" 형태로 만들기
  lab_df  <- rmse_result %>% mutate(lab = sprintf("%s\nRMSE=%.3f", level, rmse))
  lab_vec <- setNames(lab_df$lab, as.character(lab_df$level))
  
  # 박스플롯 (복제망 분포) + 관찰망 값(빨간점)
  p <- ggplot(reps, aes(x = factor(level, levels = rmse_result$level), y = rate)) +
    geom_boxplot(outlier.shape = NA) +
    geom_point(data = obs,
               aes(x = factor(level, levels = rmse_result$level), y = rate),
               color = "red", size = 2, na.rm = TRUE) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(x = "level (s for CN)",
         y = "closure rate",
         title = "Posterior Predictive Check: High-order Transitivity") +
    scale_x_discrete(labels = lab_vec) +
    theme_bw() +
    theme(axis.text.x = element_text(vjust = 1, hjust = 0.5))
  
  # 텍스트 추가 (옵션)
  if(!is.null(custom_text)){
    p <- p + annotate("text",
                      x = Inf, y = Inf, label = custom_text,
                      hjust = 1.1, vjust = 2, size = 5, color = "blue")
  }
  
  p
}


## ======================= practice =======================
# 빈 네트워크(노드 수만 지정)
n <- 100
g0 <- network.initialize(n, directed = FALSE)

# gwesp 계수에 따라 샘플링하는 함수
sample_with_gwesp <- function(gwesp_coef, theta_edges = 1, decay = 0.5) {
  sims <- simulate(
    g0 ~ edges + gwesp(decay, fixed = TRUE),
    coef = c(theta_edges, gwesp_coef),   # edges, gwesp 순서
    nsim = 1,
    output = "network",
    control = control.simulate(
      MCMC.burnin = 1e5,
      MCMC.interval = 1e3
    )
  )
  return(sims)
}

# gwesp 값을 낮음 -> 높음으로 바꿔가며 샘플링
decay_values <- seq(from = 0.1, to = 1.2, by = 0.05)
n <- length(decay_values)

nets <- list()
for(i in 1:n){
  nets[[i]] <- sample_with_gwesp(gwesp_coef = 1.0, theta_edges = -3.5, decay = decay_values[i])
}

decay_values
# 지표 확인 (전이도 / 삼각형 수)
for (i in seq_along(nets)) {
  g_ig <- intergraph::asIgraph(nets[[i]])
  cat(sprintf("decay_par = % .2f | edges = %d | transitivity = %.3f | triangles = %d\n",
              decay_values[i],
              gsize(g_ig),
              transitivity(g_ig, type = "global"),
              sum(igraph::count_triangles(g_ig))))
  plot(g_ig,
       main = paste("decay parameter =", decay_values[i]),
       vertex.size = 6, vertex.label = NA, edge.width = 1)
}



# compute communicability
library(brainGraph)
for(i in 1:n){
  comm       <- communicability(intergraph::asIgraph(nets[[i]]))
  comm_angle <- comm / sqrt(outer(diag(comm), diag(comm)))
  comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
  comm_dist.scaled  <- matrix(scale(as.numeric(nets[[1]]$comm_dist)), nrow = 100, ncol = 100)
  
  nets[[i]]$comm <- comm
  nets[[i]]$comm_angle <- comm_angle
  nets[[i]]$comm_dist <- comm_dist
  nets[[i]]$comm_dist.scaled <- comm_dist.scaled
}

mod1 <- list()
mod2 <- list()

for(i in 1:length(nets)){
  mod1[[i]] <- ergmm(nets[[i]] ~ euclidean(d = 2),
                control = ergmm.control(burnin = 2000,
                                        sample.size = 6000,
                                        interval = 5))
  
  network::set.network.attribute(nets[[i]], "comm_dist.scaled", comm_dist.scaled)
  mod2[[i]] <- ergmm(nets[[i]] ~ edgecov("comm_dist.scaled") + euclidean(d = 2),
                control = ergmm.control(burnin = 2000,
                                        sample.size = 6000,
                                        interval = 5))
}

par(mfrow = c(1,2))
for(i in 1:length(nets)){
  obs_A <- as.matrix.network.adjacency(nets[[i]])
  sequences <- seq(from = 2, to = 10, by = 1)

  ppc_out <- ppc_two_metrics_box(obs_A, mod1[[i]], B = 1000, S_grid = sequences, L_grid = sequences)
  p1 <- plot_two_metrics_box(ppc_out, custom_text = paste0(i, "th network without communality with decay parameter ",decay_values[i]))
  
  
  ppc_out <- ppc_two_metrics_box(obs_A, mod2[[i]], B = 1000, S_grid = sequences, L_grid = sequences)
  p2 <- plot_two_metrics_box(ppc_out, custom_text = paste0(i, "th network with communality with decay parameter ",decay_values[i]))
  
  grid.arrange(p1, p2, ncol = 2)
}
# sampled network has underestimated closure rate with decay parameter lower than 0.65
# on decay parameter 0.4, 9th closure rate is overestimated than true closure rate
# with lower decay parameter(generally under 0.4) high ordered closure rate such that over 7, has been overestimated than true closure rate
# with decay parameter over 0.7, almost of all true rates are covered by boxplot.

# question: generalized overestimation and peculiar underestimation under decay parameter lower than 0.7
# 1. generalized overestimation:
# 
# 2. underestimation of high order clusure rate on lower decay parameter(relatively sparse network):
# first, with lower decay parameter, high order closure is very sparse so true rate is very small = 0. high order shared partnership is exist,
# but closure doesnt exist. Or, NA. high order shared partnership doesn't exist.
# therefore, vanilla LSM overestimate high order shared partnership closure. -> 애초에 이런 식의 shared partnership 이 존재하지 않는데 복원한거임.
# 다른 motif 는 잘 복원하는가 보자.
# ---- 1) 기본 모티프 ----
count_edges <- function(A){
  sum(A[upper.tri(A)])   # 무방향 간선 수
}

count_kstars <- function(A, k = 2){
  d <- rowSums(A)
  sum(choose(d, k))      # 각 노드별 조합
}

count_two_paths <- function(A){
  d <- rowSums(A)
  sum(choose(d, 2))      # 각 노드 중심으로 2-경로
}

count_triangles <- function(A){
  sum(diag(A %*% A %*% A)) / 6   # 무방향 삼각형 개수
}

# ---- 2) ESP / DSP 분포 (테이블) ----

# Edgewise Shared Partners
esp_table <- function(A){
  S <- common_neighbors_mat(A)
  idx <- which(upper.tri(A) & A == 1, arr.ind = TRUE)
  mvals <- S[idx]
  as.data.frame(table(m = mvals))
}

# Dyadwise Shared Partners
dsp_table <- function(A, only_nonedges = FALSE){
  S <- common_neighbors_mat(A)
  U <- upper.tri(A)
  if(only_nonedges){
    idx <- which(U & A == 0, arr.ind = TRUE)
  } else {
    idx <- which(U, arr.ind = TRUE)
  }
  mvals <- S[idx]
  as.data.frame(table(m = mvals))
}



temp_net <- as.matrix.network.adjacency(nets[[7]])

two_metrics(temp_net,sequences,sequences)


P_hat <- prob_from_ergmm(mod1[[3]])
A_b <- simulate_from_P(P_hat)
plot(as.network(A_b))
df <- two_metrics(A_b, sequences, sequences)
df[df$metric == 'CN',]
