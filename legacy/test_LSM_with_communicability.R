rm(list = ls())
library(network)
library(ergm)
library(intergraph)
library(igraph)

set.seed(42)

# 빈 네트워크(노드 수만 지정)
n <- 100
g0 <- network.initialize(n, directed = FALSE)

# gwesp 계수에 따라 샘플링하는 함수
sample_with_gwesp <- function(gwesp_coef, theta_edges = -2, decay = 0.5) {
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
gwesp_values <- seq(from = -0.5, by = 0.1, length.out = 11)
n <- length(gwesp_values)
nets <- list()
for(i in 1:n){
  nets[[i]] <- sample_with_gwesp(gwesp_values[i])
}


# 지표 확인 (전이도 / 삼각형 수)
for (i in seq_along(nets)) {
  g_ig <- intergraph::asIgraph(nets[[i]])
  cat(sprintf("gwesp = % .1f | edges = %d | transitivity = %.3f | triangles = %d\n",
              gwesp_values[i],
              gsize(g_ig),
              transitivity(g_ig, type = "global"),
              sum(count_triangles(g_ig))))
  plot(g_ig,
       main = paste("gwesp =", gwesp_values[i]),
       vertex.size = 6, vertex.label = NA, edge.width = 1)
}


# 2. masking
mask_test <- function(g) {
  A   <- as_adjacency_matrix(intergraph::asIgraph(g), sparse = FALSE)
  idx <- which(upper.tri(A), arr.ind = TRUE)
  
  test_id   <- idx[sample(nrow(idx), size = round(0.10 * nrow(idx))), ]
  test_id_rev <- test_id[, c(2, 1)]
  test_id <- rbind(test_id, test_id_rev)
  truth_vec <- A[test_id]
  A[test_id] <- NA                         # NA = “missing” for latentnet
  list(adj_train = A, test_id = test_id, truth = truth_vec)
}
graphs <- lapply(nets, \(x) c(x, mask_test(x)))


# 3. compute communicability
library(brainGraph)
for(i in 1:n){
  comm       <- communicability(intergraph::asIgraph(nets[[i]]))
  comm_angle <- comm / sqrt(outer(diag(comm), diag(comm)))
  comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
  comm_dist.scaled  <- scale(comm_dist)
  
  graphs[[i]]$comm <- comm
  graphs[[i]]$comm_angle <- comm_angle
  graphs[[i]]$comm_dist <- comm_dist
  graphs[[i]]$comm_dist.scaled <- comm_dist.scaled
}


par(mfrow = c(3,1))
for(i in 1:n){
  hist(as.numeric(graphs[[i]]$comm), main = i)
  hist(as.numeric(graphs[[i]]$comm_dist), main = i)
  hist(as.numeric(graphs[[i]]$comm_dist.scaled), xlim = c(-5,5), main = i, breaks = 100)
}


# 4. fitting LSM with/without communicability
library(network)
library(latentnet)

fit_models <- function(adj_train, comm_dist.scaled){
  
  net_train  <- network::network(adj_train, directed = FALSE, ignore.eval=FALSE,
                                names.eval="y", vertex.attr=NULL)
  
  network::set.network.attribute(net_train, "comm_dist.scaled", comm_dist.scaled)
  
  mod1 <- ergmm(net_train ~ euclidean(d = 2),
                control = ergmm.control(burnin = 1000,
                                        sample.size = 5000,
                                        interval = 5))
  
  mod2 <- ergmm(net_train ~ edgecov("comm_dist.scaled") + euclidean(d = 2),
                control = ergmm.control(burnin = 1000,
                                        sample.size = 5000,
                                        interval = 5))
  # `latentcov()`가 쌍별 행렬형 edge covariate를 받음 :contentReference[oaicite:3]{index=3}
  
  list(m1 = mod1, m2 = mod2)
}

for(i in 1:n){
  graphs[[i]]$result <- fit_models(graphs[[i]]$adj_train, graphs[[i]]$comm_dist.scaled)
  print(i)
}


# 5. computing MSE
get_sse <- function(mod, test_id, truth){
  pmat   <- predict(mod, type = "post")     # 각 dyad의 링크 확률
  pmat   <- t(pmat)
  preds  <- pmat[test_id]
  return(mean((truth - preds)^2))
}

graphs <- lapply(graphs, \(x){
  x$sse1 <- get_sse(x$result$m1, x$test_id, x$truth)
  x$sse2 <- get_sse(x$result$m2, x$test_id, x$truth)
  x     
})


# 6. result reporting / visualization
library(tidyverse)

results <- map_dfr(graphs, \(x){
  tibble(
    model      = c("LSM", "LSM+Comm"),
    SSE        = c(x$sse1, x$sse2)
  )
})

results <- add_column(results, gwesp_values = rep(seq(from = -0.5, by = 0.1, length.out = n), each = 2))

ggplot(results, aes(gwesp_values, SSE, color = model)) +
  geom_point(size = 3) + geom_line() +
  labs(x = "gwesp_values",
       title = "Prediction MSE vs. gwesp_values")


# 7. AUC/ROC curve
library(pROC)    # ROC/AUC
library(PRROC)  # PR AUC 원하면 사용

# 1) 한 모델에서 테스트 셋 예측확률 추출
get_test_probs <- function(mod, test_id) {
  # latentnet의 predict: type = "post" 로 posterior mean p_ij 행렬 반환
  pmat <- t(predict(mod, type = "post"))   # 행렬 → (i,j) 순서 주의
  return(as.numeric(pmat[test_id[1:(dim(test_id)[1]/2),]]))                # test_id로 선택
}


# 2) 한 그래프에 대해 ROC/AUC 계산 및 ROC 객체 반환
eval_roc_one <- function(graph_result, test_id, truth) {
  p1 <- get_test_probs(graph_result$m1, test_id)
  p2 <- get_test_probs(graph_result$m2, test_id)
  
  # 수치 안정화(혹시 모를 경계값/NA)
  clamp <- function(x) pmin(pmax(x, 1e-12), 1 - 1e-12)
  p1 <- clamp(p1); p2 <- clamp(p2)
  
  roc1 <- roc(response = truth[1:(length(truth)/2)], predictor = p1, quiet = TRUE)
  roc2 <- roc(response = truth[1:(length(truth)/2)], predictor = p2, quiet = TRUE)
  
  list(roc1 = roc1, auc1 = as.numeric(auc(roc1)),
       roc2 = roc2, auc2 = as.numeric(auc(roc2)))
} 


# 3) 예: graphs[[1]]에 대해 그리기
for(i in 1:length(gwesp_values)){
  res1 <- eval_roc_one(graphs[[i]]$result, graphs[[i]]$test_id, graphs[[i]]$truth)
  
  
  graphs[[i]]$truth
  plot(res1$roc1, col = "steelblue", lwd = 2,
       legacy.axes = TRUE, main = paste0("ROC: LSM vs LSM+Communicability for gwesp:",gwesp_values[i]))
  lines(res1$roc2, col = "tomato", lwd = 2)
  abline(0, 1, lty = 3)
  legend("bottomright",
         legend = c(sprintf("LSM (AUC = %.3f)", res1$auc1),
                    sprintf("LSM + comm (AUC = %.3f)", res1$auc2)),
         col = c("steelblue", "tomato"), lwd = 2, bty = "n")
  
  # 4) (옵션) 여러 그래프의 AUC를 요약
  get_auc <- function(g) {
    out <- eval_roc_one(g$result, g$test_id, g$truth)
    c(lsm = out$auc1, lsm_comm = out$auc2)
  }
  auc_mat <- do.call(rbind, lapply(graphs, get_auc))
  colMeans(auc_mat)  # 평균 AUC 비교
}

for(i in 1:n){
  print(transitivity(intergraph::asIgraph(nets[[i]]), type = "global"))
}

# 8. motif goodness of fit
# 1). 