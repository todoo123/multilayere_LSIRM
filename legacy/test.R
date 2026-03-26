rm(list = ls())
library(igraph)
library(network)
library(brainGraph)
library(latentnet)
library(invgamma)
library(MASS)
g <- make_graph("Zachary")
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSM")

# ------------------------------------------------------------------------------
# 1) binary network with two cluster
par(mfrow = c(1,1))
sizes <- c(15, 15)
pref.matrix <- matrix(c(0.50, 0.01,
                        0.01, 0.30), nrow = 2)
g_1 <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)
plot(g_1,
     vertex.color = membership(cluster_louvain(g)),
     vertex.label = V(g)$name,
     vertex.size = 9,
     layout = layout_with_fr)


A_1<-as_adjacency_matrix(g, sparse = F)
dim(A_1)

# LSM 시각화
source("my_LSM_cpp.R")
bin_res <- LSM(A_1, 
               iter = 10000, burnin = 1000, thinning = 3,
               prop_z = 0.5, prop_alpha = 0.2)

# acceptance rate
bin_res$accept
# traceplot
par(mfrow = c(2,2))
for(i in 1:30){
  for(j in 1:2){
    ts.plot(bin_res$samples$z[,i,j])
  }
}
par(mfrow = c(1,1))
ts.plot(bin_res$samples$alpha)
# posterior mean of latent position
latentmap <- colMeans(bin_res$samples$z, dims = 1)
plot(latentmap,main = "latent position of binary network",
     xlab = "x-axis",
     ylab = "y-axis",
     col = "grey", # 기본 점 색상
     pch = 19)
# highlight some points
red <- c(33, 1)
# blue <- c(12)
points(x = latentmap[red,1],
       y = latentmap[red,2],
       col = "red",
       pch = 19,
       cex = 1.5)
text(latentmap[,1], latentmap[,2],
     labels = seq_len(nrow(latentmap)),  # 1..n
     pos = 3, cex = 0.7)
# points(x = latentmap[blue,1],
#        y = latentmap[blue,2],
#        col = "blue",
#        pch = 19,
#        cex = 1.5)

# ------------------------------------------------------------------------------
# 2) continuous network with one cluster
par(mfrow = c(1,1))
sizes <- c(20, 10, 10)
pref.matrix <- matrix(c(0.90, 0.0, 0.0, 0.0, 0.60, 0.0, 0.0, 0.0, 0.30), nrow = 3)
g_2 <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)

plot(g_2,
     vertex.color = membership(cluster_louvain(g_2)),
     vertex.label = V(g_2)$name,
     vertex.size = 9,
     layout = layout_with_fr)

# g_1 의 resistance distance 사용해 보기로 함. 
library(brainGraph)
library(MASS)

A_2 <- as_adjacency_matrix(g_1, sparse = FALSE)
D <- diag(rowSums(A_2), nrow = dim(A_2)[1])
L <- D - A_2
L_ginv <- ginv(L)
volG <- sum(rowSums(A_2))
diagLp <- diag(L_ginv)
# R = diag(L+)·1^T + 1·diag(L+)^T − 2L+
R_eff <- outer(diagLp, diagLp, `+`) - 2 * L_ginv
# A_2<-communicability(g_2)
hist(log(R_eff), breaks =100)
plot(density(R_eff))
R_eff <- communicability(g)
source("my_LSM_weighted_cpp.R")
d <- 2
conti_res <- LSM_conti(R_eff, 
                       sigma2 = 0.5, 
                       iter = 10000, burnin = 1000, thinning = 2,
                       d = d,
                       a0 = 0.75, b0 = 1.5,
                       prop_z = 0.3, prop_alpha = 0.1, prop_beta = 0.1)


# identifiability issue 가 있는데, 이것은 추정 자체가 잘 되지 않는 문제로 보임
# 왜? - continuous network 가 너무 복잡해서??
# 
# result
source("utils.R")
hist(log(R_eff))
sum(R_eff < 5)
sum(R_eff < 140)
result_assess(conti_res, R_eff, model = "LSM_conti", comparison_plot = T)

# acceptance rate
conti_res$accept

# traceplot
par(mfrow = c(2,d))
for(i in 1:34){
  for(j in 1:d){
    ts.plot(conti_res$samples$z[,i,j], main = paste0(i,",",j))
  }
}

# 10   - 0.0 기준
# 15   - 0.0 기준
# 16   - 0.0 기준
# 19-1 - 0.0 기준
# 19-2 - 0.8/0.4 기준
# 21-1 - 0.0 기준
# 21-2 - 0.4/0.8 기준
# 23-1 - 0.0 기준
# 23-2 - 0.4/0.8 기준
# 25-2 - 0.0 기준
# 26-1 - 0.0 기준
# 27-1 - 0.0 기준
par(mfrow = c(1,1))
plot(g_1)
conti_res$samples$z


par(mfrow = c(1,1))
ts.plot(conti_res$samples$alpha)
ts.plot(conti_res$samples$xi)
ts.plot(conti_res$samples$psi2)
ts.plot(conti_res$samples$beta)
ts.plot(conti_res$samples$kappa)

# posterior mean of latent position
latentmap <- colMeans(conti_res$samples$z, dims = 1)
plot(latentmap,
     main = "latent position of binary network",
     xlab = "x-axis",
     ylab = "y-axis",
     col = "grey", # 기본 점 색상
     pch = 19)
text(latentmap[,1], latentmap[,2],
     labels = seq_len(nrow(latentmap)),  # 1..n
     pos = 3, cex = 0.7)
# highlight some points
red <- c(3, 18)
# blue <- c(12)
points(x = latentmap[red,1],
       y = latentmap[red,2],
       col = "red",
       pch = 19,
       cex = 1.5)
# points(x = latentmap[blue,1],
#        y = latentmap[blue,2],
#        col = "blue",
#        pch = 19,
#        cex = 1.5)

# ------------------------------------------------------------------------------
source("my_LSJM_cpp.R")
lsjm_res <- LSJM_2layer(A_1, R_eff,
                        iter = 20000, burnin = 2000, thinning = 2,
                        prop_z = 0.03,
                        prop_alpha_bin = 0.3, prop_alpha_con = 0.03,
                        prop_beta_bin = 0.7, prop_beta_con = 0.1,
                        prop_kappa = 0.2
                        )

lsjm_res$accept
par(mfrow = c(2,2))
for(i in 1:30){
  for(j in 1:2){
    ts.plot(lsjm_res$samples$z[,i,j])
  }
}

ts.plot(lsjm_res$samples$alpha1)
ts.plot(lsjm_res$samples$beta1)
ts.plot(lsjm_res$samples$xi1)
ts.plot(log(lsjm_res$samples$psi1))

ts.plot(lsjm_res$samples$alpha2)
ts.plot(lsjm_res$samples$beta2)
ts.plot(lsjm_res$samples$xi2)
ts.plot(log(lsjm_res$samples$psi2))

par(mfrow = c(1,1))
ts.plot(lsjm_res$samples$kappa2)

# posterior mean of latent position
latentmap <- colMeans(lsjm_res$samples$z, dims = 1)
plot(latentmap,main = "latent position of binary network",
     xlab = "x-axis",
     ylab = "y-axis",
     col = "grey", # 기본 점 색상
     pch = 19)
text(latentmap[,1], latentmap[,2],
     labels = seq_len(nrow(latentmap)),  # 1..n
     pos = 3, cex = 0.7)  
# highlight some points
red <- c(2, 17)
# blue <- c(12)
points(x = latentmap[red,1],
       y = latentmap[red,2],
       col = "red",
       pch = 19,
       cex = 1.5)
# points(x = latentmap[blue,1],
#        y = latentmap[blue,2],
#        col = "blue",
#        pch = 19,
#        cex = 1.5)


# ------------------------------------------------------------------------------
# MNLPM
source("my_MNLPM_cpp.R")
getwd()
par(mfrow = c(1,1))
sizes <- c(20, 10, 10)
pref.matrix <- matrix(c(0.90, 0.1, 0.1, 0.1, 0.90, 0.1, 0.1, 0.1, 0.90), nrow = 3)
g_2 <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)

plot(g_2,
     vertex.color = membership(cluster_louvain(g_2)),
     vertex.label = V(g_2)$name,
     vertex.size = 9,
     layout = layout_with_fr)

A_2 <- as_adjacency_matrix(g_2, sparse = FALSE)
R_eff <- communicability(g_2)
plot(density(log(R_eff)))

g <- graph_from_adjacency_matrix(
  log(R_eff),
  mode    = "undirected",   # 또는 "directed"
  weighted = TRUE,          # 이게 중요: weight 속성으로 들어감
  diag    = FALSE
)
plot(g)
MNLPM_res <- MNLPM_2layer_Rcpp(A_2, R_eff,
                          iter = 40000, burnin = 30000, thinning = 1,
                          d = d,
                          sigma2_theta_bin = 1, sigma2_theta_con = 0.1, 
                          prop_z = 0.5,
                          prop_u_bin = 0.9, prop_u_con = 0.05,
                          prop_theta_bin = 0.7, prop_theta_con = 0.005,
                          prop_alpha_bin = 0.1, prop_beta_bin = 0.05,
                          prop_alpha_con = 0.001, prop_beta_con = 0.01, prop_kappa = 0.2,
                          )

result_assess(MNLPM_res, R_eff, model = 'MNLPM', comparison_plot = TRUE)
MNLPM_res$accept

par(mfrow = c(2,2))
for(i in 1:30){
  for(j in 1:d){
    ts.plot(MNLPM_res$samples$z[,i,j], main = paste0(i, " ",j))
  }
}
par(mfrow = c(2,2))
for(i in 1:30){
  for(j in 1:d){
    ts.plot(MNLPM_res$samples$u1[,i,j], main = paste0(i, " ",j))
  }
}
par(mfrow = c(2,2))
# u2 - 안됨 - 근데 억셉이 다 됨
for(i in 1:30){
  for(j in 1:d){
    ts.plot(MNLPM_res$samples$u2[,i,j], main = paste0(i, " ",j))
  }
}
par(mfrow = c(2,2))
for(i in 1:30){
  ts.plot(MNLPM_res$samples$theta1[,i])
}
par(mfrow = c(2,2))
# theta2 - 안됨: 엄청 좁은 구간에서 진동한다. - 값이 그만큼 Narrow?
for(i in 1:30){
  ts.plot(MNLPM_res$samples$theta2[,i])
}

# layer 1: 상대적으로 ㄱㅊ
par(mfrow =c(2,2))
ts.plot(MNLPM_res$samples$alpha1)
ts.plot(MNLPM_res$samples$beta1)
ts.plot(MNLPM_res$samples$sigma2_1)

# layer 2: 안좋음
par(mfrow =c(2,2))
## 매우 작은 지점에서 진동
ts.plot(MNLPM_res$samples$alpha2)
## beta2 값 자체가 엄청 작음
ts.plot(MNLPM_res$samples$beta2)
ts.plot(MNLPM_res$samples$sigma2_2)
# 매우 낮은 값 가짐
ts.plot(MNLPM_res$samples$kappa2)

latent_global <- colMeans(MNLPM_res$samples$z, dim = 1)
latent_1 <- colMeans(MNLPM_res$samples$u1, dim = 1)
latent_2 <- colMeans(MNLPM_res$samples$u2, dim = 1)

par(mfrow = c(2,2 ))
plot(latent_1, main = 'layer 1')
plot(latent_global, main = 'global')
plot(latent_2, main = 'layer 2')



# 안되는 이유 - continuous: proximity 의 distribution 이 lognormal 혹은 gamma 가 아닌 경우가 있다.
# 따라서 해당 데이터에 대해 맞지 않는 분포 가정을 이용해서 fitting 하려고 하니까 추정이 안 되는 거 아니냐?!?!
# 라는 것. 
# 해서, 어떠한 proximity 마다 가질 수 있는 분포를 다 확인하고, 그것에 맞는 분포 가정을 이용해 likelihood 계산을 하는 것이
# 좋지 않겠느냐 라는 것이다.

# 위의 의견에 대한 반박 - 고정된 parameter 라면 그런데, 우리는 GLM 을 사용하므로 
# distribution 자체가 lognormal, gamma 일 필요는 없다. - 따라서 그냥 잔차를 확인하고, 
# 특정한 패턴을 나타내지 않음을 확인하면 된다.

source("utils.R")
# case 별 density plot 을 통해 본 data 의 distribution

################################################################################
# resistance distance
################################################################################

# cluster 1 개
## cluster 1 개 있고, density 가 전반적으로 높은 경우
cluster <- c(30)
prob_mat <- matrix(c(0.90), nrow = 1, ncol = 1)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## cluster 1 개 있고, density 가 전반적으로 낮은 경우
cluster <- c(30)
prob_mat <- matrix(c(0.30), nrow = 1, ncol = 1)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## density 가 높은 cluster 1 개 있고, density 가 낮은 주변부가 있는 경우
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.10, 0.10,
                     0.10, 0.10, 0.10,
                     0.10, 0.10, 0.10), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## density 가 높은 cluster 1 개 있고, density 가 낮은 주변부가 있는 경우
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.90, 0.10,
                     0.90, 0.90, 0.10,
                     0.10, 0.10, 0.10), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

# cluster 2 개 있는 상황
## 동일한 수준의 density 를 가지고 약하게 연결된 cluster 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.05,
                     0.05, 0.90), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## 하나는 높은 density, 하나는 낮은 density 를 가지고 약하게 연결된 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.05,
                     0.05, 0.30), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## 둘 다 낮은 density 를 가지는 경우
cluster <- c(15, 15)
prob_mat <- matrix(c(0.30, 0.05,
                     0.05, 0.30), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## 동일한 수준의 density 를 가지고 분절된cluster 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.00,
                     0.00, 0.90), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## 하나는 높은 density, 하나는 낮은 density 를 가지고 분절된 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.00,
                     0.00, 0.30), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

## 둘 다 낮은 density 를 가지는 경우
cluster <- c(15, 15)
prob_mat <- matrix(c(0.30, 0.00,
                     0.00, 0.30), nrow = 2, ncol = 2)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance")

# cluster 3 개 있는 상황
## density 가 순차적으로 줄어드는 cluster / 연결됨
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.05, 0.05,
                     0.05, 0.60, 0.05,
                     0.05, 0.05, 0.30), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance", output = F)

## density 가 동일한 cluster / 연결됨
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.05, 0.05,
                     0.05, 0.90, 0.05,
                     0.05, 0.05, 0.90), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance", output = F)

## density 가 순차적으로 줄어드는 cluster / 분절됨
cluster <- c(20, 20, 20)
prob_mat <- matrix(c(0.90, 0.00, 0.00,
                     0.00, 0.60, 0.00,
                     0.00, 0.00, 0.30), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance", output = F)

## density 가 동일한 cluster / 분절됨
cluster <- c(20, 20, 20)
prob_mat <- matrix(c(0.90, 0.00, 0.00,
                     0.00, 0.90, 0.00,
                     0.00, 0.00, 0.90), nrow = 3, ncol = 3)
res<-sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "resistance_distance", output = F)


################################################################################
# communicability
################################################################################
source("utils.R")
# cluster 1 개
## cluster 1 개 있고, density 가 전반적으로 높은 경우
cluster <- c(30)
prob_mat <- matrix(c(0.90), nrow = 1, ncol = 1)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability")

## cluster 1 개 있고, density 가 전반적으로 낮은 경우
cluster <- c(30)
prob_mat <- matrix(c(0.30), nrow = 1, ncol = 1)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability")

## density 가 높은 cluster 1 개 있고, density 가 낮은 주변부가 있는 경우
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.10, 0.10,
                     0.10, 0.10, 0.10,
                     0.10, 0.10, 0.10), nrow = 3, ncol = 3)
res <- sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)

plot(res$g)
diag(res$proximity) <- 0
res$proximity[res$proximity < 3] <- 0
temp <- network(res$proximity,
                directed    = FALSE,
                ignore.eval = FALSE,   # 가중치 보존
                names.eval  = "w")     # edge attribute 이름
network.vertex.names(temp) <- seq_len(dim(res$proximity)[1])
w        <- get.edge.attribute(temp, "w")
edge_lwd <- 2 * w / max(w)
plot(temp,
     displaylabels = TRUE,
     edge.lwd      = edge_lwd,
     main          = "proximity network")


## density 가 높은 cluster 1 개 있고, density 가 낮은 주변부가 있는 경우
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.90, 0.10,
                     0.90, 0.90, 0.10,
                     0.10, 0.10, 0.10), nrow = 3, ncol = 3)
sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability")

# cluster 2 개 있는 상황
## 동일한 수준의 density 를 가지고 약하게 연결된 cluster 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.01,
                     0.01, 0.90), nrow = 2, ncol = 2)
res<-sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)

plot(res$g)
diag(res$proximity) <- 0
res$proximity[res$proximity < 9] <- 0
temp <- network(res$proximity,
                directed    = FALSE,
                ignore.eval = FALSE,   # 가중치 보존
                names.eval  = "w")     # edge attribute 이름
network.vertex.names(temp) <- seq_len(dim(res$proximity)[1])
w        <- get.edge.attribute(temp, "w")
edge_lwd <- 2 * w / max(w)
plot(temp,
     displaylabels = TRUE,
     edge.lwd      = edge_lwd,
     main          = "proximity network")

## 하나는 높은 density, 하나는 낮은 density 를 가지고 약하게 연결된 구조
cluster <- c(15, 15)
prob_mat <- matrix(c(0.90, 0.05,
                     0.05, 0.30), nrow = 2, ncol = 2)
res<-sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)
par(mfrow = c(1,2))
plot(res$g)
A <- as_adjacency_matrix(res$g, sparse = F)
A[res$proximity<4] <- 0
plot(network(A, directed = F))

## 둘 다 낮은 density 를 가지는 경우
cluster <- c(15, 15)
prob_mat <- matrix(c(0.30, 0.05,
                     0.05, 0.30), nrow = 2, ncol = 2)
res <- sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)

par(mfrow = c(1,2))
plot(res$g)
diag(res$proximity)<-0
res$proximity[res$proximity<1] <- 0
plot(network(res$proximity, directed = F))

# cluster 3 개 있는 상황
## density 가 순차적으로 줄어드는 cluster / 연결됨
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.05, 0.05,
                     0.05, 0.60, 0.05,
                     0.05, 0.05, 0.30), nrow = 3, ncol = 3)
res <- sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)

par(mfrow = c(1,2))
plot(res$g)
diag(res$proximity)<-0
res$proximity[res$proximity<4] <- 0
plot(network(res$proximity, directed = F))


## density 가 동일한 cluster / 연결됨
cluster <- c(10, 10, 10)
prob_mat <- matrix(c(0.90, 0.05, 0.05,
                     0.05, 0.90, 0.05,
                     0.05, 0.05, 0.90), nrow = 3, ncol = 3)
res <- sampling_prox_dist(cluster_num = cluster, prob_mat = prob_mat, prox = "communicability", output = T)


plot(res$g)
diag(res$proximity) <- 0
res$proximity[res$proximity < 6] <- 0
temp <- network(res$proximity,
                directed    = FALSE,
                ignore.eval = FALSE,   # 가중치 보존
                names.eval  = "w")     # edge attribute 이름
network.vertex.names(temp) <- seq_len(dim(res$proximity)[1])
w        <- get.edge.attribute(temp, "w")
edge_lwd <- 2 * w / max(w)
plot(temp,
     displaylabels = TRUE,
     edge.lwd      = edge_lwd,
     main          = "proximity network")
