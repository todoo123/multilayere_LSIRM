library(Matrix)
# A: dgCMatrix (N x N) 인접행렬
katz_pairwise <- function(A, alpha) {
  # stopifnot(is(A, "dgCMatrix") || is(A, "dsCMatrix"))
  N <- nrow(A)
  I <- Diagonal(N)
  # (I - alpha A)^{-1} - I
  # 큰 그래프는 solve 대신 Matrix::solve + 적절한 메서드/프리컨디셔너 사용
  K <- solve(I - alpha * A, I) - I
  K
}

# alpha 상한
eigenval <- eigen(Y)
rho <- max(eigenval$values)
# alpha 크기가 줄어들수록 국소적인 연결성에 집중하게 됨. 
alpha_max <- 1/rho * 0.9
alpha_max <- 1/rho * 0.8
alpha_max <- 1/rho * 0.5
alpha_max <- 1/rho * 0.1
alpha_max <- 1/rho * 0.05
alpha_max
plot(g)
katz_mat <- katz_pairwise(Y, alpha_max)
katz_mat <- as.matrix((katz_mat + t(katz_mat)) / 2)
katz_result<-LSM_conti(as.matrix(katz_mat), iter = 5000, burnin = 1000, thinning = 2, d = 2,
                  prop_z = 0.06, prop_alpha = 0.14, prop_beta = 0.05, prop_kappa = 0.12)

# result report
katz_result$accept
par(mfrow = c(4,2))

for(i in 1:dim(katz_result$samples$z)[2]){
  for(j in 1:2){
    ts.plot(katz_result$samples$z[,i,j])
  }
}

par(mfrow = c(1,1))
ts.plot(katz_result$samples$alpha)
ts.plot(katz_result$samples$xi)
ts.plot(katz_result$samples$psi2)
# beta 가 영 갈피를 못 잡는다
ts.plot(katz_result$samples$beta)
ts.plot(katz_result$samples$kappa)
ts.plot(katz_result$samples$ll)

# embedding plotting
hist(comm)
comm_trunc<-comm
comm_trunc[comm_trunc<10] <- 0

# original network plotting
plot(network(comm_trunc, directed = F), displaylabels = T)
rowSums(comm)
rowSums(comm_trunc)[4]
# katz_result embedding plotting
# 1, 33, 34 어디있는지
latentmap <- colMeans(katz_result$samples$z, dims = 1)
plot(latentmap,main = "colMeans(katz_result$samples$z, dims = 1)",
     xlab = "x-axis",
     ylab = "y-axis",
     col = "grey", # 기본 점 색상
     pch = 19) 

# 특정 인덱스의 점에 빨간색 칠하기
# 3번째와 7번째 점
red <- c(1, 33, 34)
blue <- c(12)
# points() 함수를 사용하여 해당 점만 덧그림
points(x = latentmap[red,1],
       y = latentmap[red,2],
       col = "red",
       pch = 19, # 기본 플롯과 동일하게 설정
       cex = 1.5) # 점 크기를 조금 더 키워서 강조

points(x = latentmap[blue,1],
       y = latentmap[blue,2],
       col = "blue",
       pch = 19, # 기본 플롯과 동일하게 설정
       cex = 1.5) # 점 크기를 조금 더 키워서 강조
# ---- 숫자 라벨 추가 ----
n <- nrow(latentmap)
lab_col <- rep("grey20", n)
lab_col[red]  <- "red"
lab_col[blue] <- "blue"

# 각 점 위(pos=3)에 번호 붙이기 (겹침 줄이려면 offset 늘리기)
text(latentmap[,1], latentmap[,2],
     labels = 1:n, col = lab_col,
     cex = 0.8, pos = 3, offset = 0.3)

