n_person <- 10
n_item   <- 5

# 1. 그룹 배정 (예: 앞 5명은 group 1, 뒤 5명은 group 2)
group <- c(rep(1, n_person/2), rep(2, n_person/2))
group
#>  [1] 1 1 1 1 1 2 2 2 2 2

# -------------------------------------------------
# 2. Binary 응답을 위한 그룹×문항별 성공확률 설정
#    - group 1: item 1~3 에서 더 긍정적
#    - group 2: item 3~5 에서 더 긍정적
# -------------------------------------------------
p_bin <- matrix(NA, nrow = 2, ncol = n_item)  # [group, item]

# group 1
p_bin[1, ] <- c(0.8, 0.7, 0.6, 0.3, 0.2)
# group 2
p_bin[2, ] <- c(0.2, 0.3, 0.5, 0.7, 0.8)

p_bin

# -------------------------------------------------
# 3. Likert(1~5) 응답을 위한 그룹×문항별 카테고리 확률
#    prob_likert[g, j, k] = group g, item j, value k의 확률
#    - group 1: 전반적으로 높은 점수 선호
#    - group 2: 전반적으로 낮은 점수 선호
# -------------------------------------------------
K <- 5  # 카테고리 수 (1~5)
prob_likert <- array(NA, dim = c(2, n_item, K))

# helper: softmax-like 가중치로 확률 벡터 만들기
make_probs <- function(weights) weights / sum(weights)

for (j in 1:n_item) {
  # group 1: 높은 카테고리 쪽에 가중치
  w1 <- c(1, 2, 3, 4, 5)  # 1~5 쪽으로 갈수록 많이 응답
  prob_likert[1, j, ] <- make_probs(w1)
  
  # group 2: 낮은 카테고리 쪽에 가중치
  w2 <- c(5, 4, 3, 2, 1)
  prob_likert[2, j, ] <- make_probs(w2)
}

# 예시: group 1, item 1의 Likert probs
prob_likert[1, 1, ]
# 예시: group 2, item 1의 Likert probs
prob_likert[2, 1, ]

# -------------------------------------------------
# 4. 응답 샘플링
# -------------------------------------------------
Y_bin  <- matrix(NA, nrow = n_person, ncol = n_item)
Y_likert <- matrix(NA, nrow = n_person, ncol = n_item)

for (i in 1:n_person) {
  g <- group[i]
  for (j in 1:n_item) {
    # (1) Binary 응답
    Y_bin[i, j] <- rbinom(1, size = 1, prob = p_bin[g, j])
    
    # (2) Likert(1~5) 응답
    probs <- prob_likert[g, j, ]
    # sample(1:5, size=1, prob=...) 을 사용
    Y_likert[i, j] <- sample(1:K, size = 1, prob = probs)
  }
}

colnames(Y_bin)    <- paste0("Item", 1:n_item)
rownames(Y_bin)    <- paste0("Person", 1:n_person)
colnames(Y_likert) <- paste0("Item", 1:n_item)
rownames(Y_likert) <- paste0("Person", 1:n_person)

# 결과 확인
Y_bin
Y_likert
group



fit <- lsirm_global_local_option1(
  Y_bin = Y_bin,
  Y_con = Y_likert,
  d     = 2,
  n_iter = 5000,
  burnin = 1000,
  thin   = 5
)
