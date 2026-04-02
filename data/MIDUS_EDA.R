rm(list = ls())

library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)

################################################################################
# 0. 전처리 스크립트 실행 (v3 기준)
################################################################################
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data")
source("MIDUS_preprocess_2_v3.R")
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data")

# lsirm_all 객체에서 5-layer matrix 사용 (P1-P3-P4 통합셋, screen_case==1)
Y_bin  <- lsirm_all$Y_bin    # binary
Y_cnt  <- lsirm_all$Y_cnt    # count
Y_ord1 <- lsirm_all$Y_ord1   # ordinal-5 (MASQ)
Y_ord2 <- lsirm_all$Y_ord2   # ordinal-4 (PSQI)
Y_con  <- lsirm_all$Y_con    # continuous

cat("\n=== Matrix dimensions ===\n")
cat(sprintf("Y_bin:  %d x %d\n", nrow(Y_bin),  ncol(Y_bin)))
cat(sprintf("Y_cnt:  %d x %d\n", nrow(Y_cnt),  ncol(Y_cnt)))
cat(sprintf("Y_ord1: %d x %d\n", nrow(Y_ord1), ncol(Y_ord1)))
cat(sprintf("Y_ord2: %d x %d\n", nrow(Y_ord2), ncol(Y_ord2)))
cat(sprintf("Y_con:  %d x %d\n", nrow(Y_con),  ncol(Y_con)))


################################################################################
# 1. alpha, beta, residual 계산 함수
################################################################################

compute_residuals <- function(Y, label) {
  # Y: n x p numeric matrix (NA 없는 상태)
  if (ncol(Y) == 0) {
    cat(sprintf("[%s] 변수 없음 — skip\n", label))
    return(NULL)
  }

  n <- nrow(Y)
  p <- ncol(Y)
  Y_num <- matrix(as.numeric(Y), n, p)

  alpha <- rowMeans(Y_num)                       # 행 평균 (응답자 효과)
  beta  <- colMeans(Y_num)                       # 열 평균 (문항 효과)

  # residual: Y_{ij} - (alpha_i + beta_j)
  resid_mat <- Y_num - outer(alpha, rep(1, p)) - outer(rep(1, n), beta)

  list(
    Y       = Y_num,
    alpha   = alpha,
    beta    = beta,
    resid   = resid_mat,
    label   = label,
    n       = n,
    p       = p,
    row_ids = lsirm_all$row_ids
  )
}


################################################################################
# 2. 5개 layer 에 대해 계산
################################################################################

layers <- list(
  compute_residuals(Y_bin,  "Binary (CESD + Biomarker)"),
  compute_residuals(Y_cnt,  "Count"),
  compute_residuals(Y_ord1, "Ordinal-5 (MASQ)"),
  compute_residuals(Y_ord2, "Ordinal-4 (PSQI)"),
  compute_residuals(Y_con,  "Continuous")
)
# NULL 제거
layers <- Filter(Negate(is.null), layers)

cat("\n=== Layer summaries ===\n")
for (L in layers) {
  cat(sprintf("[%s]  n=%d, p=%d,  alpha range=[%.3f, %.3f],  beta range=[%.3f, %.3f]\n",
              L$label, L$n, L$p,
              min(L$alpha), max(L$alpha),
              min(L$beta), max(L$beta)))
}


################################################################################
# 3. 시각화 1: 개별 응답자의 alpha_i 값 (data type 별 bar plot)
################################################################################

cat("\n=== 시각화 1: alpha_i bar plots (data type 별) ===\n")

# alpha 값들을 하나의 data.frame으로 합침
alpha_df <- do.call(rbind, lapply(layers, function(L) {
  data.frame(
    respondent = 1:L$n,
    alpha      = L$alpha,
    layer      = L$label,
    stringsAsFactors = FALSE
  )
}))

# alpha_i 를 응답자 순서대로 bar plot (facet by layer)
p1 <- ggplot(alpha_df, aes(x = respondent, y = alpha)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 1) +
  facet_wrap(~ layer, ncol = 1, scales = "free_y") +
  labs(
    title = expression(paste("Row means (", alpha[i], ") by data type")),
    x = "Respondent index (i)",
    y = expression(alpha[i])
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    plot.title = element_text(hjust = 0.5)
  )

print(p1)

# PDF 저장
ggsave("plot/MIDUS_EDA_alpha_barplots.pdf", p1, width = 14, height = 10)
cat("  → plot/MIDUS_EDA_alpha_barplots.pdf 저장 완료\n")


################################################################################
# 4. 시각화 2: residual Y_{ij} - (alpha_i + beta_j) 히스토그램 (data type 별)
################################################################################

cat("\n=== 시각화 2: residual 히스토그램 (data type 별) ===\n")

resid_df <- do.call(rbind, lapply(layers, function(L) {
  data.frame(
    residual = as.vector(L$resid),
    layer    = L$label,
    stringsAsFactors = FALSE
  )
}))

p2 <- ggplot(resid_df, aes(x = residual)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 fill = "coral", color = "white", alpha = 0.8) +
  facet_wrap(~ layer, ncol = 1, scales = "free") +
  labs(
    title = expression(paste("Distribution of ", Y[ij] - (alpha[i] + beta[j]), " by data type")),
    x = expression(Y[ij] - (alpha[i] + beta[j])),
    y = "Density"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    plot.title = element_text(hjust = 0.5)
  )

print(p2)

ggsave("plot/MIDUS_EDA_residual_histograms.pdf", p2, width = 12, height = 10)
cat("  → plot/MIDUS_EDA_residual_histograms.pdf 저장 완료\n")


################################################################################
# 5. 추가: alpha, beta 기초 통계
################################################################################

cat("\n=== alpha (row mean) summary per layer ===\n")
for (L in layers) {
  cat(sprintf("\n[%s]\n", L$label))
  print(summary(L$alpha))
}

cat("\n=== beta (column mean) summary per layer ===\n")
for (L in layers) {
  cat(sprintf("\n[%s]\n", L$label))
  names(L$beta) <- colnames(
    switch(L$label,
           "Binary (CESD + Biomarker)" = Y_bin,
           "Count" = Y_cnt,
           "Ordinal-5 (MASQ)" = Y_ord1,
           "Ordinal-4 (PSQI)" = Y_ord2,
           "Continuous" = Y_con)
  )
  print(round(L$beta, 4))
}

cat("\n=== residual summary per layer ===\n")
for (L in layers) {
  cat(sprintf("\n[%s]  mean=%.6f, sd=%.4f, range=[%.4f, %.4f]\n",
              L$label, mean(L$resid), sd(L$resid),
              min(L$resid), max(L$resid)))
}

cat("\n완료!\n")
