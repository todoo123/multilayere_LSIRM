# ==========================================================================
# SEM 시뮬레이션 + LSIRM 적합 + 잠재공간 복원 검증 (v5)
# ==========================================================================
# v4 → v5 변경사항:
#   우울 4요인 중 2요인만 사용 → 4요인 모두 사용
#   긍정정서(pos_affect) 4문항, 대인관계(interpersonal) 4문항 추가
#   두 요인은 역경과 약한 직접 경로만 부여 (매개 구조에 불참)
#   → LSIRM에서 "구조적 무관련 요인"의 위치 검증 가능
#
# LSIRM 투입 구조 (32문항, 모두 문항성 있음):
#   Y_bin: ctq1~5 (5개)                          → 아동기 역경
#   Y_con: log_il6/crp/fib (3개)                  → 염증
#   Y_cnt: n_chronic/meds/vis (3개)               → 신체 건강 부담
#   Y_ord: cesd_da1~7, som1~6, pa1~4, ip1~4 (21개) → 우울 4요인
#
# 매개경로 (v4와 동일):
#   A. 역경 → 염증 → 신체적 우울     (IE_A = 0.45 × 0.50 = 0.225)
#   B. 역경 → 우울정서 → 신체적 우울  (IE_B = 0.25 × 0.30 = 0.075)
#   C. 역경 → 염증 → 신체 건강 부담   (IE_C = 0.45 × 0.45 = 0.2025)
#
# 추가 요인 (매개 구조 불참):
#   역경 → 긍정정서   (gamma_51 = 0.05, 미약)
#   역경 → 대인관계   (gamma_61 = 0.08, 미약)
#   η₅, η₆는 다른 내생변수와 직접 경로 없음
# ==========================================================================

# if (!require("lavaan")) install.packages("lavaan", repos = "https://cran.r-project.org")
library(lavaan)
library(Rcpp)
library(RcppArmadillo)
library(vegan)
library(ggplot2)
library(ggrepel)

# LSIRM 함수 로딩
sourceCpp("my_LSIRM_4layered_nonhierarchical_v3.cpp")
source("my_LSIRM_4layered_nonhierarchical_cpp.R")

set.seed(42)

cat("\n")
cat("============================================================\n")
cat("  SEM + LSIRM 시뮬레이션 v5\n")
cat("  우울 4요인 전체 / 32문항 LSIRM\n")
cat("============================================================\n\n")


# #################################################################
# SECTION A: SEM 시뮬레이션
# #################################################################

# =========================================================================
# PART 1: 참 모수 설정 (v6 — MIDUS 문헌 기반 조정)
# =========================================================================

N <- 1255

# --- 구조 경로 계수 ---
# 매개 구조 (MIDUS 문헌 기반 조정)
gamma_11 <- 0.30   # 역경→염증: (Hostinar 2015: β≈.06–.08, 잠재변수 SEM에서 약간 상향)
gamma_21 <- 0.40   # 역경→우울정서: (O'Shields: r≈.26–.28, 경로계수로 약간 하향)
gamma_31 <- 0.00   # 역경→신체적우울 직접: 0.10→0.00 (직접효과 제거, 매개만 유지)
gamma_41 <- 0.00   # 역경→건강부담 직접: 
beta_31  <- 0.30   # 염증→신체적우울: (염증-우울 연결은 β≈.08–.14 수준)
beta_32  <- 0.40   # 우울정서→신체적우울: 0.30→0.45 (같은 CES-D 하위요인 간 강한 공변)
beta_41  <- 0.30   # 염증→건강부담: (염증-건강부담도 중간 수준으로 하향)

# v5 추가: 매개 구조 불참 요인 (더 약하게)
gamma_51 <- 0.03   # 역경→긍정정서
gamma_61 <- 0.05   # 역경→대인관계

# 통제변수 계수 (6개 내생변수: η1~η6)
# 나이→염증은 실제로 강한 예측변수 (MIDUS에서 일관되게 보고)
# 그러나 일단 각 요인 별 공평한 비교를 위하여 제거함
# c_age <- c(0.25, 0.05, 0.08, 0.20, 0.06, 0.05)
# d_sex <- c(0.05, 0.25, 0.08, 0.05, 0.12, 0.08)

# --- 매개효과 ---
IE_A <- gamma_11 * beta_31; IE_B <- gamma_21 * beta_32; IE_C <- gamma_11 * beta_41
DE_som <- gamma_31; DE_hlth <- gamma_41

cat("--- 매개효과 설정 (v6 조정) ---\n")
cat(sprintf("IE_A (역경→염증→신체적우울):     %.4f\n", IE_A))
cat(sprintf("IE_B (역경→우울정서→신체적우울): %.4f\n", IE_B))
cat(sprintf("IE_C (역경→염증→신체건강부담):   %.4f\n", IE_C))
cat(sprintf("TE  (역경→신체적우울 총효과):    %.4f\n", DE_som + IE_A + IE_B))
cat(sprintf("IE_B/IE_A = %.1f배\n\n", IE_B / IE_A))

# --- 요인적재량 --- (기존 유지, MIDUS CFA와 유사)
lambda_xi   <- c(0.75, 0.70, 0.60, 0.72, 0.68)
lambda_eta1 <- c(0.80, 0.75, 0.65)
lambda_eta2 <- c(0.75, 0.72, 0.70, 0.68, 0.65, 0.63, 0.60)
lambda_eta3 <- c(0.70, 0.68, 0.65, 0.63, 0.60, 0.55)
lambda_eta4 <- c(0.70, 0.65, 0.60)
lambda_eta5 <- c(0.72, 0.68, 0.65, 0.60)
lambda_eta6 <- c(0.70, 0.66, 0.62, 0.58)

# --- 오차분산 자동 계산: Var(η_k) = 1이 되도록 역산 ---
# 위에서 아래로 순서대로 (상위 변수부터)

# η₁ (염증): η₁ = γ₁₁·ξ₁ + ζ₁
# Var(η₁) = γ₁₁² · Var(ξ₁) + ψ_ζ1 = 1
psi_zeta1 <- 1 - gamma_11^2                          # 1 - 0.09 = 0.91

# η₂ (우울정서): η₂ = γ₂₁·ξ₁ + ζ₂
psi_zeta2 <- 1 - gamma_21^2                          # 1 - 0.16 = 0.84

# η₃ (신체적우울): η₃ = β₃₁·η₁ + β₃₂·η₂ + γ₃₁·ξ₁ + ζ₃
# 이 시점에서 Var(η₁) = Var(η₂) = 1 (위에서 확보)
# Cov(η₁, η₂) = γ₁₁·γ₂₁ (공통원인 ξ₁ 공유, ζ는 독립)
# Cov(η₁, ξ₁) = γ₁₁,  Cov(η₂, ξ₁) = γ₂₁
cov_eta12 <- gamma_11 * gamma_21
expl_3 <- beta_31^2 + beta_32^2 + gamma_31^2 +
  2*beta_31*beta_32*cov_eta12 +
  2*beta_31*gamma_31*gamma_11 +
  2*beta_32*gamma_31*gamma_21
psi_zeta3 <- 1 - expl_3

# η₄ (건강부담): η₄ = β₄₁·η₁ + γ₄₁·ξ₁ + ζ₄
expl_4 <- beta_41^2 + gamma_41^2 +
  2*beta_41*gamma_41*gamma_11
psi_zeta4 <- 1 - expl_4

# η₅ (긍정정서): η₅ = γ₅₁·ξ₁ + ζ₅
psi_zeta5 <- 1 - gamma_51^2                          # 1 - 0.0009 ≈ 0.999

# η₆ (대인관계): η₆ = γ₆₁·ξ₁ + ζ₆
psi_zeta6 <- 1 - gamma_61^2                          # 1 - 0.0025 ≈ 0.998

# --- 검증 출력 ---
cat("--- 오차분산 (Var(η)=1 역산) ---\n")
cat(sprintf("  ψ_ζ1 (염증):      %.4f  → R²=%.4f\n", psi_zeta1, 1-psi_zeta1))
cat(sprintf("  ψ_ζ2 (우울정서):  %.4f  → R²=%.4f\n", psi_zeta2, 1-psi_zeta2))
cat(sprintf("  ψ_ζ3 (신체적우울):%.4f  → R²=%.4f\n", psi_zeta3, 1-psi_zeta3))
cat(sprintf("  ψ_ζ4 (건강부담):  %.4f  → R²=%.4f\n", psi_zeta4, 1-psi_zeta4))
cat(sprintf("  ψ_ζ5 (긍정정서):  %.4f  → R²=%.4f\n", psi_zeta5, 1-psi_zeta5))
cat(sprintf("  ψ_ζ6 (대인관계):  %.4f  → R²=%.4f\n", psi_zeta6, 1-psi_zeta6))

# 안전장치: ψ_ζ < 0이면 경로계수가 너무 큰 것
for (i in 1:6) {
  psi <- get(paste0("psi_zeta", i))
  if (psi <= 0) warning(sprintf("ψ_ζ%d = %.4f ≤ 0! 경로계수 합이 분산 1을 초과합니다.", i, psi))
}
# --- 문항 생성 관련 --- (기존 유지)
thresh_cesd    <- c(0.8, 1.5, 2.2)
thresh_ctq_bin <- 0.3
count_mu    <- c(log(5.0), log(6), log(8.0))
count_sigma <- c(0.7, 0.6, 0.5)

# =========================================================================
# PART 2: 데이터 생성
# =========================================================================

generate_sem_data_v5 <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  xi_1 <- rnorm(n, 0, 1)
  # age <- rnorm(n, 55, 10); age_z <- scale(age)[, 1]
  # sex <- rbinom(n, 1, 0.57)
  
  # --- 잠재변수 생성 (구조방정식 순서) ---
  zeta_1 <- rnorm(n, 0, sqrt(psi_zeta1))
  # eta_1  <- gamma_11*xi_1 + c_age[1]*age_z + d_sex[1]*sex + zeta_1   # 염증
  eta_1  <- gamma_11*xi_1 + zeta_1
  
  zeta_2 <- rnorm(n, 0, sqrt(psi_zeta2))
  # eta_2  <- gamma_21*xi_1 + c_age[2]*age_z + d_sex[2]*sex + zeta_2   # 우울정서
  eta_2  <- gamma_21*xi_1 + zeta_2   # 우울정서
  
  zeta_3 <- rnorm(n, 0, sqrt(psi_zeta3))
  # eta_3  <- beta_31*eta_1 + beta_32*eta_2 + gamma_31*xi_1 +
  #   c_age[3]*age_z + d_sex[3]*sex + zeta_3                   
  eta_3  <- beta_31*eta_1 + beta_32*eta_2 + gamma_31*xi_1 + zeta_3     # 신체적우울
  
  zeta_4 <- rnorm(n, 0, sqrt(psi_zeta4))
  # eta_4  <- beta_41*eta_1 + gamma_41*xi_1 +
  #   c_age[4]*age_z + d_sex[4]*sex + zeta_4                    
  eta_4  <- beta_41*eta_1 + gamma_41*xi_1 + zeta_4                    # 신체건강부담
  
  # v5 추가: 매개 구조 불참 요인
  zeta_5 <- rnorm(n, 0, sqrt(psi_zeta5))
  # eta_5  <- gamma_51*xi_1 + c_age[5]*age_z + d_sex[5]*sex + zeta_5   # 긍정정서
  eta_5  <- gamma_51*xi_1 + zeta_5   # 긍정정서
  
  zeta_6 <- rnorm(n, 0, sqrt(psi_zeta6))
  # eta_6  <- gamma_61*xi_1 + c_age[6]*age_z + d_sex[6]*sex + zeta_6   # 대인관계
  eta_6  <- gamma_61*xi_1 + zeta_6   # 대인관계
  
  # === 관측변수 생성 ===
  
  # (A) CTQ 이진형
  ctq <- matrix(NA, n, 5)
  for (j in 1:5) {
    y_star <- lambda_xi[j]*xi_1 + rnorm(n, 0, sqrt(1 - lambda_xi[j]^2))
    ctq[, j] <- as.integer(y_star >= thresh_ctq_bin)
  }
  colnames(ctq) <- paste0("ctq", 1:5)
  
  # (B) 바이오마커 연속형
  inflam <- matrix(NA, n, 3)
  inflam_mu <- c(log(2.0), log(3.0), log(300)); inflam_sigma <- c(0.8, 1.0, 0.3)
  for (j in 1:3) {
    y_star <- lambda_eta1[j]*eta_1 + rnorm(n, 0, sqrt(1 - lambda_eta1[j]^2))
    inflam[, j] <- exp(inflam_mu[j] + inflam_sigma[j]*y_star)
  }
  colnames(inflam) <- c("il6", "crp", "fibrinogen")
  
  # (C) CES-D 우울정서 서열형 (7문항)
  dep_aff <- matrix(NA, n, 7)
  for (j in 1:7) {
    y_star <- lambda_eta2[j]*eta_2 + rnorm(n, 0, sqrt(1 - lambda_eta2[j]^2))
    dep_aff[, j] <- cut(y_star, breaks = c(-Inf, thresh_cesd, Inf), labels = FALSE) - 1
  }
  colnames(dep_aff) <- paste0("cesd_da", 1:7)
  
  # (D) CES-D 신체적우울 서열형 (6문항)
  som_dep <- matrix(NA, n, 6)
  for (j in 1:6) {
    y_star <- lambda_eta3[j]*eta_3 + rnorm(n, 0, sqrt(1 - lambda_eta3[j]^2))
    som_dep[, j] <- cut(y_star, breaks = c(-Inf, thresh_cesd, Inf), labels = FALSE) - 1
  }
  colnames(som_dep) <- paste0("cesd_som", 1:6)
  
  # (E) 신체건강부담 빈도형 (3문항)
  health <- matrix(NA, n, 3)
  for (j in 1:3) {
    y_star <- lambda_eta4[j]*eta_4 + rnorm(n, 0, sqrt(1 - lambda_eta4[j]^2))
    health[, j] <- rpois(n, lambda = exp(count_mu[j] + count_sigma[j]*y_star))
  }
  colnames(health) <- c("n_chronic", "n_meds", "n_visits")
  
  # (F) v5 추가: CES-D 긍정정서 서열형 (4문항)
  pos_aff <- matrix(NA, n, 4)
  for (j in 1:4) {
    y_star <- lambda_eta5[j]*eta_5 + rnorm(n, 0, sqrt(1 - lambda_eta5[j]^2))
    pos_aff[, j] <- cut(y_star, breaks = c(-Inf, thresh_cesd, Inf), labels = FALSE) - 1
  }
  colnames(pos_aff) <- paste0("cesd_pa", 1:4)
  
  # (G) v5 추가: CES-D 대인관계 서열형 (4문항)
  interp <- matrix(NA, n, 4)
  for (j in 1:4) {
    y_star <- lambda_eta6[j]*eta_6 + rnorm(n, 0, sqrt(1 - lambda_eta6[j]^2))
    interp[, j] <- cut(y_star, breaks = c(-Inf, thresh_cesd, Inf), labels = FALSE) - 1
  }
  colnames(interp) <- paste0("cesd_ip", 1:4)
  
  
  # === 데이터프레임 결합 ===
  # dat <- data.frame(ctq, inflam, dep_aff, som_dep, pos_aff, interp, health,
                    # age = age_z, sex = sex)
  dat <- data.frame(ctq, inflam, dep_aff, som_dep, pos_aff, interp, health)
  
  dat$log_il6 <- log(dat$il6)
  dat$log_crp <- log(dat$crp)
  dat$log_fibrinogen <- log(dat$fibrinogen)
  dat$log_chronic <- log(dat$n_chronic + 0.5)
  dat$log_meds    <- log(dat$n_meds + 0.5)
  dat$log_visits  <- log(dat$n_visits + 0.5)
  
  attr(dat, "true_scores") <- data.frame(xi_1, eta_1, eta_2, eta_3, eta_4, eta_5, eta_6)
  return(dat)
}


cat("--- 데이터 생성 ---\n")
dat <- generate_sem_data_v5(N, seed = 42)


# ordered factor 변환 (lavaan용)
ord_items <- c(paste0("ctq", 1:5),
               paste0("cesd_da", 1:7), paste0("cesd_som", 1:6),
               paste0("cesd_pa", 1:4), paste0("cesd_ip", 1:4))
for (v in ord_items) dat[[v]] <- ordered(dat[[v]])

cat(sprintf("N = %d, 변수 = %d\n", nrow(dat), ncol(dat)))

cat("\n--- CTQ 이진형 분포 ---\n")
for (v in paste0("ctq", 1:5)) {
  x <- as.integer(as.character(dat[[v]]))
  cat(sprintf("  %s: 0=%d(%.0f%%) 1=%d(%.0f%%)\n",
              v, sum(x==0), mean(x==0)*100, sum(x==1), mean(x==1)*100))
}

cat("\n--- v5 추가 문항 분포 ---\n")
for (v in c(paste0("cesd_pa", 1:4), paste0("cesd_ip", 1:4))) {
  x <- as.integer(as.character(dat[[v]]))
  cat(sprintf("  %s: ", v))
  for (k in 0:3) cat(sprintf("%d=%d(%.0f%%) ", k, sum(x==k), mean(x==k)*100))
  cat("\n")
}
cat("\n")


# =========================================================================
# PART 3: CFA 적합
# =========================================================================

cat("============================================================\n")
cat("  PART 3: CFA 측정 모형 (7요인)\n")
cat("============================================================\n\n")

cfa_model <- '
  adversity      =~ ctq1 + ctq2 + ctq3 + ctq4 + ctq5
  inflammation   =~ log_il6 + log_crp + log_fibrinogen
  dep_affect     =~ cesd_da1 + cesd_da2 + cesd_da3 + cesd_da4 + cesd_da5 + cesd_da6 + cesd_da7
  somatic_dep    =~ cesd_som1 + cesd_som2 + cesd_som3 + cesd_som4 + cesd_som5 + cesd_som6
  pos_affect     =~ cesd_pa1 + cesd_pa2 + cesd_pa3 + cesd_pa4
  interpersonal  =~ cesd_ip1 + cesd_ip2 + cesd_ip3 + cesd_ip4
  health_burden  =~ log_chronic + log_meds + log_visits
'

cfa_ord_items <- c(paste0("ctq", 1:5),
                   paste0("cesd_da", 1:7), paste0("cesd_som", 1:6),
                   paste0("cesd_pa", 1:4), paste0("cesd_ip", 1:4))

cfa_fit <- cfa(cfa_model, data = dat,
               ordered = cfa_ord_items,
               estimator = "WLSMV")

fit_idx <- fitMeasures(cfa_fit, c("cfi","tli","rmsea","srmr"))
cat(sprintf("  CFI=%.3f  TLI=%.3f  RMSEA=%.3f  SRMR=%.3f\n\n",
            fit_idx["cfi"], fit_idx["tli"], fit_idx["rmsea"], fit_idx["srmr"]))

cfa_params <- parameterEstimates(cfa_fit, standardized = TRUE)
all_factors <- c("adversity","inflammation","dep_affect","somatic_dep",
                 "pos_affect","interpersonal","health_burden")
fcors <- cfa_params[cfa_params$op == "~~" & cfa_params$lhs != cfa_params$rhs &
                      cfa_params$lhs %in% all_factors,
                    c("lhs","rhs","std.all","pvalue")]
fcors$sig <- ifelse(fcors$pvalue < .001, "***",
                    ifelse(fcors$pvalue < .01,  "**",
                           ifelse(fcors$pvalue < .05,  "*", "ns")))

cat("--- CFA 요인 간 상관 ---\n")
for (i in 1:nrow(fcors)) {
  with(fcors[i,], cat(sprintf("  %-15s ~ %-15s: %.3f %s\n", lhs, rhs, std.all, sig)))
}


# =========================================================================
# PART 4: SEM 적합
# =========================================================================

cat("\n============================================================\n")
cat("  PART 4: SEM 적합 (7요인)\n")
cat("============================================================\n\n")

sem_model <- '
  # === 측정 모형 ===
  adversity     =~ ctq1+ctq2+ctq3+ctq4+ctq5
  inflammation  =~ log_il6+log_crp+log_fibrinogen
  dep_affect    =~ cesd_da1+cesd_da2+cesd_da3+cesd_da4+cesd_da5+cesd_da6+cesd_da7
  somatic_dep   =~ cesd_som1+cesd_som2+cesd_som3+cesd_som4+cesd_som5+cesd_som6
  pos_affect    =~ cesd_pa1+cesd_pa2+cesd_pa3+cesd_pa4
  interpersonal =~ cesd_ip1+cesd_ip2+cesd_ip3+cesd_ip4
  health_burden =~ log_chronic+log_meds+log_visits

  # === 구조 모형: 매개 경로 (v4 동일) ===
  inflammation  ~ g11*adversity
  dep_affect    ~ g21*adversity
  somatic_dep   ~ b31*inflammation + b32*dep_affect + g31*adversity
  health_burden ~ b41*inflammation + g41*adversity

  # === 구조 모형: 매개 불참 요인 (v5 추가) ===
  pos_affect    ~ g51*adversity
  interpersonal ~ g61*adversity

  # === 매개효과 정의 ===
  ie_A := g11*b31
  ie_B := g21*b32
  ie_C := g11*b41
  de_som := g31
  te_som := g31 + g11*b31 + g21*b32
  ie_C_val := g11*b41
  de_hlth := g41
  te_hlth := g41 + g11*b41
  ie_AB_diff := g11*b31 - g21*b32
'

sem_fit <- sem(sem_model, data = dat,
               ordered = cfa_ord_items,
               estimator = "WLSMV")

sem_idx <- fitMeasures(sem_fit, c("cfi","tli","rmsea","srmr"))
cat(sprintf("  CFI=%.3f  TLI=%.3f  RMSEA=%.3f  SRMR=%.3f\n",
            sem_idx["cfi"], sem_idx["tli"], sem_idx["rmsea"], sem_idx["srmr"]))

sem_params <- parameterEstimates(sem_fit, standardized = TRUE)

# 매개효과
def_comp <- sem_params[sem_params$op == ":=", c("lhs","est","std.all","pvalue")]
def_comp

# 개별 경로의 est와 std.all 확인
struct_params <- sem_params[sem_params$op == "~", ]
struct_params[struct_params$rhs == "adversity", c("lhs","rhs","est","std.all","pvalue")]
struct_params[struct_params$rhs == "inflammation", c("lhs","rhs","est","std.all","pvalue")]
struct_params[struct_params$rhs == "dep_affect", c("lhs","rhs","est","std.all","pvalue")]

cat("\n--- 매개효과 (std.all) ---\n")
for (i in 1:nrow(def_comp)) {
  with(def_comp[i,], cat(sprintf("  %-12s: std=%.3f  p=%.4f\n", lhs, std.all, pvalue)))
}

p_ab <- def_comp$pvalue[def_comp$lhs == "ie_AB_diff"]
cat(sprintf("\n  IE_A vs IE_B: p=%.4f %s\n", p_ab,
            ifelse(p_ab < .05, "→ 염증 매개 유의하게 더 큼 ✓", "→ 염증 매개 유의하게 크지 않음 x")))

# v5 추가: 매개 불참 요인의 경로계수
cat("\n--- v5 추가: 매개 불참 요인의 경로 ---\n")
struct_params <- sem_params[sem_params$op == "~", ]
for (info in list(
  list("pos_affect",    "adversity", "g51"),
  list("interpersonal", "adversity", "g61")
)) {
  row <- struct_params[struct_params$lhs == info[[1]] & struct_params$rhs == info[[2]], ]
  if (nrow(row) > 0) {
    cat(sprintf("  %s ~ %s: std=%.3f (p=%.4f) %s\n",
                info[[1]], info[[2]], row$std.all[1], row$pvalue[1],
                ifelse(row$pvalue[1] >= .05, "→ 비유의 (기대대로)", "→ 유의")))
  }
}


# =========================================================================
# PART 4B: 부트스트랩 매개효과 신뢰구간 (std.all)
# =========================================================================

cat("\n============================================================\n")
cat("  PART 4B: 부트스트랩 매개효과 신뢰구간\n")
cat("============================================================\n\n")

n_boot <- 100  # 시연용 (실제 분석시 5000회 권장)
cat(sprintf("부트스트랩 %d회 수행 중...\n", n_boot))

boot_names <- c("ie_A", "ie_B", "ie_C", "de_som", "de_hlth",
                "te_som", "te_hlth", "ie_AB_diff")
boot_results <- matrix(NA, n_boot, length(boot_names))
colnames(boot_results) <- boot_names
boot_success <- 0

for (b in 1:n_boot) {
  idx <- sample(1:N, N, replace = TRUE)
  dat_boot <- dat[idx, ]
  
  tryCatch({
    fit_b <- sem(sem_model, data = dat_boot,
                 ordered = cfa_ord_items,
                 estimator = "WLSMV", warn = FALSE)
    
    params_b <- parameterEstimates(fit_b, standardized = TRUE)
    def_b <- params_b[params_b$op == ":=", ]
    
    for (nm in boot_names) {
      val <- def_b$std.all[def_b$lhs == nm]
      if (length(val) > 0) boot_results[b, nm] <- val[1]
    }
    boot_success <- boot_success + 1
  }, error = function(e) { })
  
  if (b %% 25 == 0) cat(sprintf("  완료: %d/%d\n", b, n_boot))
}

cat(sprintf("\n부트스트랩 성공: %d/%d (%.0f%%)\n\n",
            boot_success, n_boot, boot_success / n_boot * 100))

boot_true <- c(IE_A, IE_B, IE_C, DE_som, DE_hlth,
               IE_A + IE_B + DE_som, IE_C + DE_hlth,
               IE_A - IE_B)
names(boot_true) <- boot_names

cat(sprintf("%-12s %7s %7s %18s %10s %10s\n",
            "효과", "참값", "평균", "95% CI", "유의", "참값포함"))
cat(paste(rep("-", 72), collapse = ""), "\n")
for (nm in boot_names) {
  vals <- na.omit(boot_results[, nm])
  if (length(vals) < 10) next
  ci <- quantile(vals, c(0.025, 0.975))
  m <- mean(vals)
  sig_str <- ifelse(ci[1] > 0 | ci[2] < 0, "유의", "비유의")
  inc_str <- ifelse(ci[1] <= boot_true[nm] & boot_true[nm] <= ci[2], "OK", "미포함!")
  cat(sprintf("%-12s %7.3f %7.3f [%6.3f, %6.3f] %10s %10s\n",
              nm, boot_true[nm], m, ci[1], ci[2], sig_str, inc_str))
}


# =========================================================================
# PART 4C: 반복 시뮬레이션 (모수 복원력)
# =========================================================================

cat("\n============================================================\n")
cat("  PART 4C: 반복 시뮬레이션 (모수 복원력)\n")
cat("============================================================\n\n")

n_reps <- 20  # 시연용 (실제 분석시 100회 권장)
cat(sprintf("%d회 반복 중...\n", n_reps))

# v5: g51, g61 추가
sim_params <- c("g11", "g21", "g31", "g41", "g51", "g61",
                "b31", "b32", "b41",
                "ie_A", "ie_B", "ie_C", "ie_AB_diff")
sim_true <- c(gamma_11, gamma_21, gamma_31, gamma_41, gamma_51, gamma_61,
              beta_31, beta_32, beta_41,
              IE_A, IE_B, IE_C, IE_A - IE_B)
names(sim_true) <- sim_params

std_mat  <- matrix(NA, n_reps, length(sim_params)); colnames(std_mat)  <- sim_params
est_mat  <- matrix(NA, n_reps, length(sim_params)); colnames(est_mat)  <- sim_params
pval_mat <- matrix(NA, n_reps, length(sim_params)); colnames(pval_mat) <- sim_params

get_val <- function(df, dv, iv, col) {
  val <- df[[col]][df$lhs == dv & df$rhs == iv]
  if (length(val) == 0) NA else val[1]
}

for (r in 1:n_reps) {
  dat_r <- generate_sem_data_v5(N, seed = 300 + r)
  for (v in ord_items) dat_r[[v]] <- ordered(dat_r[[v]])
  
  tryCatch({
    fit_r <- sem(sem_model, data = dat_r,
                 ordered = cfa_ord_items,
                 estimator = "WLSMV", warn = FALSE)
    
    params_r <- parameterEstimates(fit_r, standardized = TRUE)
    sr <- params_r[params_r$op == "~", ]
    dr <- params_r[params_r$op == ":=", ]
    
    # 구조 경로
    for (info in list(
      list("g11", "inflammation",  "adversity"),
      list("g21", "dep_affect",    "adversity"),
      list("g31", "somatic_dep",   "adversity"),
      list("g41", "health_burden", "adversity"),
      list("g51", "pos_affect",    "adversity"),
      list("g61", "interpersonal", "adversity"),
      list("b31", "somatic_dep",   "inflammation"),
      list("b32", "somatic_dep",   "dep_affect"),
      list("b41", "health_burden", "inflammation")
    )) {
      std_mat[r, info[[1]]]  <- get_val(sr, info[[2]], info[[3]], "std.all")
      est_mat[r, info[[1]]]  <- get_val(sr, info[[2]], info[[3]], "est")
      pval_mat[r, info[[1]]] <- get_val(sr, info[[2]], info[[3]], "pvalue")
    }
    
    # 매개효과
    for (nm in c("ie_A", "ie_B", "ie_C", "ie_AB_diff")) {
      std_mat[r, nm]  <- dr$std.all[dr$lhs == nm]
      est_mat[r, nm]  <- dr$est[dr$lhs == nm]
      pval_mat[r, nm] <- dr$pvalue[dr$lhs == nm]
    }
    
  }, error = function(e) {
    cat(sprintf("  반복 %d: 적합 실패\n", r))
  })
  
  if (r %% 5 == 0) cat(sprintf("  완료: %d/%d\n", r, n_reps))
}

# --- 결과 출력 ---
cat("\n--- 모수 복원력: est vs std.all ---\n\n")
cat(sprintf("%-12s %7s | %7s %7s %7s | %7s %7s %7s | %7s\n",
            "모수", "참값",
            "est평균", "est편향", "estRMSE",
            "std평균", "std편향", "stdRMSE", "검정력"))
cat(paste(rep("-", 95), collapse = ""), "\n")

for (p in sim_params) {
  ev <- na.omit(est_mat[, p])
  sv <- na.omit(std_mat[, p])
  pv <- na.omit(pval_mat[, p])
  if (length(ev) == 0) next
  
  cat(sprintf("%-12s %7.3f | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %6.0f%%\n",
              p, sim_true[p],
              mean(ev), mean(ev) - sim_true[p], sqrt(mean((ev - sim_true[p])^2)),
              mean(sv), mean(sv) - sim_true[p], sqrt(mean((sv - sim_true[p])^2)),
              mean(pv < .05) * 100))
}

cat("  - 검정력 < 80%: 효과 크기 대비 표본 부족\n")

# 실제로 simulation 된 데이터가 그렇게 나오는가 확인
source("simulation_SEM_LSIRM_v5_data_viz.R")

cat("\n============================================================\n")
cat("  PART 5: LSIRM 투입 데이터 분류 (32문항)\n")
cat("============================================================\n\n")

# Y_bin: CTQ 이진형
bin_vars <- paste0("ctq", 1:5)
Y_bin <- as.matrix(data.frame(lapply(dat[, bin_vars], function(x) as.integer(as.character(x)))))
storage.mode(Y_bin) <- "double"

# Y_con: 바이오마커 연속형
con_vars <- c("log_il6", "log_crp", "log_fibrinogen")
Y_con <- as.matrix(dat[, con_vars])

# Y_cnt: 신체건강부담 빈도형
cnt_vars <- c("n_chronic", "n_meds", "n_visits")
Y_cnt <- as.matrix(dat[, cnt_vars])
storage.mode(Y_cnt) <- "double"

# Y_ord: CES-D 서열형 (v5: 4요인 21문항)
ord_vars <- c(paste0("cesd_da", 1:7), paste0("cesd_som", 1:6),
              paste0("cesd_pa", 1:4), paste0("cesd_ip", 1:4))
Y_ord <- as.matrix(data.frame(lapply(dat[, ord_vars], function(x) as.integer(as.character(x)) + 1L)))
storage.mode(Y_ord) <- "integer"

cat(sprintf("Y_bin (이진형-CTQ):    %d × %d  [%s]\n",
            nrow(Y_bin), ncol(Y_bin), paste(bin_vars, collapse=", ")))
cat(sprintf("Y_con (연속형-바이오): %d × %d  [%s]\n",
            nrow(Y_con), ncol(Y_con), paste(con_vars, collapse=", ")))
cat(sprintf("Y_cnt (빈도형-건강):   %d × %d  [%s]\n",
            nrow(Y_cnt), ncol(Y_cnt), paste(cnt_vars, collapse=", ")))
cat(sprintf("Y_ord (서열형-CES-D):  %d × %d  [%s]  (1-based, max=%d)\n",
            nrow(Y_ord), ncol(Y_ord), paste(ord_vars, collapse=", "), max(Y_ord)))
cat(sprintf("총: %d문항 (age/sex 제외)\n\n",
            ncol(Y_bin)+ncol(Y_con)+ncol(Y_cnt)+ncol(Y_ord)))

# 변수-요인 매핑 (v5: 7요인)
item_info <- data.frame(
  variable = c(bin_vars, con_vars, cnt_vars, ord_vars),
  type = c(rep("binary",5), rep("continuous",3), rep("count",3), rep("ordinal",21)),
  factor = c(rep("adversity",5), rep("inflammation",3), rep("health_burden",3),
             rep("dep_affect",7), rep("somatic_dep",6),
             rep("pos_affect",4), rep("interpersonal",4)),
  stringsAsFactors = FALSE
)
cat("--- 변수-요인 매핑 ---\n")
print(table(item_info$factor, item_info$type))

# sampled data 경향성 확인
source("")


# #################################################################
# SECTION B: LSIRM 적합
# #################################################################

# =========================================================================
# PART 6: LSIRM MCMC 적합
# =========================================================================

cat("\n============================================================\n")
cat("  PART 6: LSIRM 적합 (32문항)\n")
cat("============================================================\n\n")

D <- 2

fit <- lsirm_sharedpos_layer4_lsgrm_cpp(
  Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt, Y_ord = Y_ord,
  d = D, n_iter = 50000, burnin = 1000, thin = 5,
  prop_sd = list(
    alpha=0.5, log_gamma=0.01, a=0.3,
    beta1=0.5, beta2=0.2, beta3=0.3,
    b1=0.2, b2=0.05, b3=0.1, b4=0.2,
    log_kappa=0.20, u=0.3, delta=0.3
  ),
  verbose = TRUE
)


cat("\n--- 수용률 ---\n")
acc <- fit$accept
cat(sprintf("  alpha=%.3f  b1(bin)=%.3f  b2(con)=%.3f  b3(cnt)=%.3f  b4(ord)=%.3f  thr=%.3f\n",
            mean(acc$alpha), mean(acc$b1), mean(acc$b2), mean(acc$b3), mean(acc$b4), mean(acc$thr)))
cat(sprintf("  a(person)=%.3f  log_gamma=%.3f  log_kappa=%.3f\n",
            mean(acc$a), acc$log_gamma, acc$log_kappa))

ts.plot(exp(fit$log_gamma))


# =========================================================================
# PART 7: 사후 평균 잠재 위치 + 라벨링 (d차원 일반화)
# =========================================================================

cat("\n============================================================\n")
cat(sprintf("  PART 7: 사후 평균 + 라벨링 (7요인, d=%d)\n", D))
cat("============================================================\n\n")

# 사후 평균: 각 (n_item × D) 행렬
b1_mean <- apply(fit$b1, c(2,3), mean)  # 5 × D (CTQ)
b2_mean <- apply(fit$b2, c(2,3), mean)  # 3 × D (바이오마커)
b3_mean <- apply(fit$b3, c(2,3), mean)  # 3 × D (건강부담)
b4_mean <- apply(fit$b4, c(2,3), mean)  # 21 × D (CES-D 4요인)
a_mean  <- apply(fit$a,  c(2,3), mean)  # N × D

# d차원 일반화 데이터프레임 구성 헬퍼
make_item_df <- function(mat, vars, type_label) {
  df <- as.data.frame(mat)
  colnames(df) <- paste0("dim", 1:D)
  df$variable <- vars
  df$type     <- type_label
  df
}

item_positions <- rbind(
  make_item_df(b1_mean, bin_vars, "binary"),
  make_item_df(b2_mean, con_vars, "continuous"),
  make_item_df(b3_mean, cnt_vars, "count"),
  make_item_df(b4_mean, ord_vars, "ordinal")
)

factor_map <- c(
  setNames(rep("adversity",5),     bin_vars),
  setNames(rep("inflammation",3),  con_vars),
  setNames(rep("health_burden",3), cnt_vars),
  setNames(rep("dep_affect",7),    paste0("cesd_da",1:7)),
  setNames(rep("somatic_dep",6),   paste0("cesd_som",1:6)),
  setNames(rep("pos_affect",4),    paste0("cesd_pa",1:4)),
  setNames(rep("interpersonal",4), paste0("cesd_ip",1:4))
)
item_positions$factor <- factor_map[item_positions$variable]

dim_cols <- paste0("dim", 1:D)

cat("--- 문항 잠재 위치 ---\n")
print(item_positions[, c("variable", "type", "factor", dim_cols)], digits=3)


# =========================================================================
# PART 8: 시각화 (d차원 → 2D 쌍별 투영)
# =========================================================================

cat("\n============================================================\n")
cat(sprintf("  PART 8: 시각화 (7요인, d=%d → 2D 쌍별 투영)\n", D))
cat("============================================================\n\n")

factor_colors <- c(
  adversity="#E41A1C", inflammation="#FF7F00", dep_affect="#377EB8",
  somatic_dep="#4DAF4A", pos_affect="#F781BF", interpersonal="#A65628",
  health_burden="#984EA3"
)
shape_map <- c(binary=18, continuous=16, count=17, ordinal=15)

# 응답자 위치 데이터프레임
person_df <- as.data.frame(a_mean)
colnames(person_df) <- paste0("dim", 1:D)

# --- 모든 2D 쌍에 대한 플롯 ---
dim_pairs <- combn(D, 2)  # 2×C(D,2) 행렬

for (k in 1:ncol(dim_pairs)) {
  di <- dim_pairs[1, k]
  dj <- dim_pairs[2, k]
  dx <- paste0("dim", di)
  dy <- paste0("dim", dj)
  
  p <- ggplot() +
    geom_point(data=person_df, aes(x=.data[[dx]], y=.data[[dy]]),
               color="grey80", size=0.5, alpha=0.3) +
    geom_point(data=item_positions,
               aes(x=.data[[dx]], y=.data[[dy]], color=factor, shape=type),
               size=4, alpha=0.8) +
    geom_text_repel(data=item_positions,
                    aes(x=.data[[dx]], y=.data[[dy]], label=variable, color=factor),
                    size=2.5, max.overlaps=30, show.legend=FALSE) +
    scale_color_manual(values=factor_colors, name="latent factor") +
    scale_shape_manual(values=shape_map, name="data type") +
    theme_minimal(base_size=12) +
    labs(title=sprintf("LSIRM latent space (dim%d vs dim%d)", di, dj),
         x=sprintf("dimension %d", di), y=sprintf("dimension %d", dj))
  
  fname <- sprintf("v6_items_dim%d_vs_dim%d.png", di, dj)
  ggsave(fname, p, width=13, height=9, dpi=150)
  cat(sprintf("저장: %s\n", fname))
}

# --- 군집 중심점 (d차원) ---
core_factors <- c("adversity","inflammation","dep_affect","somatic_dep",
                  "pos_affect","interpersonal","health_burden")

centroids <- do.call(rbind, lapply(core_factors, function(f) {
  sub <- item_positions[item_positions$factor == f, dim_cols, drop=FALSE]
  means <- colMeans(sub)
  df <- as.data.frame(t(means))
  df$factor <- f
  df$n <- nrow(sub)
  df
}))

# 중심점 2D 쌍별 플롯
for (k in 1:ncol(dim_pairs)) {
  di <- dim_pairs[1, k]
  dj <- dim_pairs[2, k]
  dx <- paste0("dim", di)
  dy <- paste0("dim", dj)
  
  p <- ggplot() +
    geom_point(data=item_positions,
               aes(x=.data[[dx]], y=.data[[dy]], color=factor),
               size=2, alpha=0.4) +
    geom_point(data=centroids,
               aes(x=.data[[dx]], y=.data[[dy]], color=factor),
               size=8, shape=18) +
    geom_text_repel(data=centroids,
                    aes(x=.data[[dx]], y=.data[[dy]], label=factor),
                    size=3.5, fontface="bold") +
    scale_color_manual(values=factor_colors) +
    theme_minimal(base_size=12) +
    labs(title=sprintf("요인 군집 중심점 (dim%d vs dim%d)", di, dj),
         x=sprintf("차원 %d", di), y=sprintf("차원 %d", dj))
  
  fname <- sprintf("v6_centroids_dim%d_vs_dim%d.png", di, dj)
  ggsave(fname, p, width=11, height=9, dpi=150)
  cat(sprintf("저장: %s\n", fname))
}


# =========================================================================
# PART 9: 정량적 복원력 평가 (d차원)
# =========================================================================

cat("\n============================================================\n")
cat(sprintf("  PART 9: 정량적 복원력 평가 (7요인, d=%d)\n", D))
cat("============================================================\n\n")

# --- 9.1 군집 중심점 간 거리 (d차원 유클리드) ---
centroid_mat <- as.matrix(centroids[, dim_cols])
rownames(centroid_mat) <- centroids$factor
dist_mat <- as.matrix(dist(centroid_mat))  # d차원 유클리드 거리

cat("--- 군집 중심점 간 거리 행렬 (d차원 유클리드) ---\n")
print(round(dist_mat, 3))

# --- 9.2 SEM 예측 vs LSIRM 거리 ---
cat("\n--- SEM 예측 vs LSIRM 거리 순위 ---\n")

predictions <- data.frame(
  pair = c(
    "inflammation-somatic_dep", "inflammation-health_burden",
    "adversity-inflammation", "dep_affect-somatic_dep",
    "adversity-dep_affect", "adversity-somatic_dep",
    "dep_affect-health_burden", "adversity-health_burden",
    "somatic_dep-health_burden",
    "adversity-pos_affect", "adversity-interpersonal",
    "pos_affect-inflammation", "interpersonal-inflammation",
    "pos_affect-somatic_dep", "interpersonal-somatic_dep",
    "pos_affect-health_burden", "interpersonal-health_burden",
    "pos_affect-dep_affect", "interpersonal-dep_affect",
    "pos_affect-interpersonal"
  ),
  expected = c(
    "가까움(b31=.50)", "가까움(b41=.45)", "가까움(g11=.45)",
    "중간(b32=.30)", "중간(g21=.25)", "중간(혼합경로)",
    "멀음(직접경로없음)", "멀음(g41=.10)", "중간(공통원인만)",
    "멀음(g51=.05,미약)", "멀음(g61=.08,미약)",
    "멀음(구조경로없음)", "멀음(구조경로없음)",
    "멀음(구조경로없음)", "멀음(구조경로없음)",
    "멀음(구조경로없음)", "멀음(구조경로없음)",
    "우울내부(CFA상관)", "우울내부(CFA상관)",
    "우울내부(CFA상관)"
  ),
  stringsAsFactors = FALSE
)

predictions$distance <- sapply(1:nrow(predictions), function(i) {
  p <- strsplit(predictions$pair[i], "-")[[1]]
  if (all(p %in% rownames(dist_mat))) dist_mat[p[1], p[2]] else NA
})
predictions <- predictions[!is.na(predictions$distance), ]
predictions <- predictions[order(predictions$distance), ]
predictions$rank <- 1:nrow(predictions)

cat(sprintf("%-35s %-25s %7s %4s\n", "쌍", "SEM예측", "거리", "순위"))
cat(paste(rep("-", 80), collapse=""), "\n")
for (i in 1:nrow(predictions)) {
  with(predictions[i,], cat(sprintf("%-35s %-25s %7.3f %4d\n", pair, expected, distance, rank)))
}


# --- 9.3 ARI (k=7, d차원) ---
cat("\n--- k-means (k=7) vs 참 요인 ---\n")
core_items <- item_positions[item_positions$factor %in% core_factors, ]
set.seed(123)
km <- kmeans(as.matrix(core_items[, dim_cols]), centers=7, nstart=50)
core_items$cluster <- km$cluster

compute_ari <- function(l1, l2) {
  tab <- table(l1, l2); sc <- sum(choose(tab,2))
  sr <- sum(choose(rowSums(tab),2)); scl <- sum(choose(colSums(tab),2)); n <- sum(tab)
  ex <- sr*scl/choose(n,2); mx <- (sr+scl)/2
  if (mx == ex) return(1); (sc - ex)/(mx - ex)
}

ari <- compute_ari(as.integer(as.factor(core_items$factor)), core_items$cluster)
cat(sprintf("  ARI = %.3f %s\n", ari,
            ifelse(ari > .8, "(우수)", ifelse(ari > .5, "(부분적)", "(미흡)"))))
cat("\n--- 군집-요인 교차표 ---\n")
print(table(Factor=core_items$factor, Cluster=core_items$cluster))

# 우울 4요인 ARI (k=4, d차원)
cat("\n--- v5 추가: 우울 4요인 내부 분리도 (k=4) ---\n")
dep_factors <- c("dep_affect","somatic_dep","pos_affect","interpersonal")
dep_items <- item_positions[item_positions$factor %in% dep_factors, ]
set.seed(123)
km_dep <- kmeans(as.matrix(dep_items[, dim_cols]), centers=4, nstart=50)
dep_items$cluster <- km_dep$cluster
ari_dep <- compute_ari(as.integer(as.factor(dep_items$factor)), dep_items$cluster)
cat(sprintf("  우울 4요인 ARI = %.3f %s\n", ari_dep,
            ifelse(ari_dep > .8, "(우수)", ifelse(ari_dep > .5, "(부분적)", "(미흡)"))))
cat("\n--- 우울 4요인 교차표 ---\n")
print(table(Factor=dep_items$factor, Cluster=dep_items$cluster))


# --- 9.4 CFA 상관 vs LSIRM 유사도 ---
cat("\n--- CFA 상관 vs LSIRM 거리 ---\n")
cfa_cors <- fcors[, c("lhs","rhs","std.all")]
cfa_cors$dist <- sapply(1:nrow(cfa_cors), function(i) {
  f1 <- cfa_cors$lhs[i]; f2 <- cfa_cors$rhs[i]
  if (f1 %in% rownames(dist_mat) & f2 %in% rownames(dist_mat)) dist_mat[f1,f2] else NA
})
cfa_cors$sim <- 1 / (1 + cfa_cors$dist)

cat(sprintf("%-15s %-15s %7s %7s %7s\n", "요인1", "요인2", "CFA_r", "거리", "유사도"))
cat(paste(rep("-", 60), collapse=""), "\n")
for (i in 1:nrow(cfa_cors)) {
  with(cfa_cors[i,], cat(sprintf("%-15s %-15s %7.3f %7.3f %7.3f\n", lhs, rhs, std.all, dist, sim)))
}

valid <- complete.cases(cfa_cors$std.all, cfa_cors$sim)
if (sum(valid) >= 3) {
  r_pear <- cor(cfa_cors$std.all[valid], cfa_cors$sim[valid])
  r_spear <- cor(cfa_cors$std.all[valid], cfa_cors$sim[valid], method="spearman")
  cat(sprintf("\n  Pearson r = %.3f, Spearman rho = %.3f\n", r_pear, r_spear))
  
  cfa_cors$mediation <- ifelse(
    cfa_cors$lhs %in% c("pos_affect","interpersonal") |
      cfa_cors$rhs %in% c("pos_affect","interpersonal"),
    "매개불참 포함", "매개참여만"
  )
  
  p3 <- ggplot(cfa_cors[valid,], aes(x=std.all, y=sim)) +
    geom_point(aes(color=mediation), size=3) +
    geom_text_repel(aes(label=paste0(lhs,"-",rhs)), size=2.2) +
    geom_smooth(method="lm", se=TRUE, color="red") +
    scale_color_manual(values=c("매개참여만"="black", "매개불참 포함"="steelblue")) +
    theme_minimal() +
    labs(title=sprintf("CFA상관 vs LSIRM유사도 (r=%.2f, 7요인, d=%d)", r_pear, D),
         x="CFA 요인 상관", y="LSIRM 유사도", color="매개 참여")
  ggsave("v6_cfa_vs_lsirm.png", p3, width=10, height=8, dpi=150)
  cat("저장: v6_cfa_vs_lsirm.png\n")
}


# =========================================================================
# PART 10: SEM 매개구조 공간적 검증 (d차원 거리)
# =========================================================================

cat("\n============================================================\n")
cat(sprintf("  PART 10: SEM 매개구조 공간적 검증 (d=%d)\n", D))
cat("============================================================\n\n")

d_adv_inf  <- dist_mat["adversity","inflammation"]
d_inf_som  <- dist_mat["inflammation","somatic_dep"]
d_adv_som  <- dist_mat["adversity","somatic_dep"]
d_adv_dep  <- dist_mat["adversity","dep_affect"]
d_dep_som  <- dist_mat["dep_affect","somatic_dep"]
d_inf_hlth <- dist_mat["inflammation","health_burden"]
d_adv_hlth <- dist_mat["adversity","health_burden"]
d_dep_hlth <- dist_mat["dep_affect","health_burden"]

cat("--- 경로 A: 역경→염증→신체적우울 ---\n")
cat(sprintf("  d(역경,염증)=%.3f  d(염증,신체적우울)=%.3f  d(역경,신체적우울)=%.3f\n",
            d_adv_inf, d_inf_som, d_adv_som))
cat(sprintf("  염증이 중간 위치? %s\n",
            ifelse(d_adv_inf < d_adv_som & d_inf_som < d_adv_som, "YES ✓", "NO")))

cat("\n--- 경로 B: 역경→우울정서→신체적우울 ---\n")
cat(sprintf("  d(역경,우울정서)=%.3f  d(우울정서,신체적우울)=%.3f  d(역경,신체적우울)=%.3f\n",
            d_adv_dep, d_dep_som, d_adv_som))
cat(sprintf("  우울정서가 중간 위치? %s\n",
            ifelse(d_adv_dep < d_adv_som & d_dep_som < d_adv_som, "YES ✓", "NO")))

cat("\n--- 경로 C: 역경→염증→신체건강부담 ---\n")
cat(sprintf("  d(역경,염증)=%.3f  d(염증,건강부담)=%.3f  d(역경,건강부담)=%.3f\n",
            d_adv_inf, d_inf_hlth, d_adv_hlth))
cat(sprintf("  염증이 중간 위치? %s\n",
            ifelse(d_adv_inf < d_adv_hlth & d_inf_hlth < d_adv_hlth, "YES ✓", "NO")))

cat("\n--- 핵심: 건강부담-우울정서 독립성 ---\n")
cat(sprintf("  d(건강부담, 우울정서) = %.3f\n", d_dep_hlth))
cat(sprintf("  d(건강부담, 염증)     = %.3f\n", d_inf_hlth))
cat(sprintf("  건강부담이 우울정서보다 염증에 더 가까운가? %s\n",
            ifelse(d_inf_hlth < d_dep_hlth, "YES ✓ (독립성 확인)", "NO ✗")))


# --- v5 추가: 매개 불참 요인의 공간적 독립성 검증 ---
cat("\n--- v5 추가: 매개 불참 요인의 공간적 독립성 ---\n\n")

d_pa_inf  <- dist_mat["pos_affect","inflammation"]
d_pa_som  <- dist_mat["pos_affect","somatic_dep"]
d_pa_hlth <- dist_mat["pos_affect","health_burden"]
d_pa_dep  <- dist_mat["pos_affect","dep_affect"]
d_pa_adv  <- dist_mat["pos_affect","adversity"]
d_ip_inf  <- dist_mat["interpersonal","inflammation"]
d_ip_som  <- dist_mat["interpersonal","somatic_dep"]
d_ip_hlth <- dist_mat["interpersonal","health_burden"]
d_ip_dep  <- dist_mat["interpersonal","dep_affect"]
d_ip_adv  <- dist_mat["interpersonal","adversity"]

cat("--- 긍정정서(pos_affect) ---\n")
cat(sprintf("  d(긍정정서, 역경)    = %.3f  (기대: 멀음, g51=0.05)\n", d_pa_adv))
cat(sprintf("  d(긍정정서, 염증)    = %.3f  (기대: 멀음, 경로 없음)\n", d_pa_inf))
cat(sprintf("  d(긍정정서, 신체우울) = %.3f  (기대: 멀음, 경로 없음)\n", d_pa_som))
cat(sprintf("  d(긍정정서, 건강부담) = %.3f  (기대: 멀음, 경로 없음)\n", d_pa_hlth))
cat(sprintf("  d(긍정정서, 우울정서) = %.3f  (기대: 우울 내부 CFA 상관)\n", d_pa_dep))

cat("\n--- 대인관계(interpersonal) ---\n")
cat(sprintf("  d(대인관계, 역경)    = %.3f  (기대: 멀음, g61=0.08)\n", d_ip_adv))
cat(sprintf("  d(대인관계, 염증)    = %.3f  (기대: 멀음, 경로 없음)\n", d_ip_inf))
cat(sprintf("  d(대인관계, 신체우울) = %.3f  (기대: 멀음, 경로 없음)\n", d_ip_som))
cat(sprintf("  d(대인관계, 건강부담) = %.3f  (기대: 멀음, 경로 없음)\n", d_ip_hlth))
cat(sprintf("  d(대인관계, 우울정서) = %.3f  (기대: 우울 내부 CFA 상관)\n", d_ip_dep))

cat("\n--- 핵심 검증: 매개 불참 요인 vs 매개 구조 변수 ---\n")
d_nonmed_to_med <- mean(c(d_pa_inf, d_pa_hlth, d_ip_inf, d_ip_hlth))
d_nonmed_to_dep <- mean(c(d_pa_dep, d_pa_som, d_ip_dep, d_ip_som))
d_meddep_to_med <- mean(c(d_inf_som, d_dep_som, d_inf_hlth, d_dep_hlth))

cat(sprintf("  매개불참 → 매개변수(염증,건강) 평균거리: %.3f\n", d_nonmed_to_med))
cat(sprintf("  매개불참 → 우울요인(정서,신체) 평균거리: %.3f\n", d_nonmed_to_dep))
cat(sprintf("  매개참여 우울 → 매개변수 평균거리:       %.3f\n", d_meddep_to_med))

cat(sprintf("\n  매개불참이 매개변수와 더 먼가? %s\n",
            ifelse(d_nonmed_to_med > d_meddep_to_med,
                   "YES ✓ (구조적 경로의 공간적 반영 확인)",
                   "NO ✗")))
cat(sprintf("  매개불참이 우울 내부에 더 가까운가? %s\n",
            ifelse(d_nonmed_to_dep < d_nonmed_to_med,
                   "YES ✓ (CFA 상관만 반영, 매개 구조 미참여 확인)",
                   "NO ✗")))


# =========================================================================
# PART 10B: 차원별 기여도 분석 (d≥3일 때 유용)
# =========================================================================

if (D >= 3) {
  cat("\n============================================================\n")
  cat("  PART 10B: 차원별 기여도 분석\n")
  cat("============================================================\n\n")
  
  # 각 차원에서의 요인 분산 (어떤 차원이 어떤 구조를 담당하는지)
  cat("--- 요인별 차원별 분산 ---\n")
  cat(sprintf("%-15s", "요인"))
  for (dd in 1:D) cat(sprintf(" dim%d_var", dd))
  cat("  total_var\n")
  cat(paste(rep("-", 15 + D*10 + 12), collapse=""), "\n")
  
  for (f in core_factors) {
    sub <- item_positions[item_positions$factor == f, dim_cols, drop=FALSE]
    vars <- apply(sub, 2, var)
    cat(sprintf("%-15s", f))
    for (dd in 1:D) cat(sprintf(" %8.4f", vars[dd]))
    cat(sprintf("  %8.4f\n", sum(vars)))
  }
  
  # 각 차원 쌍에서의 중심점 거리 분해
  cat("\n--- 주요 경로의 차원별 거리 분해 ---\n")
  key_pairs <- list(
    c("adversity","inflammation"),
    c("adversity","somatic_dep"),
    c("adversity","dep_affect"),
    c("inflammation","somatic_dep"),
    c("inflammation","health_burden"),
    c("dep_affect","somatic_dep")
  )
  
  cat(sprintf("%-35s %7s", "쌍", "총거리"))
  for (dd in 1:D) cat(sprintf(" dim%d_sq", dd))
  cat("\n")
  cat(paste(rep("-", 35 + 8 + D*8), collapse=""), "\n")
  
  for (pr in key_pairs) {
    c1 <- centroid_mat[pr[1], ]
    c2 <- centroid_mat[pr[2], ]
    diff_sq <- (c1 - c2)^2
    total_d <- sqrt(sum(diff_sq))
    pair_label <- paste(pr, collapse="-")
    cat(sprintf("%-35s %7.3f", pair_label, total_d))
    for (dd in 1:D) cat(sprintf(" %7.3f", diff_sq[dd]))
    cat("\n")
  }
  cat("(dim_sq: 각 차원에서의 거리 제곱 기여분, 합의 제곱근 = 총거리)\n")
}


# =========================================================================
# PART 11: 최종 요약
# =========================================================================

cat("\n\n============================================================\n")
cat(sprintf("  최종 요약 (v6: 7요인, d=%d)\n", D))
cat("============================================================\n\n")

cat(sprintf("1. 데이터: 32문항 (이진5 + 연속3 + 빈도3 + 서열21), age/sex 제외\n"))
cat(sprintf("2. LSIRM 잠재 차원: d=%d\n", D))
cat(sprintf("3. CFA (7요인): CFI=%.3f, RMSEA=%.3f\n", fit_idx["cfi"], fit_idx["rmsea"]))
cat(sprintf("4. SEM: IE_A=%.3f > IE_B=%.3f (p=%.4f)\n",
            def_comp$std.all[def_comp$lhs=="ie_A"],
            def_comp$std.all[def_comp$lhs=="ie_B"], p_ab))
cat(sprintf("5. SEM 매개불참 경로: g51(긍정정서)=%.3f, g61(대인관계)=%.3f (기대: 비유의)\n",
            gamma_51, gamma_61))
cat(sprintf("6. LSIRM ARI (7군집) = %.3f\n", ari))
cat(sprintf("7. 우울 4요인 ARI = %.3f\n", ari_dep))
if (exists("r_pear")) {
  cat(sprintf("8. CFA~LSIRM: Pearson r=%.3f, Spearman rho=%.3f\n", r_pear, r_spear))
}
cat(sprintf("9. 독립성(v4): d(건강부담,우울정서)=%.3f > d(건강부담,염증)=%.3f → %s\n",
            d_dep_hlth, d_inf_hlth,
            ifelse(d_inf_hlth < d_dep_hlth, "확인 ✓", "미확인 ✗")))
cat(sprintf("10. 독립성(v5): 매개불참→매개변수 거리(%.3f) > 매개참여→매개변수 거리(%.3f) → %s\n",
            d_nonmed_to_med, d_meddep_to_med,
            ifelse(d_nonmed_to_med > d_meddep_to_med, "확인 ✓", "미확인 ✗")))

cat("\n============================================================\n")

