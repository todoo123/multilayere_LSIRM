rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. load
################################################################################
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data/MIDUS_2")

# ─── ICPSR_04652: Survey (MIDUS 2, 2004-2006) ────────────────────────────────
load("ICPSR_04652/DS0001/04652-0001-Data.rda")
survey_0001 <- da04652.0001   # Aggregate

# ─── ICPSR_25281: Cognitive Project (MIDUS 2, 2004-2006) ─────────────────────
load("ICPSR_25281/DS0001/25281-0001-Data.rda")
cognitive_0001 <- da25281.0001

# ─── ICPSR_29282: Biomarker Project (MIDUS 2, 2004-2009) ─────────────────────
load("ICPSR_29282/DS0001/29282-0001-Data.rda")
biomarker_0001 <- da29282.0001

dim(survey_0001)      # Aggregate
dim(cognitive_0001)   # Cognitive
dim(biomarker_0001)   # Biomarker


################################################################################
# 0-1. helper: ICPSR factor → numeric code
################################################################################
# ICPSR factor label "(01) YES" → 1, "(97) DON'T KNOW" → 97
icpsr_code <- function(x) {
  as.numeric(sub("^\\(0*([0-9]+)\\).*$", "\\1", as.character(x)))
}

# ICPSR factor → numeric, code ≥ 7 을 NA 처리 (DK/REF/INAP)
# binary/ordinal 변수용: 유효 응답 코드는 보통 1~6 범위
icpsr_to_int <- function(x) {
  v <- icpsr_code(x)
  v[v >= 7] <- NA_real_
  as.integer(v)
}

# continuous 변수용: factor이면 code 추출, numeric이면 그대로
to_numeric <- function(x) {
  if (is.factor(x)) {
    v <- icpsr_code(x)
    v[v >= 97] <- NA_real_    # continuous 에서 97/98/99 = DK/REF/INAP
    return(v)
  }
  as.numeric(x)
}


################################################################################
# 1. variable selection (based on MIDUS_wave_2_변수명세.xlsx Sheet2)
################################################################################

# ─── 04652 (M2P1): Survey — screening 변수 ───────────────────────────────────

survey_vars <- c(
  # sadness branch
  "B1PA60",   # sadness gate: Felt sad/blue/depressed 2+ weeks       [binary]
  "B1PA61",   # sadness duration: Lasted how long                    [ordinal]
  "B1PA62",   # sadness frequency: How often felt sad/depressed      [ordinal]
  # anhedonia branch
  "B1PA72",   # anhedonia gate: Lost interest 2+ weeks               [binary]
  "B1PA73",   # anhedonia duration: How long loss of interest lasted  [ordinal]
  "B1PA74"    # anhedonia frequency: How often felt loss of interest  [ordinal]
)

survey_selected <- survey_0001 %>%
  dplyr::select(M2ID, all_of(survey_vars))

cat("\n=== Survey (04652 / M2P1) selected ===\n")
cat("dim:", dim(survey_selected), "\n\n")

cat("── 각 변수 값 분포 (raw factor) ──\n")
for (v in survey_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(table(survey_selected[[v]], useNA = "ifany"))
}


# ─── 29282 (M2P4): Biomarker — phenotype 변수 ────────────────────────────────

# ordinal-5: MASQ 5-point Likert (1=Not at all ~ 5=Extremely)
bio_ord5_vars <- c(
  "B4Q1C",    # positive affect: MASQ Felt cheerful         [역코딩 대상]
  "B4Q1E",    # depressed affect: MASQ Felt discouraged
  "B4Q1I",    # self-worth: MASQ Felt worthless
  "B4Q1J",    # positive affect: MASQ Felt really happy     [역코딩 대상]
  "B4Q1K"     # anxious distress: MASQ Felt nervous
)
bio_ord5_reverse <- c("B4Q1C", "B4Q1J")  # positive affect → 역코딩 (6 - x)

# ordinal-4: PSQI sleep quality 4-point Likert (1~4, 지난 1달 빈도)
bio_ord4_vars <- c(
  "B4S11B",   # sleep quality: Could not get to sleep within 30 minutes
  "B4S11C",   # sleep quality: Woke up in the middle of the night or early morning
  "B4S11H",   # sleep quality: Had bad dreams
  "B4S11I",   # sleep quality: Had pain
  "B4S11J"    # sleep quality: Other reasons
)

bio_ordinal_vars <- c(bio_ord5_vars, bio_ord4_vars)

bio_binary_vars <- character(0)   # binary layer 변수 없음

bio_continuous_vars <- c(
  "B4BSCL4A", # HPA axis: Saliva cortisol final average
  "B4BSCL14", # HPA axis: Saliva cortisol all sample average
  "B4BNE12",  # autonomic arousal: Urine Norepinephrine 12 hour
  "B4P2A",    # obesity: Waist in centimeters
  "B4PWHR",   # central adiposity: Waist-Hip Ratio
  "B4BIL6",   # inflammation: Blood Serum IL6 (pg/mL)
  "B4BFGN",   # inflammation: Blood Fibrinogen (mg/dL)
  "B4BCRP"    # inflammation: Blood C-Reactive Protein (ug/mL)
)

biomarker_vars <- c(bio_ordinal_vars, bio_binary_vars, bio_continuous_vars)

biomarker_selected <- biomarker_0001 %>%
  dplyr::select(M2ID, all_of(biomarker_vars))

cat("\n=== Biomarker (29282 / M2P4) selected ===\n")
cat("dim:", dim(biomarker_selected), "\n\n")

cat("── 각 변수 값 분포 (raw) ──\n")
for (v in biomarker_vars) {
  cat(sprintf("\n[%s]\n", v))
  if (is.factor(biomarker_selected[[v]])) {
    print(table(biomarker_selected[[v]], useNA = "ifany"))
  } else {
    cat("  class:", class(biomarker_selected[[v]]), "\n")
    print(summary(biomarker_selected[[v]]))
  }
}


# ─── 25281 (M2P3): Cognitive — phenotype 변수 ────────────────────────────────

cognitive_vars <- c(
  "B3TCOMPZ3", # global cognition: Zscore BTACT Composite Score
  "B3TEMZ3",   # episodic memory: Zscore Episodic Memory
  "B3TEFZ3",   # executive function: Zscore Executive Functioning
  "B3TWLF",    # memory retention: Word List Proportion forgot
  "B3TSMXNS",  # switching burden: SGST normal switch median RT
  "B3TSMXRS"   # switching burden: SGST reverse switch median RT
)

cognitive_selected <- cognitive_0001 %>%
  dplyr::select(M2ID, all_of(cognitive_vars))

cat("\n=== Cognitive (25281 / M2P3) selected ===\n")
cat("dim:", dim(cognitive_selected), "\n\n")

cat("── 각 변수 값 분포 (raw) ──\n")
for (v in cognitive_vars) {
  cat(sprintf("\n[%s]\n", v))
  if (is.factor(cognitive_selected[[v]])) {
    print(table(cognitive_selected[[v]], useNA = "ifany"))
  } else {
    cat("  class:", class(cognitive_selected[[v]]), "\n")
    print(summary(cognitive_selected[[v]]))
  }
}


################################################################################
# 2. 데이터타입 변환: ICPSR factor → integer / numeric
################################################################################

# ─── P4: Biomarker ────────────────────────────────────────────────────────────
# NOTE: ordinal-5 (MASQ, 1~5) 와 ordinal-4 (PSQI sleep, 1~4) 가 혼재함.
#       LSIRM ordinal layer 에서 변수별 category 수를 별도 지정하거나,
#       sub-layer 로 분리하는 것을 고려할 것.
biomarker_selected <- biomarker_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(bio_ordinal_vars),    icpsr_to_int),
    across(all_of(bio_continuous_vars), to_numeric),
    # 역코딩: positive affect 문항 (5-point → 6 - x, 높을수록 우울 방향)
    across(all_of(bio_ord5_reverse), ~ 6L - .x)
  )

cat("\n=== P4 변환 후 class 확인 ===\n")
str(biomarker_selected, give.attr = FALSE)

cat("\n── P4 변환 후 summary ──\n")
for (v in biomarker_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(summary(biomarker_selected[[v]]))
}

# ─── P3: Cognitive ────────────────────────────────────────────────────────────
cognitive_selected <- cognitive_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(cognitive_vars), to_numeric)
  )

cat("\n=== P3 변환 후 class 확인 ===\n")
str(cognitive_selected, give.attr = FALSE)

cat("\n── P3 변환 후 summary ──\n")
for (v in cognitive_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(summary(cognitive_selected[[v]]))
}


################################################################################
# 3. 시간 관련 변수 분포 확인 (count 모델 적합성 검토)
################################################################################
# continuous로 분류되어 있지만 negative binomial 기반 count로 고려할 수 있는 변수들
time_vars <- list(
  "B3TSMXNS" = list(data = cognitive_selected,  label = "SGST normal switch median RT"),
  "B3TSMXRS" = list(data = cognitive_selected,  label = "SGST reverse switch median RT")
)

par(mfrow = c(length(time_vars), 1), mar = c(4, 4, 3, 1))
for (vname in names(time_vars)) {
  vals <- time_vars[[vname]]$data[[vname]]
  vals <- vals[!is.na(vals)]
  hist(vals, breaks = 30, col = "steelblue", border = "white",
       main = sprintf("%s (%s)", vname, time_vars[[vname]]$label),
       xlab = "Value", ylab = "Frequency")
}
par(mfrow = c(1, 1))


################################################################################
# 4. Screening & case indicator
################################################################################

# ── branch 통과 판정 함수 ────────────────────────────────────────────────────
# gate : binary (1=YES, 2=NO, ≥7=DK/REF)
# dur  : ordinal (1=ALL DAY, 2=MOST OF DAY → pass; 3,4 → fail; ≥7 → NA)
# freq : ordinal (1=EVERY DAY, 2=ALMOST EVERY, 3=MOST DAYS → pass; 4+ → fail; ≥7 → NA)
branch_pass <- function(gate, dur, freq) {
  g <- icpsr_code(gate)
  d <- icpsr_code(dur)
  f <- icpsr_code(freq)

  case_when(
    g >= 7 | is.na(g)                     ~ NA_real_,  # gate DK/REF/NA
    g == 2                                 ~ 0,         # gate = NO → 비해당
    g == 1 & (d >= 7 | f >= 7 |
              is.na(d) | is.na(f))         ~ NA_real_,  # gate YES, 후속 DK/REF
    g == 1 & d <= 2 & f <= 3              ~ 1,          # 통과
    g == 1                                 ~ 0,          # gate YES, 기준 미달
    TRUE                                   ~ NA_real_
  )
}

df_screen <- survey_0001 %>%
  transmute(
    M2ID = as.numeric(M2ID),
    # 원변수 보존 (factor 그대로 — screening rule에서만 해석)
    B1PA60 = B1PA60, B1PA61 = B1PA61, B1PA62 = B1PA62,
    B1PA72 = B1PA72, B1PA73 = B1PA73, B1PA74 = B1PA74,
    # branch 판정
    sadness_pass   = branch_pass(B1PA60, B1PA61, B1PA62),
    anhedonia_pass = branch_pass(B1PA72, B1PA73, B1PA74),
    # case indicator: 둘 중 하나라도 통과 → 1
    screen_case = case_when(
      sadness_pass == 1 | anhedonia_pass == 1   ~ 1L,
      sadness_pass == 0 & anhedonia_pass == 0   ~ 0L,
      TRUE                                       ~ NA_integer_
    ),
    # 기본 공변량
    age = as.numeric(icpsr_code(B1PAGE_M2)),
    sex = as.integer(icpsr_code(B1PRSEX))     # 1=Male, 2=Female
  )

cat("\n=== Screening 결과 ===\n")
cat("전체:", nrow(df_screen), "\n")
print(table(screen_case = df_screen$screen_case, useNA = "ifany"))
cat("\nbranch 별 통과 현황:\n")
print(table(sadness = df_screen$sadness_pass,
            anhedonia = df_screen$anhedonia_pass, useNA = "ifany"))


################################################################################
# 5. Phenotype 데이터프레임 (변환 완료된 데이터에서 추출)
################################################################################

# ─── df_p4_pheno: P4 phenotype (이미 numeric 변환됨) ─────────────────────────
df_p4_pheno <- biomarker_selected   # M2ID + biomarker_vars, 모두 numeric

cat("\n=== df_p4_pheno ===\n")
cat("dim:", dim(df_p4_pheno), "\n")

# ─── df_p3_pheno: P3 phenotype (이미 numeric 변환됨) ─────────────────────────
df_p3_pheno <- cognitive_selected   # M2ID + cognitive_vars, 모두 numeric

cat("\n=== df_p3_pheno ===\n")
cat("dim:", dim(df_p3_pheno), "\n")


################################################################################
# 6. 분석용 데이터셋 (inner join)
################################################################################

# (1) P1–P4 분석셋: 수면·HPA·염증·비만 phenotype
df_analysis_p4 <- df_screen %>%
  inner_join(df_p4_pheno, by = "M2ID")

# (2) P1–P3 분석셋: cognitive phenotype
df_analysis_p3 <- df_screen %>%
  inner_join(df_p3_pheno, by = "M2ID")

# (3) P1–P3–P4 통합셋
df_analysis_all <- df_screen %>%
  inner_join(df_p3_pheno, by = "M2ID") %>%
  inner_join(df_p4_pheno, by = "M2ID")

cat("\n=== 분석용 데이터셋 크기 ===\n")
cat("df_analysis_p4  (P1×P4):      ", nrow(df_analysis_p4),  "명 ×", ncol(df_analysis_p4),  "변수\n")
cat("df_analysis_p3  (P1×P3):      ", nrow(df_analysis_p3),  "명 ×", ncol(df_analysis_p3),  "변수\n")
cat("df_analysis_all (P1×P3×P4):   ", nrow(df_analysis_all), "명 ×", ncol(df_analysis_all), "변수\n")

# screen_case 분포
cat("\n── df_analysis_p4 screen_case ──\n")
print(table(df_analysis_p4$screen_case, useNA = "ifany"))

cat("\n── df_analysis_p3 screen_case ──\n")
print(table(df_analysis_p3$screen_case, useNA = "ifany"))

cat("\n── df_analysis_all screen_case ──\n")
print(table(df_analysis_all$screen_case, useNA = "ifany"))

# NA 현황
cat("\n── df_analysis_p4: NA per column ──\n")
na_p4 <- colSums(is.na(df_analysis_p4))
print(na_p4[na_p4 > 0])

cat("\n── df_analysis_p3: NA per column ──\n")
na_p3 <- colSums(is.na(df_analysis_p3))
print(na_p3[na_p3 > 0])

cat("\n── df_analysis_all: NA per column ──\n")
na_all <- colSums(is.na(df_analysis_all))
print(na_all[na_all > 0])


################################################################################
# 7. screen_case == 1 필터링 → data type 별 matrix 분류 (LSIRM input)
################################################################################
# LSIRM 입력: Y_bin (binary matrix), Y_ord (ordinal matrix), Y_con (continuous matrix)
# - M2ID 제거하되 row/col 매핑 보존
# - ordinal 은 5-point(MASQ) 와 4-point(PSQI) 를 별도 layer 로 분리
#   → Y_ord1 (K1=5), Y_ord2 (K2=4)

# ── helper: 분석셋 → screen_case==1 필터 + NA 제거 + 5-layer matrix 분리 ────
split_for_lsirm <- function(df, bin_vars, ord1_vars, ord2_vars, con_vars, label) {

  # screen_case == 1 만 추출
  df_case <- df %>% filter(screen_case == 1)
  cat(sprintf("\n=== %s: screen_case==1 → %d명 ===\n", label, nrow(df_case)))

  # phenotype 변수만 추출 (screening/공변량 제외)
  pheno_vars <- c(bin_vars, ord1_vars, ord2_vars, con_vars)
  df_pheno <- df_case %>% dplyr::select(M2ID, all_of(pheno_vars))

  # 변수별 NA 현황 (제거 전)
  na_cnt <- colSums(is.na(df_pheno[, pheno_vars, drop = FALSE]))
  cat("  변수별 NA (제거 전):\n")
  print(na_cnt[na_cnt > 0])
  cat(sprintf("  NA 없는 complete case: %d / %d명\n",
              sum(complete.cases(df_pheno[, pheno_vars])), nrow(df_pheno)))

  # NA 제거
  df_clean <- na.omit(df_pheno)
  cat(sprintf("  → NA 제거 후: %d명 (제거 %d명)\n",
              nrow(df_clean), nrow(df_pheno) - nrow(df_clean)))

  # row/col 매핑 보존
  row_ids  <- df_clean$M2ID

  # M2ID 제거 후 matrix 분리
  make_mat <- function(vars) {
    if (length(vars) == 0) return(matrix(nrow = nrow(df_clean), ncol = 0))
    as.matrix(df_clean[, vars, drop = FALSE])
  }

  Y_bin  <- make_mat(bin_vars)
  Y_ord1 <- make_mat(ord1_vars)
  Y_ord2 <- make_mat(ord2_vars)
  Y_con  <- make_mat(con_vars)
  storage.mode(Y_ord1) <- "integer"
  storage.mode(Y_ord2) <- "integer"

  cat(sprintf("  Y_bin: %d × %d  |  Y_ord1: %d × %d (K1=%s)  |  Y_ord2: %d × %d (K2=%s)  |  Y_con: %d × %d\n",
              nrow(Y_bin), ncol(Y_bin),
              nrow(Y_ord1), ncol(Y_ord1),
              ifelse(ncol(Y_ord1) > 0, as.character(max(Y_ord1, na.rm=TRUE)), "–"),
              nrow(Y_ord2), ncol(Y_ord2),
              ifelse(ncol(Y_ord2) > 0, as.character(max(Y_ord2, na.rm=TRUE)), "–"),
              nrow(Y_con), ncol(Y_con)))

  list(
    Y_bin  = Y_bin,
    Y_ord1 = Y_ord1,
    Y_ord2 = Y_ord2,
    Y_con  = Y_con,
    row_ids  = row_ids,                    # M2ID (행 순서)
    col_bin  = colnames(Y_bin),
    col_ord1 = colnames(Y_ord1),
    col_ord2 = colnames(Y_ord2),
    col_con  = colnames(Y_con)
  )
}


# ─── P1–P4 ──────────────────────────────────────────────────────────────────
lsirm_p4 <- split_for_lsirm(
  df        = df_analysis_p4,
  bin_vars  = bio_binary_vars,
  ord1_vars = bio_ord5_vars,          # MASQ 5-point (K1=5)
  ord2_vars = bio_ord4_vars,          # PSQI 4-point (K2=4)
  con_vars  = bio_continuous_vars,
  label     = "P1-P4"
)

# ─── P1–P3 ──────────────────────────────────────────────────────────────────
# P3 는 모두 continuous → binary/ordinal layer 는 빈 matrix
lsirm_p3 <- split_for_lsirm(
  df        = df_analysis_p3,
  bin_vars  = character(0),
  ord1_vars = character(0),
  ord2_vars = character(0),
  con_vars  = cognitive_vars,
  label     = "P1-P3"
)

# ─── P1–P3–P4 ───────────────────────────────────────────────────────────────
lsirm_all <- split_for_lsirm(
  df        = df_analysis_all,
  bin_vars  = bio_binary_vars,
  ord1_vars = bio_ord5_vars,
  ord2_vars = bio_ord4_vars,
  con_vars  = c(bio_continuous_vars, cognitive_vars),
  label     = "P1-P3-P4"
)

