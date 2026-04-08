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
icpsr_code <- function(x) {
  as.numeric(sub("^\\(0*([0-9]+)\\).*$", "\\1", as.character(x)))
}

icpsr_to_int <- function(x) {
  v <- icpsr_code(x)
  v[v >= 7] <- NA_real_
  as.integer(v)
}

# binary 전용: 1=YES→1, 2=NO→0, ≥7→NA
icpsr_to_binary <- function(x) {
  v <- icpsr_code(x)
  v[v >= 7] <- NA_real_
  ifelse(v == 1, 1L, ifelse(v == 2, 0L, NA_integer_))
}

to_numeric <- function(x) {
  if (is.factor(x)) {
    v <- icpsr_code(x)
    v[v >= 97] <- NA_real_
    return(v)
  }
  as.numeric(x)
}


################################################################################
# 1. variable selection (v3: 변수 추가/제거 반영)
################################################################################

# ─── 04652 (M2P1): Survey — screening + phenotype 변수 ───────────────────────

survey_screen_vars <- c(
  "B1PA60", "B1PA61", "B1PA62",   # sadness branch
  "B1PA72", "B1PA73", "B1PA74"    # anhedonia branch
)

# v3 추가: 비처방 진통제 (binary) → v4: 합산하여 5-level ordinal (painmed_total)
# survey_bin_vars <- c(
#   "B1SA13A",  # nonprescription pain med: Aspirin
#   "B1SA13B",  # nonprescription pain med: Acetaminophen
#   "B1SA13C",  # nonprescription pain med: Ibuprofen
#   "B1SA13D"   # nonprescription pain med: Naproxen Sodium
# )
survey_bin_vars <- character(0)

# 합산용 원본 변수 (binary 4개 → 0~4 합산 후 +1 → 1~5 ordinal)
survey_painmed_src <- c("B1SA13A", "B1SA13B", "B1SA13C", "B1SA13D")

survey_cnt_vars <- character(0)

survey_vars <- c(survey_screen_vars, survey_bin_vars, survey_cnt_vars, survey_painmed_src)

survey_selected <- survey_0001 %>%
  dplyr::select(M2ID, all_of(survey_vars))

cat("\n=== Survey (04652 / M2P1) selected ===\n")
cat("dim:", dim(survey_selected), "\n\n")


# ─── 29282 (M2P4): Biomarker — phenotype 변수 ────────────────────────────────

# ordinal-5: MASQ 5-point Likert (1~5)
# v3: B4Q1C, B4Q1J 제거 / B4Q1LL, B4Q1GG, B4Q1B 추가
# bio_ord5_vars <- c(
#   "B4Q1E",    # depressed affect: MASQ Felt discouraged
#   "B4Q1I",    # self-worth: MASQ Felt worthless
#   "B4Q1K",    # anxious distress: MASQ Felt nervous
#   "B4Q1LL",   # low energy: MASQ Felt sluggish or tired           [v3 추가]
#   "B4Q1GG",   # psychomotor slowing: MASQ Felt really slowed down [v3 추가]
#   "B4Q1B"     # anxious arousal: MASQ Startled easily             [v3 추가]
# )
bio_ord5_vars <- c(
  # --- General Distress–Anxious (GDA, 11) ---
  "B4Q1D",    # MASQ Felt afraid
  "B4Q1H",    # MASQ Had diarrhea
  "B4Q1K",    # MASQ Felt nervous
  "B4Q1N",    # MASQ Felt uneasy
  "B4Q1P",    # MASQ Had a lump in my throat
  "B4Q1T",    # MASQ Had an upset stomach
  "B4Q1Z",    # MASQ Felt keyed up, on edge
  "B4Q1FF",   # MASQ Was unable to relax
  "B4Q1II",   # MASQ Felt nauseous
  "B4Q1CCC",  # MASQ Felt tense or high-strung
  "B4Q1GGG",  # MASQ Muscles were tense or sore
  # --- Anxious Arousal (AA, 17) ---
  "B4Q1B",    # MASQ Startled easily
  "B4Q1F",    # MASQ Hands were shaky
  "B4Q1M",    # MASQ Was short of breath
  "B4Q1Q",    # MASQ Felt faint
  "B4Q1S",    # MASQ Had hot or cold spells
  "B4Q1X",    # MASQ Hands were cold or sweaty
  "B4Q1BB",   # MASQ Was trembling or shaking
  "B4Q1DD",   # MASQ Had trouble swallowing
  "B4Q1KK",   # MASQ Felt dizzy or lightheaded
  "B4Q1NN",   # MASQ Had pain in my chest
  "B4Q1PP",   # MASQ Felt like I was choking
  "B4Q1RR",   # MASQ Muscles twitched or trembled
  "B4Q1TT",   # MASQ Had a very dry mouth
  "B4Q1VV",   # MASQ Was afraid I was going to die
  "B4Q1ZZ",   # MASQ Heart was racing or pounding
  "B4Q1BBB",  # MASQ Felt numbness or tingling in body
  "B4Q1JJJ"   # MASQ Had to urinate frequently
)
bio_ord5_reverse <- character(0)

# ordinal-4: PSQI sleep quality (1~4)
# v3: B4S11J 제거 / B4S5 추가
# v4: B4S11B, B4S11C, B4S11I → sleep_trouble_count로 대체
bio_ord4_vars <- c(
  # "B4S11B",   # sleep trouble: Woke up in the middle of the night or early morning
  # "B4S11C",   # sleep trouble: Had to get up to use the bathroom
  # "B4S11H",   # sleep trouble: Had bad dreams
  # "B4S11I",   # sleep trouble: Had pain
  "B4S5"      # rate sleep quality overall (1=Very good ~ 4=Very bad) [v3 추가]
)

# ordinal-4: CESD 20문항 (1~4)  [v5 추가]
bio_cesd_vars <- c(
  "B4Q3A",    # CESD I was bothered by things that usually don't bother me
  "B4Q3B",    # CESD I did not feel like eating; my appetite was poor
  "B4Q3C",    # CESD I felt that I could not shake off the blues
  "B4Q3D",    # CESD I felt that I was just as good as other people  ★ reverse
  "B4Q3E",    # CESD I had trouble keeping my mind on what I was doing
  "B4Q3F",    # CESD I felt depressed
  "B4Q3G",    # CESD I felt that everything I did was an effort
  "B4Q3H",    # CESD I felt hopeful about the future                ★ reverse
  "B4Q3I",    # CESD I thought my life had been a failure
  "B4Q3J",    # CESD I felt fearful
  "B4Q3K",    # CESD My sleep was restless
  "B4Q3L",    # CESD I was happy                                    ★ reverse
  "B4Q3M",    # CESD I talked less than usual
  "B4Q3N",    # CESD I felt lonely
  "B4Q3O",    # CESD People were unfriendly
  "B4Q3P",    # CESD I enjoyed life                                 ★ reverse
  "B4Q3Q",    # CESD I had crying spells
  "B4Q3R",    # CESD I felt sad
  "B4Q3S",    # CESD I felt that people dislike me
  "B4Q3T"     # CESD I could not get going
)
bio_cesd_reverse <- c("B4Q3D", "B4Q3H", "B4Q3L", "B4Q3P")

# v4: B4S11A~J 이진화(≥2→1, 1→0) 후 합산 → sleep_trouble_count (count 변수)
bio_sleep_src <- c(
  "B4S11A",   # Cannot get to sleep within 30 minutes
  "B4S11B",   # Wake up in the middle of the night or early morning
  "B4S11C",   # Have to get up to use the bathroom
  "B4S11D",   # Cannot breathe comfortably
  "B4S11E",   # Cough or snore loudly
  "B4S11F",   # Feel too cold
  "B4S11G",   # Feel too hot
  "B4S11H",   # Had bad dreams
  "B4S11I",   # Have pain
  "B4S11J"    # Other reason(s)
)

bio_ordinal_vars <- c(bio_ord5_vars, bio_ord4_vars, bio_cesd_vars)

# v3 추가: binary 변수
bio_binary_vars <- c(
  # "B4H26",    # tobacco exposure: Have you ever smoked cigarettes regularly?
  # "B4H33"     # alcohol use: Past month, had at least one drink?
)

# v3 추가: count 변수 (0.5 단위 절사)
bio_count_vars <- c(
  "B4S4",     # sleep hours: hours of actual sleep at night
  # "B4O9",     # tobacco quantity: cigarettes per day on average
  "B4H40"     # alcohol quantity: drinks per drinking day (period drank most)
)

bio_continuous_vars <- c(
  # "B4BSCL4A", # HPA axis: Saliva cortisol final average — refresher_1에 없으므로 제거
  "B4BSCL14", # HPA axis: Saliva cortisol all sample average
  "B4BNE12",  # autonomic arousal: Urine Norepinephrine 12 hour
  # "B4P2A",    # obesity: Waist in centimeters
  # "B4PWHR",   # central adiposity: Waist-Hip Ratio
  "B4BIL6",   # inflammation: Blood Serum IL6 (pg/mL)
  "B4BFGN",   # inflammation: Blood Fibrinogen (mg/dL)
  "B4BCRP"    # inflammation: Blood C-Reactive Protein (ug/mL)
)

biomarker_vars <- c(bio_ordinal_vars, bio_binary_vars, bio_count_vars, bio_continuous_vars, bio_sleep_src)

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

# continuous 변수 (높을수록 인지↓)
cog_con_vars <- c(
  # "B3TCOMPZ3", # global cognition: Zscore BTACT Composite — 하위지표와 중복하므로 제외
  "B3TEMZ3",   # episodic memory: Zscore Episodic Memory
  "B3TEFZ3",   # executive function: Zscore Executive Functioning
  "B3TWLF",    # memory retention: Word List Proportion forgot (높=많이 잊음)
  "B3TSMN",    # processing speed: SGST Normal single-task median RT (초)
  "B3TSMR",    # processing speed: SGST Reverse single-task median RT (초)
  "B3TSMXBS",  # task switching: SGST Mixed block all switch median RT (초)
  "B3TSMXNO",  # task switching: SGST Mixed block normal non-switch median RT (초)
  "B3TSMXRO"   # task switching: SGST Mixed block reverse non-switch median RT (초)
  # "B3TSMXNS",  # B3TSMXBS로 대체
  # "B3TSMXRS"   # B3TSMXBS로 대체
)

# count 변수 (높을수록 인지↓)
cog_cnt_vars <- c(
  "B3TSPN",    # EF/speed: SGST Normal single-task 정확도 → 오답 수로 변환 (0~20)
  "B3TSPR",    # EF/speed: SGST Reverse single-task 정확도 → 오답 수로 변환 (0~20)
  "B3TCTFLR",  # verbal fluency: Category fluency 반복 수 (zero-inflated)
  "B3TCTFLI"   # verbal fluency: Category fluency 침입 오류 수 (zero-inflated)
)

cognitive_vars <- c(cog_con_vars, cog_cnt_vars)

cognitive_selected <- cognitive_0001 %>%
  dplyr::select(M2ID, all_of(cognitive_vars))

cat("\n=== Cognitive (25281 / M2P3) selected ===\n")
cat("dim:", dim(cognitive_selected), "\n\n")


################################################################################
# 2. 데이터타입 변환
################################################################################

# log 변환 대상
# bio_log_vars <- c("B4BSCL4A", "B4BSCL14", "B4BNE12", "B4BIL6", "B4BCRP")
# cog_log_vars <- c("B3TSMXNS", "B3TSMXRS")

# ─── P1: Survey phenotype 변환 ───────────────────────────────────────────────
survey_selected <- survey_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(survey_bin_vars), icpsr_to_binary),
    across(all_of(survey_cnt_vars), to_numeric),
    across(all_of(survey_cnt_vars), ~ as.integer(floor(.x))),
    # v4: B1SA13A~D 이진화 후 합산 → painmed_total (1~5 ordinal)
    across(all_of(survey_painmed_src), icpsr_to_binary),
    painmed_total = as.integer(rowSums(across(all_of(survey_painmed_src))) + 1L)
  )

cat("\n=== P1 Survey phenotype 변환 후 ===\n")
for (v in c(survey_bin_vars, survey_cnt_vars, "painmed_total")) {
  cat(sprintf("[%s] ", v))
  print(table(survey_selected[[v]], useNA = "ifany"))
}

# ─── P4: Biomarker 변환 ─────────────────────────────────────────────────────
biomarker_selected <- biomarker_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(bio_ordinal_vars),    icpsr_to_int),
    across(all_of(bio_binary_vars),     icpsr_to_binary),
    across(all_of(bio_continuous_vars), to_numeric),
    # log 변환: right-skewed biomarker
    # across(all_of(bio_log_vars), ~ log(.x)),
    # count 변수: to_numeric + floor (0.5 단위 절사)
    across(all_of(bio_count_vars), to_numeric),
    # skip pattern 처리: 비흡연자 → 흡연량 0
    # B4O9 = ifelse(B4H26 == 0 & is.na(B4O9), 0, B4O9),
    # skip pattern 처리: 비음주자 → 음주량/기간 0 (B4H33 주석처리로 비활성)
    # B4H40 = ifelse(B4H33 == 0 & is.na(B4H40), 0, B4H40),
    # count 절사 (floor)
    across(all_of(bio_count_vars), ~ as.integer(floor(.x))),
    # CESD reverse items: 먼저 역코딩 (1↔4, 2↔3) 후 이진화
    across(all_of(bio_cesd_reverse), ~ 5L - .x),
    # CESD 전체 이진화 (icpsr 코딩 1→0, 2/3/4→1)
    across(all_of(bio_cesd_vars), ~ ifelse(.x == 1, 0L, 1L)),
    # v4: B4S11A~J 이진화(≥2 → 1, 1 → 0) 후 합산 → sleep_trouble_count
    across(all_of(bio_sleep_src), icpsr_to_int),
    across(all_of(bio_sleep_src), ~ ifelse(.x >= 2, 1L, 0L)),
    sleep_trouble_count = as.integer(rowSums(across(all_of(bio_sleep_src))))
  )

cat("\n=== P4 변환 후 class 확인 ===\n")
str(biomarker_selected, give.attr = FALSE)

cat("\n── P4 변환 후 summary ──\n")
for (v in biomarker_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(summary(biomarker_selected[[v]]))
}

# ─── P3: Cognitive 변환 ─────────────────────────────────────────────────────
# Z-score 변수 중 높을수록 좋은 방향 → 부호 반전 (높=인지↓)
cog_reverse_vars <- c("B3TEMZ3", "B3TEFZ3")

cognitive_selected <- cognitive_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(cognitive_vars), to_numeric),
    # Z-score: 부호 반전 (높=좋음 → 높=나쁨)
    across(all_of(cog_reverse_vars), ~ -.x),
    # 정확도 → 오답 수 변환: round(x/0.05) gives 0~20, then 20 - that = errors
    across(all_of(c("B3TSPN", "B3TSPR")), ~ as.integer(abs(round(.x / 0.05) - 20L))),
    # count 변수: integer 변환
    across(all_of(c("B3TCTFLR", "B3TCTFLI")), ~ as.integer(floor(.x)))
  )

cat("\n=== P3 변환 후 class 확인 ===\n")
str(cognitive_selected, give.attr = FALSE)


################################################################################
# 3. 시간 관련 변수 분포 확인
################################################################################
# time_vars <- list(
#   "B3TSMXNS" = list(data = cognitive_selected,  label = "SGST normal switch median RT"),
#   "B3TSMXRS" = list(data = cognitive_selected,  label = "SGST reverse switch median RT")
# )
# 
# par(mfrow = c(length(time_vars), 1), mar = c(4, 4, 3, 1))
# for (vname in names(time_vars)) {
#   vals <- time_vars[[vname]]$data[[vname]]
#   vals <- vals[!is.na(vals)]
#   hist(vals, breaks = 30, col = "steelblue", border = "white",
#        main = sprintf("%s (%s)", vname, time_vars[[vname]]$label),
#        xlab = "Value", ylab = "Frequency")
# }
# par(mfrow = c(1, 1))


################################################################################
# 4. Screening & case indicator
################################################################################

branch_pass <- function(gate, dur, freq) {
  g <- icpsr_code(gate)
  d <- icpsr_code(dur)
  f <- icpsr_code(freq)

  case_when(
    g >= 7 | is.na(g)                     ~ NA_real_,
    g == 2                                 ~ 0,
    g == 1 & (d >= 7 | f >= 7 |
              is.na(d) | is.na(f))         ~ NA_real_,
    g == 1 & d <= 2 & f <= 3              ~ 1,
    g == 1                                 ~ 0,
    TRUE                                   ~ NA_real_
  )
}

df_screen <- survey_0001 %>%
  transmute(
    M2ID = as.numeric(M2ID),
    B1PA60 = B1PA60, B1PA61 = B1PA61, B1PA62 = B1PA62,
    B1PA72 = B1PA72, B1PA73 = B1PA73, B1PA74 = B1PA74,
    sadness_pass   = branch_pass(B1PA60, B1PA61, B1PA62),
    anhedonia_pass = branch_pass(B1PA72, B1PA73, B1PA74),
    screen_case = case_when(
      sadness_pass == 1 | anhedonia_pass == 1   ~ 1L,
      sadness_pass == 0 & anhedonia_pass == 0   ~ 0L,
      TRUE                                       ~ NA_integer_
    ),
    age = as.numeric(icpsr_code(B1PAGE_M2)),
    sex = as.integer(icpsr_code(B1PRSEX))
  )

# survey phenotype 변수를 df_screen에 merge
df_screen <- df_screen %>%
  left_join(
    # survey_selected %>% dplyr::select(M2ID, all_of(survey_bin_vars), all_of(survey_cnt_vars), painmed_total),
    survey_selected %>% dplyr::select(M2ID, all_of(survey_bin_vars), all_of(survey_cnt_vars)),
    by = "M2ID"
  )

cat("\n=== Screening 결과 ===\n")
cat("전체:", nrow(df_screen), "\n")
print(table(screen_case = df_screen$screen_case, useNA = "ifany"))
cat("\nbranch 별 통과 현황:\n")
print(table(sadness = df_screen$sadness_pass,
            anhedonia = df_screen$anhedonia_pass, useNA = "ifany"))


################################################################################
# 5. Phenotype 데이터프레임
################################################################################

df_p4_pheno <- biomarker_selected   # M2ID + biomarker_vars
df_p3_pheno <- cognitive_selected   # M2ID + cognitive_vars

cat("\n=== df_p4_pheno ===\n")
cat("dim:", dim(df_p4_pheno), "\n")
cat("\n=== df_p3_pheno ===\n")
cat("dim:", dim(df_p3_pheno), "\n")


################################################################################
# 6. 분석용 데이터셋 (inner join)
################################################################################

df_analysis_p4 <- df_screen %>%
  inner_join(df_p4_pheno, by = "M2ID")

df_analysis_p3 <- df_screen %>%
  inner_join(df_p3_pheno, by = "M2ID")

df_analysis_all <- df_screen %>%
  inner_join(df_p3_pheno, by = "M2ID") %>%
  inner_join(df_p4_pheno, by = "M2ID")

cat("\n=== 분석용 데이터셋 크기 ===\n")
cat("df_analysis_p4  (P1×P4):      ", nrow(df_analysis_p4),  "명 ×", ncol(df_analysis_p4),  "변수\n")
cat("df_analysis_p3  (P1×P3):      ", nrow(df_analysis_p3),  "명 ×", ncol(df_analysis_p3),  "변수\n")
cat("df_analysis_all (P1×P3×P4):   ", nrow(df_analysis_all), "명 ×", ncol(df_analysis_all), "변수\n")

print(table(df_analysis_p4$screen_case, useNA = "ifany"))
print(table(df_analysis_p3$screen_case, useNA = "ifany"))
print(table(df_analysis_all$screen_case, useNA = "ifany"))

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
# 7. screen_case == 1 필터링 → 5-layer matrix 분류 (LSIRM input)
################################################################################
# Y_bin (binary), Y_con (continuous), Y_cnt (count), Y_ord1 (5-point), Y_ord2 (4-point)

split_for_lsirm <- function(df, bin_vars, cnt_vars, ord1_vars, ord2_vars, con_vars, label) {

  df_case <- df %>% filter(screen_case == 1)
  cat(sprintf("\n=== %s: screen_case==1 → %d명 ===\n", label, nrow(df_case)))

  pheno_vars <- c(bin_vars, cnt_vars, ord1_vars, ord2_vars, con_vars)
  df_pheno <- df_case %>% dplyr::select(M2ID, all_of(pheno_vars))

  na_cnt <- colSums(is.na(df_pheno[, pheno_vars, drop = FALSE]))
  cat("  변수별 NA (제거 전):\n")
  print(na_cnt[na_cnt > 0])
  cat(sprintf("  NA 없는 complete case: %d / %d명\n",
              sum(complete.cases(df_pheno[, pheno_vars])), nrow(df_pheno)))

  df_clean <- na.omit(df_pheno)
  cat(sprintf("  → NA 제거 후: %d명 (제거 %d명)\n",
              nrow(df_clean), nrow(df_pheno) - nrow(df_clean)))

  row_ids  <- df_clean$M2ID
  branch_info <- df_case %>%
    dplyr::select(M2ID, sadness_pass, anhedonia_pass) %>%
    filter(M2ID %in% row_ids) %>%
    arrange(match(M2ID, row_ids)) %>%
    mutate(branch = case_when(
      sadness_pass == 1 & anhedonia_pass == 1 ~ "Both",
      sadness_pass == 1                       ~ "Sadness",
      anhedonia_pass == 1                     ~ "Anhedonia",
      TRUE                                    ~ "Unknown"
    ))

  make_mat <- function(vars) {
    if (length(vars) == 0) return(matrix(nrow = nrow(df_clean), ncol = 0))
    as.matrix(df_clean[, vars, drop = FALSE])
  }

  Y_bin  <- make_mat(bin_vars)
  Y_cnt  <- make_mat(cnt_vars)
  Y_ord1 <- make_mat(ord1_vars)
  Y_ord2 <- make_mat(ord2_vars)
  Y_con  <- make_mat(con_vars)
  storage.mode(Y_bin)  <- "integer"
  storage.mode(Y_cnt)  <- "integer"
  storage.mode(Y_ord1) <- "integer"
  storage.mode(Y_ord2) <- "integer"

  cat(sprintf("  Y_bin: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d (K1=%s) | Y_ord2: %d×%d (K2=%s) | Y_con: %d×%d\n",
              nrow(Y_bin), ncol(Y_bin),
              nrow(Y_cnt), ncol(Y_cnt),
              nrow(Y_ord1), ncol(Y_ord1),
              ifelse(ncol(Y_ord1) > 0, as.character(max(Y_ord1, na.rm=TRUE)), "–"),
              nrow(Y_ord2), ncol(Y_ord2),
              ifelse(ncol(Y_ord2) > 0, as.character(max(Y_ord2, na.rm=TRUE)), "–"),
              nrow(Y_con), ncol(Y_con)))

  list(
    Y_bin  = Y_bin,
    Y_cnt  = Y_cnt,
    Y_ord1 = Y_ord1,
    Y_ord2 = Y_ord2,
    Y_con  = Y_con,
    row_ids  = row_ids,
    branch   = branch_info$branch,
    col_bin  = colnames(Y_bin),
    col_cnt  = colnames(Y_cnt),
    col_ord1 = colnames(Y_ord1),
    col_ord2 = colnames(Y_ord2),
    col_con  = colnames(Y_con)
  )
}


# ─── P1–P4 ──────────────────────────────────────────────────────────────────
lsirm_p4 <- split_for_lsirm(
  df        = df_analysis_p4,
  bin_vars  = c(survey_bin_vars, bio_binary_vars, bio_cesd_vars),
  cnt_vars  = c(survey_cnt_vars, bio_count_vars, "sleep_trouble_count"),
  ord1_vars = c(bio_ord5_vars),
  ord2_vars = c(bio_ord4_vars),
  con_vars  = bio_continuous_vars,
  label     = "P1-P4"
)

# ─── P1–P3 ──────────────────────────────────────────────────────────────────
lsirm_p3 <- split_for_lsirm(
  df        = df_analysis_p3,
  bin_vars  = survey_bin_vars,
  cnt_vars  = c(survey_cnt_vars, cog_cnt_vars),
  # ord1_vars = c("painmed_total"),
  ord1_vars = character(0),
  ord2_vars = character(0),
  con_vars  = cog_con_vars,
  label     = "P1-P3"
)

# ─── P1–P3–P4 ───────────────────────────────────────────────────────────────
lsirm_all <- split_for_lsirm(
  df        = df_analysis_all,
  bin_vars  = c(survey_bin_vars, bio_binary_vars, bio_cesd_vars),
  cnt_vars  = c(survey_cnt_vars, bio_count_vars, "sleep_trouble_count", cog_cnt_vars),
  ord1_vars = c(bio_ord5_vars),
  ord2_vars = c(bio_ord4_vars),
  con_vars  = c(bio_continuous_vars, cog_con_vars),
  label     = "P1-P3-P4"
)

