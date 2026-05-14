rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. load — MIDUS Refresher 1 데이터
################################################################################
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data/MIDUS_refresher_1")

# ─── ICPSR_36532: Survey (MIDUS Refresher 1) ────────────────────────────────
load("ICPSR_36532/DS0001/36532-0001-Data.rda")
survey_0001 <- da36532.0001

# ─── ICPSR_36901: Biomarker (MIDUS Refresher 1) ─────────────────────────────
load("ICPSR_36901/DS0001/36901-0001-Data.rda")
biomarker_0001 <- da36901.0001

# ─── ICPSR_37081: Cognitive (MIDUS Refresher 1) ─────────────────────────────
load("ICPSR_37081/DS0001/37081-0001-Data.rda")
cognitive_0001 <- da37081.0001

dim(survey_0001)      # Survey
dim(biomarker_0001)   # Biomarker
dim(cognitive_0001)   # Cognitive

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
# 1. variable selection (v3: refresher_1 변수 매핑)
################################################################################

# ─── 36532: Survey — screening + phenotype 변수 ─────────────────────────────

survey_screen_vars <- c(
  "RA1PA60", "RA1PA61", "RA1PA62",   # sadness branch
  "RA1PA72", "RA1PA73", "RA1PA74"    # anhedonia branch
)

survey_bin_vars <- character(0)

# 합산용 원본 변수 (binary 4개 → 0~4 합산 후 +1 → 1~5 ordinal)
survey_painmed_src <- c("RA1SA13A", "RA1SA13B", "RA1SA13C", "RA1SA13D")

survey_cnt_vars <- character(0)

survey_vars <- c(survey_screen_vars, survey_bin_vars, survey_cnt_vars, survey_painmed_src)

survey_selected <- survey_0001 %>%
  dplyr::select(MRID, all_of(survey_vars))

cat("\n=== Survey (36532) selected ===\n")
cat("dim:", dim(survey_selected), "\n\n")


# ─── 36901: Biomarker — phenotype 변수 ──────────────────────────────────────

# ordinal-5: MASQ 5-point Likert (1~5)
bio_ord5_vars <- c(
  # --- General Distress–Anxious (GDA, 11) ---
  "RA4Q1D",    # MASQ Felt afraid
  "RA4Q1H",    # MASQ Had diarrhea
  "RA4Q1K",    # MASQ Felt nervous
  "RA4Q1N",    # MASQ Felt uneasy
  "RA4Q1P",    # MASQ Had a lump in my throat
  "RA4Q1T",    # MASQ Had an upset stomach
  "RA4Q1Z",    # MASQ Felt keyed up, on edge
  "RA4Q1FF",   # MASQ Was unable to relax
  "RA4Q1II",   # MASQ Felt nauseous
  "RA4Q1CCC",  # MASQ Felt tense or high-strung
  "RA4Q1GGG",  # MASQ Muscles were tense or sore
  # --- Anxious Arousal (AA, 17) ---
  "RA4Q1B",    # MASQ Startled easily
  "RA4Q1F",    # MASQ Hands were shaky
  "RA4Q1M",    # MASQ Was short of breath
  "RA4Q1Q",    # MASQ Felt faint
  "RA4Q1S",    # MASQ Had hot or cold spells
  "RA4Q1X",    # MASQ Hands were cold or sweaty
  "RA4Q1BB",   # MASQ Was trembling or shaking
  "RA4Q1DD",   # MASQ Had trouble swallowing
  "RA4Q1KK",   # MASQ Felt dizzy or lightheaded
  "RA4Q1NN",   # MASQ Had pain in my chest
  "RA4Q1PP",   # MASQ Felt like I was choking
  "RA4Q1RR",   # MASQ Muscles twitched or trembled
  "RA4Q1TT",   # MASQ Had a very dry mouth
  "RA4Q1VV",   # MASQ Was afraid I was going to die
  "RA4Q1ZZ",   # MASQ Heart was racing or pounding
  "RA4Q1BBB",  # MASQ Felt numbness or tingling in body
  "RA4Q1JJJ"   # MASQ Had to urinate frequently
)
bio_ord5_reverse <- character(0)

# ordinal-4: PSQI sleep quality (1~4)
bio_ord4_vars <- c(
  "RA4S5"      # rate sleep quality overall (1=Very good ~ 4=Very bad)
)

# ordinal-4: CESD 20문항 (1~4) → binary 이진화
bio_cesd_vars <- c(
  "RA4Q3A",    # CESD I was bothered by things that usually don't bother me
  "RA4Q3B",    # CESD I did not feel like eating; my appetite was poor
  "RA4Q3C",    # CESD I felt that I could not shake off the blues
  "RA4Q3D",    # CESD I felt that I was just as good as other people  ★ reverse
  "RA4Q3E",    # CESD I had trouble keeping my mind on what I was doing
  "RA4Q3F",    # CESD I felt depressed
  "RA4Q3G",    # CESD I felt that everything I did was an effort
  "RA4Q3H",    # CESD I felt hopeful about the future                ★ reverse
  "RA4Q3I",    # CESD I thought my life had been a failure
  "RA4Q3J",    # CESD I felt fearful
  "RA4Q3K",    # CESD My sleep was restless
  "RA4Q3L",    # CESD I was happy                                    ★ reverse
  "RA4Q3M",    # CESD I talked less than usual
  "RA4Q3N",    # CESD I felt lonely
  "RA4Q3O",    # CESD People were unfriendly
  "RA4Q3P",    # CESD I enjoyed life                                 ★ reverse
  "RA4Q3Q",    # CESD I had crying spells
  "RA4Q3R",    # CESD I felt sad
  "RA4Q3S",    # CESD I felt that people dislike me
  "RA4Q3T"     # CESD I could not get going
)
bio_cesd_reverse <- c("RA4Q3D", "RA4Q3H", "RA4Q3L", "RA4Q3P")

# sleep trouble 합산용 원본 변수 (B4S11A~J → RA4S11A~J)
bio_sleep_src <- c(
  "RA4S11A",   # Cannot get to sleep within 30 minutes
  "RA4S11B",   # Wake up in the middle of the night or early morning
  "RA4S11C",   # Have to get up to use the bathroom
  "RA4S11D",   # Cannot breathe comfortably
  "RA4S11E",   # Cough or snore loudly
  "RA4S11F",   # Feel too cold
  "RA4S11G",   # Feel too hot
  "RA4S11H",   # Had bad dreams
  "RA4S11I",   # Have pain
  "RA4S11J"    # Other reason(s)
)

bio_ordinal_vars <- c(bio_ord5_vars, bio_ord4_vars, bio_cesd_vars)

# binary 변수
bio_binary_vars <- c(
  # "RA4H38",    # tobacco exposure: Have you ever smoked cigarettes regularly?
  # "RA4H49"     # alcohol use: Past month, had at least one drink?
)

# count 변수 (0.5 단위 절사)
bio_count_vars <- c(
  "RA4S4",     # sleep hours: hours of actual sleep at night
  "RA4H56"     # alcohol quantity: drinks per drinking day (period drank most)
)

# continuous 변수 — RA4BSCL4A 제거 (refresher_1에 존재하지 않음)
bio_continuous_vars <- c(
  # "RA4BSCL4A", # HPA axis: Saliva cortisol final average — refresher_1에 없음
  "RA4BSCLAV", # HPA axis: Saliva cortisol all sample average
  "RA4BNE12",  # autonomic arousal: Urine Norepinephrine 12 hour
  # "RA4P2A",    # obesity: Waist in centimeters
  # "RA4PWHR",   # central adiposity: Waist-Hip Ratio
  "RA4BIL6",   # inflammation: Blood Serum IL6 (pg/mL)
  "RA4BFGN",   # inflammation: Blood Fibrinogen (mg/dL)
  "RA4BCRP"    # inflammation: Blood C-Reactive Protein (ug/mL)
)

biomarker_vars <- c(bio_ordinal_vars, bio_binary_vars, bio_count_vars, bio_continuous_vars, bio_sleep_src)

biomarker_selected <- biomarker_0001 %>%
  dplyr::select(MRID, all_of(biomarker_vars))

cat("\n=== Biomarker (36901) selected ===\n")
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


# ─── 37081: Cognitive — phenotype 변수 ──────────────────────────────────────

# continuous 변수 (높을수록 인지↓)
cog_con_vars <- c(
  # "RA3TCOMPZ3", # global cognition: Zscore BTACT Composite — 하위지표와 중복하므로 제외
  "RA3TEMZ3",   # episodic memory: Zscore Episodic Memory
  "RA3TEFZ3",   # executive function: Zscore Executive Functioning
  "RA3TWLF",    # memory retention: Word List Proportion forgot (높=많이 잊음)
  "RA3TSMN",    # processing speed: SGST Normal single-task median RT (초)
  "RA3TSMR",    # processing speed: SGST Reverse single-task median RT (초)
  "RA3TSMXBS",  # task switching: SGST Mixed block all switch median RT (초)
  "RA3TSMXNO",  # task switching: SGST Mixed block normal non-switch median RT (초)
  "RA3TSMXRO"   # task switching: SGST Mixed block reverse non-switch median RT (초)
  # "RA3TSMXNS",  # RA3TSMXBS로 대체
  # "RA3TSMXRS"   # RA3TSMXBS로 대체
)

# count 변수 (높을수록 인지↓)
cog_cnt_vars <- c(
  "RA3TSPN",    # EF/speed: SGST Normal single-task 정확도 → 오답 수로 변환 (0~20)
  "RA3TSPR",    # EF/speed: SGST Reverse single-task 정확도 → 오답 수로 변환 (0~20)
  "RA3TCTFLR",  # verbal fluency: Category fluency 반복 수 (zero-inflated)
  "RA3TCTFLI"   # verbal fluency: Category fluency 침입 오류 수 (zero-inflated)
)

cognitive_vars <- c(cog_con_vars, cog_cnt_vars)

cognitive_selected <- cognitive_0001 %>%
  dplyr::select(MRID, all_of(cognitive_vars))

cat("\n=== Cognitive (37081) selected ===\n")
cat("dim:", dim(cognitive_selected), "\n\n")


################################################################################
# 2. 데이터타입 변환
################################################################################

# ─── Survey phenotype 변환 ──────────────────────────────────────────────────
survey_selected <- survey_selected %>%
  mutate(
    MRID = as.numeric(MRID),
    across(all_of(survey_bin_vars), icpsr_to_binary),
    across(all_of(survey_cnt_vars), to_numeric),
    across(all_of(survey_cnt_vars), ~ as.integer(floor(.x))),
    # painmed: B1SA13A~D → RA1SA13A~D 이진화 후 합산 → painmed_total (1~5 ordinal)
    across(all_of(survey_painmed_src), icpsr_to_binary),
    painmed_total = as.integer(rowSums(across(all_of(survey_painmed_src))) + 1L)
  )

cat("\n=== Survey phenotype 변환 후 ===\n")
for (v in c(survey_bin_vars, survey_cnt_vars, "painmed_total")) {
  cat(sprintf("[%s] ", v))
  print(table(survey_selected[[v]], useNA = "ifany"))
}

# ─── Biomarker 변환 ────────────────────────────────────────────────────────
biomarker_selected <- biomarker_selected %>%
  mutate(
    MRID = as.numeric(MRID),
    across(all_of(bio_ordinal_vars),    icpsr_to_int),
    across(all_of(bio_binary_vars),     icpsr_to_binary),
    across(all_of(bio_continuous_vars), to_numeric),
    # count 변수: to_numeric + floor (0.5 단위 절사)
    across(all_of(bio_count_vars), to_numeric),
    # skip pattern 처리: 비음주자 → 음주량 0 (RA4H49 주석처리로 비활성)
    # RA4H56 = ifelse(RA4H49 == 0 & is.na(RA4H56), 0, RA4H56),
    # count 절사 (floor)
    across(all_of(bio_count_vars), ~ as.integer(floor(.x))),
    # CESD reverse items: 먼저 역코딩 (1↔4, 2↔3) 후 이진화
    across(all_of(bio_cesd_reverse), ~ 5L - .x),
    # CESD 전체 이진화 (icpsr 코딩 1→0, 2/3/4→1)
    across(all_of(bio_cesd_vars), ~ ifelse(.x == 1, 0L, 1L)),
    # sleep trouble: RA4S11A~J 이진화(≥2 → 1, 1 → 0) 후 합산
    across(all_of(bio_sleep_src), icpsr_to_int),
    across(all_of(bio_sleep_src), ~ ifelse(.x >= 2, 1L, 0L)),
    sleep_trouble_count = as.integer(rowSums(across(all_of(bio_sleep_src))))
  )

cat("\n=== Biomarker 변환 후 class 확인 ===\n")
str(biomarker_selected, give.attr = FALSE)

cat("\n── Biomarker 변환 후 summary ──\n")
for (v in biomarker_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(summary(biomarker_selected[[v]]))
}

# ─── Cognitive 변환 ─────────────────────────────────────────────────────────
# Z-score 변수 중 높을수록 좋은 방향 → 부호 반전 (높=인지↓)
cog_reverse_vars <- c("RA3TEMZ3", "RA3TEFZ3")

cognitive_selected <- cognitive_selected %>%
  mutate(
    MRID = as.numeric(MRID),
    across(all_of(cognitive_vars), to_numeric),
    # Z-score: 부호 반전 (높=좋음 → 높=나쁨)
    across(all_of(cog_reverse_vars), ~ -.x),
    # 정확도 → 오답 수 변환: round(x/0.05) gives 0~20, then 20 - that = errors
    across(all_of(c("RA3TSPN", "RA3TSPR")), ~ as.integer(abs(round(.x / 0.05) - 20L))),
    # count 변수: integer 변환
    across(all_of(c("RA3TCTFLR", "RA3TCTFLI")), ~ as.integer(floor(.x)))
  )

cat("\n=== Cognitive 변환 후 class 확인 ===\n")
str(cognitive_selected, give.attr = FALSE)


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
    MRID = as.numeric(MRID),
    RA1PA60 = RA1PA60, RA1PA61 = RA1PA61, RA1PA62 = RA1PA62,
    RA1PA72 = RA1PA72, RA1PA73 = RA1PA73, RA1PA74 = RA1PA74,
    sadness_pass   = branch_pass(RA1PA60, RA1PA61, RA1PA62),
    anhedonia_pass = branch_pass(RA1PA72, RA1PA73, RA1PA74),
    screen_case = case_when(
      sadness_pass == 1 | anhedonia_pass == 1   ~ 1L,
      sadness_pass == 0 & anhedonia_pass == 0   ~ 0L,
      TRUE                                       ~ NA_integer_
    ),
    age = as.numeric(icpsr_code(RA1PRAGE)),
    sex = as.integer(icpsr_code(RA1PRSEX))
  )

# survey phenotype 변수를 df_screen에 merge
df_screen <- df_screen %>%
  left_join(
    survey_selected %>% dplyr::select(MRID, all_of(survey_bin_vars), all_of(survey_cnt_vars)),
    by = "MRID"
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

df_p4_pheno <- biomarker_selected   # MRID + biomarker_vars
df_p3_pheno <- cognitive_selected   # MRID + cognitive_vars

cat("\n=== df_p4_pheno ===\n")
cat("dim:", dim(df_p4_pheno), "\n")
cat("\n=== df_p3_pheno ===\n")
cat("dim:", dim(df_p3_pheno), "\n")


################################################################################
# 6. 분석용 데이터셋 (inner join)
################################################################################

df_analysis_p4 <- df_screen %>%
  inner_join(df_p4_pheno, by = "MRID")

df_analysis_p3 <- df_screen %>%
  inner_join(df_p3_pheno, by = "MRID")

df_analysis_all <- df_screen %>%
  inner_join(df_p3_pheno, by = "MRID") %>%
  inner_join(df_p4_pheno, by = "MRID")

cat("\n=== 분석용 데이터셋 크기 ===\n")
cat("df_analysis_p4  (Survey×Bio):      ", nrow(df_analysis_p4),  "명 ×", ncol(df_analysis_p4),  "변수\n")
cat("df_analysis_p3  (Survey×Cog):      ", nrow(df_analysis_p3),  "명 ×", ncol(df_analysis_p3),  "변수\n")
cat("df_analysis_all (Survey×Cog×Bio):  ", nrow(df_analysis_all), "명 ×", ncol(df_analysis_all), "변수\n")

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

split_for_lsirm <- function(df, bin_vars, cnt_vars, ord1_vars, ord2_vars, con_vars, label) {

  df_case <- df %>% filter(screen_case == 1)
  cat(sprintf("\n=== %s: screen_case==1 → %d명 ===\n", label, nrow(df_case)))

  pheno_vars <- c(bin_vars, cnt_vars, ord1_vars, ord2_vars, con_vars)
  df_pheno <- df_case %>% dplyr::select(MRID, all_of(pheno_vars))

  na_cnt <- colSums(is.na(df_pheno[, pheno_vars, drop = FALSE]))
  cat("  변수별 NA (제거 전):\n")
  print(na_cnt[na_cnt > 0])
  cat(sprintf("  NA 없는 complete case: %d / %d명\n",
              sum(complete.cases(df_pheno[, pheno_vars])), nrow(df_pheno)))

  df_clean <- na.omit(df_pheno)
  cat(sprintf("  → NA 제거 후: %d명 (제거 %d명)\n",
              nrow(df_clean), nrow(df_pheno) - nrow(df_clean)))

  row_ids  <- df_clean$MRID
  branch_info <- df_case %>%
    dplyr::select(MRID, sadness_pass, anhedonia_pass) %>%
    filter(MRID %in% row_ids) %>%
    arrange(match(MRID, row_ids)) %>%
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


# ─── Survey–Biomarker ───────────────────────────────────────────────────────
lsirm_p4 <- split_for_lsirm(
  df        = df_analysis_p4,
  bin_vars  = c(survey_bin_vars, bio_binary_vars, bio_cesd_vars),
  cnt_vars  = c(survey_cnt_vars, bio_count_vars, "sleep_trouble_count"),
  ord1_vars = c(bio_ord5_vars),
  ord2_vars = c(bio_ord4_vars),
  con_vars  = bio_continuous_vars,
  label     = "Survey-Bio"
)

# ─── Survey–Cognitive ────────────────────────────────────────────────────────
lsirm_p3 <- split_for_lsirm(
  df        = df_analysis_p3,
  bin_vars  = survey_bin_vars,
  cnt_vars  = c(survey_cnt_vars, cog_cnt_vars),
  ord1_vars = character(0),
  ord2_vars = character(0),
  con_vars  = cog_con_vars,
  label     = "Survey-Cog"
)

# ─── Survey–Cognitive–Biomarker ──────────────────────────────────────────────
lsirm_all <- split_for_lsirm(
  df        = df_analysis_all,
  bin_vars  = c(survey_bin_vars, bio_binary_vars, bio_cesd_vars),
  cnt_vars  = c(survey_cnt_vars, bio_count_vars, "sleep_trouble_count", cog_cnt_vars),
  ord1_vars = c(bio_ord5_vars),
  ord2_vars = c(bio_ord4_vars),
  con_vars  = c(bio_continuous_vars, cog_con_vars),
  label     = "Survey-Cog-Bio"
)