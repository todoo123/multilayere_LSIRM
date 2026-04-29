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
# 1. variable selection (v4: 5-layer with empty ord2)
#    - binary: bio_cesd_vars (이진화)
#    - continuous: bio_continuous_vars (염증 관련 등)
#    - count: 새 cognitive score count 변수 (정수)
#    - ordinal-5: bio_ord5_vars (MASQ)
#    - ordinal-4: 비워둠
################################################################################

# ─── 04652 (M2P1): Survey — screening 변수 ──────────────────────────────────
survey_screen_vars <- c(
  "B1PA60", "B1PA61", "B1PA62",   # sadness branch
  "B1PA72", "B1PA73", "B1PA74"    # anhedonia branch
)

survey_bin_vars <- character(0)
survey_cnt_vars <- character(0)

survey_vars <- c(survey_screen_vars, survey_bin_vars, survey_cnt_vars)

survey_selected <- survey_0001 %>%
  dplyr::select(M2ID, all_of(survey_vars))

cat("\n=== Survey (04652 / M2P1) selected ===\n")
cat("dim:", dim(survey_selected), "\n\n")


# ─── 29282 (M2P4): Biomarker — CESD / MASQ / inflammation 변수 ──────────────

# ordinal-5: MASQ 5-point Likert (1~5), GDA(11) + AA(17)
bio_ord5_vars <- c(
  # --- General Distress–Anxious (GDA, 11) ---
  "B4Q1D", "B4Q1H", "B4Q1K", "B4Q1N", "B4Q1P", "B4Q1T",
  "B4Q1Z", "B4Q1FF", "B4Q1II", "B4Q1CCC", "B4Q1GGG",
  # --- Anxious Arousal (AA, 17) ---
  "B4Q1B", "B4Q1F", "B4Q1M", "B4Q1Q", "B4Q1S", "B4Q1X",
  "B4Q1BB", "B4Q1DD", "B4Q1KK", "B4Q1NN", "B4Q1PP", "B4Q1RR",
  "B4Q1TT", "B4Q1VV", "B4Q1ZZ", "B4Q1BBB", "B4Q1JJJ"
)

# CESD 20문항 (1~4) → binary 이진화 (icpsr 1 → 0, 2/3/4 → 1)
bio_cesd_vars <- c(
  "B4Q3A", "B4Q3B", "B4Q3C", "B4Q3D", "B4Q3E", "B4Q3F", "B4Q3G",
  "B4Q3H", "B4Q3I", "B4Q3J", "B4Q3K", "B4Q3L", "B4Q3M", "B4Q3N",
  "B4Q3O", "B4Q3P", "B4Q3Q", "B4Q3R", "B4Q3S", "B4Q3T"
)
bio_cesd_reverse <- c("B4Q3D", "B4Q3H", "B4Q3L", "B4Q3P")

# continuous: HPA / autonomic / inflammation 관련 biomarker
bio_continuous_vars <- c(
  "B4BSCL14", # HPA axis: Saliva cortisol all sample average
  "B4BNE12",  # autonomic arousal: Urine Norepinephrine 12 hour
  "B4BIL6",   # inflammation: Blood Serum IL6 (pg/mL)
  "B4BFGN",   # inflammation: Blood Fibrinogen (mg/dL)
  "B4BCRP"    # inflammation: Blood C-Reactive Protein (ug/mL)
)

biomarker_vars <- c(bio_ord5_vars, bio_cesd_vars, bio_continuous_vars)

biomarker_selected <- biomarker_0001 %>%
  dplyr::select(M2ID, all_of(biomarker_vars))

cat("\n=== Biomarker (29282 / M2P4) selected ===\n")
cat("dim:", dim(biomarker_selected), "\n\n")


# ─── 25281 (M2P3): Cognitive — 새 cognitive score count 변수 ────────────────
# 원변수 → 인지 오류/누락 count (높을수록 인지↓)
#   wl_immediate_omit         = 15 - B3TWLITU
#   wl_immediate_repetition   = B3TWLITR
#   wl_immediate_intrusion    = B3TWLITI
#   wl_delayed_omit           = 15 - B3TWLDTU
#   wl_delayed_repetition     = B3TWLDTR
#   wl_delayed_intrusion      = B3TWLDTI
#   catflu_repetition         = B3TCTFLR
#   catflu_intrusion          = B3TCTFLI
#   numseries_incorrect       = 5  - B3TNSTOT
#   backcount_error           = B3TBKERR
#   sgst_normal_incorrect     = 20 - B3TSTN
#   sgst_reverse_incorrect    = 20 - B3TSTR
#   sgst_mixed_nonswitch_incorrect = 23 - B3TSTXBO
#   sgst_mixed_switch_incorrect    =  6 - B3TSTXBS
#   sgst_mixed_all_incorrect       = 29 - B3TSTXBB

cog_cnt_src <- c(
  "B3TWLITU", "B3TWLITR", "B3TWLITI",
  "B3TWLDTU", "B3TWLDTR", "B3TWLDTI",
  "B3TCTFLR", "B3TCTFLI",
  "B3TNSTOT", "B3TBKERR",
  "B3TSTN",   "B3TSTR",
  "B3TSTXBO", "B3TSTXBS", "B3TSTXBB"
)

cognitive_selected <- cognitive_0001 %>%
  dplyr::select(M2ID, all_of(cog_cnt_src))

cat("\n=== Cognitive (25281 / M2P3) selected ===\n")
cat("dim:", dim(cognitive_selected), "\n\n")


################################################################################
# 2. 데이터타입 변환
################################################################################

# ─── Survey: screening 변수만 사용하므로 변환 불필요 ───────────────────────
survey_selected <- survey_selected %>%
  mutate(M2ID = as.numeric(M2ID))

# ─── Biomarker 변환 ─────────────────────────────────────────────────────────
biomarker_selected <- biomarker_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(bio_ord5_vars),        icpsr_to_int),
    across(all_of(bio_cesd_vars),        icpsr_to_int),
    across(all_of(bio_continuous_vars),  to_numeric),
    # CESD reverse items: 먼저 역코딩 (1↔4, 2↔3) 후 이진화
    across(all_of(bio_cesd_reverse), ~ 5L - .x),
    # CESD 전체 이진화 (icpsr 코딩 1→0, 2/3/4→1)
    across(all_of(bio_cesd_vars), ~ ifelse(.x == 1, 0L, 1L))
  )

cat("\n=== Biomarker 변환 후 class 확인 ===\n")
str(biomarker_selected, give.attr = FALSE)


# ─── Cognitive 변환: 원변수 → count 형태 (높을수록 인지↓) ──────────────────
cognitive_selected <- cognitive_selected %>%
  mutate(
    M2ID = as.numeric(M2ID),
    across(all_of(cog_cnt_src), to_numeric),
    # Word List immediate/delayed: omit = 15 - correct (정수)
    wl_immediate_omit        = as.integer(15L - round(B3TWLITU)),
    wl_immediate_repetition  = as.integer(round(B3TWLITR)),
    wl_immediate_intrusion   = as.integer(round(B3TWLITI)),
    wl_delayed_omit          = as.integer(15L - round(B3TWLDTU)),
    wl_delayed_repetition    = as.integer(round(B3TWLDTR)),
    wl_delayed_intrusion     = as.integer(round(B3TWLDTI)),
    # Category fluency (raw count)
    catflu_repetition        = as.integer(round(B3TCTFLR)),
    catflu_intrusion         = as.integer(round(B3TCTFLI)),
    # Number Series 오답: 5 - total correct
    numseries_incorrect      = as.integer(5L - round(B3TNSTOT)),
    # Backward Counting: 이미 오류 수
    backcount_error          = as.integer(round(B3TBKERR)),
    # SGST (correct count → incorrect count = max - correct)
    sgst_normal_incorrect           = as.integer(20L - round(B3TSTN)),
    sgst_reverse_incorrect          = as.integer(20L - round(B3TSTR)),
    sgst_mixed_nonswitch_incorrect  = as.integer(23L - round(B3TSTXBO)),
    sgst_mixed_switch_incorrect     = as.integer( 6L - round(B3TSTXBS)),
    sgst_mixed_all_incorrect        = as.integer(29L - round(B3TSTXBB))
  )

# 새 count 변수 이름 (LSIRM에서 사용할 최종 변수)
cog_cnt_vars <- c(
  "wl_immediate_omit", "wl_immediate_repetition", "wl_immediate_intrusion",
  "wl_delayed_omit",   "wl_delayed_repetition",   "wl_delayed_intrusion",
  "catflu_repetition", "catflu_intrusion",
  "numseries_incorrect", "backcount_error",
  "sgst_normal_incorrect", "sgst_reverse_incorrect",
  "sgst_mixed_nonswitch_incorrect", "sgst_mixed_switch_incorrect",
  "sgst_mixed_all_incorrect"
)

cat("\n=== Cognitive 변환 후 summary (new count vars) ===\n")
for (v in cog_cnt_vars) {
  cat(sprintf("\n[%s]\n", v))
  print(summary(cognitive_selected[[v]]))
}


################################################################################
# 3. Screening & case indicator
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

cat("\n=== Screening 결과 ===\n")
cat("전체:", nrow(df_screen), "\n")
print(table(screen_case = df_screen$screen_case, useNA = "ifany"))


################################################################################
# 4. Phenotype 데이터프레임
################################################################################

df_p4_pheno <- biomarker_selected   # M2ID + biomarker_vars
df_p3_pheno <- cognitive_selected %>%
  dplyr::select(M2ID, all_of(cog_cnt_vars))

cat("\n=== df_p4_pheno ===\n"); cat("dim:", dim(df_p4_pheno), "\n")
cat("\n=== df_p3_pheno ===\n"); cat("dim:", dim(df_p3_pheno), "\n")


################################################################################
# 5. 분석용 데이터셋 (inner join)
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


################################################################################
# 6. screen_case == 1 필터링 → 5-layer matrix 분류 (LSIRM input)
################################################################################
# v4 layer 구성:
#   Y_bin  : bio_cesd_vars (20)
#   Y_con  : bio_continuous_vars (5)
#   Y_cnt  : cog_cnt_vars (15, 새 cognitive score count)
#   Y_ord1 : bio_ord5_vars (28, MASQ)
#   Y_ord2 : EMPTY

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


# ─── P1–P4 (Survey × Biomarker) ─────────────────────────────────────────────
lsirm_p4 <- split_for_lsirm(
  df        = df_analysis_p4,
  bin_vars  = bio_cesd_vars,
  cnt_vars  = character(0),
  ord1_vars = bio_ord5_vars,
  ord2_vars = character(0),
  con_vars  = bio_continuous_vars,
  label     = "P1-P4"
)

# ─── P1–P3 (Survey × Cognitive) ─────────────────────────────────────────────
lsirm_p3 <- split_for_lsirm(
  df        = df_analysis_p3,
  bin_vars  = character(0),
  cnt_vars  = cog_cnt_vars,
  ord1_vars = character(0),
  ord2_vars = character(0),
  con_vars  = character(0),
  label     = "P1-P3"
)

# ─── P1–P3–P4 (Survey × Cognitive × Biomarker) ──────────────────────────────
lsirm_all <- split_for_lsirm(
  df        = df_analysis_all,
  bin_vars  = bio_cesd_vars,
  cnt_vars  = cog_cnt_vars,
  ord1_vars = bio_ord5_vars,
  ord2_vars = character(0),
  con_vars  = bio_continuous_vars,
  label     = "P1-P3-P4"
)
