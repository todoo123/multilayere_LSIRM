rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. load — MIDUS Refresher 1 데이터
################################################################################
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data/MIDUS_refresher_1")

# ─── ICPSR_36532: Survey (MIDUS Refresher 1) ────────────────────────────────
load("ICPSR_36532/DS0001/36532-0001-Data.rda")
survey_0001 <- da36532.0001

# ─── ICPSR_36901: Biomarker (MIDUS Refresher 1) ─────────────────────────────
load("ICPSR_36901/DS0001/36901-0001-Data.rda")
biomarker_0001 <- da36901.0001

# ─── ICPSR_37081: Cognitive (MIDUS Refresher 1) ─────────────────────────────
load("ICPSR_37081/DS0001/37081-0001-Data.rda")
cognitive_0001 <- da37081.0001

dim(survey_0001)
dim(biomarker_0001)
dim(cognitive_0001)

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
# 1. variable selection (v4: 5-layer with empty ord2, refresher_1)
################################################################################

# ─── 36532: Survey — screening ───────────────────────────────────────────────
survey_screen_vars <- c(
  "RA1PA60", "RA1PA61", "RA1PA62",   # sadness branch
  "RA1PA72", "RA1PA73", "RA1PA74"    # anhedonia branch
)

survey_bin_vars <- character(0)
survey_cnt_vars <- character(0)

survey_vars <- c(survey_screen_vars, survey_bin_vars, survey_cnt_vars)

survey_selected <- survey_0001 %>%
  dplyr::select(MRID, all_of(survey_vars))

cat("\n=== Survey (36532) selected ===\n")
cat("dim:", dim(survey_selected), "\n\n")


# ─── 36901: Biomarker — CESD / MASQ / inflammation 변수 ─────────────────────

# ordinal-5: MASQ (1~5) — GDA(11) + AA(17)
bio_ord5_vars <- c(
  "RA4Q1D", "RA4Q1H", "RA4Q1K", "RA4Q1N", "RA4Q1P", "RA4Q1T",
  "RA4Q1Z", "RA4Q1FF", "RA4Q1II", "RA4Q1CCC", "RA4Q1GGG",
  "RA4Q1B", "RA4Q1F", "RA4Q1M", "RA4Q1Q", "RA4Q1S", "RA4Q1X",
  "RA4Q1BB", "RA4Q1DD", "RA4Q1KK", "RA4Q1NN", "RA4Q1PP", "RA4Q1RR",
  "RA4Q1TT", "RA4Q1VV", "RA4Q1ZZ", "RA4Q1BBB", "RA4Q1JJJ"
)

# CESD 20문항 → binary 이진화
bio_cesd_vars <- c(
  "RA4Q3A", "RA4Q3B", "RA4Q3C", "RA4Q3D", "RA4Q3E", "RA4Q3F", "RA4Q3G",
  "RA4Q3H", "RA4Q3I", "RA4Q3J", "RA4Q3K", "RA4Q3L", "RA4Q3M", "RA4Q3N",
  "RA4Q3O", "RA4Q3P", "RA4Q3Q", "RA4Q3R", "RA4Q3S", "RA4Q3T"
)
bio_cesd_reverse <- c("RA4Q3D", "RA4Q3H", "RA4Q3L", "RA4Q3P")

# continuous — RA4BSCL4A 제거 (refresher_1에 없음)
bio_continuous_vars <- c(
  "RA4BSCLAV", # HPA axis: Saliva cortisol all sample average
  "RA4BNE12",  # autonomic arousal: Urine Norepinephrine 12 hour
  "RA4BIL6",   # inflammation: Blood Serum IL6 (pg/mL)
  "RA4BFGN",   # inflammation: Blood Fibrinogen (mg/dL)
  "RA4BCRP"    # inflammation: Blood C-Reactive Protein (ug/mL)
)

biomarker_vars <- c(bio_ord5_vars, bio_cesd_vars, bio_continuous_vars)

biomarker_selected <- biomarker_0001 %>%
  dplyr::select(MRID, all_of(biomarker_vars))

cat("\n=== Biomarker (36901) selected ===\n")
cat("dim:", dim(biomarker_selected), "\n\n")


# ─── 37081: Cognitive — 새 cognitive score count 변수 ───────────────────────
# 원변수 prefix RA3T (wave 2의 B3T에 대응)
cog_cnt_src <- c(
  "RA3TWLITU", "RA3TWLITR", "RA3TWLITI",
  "RA3TWLDTU", "RA3TWLDTR", "RA3TWLDTI",
  "RA3TCTFLR", "RA3TCTFLI",
  "RA3TNSTOT", "RA3TBKERR",
  "RA3TSTN",   "RA3TSTR",
  "RA3TSTXBO", "RA3TSTXBS", "RA3TSTXBB"
)

cognitive_selected <- cognitive_0001 %>%
  dplyr::select(MRID, all_of(cog_cnt_src))

cat("\n=== Cognitive (37081) selected ===\n")
cat("dim:", dim(cognitive_selected), "\n\n")


################################################################################
# 2. 데이터타입 변환
################################################################################

# ─── Survey ─────────────────────────────────────────────────────────────────
survey_selected <- survey_selected %>%
  mutate(MRID = as.numeric(MRID))

# ─── Biomarker ──────────────────────────────────────────────────────────────
biomarker_selected <- biomarker_selected %>%
  mutate(
    MRID = as.numeric(MRID),
    across(all_of(bio_ord5_vars),        icpsr_to_int),
    across(all_of(bio_cesd_vars),        icpsr_to_int),
    across(all_of(bio_continuous_vars),  to_numeric),
    across(all_of(bio_cesd_reverse), ~ 5L - .x),
    across(all_of(bio_cesd_vars), ~ ifelse(.x == 1, 0L, 1L))
  )

cat("\n=== Biomarker 변환 후 class 확인 ===\n")
str(biomarker_selected, give.attr = FALSE)


# ─── Cognitive 변환: 원변수 → count 형태 (높을수록 인지↓) ──────────────────
# wave 2 v4와 동일한 변환 로직
cognitive_selected <- cognitive_selected %>%
  mutate(
    MRID = as.numeric(MRID),
    across(all_of(cog_cnt_src), to_numeric),
    wl_immediate_omit        = as.integer(15L - round(RA3TWLITU)),
    wl_immediate_repetition  = as.integer(round(RA3TWLITR)),
    wl_immediate_intrusion   = as.integer(round(RA3TWLITI)),
    wl_delayed_omit          = as.integer(15L - round(RA3TWLDTU)),
    wl_delayed_repetition    = as.integer(round(RA3TWLDTR)),
    wl_delayed_intrusion     = as.integer(round(RA3TWLDTI)),
    catflu_repetition        = as.integer(round(RA3TCTFLR)),
    catflu_intrusion         = as.integer(round(RA3TCTFLI)),
    numseries_incorrect      = as.integer(5L - round(RA3TNSTOT)),
    backcount_error          = as.integer(round(RA3TBKERR)),
    sgst_normal_incorrect           = as.integer(20L - round(RA3TSTN)),
    sgst_reverse_incorrect          = as.integer(20L - round(RA3TSTR)),
    sgst_mixed_nonswitch_incorrect  = as.integer(23L - round(RA3TSTXBO)),
    sgst_mixed_switch_incorrect     = as.integer( 6L - round(RA3TSTXBS)),
    sgst_mixed_all_incorrect        = as.integer(29L - round(RA3TSTXBB))
  )

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

cat("\n=== Screening 결과 ===\n")
cat("전체:", nrow(df_screen), "\n")
print(table(screen_case = df_screen$screen_case, useNA = "ifany"))


################################################################################
# 4. Phenotype 데이터프레임
################################################################################

df_p4_pheno <- biomarker_selected
df_p3_pheno <- cognitive_selected %>%
  dplyr::select(MRID, all_of(cog_cnt_vars))

cat("\n=== df_p4_pheno ===\n"); cat("dim:", dim(df_p4_pheno), "\n")
cat("\n=== df_p3_pheno ===\n"); cat("dim:", dim(df_p3_pheno), "\n")


################################################################################
# 5. 분석용 데이터셋 (inner join)
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


################################################################################
# 6. screen_case == 1 필터링 → 5-layer matrix 분류 (LSIRM input)
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


# ─── Survey × Biomarker ─────────────────────────────────────────────────────
lsirm_p4 <- split_for_lsirm(
  df        = df_analysis_p4,
  bin_vars  = bio_cesd_vars,
  cnt_vars  = character(0),
  ord1_vars = bio_ord5_vars,
  ord2_vars = character(0),
  con_vars  = bio_continuous_vars,
  label     = "Survey-Bio"
)

# ─── Survey × Cognitive ─────────────────────────────────────────────────────
lsirm_p3 <- split_for_lsirm(
  df        = df_analysis_p3,
  bin_vars  = character(0),
  cnt_vars  = cog_cnt_vars,
  ord1_vars = character(0),
  ord2_vars = character(0),
  con_vars  = character(0),
  label     = "Survey-Cog"
)

# ─── Survey × Cognitive × Biomarker ─────────────────────────────────────────
lsirm_all <- split_for_lsirm(
  df        = df_analysis_all,
  bin_vars  = bio_cesd_vars,
  cnt_vars  = cog_cnt_vars,
  ord1_vars = bio_ord5_vars,
  ord2_vars = character(0),
  con_vars  = bio_continuous_vars,
  label     = "Survey-Cog-Bio"
)
