KSAH<-read.csv("/Users/hyunseokyoon/Desktop/학교/대학원/Project/사회학과_연구협력_염유식교수님/KSAH_데이터 및 코드북/KSAH_student_230521.csv")
unique(KSAH$wave)
KSAH_wave1 <- KSAH[KSAH$wave == 1, ]

####################################################
# continuous item
####################################################
# 5점 likert
KSAH_wave1$swb_6 <- 6 - KSAH_wave1$swb_6
KSAH_wave1_swb <- 6 - KSAH_wave1[, paste0("swb_", 1:6)]
KSAH_wave1_dad <- 6 - KSAH_wave1[, paste0("dad_", 1:8)]
KSAH_wave1_mom <- 6 - KSAH_wave1[, paste0("mom_", 1:8)]
KSAH_wave1_tutor_rel <- 6 - KSAH_wave1[, paste0("tutor_rel_", 1:4)]
KSAH_wave1[, paste0("pss_", c(1,2,3,6,9,10))] <- 6 - KSAH_wave1[, paste0("pss_", c(1,2,3,6,9,10))]
KSAH_wave1_pss <- 6 - KSAH_wave1[, paste0("pss_", 1:10)]
# NA 만 있음. 아마 다른 wave에서 수집한듯
# KSAH_wave1_spane <- KSAH_wave1[, paste0("spane_", 1:12)]
KSAH_wave1_vaccine <- KSAH_wave1[, paste0("vaccine")]
KSAH_wave1[, "gender_1"] <- 6 - KSAH_wave1[, "gender_1"]
KSAH_wave1_gender <- KSAH_wave1[, paste0("gender_", 1:2)]
likert_5 <- cbind(KSAH_wave1_swb, KSAH_wave1_dad, KSAH_wave1_mom, KSAH_wave1_tutor_rel, 
                  KSAH_wave1_pss, KSAH_wave1_vaccine, KSAH_wave1_gender)

colnames(likert_5)[35]
colnames(likert_5)[29]
colnames(likert_5)[37]
colnames(likert_5)[39]
colnames(likert_5)[31]
colnames(likert_5)[30]
colnames(likert_5)[28]
# 4점 likert
KSAH_wave1_phq <- KSAH_wave1[, paste0("phq_", 1:9)]
KSAH_wave1_gad <- KSAH_wave1[, paste0("gad_", 1:7)]
# NA 만 있음. 아마 다른 wave에서 수집한듯
# KSAH_wave1_bisbas <- KSAH_wave1[, paste0("bisbas_", 1:20)]
KSAH_wave1_dieting <- KSAH_wave1[, paste0("dieting_", 1:3)]
KSAH_wave1_distress <- KSAH_wave1[, paste0("distress")]

likert_4 <- cbind(KSAH_wave1_phq, KSAH_wave1_gad, KSAH_wave1_dieting, KSAH_wave1_distress)

# 3점 likert
KSAH_wave1_lonely <- KSAH_wave1[, paste0("lonely_", 1:3)]
likert_3 <- KSAH_wave1_lonely

# 7점 likert
KSAH_wave1_tipi <- KSAH_wave1[, paste0("tipi_", 1:10)]
# NA 만 있음. 아마 다른 wave에서 수집한듯
# KSAH_wave1_flourish <- KSAH_wave1[, paste0("flourish_", 1:8)]
likert_7 <- KSAH_wave1_tipi

# 10점 likert
KSAH_wave1_sdo <- KSAH_wave1[, paste0("sdo_", 1:4)]
likert_10 <- KSAH_wave1_sdo

####################################################
# binary item
####################################################
KSAH_wave1_cohab <- KSAH_wave1[, paste0("cohab_", 1:11)]
KSAH_wave1_bully <- KSAH_wave1[, paste0("bully")] - 1
KSAH_wave1_victim <- KSAH_wave1[, paste0("victim")] - 1
# NA 값만있음 - 고려 안할것
# KSAH_wave1_worldview_2 <- KSAH_wave1[, paste0("worldview_2")]
KSAH_wave1_overeating <- KSAH_wave1[, paste0("overeating")] - 1
KSAH_wave1_purging <- KSAH_wave1[, paste0("purging")] - 1

# bin <- cbind(KSAH_wave1_cohab, KSAH_wave1_bully, KSAH_wave1_victim,
#                   KSAH_wave1_overeating, KSAH_wave1_purging)

bin <- cbind(KSAH_wave1_overeating, KSAH_wave1_purging)

####################################################
# count item
####################################################
KSAH_wave1_tutor_h <- KSAH_wave1[, paste0("tutor_h")]

KSAH_wave1_tutor_h[is.na(KSAH_wave1_tutor_h)] = 0
KSAH_wave1_otutor_h <- KSAH_wave1[, paste0("otutor_h")]
# 0 값이 너무 많아서 negative binomial 로 설명하기 곤란해 보임
# KSAH_wave1_otutor_h[is.na(KSAH_wave1_otutor_h)] = 0
# hist(KSAH_wave1_otutor_h)
KSAH_wave1_wday_slp_h <- KSAH_wave1[, paste0("wday_slp_h")]
KSAH_wave1_wday_slp_h[KSAH_wave1_wday_slp_h == 50] = NA
KSAH_wave1_wday_std_h <- KSAH_wave1[, paste0("wday_std_h")]
KSAH_wave1_wday_std_h[KSAH_wave1_wday_std_h == 17] = NA
KSAH_wave1_wend_slp_h <- KSAH_wave1[, paste0("wend_slp_h")]
KSAH_wave1_wend_slp_h[KSAH_wave1_wend_slp_h == 20] = NA
KSAH_wave1_wend_std_h <- KSAH_wave1[, paste0("wend_std_h")]
# 한 개만 1 이고 나머지 NA 여서 제거
# KSAH_wave1_victim_n <- KSAH_wave1[, paste0("victim_n")]
# 분포가 negative binomial 로 설명하기 곤란함.
# KSAH_wave1_overeating <- KSAH_wave1[, paste0("overeating")]
# KSAH_wave1_purging_n <- KSAH_wave1[, paste0("purging_n")]
# KSAH_wave1_purging_n[is.na(KSAH_wave1_purging_n)] = 0

# count <- cbind(KSAH_wave1_tutor_h, KSAH_wave1_otutor_h, KSAH_wave1_wday_slp_h,
#                KSAH_wave1_wday_std_h, KSAH_wave1_wend_slp_h, KSAH_wave1_wend_std_h)
count <- cbind(KSAH_wave1_tutor_h, KSAH_wave1_wday_slp_h)
dim(count)
dim(bin)
dim(likert_3)
dim(likert_4)
dim(likert_5)
dim(likert_7)
dim(likert_10)
# colnames(likert_5[,c(7,8,9,10)])
# colnames(likert_5[,c(11,24)])
# colnames(bin[,c(10,12,13,14,15)])
index <- rowSums(is.na(count)) + rowSums(is.na(bin)) + rowSums(is.na(likert_3)) +
  rowSums(is.na(likert_4)) + rowSums(is.na(likert_5)) + rowSums(is.na(likert_7)) + 
  rowSums(is.na(likert_10))
index <- index == 0

Y_cnt <- count[index, ]
Y_bin <- bin[index, ]
Y_likert_3 <- likert_3[index, ]
Y_likert_4 <- likert_4[index, ]
Y_likert_5 <- likert_5[index, ]
Y_likert_7 <- likert_7[index, ]
Y_likert_10 <- likert_10[index, ]

