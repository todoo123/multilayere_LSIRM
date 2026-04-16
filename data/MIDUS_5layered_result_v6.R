rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. 경로 설정
################################################################################
data_dir <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data"
setwd(data_dir)

################################################################################
# 0-1. 데이터 준비: Wave 2 + Refresher 1 합치기
################################################################################

# ── Wave 2 전처리 (격리 환경) ──
cat("\n====== Wave 2 전처리 ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_2_v3.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all
lsirm_p4_w2  <- env_w2$lsirm_p4
lsirm_p3_w2  <- env_w2$lsirm_p3

# ── Refresher 1 전처리 (격리 환경) ──
cat("\n====== Refresher 1 전처리 ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v3.R"), local = env_r1)
lsirm_all_r1 <- env_r1$lsirm_all
lsirm_p4_r1  <- env_r1$lsirm_p4
lsirm_p3_r1  <- env_r1$lsirm_p3

# ── 합치기 함수 ──
combine_lsirm <- function(l_w2, l_r1, label = "") {
  rbind_mat <- function(m1, m2) {
    if (ncol(m1) == 0 && ncol(m2) == 0) return(m1)
    colnames(m2) <- colnames(m1)
    rbind(m1, m2)
  }

  Y_bin  <- rbind_mat(l_w2$Y_bin,  l_r1$Y_bin)
  Y_cnt  <- rbind_mat(l_w2$Y_cnt,  l_r1$Y_cnt)
  Y_ord1 <- rbind_mat(l_w2$Y_ord1, l_r1$Y_ord1)
  Y_ord2 <- rbind_mat(l_w2$Y_ord2, l_r1$Y_ord2)
  Y_con  <- rbind_mat(l_w2$Y_con,  l_r1$Y_con)

  combined <- list(
    Y_bin  = Y_bin,  Y_cnt  = Y_cnt,
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2,
    Y_con  = Y_con,
    row_ids  = c(l_w2$row_ids, l_r1$row_ids),
    branch   = c(l_w2$branch,  l_r1$branch),
    source   = c(rep("wave2", length(l_w2$row_ids)),
                 rep("refresher1", length(l_r1$row_ids))),
    col_bin  = l_w2$col_bin,  col_cnt  = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )

  n_w2 <- length(l_w2$row_ids)
  n_r1 <- length(l_r1$row_ids)
  cat(sprintf("\n=== %s: Combined %d (W2) + %d (R1) = %d명 ===\n",
              label, n_w2, n_r1, n_w2 + n_r1))
  cat(sprintf("  Y_bin: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d | Y_con: %d×%d\n",
              nrow(Y_bin), ncol(Y_bin), nrow(Y_cnt), ncol(Y_cnt),
              nrow(Y_ord1), ncol(Y_ord1), nrow(Y_ord2), ncol(Y_ord2),
              nrow(Y_con), ncol(Y_con)))
  combined
}

lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1, label = "P1-P3-P4")

# 5-layered hierarchical LSIRM (v6: GRM ordinal + robust continuous + hierarchical positions) 모델 로드
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_5layered_hierarchical_cpp_v6.R"))
source(file.path(data_dir, "utils.R"))

################################################################################
# Helper: array/matrix 가 유효한 데이터를 가지고 있는지 확인
################################################################################
has_valid <- function(x) {
  if (is.null(x)) return(FALSE)
  if (is.matrix(x) && ncol(x) == 0) return(FALSE)
  if (is.array(x) && length(dim(x)) == 3 && dim(x)[2] == 0) return(FALSE)
  if (all(is.na(x))) return(FALSE)
  TRUE
}

################################################################################
# 1. 공통 hyperparameter / MCMC 설정
################################################################################
common_hyper <- list(
  a_sigma=1, b_sigma=1,
  a_tau1=1, b_tau1=1, a_tau2=1, b_tau2=1, a_tau3=1, b_tau3=1,
  a_sigma0=3, b_sigma0=2,
  mu_log_gamma1=0, sd_log_gamma1=0.5,
  mu_log_gamma2=0, sd_log_gamma2=0.5,
  mu_log_gamma3=0, sd_log_gamma3=0.5,
  mu_log_gamma4=0, sd_log_gamma4=0.5,
  mu_log_gamma5=0, sd_log_gamma5=0.5,
  mu_log_kappa=0, sd_log_kappa=0.5,
  mu_beta4=0, sd_beta4=2,
  mu_beta5=0, sd_beta5=2,
  a_sigma_global=3, b_sigma_global=1
)

common_prop_sd <- list(
  alpha1=0.5, alpha2=0.5, alpha3=0.5, alpha4=0.5, alpha5=0.5,
  log_gamma1=0.1, log_gamma2=0.05, log_gamma3=0.10, log_gamma4=0.2, log_gamma5=0.2,
  a1=0.3, a2=0.3, a3=0.3, a4=0.3, a5=0.3,
  beta1=0.6, beta2=0.25, beta3=0.2, beta4=0.3, beta5=0.3,
  b1=0.35, b2=0.2, b3=0.2, b4=0.5, b5=0.5,
  log_kappa=0.4
)

common_mcmc <- list(d = 3, n_iter = 50000, burnin = 10000, thin = 10)

# Robust continuous layer: degrees of freedom
nu2 <- 4

# plot 루트 디렉토리
plot_root <- file.path(data_dir, "plot")
if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE)

################################################################################
# 2. Empty matrix helper
################################################################################
make_empty <- function(n) matrix(0L, nrow = n, ncol = 0)

################################################################################
# 3. Case 정의 (lsirm_all 기준, Y_ord2 항상 제외)
################################################################################
n_all <- nrow(lsirm_all$Y_con)

Y_con_full  <- scale(lsirm_all$Y_con)
# Y_con_full  <- Y_con_full[, , drop = FALSE]          # strip scale/center attrs
# attr(Y_con_full, "scaled:center") <- NULL
# attr(Y_con_full, "scaled:scale")  <- NULL
# 
# # --- Yeo-Johnson 변환 (car::yjPower + 직접 MLE) ---
# library(car)
# 
# # Yeo-Johnson 변환 함수 (car::yjPower와 동일)
# yj_transform <- function(x, lambda) {
#   car::yjPower(x, lambda = lambda)
# }
# 
# # Yeo-Johnson log-likelihood → lambda MLE
# yj_mle_lambda <- function(x) {
#   neg_ll <- function(lambda) {
#     xt <- yj_transform(x, lambda)
#     n <- length(x)
#     sigma2 <- var(xt) * (n - 1) / n
#     ll <- -n/2 * log(sigma2)
#     # Jacobian: (|x|+1)^(lambda-1) * sign 보정
#     ll <- ll + (lambda - 1) * sum(sign(x) * log(abs(x) + 1))
#     -ll
#   }
#   opt <- optimize(neg_ll, interval = c(-3, 3))
#   opt$minimum
# }
# 
# yj_lambdas <- numeric(ncol(Y_con_full))
# names(yj_lambdas) <- colnames(Y_con_full)
# 
# for (j in seq_len(ncol(Y_con_full))) {
#   x <- as.numeric(Y_con_full[, j])
#   valid <- which(!is.na(x))
#   if (length(valid) < 10) next
#   xv <- x[valid]
#   # lambda 추정 (직접 MLE)
#   lam <- yj_mle_lambda(xv)
#   yj_lambdas[j] <- lam
#   # Yeo-Johnson 변환 적용
#   Y_con_full[valid, j] <- yj_transform(xv, lam)
#   cat(sprintf("  [%d] %s: n=%d, lambda=%.4f\n",
#               j, colnames(Y_con_full)[j], length(xv), lam))
# }
# 
# cat("\n=== Yeo-Johnson lambda estimates ===\n")
# print(round(yj_lambdas, 4))
# 
# # 변환 후 히스토그램 확인
# par(mfrow = c(ceiling(ncol(Y_con_full) / 4), 4), mar = c(3, 3, 2, 1))
# for (j in seq_len(ncol(Y_con_full))) {
#   hist(Y_con_full[, j], breaks = 30,
#        main = sprintf("%s (λ=%.2f)", colnames(Y_con_full)[j], yj_lambdas[j]),
#        xlab = "", col = "steelblue", border = "white")
# }

Y_bin_full  <- lsirm_all$Y_bin
Y_cnt_full  <- lsirm_all$Y_cnt
Y_ord1_full <- lsirm_all$Y_ord1

E  <- make_empty(n_all)           # reusable empty matrix
Eo <- matrix(0L, nrow=n_all, ncol=0)  # empty integer matrix for ord

cases <- list(
  list(
    name = "case1_all",
    label = "Case 1: All (bin+con+cnt+ord)",
    Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = Y_cnt_full, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = lsirm_all$col_con,
    col_cnt = lsirm_all$col_cnt, col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  # --- 2-type combinations ---
  list(
    name = "case2_con_bin",
    label = "Case 2: Continuous + Binary",
    Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = E, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = lsirm_all$col_con,
    col_cnt = character(0), col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case3_con_ord",
    label = "Case 3: Continuous + Ordinal",
    Y_bin = E, Y_con = Y_con_full, Y_cnt = E, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = character(0), col_con = lsirm_all$col_con,
    col_cnt = character(0), col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  list(
    name = "case4_bin_ord",
    label = "Case 4: Binary + Ordinal",
    Y_bin = Y_bin_full, Y_con = E, Y_cnt = E, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = character(0),
    col_cnt = character(0), col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  list(
    name = "case5_cnt_con",
    label = "Case 5: Count + Continuous",
    Y_bin = E, Y_con = Y_con_full, Y_cnt = Y_cnt_full, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = character(0), col_con = lsirm_all$col_con,
    col_cnt = lsirm_all$col_cnt, col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case6_cnt_bin",
    label = "Case 6: Count + Binary",
    Y_bin = Y_bin_full, Y_con = E, Y_cnt = Y_cnt_full, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = character(0),
    col_cnt = lsirm_all$col_cnt, col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case7_cnt_ord",
    label = "Case 7: Count + Ordinal",
    Y_bin = E, Y_con = E, Y_cnt = Y_cnt_full, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = character(0), col_con = character(0),
    col_cnt = lsirm_all$col_cnt, col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  # --- 3-type combinations ---
  list(
    name = "case8_cnt_con_bin",
    label = "Case 8: Count + Continuous + Binary",
    Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = Y_cnt_full, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = lsirm_all$col_con,
    col_cnt = lsirm_all$col_cnt, col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case9_cnt_con_ord",
    label = "Case 9: Count + Continuous + Ordinal",
    Y_bin = E, Y_con = Y_con_full, Y_cnt = Y_cnt_full, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = character(0), col_con = lsirm_all$col_con,
    col_cnt = lsirm_all$col_cnt, col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  list(
    name = "case10_con_bin_ord",
    label = "Case 10: Continuous + Binary + Ordinal",
    Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = E, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = lsirm_all$col_con,
    col_cnt = character(0), col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  list(
    name = "case11_cnt_bin_ord",
    label = "Case 11: Count + Binary + Ordinal",
    Y_bin = Y_bin_full, Y_con = E, Y_cnt = Y_cnt_full, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = character(0),
    col_cnt = lsirm_all$col_cnt, col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  ),
  # --- 1-type (single layer) ---
  list(
    name = "case12_bin_only",
    label = "Case 12: Binary only",
    Y_bin = Y_bin_full, Y_con = E, Y_cnt = E, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = lsirm_all$col_bin, col_con = character(0),
    col_cnt = character(0), col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case13_con_only",
    label = "Case 13: Continuous only",
    Y_bin = E, Y_con = Y_con_full, Y_cnt = E, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = character(0), col_con = lsirm_all$col_con,
    col_cnt = character(0), col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case14_cnt_only",
    label = "Case 14: Count only",
    Y_bin = E, Y_con = E, Y_cnt = Y_cnt_full, Y_ord1 = Eo, Y_ord2 = E,
    col_bin = character(0), col_con = character(0),
    col_cnt = lsirm_all$col_cnt, col_ord1 = character(0), col_ord2 = character(0)
  ),
  list(
    name = "case15_ord_only",
    label = "Case 15: Ordinal only",
    Y_bin = E, Y_con = E, Y_cnt = E, Y_ord1 = Y_ord1_full, Y_ord2 = E,
    col_bin = character(0), col_con = character(0),
    col_cnt = character(0), col_ord1 = lsirm_all$col_ord1, col_ord2 = character(0)
  )
)


################################################################################
# 4. Traceplot helper (V6: z, a1-a5 hierarchical)
################################################################################
make_traceplots <- function(result, prefix, lsirm_data, plot_dir) {

  res <- list()
  if (is.null(result$samples)) res$samples <- result else res$samples <- result$samples

  leg_no_true <- list(
    x = "topright",
    legend = c("Posterior mean", "95% credible interval"),
    col = c("darkgreen", "blue"), lwd = 2, lty = c(1, 3), bty = "n", cex = 0.8
  )

  plot_trace_vec <- function(samples_mat, name = "param", mfrow = c(2,2), leg = leg_no_true) {
    samples_mat <- as.matrix(samples_mat)
    par(mfrow = mfrow)
    for(i in seq_len(ncol(samples_mat))) {
      x <- samples_mat[, i]
      q <- quantile(x, c(.025, .975), na.rm = TRUE)
      ts.plot(x, main = sprintf("%s_%d", name, i))
      abline(h = c(mean(x, na.rm=TRUE), q), col = c("darkgreen","blue","blue"), lwd = 2, lty = c(1,3,3))
      do.call(legend, leg)
    }
  }

  # z (global latent positions)
  if (has_valid(res$samples$z) && dim(res$samples$z)[2] > 0) {
    n_show <- dim(res$samples$z)[2]
    pdf(file.path(plot_dir, paste0(prefix, "_trace_z.pdf")), width = 8, height = 12)
    par(mfrow = c(4,2), mar = c(3,3,2,1))
    for (i in 1:n_show) {
      for (j in 1:dim(res$samples$z)[3]) {
        ts.plot(res$samples$z[, i, j], main = paste0("z_global: ", i, "_", j))
      }
    }
    dev.off()
  }

  # a1-a5 (layer-specific local positions)
  layer_names <- c("a1_bin", "a2_con", "a3_cnt", "a4_ord1", "a5_ord2")
  for (al in 1:5) {
    aname <- paste0("a", al)
    if (has_valid(res$samples[[aname]]) && dim(res$samples[[aname]])[2] > 0) {
      n_show <- dim(res$samples[[aname]])[2]
      pdf(file.path(plot_dir, paste0(prefix, "_trace_", aname, "_pos.pdf")), width = 8, height = 12)
      par(mfrow = c(4,2), mar = c(3,3,2,1))
      for (i in 1:n_show) {
        for (j in 1:dim(res$samples[[aname]])[3]) {
          ts.plot(res$samples[[aname]][, i, j], main = paste0(layer_names[al], ": ", i, "_", j))
        }
      }
      dev.off()
    }
  }

  # b1 (binary item positions)
  if (has_valid(res$samples$b1) && dim(res$samples$b1)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b1.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b1)[2]) {
      for (j in 1:dim(res$samples$b1)[3]) {
        ts.plot(res$samples$b1[, i, j], main = paste0("b1_bin: ", lsirm_data$col_bin[i], "_d", j))
      }
    }
    dev.off()
  }

  # b2 (continuous item positions)
  if (has_valid(res$samples$b2) && dim(res$samples$b2)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b2.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b2)[2]) {
      for (j in 1:dim(res$samples$b2)[3]) {
        ts.plot(res$samples$b2[, i, j], main = paste0("b2_con: ", lsirm_data$col_con[i], "_d", j))
      }
    }
    dev.off()
  }

  # b3 (count item positions)
  if (has_valid(res$samples$b3) && dim(res$samples$b3)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b3.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b3)[2]) {
      for (j in 1:dim(res$samples$b3)[3]) {
        ts.plot(res$samples$b3[, i, j], main = paste0("b3_cnt: ", lsirm_data$col_cnt[i], "_d", j))
      }
    }
    dev.off()
  }

  # b4 (ordinal-1 item positions)
  if (has_valid(res$samples$b4) && dim(res$samples$b4)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b4.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b4)[2]) {
      for (j in 1:dim(res$samples$b4)[3]) {
        ts.plot(res$samples$b4[, i, j], main = paste0("b4_ord1: ", lsirm_data$col_ord1[i], "_d", j))
      }
    }
    dev.off()
  }

  # b5 (ordinal-2 item positions)
  if (has_valid(res$samples$b5) && dim(res$samples$b5)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b5.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b5)[2]) {
      for (j in 1:dim(res$samples$b5)[3]) {
        ts.plot(res$samples$b5[, i, j], main = paste0("b5_ord2: ", lsirm_data$col_ord2[i], "_d", j))
      }
    }
    dev.off()
  }

  # alpha (layer-specific)
  for (al in 1:5) {
    aname <- paste0("alpha", al)
    if (has_valid(res$samples[[aname]]) && ncol(res$samples[[aname]]) > 0) {
      pdf(file.path(plot_dir, paste0(prefix, "_trace_", aname, ".pdf")), width = 8, height = 12)
      plot_trace_vec(res$samples[[aname]][, 1:ncol(res$samples[[aname]])], name = aname, mfrow = c(3,2))
      dev.off()
    }
  }

  # beta1
  if (has_valid(res$samples$beta1) && ncol(res$samples$beta1) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta1.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta1, name = "beta1", mfrow = c(3,2))
    dev.off()
  }

  # beta2
  if (has_valid(res$samples$beta2) && ncol(res$samples$beta2) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta2.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta2, name = "beta2", mfrow = c(3,2))
    dev.off()
  }

  # beta3
  if (has_valid(res$samples$beta3) && ncol(res$samples$beta3) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta3.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta3, name = "beta3", mfrow = c(3,2))
    dev.off()
  }

  # beta4 (GRM thresholds L4)
  if (has_valid(res$samples$beta4)) {
    b4_s <- res$samples$beta4
    P4_d <- dim(b4_s)[2]
    Km1  <- dim(b4_s)[3]
    col_ord1 <- if (length(lsirm_data$col_ord1) > 0) lsirm_data$col_ord1 else paste0("ord1_j", 1:P4_d)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta4_thr.pdf")), width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P4_d) {
      for (k in 1:Km1) {
        x <- b4_s[, j, k]
        q <- quantile(x, c(.025, .975), na.rm = TRUE)
        ts.plot(x, main = sprintf("beta4[%s, k=%d]", col_ord1[j], k))
        abline(h = c(mean(x, na.rm = TRUE), q),
               col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
      }
    }
    dev.off()
  }

  # beta5 (GRM thresholds L5)
  if (has_valid(res$samples$beta5)) {
    b5_s <- res$samples$beta5
    P5_d <- dim(b5_s)[2]
    Km1  <- dim(b5_s)[3]
    col_ord2 <- if (length(lsirm_data$col_ord2) > 0) lsirm_data$col_ord2 else paste0("ord2_j", 1:P5_d)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta5_thr.pdf")), width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P5_d) {
      for (k in 1:Km1) {
        x <- b5_s[, j, k]
        q <- quantile(x, c(.025, .975), na.rm = TRUE)
        ts.plot(x, main = sprintf("beta5[%s, k=%d]", col_ord2[j], k))
        abline(h = c(mean(x, na.rm = TRUE), q),
               col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
      }
    }
    dev.off()
  }

  # Extra scalar parameters (including sigma1_sq)
  pdf(file.path(plot_dir, paste0(prefix, "_trace_extra.pdf")), width = 8, height = 20)
  par(mfrow = c(8,2), mar = c(3,3,2,1))
  if (has_valid(res$samples$sigma0_sq))  plot_trace_scalar(res$samples$sigma0_sq,  true = NA, main = "sigma0_sq")
  if (has_valid(res$samples$sigma1_sq))  plot_trace_scalar(res$samples$sigma1_sq,  true = NA, main = "sigma1_sq (global-local coupling)")
  if (has_valid(res$samples$log_gamma1)) plot_trace_scalar(res$samples$log_gamma1, true = NA, main = "gamma1 (Bin)", transform = exp)
  if (has_valid(res$samples$log_gamma2)) plot_trace_scalar(res$samples$log_gamma2, true = NA, main = "gamma2 (Con)", transform = exp)
  if (has_valid(res$samples$log_gamma3)) plot_trace_scalar(res$samples$log_gamma3, true = NA, main = "gamma3 (Cnt)", transform = exp)
  if (has_valid(res$samples$log_gamma4)) plot_trace_scalar(res$samples$log_gamma4, true = NA, main = "gamma4 (Ord1)", transform = exp)
  if (has_valid(res$samples$log_gamma5)) plot_trace_scalar(res$samples$log_gamma5, true = NA, main = "gamma5 (Ord2)", transform = exp)
  if (has_valid(res$samples$log_kappa))  plot_trace_scalar(res$samples$log_kappa,  true = NA, main = "kappa", transform = exp)
  for (al in 1:5) {
    sname <- paste0("sigma_alpha", al, "_sq")
    if (has_valid(res$samples[[sname]])) plot_trace_scalar(res$samples[[sname]], true = NA, main = sname)
  }
  if (has_valid(res$samples$lambda2_mean)) {
    plot_trace_scalar(res$samples$lambda2_mean, true = NA, main = "lambda2_mean (robust weight)")
  }
  dev.off()

  # --- Lambda2 per-edge traceplots ---
  if (has_valid(res$samples$lambda2) && length(dim(res$samples$lambda2)) == 3 && dim(res$samples$lambda2)[3] > 0) {
    lam <- res$samples$lambda2
    n_resp <- dim(lam)[2]
    n_item <- dim(lam)[3]

    n_edge_show <- min(12, n_resp * n_item)
    set.seed(42)
    all_edges <- expand.grid(i = 1:n_resp, j = 1:n_item)
    edge_idx  <- all_edges[sort(sample(nrow(all_edges), n_edge_show)), ]

    pdf(file.path(plot_dir, paste0(prefix, "_trace_lambda2_edges.pdf")), width = 10, height = 12)
    par(mfrow = c(4, 3), mar = c(3, 3, 2, 1))
    for (r in seq_len(nrow(edge_idx))) {
      ii <- edge_idx$i[r]; jj <- edge_idx$j[r]
      x <- lam[, ii, jj]
      jname <- if (length(lsirm_data$col_con) >= jj) lsirm_data$col_con[jj] else jj
      ts.plot(x, main = bquote(lambda[2] ~ "(" * .(ii) * "," * .(jname) * ")"))
      abline(h = mean(x), col = "darkgreen", lwd = 2)
      abline(h = 1, col = "red", lty = 2)
    }
    dev.off()
  }

  # --- Lambda2 posterior mean heatmap ---
  if (has_valid(res$samples$lambda2_postmean) && ncol(res$samples$lambda2_postmean) > 0) {
    lam_pm <- res$samples$lambda2_postmean
    col_names <- if (length(lsirm_data$col_con) > 0) lsirm_data$col_con else paste0("j", 1:ncol(lam_pm))

    pdf(file.path(plot_dir, paste0(prefix, "_lambda2_postmean_heatmap.pdf")), width = 10, height = 8)
    col_pal <- colorRampPalette(c("red", "white", "steelblue"))(100)
    lam_clipped <- pmin(lam_pm, quantile(lam_pm, 0.99))
    image(1:nrow(lam_pm), 1:ncol(lam_pm), lam_clipped,
          col = col_pal, xlab = "Respondent (i)", ylab = "Item (j)",
          main = expression("Posterior Mean of " * lambda[ij]^{(2)}),
          axes = FALSE)
    axis(1, at = seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm))),
         labels = seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm))))
    axis(2, at = 1:ncol(lam_pm), labels = col_names, las = 2, cex.axis = 0.7)
    box()
    dev.off()

    pdf(file.path(plot_dir, paste0(prefix, "_lambda2_postmean_boxplot.pdf")), width = 10, height = 6)
    colnames(lam_pm) <- col_names
    boxplot(as.data.frame(lam_pm), las = 2, cex.axis = 0.7,
            main = expression("Posterior Mean " * lambda[ij]^{(2)} * " by Item"),
            ylab = expression(lambda[ij]^{(2)}), col = "lightyellow", outline = TRUE)
    abline(h = 1, col = "red", lty = 2)
    dev.off()

    cat("\n── Lambda2 Posterior Mean Summary ──\n")
    cat(sprintf("  Overall mean: %.3f\n", mean(lam_pm)))
    cat(sprintf("  Min: %.3f | Q1: %.3f | Median: %.3f | Q3: %.3f | Max: %.3f\n",
                min(lam_pm), quantile(lam_pm, 0.25), median(lam_pm),
                quantile(lam_pm, 0.75), max(lam_pm)))
    n_outlier <- sum(lam_pm < 0.5)
    cat(sprintf("  Edges with lambda < 0.5 (potential outliers): %d / %d (%.1f%%)\n",
                n_outlier, length(lam_pm), 100 * n_outlier / length(lam_pm)))
  }

  # Gamma comparison: independent per layer
  gamma_names <- c("Bin", "Con", "Cnt", "Ord1", "Ord2")
  gamma_cols  <- c("red", "blue", "darkgreen", "purple", "deeppink")
  gamma_list  <- list(res$samples$log_gamma1, res$samples$log_gamma2,
                      res$samples$log_gamma3, res$samples$log_gamma4,
                      res$samples$log_gamma5)
  gamma_exist <- sapply(gamma_list, has_valid)

  if (sum(gamma_exist) >= 1) {
    gamma_val_list <- lapply(gamma_list[gamma_exist], exp)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_gamma_compare.pdf")), width = 10, height = 8)
    par(mfrow = c(2,1), mar = c(4,4,3,1))

    yr <- range(unlist(gamma_val_list))
    base::plot(gamma_val_list[[1]], type = "l", col = gamma_cols[gamma_exist][1],
         ylim = yr, xlab = "Iteration", ylab = expression(gamma[l]),
         main = "Layer-Specific Gamma Traceplots")
    if (length(gamma_val_list) > 1) {
      for (k in seq_along(gamma_val_list)[-1]) {
        lines(gamma_val_list[[k]], col = gamma_cols[gamma_exist][k])
      }
    }
    legend("topright", legend = gamma_names[gamma_exist],
           col = gamma_cols[gamma_exist], lwd = 1, bty = "n", cex = 0.9)

    dens_list <- lapply(gamma_val_list, density)
    xr <- range(unlist(lapply(dens_list, function(d) d$x)))
    yr <- range(unlist(lapply(dens_list, function(d) d$y)))
    base::plot(dens_list[[1]], col = gamma_cols[gamma_exist][1], lwd = 2,
         xlim = xr, ylim = yr, xlab = expression(gamma[l]), ylab = "Density",
         main = "Posterior Density: Layer-Specific Gamma")
    if (length(dens_list) > 1) {
      for (k in seq_along(dens_list)[-1]) {
        lines(dens_list[[k]], col = gamma_cols[gamma_exist][k], lwd = 2)
      }
    }
    for (k in seq_along(gamma_val_list)) {
      abline(v = mean(gamma_val_list[[k]]), col = gamma_cols[gamma_exist][k], lty = 3)
    }
    legend("topright", legend = paste0(gamma_names[gamma_exist], " (mean=",
           sprintf("%.3f", sapply(gamma_val_list, mean)), ")"),
           col = gamma_cols[gamma_exist], lwd = 2, bty = "n", cex = 0.9)

    dev.off()
  }
}


################################################################################
# 5. Biplot helper (V6: z as global, a1-a5 as local, b1-b5 as items)
################################################################################
make_biplot <- function(result, lsirm_data, title, filename, plot_dir) {

  res <- if (is.null(result$samples)) result else result$samples

  # Global respondent position (z)
  Z_hat <- apply(res$z, c(2,3), mean)

  has_bin  <- length(lsirm_data$col_bin)  > 0 && has_valid(res$b1) && dim(res$b1)[2] > 0
  has_con  <- length(lsirm_data$col_con)  > 0 && has_valid(res$b2) && dim(res$b2)[2] > 0
  has_cnt  <- length(lsirm_data$col_cnt)  > 0 && has_valid(res$b3) && dim(res$b3)[2] > 0
  has_ord1 <- length(lsirm_data$col_ord1) > 0 && has_valid(res$b4) && dim(res$b4)[2] > 0
  has_ord2 <- length(lsirm_data$col_ord2) > 0 && has_valid(res$b5) && dim(res$b5)[2] > 0

  B1_hat <- if (has_bin)  apply(res$b1, c(2,3), mean) else NULL
  B2_hat <- if (has_con)  apply(res$b2, c(2,3), mean) else NULL
  B3_hat <- if (has_cnt)  apply(res$b3, c(2,3), mean) else NULL
  B4_hat <- if (has_ord1) apply(res$b4, c(2,3), mean) else NULL
  B5_hat <- if (has_ord2) apply(res$b5, c(2,3), mean) else NULL

  all_pts <- rbind(Z_hat, B1_hat, B2_hat, B3_hat, B4_hat, B5_hat)
  expand <- 0.08
  xr <- range(all_pts[,1], na.rm = TRUE)
  yr <- range(all_pts[,2], na.rm = TRUE)
  dx <- diff(xr); dy <- diff(yr)
  xlim <- xr + c(-1,1) * expand * dx
  ylim <- yr + c(-1,1) * expand * dy
  if (dx == 0) xlim <- xr + c(-1,1)
  if (dy == 0) ylim <- yr + c(-1,1)

  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  base::plot(Z_hat, pch = 21, col = "black", bg = "gray80", cex = 0.8,
       xlab = "Dim1", ylab = "Dim2", main = title,
       xlim = xlim, ylim = ylim)

  if (has_bin) {
    points(B1_hat, pch = 21, bg = "forestgreen", col = "forestgreen", cex = 1.2)
    text(B1_hat, labels = lsirm_data$col_bin, cex = 0.7, pos = 4, col = "darkgreen")
  }
  if (has_con) {
    points(B2_hat, pch = 21, bg = "orange", col = "orange", cex = 1.2)
    text(B2_hat, labels = lsirm_data$col_con, cex = 0.7, pos = 4, col = "orange4")
  }
  if (has_cnt) {
    points(B3_hat, pch = 21, bg = "cyan3", col = "cyan3", cex = 1.2)
    text(B3_hat, labels = lsirm_data$col_cnt, cex = 0.7, pos = 4, col = "cyan4")
  }
  if (has_ord1) {
    points(B4_hat, pch = 21, bg = "purple", col = "purple", cex = 1.2)
    text(B4_hat, labels = lsirm_data$col_ord1, cex = 0.7, pos = 4, col = "purple4")
  }
  if (has_ord2) {
    points(B5_hat, pch = 21, bg = "deeppink", col = "deeppink", cex = 1.2)
    text(B5_hat, labels = lsirm_data$col_ord2, cex = 0.7, pos = 4, col = "deeppink4")
  }

  resp_legend <- "Respondent (z: global)"
  resp_bg  <- "gray80"
  resp_pch <- 21

  item_legend <- c(
    if(has_bin)  "Binary (CESD)" else NULL,
    if(has_con)  "Continuous (bio/cog)" else NULL,
    if(has_cnt)  "Count (sleep/drink/cog)" else NULL,
    if(has_ord1) "Ordinal-5 (MASQ)" else NULL,
    if(has_ord2) "Ordinal-4 (PSQI)" else NULL
  )
  item_bg <- c(
    if(has_bin)  "forestgreen" else NULL,
    if(has_con)  "orange" else NULL,
    if(has_cnt)  "cyan3" else NULL,
    if(has_ord1) "purple" else NULL,
    if(has_ord2) "deeppink" else NULL
  )

  legend("topright",
         legend = c(resp_legend, item_legend),
         pch = c(resp_pch, rep(21, length(item_legend))),
         pt.bg = c(resp_bg, item_bg),
         bty = "n", cex = 0.8)

  dev.off()
}


################################################################################
# 5b. 3D Biplot helper (requires rgl) - V6: uses z (global)
################################################################################
make_biplot_3d <- function(result, lsirm_data, title, filename, plot_dir) {

  if (!requireNamespace("rgl", quietly = TRUE)) {
    cat("  [SKIP] rgl package not installed -skipping 3D biplot\n")
    return(invisible(NULL))
  }
  library(rgl)

  res <- if (is.null(result$samples)) result else result$samples

  Z_hat <- apply(res$z, c(2,3), mean)
  d <- ncol(Z_hat)
  if (d < 3) {
    cat("  [SKIP] d < 3 -skipping 3D biplot\n")
    return(invisible(NULL))
  }

  has_bin  <- length(lsirm_data$col_bin)  > 0 && has_valid(res$b1) && dim(res$b1)[2] > 0
  has_con  <- length(lsirm_data$col_con)  > 0 && has_valid(res$b2) && dim(res$b2)[2] > 0
  has_cnt  <- length(lsirm_data$col_cnt)  > 0 && has_valid(res$b3) && dim(res$b3)[2] > 0
  has_ord1 <- length(lsirm_data$col_ord1) > 0 && has_valid(res$b4) && dim(res$b4)[2] > 0
  has_ord2 <- length(lsirm_data$col_ord2) > 0 && has_valid(res$b5) && dim(res$b5)[2] > 0

  B1_hat <- if (has_bin)  apply(res$b1, c(2,3), mean) else NULL
  B2_hat <- if (has_con)  apply(res$b2, c(2,3), mean) else NULL
  B3_hat <- if (has_cnt)  apply(res$b3, c(2,3), mean) else NULL
  B4_hat <- if (has_ord1) apply(res$b4, c(2,3), mean) else NULL
  B5_hat <- if (has_ord2) apply(res$b5, c(2,3), mean) else NULL

  # --- Interactive 3D plot ---
  open3d()
  par3d(windowRect = c(50, 50, 850, 650))

  # Respondents (global z)
  plot3d(Z_hat[,1], Z_hat[,2], Z_hat[,3],
         xlab = "Dim1", ylab = "Dim2", zlab = "Dim3",
         col = "gray70", size = 3, type = "s",
         main = title)

  if (has_bin) {
    spheres3d(B1_hat[,1], B1_hat[,2], B1_hat[,3], radius = 0.08, col = "forestgreen")
    text3d(B1_hat[,1], B1_hat[,2], B1_hat[,3], texts = lsirm_data$col_bin,
           adj = c(-0.2, 0.5), cex = 0.7, col = "darkgreen")
  }
  if (has_con) {
    spheres3d(B2_hat[,1], B2_hat[,2], B2_hat[,3], radius = 0.08, col = "orange")
    text3d(B2_hat[,1], B2_hat[,2], B2_hat[,3], texts = lsirm_data$col_con,
           adj = c(-0.2, 0.5), cex = 0.7, col = "orange4")
  }
  if (has_cnt) {
    spheres3d(B3_hat[,1], B3_hat[,2], B3_hat[,3], radius = 0.08, col = "cyan3")
    text3d(B3_hat[,1], B3_hat[,2], B3_hat[,3], texts = lsirm_data$col_cnt,
           adj = c(-0.2, 0.5), cex = 0.7, col = "cyan4")
  }
  if (has_ord1) {
    spheres3d(B4_hat[,1], B4_hat[,2], B4_hat[,3], radius = 0.08, col = "purple")
    text3d(B4_hat[,1], B4_hat[,2], B4_hat[,3], texts = lsirm_data$col_ord1,
           adj = c(-0.2, 0.5), cex = 0.7, col = "purple4")
  }
  if (has_ord2) {
    spheres3d(B5_hat[,1], B5_hat[,2], B5_hat[,3], radius = 0.08, col = "deeppink")
    text3d(B5_hat[,1], B5_hat[,2], B5_hat[,3], texts = lsirm_data$col_ord2,
           adj = c(-0.2, 0.5), cex = 0.7, col = "deeppink4")
  }

  legend3d("topright",
           legend = c("Respondent (z: global)",
                      if(has_bin)  "Binary (CESD)" else NULL,
                      if(has_con)  "Continuous (bio/cog)" else NULL,
                      if(has_cnt)  "Count (sleep/drink/cog)" else NULL,
                      if(has_ord1) "Ordinal-5 (MASQ)" else NULL,
                      if(has_ord2) "Ordinal-4 (PSQI)" else NULL),
           pch = 16,
           col = c("gray70",
                   if(has_bin)  "forestgreen" else NULL,
                   if(has_con)  "orange" else NULL,
                   if(has_cnt)  "cyan3" else NULL,
                   if(has_ord1) "purple" else NULL,
                   if(has_ord2) "deeppink" else NULL),
           bty = "n", cex = 0.9)

  # Save as PNG snapshot
  rgl.snapshot(file.path(plot_dir, filename), fmt = "png")
  close3d()

  # --- Also save 2D pairwise projections as PDF ---
  pdf_file <- sub("\\.[^.]+$", "_pairs.pdf", file.path(plot_dir, filename))
  pdf(pdf_file, width = 12, height = 4.5)
  par(mfrow = c(1,3), mar = c(4,4,3,1))

  dim_pairs <- list(c(1,2), c(1,3), c(2,3))
  dim_labels <- c("Dim1", "Dim2", "Dim3")

  for (dp in dim_pairs) {
    d1 <- dp[1]; d2 <- dp[2]
    all_pts <- rbind(Z_hat[, c(d1,d2)], B1_hat[, c(d1,d2)], B2_hat[, c(d1,d2)],
                     B3_hat[, c(d1,d2)], B4_hat[, c(d1,d2)], B5_hat[, c(d1,d2)])
    xr <- range(all_pts[,1], na.rm = TRUE); yr <- range(all_pts[,2], na.rm = TRUE)
    expand <- 0.08
    xlim <- xr + c(-1,1) * expand * diff(xr)
    ylim <- yr + c(-1,1) * expand * diff(yr)
    if (diff(xr) == 0) xlim <- xr + c(-1,1)
    if (diff(yr) == 0) ylim <- yr + c(-1,1)

    base::plot(Z_hat[,d1], Z_hat[,d2], pch = 21, bg = "gray80", col = "black", cex = 0.8,
         xlab = dim_labels[d1], ylab = dim_labels[d2],
         main = paste0(dim_labels[d1], " vs ", dim_labels[d2]),
         xlim = xlim, ylim = ylim)

    if (has_bin) {
      points(B1_hat[,d1], B1_hat[,d2], pch = 21, bg = "forestgreen", col = "forestgreen", cex = 1.2)
      text(B1_hat[,d1], B1_hat[,d2], labels = lsirm_data$col_bin, cex = 0.6, pos = 4, col = "darkgreen")
    }
    if (has_con) {
      points(B2_hat[,d1], B2_hat[,d2], pch = 21, bg = "orange", col = "orange", cex = 1.2)
      text(B2_hat[,d1], B2_hat[,d2], labels = lsirm_data$col_con, cex = 0.6, pos = 4, col = "orange4")
    }
    if (has_cnt) {
      points(B3_hat[,d1], B3_hat[,d2], pch = 21, bg = "cyan3", col = "cyan3", cex = 1.2)
      text(B3_hat[,d1], B3_hat[,d2], labels = lsirm_data$col_cnt, cex = 0.6, pos = 4, col = "cyan4")
    }
    if (has_ord1) {
      points(B4_hat[,d1], B4_hat[,d2], pch = 21, bg = "purple", col = "purple", cex = 1.2)
      text(B4_hat[,d1], B4_hat[,d2], labels = lsirm_data$col_ord1, cex = 0.6, pos = 4, col = "purple4")
    }
    if (has_ord2) {
      points(B5_hat[,d1], B5_hat[,d2], pch = 21, bg = "deeppink", col = "deeppink", cex = 1.2)
      text(B5_hat[,d1], B5_hat[,d2], labels = lsirm_data$col_ord2, cex = 0.6, pos = 4, col = "deeppink4")
    }
  }
  dev.off()
}

################################################################################
# 5c. Respondent position scatter plots (global z + local a1-a5) with labels
################################################################################
make_respondent_scatter <- function(result, title, filename, plot_dir) {

  res <- if (is.null(result$samples)) result else result$samples

  Z_hat  <- apply(res$z,  c(2,3), mean)  # n x d
  A1_hat <- apply(res$a1, c(2,3), mean)
  A2_hat <- apply(res$a2, c(2,3), mean)
  A3_hat <- apply(res$a3, c(2,3), mean)
  A4_hat <- apply(res$a4, c(2,3), mean)
  A5_hat <- apply(res$a5, c(2,3), mean)

  n <- nrow(Z_hat)
  labels <- as.character(1:n)

  layer_names <- c("Global (z)", "L1: Binary", "L2: Continuous",
                   "L3: Count", "L4: Ordinal-5", "L5: Ordinal-4")
  layer_mats  <- list(Z_hat, A1_hat, A2_hat, A3_hat, A4_hat, A5_hat)
  layer_cols   <- c("black", "forestgreen", "orange", "cyan4", "purple", "deeppink")
  layer_bgs    <- c("gray80", "lightgreen", "lightyellow", "lightskyblue", "plum", "lightpink")

  # --- (a) All-in-one overlay plot ---
  all_pts <- do.call(rbind, layer_mats)
  xr <- range(all_pts[,1], na.rm = TRUE); yr <- range(all_pts[,2], na.rm = TRUE)
  expand <- 0.1
  xlim <- xr + c(-1,1) * expand * max(diff(xr), 0.1)
  ylim <- yr + c(-1,1) * expand * max(diff(yr), 0.1)

  pdf(file.path(plot_dir, filename), width = 12, height = 10)
  base::plot(NULL, xlim = xlim, ylim = ylim,
       xlab = "Dim1", ylab = "Dim2",
       main = paste0(title, " - All Respondent Positions"))

  for (l in seq_along(layer_mats)) {
    points(layer_mats[[l]], pch = 21, bg = layer_bgs[l], col = layer_cols[l], cex = 0.7)
  }
  # Label only global z to avoid clutter
  text(Z_hat, labels = labels, cex = 0.5, pos = 3, col = "black")

  legend("topright", legend = layer_names, pch = 21, pt.bg = layer_bgs,
         col = layer_cols, bty = "n", cex = 0.8)
  dev.off()

  # --- (b) 6-panel: one panel per layer ---
  pdf_panel <- sub("\\.pdf$", "_panels.pdf", file.path(plot_dir, filename))
  pdf(pdf_panel, width = 15, height = 10)
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

  for (l in seq_along(layer_mats)) {
    mat <- layer_mats[[l]]
    base::plot(mat, pch = 21, bg = layer_bgs[l], col = layer_cols[l], cex = 0.8,
         xlab = "Dim1", ylab = "Dim2", main = layer_names[l],
         xlim = xlim, ylim = ylim)
    text(mat, labels = labels, cex = 0.45, pos = 3, col = layer_cols[l])
  }
  dev.off()
}


################################################################################
# 5d. Global-local distance histograms (||z_i - a_l_i|| per layer)
################################################################################
make_distance_histogram <- function(result, title, filename, plot_dir) {

  res <- if (is.null(result$samples)) result else result$samples

  Z_hat  <- apply(res$z,  c(2,3), mean)  # n x d
  A1_hat <- apply(res$a1, c(2,3), mean)
  A2_hat <- apply(res$a2, c(2,3), mean)
  A3_hat <- apply(res$a3, c(2,3), mean)
  A4_hat <- apply(res$a4, c(2,3), mean)
  A5_hat <- apply(res$a5, c(2,3), mean)

  # Euclidean distance ||z_i - a_l_i|| for each respondent
  euc_dist <- function(A, B) sqrt(rowSums((A - B)^2))

  dist1 <- euc_dist(Z_hat, A1_hat)
  dist2 <- euc_dist(Z_hat, A2_hat)
  dist3 <- euc_dist(Z_hat, A3_hat)
  dist4 <- euc_dist(Z_hat, A4_hat)
  dist5 <- euc_dist(Z_hat, A5_hat)

  layer_names <- c("L1: Binary", "L2: Continuous", "L3: Count",
                   "L4: Ordinal-5", "L5: Ordinal-4")
  dist_list <- list(dist1, dist2, dist3, dist4, dist5)
  layer_cols <- c("forestgreen", "orange", "cyan4", "purple", "deeppink")

  # --- (a) 5-panel histograms ---
  pdf(file.path(plot_dir, filename), width = 15, height = 6)
  par(mfrow = c(1, 5), mar = c(4, 4, 3, 1))

  x_max <- max(unlist(dist_list), na.rm = TRUE) * 1.05
  breaks_common <- seq(0, ceiling(x_max * 10) / 10, length.out = 25)

  for (l in 1:5) {
    hist(dist_list[[l]], breaks = breaks_common, col = layer_cols[l], border = "white",
         main = layer_names[l],
         xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
         ylab = "Frequency", xlim = c(0, x_max))
    abline(v = mean(dist_list[[l]]), col = "red", lwd = 2, lty = 2)
    legend("topright",
           legend = sprintf("mean=%.3f\nsd=%.3f", mean(dist_list[[l]]), sd(dist_list[[l]])),
           bty = "n", cex = 0.8)
  }
  dev.off()

  # --- (b) Overlaid density plot ---
  pdf_density <- sub("\\.pdf$", "_density.pdf", file.path(plot_dir, filename))
  pdf(pdf_density, width = 8, height = 6)

  dens_list <- lapply(dist_list, density)
  y_max <- max(sapply(dens_list, function(d) max(d$y)))

  base::plot(NULL, xlim = c(0, x_max), ylim = c(0, y_max * 1.1),
       xlab = expression(paste("||", z[i], " - ", a[i]^(l), "||")),
       ylab = "Density",
       main = paste0(title, " -Global-Local Distance Density"))

  for (l in 1:5) {
    lines(dens_list[[l]], col = layer_cols[l], lwd = 2)
  }
  legend("topright", legend = layer_names, col = layer_cols, lwd = 2, bty = "n", cex = 0.9)
  dev.off()
}


################################################################################
# 6. 순차 실행
################################################################################

################################################################################
# 6-0. Continuous layer variable subsets (switch active_con_group to change)
################################################################################
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
cognition_vars    <- c("B3TCOMPZ3", "B3TEMZ3", "B3TEFZ3", "B3TWLF",
                        "B3TSMXNS", "B3TSMXRS", "B3TSMN", "B3TSMR",
                        "B3TSMXBS", "B3TSMXNO", "B3TSMXRO")

# ★ Switch here: "inflammation" or "cognition"
active_con_group <- "cognition"

if (active_con_group == "inflammation") {
  active_con_vars <- inflammation_vars
} else {
  active_con_vars <- cognition_vars
}

# Subset Y_con to active group
con_all_names <- colnames(Y_con_full)
active_idx <- which(con_all_names %in% active_con_vars)
if (length(active_idx) == 0) stop("No matching continuous variables found for: ", active_con_group)

Y_con_subset <- Y_con_full[, active_idx, drop = FALSE]
col_con_subset <- con_all_names[active_idx]

cat(sprintf("\n=== Active continuous group: %s (%d vars) ===\n", active_con_group, length(active_idx)))
cat(sprintf("  Variables: %s\n", paste(col_con_subset, collapse = ", ")))

################################################################################
# 6. 순차 실행
################################################################################

for (cs in cases[1]) {
  cs <- cases[[1]]

  # Override Y_con with subset
  cs$Y_con  <- Y_con_subset
  cs$col_con <- col_con_subset

  cat(sprintf("\n\n========== %s [con=%s] ==========\n", cs$label, active_con_group))

  # Case-specific plot directory (include group name)
  case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", active_con_group))
  if (!dir.exists(case_plot_dir)) dir.create(case_plot_dir, recursive = TRUE)

  cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
              nrow(cs$Y_bin), ncol(cs$Y_con),
              nrow(cs$Y_con), ncol(cs$Y_con),
              nrow(cs$Y_cnt), ncol(cs$Y_cnt),
              nrow(cs$Y_ord1), ncol(cs$Y_ord1),
              nrow(cs$Y_ord2), ncol(cs$Y_ord2)))

  result <- lsirm_hierarchical_layer5_grm_cpp(
    cs$Y_bin, round(cs$Y_con, 1), cs$Y_cnt, cs$Y_ord1, cs$Y_ord2,
    d = common_mcmc$d,
    n_iter = common_mcmc$n_iter,
    burnin = common_mcmc$burnin,
    thin   = common_mcmc$thin,
    nu2    = nu2,
    hyper  = common_hyper,
    prop_sd = common_prop_sd,
    init = NULL,
    verbose = TRUE
  )

  cat(sprintf("\n-- %s [con=%s]: Acceptance Rates --\n", cs$name, active_con_group))
  print(result$accept)

  # lsirm_data for plot labels
  lsirm_data <- list(
    col_bin  = cs$col_bin,  col_con  = cs$col_con,
    col_cnt  = cs$col_cnt,  col_ord1 = cs$col_ord1, col_ord2 = cs$col_ord2,
    branch   = lsirm_all$branch
  )

  prefix <- paste0("v6_", cs$name, "_", active_con_group)

  make_traceplots(result, prefix = prefix, lsirm_data = lsirm_data, plot_dir = case_plot_dir)
  make_biplot(result, lsirm_data = lsirm_data,
              title = paste0("MIDUS W2+R1: ", cs$label, " [", active_con_group, "] (v6)"),
              filename = paste0(prefix, "_biplot.pdf"),
              plot_dir = case_plot_dir)
  make_respondent_scatter(result,
                          title = paste0("MIDUS W2+R1: ", cs$label, " [", active_con_group, "]"),
                          filename = paste0(prefix, "_respondent_scatter.pdf"),
                          plot_dir = case_plot_dir)
  make_distance_histogram(result,
                          title = paste0("MIDUS W2+R1: ", cs$label, " [", active_con_group, "]"),
                          filename = paste0(prefix, "_dist_histogram.pdf"),
                          plot_dir = case_plot_dir)

  # Save result object
  result_file <- file.path(case_plot_dir, paste0(cs$name, "_", active_con_group, "_v6_result.rds"))
  saveRDS(result, result_file)
  cat(sprintf("  -> Plots & result saved to: %s\n", case_plot_dir))
}








# 염증지표
# B4BSCL14
# B4BNE12
# B4BIL6
# B4BFGN
# B4BCRP

# 인지
# B3TCOMPZ3
# B3TEMZ3
# B3TEFZ3
# B3TWLF
# B3TSMXNS
# B3TSMXRS
# B3TSMN
# B3TSMR
# B3TSMXBS
# B3TSMXNO
# B3TSMXRO

################################################################################
# Continuous layer variable-group histograms (colMeans, rowMeans removed)
################################################################################

Y_con_raw <- cases[[1]]$Y_con
con_names <- colnames(Y_con_raw)

# Variable groups
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
cognition_vars    <- c("B3TCOMPZ3", "B3TEMZ3", "B3TEFZ3", "B3TWLF",
                        "B3TSMXNS", "B3TSMXRS", "B3TSMN", "B3TSMR",
                        "B3TSMXBS", "B3TSMXNO", "B3TSMXRO")

# Double-centering: remove colMeans and rowMeans
double_center <- function(M) {
  M <- as.matrix(M)
  cm <- colMeans(M, na.rm = TRUE)
  M_c <- sweep(M, 2, cm, "-")       # remove column means
  rm <- rowMeans(M_c, na.rm = TRUE)
  M_dc <- sweep(M_c, 1, rm, "-")    # remove row means
  M_dc
}

# --- Inflammation group ---
inf_idx <- which(con_names %in% inflammation_vars)
if (length(inf_idx) > 0) {
  Y_inf <- double_center(Y_con_raw[, inf_idx, drop = FALSE])

  pdf(file.path(data_dir, "plot", "v6_hist_inflammation_double_centered.pdf"),
      width = 12, height = ceiling(length(inf_idx) / 3) * 3)
  par(mfrow = c(ceiling(length(inf_idx) / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(ncol(Y_inf))) {
    vals <- Y_inf[, j]
    hist(vals[!is.na(vals)], breaks = 30, col = "salmon", border = "white",
         main = paste0(colnames(Y_inf)[j], " (double-centered)"),
         xlab = "Value", ylab = "Frequency")
    abline(v = 0, col = "red", lty = 2)
  }
  dev.off()
  cat(sprintf("Inflammation group: %d variables found in Y_con\n", length(inf_idx)))
} else {
  cat("Inflammation group: no matching variables found in Y_con\n")
}

# --- Cognition group ---
cog_idx <- which(con_names %in% cognition_vars)
if (length(cog_idx) > 0) {
  Y_cog <- double_center(Y_con_raw[, cog_idx, drop = FALSE])

  pdf(file.path(data_dir, "plot", "v6_hist_cognition_double_centered.pdf"),
      width = 12, height = ceiling(length(cog_idx) / 3) * 4)
  par(mfrow = c(ceiling(length(cog_idx) / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(ncol(Y_cog))) {
    vals <- Y_cog[, j]
    hist(vals[!is.na(vals)], breaks = 30, col = "steelblue", border = "white",
         main = paste0(colnames(Y_cog)[j], " (double-centered)"),
         xlab = "Value", ylab = "Frequency")
    abline(v = 0, col = "red", lty = 2)
  }
  dev.off()
  cat(sprintf("Cognition group: %d variables found in Y_con\n", length(cog_idx)))
} else {
  cat("Cognition group: no matching variables found in Y_con\n")
}

# --- Summary ---
cat("\n=== Y_con column names ===\n")
print(con_names)
cat(sprintf("\nInflammation matched: %s\n", paste(con_names[inf_idx], collapse = ", ")))
cat(sprintf("Cognition matched:    %s\n", paste(con_names[cog_idx], collapse = ", ")))

# --- Combined: both groups in one plot ---
both_idx <- c(inf_idx, cog_idx)
if (length(both_idx) > 0) {
  Y_both <- double_center(Y_con_raw[, both_idx, drop = FALSE])
  n_vars <- ncol(Y_both)
  n_inf  <- length(inf_idx)
  n_cog  <- length(cog_idx)

  # Color: salmon for inflammation, steelblue for cognition
  var_cols <- c(rep("salmon", n_inf), rep("steelblue", n_cog))
  group_labels <- c(rep("Inflammation", n_inf), rep("Cognition", n_cog))

  ncol_layout <- 4
  nrow_layout <- ceiling(n_vars / ncol_layout)

  pdf(file.path(data_dir, "plot", "v6_hist_combined_double_centered.pdf"),
      width = 14, height = nrow_layout * 3.5)
  par(mfrow = c(nrow_layout, ncol_layout), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_vars)) {
    vals <- Y_both[, j]
    hist(vals[!is.na(vals)], breaks = 30, col = var_cols[j], border = "white",
         main = paste0(colnames(Y_both)[j], "\n(", group_labels[j], ")"),
         xlab = "Value", ylab = "Frequency", cex.main = 0.9)
    abline(v = 0, col = "red", lty = 2)
  }
  dev.off()
  cat(sprintf("Combined plot: %d inflammation + %d cognition = %d variables\n",
              n_inf, n_cog, n_vars))
}



