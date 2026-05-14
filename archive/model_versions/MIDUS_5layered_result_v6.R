rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. 경로 설정
################################################################################
data_dir <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data"
setwd(data_dir)

################################################################################
# 0-1. 데이터 준비: Wave 2 + Refresher 1 합치기 (v4 preprocess 사용)
################################################################################

# ── Wave 2 전처리 (격리 환경) ──
cat("\n====== Wave 2 전처리 ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_v4.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all
lsirm_p4_w2  <- env_w2$lsirm_p4
lsirm_p3_w2  <- env_w2$lsirm_p3

# ── Refresher 1 전처리 (격리 환경) ──
cat("\n====== Refresher 1 전처리 ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), local = env_r1)
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

# 5-layered LSIRM (v6: per-item kappa_j, nonhierarchical) 모델 로드
# sourceCpp는 fork 전에 한 번만 실행해야 함
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_5layered_nonhierarchical_cpp_v6.R"))
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
  a_sigma0=1, b_sigma0=1,
  mu_log_gamma1=0, sd_log_gamma1=0.5,
  mu_log_gamma2=0, sd_log_gamma2=0.5,
  mu_log_gamma3=0, sd_log_gamma3=0.5,
  mu_log_gamma4=0, sd_log_gamma4=0.5,
  mu_log_gamma5=0, sd_log_gamma5=0.5,
  # v6: shared prior hyperparameters for per-item log_kappa_j (v5와 동일)
  mu_log_kappa=0, sd_log_kappa=0.1,
  mu_beta4=0, sd_beta4=2,
  mu_beta5=0, sd_beta5=2
)

common_prop_sd <- list(
  alpha1=0.5, alpha2=0.5, alpha3=0.5, alpha4=0.5, alpha5=0.5,
  log_gamma1=0.1, log_gamma2=0.05, log_gamma3=0.10, log_gamma4=0.2, log_gamma5=0.2, a=0.3,
  beta1=0.6, beta2=0.25, beta3=0.2, beta4=0.3, beta5=0.3,
  b1=0.35, b2=0.2, b3=0.2, b4=0.5, b5=0.5,
  log_kappa=0.4
)

common_mcmc <- list(d = 2, n_iter = 100000, burnin = 20000, thin = 10)

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
# 4. Traceplot helper
#    v6 change: log_kappa is now a matrix (n_save × P3) — plot as vector trace
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

  # a (latent positions)
  if (has_valid(res$samples$a) && dim(res$samples$a)[2] > 0) {
    n_show <- dim(res$samples$a)[2]
    pdf(file.path(plot_dir, paste0(prefix, "_trace_a.pdf")), width = 8, height = 12)
    par(mfrow = c(4,2), mar = c(3,3,2,1))
    for (i in 1:n_show) {
      for (j in 1:dim(res$samples$a)[3]) {
        ts.plot(res$samples$a[, i, j], main = paste0("a: ", i, "_", j))
      }
    }
    dev.off()
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

  # b4 / b5 (ordinal item positions)
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

  # beta1/2/3
  for (bn in c("beta1","beta2","beta3")) {
    if (has_valid(res$samples[[bn]]) && ncol(res$samples[[bn]]) > 0) {
      pdf(file.path(plot_dir, paste0(prefix, "_trace_", bn, ".pdf")), width = 8, height = 12)
      plot_trace_vec(res$samples[[bn]], name = bn, mfrow = c(3,2))
      dev.off()
    }
  }

  # beta4 (GRM thresholds L4)
  if (has_valid(res$samples$beta4)) {
    b4_s <- res$samples$beta4
    P4_d <- dim(b4_s)[2]; Km1  <- dim(b4_s)[3]
    col_ord1 <- if (length(lsirm_data$col_ord1) > 0) lsirm_data$col_ord1 else paste0("ord1_j", 1:P4_d)
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta4_thr.pdf")), width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P4_d) for (k in 1:Km1) {
      x <- b4_s[, j, k]; q <- quantile(x, c(.025, .975), na.rm = TRUE)
      ts.plot(x, main = sprintf("beta4[%s, k=%d]", col_ord1[j], k))
      abline(h = c(mean(x, na.rm = TRUE), q),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()
  }

  # beta5 (GRM thresholds L5)
  if (has_valid(res$samples$beta5)) {
    b5_s <- res$samples$beta5
    P5_d <- dim(b5_s)[2]; Km1  <- dim(b5_s)[3]
    col_ord2 <- if (length(lsirm_data$col_ord2) > 0) lsirm_data$col_ord2 else paste0("ord2_j", 1:P5_d)
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta5_thr.pdf")), width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P5_d) for (k in 1:Km1) {
      x <- b5_s[, j, k]; q <- quantile(x, c(.025, .975), na.rm = TRUE)
      ts.plot(x, main = sprintf("beta5[%s, k=%d]", col_ord2[j], k))
      abline(h = c(mean(x, na.rm = TRUE), q),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()
  }

  # v6: per-item kappa_j traceplot (log_kappa is n_save × P3 matrix)
  if (has_valid(res$samples$log_kappa) && is.matrix(res$samples$log_kappa) && ncol(res$samples$log_kappa) > 0) {
    lk_mat <- res$samples$log_kappa
    P3_d <- ncol(lk_mat)
    col_cnt_names <- if (length(lsirm_data$col_cnt) >= P3_d) lsirm_data$col_cnt else paste0("cnt_j", 1:P3_d)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_kappa_per_item.pdf")), width = 10, height = 12)
    par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P3_d) {
      kx <- exp(lk_mat[, j])
      q <- quantile(kx, c(.025, .975), na.rm = TRUE)
      ts.plot(kx, main = sprintf("kappa[%s]", col_cnt_names[j]))
      abline(h = c(mean(kx, na.rm = TRUE), q),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()

    # Summary: posterior mean kappa_j + 95% CI per item
    kappa_summary <- data.frame(
      item   = col_cnt_names,
      mean   = apply(exp(lk_mat), 2, mean),
      q2.5   = apply(exp(lk_mat), 2, quantile, probs = 0.025),
      q97.5  = apply(exp(lk_mat), 2, quantile, probs = 0.975)
    )
    write.csv(kappa_summary,
              file.path(plot_dir, paste0(prefix, "_kappa_per_item_summary.csv")),
              row.names = FALSE)
    cat("\n── Per-item kappa_j Posterior Summary ──\n")
    print(round(kappa_summary[, -1], 3))
  }

  # Extra scalar parameters (v6: log_kappa removed from scalar trace — plotted above)
  pdf(file.path(plot_dir, paste0(prefix, "_trace_extra.pdf")), width = 8, height = 18)
  par(mfrow = c(7,2), mar = c(3,3,2,1))
  if (has_valid(res$samples$sigma0_sq))  plot_trace_scalar(res$samples$sigma0_sq,  true = NA, main = "sigma0_sq")
  if (has_valid(res$samples$log_gamma1)) plot_trace_scalar(res$samples$log_gamma1, true = NA, main = "gamma1 (Bin)", transform = exp)
  if (has_valid(res$samples$log_gamma2)) plot_trace_scalar(res$samples$log_gamma2, true = NA, main = "gamma2 (Con)", transform = exp)
  if (has_valid(res$samples$log_gamma3)) plot_trace_scalar(res$samples$log_gamma3, true = NA, main = "gamma3 (Cnt)", transform = exp)
  if (has_valid(res$samples$log_gamma4)) plot_trace_scalar(res$samples$log_gamma4, true = NA, main = "gamma4 (Ord1)", transform = exp)
  if (has_valid(res$samples$log_gamma5)) plot_trace_scalar(res$samples$log_gamma5, true = NA, main = "gamma5 (Ord2)", transform = exp)
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
    n_resp <- dim(lam)[2]; n_item <- dim(lam)[3]
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
    if (length(gamma_val_list) > 1) for (k in seq_along(gamma_val_list)[-1])
      lines(gamma_val_list[[k]], col = gamma_cols[gamma_exist][k])
    legend("topright", legend = gamma_names[gamma_exist],
           col = gamma_cols[gamma_exist], lwd = 1, bty = "n", cex = 0.9)
    dens_list <- lapply(gamma_val_list, density)
    xr <- range(unlist(lapply(dens_list, function(d) d$x)))
    yr <- range(unlist(lapply(dens_list, function(d) d$y)))
    base::plot(dens_list[[1]], col = gamma_cols[gamma_exist][1], lwd = 2,
         xlim = xr, ylim = yr, xlab = expression(gamma[l]), ylab = "Density",
         main = "Posterior Density: Layer-Specific Gamma")
    if (length(dens_list) > 1) for (k in seq_along(dens_list)[-1])
      lines(dens_list[[k]], col = gamma_cols[gamma_exist][k], lwd = 2)
    for (k in seq_along(gamma_val_list))
      abline(v = mean(gamma_val_list[[k]]), col = gamma_cols[gamma_exist][k], lty = 3)
    legend("topright", legend = paste0(gamma_names[gamma_exist], " (mean=",
           sprintf("%.3f", sapply(gamma_val_list, mean)), ")"),
           col = gamma_cols[gamma_exist], lwd = 2, bty = "n", cex = 0.9)
    dev.off()
  }
}


################################################################################
# 5. Biplot helper
################################################################################
make_biplot <- function(result, lsirm_data, title, filename, plot_dir) {

  res <- if (is.null(result$samples)) result else result$samples

  A_hat <- apply(res$a, c(2,3), mean)

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

  branch_col <- rep("gray80", nrow(A_hat))
  branch_pch <- rep(21, nrow(A_hat))

  all_pts <- rbind(A_hat, B1_hat, B2_hat, B3_hat, B4_hat, B5_hat)
  expand <- 0.08
  xr <- range(all_pts[,1], na.rm = TRUE); yr <- range(all_pts[,2], na.rm = TRUE)
  dx <- diff(xr); dy <- diff(yr)
  xlim <- xr + c(-1,1) * expand * dx; ylim <- yr + c(-1,1) * expand * dy
  if (dx == 0) xlim <- xr + c(-1,1); if (dy == 0) ylim <- yr + c(-1,1)

  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  base::plot(A_hat, pch = branch_pch, col = "black", bg = branch_col, cex = 0.8,
       xlab = "Dim1", ylab = "Dim2", main = title,
       xlim = xlim, ylim = ylim)
  if (has_bin)  { points(B1_hat, pch=21, bg="forestgreen", col="forestgreen", cex=1.2); text(B1_hat, labels=lsirm_data$col_bin, cex=0.7, pos=4, col="darkgreen") }
  if (has_con)  { points(B2_hat, pch=21, bg="orange",      col="orange",      cex=1.2); text(B2_hat, labels=lsirm_data$col_con, cex=0.7, pos=4, col="orange4") }
  if (has_cnt)  { points(B3_hat, pch=21, bg="cyan3",       col="cyan3",       cex=1.2); text(B3_hat, labels=lsirm_data$col_cnt, cex=0.7, pos=4, col="cyan4") }
  if (has_ord1) { points(B4_hat, pch=21, bg="purple",      col="purple",      cex=1.2); text(B4_hat, labels=lsirm_data$col_ord1, cex=0.7, pos=4, col="purple4") }
  if (has_ord2) { points(B5_hat, pch=21, bg="deeppink",    col="deeppink",    cex=1.2); text(B5_hat, labels=lsirm_data$col_ord2, cex=0.7, pos=4, col="deeppink4") }

  item_legend <- c(
    if(has_bin)  "Binary (CESD)" else NULL,
    if(has_con)  "Continuous (bio)" else NULL,
    if(has_cnt)  "Count (cog-error)" else NULL,
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
         legend = c("Respondent", item_legend),
         pch = c(21, rep(21, length(item_legend))),
         pt.bg = c("gray80", item_bg),
         bty = "n", cex = 0.8)
  dev.off()
}


################################################################################
# 6. Variable subsets (v5와 동일한 switch 구조)
################################################################################
# 6-0. Continuous layer
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
# v4: continuous layer에 cognitive Z-score 변수 없음
cognition_vars    <- character(0)

active_con_group <- "inflammation"   # "inflammation" or "cognition"

if (active_con_group == "inflammation") {
  active_con_vars <- inflammation_vars
} else {
  active_con_vars <- cognition_vars
}

con_all_names <- colnames(Y_con_full)
active_idx <- which(con_all_names %in% active_con_vars)
if (length(active_idx) == 0) stop("No matching continuous variables found for: ", active_con_group)
Y_con_subset <- Y_con_full[, active_idx, drop = FALSE]
col_con_subset <- con_all_names[active_idx]

cat(sprintf("\n=== Active continuous group: %s (%d vars) ===\n", active_con_group, length(active_idx)))
cat(sprintf("  Variables: %s\n", paste(col_con_subset, collapse = ", ")))


# 6-1. Count layer (v4 새 cognitive score count 15개)
cnt_cognition_vars <- c(
  "wl_immediate_omit", "wl_immediate_repetition", "wl_immediate_intrusion",
  "wl_delayed_omit",   "wl_delayed_repetition",   "wl_delayed_intrusion",
  "catflu_repetition", "catflu_intrusion",
  "numseries_incorrect", "backcount_error",
  "sgst_normal_incorrect", "sgst_reverse_incorrect",
  "sgst_mixed_nonswitch_incorrect", "sgst_mixed_switch_incorrect",
  "sgst_mixed_all_incorrect"
)

active_cnt_group <- "cognition"   # "all", "cognition", "none"

if (active_cnt_group == "none") {
  Y_cnt_subset   <- matrix(0L, nrow = nrow(Y_cnt_full), ncol = 0)
  col_cnt_subset <- character(0)
} else if (active_cnt_group == "all") {
  Y_cnt_subset   <- Y_cnt_full
  col_cnt_subset <- colnames(Y_cnt_full)
} else if (active_cnt_group == "cognition") {
  cnt_all_names <- colnames(Y_cnt_full)
  cnt_idx <- which(cnt_all_names %in% cnt_cognition_vars)
  if (length(cnt_idx) == 0) stop("No matching count variables found for: ", active_cnt_group)
  Y_cnt_subset   <- Y_cnt_full[, cnt_idx, drop = FALSE]
  col_cnt_subset <- cnt_all_names[cnt_idx]
}

cat(sprintf("\n=== Active count group: %s (%d vars) ===\n", active_cnt_group, ncol(Y_cnt_subset)))
cat(sprintf("  Variables: %s\n", paste(col_cnt_subset, collapse = ", ")))


# 6-2. Ordinal layer
ord_GDA_vars <- c("B4Q1D", "B4Q1H", "B4Q1K", "B4Q1N", "B4Q1P", "B4Q1T",
                   "B4Q1Z", "B4Q1FF", "B4Q1II", "B4Q1CCC", "B4Q1GGG")
ord_AA_vars  <- c("B4Q1B", "B4Q1F", "B4Q1M", "B4Q1Q", "B4Q1S", "B4Q1X",
                   "B4Q1BB", "B4Q1DD", "B4Q1KK", "B4Q1NN", "B4Q1PP",
                   "B4Q1RR", "B4Q1TT", "B4Q1VV", "B4Q1ZZ", "B4Q1BBB", "B4Q1JJJ")

active_ord_group <- "all"   # "all", "GDA", "AA", "none"

if (active_ord_group == "none") {
  Y_ord1_subset   <- matrix(0L, nrow = nrow(Y_ord1_full), ncol = 0)
  col_ord1_subset <- character(0)
} else if (active_ord_group == "all") {
  Y_ord1_subset   <- Y_ord1_full
  col_ord1_subset <- colnames(Y_ord1_full)
} else if (active_ord_group == "GDA") {
  ord_all_names <- colnames(Y_ord1_full)
  ord_idx <- which(ord_all_names %in% ord_GDA_vars)
  if (length(ord_idx) == 0) stop("No matching ordinal variables found for: GDA")
  Y_ord1_subset   <- Y_ord1_full[, ord_idx, drop = FALSE]
  col_ord1_subset <- ord_all_names[ord_idx]
} else if (active_ord_group == "AA") {
  ord_all_names <- colnames(Y_ord1_full)
  ord_idx <- which(ord_all_names %in% ord_AA_vars)
  if (length(ord_idx) == 0) stop("No matching ordinal variables found for: AA")
  Y_ord1_subset   <- Y_ord1_full[, ord_idx, drop = FALSE]
  col_ord1_subset <- ord_all_names[ord_idx]
}

cat(sprintf("\n=== Active ordinal group: %s (%d vars) ===\n", active_ord_group, ncol(Y_ord1_subset)))
if (length(col_ord1_subset) > 0) cat(sprintf("  Variables: %s\n", paste(col_ord1_subset, collapse = ", ")))


################################################################################
# 7. 순차 실행 (v6: per-item kappa_j)
################################################################################
run_label <- paste0("v6_con_", active_con_group, "_cnt_", active_cnt_group, "_ord_", active_ord_group)

for (cs in cases[1]) {
  # Override Y_con, Y_cnt, Y_ord1 with subsets
  cs$Y_con   <- Y_con_subset
  cs$col_con <- col_con_subset
  cs$Y_cnt   <- Y_cnt_subset
  cs$col_cnt <- col_cnt_subset
  cs$Y_ord1   <- Y_ord1_subset
  cs$col_ord1 <- col_ord1_subset

  cat(sprintf("\n\n========== %s [v6 per-item kappa | con=%s, cnt=%s, ord=%s] ==========\n",
              cs$label, active_con_group, active_cnt_group, active_ord_group))

  case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", run_label))
  if (!dir.exists(case_plot_dir)) dir.create(case_plot_dir, recursive = TRUE)

  cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
              nrow(cs$Y_bin), ncol(cs$Y_bin),
              nrow(cs$Y_con), ncol(cs$Y_con),
              nrow(cs$Y_cnt), ncol(cs$Y_cnt),
              nrow(cs$Y_ord1), ncol(cs$Y_ord1),
              nrow(cs$Y_ord2), ncol(cs$Y_ord2)))

  result <- lsirm_sharedpos_layer5_grm_v6_cpp(
    cs$Y_bin, round(cs$Y_con,1), cs$Y_cnt, cs$Y_ord1, cs$Y_ord2,
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

  cat(sprintf("\n-- %s [%s]: Acceptance Rates --\n", cs$name, run_label))
  # print summary (not full vectors) for acceptance
  acc_summary <- list(
    alpha1_mean = mean(result$accept$alpha1),
    alpha2_mean = mean(result$accept$alpha2),
    alpha3_mean = mean(result$accept$alpha3),
    alpha4_mean = mean(result$accept$alpha4),
    alpha5_mean = mean(result$accept$alpha5),
    beta1_mean  = mean(result$accept$beta1),
    beta2_mean  = mean(result$accept$beta2),
    beta3_mean  = mean(result$accept$beta3),
    log_gamma1  = result$accept$log_gamma1,
    log_gamma2  = result$accept$log_gamma2,
    log_gamma3  = result$accept$log_gamma3,
    log_gamma4  = result$accept$log_gamma4,
    log_gamma5  = result$accept$log_gamma5,
    log_kappa_mean = if (length(result$accept$log_kappa) > 0) mean(result$accept$log_kappa) else NA,
    log_kappa_range = if (length(result$accept$log_kappa) > 0) range(result$accept$log_kappa) else NA,
    a_mean       = mean(result$accept$a)
  )
  print(acc_summary)

  lsirm_data <- list(
    col_bin  = cs$col_bin,  col_con  = cs$col_con,
    col_cnt  = cs$col_cnt,  col_ord1 = cs$col_ord1, col_ord2 = cs$col_ord2,
    branch   = lsirm_all$branch
  )

  prefix <- paste0(cs$name, "_", run_label)

  make_traceplots(result, prefix = prefix, lsirm_data = lsirm_data, plot_dir = case_plot_dir)
  make_biplot(result, lsirm_data = lsirm_data,
              title = paste0("MIDUS W2+R1: ", cs$label, " [", run_label, "] (v6 per-item kappa)"),
              filename = paste0(prefix, "_biplot.pdf"),
              plot_dir = case_plot_dir)

  result_file <- file.path(case_plot_dir, paste0(cs$name, "_", run_label, "_result.rds"))
  saveRDS(result, result_file)
  cat(sprintf("  -> Plots & result saved to: %s\n", case_plot_dir))
}


################################################################################
# 8. Multilayered item positions b → K-means clustering
################################################################################
b1 <- colMeans(result$b1); b2 <- colMeans(result$b2)
b3 <- colMeans(result$b3); b4 <- colMeans(result$b4)
b  <- rbind(b1, b2, b3, b4)
rownames(b) <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1)

b_layer <- c(rep("bin",  ncol(result$b1)), rep("con",  ncol(result$b2)),
             rep("cnt",  ncol(result$b3)), rep("ord1", ncol(result$b4)))

km_multi <- kmeans_cluster_b(
  b, b_layer = b_layer,
  plot_dir    = case_plot_dir,
  file_prefix = "kmeans_multilayer",
  seed        = 42
)


################################################################################
# 8b. Bipartite SBM clustering on full posterior distance trajectory
#     (paper/bipartite_SBM.tex §1.2: post-LSIRM bipartite SBM)
#
#   - Q_sbm / L_sbm 길이 1: 단일 fit
#   - Q_sbm / L_sbm 중 하나 이상이 vector: grid search + ICL 비교 + best 선택
################################################################################
setwd(data_dir)
source(file.path(data_dir, "bipartite_SBM_cpp.R"))

# Item samples per layer in the same order as section 8 (bin, con, cnt, ord1, ord2)
b_samps_list <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
b_samps_list <- b_samps_list[
  sapply(b_samps_list, function(x) !is.null(x) && length(dim(x)) == 3 && dim(x)[2] > 0)
]

D_cube <- build_distance_cube(result$a, b_samps_list)        # array (n x p x m)
cat(sprintf("\n[bipartite SBM] D_cube dim: %s (n x p x m)\n",
            paste(dim(D_cube), collapse = " x ")))

# --- Posterior-mean positions for biplots (matches §8 b/B_uni convention) ---
A_hat_pm <- apply(result$a, c(2, 3), mean)
B_hat_pm <- do.call(rbind,
                    lapply(b_samps_list, function(x) apply(x, c(2, 3), mean)))
item_names_full <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1, cs$col_ord2)
item_names_full <- item_names_full[seq_len(nrow(B_hat_pm))]
stopifnot(length(item_names_full) == nrow(B_hat_pm),
          nrow(B_hat_pm) == dim(D_cube)[2])
rownames(B_hat_pm) <- item_names_full

# --- (Q, L) grid: scalar = single fit, vector = grid search ---
Q_sbm <- 3
L_sbm <- c(3,4,5,6,7)

grid_QL <- expand.grid(Q = Q_sbm, L = L_sbm)
grid_QL$icl            <- NA_real_
grid_QL$cll            <- NA_real_
grid_QL$penalty        <- NA_real_
grid_QL$nu             <- NA_integer_
grid_QL$acc_log_kappa  <- NA_real_
grid_QL$kappa_postmean <- NA_real_

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")

# helper: SBM biplot — respondent (circle, alpha) + item (square) colored by cluster
make_sbm_biplot <- function(A_hat, B_hat, item_cluster, resp_cluster,
                            item_names, title, filename, plot_dir, pal) {
  k_max <- max(c(item_cluster, resp_cluster))
  if (k_max > length(pal)) pal <- colorRampPalette(pal)(k_max)

  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  par(mar = c(4, 4, 3, 1))
  xr <- range(A_hat[, 1], B_hat[, 1])
  yr <- range(A_hat[, 2], B_hat[, 2])
  plot(A_hat, pch = 21,
       bg  = adjustcolor(pal[resp_cluster], alpha.f = 0.45),
       col = "black", cex = 0.9,
       xlab = "Dim1", ylab = "Dim2", main = title,
       xlim = xr + c(-1, 1) * 0.1 * diff(xr),
       ylim = yr + c(-1, 1) * 0.1 * diff(yr))
  points(B_hat, pch = 22, bg = pal[item_cluster], col = "black", cex = 1.6)
  text(B_hat, labels = item_names, pos = 4, cex = 0.55)

  uq_resp <- sort(unique(resp_cluster))
  uq_item <- sort(unique(item_cluster))
  legend("topright",
         legend = c(sprintf("Resp cluster %d", uq_resp),
                    sprintf("Item cluster %d", uq_item)),
         pch    = c(rep(21, length(uq_resp)), rep(22, length(uq_item))),
         pt.bg  = c(adjustcolor(pal[uq_resp], alpha.f = 0.45), pal[uq_item]),
         bty = "n", cex = 0.75)
  dev.off()
}

fits_list <- vector("list", nrow(grid_QL))
names(fits_list) <- sprintf("Q%d_L%d", grid_QL$Q, grid_QL$L)

# --- Loop over (Q, L) ---
for (gi in seq_len(nrow(grid_QL))) {
  Q_g <- grid_QL$Q[gi]; L_g <- grid_QL$L[gi]
  cat(sprintf("\n========== bipartite SBM: Q=%d, L=%d  (%d/%d) ==========\n",
              Q_g, L_g, gi, nrow(grid_QL)))

  fit_g <- bipartite_sbm(
    D_cube,
    Q       = Q_g, L = L_g,
    hyper   = list(r = 0.1, mu_log_kappa = 0, sd_log_kappa = 2),
    prop_sd = list(log_kappa = 0.1),
    init    = NULL,
    verbose = TRUE
  )

  icl_g <- compute_sbm_icl(fit_g, D_cube)
  grid_QL$icl[gi]            <- icl_g$icl
  grid_QL$cll[gi]            <- icl_g$cll
  grid_QL$penalty[gi]        <- icl_g$penalty
  grid_QL$nu[gi]             <- icl_g$nu
  grid_QL$acc_log_kappa[gi]  <- fit_g$acc_log_kappa
  grid_QL$kappa_postmean[gi] <- mean(exp(fit_g$log_kappa))

  cat(sprintf("[Q=%d, L=%d] cll=%.2f, penalty=%.2f, ICL=%.2f, ",
              Q_g, L_g, icl_g$cll, icl_g$penalty, icl_g$icl))
  cat(sprintf("acc_kappa=%.3f, kappa_postmean=%.3f\n",
              fit_g$acc_log_kappa, mean(exp(fit_g$log_kappa))))

  # cluster mode per item / respondent
  item_cluster <- apply(fit_g$w, 2,
                        function(v) as.integer(names(sort(table(v), decreasing = TRUE))[1]))
  resp_cluster <- apply(fit_g$z, 2,
                        function(v) as.integer(names(sort(table(v), decreasing = TRUE))[1]))

  # Lambda posterior mean (Q × L block intensity)
  Lambda_pm <- apply(fit_g$Lambda, c(1, 2), mean)

  # item co-clustering probability  P(w_j = w_{j'} | D)
  n_save_sbm <- nrow(fit_g$w)
  co_w <- matrix(0, length(item_names_full), length(item_names_full))
  for (s in seq_len(n_save_sbm))
    co_w <- co_w + outer(fit_g$w[s, ], fit_g$w[s, ], `==`)
  co_w <- co_w / n_save_sbm
  rownames(co_w) <- colnames(co_w) <- item_names_full

  # --- Save per-(Q,L) artifacts ---
  prefix_g <- sprintf("%s_%s_sbm_Q%d_L%d", cs$name, run_label, Q_g, L_g)

  saveRDS(fit_g, file.path(case_plot_dir, paste0(prefix_g, "_result.rds")))
  write.csv(data.frame(item = item_names_full, sbm_cluster = item_cluster),
            file.path(case_plot_dir, paste0(prefix_g, "_item_clusters.csv")),
            row.names = FALSE)
  write.csv(data.frame(respondent = seq_along(resp_cluster), sbm_cluster = resp_cluster),
            file.path(case_plot_dir, paste0(prefix_g, "_respondent_clusters.csv")),
            row.names = FALSE)
  write.csv(co_w,
            file.path(case_plot_dir, paste0(prefix_g, "_co_cluster_w.csv")))
  write.csv(round(Lambda_pm, 4),
            file.path(case_plot_dir, paste0(prefix_g, "_Lambda_postmean.csv")),
            row.names = FALSE)

  # --- Co-clustering heatmap (items reordered by cluster) ---
  pdf(file.path(case_plot_dir, paste0(prefix_g, "_co_cluster_w.pdf")),
      width = 9, height = 8)
  ord <- order(item_cluster)
  image(seq_along(item_names_full), seq_along(item_names_full), co_w[ord, ord],
        col  = colorRampPalette(c("white", "steelblue"))(50),
        xlab = "", ylab = "", axes = FALSE,
        main = sprintf("Bipartite SBM item co-clustering (Q=%d, L=%d, ICL=%.0f)",
                       Q_g, L_g, icl_g$icl))
  axis(1, at = seq_along(item_names_full), labels = item_names_full[ord],
       las = 2, cex.axis = 0.6)
  axis(2, at = seq_along(item_names_full), labels = item_names_full[ord],
       las = 2, cex.axis = 0.6)
  box()
  dev.off()

  # --- log_kappa traceplot ---
  pdf(file.path(case_plot_dir, paste0(prefix_g, "_trace_log_kappa.pdf")),
      width = 8, height = 5)
  ts.plot(fit_g$log_kappa,
          main = sprintf("log kappa trace (Q=%d, L=%d)", Q_g, L_g),
          ylab = expression(log(kappa)))
  abline(h = mean(fit_g$log_kappa), col = "darkgreen", lwd = 2)
  dev.off()

  # --- SBM biplot (resp + item, colored by SBM cluster) ---
  make_sbm_biplot(
    A_hat = A_hat_pm, B_hat = B_hat_pm,
    item_cluster = item_cluster, resp_cluster = resp_cluster,
    item_names = item_names_full,
    title = sprintf("Bipartite SBM (Q=%d, L=%d, ICL=%.0f)  |  %s",
                    Q_g, L_g, icl_g$icl, cs$label),
    filename = paste0(prefix_g, "_biplot.pdf"),
    plot_dir = case_plot_dir,
    pal = cluster_pal
  )

  cat(sprintf("\n[Q=%d, L=%d] item-cluster table:\n", Q_g, L_g))
  print(table(item_cluster))
  cat(sprintf("[Q=%d, L=%d] respondent-cluster table:\n", Q_g, L_g))
  print(table(resp_cluster))

  fits_list[[gi]] <- fit_g
}

# --- Grid summary CSV ---
write.csv(grid_QL,
          file.path(case_plot_dir,
                    sprintf("%s_%s_sbm_grid_ICL.csv", cs$name, run_label)),
          row.names = FALSE)

best_idx <- which.max(grid_QL$icl)
cat(sprintf("\n=== Best (Q, L) by ICL: (%d, %d), ICL = %.2f ===\n",
            grid_QL$Q[best_idx], grid_QL$L[best_idx], grid_QL$icl[best_idx]))
cat("\nFull (Q, L) grid:\n")
print(grid_QL)

# --- ICL plot (skip if grid has only one row) ---
if (nrow(grid_QL) > 1) {
  pdf(file.path(case_plot_dir,
                sprintf("%s_%s_sbm_ICL.pdf", cs$name, run_label)),
      width = 9, height = 7)
  par(mar = c(4, 4, 3, 1))
  if (length(unique(grid_QL$Q)) > 1 && length(unique(grid_QL$L)) > 1) {
    Qu <- sort(unique(grid_QL$Q)); Lu <- sort(unique(grid_QL$L))
    icl_mat <- matrix(NA_real_, length(Qu), length(Lu),
                      dimnames = list(paste0("Q=", Qu), paste0("L=", Lu)))
    for (gi in seq_len(nrow(grid_QL))) {
      qi <- match(grid_QL$Q[gi], Qu); li <- match(grid_QL$L[gi], Lu)
      icl_mat[qi, li] <- grid_QL$icl[gi]
    }
    image(seq_along(Qu), seq_along(Lu), icl_mat,
          col = colorRampPalette(c("white", "tomato"))(50),
          xlab = "Q", ylab = "L", axes = FALSE,
          main = sprintf("Bipartite SBM ICL grid  (best: Q=%d, L=%d, ICL=%.0f)",
                         grid_QL$Q[best_idx], grid_QL$L[best_idx],
                         grid_QL$icl[best_idx]))
    axis(1, at = seq_along(Qu), labels = Qu)
    axis(2, at = seq_along(Lu), labels = Lu)
    for (gi in seq_len(nrow(grid_QL))) {
      qi <- match(grid_QL$Q[gi], Qu); li <- match(grid_QL$L[gi], Lu)
      text(qi, li, sprintf("%.0f", grid_QL$icl[gi]), cex = 0.75)
    }
    qi_b <- match(grid_QL$Q[best_idx], Qu)
    li_b <- match(grid_QL$L[best_idx], Lu)
    rect(qi_b - 0.5, li_b - 0.5, qi_b + 0.5, li_b + 0.5,
         border = "black", lwd = 2.5)
    box()
  } else if (length(unique(grid_QL$Q)) > 1) {
    plot(grid_QL$Q, grid_QL$icl, type = "b", pch = 19, cex = 1.2,
         xlab = "Q", ylab = "ICL",
         main = sprintf("ICL vs Q  (L = %d, best Q = %d)",
                        grid_QL$L[1], grid_QL$Q[best_idx]))
    abline(v = grid_QL$Q[best_idx], col = "red", lty = 2)
  } else {
    plot(grid_QL$L, grid_QL$icl, type = "b", pch = 19, cex = 1.2,
         xlab = "L", ylab = "ICL",
         main = sprintf("ICL vs L  (Q = %d, best L = %d)",
                        grid_QL$Q[1], grid_QL$L[best_idx]))
    abline(v = grid_QL$L[best_idx], col = "red", lty = 2)
  }
  dev.off()
}

# free large object before next section
rm(D_cube); gc(verbose = FALSE)


################################################################################
# 9. Unilayered LSIRM (모든 변수 이진화 후 단일-layer LSIRM)
################################################################################
# 9-0. Project root에서 basic LSIRM 로드
proj_root <- dirname(data_dir)
setwd(proj_root)
source(file.path(proj_root, "my_LSIRM_cpp.R"))   # compiles my_LSIRM.cpp
setwd(data_dir)

# 9-1. Dichotomize: con/cnt/ord1 layer를 mean 기준 binary로 변환
#      (bin layer는 그대로 유지: bin_method="none")
dichot_method <- "mean"   # "mean", "Q1", "Q2", "Q3", "Q4"

prep <- make_binarized_multilayer_for_lsirm(
  Y_bin       = cs$Y_bin,
  Y_con       = cs$Y_con,
  Y_cnt       = cs$Y_cnt,
  Y_ord1      = cs$Y_ord1,
  Y_ord2      = NULL,
  bin_method  = "none",
  con_method  = dichot_method,
  cnt_method  = dichot_method,
  ord1_method = dichot_method,
  strict      = TRUE,
  ord_input   = "raw"
)

Y_bin_all        <- prep$Y_bin_all
layer_labels_uni <- prep$layer_labels
uni_names        <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1)
colnames(Y_bin_all) <- uni_names

cat(sprintf("\n=== Unilayered input Y_bin_all: %d × %d (dichot=%s) ===\n",
            nrow(Y_bin_all), ncol(Y_bin_all), dichot_method))
cat(sprintf("  layer counts: %s\n",
            paste(names(table(layer_labels_uni)),
                  table(layer_labels_uni), sep = "=", collapse = ", ")))

# 9-2. Run unilayered LSIRM (lsirm_basic from my_LSIRM_cpp.R)
uni_plot_dir <- file.path(plot_root,
                          paste0(cs$name, "_", run_label, "_unilayered_", dichot_method))
if (!dir.exists(uni_plot_dir)) dir.create(uni_plot_dir, recursive = TRUE)

setwd(proj_root)   # sourceCpp relative path 때문에 proj_root에서 실행
fit_uni <- lsirm_basic(
  Y_bin  = Y_bin_all,
  d      = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  prop_sd = list(
    alpha     = 0.70,
    beta      = 0.50,
    log_gamma = 0.05,
    a         = 0.50,
    b         = 0.30
  ),
  verbose   = TRUE,
  fix_gamma = TRUE
)
setwd(data_dir)

cat("\n-- Unilayered LSIRM acceptance rates --\n")
print(list(
  alpha_mean = mean(fit_uni$accept$alpha),
  beta_mean  = mean(fit_uni$accept$beta),
  a_mean     = mean(fit_uni$accept$a),
  b_mean     = mean(fit_uni$accept$b),
  log_gamma  = fit_uni$accept$log_gamma
))


################################################################################
# 10. Unilayered LSIRM traceplots / biplot
################################################################################
samps_uni <- fit_uni$samples

# (a) latent positions a (respondents)
pdf(file.path(uni_plot_dir, "uni_trace_a.pdf"), width = 8, height = 12)
par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
for (i in 1:dim(samps_uni$a)[2]) {
  for (j in 1:dim(samps_uni$a)[3]) {
    ts.plot(samps_uni$a[, i, j], main = paste0("a: ", i, "_", j))
  }
}
dev.off()

# (b) item positions b
pdf(file.path(uni_plot_dir, "uni_trace_b.pdf"), width = 8, height = 12)
par(mfrow = c(2, 2), mar = c(3, 3, 2, 1))
for (i in 1:dim(samps_uni$b)[2]) {
  for (j in 1:dim(samps_uni$b)[3]) {
    ts.plot(samps_uni$b[, i, j],
            main = paste0("b[", uni_names[i], "]_d", j))
  }
}
dev.off()

# (c) alpha & beta
pdf(file.path(uni_plot_dir, "uni_trace_alpha.pdf"), width = 8, height = 12)
par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
for (i in 1:ncol(samps_uni$alpha)) {
  x <- samps_uni$alpha[, i]
  q <- quantile(x, c(.025, .975))
  ts.plot(x, main = sprintf("alpha_%d", i))
  abline(h = c(mean(x), q), col = c("darkgreen", "blue", "blue"),
         lwd = 2, lty = c(1, 3, 3))
}
dev.off()

pdf(file.path(uni_plot_dir, "uni_trace_beta.pdf"), width = 8, height = 12)
par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
for (i in 1:ncol(samps_uni$beta)) {
  x <- samps_uni$beta[, i]
  q <- quantile(x, c(.025, .975))
  ts.plot(x, main = sprintf("beta[%s]", uni_names[i]))
  abline(h = c(mean(x), q), col = c("darkgreen", "blue", "blue"),
         lwd = 2, lty = c(1, 3, 3))
}
dev.off()

# (d) scalar parameters
pdf(file.path(uni_plot_dir, "uni_trace_extra.pdf"), width = 8, height = 10)
par(mfrow = c(3, 1), mar = c(3, 3, 2, 1))
plot_trace_scalar(samps_uni$log_gamma, true = NA, main = "gamma", transform = exp)
plot_trace_scalar(samps_uni$sigma_alpha_sq, true = NA, main = "sigma_alpha_sq")
plot_trace_scalar(samps_uni$tau_beta_sq,    true = NA, main = "tau_beta_sq")
dev.off()

# (e) posterior-mean positions biplot
A_uni <- apply(samps_uni$a, c(2, 3), mean)
B_uni <- apply(samps_uni$b, c(2, 3), mean)
rownames(B_uni) <- uni_names

layer_colors <- c(bin  = "forestgreen", con  = "orange",
                  cnt  = "cyan3",       ord1 = "purple", ord2 = "deeppink")
pdf(file.path(uni_plot_dir, "uni_biplot.pdf"), width = 10, height = 8)
par(mar = c(4, 4, 3, 1))
xr <- range(A_uni[,1], B_uni[,1]); yr <- range(A_uni[,2], B_uni[,2])
plot(A_uni, pch = 21, bg = "gray80", col = "black", cex = 0.8,
     xlab = "Dim1", ylab = "Dim2",
     main = sprintf("Unilayered LSIRM (dichot=%s)", dichot_method),
     xlim = xr + c(-1, 1) * 0.1 * diff(xr),
     ylim = yr + c(-1, 1) * 0.1 * diff(yr))
points(B_uni, pch = 21,
       bg  = layer_colors[layer_labels_uni],
       col = layer_colors[layer_labels_uni], cex = 1.3)
text(B_uni, labels = uni_names, pos = 4, cex = 0.6,
     col = layer_colors[layer_labels_uni])
present_layers <- unique(layer_labels_uni)
legend("topright",
       legend = c("Respondents", present_layers),
       pch = 21,
       pt.bg = c("gray80", layer_colors[present_layers]),
       bty = "n", cex = 0.8)
dev.off()

saveRDS(fit_uni, file.path(uni_plot_dir, "unilayered_result.rds"))
cat(sprintf("  -> Unilayered plots & result saved to: %s\n", uni_plot_dir))


################################################################################
# 11. K-means on unilayered b + Multilayered vs Unilayered 비교
################################################################################
km_uni <- kmeans_cluster_b(
  B_uni, b_layer = layer_labels_uni,
  plot_dir    = uni_plot_dir,
  file_prefix = "kmeans_unilayer",
  seed        = 42
)

# 두 b matrix가 같은 item 순서인지 확인 (전처리에서 동일 순서로 구성)
stopifnot(identical(rownames(b), rownames(B_uni)))

comp <- merge(
  km_multi$cluster_summary[, c("item", "layer", "cluster", "sil_w")],
  km_uni$cluster_summary  [, c("item", "layer", "cluster", "sil_w")],
  by = "item", suffixes = c("_multi", "_uni")
)
comp <- comp[match(rownames(b), comp$item), ]

cat("\n=== Multilayered vs Unilayered clustering ===\n")
cat(sprintf("  Multilayered: K = %d, avg silhouette = %.3f\n",
            km_multi$K_best, km_multi$sil_mean))
cat(sprintf("  Unilayered  : K = %d, avg silhouette = %.3f\n",
            km_uni$K_best, km_uni$sil_mean))

conf_tab <- table(Multi = comp$cluster_multi, Uni = comp$cluster_uni)
cat("\n-- Cluster assignment confusion (rows=multi, cols=uni) --\n")
print(conf_tab)

if (requireNamespace("mclust", quietly = TRUE)) {
  ari <- mclust::adjustedRandIndex(comp$cluster_multi, comp$cluster_uni)
  cat(sprintf("\nAdjusted Rand Index (multi vs uni): %.3f\n", ari))
} else {
  cat("\n[info] install.packages('mclust') 로 ARI도 확인 가능\n")
}

write.csv(comp,
          file.path(uni_plot_dir, "cluster_comparison_multi_vs_uni.csv"),
          row.names = FALSE)

# Side-by-side biplot
cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF")
pdf(file.path(uni_plot_dir, "biplot_compare_multi_vs_uni.pdf"),
    width = 14, height = 7)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(b, pch = 21,
     bg  = cluster_pal[km_multi$km$cluster], col = "black", cex = 1.4,
     xlab = "Dim1", ylab = "Dim2",
     main = sprintf("Multilayered (K=%d, sil=%.3f)",
                    km_multi$K_best, km_multi$sil_mean))
text(b[,1], b[,2], labels = rownames(b), pos = 4, cex = 0.5)

plot(B_uni, pch = 21,
     bg  = cluster_pal[km_uni$km$cluster], col = "black", cex = 1.4,
     xlab = "Dim1", ylab = "Dim2",
     main = sprintf("Unilayered (K=%d, sil=%.3f)",
                    km_uni$K_best, km_uni$sil_mean))
text(B_uni[,1], B_uni[,2], labels = rownames(B_uni), pos = 4, cex = 0.5)
dev.off()

cat(sprintf("\n-> Comparison outputs saved to: %s\n", uni_plot_dir))
cat("   * kmeans_unilayer_* (diagnostic / clusters / silhouette)\n")
cat("   * cluster_comparison_multi_vs_uni.csv\n")
cat("   * biplot_compare_multi_vs_uni.pdf\n")
