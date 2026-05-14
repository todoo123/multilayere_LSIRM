rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. 경로 설정
################################################################################
data_dir <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM/data"
setwd(data_dir)

################################################################################
# 0-1. 데이터 준비: Wave 2 + Refresher 1 합치기 (v4 preprocess)
################################################################################
cat("\n====== Wave 2 전처리 ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_v4.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all

cat("\n====== Refresher 1 전처리 ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(data_dir, "MIDUS_preprocess_refresher_v4.R"), local = env_r1)
lsirm_all_r1 <- env_r1$lsirm_all

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
  list(
    Y_bin  = Y_bin,  Y_cnt  = Y_cnt,
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2, Y_con = Y_con,
    row_ids  = c(l_w2$row_ids, l_r1$row_ids),
    branch   = c(l_w2$branch,  l_r1$branch),
    source   = c(rep("wave2",      length(l_w2$row_ids)),
                 rep("refresher1", length(l_r1$row_ids))),
    col_bin  = l_w2$col_bin,  col_cnt  = l_w2$col_cnt,
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2,
    col_con  = l_w2$col_con
  )
}
lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1, label = "P1-P3-P4")

################################################################################
# 0-2. 모델 / 유틸 로드
################################################################################
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_SBM_cpp_v7.R"))   # joint v7
source(file.path(data_dir, "bipartite_SBM_cpp.R"))     # compute_sbm_icl
source(file.path(data_dir, "utils.R"))

has_valid <- function(x) {
  if (is.null(x)) return(FALSE)
  if (is.matrix(x) && ncol(x) == 0) return(FALSE)
  if (is.array(x) && length(dim(x)) == 3 && dim(x)[2] == 0) return(FALSE)
  if (all(is.na(x))) return(FALSE)
  TRUE
}

################################################################################
# 1. Hyperparameter / MCMC 설정 (LSIRM / SBM 분리)
################################################################################
common_lsirm_hyper <- list(
  a_sigma = 1, b_sigma = 1,
  a_tau1 = 1, b_tau1 = 1, a_tau2 = 1, b_tau2 = 1, a_tau3 = 1, b_tau3 = 1,
  a_sigma0 = 1, b_sigma0 = 1,
  mu_log_gamma1 = 0, sd_log_gamma1 = 0.5,
  mu_log_gamma2 = 0, sd_log_gamma2 = 0.5,
  mu_log_gamma3 = 0, sd_log_gamma3 = 0.5,
  mu_log_gamma4 = 0, sd_log_gamma4 = 0.5,
  mu_log_gamma5 = 0, sd_log_gamma5 = 0.5,
  mu_log_kappa = 0, sd_log_kappa = 0.1,
  mu_beta4 = 0, sd_beta4 = 2,
  mu_beta5 = 0, sd_beta5 = 2
)
common_sbm_hyper <- list(
  r            = 0.1,
  mu_log_kappa = 0,
  sd_log_kappa = 2
)

common_lsirm_prop_sd <- list(
  alpha1 = 0.6, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.05, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.30,
  beta1 = 0.50, beta2 = 0.1, beta3 = 0.20, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.35, b2 = 0.20, b3 = 0.20, b4 = 0.10, b5 = 0.50,
  log_kappa = 0.30
)
common_sbm_prop_sd <- list(
  log_kappa = 0.05
)

common_mcmc <- list(d = 2, n_iter = 100000, burnin = 20000, thin = 10)
nu2 <- 4

# --- (Q, L) — scalar = 단일 chain, vector = expand.grid 모든 조합 별도 chain ---
Q_sbm <- c(4, 5)
L_sbm <- c(4, 5)

plot_root <- file.path(data_dir, "plot")
if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE)

################################################################################
# 2. Case 정의 (lsirm_all 기준, Y_ord2 항상 제외; case1_all만 사용)
################################################################################
n_all <- nrow(lsirm_all$Y_con)
make_empty <- function(n) matrix(0L, nrow = n, ncol = 0)
E  <- make_empty(n_all)
Eo <- matrix(0L, nrow = n_all, ncol = 0)

Y_con_full  <- scale(lsirm_all$Y_con)
Y_bin_full  <- lsirm_all$Y_bin
Y_cnt_full  <- lsirm_all$Y_cnt
Y_ord1_full <- lsirm_all$Y_ord1

cs <- list(
  name  = "case1_all",
  label = "Case 1: All (bin+con+cnt+ord)",
  Y_bin = Y_bin_full, Y_con = Y_con_full, Y_cnt = Y_cnt_full,
  Y_ord1 = Y_ord1_full, Y_ord2 = E,
  col_bin  = lsirm_all$col_bin,
  col_con  = lsirm_all$col_con,
  col_cnt  = lsirm_all$col_cnt,
  col_ord1 = lsirm_all$col_ord1,
  col_ord2 = character(0)
)

################################################################################
# 3. Variable subsets (v6와 동일 switch)
################################################################################
inflammation_vars <- c("B4BSCL14", "B4BNE12", "B4BIL6", "B4BFGN", "B4BCRP")
con_all_names <- colnames(Y_con_full)
active_idx <- which(con_all_names %in% inflammation_vars)
Y_con_subset   <- Y_con_full[, active_idx, drop = FALSE]
col_con_subset <- con_all_names[active_idx]

cnt_cognition_vars <- c(
  "wl_immediate_omit", "wl_immediate_repetition", "wl_immediate_intrusion",
  "wl_delayed_omit",   "wl_delayed_repetition",   "wl_delayed_intrusion",
  "catflu_repetition", "catflu_intrusion",
  "numseries_incorrect", "backcount_error",
  "sgst_normal_incorrect", "sgst_reverse_incorrect",
  "sgst_mixed_nonswitch_incorrect", "sgst_mixed_switch_incorrect",
  "sgst_mixed_all_incorrect"
)
cnt_all_names <- colnames(Y_cnt_full)
cnt_idx <- which(cnt_all_names %in% cnt_cognition_vars)
Y_cnt_subset   <- Y_cnt_full[, cnt_idx, drop = FALSE]
col_cnt_subset <- cnt_all_names[cnt_idx]

Y_ord1_subset   <- Y_ord1_full
col_ord1_subset <- colnames(Y_ord1_full)

cs$Y_con   <- Y_con_subset;   cs$col_con   <- col_con_subset
cs$Y_cnt   <- Y_cnt_subset;   cs$col_cnt   <- col_cnt_subset
cs$Y_ord1  <- Y_ord1_subset;  cs$col_ord1  <- col_ord1_subset

################################################################################
# 4. (Q, L) grid setup + 공통 plotting helpers
################################################################################
grid_QL <- expand.grid(Q = Q_sbm, L = L_sbm)
grid_QL$icl                <- NA_real_
grid_QL$cll                <- NA_real_
grid_QL$penalty            <- NA_real_
grid_QL$nu                 <- NA_integer_
grid_QL$acc_log_kappa_sbm  <- NA_real_
grid_QL$kappa_sbm_postmean <- NA_real_

cat(sprintf("\n========== v7 grid: %d (Q, L) combinations ==========\n",
            nrow(grid_QL)))
print(grid_QL[, c("Q", "L")])

plot_trace_vec <- function(samples_mat, name = "param", mfrow = c(2, 2)) {
  samples_mat <- as.matrix(samples_mat)
  par(mfrow = mfrow)
  for (i in seq_len(ncol(samples_mat))) {
    x <- samples_mat[, i]
    q <- quantile(x, c(.025, .975), na.rm = TRUE)
    ts.plot(x, main = sprintf("%s_%d", name, i))
    abline(h = c(mean(x, na.rm = TRUE), q),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
}
plot_trace_scalar_local <- function(x, true = NA, main = "", transform = identity) {
  y <- transform(x)
  q <- quantile(y, c(.025, .975), na.rm = TRUE)
  ts.plot(y, main = main)
  abline(h = c(mean(y, na.rm = TRUE), q),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
}
mode_label <- function(v)
  as.integer(names(sort(table(v), decreasing = TRUE))[1])

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")
expand_pal <- function(K, pal)
  if (K > length(pal)) colorRampPalette(pal)(K) else pal[1:K]

make_sbm_biplot <- function(A_hat, B_hat, item_cluster, resp_cluster,
                            item_names, title, filename, plot_dir, pal) {
  k_max <- max(c(item_cluster, resp_cluster))
  if (k_max > length(pal)) pal <- colorRampPalette(pal)(k_max)
  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  par(mar = c(4, 4, 3, 1))
  xr <- range(A_hat[, 1], B_hat[, 1])
  yr <- range(A_hat[, 2], B_hat[, 2])
  plot(A_hat, pch = 21,
       bg = adjustcolor(pal[resp_cluster], alpha.f = 0.45),
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
         pch = c(rep(21, length(uq_resp)), rep(22, length(uq_item))),
         pt.bg = c(adjustcolor(pal[uq_resp], alpha.f = 0.45), pal[uq_item]),
         bty = "n", cex = 0.75)
  dev.off()
}

################################################################################
# 5. (Q, L) grid loop — 각 조합당 독립 v7 joint chain
################################################################################
for (gi in seq_len(nrow(grid_QL))) {
  Q_g <- grid_QL$Q[gi]
  L_g <- grid_QL$L[gi]

  run_label     <- sprintf("v7_joint_Q%d_L%d", Q_g, L_g)
  case_plot_dir <- file.path(plot_root, paste0(cs$name, "_", run_label))
  if (!dir.exists(case_plot_dir)) dir.create(case_plot_dir, recursive = TRUE)
  prefix     <- paste0(cs$name, "_", run_label)
  sbm_prefix <- paste0(prefix, "_sbm")

  cat(sprintf("\n\n========== [%d/%d] %s [%s] ==========\n",
              gi, nrow(grid_QL), cs$label, run_label))
  cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
              nrow(cs$Y_bin),  ncol(cs$Y_bin),
              nrow(cs$Y_con),  ncol(cs$Y_con),
              nrow(cs$Y_cnt),  ncol(cs$Y_cnt),
              nrow(cs$Y_ord1), ncol(cs$Y_ord1),
              nrow(cs$Y_ord2), ncol(cs$Y_ord2)))

  ##############################################################################
  # 5-A. v7 joint MCMC 실행
  ##############################################################################
  result <- lsirm_sbm_v7_cpp(
    Y_bin   = cs$Y_bin,
    Y_con   = round(cs$Y_con, 1),
    Y_cnt   = cs$Y_cnt,
    Y_ord1  = cs$Y_ord1,
    Y_ord2  = cs$Y_ord2,
    Q       = Q_g, L = L_g,
    d       = common_mcmc$d,
    n_iter  = common_mcmc$n_iter,
    burnin  = common_mcmc$burnin,
    thin    = common_mcmc$thin,
    nu2     = nu2,
    lsirm_hyper   = common_lsirm_hyper,
    sbm_hyper     = common_sbm_hyper,
    lsirm_prop_sd = common_lsirm_prop_sd,
    sbm_prop_sd   = common_sbm_prop_sd,
    lsirm_init    = NULL,
    sbm_init      = NULL,
    verbose       = TRUE,
    fix_gamma     = FALSE
  )

  ##############################################################################
  # 5-B. Acceptance summary
  ##############################################################################
  acc <- result$accept
  cat(sprintf("\n-- %s Acceptance --\n", run_label))
  cat(sprintf("  alpha1..5 mean : %.3f / %.3f / %.3f / %.3f / %.3f\n",
              mean(acc$alpha1), mean(acc$alpha2), mean(acc$alpha3),
              mean(acc$alpha4), mean(acc$alpha5)))
  cat(sprintf("  beta1..3 mean  : %.3f / %.3f / %.3f\n",
              mean(acc$beta1), mean(acc$beta2), mean(acc$beta3)))
  cat(sprintf("  log_gamma1..5  : %.3f / %.3f / %.3f / %.3f / %.3f\n",
              acc$log_gamma1, acc$log_gamma2, acc$log_gamma3,
              acc$log_gamma4, acc$log_gamma5))
  cat(sprintf("  a / b1..5 mean : %.3f / %.3f, %.3f, %.3f, %.3f, %.3f\n",
              mean(acc$a),  mean(acc$b1), mean(acc$b2),
              mean(acc$b3), mean(acc$b4), mean(acc$b5)))
  cat(sprintf("  LSIRM log_kappa per-item mean: %.3f\n", mean(acc$log_kappa)))
  cat(sprintf("  SBM   log_kappa             : %.3f\n", acc$sbm_log_kappa))

  ##############################################################################
  # 5-C. LSIRM traceplots
  ##############################################################################
  if (has_valid(result$a)) {
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_a.pdf")),
        width = 8, height = 12)
    par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
    for (i in 1:dim(result$a)[2])
      for (j in 1:dim(result$a)[3])
        ts.plot(result$a[, i, j], main = paste0("a: ", i, "_", j))
    dev.off()
  }

  for (k in 1:5) {
    bname <- paste0("b", k)
    bmat <- result[[bname]]
    if (!has_valid(bmat) || dim(bmat)[2] == 0) next
    col_layer <- switch(bname,
                        b1 = cs$col_bin, b2 = cs$col_con, b3 = cs$col_cnt,
                        b4 = cs$col_ord1, b5 = cs$col_ord2)
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", bname, ".pdf")),
        width = 8, height = 12)
    par(mfrow = c(2, 2), mar = c(3, 3, 2, 1))
    for (i in 1:dim(bmat)[2])
      for (j in 1:dim(bmat)[3])
        ts.plot(bmat[, i, j],
                main = paste0(bname, ": ",
                              (if (length(col_layer) >= i) col_layer[i] else i),
                              "_d", j))
    dev.off()
  }

  for (al in 1:5) {
    aname <- paste0("alpha", al)
    if (has_valid(result[[aname]]) && ncol(result[[aname]]) > 0) {
      pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", aname, ".pdf")),
          width = 8, height = 12)
      plot_trace_vec(result[[aname]], name = aname, mfrow = c(3, 2))
      dev.off()
    }
  }

  for (bn in c("beta1", "beta2", "beta3")) {
    if (has_valid(result[[bn]]) && ncol(result[[bn]]) > 0) {
      pdf(file.path(case_plot_dir, paste0(prefix, "_trace_", bn, ".pdf")),
          width = 8, height = 12)
      plot_trace_vec(result[[bn]], name = bn, mfrow = c(3, 2))
      dev.off()
    }
  }

  if (has_valid(result$beta4)) {
    b4s <- result$beta4
    P4d <- dim(b4s)[2]; Km1 <- dim(b4s)[3]
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_beta4_thr.pdf")),
        width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P4d) for (k in 1:Km1) {
      x <- b4s[, j, k]
      ts.plot(x,
              main = sprintf("beta4[%s, k=%d]",
                             ifelse(length(cs$col_ord1) >= j,
                                    cs$col_ord1[j], j), k))
      abline(h = c(mean(x), quantile(x, c(.025, .975))),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()
  }
  if (has_valid(result$beta5)) {
    b5s <- result$beta5
    P5d <- dim(b5s)[2]; Km1 <- dim(b5s)[3]
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_beta5_thr.pdf")),
        width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P5d) for (k in 1:Km1) {
      x <- b5s[, j, k]
      ts.plot(x,
              main = sprintf("beta5[%s, k=%d]",
                             ifelse(length(cs$col_ord2) >= j,
                                    cs$col_ord2[j], j), k))
      abline(h = c(mean(x), quantile(x, c(.025, .975))),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()
  }

  if (has_valid(result$log_kappa) && is.matrix(result$log_kappa) &&
      ncol(result$log_kappa) > 0) {
    lk <- result$log_kappa
    P3d <- ncol(lk)
    cn <- if (length(cs$col_cnt) >= P3d) cs$col_cnt else paste0("cnt_j", 1:P3d)
    pdf(file.path(case_plot_dir, paste0(prefix, "_trace_lsirm_kappa_per_item.pdf")),
        width = 10, height = 12)
    par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P3d) {
      kx <- exp(lk[, j])
      ts.plot(kx, main = sprintf("LSIRM kappa[%s]", cn[j]))
      abline(h = c(mean(kx), quantile(kx, c(.025, .975))),
             col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
    }
    dev.off()
  }

  pdf(file.path(case_plot_dir, paste0(prefix, "_trace_lsirm_extra.pdf")),
      width = 8, height = 18)
  par(mfrow = c(7, 2), mar = c(3, 3, 2, 1))
  plot_trace_scalar_local(result$sigma0_sq,  main = "sigma0_sq")
  plot_trace_scalar_local(result$log_gamma1, main = "gamma1 (Bin)",  transform = exp)
  plot_trace_scalar_local(result$log_gamma2, main = "gamma2 (Con)",  transform = exp)
  plot_trace_scalar_local(result$log_gamma3, main = "gamma3 (Cnt)",  transform = exp)
  plot_trace_scalar_local(result$log_gamma4, main = "gamma4 (Ord1)", transform = exp)
  plot_trace_scalar_local(result$log_gamma5, main = "gamma5 (Ord2)", transform = exp)
  for (al in 1:5) {
    sname <- paste0("sigma_alpha", al, "_sq")
    if (has_valid(result[[sname]])) plot_trace_scalar_local(result[[sname]], main = sname)
  }
  if (has_valid(result$lambda2_mean))
    plot_trace_scalar_local(result$lambda2_mean, main = "lambda2_mean")
  dev.off()

  ##############################################################################
  # 5-D. SBM Gibbs / MH parameter traceplots
  ##############################################################################
  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_pi.pdf")),
      width = 9, height = 6)
  par(mfrow = c(ceiling(Q_g / 2), 2), mar = c(3, 3, 2, 1))
  for (q in 1:Q_g) {
    x <- result$sbm_pi[, q]
    ts.plot(x, main = sprintf("pi[%d]", q), ylab = "")
    abline(h = c(mean(x), quantile(x, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_rho.pdf")),
      width = 9, height = 6)
  par(mfrow = c(ceiling(L_g / 2), 2), mar = c(3, 3, 2, 1))
  for (l in 1:L_g) {
    x <- result$sbm_rho[, l]
    ts.plot(x, main = sprintf("rho[%d]", l), ylab = "")
    abline(h = c(mean(x), quantile(x, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()

  Lam_arr <- result$sbm_Lambda  # (Q × L × n_save)
  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_Lambda.pdf")),
      width = 10, height = 12)
  par(mfrow = c(Q_g, L_g), mar = c(3, 3, 2, 1))
  for (q in 1:Q_g) for (l in 1:L_g) {
    x <- Lam_arr[q, l, ]
    ts.plot(x, main = sprintf("Lambda[%d,%d]", q, l), ylab = "")
    abline(h = c(mean(x), quantile(x, c(.025, .975))),
           col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  }
  dev.off()

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_kappa_Xbar.pdf")),
      width = 8, height = 8)
  par(mfrow = c(2, 1), mar = c(3, 3, 2, 1))
  xk <- exp(result$sbm_log_kappa)
  ts.plot(xk, main = "SBM kappa", ylab = expression(kappa[SBM]))
  abline(h = c(mean(xk), quantile(xk, c(.025, .975))),
         col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
  xb <- result$sbm_Xbar
  ts.plot(xb, main = "Xbar (mean distance per iter)",
          ylab = expression(bar(X)))
  abline(h = mean(xb), col = "darkgreen", lwd = 2)
  dev.off()

  ##############################################################################
  # 5-E. SBM membership traceplots
  ##############################################################################
  z_mat <- result$sbm_z   # (n_save × n)         1-based
  w_mat <- result$sbm_w   # (n_save × P_total)   1-based
  n_save <- nrow(z_mat)

  item_names_full <- c(cs$col_bin, cs$col_con, cs$col_cnt, cs$col_ord1, cs$col_ord2)
  P_total <- ncol(w_mat)
  item_names_full <- item_names_full[seq_len(P_total)]

  n_q_trace <- t(apply(z_mat, 1, function(v) tabulate(v, nbins = Q_g)))
  m_l_trace <- t(apply(w_mat, 1, function(v) tabulate(v, nbins = L_g)))

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_cluster_sizes.pdf")),
      width = 10, height = 8)
  par(mfrow = c(2, 1), mar = c(3, 4, 3, 1))
  matplot(n_q_trace, type = "l", lty = 1,
          col = seq_len(Q_g), xlab = "iter",
          ylab = expression(n[q]^{(s)}),
          main = "Respondent cluster sizes (z)")
  legend("topright", legend = paste0("q=", 1:Q_g),
         col = seq_len(Q_g), lty = 1, bty = "n", cex = 0.8)
  matplot(m_l_trace, type = "l", lty = 1,
          col = seq_len(L_g), xlab = "iter",
          ylab = expression(m[l]^{(s)}),
          main = "Item cluster sizes (w)")
  legend("topright", legend = paste0("l=", 1:L_g),
         col = seq_len(L_g), lty = 1, bty = "n", cex = 0.8)
  dev.off()

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_membership_heatmap_w.pdf")),
      width = 10, height = 7)
  par(mar = c(7, 4, 3, 1))
  ord_w <- order(item_names_full)
  image(seq_len(n_save), seq_len(P_total), w_mat[, ord_w, drop = FALSE],
        col = expand_pal(L_g, cluster_pal),
        xlab = "MCMC iteration (saved)", ylab = "",
        main = "Item membership trace  (w_j over iterations)",
        axes = FALSE)
  axis(1)
  axis(2, at = seq_len(P_total), labels = item_names_full[ord_w],
       las = 2, cex.axis = 0.55)
  box()
  legend("topright", legend = paste0("l=", 1:L_g),
         fill = expand_pal(L_g, cluster_pal), bty = "n", cex = 0.8)
  dev.off()

  n_resp <- ncol(z_mat)
  if (n_resp > 80) {
    set.seed(42); resp_show <- sort(sample.int(n_resp, 80))
  } else resp_show <- seq_len(n_resp)

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_membership_heatmap_z.pdf")),
      width = 10, height = 7)
  par(mar = c(4, 4, 3, 1))
  image(seq_len(n_save), seq_along(resp_show),
        z_mat[, resp_show, drop = FALSE],
        col = expand_pal(Q_g, cluster_pal),
        xlab = "MCMC iteration (saved)",
        ylab = "Respondent (subset)",
        main = sprintf("Respondent membership trace  (sampled %d/%d)",
                       length(resp_show), n_resp),
        axes = FALSE)
  axis(1); axis(2, at = seq_along(resp_show), labels = resp_show,
                las = 2, cex.axis = 0.6)
  box()
  legend("topright", legend = paste0("q=", 1:Q_g),
         fill = expand_pal(Q_g, cluster_pal), bty = "n", cex = 0.8)
  dev.off()

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_w_individual.pdf")),
      width = 9, height = 12)
  n_show_w <- min(P_total, 12)
  par(mfrow = c(4, 3), mar = c(3, 3, 2, 1))
  for (jj in seq_len(n_show_w))
    ts.plot(w_mat[, jj], main = sprintf("w[%s]", item_names_full[jj]),
            ylab = "cluster")
  dev.off()

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_trace_z_individual.pdf")),
      width = 9, height = 12)
  n_show_z <- min(n_resp, 12)
  par(mfrow = c(4, 3), mar = c(3, 3, 2, 1))
  for (ii in seq_len(n_show_z))
    ts.plot(z_mat[, ii], main = sprintf("z[i=%d]", ii), ylab = "cluster")
  dev.off()

  ##############################################################################
  # 5-F. SBM cluster summaries + co-clustering + biplot + ICL
  ##############################################################################
  item_cluster <- apply(w_mat, 2, mode_label)
  resp_cluster <- apply(z_mat, 2, mode_label)
  Lambda_pm    <- apply(Lam_arr, c(1, 2), mean)

  co_w <- matrix(0, P_total, P_total)
  for (s in seq_len(n_save))
    co_w <- co_w + outer(w_mat[s, ], w_mat[s, ], `==`)
  co_w <- co_w / n_save
  rownames(co_w) <- colnames(co_w) <- item_names_full

  saveRDS(result, file.path(case_plot_dir, paste0(prefix, "_result.rds")))
  write.csv(data.frame(item = item_names_full, sbm_cluster = item_cluster),
            file.path(case_plot_dir, paste0(sbm_prefix, "_item_clusters.csv")),
            row.names = FALSE)
  write.csv(data.frame(respondent = seq_along(resp_cluster),
                       sbm_cluster = resp_cluster),
            file.path(case_plot_dir, paste0(sbm_prefix, "_respondent_clusters.csv")),
            row.names = FALSE)
  write.csv(co_w,
            file.path(case_plot_dir, paste0(sbm_prefix, "_co_cluster_w.csv")))
  write.csv(round(Lambda_pm, 4),
            file.path(case_plot_dir, paste0(sbm_prefix, "_Lambda_postmean.csv")),
            row.names = FALSE)

  pdf(file.path(case_plot_dir, paste0(sbm_prefix, "_co_cluster_w.pdf")),
      width = 9, height = 8)
  ord_c <- order(item_cluster)
  image(seq_along(item_names_full), seq_along(item_names_full), co_w[ord_c, ord_c],
        col  = colorRampPalette(c("white", "steelblue"))(50),
        xlab = "", ylab = "", axes = FALSE,
        main = sprintf("SBM item co-clustering  (Q=%d, L=%d)", Q_g, L_g))
  axis(1, at = seq_along(item_names_full), labels = item_names_full[ord_c],
       las = 2, cex.axis = 0.6)
  axis(2, at = seq_along(item_names_full), labels = item_names_full[ord_c],
       las = 2, cex.axis = 0.6)
  box()
  dev.off()

  # ICL via shared helper (uses full LSIRM distance trajectory)
  b_samps_list <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
  b_samps_list <- b_samps_list[
    sapply(b_samps_list, function(x) !is.null(x) &&
                                     length(dim(x)) == 3 &&
                                     dim(x)[2] > 0)
  ]
  D_cube <- build_distance_cube(result$a, b_samps_list)
  fit_for_icl <- list(
    z         = result$sbm_z,
    w         = result$sbm_w,
    pi        = result$sbm_pi,
    rho       = result$sbm_rho,
    Lambda    = result$sbm_Lambda,
    log_kappa = result$sbm_log_kappa,
    Q = Q_g, L = L_g
  )
  icl <- compute_sbm_icl(fit_for_icl, D_cube)
  cat(sprintf("\n[ICL] cll=%.2f, penalty=%.2f, ICL=%.2f, nu=%d\n",
              icl$cll, icl$penalty, icl$icl, icl$nu))
  write.csv(data.frame(Q = Q_g, L = L_g, cll = icl$cll,
                       penalty = icl$penalty, icl = icl$icl, nu = icl$nu),
            file.path(case_plot_dir, paste0(sbm_prefix, "_ICL.csv")),
            row.names = FALSE)

  # SBM biplot
  A_hat_pm <- apply(result$a, c(2, 3), mean)
  B_hat_pm <- do.call(rbind,
                      lapply(b_samps_list, function(x) apply(x, c(2, 3), mean)))
  rownames(B_hat_pm) <- item_names_full
  make_sbm_biplot(
    A_hat = A_hat_pm, B_hat = B_hat_pm,
    item_cluster = item_cluster, resp_cluster = resp_cluster,
    item_names = item_names_full,
    title = sprintf("Joint LSIRM+SBM v7  (Q=%d, L=%d, ICL=%.0f)  |  %s",
                    Q_g, L_g, icl$icl, cs$label),
    filename = paste0(sbm_prefix, "_biplot.pdf"),
    plot_dir = case_plot_dir,
    pal = cluster_pal
  )

  cat(sprintf("\n=== [Q=%d, L=%d] item cluster table ===\n", Q_g, L_g))
  print(table(item_cluster))
  cat(sprintf("=== [Q=%d, L=%d] respondent cluster table ===\n", Q_g, L_g))
  print(table(resp_cluster))
  cat(sprintf("=== [Q=%d, L=%d] Lambda posterior mean ===\n", Q_g, L_g))
  print(round(Lambda_pm, 3))
  cat(sprintf("\n-> v7 plots & artifacts saved to: %s\n", case_plot_dir))

  # Update grid summary
  grid_QL$icl[gi]                <- icl$icl
  grid_QL$cll[gi]                <- icl$cll
  grid_QL$penalty[gi]            <- icl$penalty
  grid_QL$nu[gi]                 <- icl$nu
  grid_QL$acc_log_kappa_sbm[gi]  <- result$accept$sbm_log_kappa
  grid_QL$kappa_sbm_postmean[gi] <- mean(exp(result$sbm_log_kappa))

  # Free memory before next iteration
  rm(result, D_cube, fit_for_icl,
     z_mat, w_mat, Lam_arr, co_w,
     A_hat_pm, B_hat_pm, b_samps_list,
     n_q_trace, m_l_trace,
     item_cluster, resp_cluster, Lambda_pm)
  gc(verbose = FALSE)
}

################################################################################
# 6. Grid 종합 — ICL CSV + heatmap / line plot
################################################################################
write.csv(grid_QL,
          file.path(plot_root,
                    sprintf("%s_v7_grid_ICL.csv", cs$name)),
          row.names = FALSE)

best_idx <- which.max(grid_QL$icl)
cat(sprintf("\n=== Best (Q, L) by ICL: (%d, %d), ICL = %.2f ===\n",
            grid_QL$Q[best_idx], grid_QL$L[best_idx], grid_QL$icl[best_idx]))
cat("\nFull (Q, L) grid:\n")
print(grid_QL)

if (nrow(grid_QL) > 1) {
  pdf(file.path(plot_root, sprintf("%s_v7_grid_ICL.pdf", cs$name)),
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
          main = sprintf("v7 joint LSIRM+SBM ICL grid  (best: Q=%d, L=%d, ICL=%.0f)",
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
         main = sprintf("v7 ICL vs Q  (L = %d, best Q = %d)",
                        grid_QL$L[1], grid_QL$Q[best_idx]))
    abline(v = grid_QL$Q[best_idx], col = "red", lty = 2)
  } else {
    plot(grid_QL$L, grid_QL$icl, type = "b", pch = 19, cex = 1.2,
         xlab = "L", ylab = "ICL",
         main = sprintf("v7 ICL vs L  (Q = %d, best L = %d)",
                        grid_QL$Q[1], grid_QL$L[best_idx]))
    abline(v = grid_QL$L[best_idx], col = "red", lty = 2)
  }
  dev.off()
}

cat(sprintf("\n-> Grid summary saved to: %s\n",
            file.path(plot_root, sprintf("%s_v7_grid_ICL.csv", cs$name))))

