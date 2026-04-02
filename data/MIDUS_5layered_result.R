rm(list = ls())

################################################################################
# 0. 데이터 준비 (EDA 스크립트 실행)
################################################################################
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data")
source("MIDUS_EDA_2_v3.R")

# 5-layered LSIRM 모델 로드
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM")
source("data/my_LSIRM_5layered_nonhierarchical_cpp.R")
source("utils.R")

################################################################################
# 1. 공통 hyperparameter / MCMC 설정
################################################################################
common_hyper <- list(
  a_sigma=1, b_sigma=1,
  a_tau1=1, b_tau1=1, a_tau2=1, b_tau2=1, a_tau3=1, b_tau3=1,
  a_sigma0=1, b_sigma0=1,
  mu_log_gamma1=0, sd_log_gamma1=0.5,
  mu_log_gamma2=0, sd_log_gamma2=0.4,
  mu_log_gamma3=0, sd_log_gamma3=0.5,
  mu_log_gamma4=0, sd_log_gamma4=0.5,
  mu_log_gamma5=0, sd_log_gamma5=0.5,
  mu_log_kappa=0, sd_log_kappa=0.5,
  mu_u=0, sd_u=2, mu_delta=0, sd_delta=1
)

common_prop_sd <- list(
  alpha=0.5, log_gamma1=0.1, log_gamma2=0.05, log_gamma3=0.10, log_gamma4=0.2, log_gamma5=0.2, a=0.3,
  beta1=0.6, beta2=0.25, beta3=0.2,
  b1=0.35, b2=0.2, b3=0.2, b4=0.5, b5=0.5,
  log_kappa=0.4, u=0.3, delta=0.3, delta2=0.3
)

common_mcmc <- list(d = 2, n_iter = 100000, burnin = 20000, thin = 10)

# plot 디렉토리 생성
plot_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data/plot"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

################################################################################
# 2. Analysis Case 1: P1–P4 (MASQ + PSQI + bio continuous)
################################################################################
cat("\n\n========== Case 1: P1-P4 ==========\n")

# continuous 표준화
Y_con_p4  <- scale(lsirm_p4$Y_con)
Y_bin_p4  <- lsirm_p4$Y_bin
Y_cnt_p4  <- lsirm_p4$Y_cnt
Y_ord1_p4 <- lsirm_p4$Y_ord1                  # MASQ 5-point
Y_ord2_p4 <- lsirm_p4$Y_ord2                  # PSQI 4-point

cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
            nrow(Y_bin_p4), ncol(Y_bin_p4),
            nrow(Y_con_p4), ncol(Y_con_p4),
            nrow(Y_cnt_p4), ncol(Y_cnt_p4),
            nrow(Y_ord1_p4), ncol(Y_ord1_p4),
            nrow(Y_ord2_p4), ncol(Y_ord2_p4)))

result_p4 <- lsirm_sharedpos_layer5_lsgrm_cpp(
  Y_bin_p4, Y_con_p4, Y_cnt_p4, Y_ord1_p4, Y_ord2_p4,
  d = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  hyper  = common_hyper,
  prop_sd = common_prop_sd,
  init = NULL,
  verbose = TRUE
)

cat("\n── P1-P4 Acceptance Rates ──\n")
print(result_p4$accept)


################################################################################
# 3. Analysis Case 2: P1–P3 (cognitive only)
################################################################################
cat("\n\n========== Case 2: P1-P3 ==========\n")

Y_con_p3  <- scale(lsirm_p3$Y_con)
Y_bin_p3  <- lsirm_p3$Y_bin
Y_cnt_p3  <- lsirm_p3$Y_cnt
Y_ord1_p3 <- lsirm_p3$Y_ord1
Y_ord2_p3 <- lsirm_p3$Y_ord2

cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
            nrow(Y_bin_p3), ncol(Y_bin_p3),
            nrow(Y_con_p3), ncol(Y_con_p3),
            nrow(Y_cnt_p3), ncol(Y_cnt_p3),
            nrow(Y_ord1_p3), ncol(Y_ord1_p3),
            nrow(Y_ord2_p3), ncol(Y_ord2_p3)))

result_p3 <- lsirm_sharedpos_layer5_lsgrm_cpp(
  Y_bin_p3, Y_con_p3, Y_cnt_p3, Y_ord1_p3, Y_ord2_p3,
  d = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  hyper  = common_hyper,
  prop_sd = common_prop_sd,
  init = NULL,
  verbose = TRUE
)

cat("\n── P1-P3 Acceptance Rates ──\n")
print(result_p3$accept)

for(i in 1:dim(Y_con_all)[2]){
  hist(Y_con_all[,i], main = i)
}
lsirm_all$col_con[2]
################################################################################
# 4. Analysis Case 3: P1–P3–P4 (통합)
################################################################################
cat("\n\n========== Case 3: P1-P3-P4 ==========\n")

Y_con_all  <- scale(lsirm_all$Y_con)
Y_bin_all  <- lsirm_all$Y_bin
Y_cnt_all  <- lsirm_all$Y_cnt
Y_ord1_all <- lsirm_all$Y_ord1
Y_ord2_all <- lsirm_all$Y_ord2

Y_con_all  <- matrix(0L, nrow = nrow(Y_con_all), ncol = 0)
Y_bin_all  <- matrix(0L, nrow = nrow(Y_bin_all), ncol = 0)
Y_cnt_all  <- matrix(0L, nrow = nrow(Y_cnt_all), ncol = 0)
Y_ord1_all  <- matrix(0L, nrow = nrow(Y_ord1_all), ncol = 0)
Y_ord2_all  <- matrix(0L, nrow = nrow(Y_ord2_all), ncol = 0)

cat(sprintf("  Y_bin: %d×%d | Y_con: %d×%d | Y_cnt: %d×%d | Y_ord1: %d×%d | Y_ord2: %d×%d\n",
            nrow(Y_bin_all), ncol(Y_bin_all),
            nrow(Y_con_all), ncol(Y_con_all),
            nrow(Y_cnt_all), ncol(Y_cnt_all),
            nrow(Y_ord1_all), ncol(Y_ord1_all),
            nrow(Y_ord2_all), ncol(Y_ord2_all)))

result_all <- lsirm_sharedpos_layer5_lsgrm_cpp(
  Y_bin_all, Y_con_all, Y_cnt_all, Y_ord1_all, Y_ord2_all,
  d = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  hyper  = common_hyper,
  prop_sd = common_prop_sd,
  init = NULL,
  verbose = TRUE
)

cat("\n── P1-P3-P4 Acceptance Rates ──\n")
print(result_all$accept)


################################################################################
# 5. 진단: Traceplots
################################################################################
# ── helper ──
make_traceplots <- function(result, prefix, lsirm_data) {

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

  # a (latent positions - first 20 persons)
  if (!is.null(res$samples$a) && dim(res$samples$a)[2] > 0) {
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
  if (!is.null(res$samples$b1) && dim(res$samples$b1)[2] > 0) {
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
  if (!is.null(res$samples$b2) && dim(res$samples$b2)[2] > 0) {
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
  if (!is.null(res$samples$b3) && dim(res$samples$b3)[2] > 0) {
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
  if (!is.null(res$samples$b4) && dim(res$samples$b4)[2] > 0) {
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
  if (!is.null(res$samples$b5) && dim(res$samples$b5)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b5.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$samples$b5)[2]) {
      for (j in 1:dim(res$samples$b5)[3]) {
        ts.plot(res$samples$b5[, i, j], main = paste0("b5_ord2: ", lsirm_data$col_ord2[i], "_d", j))
      }
    }
    dev.off()
  }

  # alpha
  if (!is.null(res$samples$alpha) && ncol(res$samples$alpha) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_alpha.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$alpha[, 1:ncol(res$samples$alpha)], name = "alpha", mfrow = c(3,2))
    dev.off()
  }

  # beta1
  if (!is.null(res$samples$beta1) && ncol(res$samples$beta1) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta1.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta1, name = "beta1", mfrow = c(3,2))
    dev.off()
  }

  # beta2
  if (!is.null(res$samples$beta2) && ncol(res$samples$beta2) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta2.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta2, name = "beta2", mfrow = c(3,2))
    dev.off()
  }

  # beta3
  if (!is.null(res$samples$beta3) && ncol(res$samples$beta3) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_beta3.pdf")), width = 8, height = 12)
    plot_trace_vec(res$samples$beta3, name = "beta3", mfrow = c(3,2))
    dev.off()
  }

  # delta (L4 ordinal thresholds)
  if (!is.null(res$samples$delta)) {
    d_s <- res$samples$delta
    n_s <- dim(d_s)[1]
    delta_mat <- matrix(d_s, nrow = n_s, ncol = dim(d_s)[2] * dim(d_s)[3])
    pdf(file.path(plot_dir, paste0(prefix, "_trace_delta.pdf")), width = 8, height = 12)
    plot_trace_vec(delta_mat, name = "delta_ord1", mfrow = c(3,2))
    dev.off()
  }

  # delta2 (L5 ordinal thresholds)
  if (!is.null(res$samples$delta2)) {
    d_s <- res$samples$delta2
    n_s <- dim(d_s)[1]
    delta_mat <- matrix(d_s, nrow = n_s, ncol = dim(d_s)[2] * dim(d_s)[3])
    pdf(file.path(plot_dir, paste0(prefix, "_trace_delta2.pdf")), width = 8, height = 12)
    plot_trace_vec(delta_mat, name = "delta_ord2", mfrow = c(3,2))
    dev.off()
  }

  # Extra scalar parameters
  pdf(file.path(plot_dir, paste0(prefix, "_trace_extra.pdf")), width = 8, height = 14)
  par(mfrow = c(5,2), mar = c(3,3,2,1))
  if (!is.null(res$samples$sigma0_sq))  plot_trace_scalar(res$samples$sigma0_sq,  true = NA, main = "sigma0_sq")
  if (!is.null(res$samples$log_gamma1)) plot_trace_scalar(res$samples$log_gamma1, true = NA, main = "gamma1 (Bin)", transform = exp)
  if (!is.null(res$samples$log_gamma2)) plot_trace_scalar(res$samples$log_gamma2, true = NA, main = "gamma2 (Con)", transform = exp)
  if (!is.null(res$samples$log_gamma3)) plot_trace_scalar(res$samples$log_gamma3, true = NA, main = "gamma3 (Cnt)", transform = exp)
  if (!is.null(res$samples$log_gamma4)) plot_trace_scalar(res$samples$log_gamma4, true = NA, main = "gamma4 (Ord1)", transform = exp)
  if (!is.null(res$samples$log_gamma5)) plot_trace_scalar(res$samples$log_gamma5, true = NA, main = "gamma5 (Ord2)", transform = exp)
  if (!is.null(res$samples$log_kappa))  plot_trace_scalar(res$samples$log_kappa,  true = NA, main = "kappa", transform = exp)
  if (!is.null(res$samples$sigma_alpha_sq)) plot_trace_scalar(res$samples$sigma_alpha_sq, true = NA, main = "sigma_alpha_sq")
  dev.off()

  # Gamma comparison: independent per layer
  gamma_names <- c("Bin", "Con", "Cnt", "Ord1", "Ord2")
  gamma_cols  <- c("red", "blue", "darkgreen", "purple", "deeppink")
  gamma_list  <- list(res$samples$log_gamma1, res$samples$log_gamma2,
                      res$samples$log_gamma3, res$samples$log_gamma4,
                      res$samples$log_gamma5)
  gamma_exist <- !sapply(gamma_list, is.null)

  if (any(gamma_exist)) {
    gamma_val_list <- lapply(gamma_list[gamma_exist], exp)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_gamma_compare.pdf")), width = 10, height = 8)
    par(mfrow = c(2,1), mar = c(4,4,3,1))

    # Panel 1: Gamma traceplots per layer
    yr <- range(unlist(gamma_val_list))
    plot(gamma_val_list[[1]], type = "l", col = gamma_cols[gamma_exist][1],
         ylim = yr, xlab = "Iteration", ylab = expression(gamma[l]),
         main = "Layer-Specific Gamma Traceplots")
    for (k in seq_along(gamma_val_list)[-1]) {
      lines(gamma_val_list[[k]], col = gamma_cols[gamma_exist][k])
    }
    legend("topright", legend = gamma_names[gamma_exist],
           col = gamma_cols[gamma_exist], lwd = 1, bty = "n", cex = 0.9)

    # Panel 2: Posterior density per layer
    dens_list <- lapply(gamma_val_list, density)
    xr <- range(unlist(lapply(dens_list, function(d) d$x)))
    yr <- range(unlist(lapply(dens_list, function(d) d$y)))
    plot(dens_list[[1]], col = gamma_cols[gamma_exist][1], lwd = 2,
         xlim = xr, ylim = yr, xlab = expression(gamma[l]), ylab = "Density",
         main = "Posterior Density: Layer-Specific Gamma")
    for (k in seq_along(dens_list)[-1]) {
      lines(dens_list[[k]], col = gamma_cols[gamma_exist][k], lwd = 2)
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


# ── Case 1: P1-P4 ──
# make_traceplots(result_p4, prefix = "M2_P4", lsirm_data = lsirm_p4)

# ── Case 2: P1-P3 ──
# make_traceplots(result_p3, prefix = "M2_P3", lsirm_data = lsirm_p3)

# ── Case 3: P1-P3-P4 ──
make_traceplots(result_all, prefix = "M2_ALL", lsirm_data = lsirm_all)


################################################################################
# 6. Biplot: 잠재 공간 시각화
################################################################################

make_biplot <- function(result, lsirm_data, title, filename) {

  res <- if (is.null(result$samples)) result else result$samples

  A_hat <- apply(res$a, c(2,3), mean)
  
  has_bin  <- length(lsirm_data$col_bin)  > 0 && length(res$b1)>0
  has_con  <- length(lsirm_data$col_con)  > 0 && length(res$b2)>0
  has_cnt  <- length(lsirm_data$col_cnt)  > 0 && length(res$b3)>0
  has_ord1 <- length(lsirm_data$col_ord1) > 0 && length(res$b4)>0
  has_ord2 <- length(lsirm_data$col_ord2) > 0 && length(res$b5)>0
  
  B1_hat <- if (has_bin)  apply(res$b1, c(2,3), mean) else NULL
  B2_hat <- if (has_con)  apply(res$b2, c(2,3), mean) else NULL
  B3_hat <- if (has_cnt)  apply(res$b3, c(2,3), mean) else NULL
  B4_hat <- if (has_ord1) apply(res$b4, c(2,3), mean) else NULL
  B5_hat <- if (has_ord2) apply(res$b5, c(2,3), mean) else NULL

  # Branch coloring for respondents
  has_branch <- !is.null(lsirm_data$branch)
  if (has_branch) {
    br <- lsirm_data$branch
    branch_col <- ifelse(br == "Sadness", "#E41A1C",
                  ifelse(br == "Anhedonia", "#377EB8",
                  ifelse(br == "Both", "#984EA3", "gray80")))
    branch_pch <- ifelse(br == "Both", 24, 21)  # triangle for Both, circle otherwise
  } else {
    branch_col <- rep("gray80", nrow(A_hat))
    branch_pch <- rep(21, nrow(A_hat))
  }

  all_pts <- rbind(A_hat, B1_hat, B2_hat, B3_hat, B4_hat, B5_hat)
  expand <- 0.08
  xr <- range(all_pts[,1], na.rm = TRUE)
  yr <- range(all_pts[,2], na.rm = TRUE)
  dx <- diff(xr); dy <- diff(yr)
  xlim <- xr + c(-1,1) * expand * dx
  ylim <- yr + c(-1,1) * expand * dy
  if (dx == 0) xlim <- xr + c(-1,1)
  if (dy == 0) ylim <- yr + c(-1,1)

  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  plot(A_hat, pch = branch_pch, col = "black", bg = branch_col, cex = 0.8,
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

  # Build legend
  resp_legend <- if (has_branch) {
    c("Sadness only", "Anhedonia only", "Both")
  } else {
    "Respondent"
  }
  resp_bg <- if (has_branch) {
    c("#E41A1C", "#377EB8", "#984EA3")
  } else {
    "gray80"
  }
  resp_pch <- if (has_branch) c(21, 21, 24) else 21

  item_legend <- c(
    if(has_bin)  "Binary (med/supp/health)" else NULL,
    if(has_con)  "Continuous (bio/cog)" else NULL,
    if(has_cnt)  "Count (sleep/smoke/drink)" else NULL,
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

  # Print branch counts
  if (has_branch) {
    cat(sprintf("  [%s] Branch counts: %s\n", filename,
        paste(names(table(br)), table(br), sep="=", collapse=", ")))
  }
  dev.off()
}

# make_biplot(result_p4,  lsirm_p4,  "MIDUS_2: P1-P4 (Bio+Sleep)",  "M2_P4_biplot.pdf")
# make_biplot(result_p3,  lsirm_p3,  "MIDUS_2: P1-P3 (Cognitive)",  "M2_P3_biplot.pdf")
make_biplot(result_all, lsirm_all, "MIDUS_2: P1-P3-P4 (All)",     "M2_ALL_biplot.pdf")

################################################################################
# 7. Response pattern by branch (Sadness vs Anhedonia)
################################################################################
cat("\n\n========== Response Patterns by Branch ==========\n")

br <- lsirm_all$branch
idx_sad  <- which(br == "Sadness")
idx_anh  <- which(br == "Anhedonia")

Y_bin_all_raw <- lsirm_all$Y_bin
Y_cnt_all_raw <- lsirm_all$Y_cnt

# ── Binary items (proportion barplot) ──
if (length(lsirm_all$col_bin) > 0) {
  n_bin <- ncol(Y_bin_all_raw)
  par(mfrow = c(ceiling(n_bin / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_bin)) {
    prop_sad <- mean(Y_bin_all_raw[idx_sad, j], na.rm = TRUE)
    prop_anh <- mean(Y_bin_all_raw[idx_anh, j], na.rm = TRUE)
    mat <- rbind(Sadness = c(1 - prop_sad, prop_sad),
                 Anhedonia = c(1 - prop_anh, prop_anh))
    colnames(mat) <- c("No", "Yes")
    barplot(mat, beside = TRUE, col = c("#E41A1C", "#377EB8"),
            main = lsirm_all$col_bin[j], xlab = "Response", ylab = "Proportion",
            ylim = c(0, 1))
    if (j == 1) legend("topright", legend = c("Sadness", "Anhedonia"),
                        fill = c("#E41A1C", "#377EB8"), bty = "n", cex = 0.9)
  }
}

# ── Count items (barplot of frequencies) ──
if (length(lsirm_all$col_cnt) > 0) {
  n_cnt <- ncol(Y_cnt_all_raw)
  par(mfrow = c(ceiling(n_cnt / 2), 2), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_cnt)) {
    lvls <- sort(unique(c(Y_cnt_all_raw[idx_sad, j], Y_cnt_all_raw[idx_anh, j])))
    prop_sad <- table(factor(Y_cnt_all_raw[idx_sad, j], levels = lvls)) / length(idx_sad)
    prop_anh <- table(factor(Y_cnt_all_raw[idx_anh, j], levels = lvls)) / length(idx_anh)
    mat <- rbind(Sadness = as.numeric(prop_sad), Anhedonia = as.numeric(prop_anh))
    colnames(mat) <- lvls
    barplot(mat, beside = TRUE, col = c("#E41A1C", "#377EB8"),
            main = lsirm_all$col_cnt[j], xlab = "Count", ylab = "Proportion",
            ylim = c(0, max(mat) * 1.2))
    if (j == 1) legend("topright", legend = c("Sadness", "Anhedonia"),
                        fill = c("#E41A1C", "#377EB8"), bty = "n", cex = 0.9)
  }
}

# ── Ordinal-5 (MASQ) items ──
if (length(lsirm_all$col_ord1) > 0) {
  n_ord1 <- ncol(Y_ord1_all)
  par(mfrow = c(ceiling(n_ord1 / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_ord1)) {
    lvls <- sort(unique(c(Y_ord1_all[idx_sad, j], Y_ord1_all[idx_anh, j])))
    prop_sad <- table(factor(Y_ord1_all[idx_sad, j], levels = lvls)) / length(idx_sad)
    prop_anh <- table(factor(Y_ord1_all[idx_anh, j], levels = lvls)) / length(idx_anh)
    mat <- rbind(Sadness = as.numeric(prop_sad), Anhedonia = as.numeric(prop_anh))
    colnames(mat) <- lvls
    barplot(mat, beside = TRUE, col = c("#E41A1C", "#377EB8"),
            main = lsirm_all$col_ord1[j], xlab = "Response", ylab = "Proportion",
            ylim = c(0, max(mat) * 1.2))
    if (j == 1) legend("topright", legend = c("Sadness", "Anhedonia"),
                        fill = c("#E41A1C", "#377EB8"), bty = "n", cex = 0.9)
  }
}

# ── Ordinal-4 (PSQI) items ──
if (length(lsirm_all$col_ord2) > 0) {
  n_ord2 <- ncol(Y_ord2_all)
  par(mfrow = c(ceiling(n_ord2 / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_ord2)) {
    lvls <- sort(unique(c(Y_ord2_all[idx_sad, j], Y_ord2_all[idx_anh, j])))
    prop_sad <- table(factor(Y_ord2_all[idx_sad, j], levels = lvls)) / length(idx_sad)
    prop_anh <- table(factor(Y_ord2_all[idx_anh, j], levels = lvls)) / length(idx_anh)
    mat <- rbind(Sadness = as.numeric(prop_sad), Anhedonia = as.numeric(prop_anh))
    colnames(mat) <- lvls
    barplot(mat, beside = TRUE, col = c("#E41A1C", "#377EB8"),
            main = lsirm_all$col_ord2[j], xlab = "Response", ylab = "Proportion",
            ylim = c(0, max(mat) * 1.2))
    if (j == 1) legend("topright", legend = c("Sadness", "Anhedonia"),
                        fill = c("#E41A1C", "#377EB8"), bty = "n", cex = 0.9)
  }
}

# ── Continuous items (density plots) ──
if (length(lsirm_all$col_con) > 0) {
  n_con <- ncol(Y_con_all)
  par(mfrow = c(ceiling(n_con / 3), 3), mar = c(4, 4, 3, 1))
  for (j in seq_len(n_con)) {
    d_sad <- density(Y_con_all[idx_sad, j], na.rm = TRUE)
    d_anh <- density(Y_con_all[idx_anh, j], na.rm = TRUE)
    xr <- range(c(d_sad$x, d_anh$x))
    yr <- range(c(d_sad$y, d_anh$y))
    plot(d_sad, col = "#E41A1C", lwd = 2, xlim = xr, ylim = yr,
         main = lsirm_all$col_con[j], xlab = "Value (standardized)", ylab = "Density")
    lines(d_anh, col = "#377EB8", lwd = 2)
    if (j == 1) legend("topright", legend = c("Sadness", "Anhedonia"),
                        col = c("#E41A1C", "#377EB8"), lwd = 2, bty = "n", cex = 0.9)
  }
}

cat("\n=== 모든 분석 완료 ===\n")
cat("Plot 저장 위치:", plot_dir, "\n")


