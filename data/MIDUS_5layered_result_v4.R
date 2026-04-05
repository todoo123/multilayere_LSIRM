rm(list = ls())

################################################################################
# 0. 데이터 준비 (EDA 스크립트 실행)
################################################################################
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data")
source("MIDUS_preprocess_2_v3.R")

# 5-layered LSIRM (v4: robust continuous) 모델 로드
setwd("/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM")
source("data/my_LSIRM_5layered_nonhierarchical_cpp_v4.R")
source("utils.R")

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
  mu_log_kappa=0, sd_log_kappa=0.5,
  mu_u=0, sd_u=2, mu_delta=0, sd_delta=1
)

common_prop_sd <- list(
  alpha1=0.5, alpha2=0.5, alpha3=0.5, alpha4=0.5, alpha5=0.5,
  log_gamma1=0.1, log_gamma2=0.05, log_gamma3=0.10, log_gamma4=0.2, log_gamma5=0.2, a=0.3,
  beta1=0.6, beta2=0.25, beta3=0.2,
  b1=0.35, b2=0.2, b3=0.2, b4=0.5, b5=0.5,
  log_kappa=0.4, u=0.3, delta=0.3, delta2=0.3
)

common_mcmc <- list(d = 2, n_iter = 100000, burnin = 20000, thin = 10)

# Robust continuous layer: degrees of freedom
nu2 <- 5

# plot 디렉토리 생성
plot_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data/plot"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

################################################################################
# 2. Analysis Case 3: P1–P3–P4 (통합)
################################################################################
cat("\n\n========== Case 3: P1-P3-P4 (Robust v4) ==========\n")

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

result_all <- lsirm_sharedpos_layer5_robust_cpp(
  Y_bin_all, Y_con_all, Y_cnt_all, Y_ord1_all, Y_ord2_all,
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

cat("\n── P1-P3-P4 Acceptance Rates (Robust v4) ──\n")
print(result_all$accept)


################################################################################
# 3. 진단: Traceplots
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

  # alpha (layer-specific)
  for (al in 1:5) {
    aname <- paste0("alpha", al)
    if (!is.null(res$samples[[aname]]) && ncol(res$samples[[aname]]) > 0) {
      pdf(file.path(plot_dir, paste0(prefix, "_trace_", aname, ".pdf")), width = 8, height = 12)
      plot_trace_vec(res$samples[[aname]][, 1:ncol(res$samples[[aname]])], name = aname, mfrow = c(3,2))
      dev.off()
    }
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
  pdf(file.path(plot_dir, paste0(prefix, "_trace_extra.pdf")), width = 8, height = 18)
  par(mfrow = c(7,2), mar = c(3,3,2,1))
  if (!is.null(res$samples$sigma0_sq))  plot_trace_scalar(res$samples$sigma0_sq,  true = NA, main = "sigma0_sq")
  if (!is.null(res$samples$log_gamma1)) plot_trace_scalar(res$samples$log_gamma1, true = NA, main = "gamma1 (Bin)", transform = exp)
  if (!is.null(res$samples$log_gamma2)) plot_trace_scalar(res$samples$log_gamma2, true = NA, main = "gamma2 (Con)", transform = exp)
  if (!is.null(res$samples$log_gamma3)) plot_trace_scalar(res$samples$log_gamma3, true = NA, main = "gamma3 (Cnt)", transform = exp)
  if (!is.null(res$samples$log_gamma4)) plot_trace_scalar(res$samples$log_gamma4, true = NA, main = "gamma4 (Ord1)", transform = exp)
  if (!is.null(res$samples$log_gamma5)) plot_trace_scalar(res$samples$log_gamma5, true = NA, main = "gamma5 (Ord2)", transform = exp)
  if (!is.null(res$samples$log_kappa))  plot_trace_scalar(res$samples$log_kappa,  true = NA, main = "kappa", transform = exp)
  for (al in 1:5) {
    sname <- paste0("sigma_alpha", al, "_sq")
    if (!is.null(res$samples[[sname]])) plot_trace_scalar(res$samples[[sname]], true = NA, main = sname)
  }
  # lambda2_mean traceplot
  if (!is.null(res$samples$lambda2_mean)) {
    plot_trace_scalar(res$samples$lambda2_mean, true = NA, main = "lambda2_mean (robust weight)")
  }
  dev.off()

  # --- Lambda2 per-edge traceplots ---
  if (!is.null(res$samples$lambda2) && length(dim(res$samples$lambda2)) == 3) {
    lam <- res$samples$lambda2  # (n_save, n, P2) after aperm in wrapper
    n_resp <- dim(lam)[2]
    n_item <- dim(lam)[3]

    # Select edges to plot: up to 12 (i,j) pairs spread across the matrix
    n_edge_show <- min(12, n_resp * n_item)
    set.seed(42)
    all_edges <- expand.grid(i = 1:n_resp, j = 1:n_item)
    edge_idx  <- all_edges[sort(sample(nrow(all_edges), n_edge_show)), ]

    pdf(file.path(plot_dir, paste0(prefix, "_trace_lambda2_edges.pdf")), width = 10, height = 12)
    par(mfrow = c(4, 3), mar = c(3, 3, 2, 1))
    for (r in seq_len(nrow(edge_idx))) {
      ii <- edge_idx$i[r]; jj <- edge_idx$j[r]
      x <- lam[, ii, jj]
      jname <- if (!is.null(lsirm_data$col_con) && jj <= length(lsirm_data$col_con)) lsirm_data$col_con[jj] else jj
      ts.plot(x, main = bquote(lambda[2] ~ "(" * .(ii) * "," * .(jname) * ")"))
      abline(h = mean(x), col = "darkgreen", lwd = 2)
      abline(h = 1, col = "red", lty = 2)  # reference: lambda=1 means no downweighting
    }
    dev.off()
  }

  # --- Lambda2 posterior mean heatmap ---
  if (!is.null(res$samples$lambda2_postmean)) {
    lam_pm <- res$samples$lambda2_postmean  # (n x P2)
    col_names <- if (!is.null(lsirm_data$col_con)) lsirm_data$col_con else paste0("j", 1:ncol(lam_pm))

    pdf(file.path(plot_dir, paste0(prefix, "_lambda2_postmean_heatmap.pdf")), width = 10, height = 8)

    # Heatmap with color scale: low lambda (outliers) in red, lambda~1 in white/blue
    col_pal <- colorRampPalette(c("red", "white", "steelblue"))(100)
    lam_clipped <- pmin(lam_pm, quantile(lam_pm, 0.99))  # clip for color range
    image(1:nrow(lam_pm), 1:ncol(lam_pm), lam_clipped,
          col = col_pal, xlab = "Respondent (i)", ylab = "Item (j)",
          main = expression("Posterior Mean of " * lambda[ij]^{(2)}),
          axes = FALSE)
    axis(1, at = seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm))),
         labels = seq(1, nrow(lam_pm), length.out = min(10, nrow(lam_pm))))
    axis(2, at = 1:ncol(lam_pm), labels = col_names, las = 2, cex.axis = 0.7)
    box()

    dev.off()

    # --- Lambda2 per-item boxplot ---
    pdf(file.path(plot_dir, paste0(prefix, "_lambda2_postmean_boxplot.pdf")), width = 10, height = 6)
    colnames(lam_pm) <- col_names
    boxplot(as.data.frame(lam_pm), las = 2, cex.axis = 0.7,
            main = expression("Posterior Mean " * lambda[ij]^{(2)} * " by Item"),
            ylab = expression(lambda[ij]^{(2)}), col = "lightyellow", outline = TRUE)
    abline(h = 1, col = "red", lty = 2)
    dev.off()

    # --- Print summary ---
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


# ── Case 3: P1-P3-P4 ──
make_traceplots(result_all, prefix = "M2_ALL_v4", lsirm_data = lsirm_all)


################################################################################
# 4. Biplot: 잠재 공간 시각화
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

make_biplot(result_all, lsirm_all, "MIDUS_2: P1-P3-P4 (Robust v4)",  "M2_ALL_v4_biplot.pdf")
