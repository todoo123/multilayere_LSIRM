rm(list = ls())

library(dplyr)
library(purrr)

################################################################################
# 0. 데이터 준비: Wave 2 + Refresher 1 합치기
################################################################################
proj_root <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"

# ── Wave 2 전처리 (격리 환경) ──
cat("\n====== Wave 2 전처리 ======\n")
env_w2 <- new.env(parent = globalenv())
source(file.path(proj_root, "data/MIDUS_preprocess_2_v3.R"), local = env_w2)
lsirm_all_w2 <- env_w2$lsirm_all

# ── Refresher 1 전처리 (격리 환경) ──
cat("\n====== Refresher 1 전처리 ======\n")
env_r1 <- new.env(parent = globalenv())
source(file.path(proj_root, "data/MIDUS_preprocess_refresher_v3.R"), local = env_r1)
lsirm_all_r1 <- env_r1$lsirm_all

# ── 합치기 함수 ──
combine_lsirm <- function(l_w2, l_r1, label = "") {
  rbind_mat <- function(m1, m2) {
    if (ncol(m1) == 0 && ncol(m2) == 0) return(m1)
    colnames(m2) <- colnames(m1)
    rbind(m1, m2)
  }

  Y_ord1 <- rbind_mat(l_w2$Y_ord1, l_r1$Y_ord1)
  Y_ord2 <- rbind_mat(l_w2$Y_ord2, l_r1$Y_ord2)

  combined <- list(
    Y_ord1 = Y_ord1, Y_ord2 = Y_ord2,
    row_ids  = c(l_w2$row_ids, l_r1$row_ids),
    branch   = c(l_w2$branch,  l_r1$branch),
    source   = c(rep("wave2", length(l_w2$row_ids)),
                 rep("refresher1", length(l_r1$row_ids))),
    col_ord1 = l_w2$col_ord1, col_ord2 = l_w2$col_ord2
  )

  n_w2 <- length(l_w2$row_ids)
  n_r1 <- length(l_r1$row_ids)
  cat(sprintf("\n=== %s: Combined %d (W2) + %d (R1) = %d명 ===\n",
              label, n_w2, n_r1, n_w2 + n_r1))
  cat(sprintf("  Y_ord1: %d×%d | Y_ord2: %d×%d\n",
              nrow(Y_ord1), ncol(Y_ord1), nrow(Y_ord2), ncol(Y_ord2)))
  combined
}

lsirm_all <- combine_lsirm(lsirm_all_w2, lsirm_all_r1, label = "Ordinal-only")

# Ordinal-only LSIRM 모델 로드
setwd(proj_root)
source("test/lsirm_ordinal_only.R")
source("utils.R")

################################################################################
# 1. 공통 hyperparameter / MCMC 설정
################################################################################
common_hyper <- list(
  a_sigma=1, b_sigma=1,
  mu_log_gamma=0, sd_log_gamma=0.5,
  mu_delta=0, sd_delta=1
)

common_prop_sd <- list(
  alpha=0.5, log_gamma=0.2, a=0.3,
  b=0.5, delta=0.3
)

common_mcmc <- list(d = 2, n_iter = 100000, burnin = 20000, thin = 10)

# plot 디렉토리 생성
plot_dir <- file.path(proj_root, "test/plot")
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

################################################################################
# 2-A. Ordinal-1 (MASQ, 5-point) only
################################################################################
cat("\n\n========== Ordinal-1 (MASQ) Only ==========\n")

Y_ord1_all <- lsirm_all$Y_ord1
cat(sprintf("  Y_ord1: %d×%d\n", nrow(Y_ord1_all), ncol(Y_ord1_all)))

result_ord1 <- lsirm_ordinal_only_cpp(
  Y_ord1_all,
  d = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  hyper  = common_hyper,
  prop_sd = common_prop_sd,
  init = NULL,
  verbose = TRUE
)

cat("\n── Ordinal-1 Acceptance Rates ──\n")
print(result_ord1$accept)

################################################################################
# 2-B. Ordinal-2 (PSQI, 4-point) only
################################################################################
cat("\n\n========== Ordinal-2 (PSQI) Only ==========\n")

Y_ord2_all <- lsirm_all$Y_ord2
cat(sprintf("  Y_ord2: %d×%d\n", nrow(Y_ord2_all), ncol(Y_ord2_all)))

result_ord2 <- lsirm_ordinal_only_cpp(
  Y_ord2_all,
  d = common_mcmc$d,
  n_iter = common_mcmc$n_iter,
  burnin = common_mcmc$burnin,
  thin   = common_mcmc$thin,
  hyper  = common_hyper,
  prop_sd = common_prop_sd,
  init = NULL,
  verbose = TRUE
)

cat("\n── Ordinal-2 Acceptance Rates ──\n")
print(result_ord2$accept)

################################################################################
# 3. 진단: Traceplots
################################################################################
make_traceplots_ord <- function(result, prefix, col_names) {

  res <- result

  leg_no_true <- list(
    x = "topright",
    legend = c("Posterior mean", "95% credible interval"),
    col = c("darkgreen", "blue"), lwd = 2, lty = c(1, 3), bty = "n", cex = 0.8
  )

  plot_trace_vec_local <- function(samples_mat, name = "param", mfrow = c(2,2), leg = leg_no_true) {
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
  if (!is.null(res$a) && dim(res$a)[2] > 0) {
    n_show <- dim(res$a)[2]
    pdf(file.path(plot_dir, paste0(prefix, "_trace_a.pdf")), width = 8, height = 12)
    par(mfrow = c(4,2), mar = c(3,3,2,1))
    for (i in 1:n_show) {
      for (j in 1:dim(res$a)[3]) {
        ts.plot(res$a[, i, j], main = paste0("a: ", i, "_", j))
      }
    }
    dev.off()
  }

  # b (item positions)
  if (!is.null(res$b) && dim(res$b)[2] > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_b.pdf")), width = 8, height = 12)
    par(mfrow = c(2,2), mar = c(3,3,2,1))
    for (i in 1:dim(res$b)[2]) {
      for (j in 1:dim(res$b)[3]) {
        lab <- if (!is.null(col_names) && i <= length(col_names)) col_names[i] else paste0("j", i)
        ts.plot(res$b[, i, j], main = paste0("b_ord: ", lab, "_d", j))
      }
    }
    dev.off()
  }

  # alpha
  if (!is.null(res$alpha) && ncol(res$alpha) > 0) {
    pdf(file.path(plot_dir, paste0(prefix, "_trace_alpha.pdf")), width = 8, height = 12)
    plot_trace_vec_local(res$alpha[, 1:min(20, ncol(res$alpha))], name = "alpha", mfrow = c(4,2))
    dev.off()
  }

  # delta (ordinal thresholds)
  if (!is.null(res$delta) && length(dim(res$delta)) == 3) {
    d_s <- res$delta
    P_d <- dim(d_s)[2]
    Km2 <- dim(d_s)[3]
    col_ord <- if (!is.null(col_names)) col_names else paste0("ord_j", 1:P_d)

    pdf(file.path(plot_dir, paste0(prefix, "_trace_delta.pdf")), width = 8, height = 12)
    par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
    for (j in 1:P_d) {
      for (k in 1:Km2) {
        x <- d_s[, j, k]
        q <- quantile(x, c(.025, .975), na.rm = TRUE)
        ts.plot(x, main = sprintf("delta[%s, k=%d]", col_ord[j], k))
        abline(h = c(mean(x, na.rm = TRUE), q),
               col = c("darkgreen", "blue", "blue"), lwd = 2, lty = c(1, 3, 3))
      }
    }
    dev.off()
  }

  # Extra scalar parameters
  pdf(file.path(plot_dir, paste0(prefix, "_trace_extra.pdf")), width = 8, height = 10)
  par(mfrow = c(3,2), mar = c(3,3,2,1))
  if (!is.null(res$log_gamma))      plot_trace_scalar(res$log_gamma,      true = NA, main = "gamma (Ord)", transform = exp)
  if (!is.null(res$sigma_alpha_sq)) plot_trace_scalar(res$sigma_alpha_sq, true = NA, main = "sigma_alpha_sq")
  dev.off()
}

# ── Ordinal-1 ──
make_traceplots_ord(result_ord1, prefix = "ord1_only", col_names = lsirm_all$col_ord1)
# ── Ordinal-2 ──
# make_traceplots_ord(result_ord2, prefix = "ord2_only", col_names = lsirm_all$col_ord2)

################################################################################
# 4. Biplot: 잠재 공간 시각화
################################################################################
make_biplot_ord <- function(result, lsirm_data, col_names, title, filename) {

  res <- result

  A_hat <- apply(res$a, c(2,3), mean)
  B_hat <- apply(res$b, c(2,3), mean)

  # Branch coloring for respondents
  has_branch <- !is.null(lsirm_data$branch)
  
  branch_col <- rep("gray80", nrow(A_hat))
  branch_pch <- rep(21, nrow(A_hat))

  all_pts <- rbind(A_hat, B_hat)
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

  points(B_hat, pch = 21, bg = "purple", col = "purple", cex = 1.2)
  text(B_hat, labels = col_names, cex = 0.7, pos = 4, col = "purple4")

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

  legend("topright",
         legend = c(resp_legend, "Ordinal Items"),
         pch = c(resp_pch, 21),
         pt.bg = c(resp_bg, "purple"),
         bty = "n", cex = 0.8)

  dev.off()
}

make_biplot_ord(result_ord1, lsirm_all, lsirm_all$col_ord1,
                "MIDUS W2+R1: Ordinal-1 (MASQ) Only", "ord1_only_biplot.pdf")
make_biplot_ord(result_ord2, lsirm_all, lsirm_all$col_ord2,
                "MIDUS W2+R1: Ordinal-2 (PSQI) Only", "ord2_only_biplot.pdf")

