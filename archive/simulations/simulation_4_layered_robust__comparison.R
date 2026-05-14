rm(list = ls())

library(Rcpp)
library(RcppArmadillo)
library(vegan)
library(parallel)

################################################################################
# 0. Setup: 두 모델 컴파일
################################################################################
setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM")

source("my_LSIRM_4layered_nonhierarchical_cpp_v4.R")   # standard LSIRM
source("my_LSIRM_4layered_nonhierarchical_cpp_v5.R")   # robust LSIRM

out_dir <- "simulation_results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

################################################################################
# 1. Simulation settings (simulation_4_layered_v5.R 기반)
################################################################################
n  <- 150
P1 <- 10   # binary
P2 <- 30   # continuous
P3 <- 10   # count
P4 <- 10   # ordinal
d  <- 2
K_ord <- 5

gamma_true <- 1.0
kappa_true <- 1
sigma0_true <- 1.0
nu2_true <- 5      # Student-t df for Y_con generation
nu2_fit  <- 5      # df used in robust model fitting

centers_resp <- rbind(c(-2, -2), c(2, 2), c(0, 0))
centers_item <- rbind(c(-2, -2), c(2, 2), c(0, 0))
sd_cluster_resp <- 0.50
sd_cluster_item <- 0.30

true_sigma_alpha <- 2.0
true_tau_beta1   <- 1.5
true_tau_beta2   <- 1.0
true_tau_beta3   <- 0.5
true_mu_u     <- 0;   true_sd_u     <- 2
true_mu_delta <- 0;   true_sd_delta <- 1

# Outlier settings (scenario 2)
outlier_frac     <- 0.10    # 10% of respondents
outlier_shift_sd <- 8.0     # outlier position sd (far from clusters)

# MCMC settings
mcmc <- list(d = 2, n_iter = 50000, burnin = 10000, thin = 10)

fit_hyper <- list(
  a_sigma=1, b_sigma=0.1,
  a_tau1=1, b_tau1=0.1, a_tau2=1, b_tau2=0.1, a_tau3=1, b_tau3=0.1,
  a_sigma0=1, b_sigma0=1,
  mu_log_gamma1=0, sd_log_gamma1=1, mu_log_gamma2=0, sd_log_gamma2=1,
  mu_log_gamma3=0, sd_log_gamma3=1, mu_log_gamma4=0, sd_log_gamma4=1,
  mu_log_kappa=0, sd_log_kappa=1,
  mu_u=0, sd_u=1, mu_delta=0, sd_delta=1
)

fit_prop_sd <- list(
  alpha=0.1, log_gamma1=0.05, log_gamma2=0.05, log_gamma3=0.05, log_gamma4=0.05, a=0.1,
  beta1=0.1, beta2=0.1, beta3=0.1,
  b1=0.1, b2=0.1, b3=0.1, b4=0.1,
  log_kappa=0.05, u=0.1, delta=0.1
)

# Replication
n_rep   <- 100
n_cores <- min(detectCores() - 1, 8)


################################################################################
# 2. Helper functions
################################################################################
softplus <- function(x) log1p(exp(-abs(x))) + pmax(x, 0)

build_thresholds_fixed0 <- function(delta_vec) {
  Kminus1 <- 1 + length(delta_vec)
  b <- numeric(Kminus1)
  b[1] <- 0.0
  if (Kminus1 >= 2) for (k in 2:Kminus1) b[k] <- b[k-1] + softplus(delta_vec[k-1])
  b
}

invlogit <- function(x) 1 / (1 + exp(-x))

dist_mat <- function(A, B) {
  nr <- nrow(A); P <- nrow(B)
  out <- matrix(0, nr, P)
  for (j in 1:P) out[, j] <- sqrt(rowSums((A - matrix(B[j,], nr, ncol(A), byrow = TRUE))^2))
  out
}

sample_positions_3clusters <- function(N, centers, sd = 0.4, prob = NULL) {
  K <- nrow(centers)
  if (is.null(prob)) prob <- rep(1/K, K)
  cl <- sample(1:K, size = N, replace = TRUE, prob = prob)
  X  <- centers[cl, , drop = FALSE] + matrix(rnorm(N * ncol(centers), 0, sd), N, ncol(centers))
  list(X = X, cl = cl)
}

# ── Coverage helpers (simulation_4_layered_v3_coverage.R 기반) ──

coverage_from_samples <- function(samples_mat, true_vec, prob = c(0.025, 0.975)) {
  samples_mat <- as.matrix(samples_mat)
  if (length(true_vec) != ncol(samples_mat)) stop("true_vec length mismatch")
  q_lo <- apply(samples_mat, 2, quantile, probs = prob[1], na.rm = TRUE)
  q_hi <- apply(samples_mat, 2, quantile, probs = prob[2], na.rm = TRUE)
  widths <- q_hi - q_lo
  ok <- !is.na(true_vec)
  cov_ind <- rep(NA_integer_, length(true_vec))
  cov_ind[ok] <- as.integer(q_lo[ok] <= true_vec[ok] & true_vec[ok] <= q_hi[ok])
  list(
    n_param     = sum(ok),
    n_covered   = sum(cov_ind[ok] == 1),
    pct_covered = if (sum(ok) == 0) NA_real_ else 100 * mean(cov_ind[ok] == 1),
    mean_width  = if (sum(ok) == 0) NA_real_ else mean(widths[ok]),
    sum_width   = if (sum(ok) == 0) 0 else sum(widths[ok])
  )
}

distance_coverage_layer <- function(a_samps, b_samps, log_gamma_samps,
                                    D_true, gamma_true_val = 1.0,
                                    prob = c(0.025, 0.975)) {
  S <- dim(a_samps)[1]; nr <- dim(a_samps)[2]; dd <- dim(a_samps)[3]; P <- dim(b_samps)[2]
  gamma_s <- exp(as.numeric(log_gamma_samps))
  true_scaled <- D_true * gamma_true_val
  n_param <- nr * P; n_covered <- 0L; sum_width <- 0.0

  for (j in 1:P) {
    dx <- a_samps[, , 1] - matrix(b_samps[, j, 1], nrow = S, ncol = nr)
    if (dd >= 2) {
      dy <- a_samps[, , 2] - matrix(b_samps[, j, 2], nrow = S, ncol = nr)
      dist_s <- sqrt(dx^2 + dy^2)
    } else {
      dist_s <- abs(dx)
    }
    dist_s <- dist_s * matrix(gamma_s, nrow = S, ncol = nr)
    q_lo <- apply(dist_s, 2, quantile, probs = prob[1], na.rm = TRUE)
    q_hi <- apply(dist_s, 2, quantile, probs = prob[2], na.rm = TRUE)
    sum_width <- sum_width + sum(q_hi - q_lo)
    tj <- true_scaled[, j]
    n_covered <- n_covered + sum(q_lo <= tj & tj <= q_hi, na.rm = TRUE)
  }
  list(
    n_param     = n_param,
    n_covered   = n_covered,
    pct_covered = 100 * (n_covered / n_param),
    mean_width  = sum_width / n_param,
    sum_width   = sum_width
  )
}


################################################################################
# 3. Data generation (2 scenarios)
################################################################################
generate_data <- function(scenario = c("standard", "outlier"), seed = NULL) {
  scenario <- match.arg(scenario)
  if (!is.null(seed)) set.seed(seed)

  # ── Respondent positions ──
  resp <- sample_positions_3clusters(n, centers_resp, sd_cluster_resp)
  A_true <- resp$X

  # Scenario 2: outlier respondents — 극단적 latent position
  if (scenario == "outlier") {
    n_out <- round(n * outlier_frac)
    idx_out <- sample(1:n, n_out)
    A_true[idx_out, ] <- matrix(rnorm(n_out * d, 0, outlier_shift_sd), n_out, d)
  }

  # ── Item positions (공유) ──
  B1_true <- sample_positions_3clusters(P1, centers_item, sd_cluster_item)$X
  B2_true <- sample_positions_3clusters(P2, centers_item, sd_cluster_item)$X
  B3_true <- sample_positions_3clusters(P3, centers_item, sd_cluster_item)$X
  B4_true <- sample_positions_3clusters(P4, centers_item, sd_cluster_item)$X

  # ── Person / item parameters ──
  alpha_true <- rnorm(n, 0, true_sigma_alpha)
  beta1_true <- rnorm(P1, 0, true_tau_beta1)
  beta2_true <- rnorm(P2, 0, true_tau_beta2)
  beta3_true <- rnorm(P3, 0, true_tau_beta3)
  u_true     <- rep(0.0, P4)
  delta_true <- matrix(rnorm(P4 * (K_ord - 2), true_mu_delta, true_sd_delta), P4, K_ord - 2)

  beta4_list <- vector("list", P4)
  for (j in 1:P4) beta4_list[[j]] <- build_thresholds_fixed0(delta_true[j, ])

  # ── Distances & linear predictors ──
  D1 <- dist_mat(A_true, B1_true)
  D2 <- dist_mat(A_true, B2_true)
  D3 <- dist_mat(A_true, B3_true)
  D4 <- dist_mat(A_true, B4_true)

  ETA1 <- outer(alpha_true, rep(1, P1)) - outer(rep(1, n), beta1_true) - gamma_true * D1
  ETA2 <- outer(alpha_true, rep(1, P2)) - outer(rep(1, n), beta2_true) - gamma_true * D2
  ETA3 <- outer(alpha_true, rep(1, P3)) - outer(rep(1, n), beta3_true) - gamma_true * D3
  ETA4 <- outer(alpha_true, rep(1, P4)) - gamma_true * D4

  # ── Generate Y ──
  # Binary
  Y_bin <- matrix(rbinom(n * P1, 1, invlogit(ETA1)), n, P1)

  # Continuous: Student-t errors (Normal-Gamma mixture)
  lambda_true <- matrix(rgamma(n * P2, shape = nu2_true / 2, rate = nu2_true / 2), n, P2)
  Y_con <- ETA2 + matrix(rnorm(n * P2, 0, sigma0_true), n, P2) / sqrt(lambda_true)
  storage.mode(Y_con) <- "numeric"

  # Count (NB2)
  Y_cnt <- matrix(rnbinom(n * P3, size = 1 / kappa_true, mu = exp(ETA3)), n, P3)

  # Ordinal
  Y_ord <- matrix(NA_integer_, n, P4)
  for (j in 1:P4) {
    bv <- beta4_list[[j]]
    for (i in 1:n) {
      Cv <- invlogit(ETA4[i, j] - bv)
      p <- numeric(K_ord)
      p[1] <- 1 - Cv[1]; p[K_ord] <- Cv[K_ord - 1]
      if (K_ord > 2) for (y in 2:(K_ord - 1)) p[y] <- Cv[y - 1] - Cv[y]
      p[p < 0] <- 0; ps <- sum(p)
      if (ps <= 0) { p <- rep(0, K_ord); p[ceiling(K_ord / 2)] <- 1 } else p <- p / ps
      Y_ord[i, j] <- sample.int(K_ord, 1, prob = p)
    }
  }
  storage.mode(Y_ord) <- "integer"

  list(
    Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt, Y_ord = Y_ord,
    alpha_true = alpha_true, beta1_true = beta1_true,
    beta2_true = beta2_true, beta3_true = beta3_true,
    u_true = u_true, delta_true = delta_true,
    D1_true = D1, D2_true = D2, D3_true = D3, D4_true = D4,
    lambda_true = lambda_true
  )
}


################################################################################
# 4. Coverage computation for one fit
################################################################################
compute_coverage <- function(fit, dat) {
  samps <- if (is.null(fit$samples)) fit else fit$samples
  rows <- list()

  # ── Scalar / vector parameters ──
  for (pname in c("alpha", "beta1", "beta2", "beta3")) {
    true_name <- paste0(pname, "_true")
    if (!is.null(samps[[pname]]) && !is.null(dat[[true_name]])) {
      tmp <- coverage_from_samples(samps[[pname]], dat[[true_name]])
      rows[[pname]] <- data.frame(
        param = pname, n_param = tmp$n_param,
        n_covered = tmp$n_covered, pct = tmp$pct_covered,
        mean_width = tmp$mean_width, sum_width = tmp$sum_width,
        stringsAsFactors = FALSE
      )
    }
  }

  # u (ordinal)
  if (!is.null(samps$u) && !is.null(dat$u_true)) {
    tmp <- coverage_from_samples(samps$u, dat$u_true)
    rows[["u"]] <- data.frame(
      param = "u", n_param = tmp$n_param,
      n_covered = tmp$n_covered, pct = tmp$pct_covered,
      mean_width = tmp$mean_width, sum_width = tmp$sum_width,
      stringsAsFactors = FALSE
    )
  }

  # delta (ordinal gaps)
  if (!is.null(samps$delta) && !is.null(dat$delta_true)) {
    ns <- dim(samps$delta)[1]
    dm <- matrix(samps$delta, nrow = ns, ncol = prod(dim(samps$delta)[-1]))
    tmp <- coverage_from_samples(dm, as.vector(dat$delta_true))
    rows[["delta"]] <- data.frame(
      param = "delta", n_param = tmp$n_param,
      n_covered = tmp$n_covered, pct = tmp$pct_covered,
      mean_width = tmp$mean_width, sum_width = tmp$sum_width,
      stringsAsFactors = FALSE
    )
  }

  # ── Distance coverage per layer ──
  layers <- list(
    list(b = "b1", lg = "log_gamma1", D = "D1_true", nm = "dist_bin"),
    list(b = "b2", lg = "log_gamma2", D = "D2_true", nm = "dist_con"),
    list(b = "b3", lg = "log_gamma3", D = "D3_true", nm = "dist_cnt"),
    list(b = "b4", lg = "log_gamma4", D = "D4_true", nm = "dist_ord")
  )
  for (L in layers) {
    if (!is.null(samps$a) && !is.null(samps[[L$b]]) && !is.null(samps[[L$lg]])) {
      tmpd <- distance_coverage_layer(samps$a, samps[[L$b]], samps[[L$lg]],
                                      dat[[L$D]], gamma_true)
      rows[[L$nm]] <- data.frame(
        param = L$nm, n_param = tmpd$n_param,
        n_covered = tmpd$n_covered, pct = tmpd$pct_covered,
        mean_width = tmpd$mean_width, sum_width = tmpd$sum_width,
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, rows)
}


################################################################################
# 5. Single replication worker
################################################################################
run_one_rep <- function(rep_id, scenario) {
  seed <- rep_id * 1000 + ifelse(scenario == "standard", 0, 500)
  dat <- generate_data(scenario, seed = seed)

  # ── Fit v4: Standard LSIRM ──
  fit_v4 <- tryCatch(
    lsirm_sharedpos_layer4_lsgrm_cpp(
      dat$Y_bin, dat$Y_con, dat$Y_cnt, dat$Y_ord,
      d = mcmc$d, n_iter = mcmc$n_iter, burnin = mcmc$burnin, thin = mcmc$thin,
      hyper = fit_hyper, prop_sd = fit_prop_sd,
      init = NULL, verbose = FALSE, fix_gamma = TRUE
    ),
    error = function(e) { message(sprintf("[Rep %d | %s] v4 error: %s", rep_id, scenario, e$message)); NULL }
  )

  # ── Fit v5: Robust LSIRM ──
  fit_v5 <- tryCatch(
    lsirm_sharedpos_layer4_robust_cpp(
      dat$Y_bin, dat$Y_con, dat$Y_cnt, dat$Y_ord,
      d = mcmc$d, n_iter = mcmc$n_iter, burnin = mcmc$burnin, thin = mcmc$thin,
      nu2 = nu2_fit,
      hyper = fit_hyper, prop_sd = fit_prop_sd,
      init = NULL, verbose = FALSE, fix_gamma = TRUE
    ),
    error = function(e) { message(sprintf("[Rep %d | %s] v5 error: %s", rep_id, scenario, e$message)); NULL }
  )

  # ── Coverage 계산 ──
  out <- list()
  if (!is.null(fit_v4)) {
    cv4 <- compute_coverage(fit_v4, dat)
    cv4$model <- "standard"; cv4$scenario <- scenario; cv4$rep <- rep_id
    out[["v4"]] <- cv4
  }
  if (!is.null(fit_v5)) {
    cv5 <- compute_coverage(fit_v5, dat)
    cv5$model <- "robust"; cv5$scenario <- scenario; cv5$rep <- rep_id
    out[["v5"]] <- cv5
  }

  do.call(rbind, out)
}


################################################################################
# 6. Parallel execution
################################################################################
cat(sprintf("\n=== Simulation: %d reps × 2 scenarios × 2 models, %d cores ===\n", n_rep, n_cores))

cat("\n────── Scenario 1: Standard (Student-t errors, nu=5) ──────\n")
t1 <- proc.time()
res_std <- mclapply(1:n_rep, function(r) {
  cat(sprintf("  [standard] rep %d/%d\n", r, n_rep))
  run_one_rep(r, "standard")
}, mc.cores = n_cores)
cat(sprintf("  Scenario 1 done: %.1f min\n", (proc.time() - t1)[3] / 60))

cat("\n────── Scenario 2: Outlier (극단 latent position, %.0f%%) ──────\n", 100 * outlier_frac)
t2 <- proc.time()
res_out <- mclapply(1:n_rep, function(r) {
  cat(sprintf("  [outlier] rep %d/%d\n", r, n_rep))
  run_one_rep(r, "outlier")
}, mc.cores = n_cores)
cat(sprintf("  Scenario 2 done: %.1f min\n", (proc.time() - t2)[3] / 60))


################################################################################
# 7. Aggregate & save
################################################################################

# 실패한 rep 제거 후 결합
res_std_ok <- Filter(Negate(is.null), res_std)
res_out_ok <- Filter(Negate(is.null), res_out)
all_df <- rbind(do.call(rbind, res_std_ok), do.call(rbind, res_out_ok))

cat(sprintf("\n=== 총 결과: %d rows (성공 rep: standard %d/%d, outlier %d/%d) ===\n",
            nrow(all_df), length(res_std_ok), n_rep, length(res_out_ok), n_rep))

# ── Summary table ──
params_order <- c("alpha", "beta1", "beta2", "beta3", "u", "delta",
                  "dist_bin", "dist_con", "dist_cnt", "dist_ord")

summary_list <- list()
for (sc in c("standard", "outlier")) {
  for (md in c("standard", "robust")) {
    for (pm in params_order) {
      sub <- all_df[all_df$scenario == sc & all_df$model == md & all_df$param == pm, ]
      if (nrow(sub) > 0) {
        summary_list[[length(summary_list) + 1]] <- data.frame(
          scenario     = sc,
          model        = md,
          param        = pm,
          mean_cov     = round(mean(sub$pct, na.rm = TRUE), 2),
          sd_cov       = round(sd(sub$pct, na.rm = TRUE), 2),
          mean_width   = round(mean(sub$mean_width, na.rm = TRUE), 4),
          n_reps       = nrow(sub),
          stringsAsFactors = FALSE
        )
      }
    }

    # OVERALL (전체 parameter 통합)
    sub_all <- all_df[all_df$scenario == sc & all_df$model == md, ]
    if (nrow(sub_all) > 0) {
      overall_by_rep <- aggregate(
        cbind(n_param, n_covered, sum_width) ~ rep,
        data = sub_all, FUN = sum
      )
      overall_by_rep$pct <- 100 * overall_by_rep$n_covered / overall_by_rep$n_param
      overall_by_rep$mw  <- overall_by_rep$sum_width / overall_by_rep$n_param

      summary_list[[length(summary_list) + 1]] <- data.frame(
        scenario   = sc,
        model      = md,
        param      = "OVERALL",
        mean_cov   = round(mean(overall_by_rep$pct), 2),
        sd_cov     = round(sd(overall_by_rep$pct), 2),
        mean_width = round(mean(overall_by_rep$mw), 4),
        n_reps     = nrow(overall_by_rep),
        stringsAsFactors = FALSE
      )
    }
  }
}
summary_df <- do.call(rbind, summary_list)

# ── Save ──
write.csv(all_df,     file.path(out_dir, "robust_comparison_raw.csv"),     row.names = FALSE)
write.csv(summary_df, file.path(out_dir, "robust_comparison_summary.csv"), row.names = FALSE)

# ── Print ──
cat("\n")
cat("================================================================\n")
cat("  Coverage Comparison: Standard LSIRM vs Robust LSIRM\n")
cat("================================================================\n\n")

for (sc in c("standard", "outlier")) {
  cat(sprintf("─── Scenario: %s ───\n", sc))
  sub <- summary_df[summary_df$scenario == sc, ]

  # Wide format: standard vs robust side by side
  std_sub <- sub[sub$model == "standard", ]
  rob_sub <- sub[sub$model == "robust", ]
  merged <- merge(std_sub, rob_sub, by = "param", suffixes = c("_std", "_rob"))

  cat(sprintf("  %-12s  %8s %8s  %8s %8s   %s\n",
              "param", "std_cov", "rob_cov", "std_wid", "rob_wid", "diff(rob-std)"))
  cat(sprintf("  %-12s  %8s %8s  %8s %8s   %s\n",
              "-----", "-------", "-------", "-------", "-------", "-------------"))

  for (i in seq_len(nrow(merged))) {
    r <- merged[i, ]
    diff_cov <- r$mean_cov_rob - r$mean_cov_std
    cat(sprintf("  %-12s  %7.2f%% %7.2f%%  %8.4f %8.4f   %+.2f%%\n",
                r$param,
                r$mean_cov_std, r$mean_cov_rob,
                r$mean_width_std, r$mean_width_rob,
                diff_cov))
  }
  cat("\n")
}

cat(sprintf("Results saved to: %s/\n", out_dir))
cat(sprintf("  - robust_comparison_raw.csv     (%d rows)\n", nrow(all_df)))
cat(sprintf("  - robust_comparison_summary.csv (%d rows)\n", nrow(summary_df)))
