# benchmark_v14_variantA_vs_B.R
# ============================================================
# Stage-3 verification: Variant A (full collapse) vs Variant B (partial).
#
# Same seed, same data, same iterations. Compare:
#   (a) b_j ESS averaged over all P_total * d coordinates
#   (b) wallclock
# Gate (per plan): adopt Variant A iff
#   * b_j ESS(A) >= 1.3 * b_j ESS(B), AND
#   * wallclock(A) <= 2 * wallclock(B)
# else default stays "B".
#
# Reuses the Stage 2 simulation data (K_true=4, P=120, N=150,
# telescoping_on=TRUE) so the benchmark exercises the full v14 pipeline.
# ============================================================
suppressMessages({
  setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM")
  setwd("data")
  source("my_LSIRM_FMC_cpp_v14.R")
  library(coda)
  library(mclust)
})

# ---- Build the same data as simulation_4_layered_v14_stage2.R ----
make_data <- function() {
  set.seed(20250511)
  n  <- 150
  P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30; P5 <- 0
  P_total <- P1 + P2 + P3 + P4 + P5
  d  <- 2
  K_true <- 4
  centers <- matrix(c( 1.8,  1.8, -1.8,  1.8, -1.8, -1.8,  1.8, -1.8),
                    nrow = K_true, byrow = TRUE)
  true_S <- sample.int(K_true, P_total, replace = TRUE,
                       prob = c(0.30, 0.25, 0.25, 0.20))
  B_true <- centers[true_S, , drop = FALSE] +
            matrix(rnorm(P_total * d, 0, 0.30), P_total, d)
  A_true <- matrix(rnorm(n * d, 0, 0.8), n, d)
  alpha_true <- rnorm(n, 0, 0.5)
  beta_true  <- rnorm(P_total, 0, 0.4)
  gamma_true <- 1.2
  D_all <- outer(seq_len(n), seq_len(P_total),
                 Vectorize(function(i, j) sqrt(sum((A_true[i,] - B_true[j,])^2))))
  b_off <- 0
  b_idx <- list()
  b_idx$L1 <- (b_off + 1):(b_off + P1); b_off <- b_off + P1
  b_idx$L2 <- (b_off + 1):(b_off + P2); b_off <- b_off + P2
  b_idx$L3 <- (b_off + 1):(b_off + P3); b_off <- b_off + P3
  b_idx$L4 <- (b_off + 1):(b_off + P4); b_off <- b_off + P4
  ETA1 <- alpha_true + matrix(-beta_true[b_idx$L1], n, P1, byrow = TRUE) -
          gamma_true * D_all[, b_idx$L1]
  ETA2 <- alpha_true + matrix(-beta_true[b_idx$L2], n, P2, byrow = TRUE) -
          gamma_true * D_all[, b_idx$L2]
  ETA3 <- alpha_true + matrix(-beta_true[b_idx$L3], n, P3, byrow = TRUE) -
          gamma_true * D_all[, b_idx$L3]
  ETA4 <- alpha_true + matrix(-beta_true[b_idx$L4], n, P4, byrow = TRUE) -
          gamma_true * D_all[, b_idx$L4]
  Y_bin <- matrix(rbinom(n * P1, 1, plogis(ETA1)), n, P1)
  Y_con <- ETA2 + matrix(rnorm(n * P2, 0, 1.0), n, P2)
  Y_cnt <- matrix(rnbinom(n * P3, mu = exp(ETA3 - 1.0), size = 3), n, P3)
  K1_true <- 5
  thr_true <- matrix(NA_real_, P4, K1_true - 1)
  for (j in 1:P4) thr_true[j, ] <- sort(rnorm(K1_true - 1, 0, 1),
                                         decreasing = TRUE)
  Y_ord1 <- matrix(0L, n, P4)
  for (i in 1:n) for (j in 1:P4) {
    eta_ij <- alpha_true[i] - gamma_true * D_all[i, b_idx$L4[j]]
    p_ge <- plogis(eta_ij + thr_true[j, ])
    probs <- numeric(K1_true)
    probs[1] <- 1 - p_ge[1]
    if (K1_true >= 3) for (k in 2:(K1_true - 1)) probs[k] <- p_ge[k - 1] - p_ge[k]
    probs[K1_true] <- p_ge[K1_true - 1]
    probs <- pmax(probs, 1e-12); probs <- probs / sum(probs)
    Y_ord1[i, j] <- sample.int(K1_true, 1, prob = probs)
  }
  Y_ord2 <- matrix(0L, n, 0)
  list(Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt,
       Y_ord1 = Y_ord1, Y_ord2 = Y_ord2,
       true_S = true_S, d = d, K_true = K_true)
}

data <- make_data()
true_S <- data$true_S
d <- data$d
K_true <- data$K_true

# ---- Run both variants with identical RNG state ----
run_chain <- function(variant_str, seed = 20250512) {
  set.seed(seed)
  t0 <- Sys.time()
  res <- lsirm_fmc_v14_cpp(
    data$Y_bin, data$Y_con, data$Y_cnt, data$Y_ord1, data$Y_ord2,
    K_max = 30L, L = 4L, d = d,
    n_iter = 10000, burnin = 5000, thin = 5,
    nu2 = 5,
    telescoping_on = TRUE,
    alpha_init     = 1.0,
    s_alpha        = 0.8,
    b_variant      = variant_str,
    verbose = FALSE,
    compute_co_cluster_online = TRUE
  )
  walltime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  list(res = res, walltime = walltime)
}

# Aggregate b_j coordinate-wise ESS (averaged across P_total * d coords).
b_ess_summary <- function(res) {
  # res$b1..b5 are (n_save, P_l, d) after the wrapper's aperm.
  collect <- function(arr) {
    if (length(dim(arr)) != 3) return(numeric(0))
    P_l <- dim(arr)[2]
    d_l <- dim(arr)[3]
    ess_vals <- numeric(P_l * d_l)
    idx <- 1
    for (j in seq_len(P_l)) for (k in seq_len(d_l)) {
      ess_vals[idx] <- as.numeric(coda::effectiveSize(arr[, j, k]))
      idx <- idx + 1
    }
    ess_vals
  }
  ess_all <- c(collect(res$b1), collect(res$b2), collect(res$b3),
               collect(res$b4), collect(res$b5))
  ess_all <- ess_all[is.finite(ess_all) & ess_all > 0]
  list(mean = mean(ess_all), median = median(ess_all),
       min = min(ess_all), max = max(ess_all), n = length(ess_all))
}

dahl_ari <- function(res, true_S) {
  cc <- res$fmc_co_cluster
  n_save <- nrow(res$fmc_S)
  dahl_dist <- sapply(seq_len(n_save), function(m) {
    Sm <- res$fmc_S[m, ]
    sum((outer(Sm, Sm, "==") - cc)^2)
  })
  m_star <- which.min(dahl_dist)
  mclust::adjustedRandIndex(as.integer(res$fmc_S[m_star, ]), true_S)
}

cat("==== Benchmark: Variant B (Stage 2 default) ====\n")
runB <- run_chain("B")
cat(sprintf("  wallclock: %.1f s\n", runB$walltime))
essB <- b_ess_summary(runB$res)
cat(sprintf("  b_j ESS: mean=%.1f, median=%.1f, range=[%.1f, %.1f] (n=%d coords)\n",
            essB$mean, essB$median, essB$min, essB$max, essB$n))
ariB <- dahl_ari(runB$res, true_S)
cat(sprintf("  Dahl ARI: %.3f\n", ariB))
mode_KpB <- as.integer(names(sort(table(runB$res$fmc_K_plus), decreasing = TRUE))[1])
cat(sprintf("  K_+ mode: %d  (true %d)\n", mode_KpB, K_true))

cat("\n==== Benchmark: Variant A (Stage 3 candidate) ====\n")
runA <- run_chain("A")
cat(sprintf("  wallclock: %.1f s\n", runA$walltime))
essA <- b_ess_summary(runA$res)
cat(sprintf("  b_j ESS: mean=%.1f, median=%.1f, range=[%.1f, %.1f] (n=%d coords)\n",
            essA$mean, essA$median, essA$min, essA$max, essA$n))
ariA <- dahl_ari(runA$res, true_S)
cat(sprintf("  Dahl ARI: %.3f\n", ariA))
mode_KpA <- as.integer(names(sort(table(runA$res$fmc_K_plus), decreasing = TRUE))[1])
cat(sprintf("  K_+ mode: %d  (true %d)\n", mode_KpA, K_true))

cat("\n==== Gate ====\n")
ess_ratio <- essA$mean / essB$mean
wall_ratio <- runA$walltime / runB$walltime
cat(sprintf("  ESS(A) / ESS(B) = %.3f  (gate >= 1.30)\n", ess_ratio))
cat(sprintf("  wallclock(A) / wallclock(B) = %.3f  (gate <= 2.00)\n", wall_ratio))
cat(sprintf("  ARI improvement: A - B = %.3f\n", ariA - ariB))

gate_pass <- (ess_ratio >= 1.30) && (wall_ratio <= 2.00)
if (gate_pass) {
  cat("GATE PASS: Adopt Variant A as default.\n")
} else {
  cat("GATE FAIL: Keep Variant B as default (informational benchmark).\n")
}

saveRDS(list(runA = runA, runB = runB, essA = essA, essB = essB,
             ariA = ariA, ariB = ariB,
             ess_ratio = ess_ratio, wall_ratio = wall_ratio,
             gate_pass = gate_pass),
        file = "benchmark_v14_AB.rds")
cat("saved benchmark_v14_AB.rds\n")
