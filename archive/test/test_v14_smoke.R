# test_v14_smoke.R
# ============================================================
# Smoke test for the v14 Stage-1 sampler.
# Target: full sweep finishes in < 30 s on a small (n=30, P=24) problem.
#
# Verifies:
#   - cpp compiles via sourceCpp() through the R wrapper
#   - run completes without error
#   - output shapes match expectations
#   - posterior K_plus is sensible (>= 1, <= K_max)
#   - posterior co-cluster matrix has values in [0, 1]
# ============================================================
suppressMessages({
  if (basename(getwd()) != "joint_LSIRM") {
    candidate <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"
    if (dir.exists(candidate)) setwd(candidate)
  }
  setwd(file.path(getwd(), "data"))
  source("my_LSIRM_FMC_cpp_v14.R")
})

set.seed(20250511)

# ---- Small synthetic data ----
n  <- 30
P1 <- 8; P2 <- 8; P3 <- 4; P4 <- 4; P5 <- 0   # K2m1 = 0 if P5 = 0
P_total <- P1 + P2 + P3 + P4 + P5
d <- 2

# True item clusters (K_true = 2)
K_true <- 2
true_S <- sample.int(K_true, P_total, replace = TRUE)
centers <- matrix(c(-1.5, -1.5,
                     1.5,  1.5), nrow = K_true, byrow = TRUE)
B_true <- centers[true_S, , drop = FALSE] +
          matrix(rnorm(P_total * d, 0, 0.3), P_total, d)
A_true <- matrix(rnorm(n * d, 0, 0.5), n, d)

alpha_true <- rnorm(n, 0, 0.5)
beta_true  <- rnorm(P_total, 0, 0.5)
gamma_true <- 1.5

# Distances and linear predictors
dist_mat <- function(A, B) as.matrix(dist(rbind(A, B)))[1:nrow(A), (nrow(A)+1):(nrow(A)+nrow(B))]

D_all <- dist_mat(A_true, B_true)
# Partition into layers
b_off <- 0
b_idx <- list(L1 = 1:P1)
b_off <- b_off + P1
b_idx$L2 <- (b_off + 1):(b_off + P2); b_off <- b_off + P2
b_idx$L3 <- (b_off + 1):(b_off + P3); b_off <- b_off + P3
b_idx$L4 <- (b_off + 1):(b_off + P4); b_off <- b_off + P4

ETA1 <- alpha_true + matrix(-beta_true[b_idx$L1], n, P1, byrow = TRUE) - gamma_true * D_all[, b_idx$L1]
ETA2 <- alpha_true + matrix(-beta_true[b_idx$L2], n, P2, byrow = TRUE) - gamma_true * D_all[, b_idx$L2]
ETA3 <- alpha_true + matrix(-beta_true[b_idx$L3], n, P3, byrow = TRUE) - gamma_true * D_all[, b_idx$L3]
ETA4 <- alpha_true + matrix(-beta_true[b_idx$L4], n, P4, byrow = TRUE) - gamma_true * D_all[, b_idx$L4]

Y_bin <- matrix(rbinom(n * P1, 1, plogis(ETA1)), n, P1)
Y_con <- ETA2 + matrix(rnorm(n * P2, 0, 1.0), n, P2)
Y_cnt <- matrix(rnbinom(n * P3, mu = exp(ETA3), size = 5), n, P3)

# 3-category ordinal via DESCENDING thresholds (GRM convention used by cpp).
thr_mat <- matrix(c(1, -1), nrow = P4, ncol = 2, byrow = TRUE)  # beta_j(0) > beta_j(1)
Y_ord1 <- matrix(0L, n, P4)
for (i in 1:n) for (j in 1:P4) {
  eta_ij <- alpha_true[i] - gamma_true * D_all[i, b_idx$L4[j]]
  p_ge1 <- plogis(eta_ij + thr_mat[j, 1])
  p_ge2 <- plogis(eta_ij + thr_mat[j, 2])
  pr1 <- 1 - p_ge1
  pr2 <- p_ge1 - p_ge2
  pr3 <- p_ge2
  pr1 <- max(pr1, 0); pr2 <- max(pr2, 0); pr3 <- max(pr3, 0)
  Y_ord1[i, j] <- sample.int(3, 1, prob = c(pr1, pr2, pr3))
}

Y_ord2 <- matrix(0L, n, 0)  # empty

# ---- Run sampler ----
t0 <- Sys.time()
res <- lsirm_fmc_v14_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
  K_max = 5L, L = 3L, d = d,
  n_iter = 400, burnin = 200, thin = 1,
  nu2 = 5,
  alpha_const = 1.0,
  telescoping_on = FALSE, b_variant = "B",
  verbose = FALSE,
  compute_co_cluster_online = TRUE
)
t1 <- Sys.time()
dt <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("[smoke] wallclock: %.2f s\n", dt))
if (dt > 60) {
  warning("Smoke test exceeded 60 s target (>30 s soft target). Investigate.")
}

# ---- Shape sanity ----
n_save <- 200
stopifnot(dim(res$a)[1]    == n_save)
stopifnot(dim(res$a)[2:3]  == c(n, d))
stopifnot(dim(res$fmc_S)   == c(n_save, P_total))
stopifnot(length(res$fmc_K_plus) == n_save)
stopifnot(dim(res$fmc_eta_K) == c(n_save, 5))         # K_max = 5
stopifnot(dim(res$fmc_w_k)   == c(5, 3, n_save))       # K_max x L x n_save
stopifnot(dim(res$fmc_mu_kl) == c(5 * 3, d, n_save))   # (K_max*L) x d x n_save
stopifnot(dim(res$fmc_Sigma_kl) == c(d, d, 5 * 3, n_save))
stopifnot(dim(res$fmc_b_0k)  == c(5, d, n_save))
stopifnot(dim(res$fmc_C_0k)  == c(d, d, 5, n_save))
stopifnot(dim(res$fmc_Lambda_k) == c(5, d, n_save))
cat("[smoke] all output shapes OK\n")

# ---- Posterior plausibility ----
mean_Kplus <- mean(res$fmc_K_plus)
stopifnot(mean_Kplus >= 1, mean_Kplus <= 5)
cat(sprintf("[smoke] mean K_plus = %.2f (expected 1..5)\n", mean_Kplus))

# Co-cluster values must be in [0, 1].
cc <- res$fmc_co_cluster
stopifnot(!is.null(cc), all(cc >= 0), all(cc <= 1 + 1e-8))
cat(sprintf("[smoke] co_cluster matrix: %d x %d, range [%.3f, %.3f]\n",
            nrow(cc), ncol(cc), min(cc), max(cc)))

# Acceptance rates exist
stopifnot(!is.null(res$accept), !is.null(res$accept$b1))
cat(sprintf("[smoke] mean accept b1=%.3f, b2=%.3f, b3=%.3f, b4=%.3f\n",
            mean(res$accept$b1), mean(res$accept$b2),
            mean(res$accept$b3), mean(res$accept$b4)))

# Mu_kl posterior means shouldn't be wildly off-scale.
mu_kl_pm <- apply(res$fmc_mu_kl, c(1, 2), mean)
stopifnot(all(abs(mu_kl_pm) < 20))
cat(sprintf("[smoke] mu_kl postmean range: [%.2f, %.2f]\n",
            min(mu_kl_pm), max(mu_kl_pm)))

# Lambda_k must be positive.
stopifnot(all(res$fmc_Lambda_k > 0))
cat(sprintf("[smoke] Lambda_k range: [%.3f, %.3f]\n",
            min(res$fmc_Lambda_k), max(res$fmc_Lambda_k)))

cat("PASS test_v14_smoke.R\n")
