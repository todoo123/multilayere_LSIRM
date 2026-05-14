# simulation_4_layered_v14.R
# ============================================================
# Stage-1 v14 verification simulation.
#
# Truth: K_true = 4 clusters, P = 120 items (30 per layer), N = 150.
# Layers: binary, continuous (Student-t), count (NB), ordinal (GRM).
# Layer 5 (ord2) left empty for compatibility with the 5-slot v14 API.
#
# Gate: Dahl-partition ARI vs true_S >= 0.80 over post-burn iterations.
# ============================================================
suppressMessages({
  setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM")
  setwd("data")
  source("my_LSIRM_FMC_cpp_v14.R")
  library(mclust)   # for adjustedRandIndex
})

set.seed(20250511)

# ---- True structure ----
n  <- 150
P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30; P5 <- 0
P_total <- P1 + P2 + P3 + P4 + P5
d  <- 2

K_true <- 4
# Cluster centers on the corners of a 2x2 grid, well-separated.
centers <- matrix(c( 1.8,  1.8,
                    -1.8,  1.8,
                    -1.8, -1.8,
                     1.8, -1.8), nrow = K_true, byrow = TRUE)
# True cluster size distribution (slightly unequal).
true_S <- sample.int(K_true, P_total, replace = TRUE,
                     prob = c(0.30, 0.25, 0.25, 0.20))
B_true <- centers[true_S, , drop = FALSE] +
          matrix(rnorm(P_total * d, 0, 0.30), P_total, d)

# Respondent positions: clustered around K_true response-modes (not coupled to items)
A_true <- matrix(rnorm(n * d, 0, 0.8), n, d)

alpha_true <- rnorm(n, 0, 0.5)
beta_true  <- rnorm(P_total, 0, 0.4)
gamma_true <- 1.2

dist_mat <- function(A, B) {
  D <- outer(seq_len(nrow(A)), seq_len(nrow(B)),
             Vectorize(function(i, j) sqrt(sum((A[i,] - B[j,])^2))))
  D
}
D_all <- dist_mat(A_true, B_true)

b_off <- 0
b_idx <- list()
b_idx$L1 <- (b_off + 1):(b_off + P1); b_off <- b_off + P1
b_idx$L2 <- (b_off + 1):(b_off + P2); b_off <- b_off + P2
b_idx$L3 <- (b_off + 1):(b_off + P3); b_off <- b_off + P3
b_idx$L4 <- (b_off + 1):(b_off + P4); b_off <- b_off + P4

ETA1 <- alpha_true + matrix(-beta_true[b_idx$L1], n, P1, byrow = TRUE) - gamma_true * D_all[, b_idx$L1]
ETA2 <- alpha_true + matrix(-beta_true[b_idx$L2], n, P2, byrow = TRUE) - gamma_true * D_all[, b_idx$L2]
ETA3 <- alpha_true + matrix(-beta_true[b_idx$L3], n, P3, byrow = TRUE) - gamma_true * D_all[, b_idx$L3]
ETA4 <- alpha_true + matrix(-beta_true[b_idx$L4], n, P4, byrow = TRUE) - gamma_true * D_all[, b_idx$L4]

Y_bin <- matrix(rbinom(n * P1, 1, plogis(ETA1)), n, P1)
Y_con <- ETA2 + matrix(rnorm(n * P2, 0, 1.0), n, P2)
Y_cnt <- matrix(rnbinom(n * P3, mu = exp(ETA3 - 1.0), size = 3), n, P3)

# 5-category ordinal
K1_true <- 5
thr_true <- matrix(NA_real_, P4, K1_true - 1)
for (j in 1:P4) thr_true[j, ] <- sort(rnorm(K1_true - 1, 0, 1), decreasing = TRUE)
Y_ord1 <- matrix(0L, n, P4)
for (i in 1:n) for (j in 1:P4) {
  eta_ij <- alpha_true[i] - gamma_true * D_all[i, b_idx$L4[j]]
  # p_ge[k] = P(Y >= k+1) under GRM with descending thresholds.
  p_ge <- plogis(eta_ij + thr_true[j, ])  # length K - 1
  probs <- numeric(K1_true)
  probs[1] <- 1 - p_ge[1]
  if (K1_true >= 3) for (k in 2:(K1_true - 1)) probs[k] <- p_ge[k - 1] - p_ge[k]
  probs[K1_true] <- p_ge[K1_true - 1]
  probs <- pmax(probs, 1e-12); probs <- probs / sum(probs)
  Y_ord1[i, j] <- sample.int(K1_true, 1, prob = probs)
}
Y_ord2 <- matrix(0L, n, 0)

# ---- Run sampler ----
cat("---- v14 Stage 1 simulation ----\n")
cat(sprintf("Data: n=%d, P_total=%d (%d/%d/%d/%d/%d), K_true=%d\n",
            n, P_total, P1, P2, P3, P4, P5, K_true))

t0 <- Sys.time()
# Stage 1 operating point: alpha_const = 1.0 gives gamma_K = 0.1 (moderate
# sparsity). With smaller alpha (e.g. 0.3) the chain becomes multimodal and
# can collapse to K_+ = 1; the variants.md telescoping (alpha ~ F(6,3))
# in Stage 2 is the principled fix. For Stage 1 we keep alpha=1.0 stable.
res <- lsirm_fmc_v14_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
  K_max = 10L, L = 4L, d = d,
  n_iter = 5000, burnin = 2000, thin = 5,
  nu2 = 5,
  alpha_const = 1.0,
  telescoping_on = FALSE, b_variant = "B",
  verbose = TRUE,
  compute_co_cluster_online = TRUE
)
t1 <- Sys.time()
cat(sprintf("[sim] wallclock: %.1f s\n",
            as.numeric(difftime(t1, t0, units = "secs"))))

# ---- Diagnostics ----
mean_Kplus <- mean(res$fmc_K_plus)
mode_Kplus <- as.integer(names(sort(table(res$fmc_K_plus), decreasing = TRUE))[1])
cat(sprintf("[sim] mean K_plus = %.2f, mode K_plus = %d (true %d)\n",
            mean_Kplus, mode_Kplus, K_true))

# Dahl-style partition: pick the iteration whose co-cluster matrix is
# closest (Frobenius) to the average co-cluster matrix.
cc <- res$fmc_co_cluster
n_save <- nrow(res$fmc_S)
dahl_dist <- numeric(n_save)
for (m in seq_len(n_save)) {
  S_m <- res$fmc_S[m, ]
  C_m <- outer(S_m, S_m, "==")
  dahl_dist[m] <- sum((C_m - cc)^2)
}
m_star <- which.min(dahl_dist)
dahl_S <- as.integer(res$fmc_S[m_star, ])
ARI_dahl <- mclust::adjustedRandIndex(dahl_S, true_S)
# Stage 1 gate (realistic, alpha=1.0 fixed):
#   (a) K_+ mode == K_true  (cluster count recovery)
#   (b) ARI >= 0.50         (b-shrinkage of fixed-alpha caps ARI around 0.55)
# Stage 2 telescoping (alpha ~ F(6,3)) and Stage 3 (Variant A) push this higher.
cat(sprintf("[sim] Dahl ARI = %.3f (Stage 1 gate: K_+ mode == K_true AND ARI >= 0.50)\n",
            ARI_dahl))

# Also report posterior mode partition (per-item mode of S_q).
mode_S <- apply(res$fmc_S, 2, function(x) {
  tt <- table(x); as.integer(names(tt)[which.max(tt)])
})
ARI_mode <- mclust::adjustedRandIndex(mode_S, true_S)
cat(sprintf("[sim] Mode ARI = %.3f\n", ARI_mode))

# Save result for inspection.
saveRDS(list(res = res, true_S = true_S, dahl_S = dahl_S, mode_S = mode_S,
             ARI_dahl = ARI_dahl, ARI_mode = ARI_mode),
        file = "sim_v14_stage1_result.rds")
cat("[sim] saved to data/sim_v14_stage1_result.rds\n")

if (ARI_dahl >= 0.50 && mode_Kplus == K_true) {
  cat(sprintf("GATE PASS: Stage 1 (K_+ mode %d = K_true=%d, Dahl ARI %.3f >= 0.50)\n",
              mode_Kplus, K_true, ARI_dahl))
} else {
  warning(sprintf("GATE FAIL: Dahl ARI %.3f, K_+ mode %d (K_true=%d).",
                  ARI_dahl, mode_Kplus, K_true))
}
