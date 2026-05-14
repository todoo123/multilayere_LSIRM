# simulation_4_layered_v14_stage2.R
# ============================================================
# Stage-2 v14 verification simulation (telescoping_on = TRUE).
#
# Same synthetic data as simulation_4_layered_v14.R but with the
# telescoping sampler (K, alpha dynamic).
#
# Gates:
#   (a) K_+ mode in {3, 4, 5}  (K_true = 4)
#   (b) alpha chain ESS >= 100 over post-burnin iterations
#   (c) K trace hits K_max less than 1% of iterations
#   (d) Dahl ARI vs true partition >= 0.55 (compared to Stage 1's 0.538;
#       telescoping is expected to match or improve)
# ============================================================
suppressMessages({
  setwd("/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM")
  setwd("data")
  source("my_LSIRM_FMC_cpp_v14.R")
  library(mclust)
  library(coda)
})

set.seed(20250511)

# ---- Truth (identical to Stage 1 simulation) ----
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

dist_mat <- function(A, B) {
  outer(seq_len(nrow(A)), seq_len(nrow(B)),
        Vectorize(function(i, j) sqrt(sum((A[i,] - B[j,])^2))))
}
D_all <- dist_mat(A_true, B_true)

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

# ---- Run Stage 2 sampler ----
cat("---- v14 Stage 2 simulation (telescoping on) ----\n")
cat(sprintf("Data: n=%d, P_total=%d (%d/%d/%d/%d/%d), K_true=%d\n",
            n, P_total, P1, P2, P3, P4, P5, K_true))

t0 <- Sys.time()
res <- lsirm_fmc_v14_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
  K_max = 30L, L = 4L, d = d,
  n_iter = 10000, burnin = 5000, thin = 5,
  nu2 = 5,
  telescoping_on = TRUE,
  alpha_init     = 1.0,
  s_alpha        = 0.8,   # larger step for better mixing (target acc ~0.5)
  b_variant      = "B",
  verbose = TRUE,
  compute_co_cluster_online = TRUE
)
t1 <- Sys.time()
cat(sprintf("[sim] wallclock: %.1f s\n",
            as.numeric(difftime(t1, t0, units = "secs"))))

# ---- Diagnostics ----
mean_Kplus <- mean(res$fmc_K_plus)
mode_Kplus <- as.integer(names(sort(table(res$fmc_K_plus),
                                    decreasing = TRUE))[1])
cat(sprintf("[sim] K_+ chain: min=%d  mode=%d  mean=%.2f  max=%d (true %d)\n",
            as.integer(min(res$fmc_K_plus)), mode_Kplus, mean_Kplus,
            as.integer(max(res$fmc_K_plus)), K_true))

K_trace <- res$fmc_K
cat(sprintf("[sim] K chain: min=%d  mode=%d  mean=%.2f  max=%d (K_max=%d)\n",
            as.integer(min(K_trace)),
            as.integer(names(sort(table(K_trace), decreasing = TRUE))[1]),
            mean(K_trace), as.integer(max(K_trace)), 30L))
hit_Kmax_rate <- mean(K_trace == 30)
cat(sprintf("[sim] K hits K_max rate: %.4f (gate: < 0.01)\n", hit_Kmax_rate))

alpha_chain <- res$fmc_alpha
alpha_ESS <- coda::effectiveSize(alpha_chain)
cat(sprintf("[sim] alpha chain: mean=%.3f  range=[%.3f, %.3f]  ESS=%.1f (gate: >= 100)\n",
            mean(alpha_chain), min(alpha_chain), max(alpha_chain), alpha_ESS))
cat(sprintf("[sim] alpha MH acceptance rate: %.3f\n",
            res$fmc_alpha_mh_accept_rate))

# Dahl partition.
cc <- res$fmc_co_cluster
n_save <- nrow(res$fmc_S)
dahl_dist <- sapply(seq_len(n_save), function(m) {
  Sm <- res$fmc_S[m, ]
  Cm <- outer(Sm, Sm, "==")
  sum((Cm - cc)^2)
})
m_star <- which.min(dahl_dist)
dahl_S <- as.integer(res$fmc_S[m_star, ])
ARI_dahl <- mclust::adjustedRandIndex(dahl_S, true_S)
cat(sprintf("[sim] Dahl ARI = %.3f (gate: >= 0.55)\n", ARI_dahl))

mode_S <- apply(res$fmc_S, 2, function(x) {
  tt <- table(x); as.integer(names(tt)[which.max(tt)])
})
ARI_mode <- mclust::adjustedRandIndex(mode_S, true_S)
cat(sprintf("[sim] Mode ARI = %.3f\n", ARI_mode))

saveRDS(list(res = res, true_S = true_S, dahl_S = dahl_S, mode_S = mode_S,
             ARI_dahl = ARI_dahl, ARI_mode = ARI_mode),
        file = "sim_v14_stage2_result.rds")
cat("[sim] saved to data/sim_v14_stage2_result.rds\n")

gate_pass <- (
  mode_Kplus %in% (K_true - 1):(K_true + 1) &&
  alpha_ESS >= 100 &&
  hit_Kmax_rate < 0.01 &&
  ARI_dahl >= 0.55
)
if (gate_pass) {
  cat("GATE PASS: Stage 2 telescoping\n")
} else {
  warning(sprintf(
    "GATE FAIL: K_+ mode=%d (need %d-%d), alpha ESS=%.1f, K_max hit=%.4f, Dahl ARI=%.3f.",
    mode_Kplus, K_true - 1, K_true + 1, alpha_ESS, hit_Kmax_rate, ARI_dahl
  ))
}
