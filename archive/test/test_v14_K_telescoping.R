# test_v14_K_telescoping.R
# ============================================================
# Verifies the v14 K-telescoping sampler (variants.md Eq 3.8) reduces
# to the BNB(1,4,3) prior when there are no filled clusters
# (K_plus = 0, N_k_filled = empty). In that case Eq 3.8 collapses to:
#   p(K | C, alpha) prop p_BNB(K)
# (every other factor is 1).
#
# Also exercises the case K_plus > 0 with a fixed partition and verifies
# that the empirical histogram matches the analytical formula
# (proportionality on a small grid).
# ============================================================
suppressMessages({
  if (basename(getwd()) != "joint_LSIRM") {
    candidate <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"
    if (dir.exists(candidate)) setwd(candidate)
  }
  setwd(file.path(getwd(), "data"))
  library(Rcpp)
  library(RcppArmadillo)
  sourceCpp("my_LSIRM_FMC_v14.cpp")
})

set.seed(20250511)

# Analytic BNB(1, 4, 3) pmf on K = 1..K_max.
bnb_143 <- function(K_max) {
  log_pmf <- sapply(seq_len(K_max), function(K) {
    # log B(5, K+2) - log B(4, 3); B = beta function.
    lgamma(5) + lgamma(K + 2) - lgamma(K + 7) -
      (lgamma(4) + lgamma(3) - lgamma(7))
  })
  # Truncate to K_max and renormalize.
  pmf <- exp(log_pmf - max(log_pmf))
  pmf / sum(pmf)
}

# ---- Case 1: K_plus = 0 (no filled clusters) -> reduces to BNB prior. ----
K_max <- 30
n_iter <- 50000
samples_empty <- v14_K_telescope_sample(
  n_iter = n_iter, alpha = 1.0, K_plus = 0, K_max = K_max,
  N_k_filled_in = integer(0)
)
emp1 <- as.numeric(table(factor(samples_empty, levels = 1:K_max))) / n_iter
th1  <- bnb_143(K_max)
tv1  <- 0.5 * sum(abs(emp1 - th1))
cat(sprintf("[K-tele] K_plus=0  total-var(empirical, BNB prior)=%.4f (tol 0.02)\n", tv1))
stopifnot(tv1 < 0.02)

# ---- Case 2: K_plus > 0, fixed partition. Verify chain hits valid support. ----
K_plus <- 3
N_k_filled <- c(40L, 30L, 30L)
samples_fill <- v14_K_telescope_sample(
  n_iter = n_iter, alpha = 1.0, K_plus = K_plus, K_max = K_max,
  N_k_filled_in = N_k_filled
)
# Support must be [K_plus, K_max].
stopifnot(min(samples_fill) >= K_plus, max(samples_fill) <= K_max)

# Compute analytical formula on the grid and check empirical proportions.
log_pK <- sapply(K_plus:K_max, function(K) {
  alpha <- 1.0
  gK <- alpha / K
  s <- lgamma(5) + lgamma(K + 2) - lgamma(K + 7)
  s <- s + K_plus * log(alpha) - K_plus * log(K)
  s <- s + lgamma(K + 1) - lgamma(K - K_plus + 1)
  s <- s + sum(lgamma(N_k_filled + gK) - lgamma(1 + gK))
  s
})
pK_th <- exp(log_pK - max(log_pK)); pK_th <- pK_th / sum(pK_th)
emp2 <- as.numeric(table(factor(samples_fill, levels = K_plus:K_max))) / n_iter
tv2  <- 0.5 * sum(abs(emp2 - pK_th))
cat(sprintf("[K-tele] K_plus=3, fixed N: total-var(empirical, theory)=%.4f (tol 0.02)\n", tv2))
stopifnot(tv2 < 0.02)

# ---- Case 3: alpha-dependence at fixed K_plus, N.
#   Analytic argument (variants.md derivation): for large alpha with K_plus
#   and N fixed, the product term Pi Gamma(N_k + a/K)/Gamma(1 + a/K) -> (a/K)^{N - K_plus}.
#   Combined with the alpha^{K_plus}/K^{K_plus} factor, the K-dependence at
#   large alpha scales like 1/K^N, which pushes K toward K_start.
#   Small alpha lets K_+ -> BNB(1,4,3) approximately for the K-K_plus extra slots.
#   So large alpha -> SMALLER K (not larger).
samples_small_a <- v14_K_telescope_sample(
  n_iter = n_iter, alpha = 0.1, K_plus = K_plus, K_max = K_max,
  N_k_filled_in = N_k_filled
)
samples_large_a <- v14_K_telescope_sample(
  n_iter = n_iter, alpha = 5.0, K_plus = K_plus, K_max = K_max,
  N_k_filled_in = N_k_filled
)
cat(sprintf("[K-tele] mean K(alpha=0.1) = %.2f vs mean K(alpha=5.0) = %.2f\n",
            mean(samples_small_a), mean(samples_large_a)))
stopifnot(mean(samples_small_a) > mean(samples_large_a))

cat("PASS test_v14_K_telescoping.R\n")
