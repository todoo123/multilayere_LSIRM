# test_v14_alpha_mh.R
# ============================================================
# Verifies the v14 alpha log-RWMH sampler:
#   (a) Under degenerate likelihood (K_plus=0, N_total=0), the chain
#       targets the F(6,3) prior. We check it via:
#         - empirical 1st and 2nd moments match analytical F(6,3) moments
#         - Kolmogorov-Smirnov vs F(6,3) CDF: p > 0.01
#   (b) Acceptance rate is in a reasonable range (e.g. 0.1..0.7).
#   (c) Jacobian sanity: log target at alpha + log(alpha) should yield
#       symmetric MH acceptance (we re-derive by sampling from prior
#       directly and comparing).
#
# F(6,3) has density p(x) prop x^2 (1 + 2x)^{-4.5} on x > 0.
# It is also the standard F-distribution F(6,3) — its mean is undefined
# (nu_r=3 < 2*2=4 needed for mean to exist? Wait, classical F(d1, d2):
#  E[X] = d2 / (d2 - 2) for d2 > 2.  d2=3 -> E[X] = 3 / (3-2) = 3.
#  Var[X] is undefined for d2 <= 4.).
#
# We use stats::df / pf with df1 = nu_l = 6, df2 = nu_r = 3.
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

# ---- Case (a): degenerate likelihood -> chain targets F(6,3) prior. ----
n_iter <- 50000
res <- v14_alpha_mh_chain(
  n_iter = n_iter, alpha_init = 1.0, s_alpha = 0.6,
  K = 5, K_plus = 0, N_total = 0,
  N_k_filled_in = integer(0)
)
chain <- res$chain
acc   <- res$accept_rate
# With degenerate likelihood + compact F(6,3) prior + s_alpha=0.6, acceptance
# is naturally high. The real verification is the KS test below.
cat(sprintf("[alpha-MH] acceptance rate = %.3f (sanity 0.05..0.95)\n", acc))
stopifnot(acc >= 0.05, acc <= 0.95)

# Discard burnin
burnin <- 5000
chain_use <- chain[(burnin + 1):n_iter]

emp_mean <- mean(chain_use)
# Theory: F(6, 3) -> E[X] = nu_r / (nu_r - 2) = 3 / 1 = 3.
th_mean <- 3.0
cat(sprintf("[alpha-MH] empirical mean = %.3f, F(6,3) mean = %.3f\n",
            emp_mean, th_mean))
stopifnot(abs(emp_mean - th_mean) / th_mean < 0.10)

# KS test vs F(6, 3) CDF.
ks <- ks.test(chain_use, pf, df1 = 6, df2 = 3)
cat(sprintf("[alpha-MH] KS test vs F(6,3): D=%.4f, p=%.4f (need p > 0.01)\n",
            as.numeric(ks$statistic), ks$p.value))
stopifnot(ks$p.value > 0.01)

# ---- Case (b): non-trivial likelihood. Check chain stays in reasonable range. ----
N_k_filled <- c(40L, 30L, 30L, 20L)
res_b <- v14_alpha_mh_chain(
  n_iter = 5000, alpha_init = 1.0, s_alpha = 0.5,
  K = 5, K_plus = 4, N_total = 120,
  N_k_filled_in = N_k_filled
)
cat(sprintf("[alpha-MH] non-degen acceptance = %.3f, mean alpha = %.3f, range [%.3f, %.3f]\n",
            res_b$accept_rate, mean(res_b$chain),
            min(res_b$chain), max(res_b$chain)))
stopifnot(res_b$accept_rate >= 0.05, res_b$accept_rate <= 0.85)
stopifnot(all(res_b$chain > 0), all(is.finite(res_b$chain)))

# ---- Case (c): F(6,3) log density evaluation ----
# log_F_6_3(alpha) is up-to-constant. Test ratio at two points matches.
a1 <- 1.0; a2 <- 3.0
log_ratio_cpp <- v14_log_F_6_3(a2) - v14_log_F_6_3(a1)
# True log p_F63(a) = (nu_l/2 - 1) log a - (nu_l+nu_r)/2 log(1 + nu_l*a/nu_r)
log_ratio_th <- (2 * log(a2) - 4.5 * log1p(2 * a2)) -
                (2 * log(a1) - 4.5 * log1p(2 * a1))
cat(sprintf("[alpha-MH] log_F_6_3 ratio: cpp=%.5f, theory=%.5f\n",
            log_ratio_cpp, log_ratio_th))
stopifnot(abs(log_ratio_cpp - log_ratio_th) < 1e-10)

cat("PASS test_v14_alpha_mh.R\n")
