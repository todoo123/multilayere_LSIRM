# test_v14_gig.R
# ============================================================
# Verifies the v14_gig_sanity wiring through GIGrvg::rgig.
#
# v14_gig_sanity(n, lambda, chi, psi) draws n samples from GIG with
# variants.md notation GIG(p, a, b) = GIG(lambda=p, chi=b, psi=a)
# (density prop x^{p-1} exp(-0.5 (a x + b/x))).
#
# Mean: E[X] = sqrt(chi/psi) * K_{lambda+1}(sqrt(chi*psi)) / K_lambda(sqrt(chi*psi)).
# Variance and higher moments via similar Bessel-K ratios.
# ============================================================
suppressMessages({
  if (basename(getwd()) != "joint_LSIRM") {
    candidate <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"
    if (dir.exists(candidate)) setwd(candidate)
  }
  setwd(file.path(getwd(), "data"))
  library(Rcpp)
  library(RcppArmadillo)
  if (!requireNamespace("GIGrvg", quietly = TRUE)) {
    install.packages("GIGrvg", repos = "https://cloud.r-project.org")
  }
  suppressMessages(library(GIGrvg))
  sourceCpp("my_LSIRM_FMC_v14.cpp")
})

# Analytical GIG mean: E[X] = sqrt(chi/psi) * K_{p+1}(omega) / K_p(omega), omega = sqrt(chi*psi).
gig_mean <- function(p, chi, psi) {
  omega <- sqrt(chi * psi)
  besselK(omega, p + 1) / besselK(omega, p) * sqrt(chi / psi)
}

run_test <- function(p, chi, psi, n = 50000, tol_rel = 0.03) {
  # Use cpp helper. v14_gig_sanity(n, lambda=p, chi, psi)
  x <- v14_gig_sanity(n, p, chi, psi)
  emp_mean <- mean(x)
  th_mean  <- gig_mean(p, chi, psi)
  rel_err  <- abs(emp_mean - th_mean) / abs(th_mean)
  pass <- rel_err < tol_rel
  cat(sprintf("[gig] p=%g chi=%g psi=%g: emp=%.4f theory=%.4f rel-err=%.4f -> %s\n",
              p, chi, psi, emp_mean, th_mean, rel_err,
              if (pass) "PASS" else "FAIL"))
  list(pass = pass, emp = emp_mean, theory = th_mean, rel_err = rel_err)
}

set.seed(20250511)

# Recommended operating point: p = nu - L/2 = 10 - 2 = 8, psi = 2 nu = 20, chi = arbitrary > 0
r1 <- run_test(p = 8.0,  chi = 1.0,  psi = 20.0,  n = 50000)
r2 <- run_test(p = 8.0,  chi = 0.01, psi = 20.0,  n = 50000)
r3 <- run_test(p = 7.5,  chi = 4.0,  psi = 20.0,  n = 50000)

# Cross-validate against direct GIGrvg::rgig call.
cat("\n[gig] Cross-validation against GIGrvg::rgig directly:\n")
direct <- GIGrvg::rgig(n = 50000, lambda = 8.0, chi = 1.0, psi = 20.0)
cat(sprintf("  cpp wrapper mean: %.4f   GIGrvg direct mean: %.4f\n",
            mean(v14_gig_sanity(50000, 8.0, 1.0, 20.0)),
            mean(direct)))

stopifnot(r1$pass, r2$pass, r3$pass)
cat("PASS test_v14_gig.R\n")
