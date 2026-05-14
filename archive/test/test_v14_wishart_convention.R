# test_v14_wishart_convention.R
# ============================================================
# Verifies that v14_wishart_fs_sanity (cpp helper) draws X ~ W_d(c, C)
# under Fruhwirth-Schnatter convention:
#   p(X) prop |X|^{c-(d+1)/2} exp(-tr(C X)),  E[X] = c * C^{-1}.
#
# Test: with d=2, c=5, C=I_2:
#   E[X] = 5 * I_2; Var(X_{ii}) = c (analytic).
#   tr(X)/d should have mean = c.
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

run_test <- function(d, c_fs, C_fs, n = 20000, tol_rel = 0.05) {
  X <- v14_wishart_fs_sanity(n, c_fs, C_fs)
  # Empirical mean across n draws
  mean_X <- apply(X, c(1, 2), mean)
  # Theoretical FS mean: c * C^{-1}
  E_X_theory <- c_fs * solve(C_fs)

  rel_err <- max(abs(mean_X - E_X_theory)) / max(abs(E_X_theory))
  pass <- rel_err < tol_rel
  cat(sprintf("[wishart] d=%d c=%g: max-rel-err E[X] = %.4f  (tol %.2f) -> %s\n",
              d, c_fs, rel_err, tol_rel, if (pass) "PASS" else "FAIL"))
  list(pass = pass, rel_err = rel_err,
       mean_X = mean_X, E_X_theory = E_X_theory)
}

set.seed(20250511)

# Case 1: d=2, c=5, C=I_2  ->  E[X] = 5 * I_2
res1 <- run_test(2, 5.0, diag(2), n = 20000)

# Case 2: d=2, c=10, C=diag(2,3)  ->  E[X] = 10 * diag(1/2, 1/3)
C2 <- diag(c(2, 3))
res2 <- run_test(2, 10.0, C2, n = 20000)

# Case 3: d=3, c=8, C = a non-trivial PD matrix
C3 <- matrix(c(2, 0.5, 0.1,
               0.5, 1.5, 0.2,
               0.1, 0.2, 1.0), 3, 3)
res3 <- run_test(3, 8.0, C3, n = 20000, tol_rel = 0.07)

# Case 4 (boundary): d=2, c just above (d-1)/2 = 0.5
res4 <- run_test(2, 1.5, diag(2), n = 30000, tol_rel = 0.10)

cat("\n[wishart] cases pass:", res1$pass && res2$pass && res3$pass && res4$pass, "\n")
stopifnot(res1$pass, res2$pass, res3$pass, res4$pass)
cat("PASS test_v14_wishart_convention.R\n")
