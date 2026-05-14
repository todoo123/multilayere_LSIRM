## Per-cluster Sigma analysis on v11t MIDUS result, conditional on Dahl mode
## partition (resolves label switching).
##
## Three analyses:
##   (1) Within-cluster spread of b posterior MEAN per Dahl cluster
##       (simple, no label matching needed)
##   (2) Per-iter Sigma_l matched to Dahl clusters via majority-vote label
##       matching (true cluster-conditional Sigma posterior)
##   (3) Compare to overall b spread + prior expectation

suppressPackageStartupMessages({ library(abind) })

res_path <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_1_nu12_M100_nut3_tunedb/case1_all_v11t_fmc_d2_K10_e0.05_S0init_1_nu12_M100_nut3_tunedb_result.rds"
res <- readRDS(res_path)

Sigma   <- res$fmc_Sigma                 # 2x2x10x9000
fmc_c   <- res$fmc_c                     # 9000 x P
co_pp   <- res$fmc_co_cluster            # P x P PSM (assumed available)
n_save  <- dim(Sigma)[4]; K_star <- dim(Sigma)[3]; d <- dim(Sigma)[1]
P       <- ncol(fmc_c)
nu0     <- 12
S0_diag <- 1
prior_mean_perdim <- S0_diag / (nu0 - d - 1)
cat(sprintf("n_save=%d, K_star=%d, d=%d, P_total=%d, nu_t=%g, e0=%g\n",
            n_save, K_star, d, P, res$nu_t, res$fmc_e0))
cat(sprintf("Prior E[Sigma_l] per-dim = %.4f -> sd %.3f\n",
            prior_mean_perdim, sqrt(prior_mean_perdim)))

## --- Stack b posterior into one (n_save, P, d) array ---
b_all_iter <- abind::abind(res$b1, res$b2, res$b3, res$b4, along = 2)  # 9000 x P x 2
stopifnot(dim(b_all_iter)[2] == P)
b_post_mean <- apply(b_all_iter, c(2,3), mean)                          # P x 2

## --- Compute Dahl partition ---
ut <- upper.tri(co_pp)
C_ut <- co_pp[ut]
loss <- numeric(n_save)
for (s in 1:n_save) {
  cs <- fmc_c[s, ]
  Cs_ut <- (outer(cs, cs, "=="))[ut]
  loss[s] <- sum((Cs_ut - C_ut)^2)
}
s_star <- which.min(loss)
dahl_raw <- as.integer(fmc_c[s_star, ])
dahl <- as.integer(factor(dahl_raw, levels = unique(dahl_raw)))
K_dahl <- length(unique(dahl))
cat(sprintf("\nDahl partition (iter %d, loss %.2f): K=%d, sizes=%s\n",
            s_star, loss[s_star], K_dahl, paste(table(dahl), collapse=",")))

## ============================================================
## (1) Within-cluster spread of b posterior MEAN per Dahl cluster
## ============================================================
cat("\n=== (1) Within-cluster spread of b posterior mean (per Dahl cluster) ===\n")
cat(sprintf("%-3s %-5s %-15s %-15s %-15s %-12s\n",
            "k","n_k","mean_perdim_var","perdim_sd","2D_total_var","cluster_center"))
for (k in 1:K_dahl) {
  idx <- which(dahl == k)
  if (length(idx) < 2) {
    cat(sprintf("  cluster %d (n=%d): too small to compute spread\n", k, length(idx)))
    next
  }
  b_k <- b_post_mean[idx, , drop=FALSE]
  center_k <- colMeans(b_k)
  v_perdim <- apply(b_k, 2, var)
  total_var <- sum(v_perdim)
  cat(sprintf("%-3d %-5d %-15.4f %-15.3f %-15.4f (%.2f, %.2f)\n",
              k, length(idx), mean(v_perdim), sqrt(mean(v_perdim)),
              total_var, center_k[1], center_k[2]))
}

cat(sprintf("\n  pooled within-cluster variance per-dim = %.4f -> sd %.3f\n",
            {
              v_acc <- 0; n_acc <- 0
              for (k in 1:K_dahl) {
                idx <- which(dahl == k)
                if (length(idx) < 2) next
                v_acc <- v_acc + (length(idx) - 1) * mean(apply(b_post_mean[idx,], 2, var))
                n_acc <- n_acc + (length(idx) - 1)
              }
              v_acc / n_acc
            },
            sqrt({
              v_acc <- 0; n_acc <- 0
              for (k in 1:K_dahl) {
                idx <- which(dahl == k)
                if (length(idx) < 2) next
                v_acc <- v_acc + (length(idx) - 1) * mean(apply(b_post_mean[idx,], 2, var))
                n_acc <- n_acc + (length(idx) - 1)
              }
              v_acc / n_acc
            })))

## ============================================================
## (2) Per-iter Sigma_l matched to Dahl clusters via majority overlap
## ============================================================
cat("\n=== (2) Per-iter Sigma matched to Dahl clusters (label switching resolved) ===\n")
matched_Sigma_trace <- matrix(NA_real_, n_save, K_dahl)  # mean per-dim variance
for (s in 1:n_save) {
  cs <- fmc_c[s, ]
  for (k in 1:K_dahl) {
    idx_dahl <- which(dahl == k)
    cs_in_k <- cs[idx_dahl]
    ## majority vote: which cluster label l does Dahl cluster k correspond to in iter s?
    tab <- table(cs_in_k)
    if (length(tab) == 0) next
    matched_l <- as.integer(names(tab)[which.max(tab)])
    if (matched_l < 1 || matched_l > K_star) next
    matched_Sigma_trace[s, k] <- sum(diag(Sigma[,,matched_l,s])) / d
  }
}

cat(sprintf("%-3s %-15s %-12s %-25s\n",
            "k","mean_trace/d","perdim_sd","q25 q50 q75"))
for (k in 1:K_dahl) {
  v <- matched_Sigma_trace[,k]
  v <- v[!is.na(v)]
  if (length(v) < 50) {
    cat(sprintf("  cluster %d: too few matched iters\n", k)); next
  }
  cat(sprintf("%-3d %-15.4f %-12.3f %-25s\n", k, mean(v), sqrt(mean(v)),
              sprintf("%.4f / %.4f / %.4f",
                      quantile(v,.25), quantile(v,.50), quantile(v,.75))))
}

## ============================================================
## (3) Compare to overall b spread and prior
## ============================================================
cat("\n=== (3) Comparison ===\n")
overall_var <- apply(b_post_mean, 2, var)
overall_perdim <- mean(overall_var)
cat(sprintf("  Prior E[Sigma_l] per-dim:                  %.4f  (sd %.3f)\n",
            prior_mean_perdim, sqrt(prior_mean_perdim)))
cat(sprintf("  Overall b posterior-mean variance per-dim: %.4f  (sd %.3f)\n",
            overall_perdim, sqrt(overall_perdim)))

within_pooled <- {
  v_acc <- 0; n_acc <- 0
  for (k in 1:K_dahl) {
    idx <- which(dahl == k)
    if (length(idx) < 2) next
    v_acc <- v_acc + (length(idx)-1) * mean(apply(b_post_mean[idx,],2,var))
    n_acc <- n_acc + (length(idx)-1)
  }
  v_acc / n_acc
}
cat(sprintf("  Within-Dahl-cluster pooled var per-dim:    %.4f  (sd %.3f)\n",
            within_pooled, sqrt(within_pooled)))

between_var <- overall_perdim - within_pooled
cat(sprintf("  Between-cluster variance per-dim (approx): %.4f  (sd %.3f)\n",
            between_var, sqrt(max(between_var, 0))))

cat(sprintf("\n  Ratio (within posterior estimate / actual within data):\n"))
matched_overall <- mean(matched_Sigma_trace, na.rm=TRUE)
cat(sprintf("    matched-Sigma posterior mean / actual within-data var = %.4f / %.4f = %.2fx\n",
            matched_overall, within_pooled, matched_overall / within_pooled))

cat(sprintf("\n--> Interpretation:\n"))
cat(sprintf("  prior/data within ratio = %.2f\n", prior_mean_perdim / within_pooled))
cat(sprintf("  posterior/data within ratio = %.2f\n", matched_overall / within_pooled))
