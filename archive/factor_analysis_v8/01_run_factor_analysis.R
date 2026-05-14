################################################################################
# Direct factor analysis on the LSIRM-induced item profiles
# (independent benchmark for the v8 FMC clustering result)
#
# Pipeline:
#   1. Load v8 result (.rds) saved by MIDUS_5layered_result_v8.R
#   2. Compute posterior mean a (n x d) and stacked b (P x d)
#   3. Construct x[i, j] = log ||a_i - b_j||  (n x P)
#      Row-centre to match the v8 model's preprocessing.
#   4. PCA / SVD on x to obtain item factor scores in R^r
#   5. Decide optimal r via:
#        (a) Kaiser criterion  (eigenvalue > 1 on the correlation matrix)
#        (b) Parallel analysis (compare to eigenvalues of permuted x)
#        (c) Cumulative variance explained
#        (d) Scree plot
#   6. K-means on item factor scores with K = 1..K_max:
#        (a) Total within-cluster SS (elbow)
#        (b) Average silhouette width (cluster::silhouette)
#        (c) Gap statistic (cluster::clusGap)
#   7. Save plots + CSV diagnostics to ./output/<run_label>/
#
# Note: items are observations and respondents are variables. Because P < n,
# a covariance-based PCA / FA is well-defined; we use SVD directly to avoid
# any singular n x n covariance issues.
################################################################################

suppressMessages({
  library(cluster)   # silhouette, clusGap
  library(ggplot2)   # for prettier plots (optional; falls back to base if absent)
})

################################################################################
# 0. Configuration
################################################################################
project_root <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"

# Choose which v8 result directory to analyse.
# Default = the most recently modified case1_all_v8_fmc_* directory.
plot_root   <- file.path(project_root, "data", "plot")
v8_dirs     <- list.dirs(plot_root, recursive = FALSE)
v8_dirs     <- v8_dirs[grepl("case1_all_v8_fmc_", basename(v8_dirs))]
stopifnot(length(v8_dirs) > 0)
v8_dir      <- v8_dirs[which.max(file.info(v8_dirs)$mtime)]
run_label   <- basename(v8_dir)
result_rds  <- file.path(v8_dir, paste0(run_label, "_result.rds"))
cat(sprintf("Using v8 result: %s\n", result_rds))
stopifnot(file.exists(result_rds))

# Output folder (sibling under factor_analysis_v8/output/<run_label>/)
out_dir <- file.path(project_root, "factor_analysis_v8", "output", run_label)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
cat(sprintf("Output -> %s\n", out_dir))

# Tuning knobs
K_max    <- 10L      # k-means cluster grid: 1..K_max
B_par    <- 100L     # parallel-analysis bootstrap reps
gap_B    <- 100L     # gap-statistic bootstrap reps
seed     <- 20260428
set.seed(seed)

################################################################################
# 1. Load result and compute posterior-mean positions
################################################################################
res  <- readRDS(result_rds)
info <- res$info
n    <- info$n
d    <- common_d <- dim(res$a)[3]
P_total <- info$P_total
cat(sprintf("\nDimensions: n = %d  P_total = %d  d = %d\n", n, P_total, d))

# Posterior mean of a (n x d) -- result$a is (n_save x n x d) after wrapper aperm
A_pm <- apply(res$a, c(2, 3), mean)

# Stack b1..b5 into B_pm (P_total x d) using the same canonical order as v8
b_arrs <- list(res$b1, res$b2, res$b3, res$b4, res$b5)
B_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
  if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
    return(matrix(0, 0, d))
  apply(arr, c(2, 3), mean)
}))
stopifnot(nrow(B_pm) == P_total)

# Try to recover item names if previously saved
item_names <- tryCatch({
  ic <- read.csv(file.path(v8_dir, paste0(run_label, "_fmc_item_clusters.csv")),
                 stringsAsFactors = FALSE)
  ic$item
}, error = function(e) paste0("item_", seq_len(P_total)))
stopifnot(length(item_names) == P_total)

################################################################################
# 2. Build x[i, j] = log ||a_i - b_j||  and row-centre
################################################################################
EPS <- 1e-8
D_mat  <- as.matrix(dist(rbind(A_pm, B_pm)))[1:n, n + seq_len(P_total), drop = FALSE]
D_mat  <- pmax(D_mat, EPS)
x_log  <- log(D_mat + 1)              # n x P_total
x_rc   <- x_log - rowMeans(x_log) # row-centre (matches v8's row-centring)

cat(sprintf("\nx_log raw       : range = [%.3f, %.3f], sd = %.3f\n",
            min(x_log), max(x_log), sd(x_log)))
cat(sprintf("x_log row-centred: range = [%.3f, %.3f], sd = %.3f\n",
            min(x_rc), max(x_rc), sd(x_rc)))
cat(sprintf("between-item sd of column means (raw): %.3f\n",
            sd(colMeans(x_log))))

# Save the matrix used for FA so downstream scripts can reuse it
saveRDS(list(x_log = x_log, x_rc = x_rc, A_pm = A_pm, B_pm = B_pm,
             item_names = item_names),
        file.path(out_dir, "x_log.rds"))

################################################################################
# 3. PCA / SVD on x_rc with items as observations
#
#    Treat each column of x_rc (= one respondent's profile across items)? NO.
#    We treat each ROW of t(x_rc) (= one item's profile across respondents)
#    as an observation. So input matrix is X_items: P_total x n.
################################################################################
X_items <- t(x_rc)                 # P_total x n
X_items <- scale(X_items, center = TRUE, scale = FALSE)   # column-centre

# SVD for full-rank decomposition (works for any P x n shape)
sv <- svd(X_items)
# Eigenvalues of the cross-product / (P-1) (covariance-based):
sigma2 <- (sv$d^2) / (P_total - 1)
total_var <- sum(sigma2)
prop_var  <- sigma2 / total_var
cum_var   <- cumsum(prop_var)
n_eigs    <- length(sigma2)
cat(sprintf("\n# of non-zero eigenvalues from SVD: %d\n", n_eigs))

# Item factor scores: principal component scores (rows = items, cols = factors)
F_full <- sv$u %*% diag(sv$d)              # P_total x n_eigs

################################################################################
# 4. Choose # factors via four criteria
################################################################################
# (a) Kaiser-style: eigenvalue > 1 on the CORRELATION matrix scale.
#     Here we use eigenvalue > mean(eigenvalue) as a scale-free analogue.
mean_eig <- mean(sigma2)
r_kaiser <- sum(sigma2 > mean_eig)

# (b) Parallel analysis: compare observed eigenvalues to the 95th percentile
#     of eigenvalues from B random column-permutations of X_items.
cat(sprintf("\nParallel analysis (B = %d permutations)... ", B_par))
sim_eigs <- matrix(NA_real_, nrow = B_par, ncol = n_eigs)
for (b in seq_len(B_par)) {
  Xb <- apply(X_items, 2, sample)        # break column dependencies
  sb <- svd(scale(Xb, center = TRUE, scale = FALSE))$d
  sim_eigs[b, seq_along(sb)] <- (sb^2) / (P_total - 1)
}
sim_q95 <- apply(sim_eigs, 2, quantile, probs = 0.95, na.rm = TRUE)
r_par   <- sum(sigma2 > sim_q95)
cat("done.\n")

# (c) Cumulative variance: smallest r with cum_var >= 0.80 / 0.90
r_var80 <- which(cum_var >= 0.80)[1]
r_var90 <- which(cum_var >= 0.90)[1]

# (d) Scree elbow: simple "largest curvature" heuristic
diffs <- -diff(sigma2)
elbow_r <- which.max(diffs[seq_len(min(n_eigs - 1, 15))]) + 0   # purely diagnostic

# Print summary table
factor_summary <- data.frame(
  factor      = seq_len(min(n_eigs, 20)),
  eigenvalue  = sigma2[1:min(n_eigs, 20)],
  prop_var    = prop_var[1:min(n_eigs, 20)],
  cum_var     = cum_var[1:min(n_eigs, 20)],
  parallel_q95= sim_q95[1:min(n_eigs, 20)]
)
cat("\nFirst 20 factors:\n")
print(round(factor_summary, 4))
cat(sprintf("\n# factors selected by:\n"))
cat(sprintf("  (a) Kaiser-style (eig > mean)    : %d\n", r_kaiser))
cat(sprintf("  (b) Parallel analysis (q95)      : %d  <-- recommended\n", r_par))
cat(sprintf("  (c) Cumulative variance >= 80%%   : %d\n", r_var80))
cat(sprintf("  (c) Cumulative variance >= 90%%   : %d\n", r_var90))
cat(sprintf("  (d) Scree elbow heuristic        : %d\n", elbow_r))

write.csv(factor_summary,
          file.path(out_dir, "factor_summary.csv"),
          row.names = FALSE)

# Scree + parallel-analysis plot
pdf(file.path(out_dir, "scree_plot.pdf"), width = 8, height = 6)
nshow <- min(n_eigs, 20)
plot(seq_len(nshow), sigma2[1:nshow], type = "b", pch = 19,
     xlab = "Factor index", ylab = "Eigenvalue",
     main = sprintf("Scree + Parallel analysis  (%s)", run_label),
     ylim = range(c(sigma2[1:nshow], sim_q95[1:nshow])))
lines(seq_len(nshow), sim_q95[1:nshow], type = "b", pch = 1, col = "red", lty = 2)
abline(h = mean_eig, col = "blue", lty = 3)
legend("topright",
       legend = c("Observed eigenvalues",
                  "Parallel analysis (q95)",
                  sprintf("Mean eigenvalue (Kaiser-style cutoff = %.2f)", mean_eig)),
       col = c("black", "red", "blue"), lty = c(1, 2, 3),
       pch = c(19, 1, NA), bty = "n", cex = 0.8)
abline(v = r_par, col = "darkgreen", lwd = 2, lty = 2)
text(r_par, max(sigma2[1:nshow]) * 0.9,
     sprintf("r* = %d (parallel)", r_par), pos = 4, col = "darkgreen")
dev.off()

# Settle on r = parallel-analysis pick (most defensible)
r_choice <- max(2L, as.integer(r_par))
cat(sprintf("\n=> Using r = %d for downstream k-means.\n", r_choice))

# Item factor scores at chosen r
F_r <- F_full[, seq_len(r_choice), drop = FALSE]
rownames(F_r) <- item_names
colnames(F_r) <- paste0("F", seq_len(r_choice))
write.csv(round(F_r, 4),
          file.path(out_dir, "item_factor_scores.csv"))

################################################################################
# 5. K-means on item factor scores -- determine optimal K
################################################################################
cat(sprintf("\nK-means sweep: K = 2 .. %d (with multiple inits)\n", K_max))

km_results <- vector("list", K_max)
wss_vec    <- numeric(K_max)
sil_vec    <- rep(NA_real_, K_max)
dist_F     <- dist(F_r)

for (K in 1:K_max) {
  set.seed(seed + K)
  km <- kmeans(F_r, centers = K, nstart = 50, iter.max = 100)
  km_results[[K]] <- km
  wss_vec[K] <- km$tot.withinss
  if (K >= 2) {
    sil <- silhouette(km$cluster, dist_F)
    sil_vec[K] <- mean(sil[, "sil_width"])
  }
}

# Gap statistic (cluster::clusGap)
cat(sprintf("Gap statistic (B = %d)... ", gap_B))
gap_res <- clusGap(F_r, FUN = kmeans, K.max = K_max, B = gap_B,
                   nstart = 25, iter.max = 100)
gap_tab <- gap_res$Tab
cat("done.\n")

# Optimal K via different rules:
K_elbow_first   <- {
  # Knee detection using the largest drop in WSS
  drops <- -diff(wss_vec)
  which.max(drops[seq_len(min(K_max - 1, 8))])      # location of biggest drop
} + 1L
K_silhouette    <- if (any(!is.na(sil_vec))) which.max(sil_vec) else NA_integer_
K_gap_global    <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "globalmax")
K_gap_firstSEmx <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "firstSEmax")
K_gap_TibsSEmax <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "Tibs2001SEmax")

cluster_summary <- data.frame(
  K        = seq_len(K_max),
  wss      = wss_vec,
  silhouette = sil_vec,
  gap      = gap_tab[, "gap"],
  gap_SE   = gap_tab[, "SE.sim"]
)
cat("\nCluster validation table:\n")
print(round(cluster_summary, 4))
cat(sprintf("\nOptimal K via:\n"))
cat(sprintf("  Elbow (largest WSS drop)              : %d\n", K_elbow_first))
cat(sprintf("  Silhouette (max average sil-width)    : %s  <-- often most informative\n",
            K_silhouette))
cat(sprintf("  Gap statistic (globalmax)             : %s\n", K_gap_global))
cat(sprintf("  Gap statistic (firstSEmax)            : %s\n", K_gap_firstSEmx))
cat(sprintf("  Gap statistic (Tibs2001SEmax, default): %s\n", K_gap_TibsSEmax))

write.csv(cluster_summary,
          file.path(out_dir, "kmeans_validation.csv"),
          row.names = FALSE)

################################################################################
# 6. K-means validation plots
################################################################################
pdf(file.path(out_dir, "kmeans_validation.pdf"), width = 10, height = 8)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# (a) Elbow plot
plot(seq_len(K_max), wss_vec, type = "b", pch = 19,
     xlab = "K (number of clusters)", ylab = "Total within-cluster SS",
     main = sprintf("Elbow plot  (knee around K=%d)", K_elbow_first))
abline(v = K_elbow_first, col = "red", lty = 2)

# (b) Silhouette plot
plot(seq_len(K_max), sil_vec, type = "b", pch = 19,
     xlab = "K", ylab = "Average silhouette width",
     main = sprintf("Silhouette  (max at K=%s)", K_silhouette))
if (!is.na(K_silhouette)) abline(v = K_silhouette, col = "red", lty = 2)

# (c) Gap statistic
plot(gap_tab[, "gap"], type = "b", pch = 19,
     xlab = "K", ylab = "Gap(K)",
     main = sprintf("Gap statistic  (Tibs2001SEmax = K%s)", K_gap_TibsSEmax))
arrows(seq_len(K_max), gap_tab[, "gap"] - gap_tab[, "SE.sim"],
       seq_len(K_max), gap_tab[, "gap"] + gap_tab[, "SE.sim"],
       angle = 90, code = 3, length = 0.05, col = "gray60")
abline(v = K_gap_TibsSEmax, col = "red", lty = 2)

# (d) First two factor scores coloured by chosen K (silhouette pick if available)
K_pick <- if (!is.na(K_silhouette)) K_silhouette else K_elbow_first
K_pick <- 4
km_pick <- km_results[[K_pick]]
plot(F_r[, 1], F_r[, 2], pch = 19, cex = 1.3,
     col = km_pick$cluster,
     xlab = "Factor 1", ylab = "Factor 2",
     main = sprintf("Items in factor space (K = %d, '%s' rule)",
                    K_pick,
                    if (!is.na(K_silhouette)) "silhouette" else "elbow"))
text(F_r[, 1], F_r[, 2], labels = item_names, pos = 4, cex = 0.5)
dev.off()

# Save the chosen partition
write.csv(data.frame(item     = item_names,
                     cluster  = km_pick$cluster,
                     F_r),
          file.path(out_dir,
                    sprintf("partition_K%d_silhouette.csv", K_pick)),
          row.names = FALSE)

################################################################################
# 7. Final summary
################################################################################
summary_txt <- file.path(out_dir, "summary.txt")
sink(summary_txt)
cat(sprintf("Factor analysis on v8 LSIRM result\n"))
cat(sprintf("Source: %s\n\n", result_rds))
cat(sprintf("Dimensions: n = %d  P = %d  d (LSIRM) = %d\n", n, P_total, d))
cat(sprintf("\n# Factor selection\n"))
cat(sprintf("  Kaiser-style (eig > mean)     : %d\n", r_kaiser))
cat(sprintf("  Parallel analysis (q95)       : %d   <-- chosen\n", r_par))
cat(sprintf("  Cumulative var >= 80%%         : %d\n", r_var80))
cat(sprintf("  Cumulative var >= 90%%         : %d\n", r_var90))
cat(sprintf("\n# K-means cluster selection (using r = %d factors)\n", r_choice))
cat(sprintf("  Elbow (largest WSS drop)      : K = %d\n", K_elbow_first))
cat(sprintf("  Silhouette (max average)      : K = %s\n", K_silhouette))
cat(sprintf("  Gap (Tibs2001SEmax, default)  : K = %s\n", K_gap_TibsSEmax))
cat(sprintf("  Gap (firstSEmax)              : K = %s\n", K_gap_firstSEmx))
cat(sprintf("  Gap (globalmax)               : K = %s\n", K_gap_global))
cat(sprintf("\n# Items per cluster (chosen rule):\n"))
print(table(km_pick$cluster))
sink()
cat(sprintf("\nSummary written to: %s\n", summary_txt))
cat(sprintf("All outputs in   : %s\n", out_dir))

