################################################################################
# Gaussian Mixture Model clustering on the v8 LSIRM-induced item factor scores
#
# Companion to 01_run_factor_analysis.R. Loads the item factor scores produced
# there and refits clusters using a finite Gaussian mixture (mclust). Compared
# to k-means, GMM:
#   - allows ellipsoidal (not just spherical) clusters
#   - gives soft (probabilistic) assignments
#   - provides BIC-based model selection over BOTH the number of clusters K
#     AND the covariance structure (EII, EEI, EEE, VVV, ...).
#
# Pipeline:
#   1. Load r-dimensional item factor scores from 01's output
#   2. Mclust with G = 1..G_max and all covariance models -> BIC matrix
#   3. Pick the BIC-optimal (G, modelName)
#   4. Save BIC plot, classification, soft-assignment uncertainty,
#      comparison vs k-means partition (same K) using ARI
#   5. Visualise items in factor space coloured by GMM cluster
################################################################################

suppressMessages({
  library(mclust)    # Mclust, mclustBIC, adjustedRandIndex
  library(cluster)   # silhouette (for diagnostic)
})

################################################################################
# 0. Locate the latest 01 output to reuse
################################################################################
project_root <- "/Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM"
out_root <- file.path(project_root, "factor_analysis_v8", "output")

run_dirs <- list.dirs(out_root, recursive = FALSE)
stopifnot(length(run_dirs) > 0)
out_dir   <- run_dirs[which.max(file.info(run_dirs)$mtime)]
run_label <- basename(out_dir)
cat(sprintf("Using factor scores from: %s\n", out_dir))

scores_csv <- file.path(out_dir, "item_factor_scores.csv")
stopifnot(file.exists(scores_csv))
F_df <- read.csv(scores_csv, row.names = 1, check.names = FALSE)
F_r  <- as.matrix(F_df)
P    <- nrow(F_r)
r    <- ncol(F_r)
item_names <- rownames(F_r)
cat(sprintf("Factor score matrix: %d items x %d factors\n", P, r))

################################################################################
# 1. Mclust BIC sweep over G = 1..G_max and all covariance models
################################################################################
G_max <- 10L
seed  <- 20260428
set.seed(seed)

cat(sprintf("\nFitting GMM with G = 1..%d and all covariance models... ", G_max))
bic_obj <- mclustBIC(F_r, G = 1:G_max, verbose = FALSE)
cat("done.\n")

# Pick best (G, modelName) by max BIC (mclust convention: higher BIC = better)
best <- mclust::pickBIC(bic_obj, krange = 1:G_max)
# Fallback: pick from the BIC matrix manually if pickBIC unavailable
bic_mat   <- as.matrix(bic_obj)
best_idx  <- which(bic_mat == max(bic_mat, na.rm = TRUE), arr.ind = TRUE)[1, ]
best_G    <- as.integer(rownames(bic_mat)[best_idx[1]])
best_mod  <- colnames(bic_mat)[best_idx[2]]
best_bic  <- bic_mat[best_idx[1], best_idx[2]]
cat(sprintf("BIC-optimal model: G = %d, covariance = '%s', BIC = %.2f\n",
            best_G, best_mod, best_bic))

# Refit final model (this also gives soft probabilities and uncertainty)
gmm <- Mclust(F_r, G = best_G, modelNames = best_mod, verbose = FALSE)

################################################################################
# 2. Save BIC table + plot
################################################################################
write.csv(bic_mat,
          file.path(out_dir, "gmm_BIC_table.csv"))

pdf(file.path(out_dir, "gmm_BIC_plot.pdf"), width = 10, height = 6)
par(mar = c(4, 4, 3, 6))
matplot(seq_len(nrow(bic_mat)), bic_mat, type = "b", pch = 1:ncol(bic_mat),
        col = seq_len(ncol(bic_mat)), lty = 1,
        xlab = "G (number of mixture components)", ylab = "BIC",
        main = sprintf("Mclust BIC sweep  (best: G=%d, %s)", best_G, best_mod))
legend("right", inset = c(-0.18, 0), legend = colnames(bic_mat),
       col = seq_len(ncol(bic_mat)), pch = 1:ncol(bic_mat),
       lty = 1, bty = "n", cex = 0.7, xpd = TRUE)
abline(v = best_G, col = "red", lty = 2)
dev.off()

################################################################################
# 3. Cluster summary, partition, soft assignments
################################################################################
cls <- gmm$classification        # hard assignment (length P)
unc <- gmm$uncertainty           # 1 - max posterior prob per item
z   <- gmm$z                     # P x G soft probabilities

cat("\nGMM cluster sizes:\n"); print(table(cls))
cat(sprintf("\nMean uncertainty (1 - max posterior prob): %.3f\n", mean(unc)))
cat("Distribution of uncertainty:\n")
print(round(quantile(unc, c(0, .25, .5, .75, 1)), 3))

# Silhouette for the GMM partition (for direct comparison vs k-means)
sil_gmm <- silhouette(cls, dist(F_r))
cat(sprintf("Average silhouette width (GMM partition): %.3f\n",
            mean(sil_gmm[, "sil_width"])))

# Save partition + soft probabilities
write.csv(data.frame(item = item_names,
                     cluster = cls,
                     uncertainty = round(unc, 4),
                     z = round(z, 3),
                     F_r),
          file.path(out_dir,
                    sprintf("gmm_partition_G%d_%s.csv", best_G, best_mod)),
          row.names = FALSE)

################################################################################
# 4. Compare with k-means partition at the same K (if available)
################################################################################
km_files <- list.files(out_dir, pattern = "^partition_K\\d+_silhouette\\.csv$",
                       full.names = TRUE)
ari_msg <- ""
if (length(km_files) >= 1) {
  km_df <- read.csv(km_files[1], stringsAsFactors = FALSE)
  km_cls <- km_df$cluster
  if (length(km_cls) == P) {
    ari_same_K <- adjustedRandIndex(cls, km_cls)
    ari_msg <- sprintf("Adjusted Rand Index vs k-means (%s, K = %d): %.3f",
                       basename(km_files[1]), max(km_cls), ari_same_K)
    cat(sprintf("\n%s\n", ari_msg))
  }
}

################################################################################
# 5. 2D / pairs visualisation in factor space
################################################################################
pal <- rainbow(best_G, alpha = 0.85)

if (r == 2) {
  pdf(file.path(out_dir, "gmm_factor_scatter.pdf"), width = 9, height = 8)
  par(mar = c(4, 4, 3, 1))
  plot(F_r[, 1], F_r[, 2], pch = 19, cex = 1.4 * (1 - 0.6 * unc),
       col = pal[cls],
       xlab = colnames(F_r)[1], ylab = colnames(F_r)[2],
       main = sprintf("GMM partition  (G=%d, %s)\nsize ~ confidence",
                      best_G, best_mod))
  text(F_r[, 1], F_r[, 2], labels = item_names, pos = 4, cex = 0.5)
  legend("topright", legend = paste0("c", seq_len(best_G)),
         pch = 19, col = pal, bty = "n", cex = 0.8)
  dev.off()
} else if (r >= 3) {
  pdf(file.path(out_dir, "gmm_factor_pairs.pdf"), width = 10, height = 10)
  pairs(F_r, col = pal[cls], pch = 19, cex = 1.2 * (1 - 0.6 * unc),
        labels = colnames(F_r),
        main = sprintf("GMM partition  (G=%d, %s) -- pairwise factor scores",
                       best_G, best_mod))
  dev.off()

  # Also produce a focused scatter of first two factors
  pdf(file.path(out_dir, "gmm_factor_F1F2.pdf"), width = 9, height = 8)
  par(mar = c(4, 4, 3, 1))
  plot(F_r[, 1], F_r[, 2], pch = 19, cex = 1.4 * (1 - 0.6 * unc),
       col = pal[cls],
       xlab = colnames(F_r)[1], ylab = colnames(F_r)[2],
       main = sprintf("GMM partition (F1 vs F2)  (G=%d, %s)\nsize ~ confidence",
                      best_G, best_mod))
  text(F_r[, 1], F_r[, 2], labels = item_names, pos = 4, cex = 0.5)
  legend("topright", legend = paste0("c", seq_len(best_G)),
         pch = 19, col = pal, bty = "n", cex = 0.8)
  dev.off()
}

################################################################################
# 6. Soft-assignment uncertainty plot
################################################################################
pdf(file.path(out_dir, "gmm_uncertainty.pdf"), width = 9, height = 5)
par(mar = c(7, 4, 3, 1))
ord <- order(cls, -unc)
barplot(unc[ord], names.arg = item_names[ord], las = 2, cex.names = 0.55,
        col = pal[cls[ord]],
        ylab = "Uncertainty (1 - max posterior prob)",
        main = sprintf("Per-item GMM assignment uncertainty (G=%d, %s)",
                       best_G, best_mod))
abline(h = 0.5, col = "red", lty = 2)
legend("topright", legend = paste0("c", seq_len(best_G)),
       fill = pal, bty = "n", cex = 0.8)
dev.off()

################################################################################
# 7. Append to summary
################################################################################
gmm_summary <- file.path(out_dir, "gmm_summary.txt")
sink(gmm_summary)
cat(sprintf("GMM clustering on item factor scores (r = %d)\n", r))
cat(sprintf("Source factor scores: %s\n\n", scores_csv))
cat(sprintf("BIC-optimal model: G = %d, covariance = '%s', BIC = %.2f\n",
            best_G, best_mod, best_bic))
cat(sprintf("Average silhouette width: %.3f\n",
            mean(sil_gmm[, "sil_width"])))
cat(sprintf("Mean assignment uncertainty: %.3f\n", mean(unc)))
cat(sprintf("# items with uncertainty > 0.5: %d / %d\n", sum(unc > 0.5), P))
if (nzchar(ari_msg)) cat(sprintf("\n%s\n", ari_msg))
cat("\nCluster sizes:\n"); print(table(cls))
cat("\nTop 5 covariance models by BIC:\n")
top5 <- sort(bic_mat[!is.na(bic_mat)], decreasing = TRUE)[1:5]
for (val in top5) {
  idx <- which(bic_mat == val, arr.ind = TRUE)[1, ]
  cat(sprintf("  G = %s, model = %s, BIC = %.2f\n",
              rownames(bic_mat)[idx[1]], colnames(bic_mat)[idx[2]], val))
}
sink()
cat(sprintf("\nGMM summary: %s\n", gmm_summary))
cat(sprintf("All outputs in: %s\n", out_dir))
