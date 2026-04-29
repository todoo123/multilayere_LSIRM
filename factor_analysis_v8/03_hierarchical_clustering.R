################################################################################
# Hierarchical clustering on the v8 LSIRM-induced item factor scores
#
# Companion to 01_run_factor_analysis.R and 02_gmm_clustering.R.
#
# This script answers two questions:
#   (Q1) What does a hierarchical partition of the items look like?
#   (Q2) Is the hierarchy itself a valid representation of the data?
#        (i.e., does the dendrogram faithfully preserve pairwise distances,
#        and is the clustering structure strong enough to be meaningful?)
#
# Pipeline:
#   1. Load item factor scores from 01's output.
#   2. Compute Euclidean distance matrix in factor space.
#   3. Hierarchical clustering with FOUR linkages (ward.D2, average,
#      complete, single) -- different linkages reveal different aspects.
#   4. Hierarchy-validity diagnostics:
#        (a) Cophenetic correlation: how well the dendrogram preserves
#            input distances (target > 0.7).
#        (b) Agglomerative coefficient (cluster::agnes): strength of the
#            hierarchical structure (target > 0.7).
#        (c) Inversions / number-of-leaves consistency check.
#        (d) Bootstrap stability via the simpler resampling-based AU/BP
#            p-values (subsample columns of the original profile matrix x_rc
#            and compare partition agreement via ARI).
#   5. Optimal cut: per linkage, find K that maximises silhouette.
#   6. Cross-method comparison via ARI: hclust vs k-means vs GMM.
#   7. Plots: dendrograms, cophenetic scatter, silhouette curves,
#             hierarchy heatmap with reordered rows/cols.
################################################################################

suppressMessages({
  library(cluster)   # agnes, silhouette
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

# Load row-centred profile matrix for bootstrap diagnostic
x_rds <- file.path(out_dir, "x_log.rds")
x_dat <- readRDS(x_rds)
x_rc  <- x_dat$x_rc                  # n x P
n     <- nrow(x_rc)
stopifnot(ncol(x_rc) == P)

################################################################################
# 1. Distance matrix + hclust with 4 linkages
################################################################################
D    <- dist(F_r, method = "euclidean")
linkages <- c("ward.D2", "average", "complete", "single")
hc_list  <- lapply(linkages, function(m) hclust(D, method = m))
names(hc_list) <- linkages

################################################################################
# 2. Hierarchy-validity diagnostics
################################################################################
diag_tbl <- data.frame(
  linkage              = linkages,
  cophenetic_corr      = NA_real_,
  agglom_coef          = NA_real_,
  num_inversions       = NA_integer_,
  best_silhouette_K    = NA_integer_,
  best_silhouette_val  = NA_real_,
  stringsAsFactors = FALSE
)

# Cophenetic correlations: how well does the dendrogram preserve D?
for (i in seq_along(linkages)) {
  coph <- cophenetic(hc_list[[i]])
  diag_tbl$cophenetic_corr[i] <- cor(D, coph)
}

# Agglomerative coefficient via cluster::agnes (refit identically)
for (i in seq_along(linkages)) {
  ag <- agnes(F_r, method = ifelse(linkages[i] == "ward.D2", "ward",
                                   linkages[i]),
              metric = "euclidean")
  diag_tbl$agglom_coef[i] <- ag$ac
}

# Inversion count: a "true" hierarchy should have monotone non-decreasing
# merge heights. Count how many subsequent merges have lower height than
# the previous (only matters for centroid/median; we check anyway).
for (i in seq_along(linkages)) {
  h <- hc_list[[i]]$height
  diag_tbl$num_inversions[i] <- sum(diff(h) < 0)
}

cat("\n--- Hierarchy validity diagnostics ---\n")
print(transform(diag_tbl[, 1:4],
                cophenetic_corr = round(cophenetic_corr, 3),
                agglom_coef     = round(agglom_coef, 3)),
      row.names = FALSE)
cat("\nGuideline:\n")
cat("  cophenetic_corr > 0.75 : hierarchy faithfully preserves distances\n")
cat("                  > 0.85 : strong\n")
cat("  agglom_coef     > 0.65 : moderate clustering structure\n")
cat("                  > 0.85 : strong\n")
cat("  num_inversions  == 0   : monotone dendrogram (no upside-down branches)\n")

################################################################################
# 3. Per-linkage silhouette sweep -> optimal K
################################################################################
K_max <- 10L
sil_mat <- matrix(NA_real_, nrow = K_max, ncol = length(linkages),
                  dimnames = list(K = paste0("K=", 1:K_max), linkages))
for (i in seq_along(linkages)) {
  for (K in 2:K_max) {
    cl <- cutree(hc_list[[i]], k = K)
    if (length(unique(cl)) >= 2) {
      sil <- silhouette(cl, D)
      sil_mat[K, i] <- mean(sil[, "sil_width"])
    }
  }
  diag_tbl$best_silhouette_K[i]   <- which.max(sil_mat[, i])
  diag_tbl$best_silhouette_val[i] <- max(sil_mat[, i], na.rm = TRUE)
}

cat("\n--- Per-linkage silhouette sweep ---\n")
print(round(sil_mat, 3))
cat("\nBest K per linkage (silhouette):\n")
print(diag_tbl[, c("linkage", "best_silhouette_K", "best_silhouette_val")],
      row.names = FALSE)

################################################################################
# 4. Bootstrap stability -- resample respondents, recompute factor scores,
#    refit hclust, measure cluster agreement (ARI) with the reference partition.
#
#    This tests: does the hierarchical partition recur if respondents are
#    perturbed? Stable hierarchy => high ARI across bootstrap reps.
################################################################################
B_boot <- 50L
set.seed(20260428)

# Reference: ward.D2 cut at its silhouette-optimal K
ref_link <- "ward.D2"
ref_K    <- diag_tbl$best_silhouette_K[diag_tbl$linkage == ref_link]
ref_part <- cutree(hc_list[[ref_link]], k = ref_K)

ari_one <- function(a, b) {
  tab <- table(a, b)
  N   <- sum(tab)
  sum_choose <- function(x) sum(choose(x, 2))
  index <- sum_choose(tab)
  expected <- sum_choose(rowSums(tab)) * sum_choose(colSums(tab)) / choose(N, 2)
  maxIndex <- 0.5 * (sum_choose(rowSums(tab)) + sum_choose(colSums(tab)))
  if (isTRUE(all.equal(maxIndex, expected))) return(0)
  (index - expected) / (maxIndex - expected)
}

cat(sprintf("\n--- Bootstrap stability of ward.D2 hclust at K=%d (%d reps) ---\n",
            ref_K, B_boot))
ari_boot <- numeric(B_boot)
for (b in seq_len(B_boot)) {
  idx <- sample.int(n, n, replace = TRUE)
  Xb  <- x_rc[idx, , drop = FALSE]
  # Re-row-centre on the bootstrap sample
  Xb  <- Xb - rowMeans(Xb)
  # PCA via SVD on transposed matrix (items as rows)
  Xi  <- t(Xb)
  Xi  <- scale(Xi, center = TRUE, scale = FALSE)
  sv  <- svd(Xi)
  Fb  <- sv$u %*% diag(sv$d)
  Fb  <- Fb[, 1:r, drop = FALSE]
  # hclust with same linkage
  Db  <- dist(Fb)
  hb  <- hclust(Db, method = "ward.D2")
  pb  <- cutree(hb, k = ref_K)
  ari_boot[b] <- ari_one(ref_part, pb)
}
cat(sprintf("Bootstrap ARI: mean = %.3f, median = %.3f, sd = %.3f\n",
            mean(ari_boot), median(ari_boot), sd(ari_boot)))
cat(sprintf("  q10 = %.3f, q90 = %.3f\n",
            quantile(ari_boot, 0.10), quantile(ari_boot, 0.90)))

################################################################################
# 5. Cross-method comparison via ARI: hclust vs k-means vs GMM
################################################################################
ext_compare <- data.frame()
# k-means partition (silhouette-optimal K from script 01)
km_files <- list.files(out_dir, pattern = "^partition_K\\d+_silhouette\\.csv$",
                       full.names = TRUE)
gmm_files <- list.files(out_dir, pattern = "^gmm_partition_G\\d+_.+\\.csv$",
                        full.names = TRUE)

for (link_name in linkages) {
  cl_h <- cutree(hc_list[[link_name]],
                 k = diag_tbl$best_silhouette_K[diag_tbl$linkage == link_name])
  for (km_file in km_files) {
    km_df <- read.csv(km_file, stringsAsFactors = FALSE)
    if (length(km_df$cluster) == P) {
      ext_compare <- rbind(ext_compare,
        data.frame(method_A = paste0("hclust:", link_name),
                   method_B = paste0("kmeans:", basename(km_file)),
                   ARI = round(ari_one(cl_h, km_df$cluster), 3)))
    }
  }
  for (gmm_file in gmm_files) {
    gmm_df <- read.csv(gmm_file, stringsAsFactors = FALSE)
    if (length(gmm_df$cluster) == P) {
      ext_compare <- rbind(ext_compare,
        data.frame(method_A = paste0("hclust:", link_name),
                   method_B = paste0("gmm:", basename(gmm_file)),
                   ARI = round(ari_one(cl_h, gmm_df$cluster), 3)))
    }
  }
}
cat("\n--- Cross-method ARI ---\n")
print(ext_compare, row.names = FALSE)

################################################################################
# 6. Plots
################################################################################
# (A) 4 dendrograms (one per linkage) with optimal cut highlighted
pdf(file.path(out_dir, "hclust_dendrograms.pdf"), width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(8, 4, 3, 1))
for (i in seq_along(linkages)) {
  K_opt <- diag_tbl$best_silhouette_K[i]
  plot(hc_list[[i]], labels = item_names, cex = 0.5,
       main = sprintf("%s  (cophenetic = %.2f, AC = %.2f, K* = %d, sil = %.2f)",
                      linkages[i],
                      diag_tbl$cophenetic_corr[i],
                      diag_tbl$agglom_coef[i],
                      K_opt,
                      diag_tbl$best_silhouette_val[i]),
       xlab = "", sub = "", hang = -1)
  rect.hclust(hc_list[[i]], k = K_opt, border = 2:(K_opt + 1))
}
dev.off()

# (B) Cophenetic-distance scatter for each linkage
pdf(file.path(out_dir, "hclust_cophenetic_scatter.pdf"), width = 12, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (i in seq_along(linkages)) {
  coph <- cophenetic(hc_list[[i]])
  plot(as.vector(D), as.vector(coph), pch = 20, cex = 0.4,
       xlab = "Original Euclidean distance",
       ylab = "Cophenetic distance",
       main = sprintf("%s  (corr = %.3f)",
                      linkages[i], diag_tbl$cophenetic_corr[i]))
  abline(0, 1, col = "red", lty = 2)
}
dev.off()

# (C) Silhouette-vs-K curves overlaid for the four linkages
pdf(file.path(out_dir, "hclust_silhouette.pdf"), width = 10, height = 6)
par(mar = c(4, 4, 3, 1))
matplot(seq_len(K_max), sil_mat, type = "b", pch = 1:length(linkages),
        col = seq_along(linkages), lty = 1,
        xlab = "K (number of clusters from hclust cut)",
        ylab = "Average silhouette width",
        main = "Silhouette vs K  per linkage")
legend("bottomright", legend = linkages,
       col = seq_along(linkages), pch = 1:length(linkages),
       lty = 1, bty = "n", cex = 0.8)
abline(h = 0.5, col = "gray70", lty = 3)
dev.off()

# (D) Heatmap with hierarchical reordering (ward.D2)
pdf(file.path(out_dir, "hclust_heatmap_ward.pdf"), width = 10, height = 9)
ord <- hc_list[["ward.D2"]]$order
D_mat <- as.matrix(D)
rownames(D_mat) <- colnames(D_mat) <- item_names
par(mar = c(7, 7, 3, 1))
image(seq_len(P), seq_len(P),
      D_mat[ord, ord],
      col = colorRampPalette(c("steelblue", "white"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("Pairwise distance heatmap reordered by ward.D2  (K* = %d)",
                     diag_tbl$best_silhouette_K[1]))
axis(1, at = seq_len(P), labels = item_names[ord], las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P), labels = item_names[ord], las = 2, cex.axis = 0.55)
box()
# overlay K* cluster boundaries
cl_ord <- cutree(hc_list[["ward.D2"]], k = diag_tbl$best_silhouette_K[1])[ord]
brks <- which(diff(cl_ord) != 0) + 0.5
abline(h = brks, v = brks, col = "red", lwd = 1.2)
dev.off()

################################################################################
# 7. Save final ward.D2 partition + diagnostics
################################################################################
write.csv(diag_tbl,
          file.path(out_dir, "hclust_diagnostics.csv"),
          row.names = FALSE)
write.csv(round(sil_mat, 4),
          file.path(out_dir, "hclust_silhouette_per_K.csv"))

ward_K <- diag_tbl$best_silhouette_K[diag_tbl$linkage == "ward.D2"]
ward_partition <- cutree(hc_list[["ward.D2"]], k = ward_K)
write.csv(data.frame(item     = item_names,
                     cluster  = ward_partition,
                     F_r),
          file.path(out_dir, sprintf("hclust_ward_K%d.csv", ward_K)),
          row.names = FALSE)

# Append to a hierarchy summary
sink(file.path(out_dir, "hclust_summary.txt"))
cat(sprintf("Hierarchical clustering on item factor scores  (r = %d)\n", r))
cat(sprintf("Source factor scores: %s\n\n", scores_csv))
cat("Validity diagnostics per linkage:\n")
print(transform(diag_tbl,
                cophenetic_corr     = round(cophenetic_corr, 3),
                agglom_coef         = round(agglom_coef, 3),
                best_silhouette_val = round(best_silhouette_val, 3)),
      row.names = FALSE)
cat("\nGuideline (rule-of-thumb thresholds):\n")
cat("  cophenetic_corr > 0.75 -> hierarchy faithfully preserves input distances\n")
cat("  agglom_coef     > 0.65 -> moderate clustering structure (1 = perfect)\n")
cat("  num_inversions  == 0   -> monotone dendrogram\n\n")
cat(sprintf("Bootstrap stability (ward.D2, K = %d, B = %d):\n",
            ref_K, B_boot))
cat(sprintf("  ARI mean = %.3f, median = %.3f, sd = %.3f, q10 = %.3f, q90 = %.3f\n",
            mean(ari_boot), median(ari_boot), sd(ari_boot),
            quantile(ari_boot, 0.10), quantile(ari_boot, 0.90)))
cat("\nGuideline:\n")
cat("  ARI mean > 0.7  -> highly stable\n")
cat("  ARI mean > 0.5  -> moderately stable\n")
cat("  ARI mean < 0.4  -> unstable / hierarchy not robust\n\n")
cat("Cross-method ARI:\n")
print(ext_compare, row.names = FALSE)
sink()

cat(sprintf("\nDiagnostics + plots saved under: %s\n", out_dir))
