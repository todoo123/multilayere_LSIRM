################################################################################
# Item clustering directly on LSIRM item positions b_j (Option C)
#
# Bypasses the FMC FA layer entirely. Uses b_j^{(t)} MCMC samples from the
# joint LSIRM+FMC chain (v9). Each item j has a 2-d latent position b_j^{(t)}
# at each MCMC iteration. We cluster items by geometric distance in this
# d=2 LSIRM latent space.
#
# Two inputs (analogous to 04):
#   (P1) B_PM  = posterior mean of b_j               (P x d=2)
#   (P2) D_bar = (1/T) sum_t || b_j^{(t)} - b_k^{(t)} ||  (P x P)
#
# Procrustes alignment is applied by the wrapper, so per-iteration b_j are
# already in a common reference frame; pairwise distances are invariant
# under Procrustes anyway.
#
# Pipeline:
#   1. Load v9 result.rds, stack b1..b5 -> B_arr (T x P x d)
#   2. Build B_PM (P1) and D_bar (P2)
#   3. hclust (4 linkages) + kmeans / PAM, K = 2..8 silhouette sweep
#   4. Cross-method ARI + comparison vs in-model PSM partition
#   5. Save plots + CSVs, prefix "bj_clust_*"
################################################################################

suppressMessages({
  library(cluster)   # silhouette, agnes, pam, clusGap
})

################################################################################
# 0. Locate v9 result
################################################################################
project_root <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
plot_root    <- file.path(project_root, "data", "plot")
v9_dirs <- list.dirs(plot_root, recursive = FALSE)
v9_dirs <- v9_dirs[grepl("v9_fmc_", basename(v9_dirs))]
stopifnot(length(v9_dirs) > 0)
v9_dir   <- v9_dirs[which.max(file.info(v9_dirs)$mtime)]
run_label <- basename(v9_dir)
result_rds <- file.path(v9_dir, paste0(run_label, "_result.rds"))
stopifnot(file.exists(result_rds))
cat(sprintf("Using v9 result: %s\n", result_rds))

# Output directory: same as v9 plot dir, prefix bj_clust_*
out_dir <- v9_dir
cat(sprintf("Outputs -> %s\n", out_dir))

K_max <- 8L
gap_B <- 100L
seed  <- 20260430
set.seed(seed)

################################################################################
# 1. Load result, stack b1..b5
################################################################################
have_res <- exists("result", envir = .GlobalEnv, inherits = FALSE) &&
            is.list(get("result", envir = .GlobalEnv)) &&
            !is.null(get("result", envir = .GlobalEnv)$b1)
if (have_res) {
  res <- get("result", envir = .GlobalEnv)
  cat("Using `result` already in workspace.\n")
} else {
  cat("Loading result.rds ...\n")
  res <- readRDS(result_rds)
}

info <- res$info
n <- info$n
P_total <- info$P_total
d <- 2L  # LSIRM latent dim (fixed by v8/v9)

# wrapper aperms b1..b5 to (n_save x P_layer x d)
b_list <- list(res$b1, res$b2, res$b3, res$b4, res$b5)
b_list <- Filter(function(arr) {
  !is.null(arr) && length(dim(arr)) == 3 && dim(arr)[2] > 0
}, b_list)

T_mc <- dim(b_list[[1]])[1]
# Stack along the P axis: result is (T_mc x P x d)
B_arr <- abind::abind(b_list, along = 2)
stopifnot(dim(B_arr)[1] == T_mc, dim(B_arr)[2] == P_total, dim(B_arr)[3] == d)
cat(sprintf("B_arr: %d iter x %d items x %d dim\n", T_mc, P_total, d))

# Item names (consistent with v9 output)
item_names <- tryCatch({
  ic <- read.csv(file.path(v9_dir, paste0(run_label, "_fmc_item_clusters.csv")),
                 stringsAsFactors = FALSE)
  ic$item
}, error = function(e) paste0("item_", seq_len(P_total)))
stopifnot(length(item_names) == P_total)

################################################################################
# 2. Build inputs (B_PM, D_bar)
################################################################################
B_PM <- apply(B_arr, c(2, 3), mean)        # P x d
rownames(B_PM) <- item_names
colnames(B_PM) <- paste0("b", seq_len(d))

cat(sprintf("\nB_PM: range = [%.3f, %.3f], norm range = [%.3f, %.3f]\n",
            min(B_PM), max(B_PM),
            min(sqrt(rowSums(B_PM^2))), max(sqrt(rowSums(B_PM^2)))))

cat("\nComputing posterior-mean pairwise distance D_bar from MCMC samples ...\n")
D_bar <- matrix(0, P_total, P_total)
for (t in seq_len(T_mc)) {
  Dt <- as.matrix(dist(B_arr[t, , ]))  # P x P
  D_bar <- D_bar + Dt
}
D_bar <- D_bar / T_mc
rownames(D_bar) <- colnames(D_bar) <- item_names
D_bar_dist <- as.dist(D_bar)
D_PM <- dist(B_PM)
cat(sprintf("D_bar (LSIRM space): range = [%.3f, %.3f], mean off-diag = %.3f\n",
            min(D_bar[upper.tri(D_bar)]),
            max(D_bar[upper.tri(D_bar)]),
            mean(D_bar[upper.tri(D_bar)])))
cat(sprintf("D_PM:                range = [%.3f, %.3f], mean = %.3f\n",
            min(D_PM), max(D_PM), mean(D_PM)))

# Persist
saveRDS(list(B_PM = B_PM, D_bar = D_bar, item_names = item_names),
        file.path(out_dir, "bj_clust_inputs.rds"))
write.csv(round(B_PM, 4), file.path(out_dir, "bj_clust_B_postmean.csv"))
write.csv(round(D_bar, 4), file.path(out_dir, "bj_clust_Dbar.csv"))

################################################################################
# 3. Hierarchical clustering (4 linkages) on each input
################################################################################
linkages <- c("ward.D2", "average", "complete", "single")
fit_hclust <- function(d) setNames(lapply(linkages, function(m) hclust(d, method = m)), linkages)
hc_PM <- fit_hclust(D_PM)
hc_Db <- fit_hclust(D_bar_dist)

agnes_ac <- function(input, is_dist) {
  vapply(linkages, function(lk) {
    method <- if (lk == "ward.D2") "ward" else lk
    if (is_dist) ag <- agnes(input, method = method, diss = TRUE)
    else         ag <- agnes(input, method = method, metric = "euclidean")
    ag$ac
  }, numeric(1))
}

hclust_diag <- function(hc_list, d_input, tag) {
  out <- data.frame(
    input = tag, linkage = linkages,
    cophenetic_corr = NA_real_, agglom_coef = NA_real_,
    num_inversions = NA_integer_,
    best_silhouette_K = NA_integer_, best_silhouette_val = NA_real_,
    stringsAsFactors = FALSE)
  sil_mat <- matrix(NA_real_, nrow = K_max, ncol = length(linkages),
                    dimnames = list(K = paste0("K=", 1:K_max), linkages))
  for (i in seq_along(linkages)) {
    coph <- cophenetic(hc_list[[i]])
    out$cophenetic_corr[i] <- cor(d_input, coph)
    out$num_inversions[i] <- sum(diff(hc_list[[i]]$height) < 0)
    for (K in 2:K_max) {
      cl <- cutree(hc_list[[i]], k = K)
      if (length(unique(cl)) >= 2)
        sil_mat[K, i] <- mean(silhouette(cl, d_input)[, "sil_width"])
    }
    out$best_silhouette_K[i]   <- which.max(sil_mat[, i])
    out$best_silhouette_val[i] <- max(sil_mat[, i], na.rm = TRUE)
  }
  list(diag = out, sil = sil_mat)
}

PM <- hclust_diag(hc_PM, D_PM, "B_PM")
Db <- hclust_diag(hc_Db, D_bar_dist, "D_bar")
PM$diag$agglom_coef <- agnes_ac(B_PM, FALSE)
Db$diag$agglom_coef <- agnes_ac(D_bar_dist, TRUE)

diag_tbl <- rbind(PM$diag, Db$diag)
sil_PM   <- PM$sil
sil_Db   <- Db$sil

cat("\n--- Hierarchy validity diagnostics ---\n")
print(transform(diag_tbl,
                cophenetic_corr = round(cophenetic_corr, 3),
                agglom_coef = round(agglom_coef, 3),
                best_silhouette_val = round(best_silhouette_val, 3)),
      row.names = FALSE)

cat("\n--- hclust silhouette per K (B_PM input) ---\n")
print(round(sil_PM, 3))
cat("\n--- hclust silhouette per K (D_bar input) ---\n")
print(round(sil_Db, 3))

################################################################################
# 4. K-means on B_PM, PAM on D_bar, K = 2..8
################################################################################
km_results <- vector("list", K_max)
wss_vec    <- numeric(K_max); wss_vec[]    <- NA
sil_km_vec <- numeric(K_max); sil_km_vec[] <- NA
for (K in 1:K_max) {
  set.seed(seed + K)
  km <- kmeans(B_PM, centers = K, nstart = 50, iter.max = 100)
  km_results[[K]] <- km
  wss_vec[K] <- km$tot.withinss
  if (K >= 2) sil_km_vec[K] <- mean(silhouette(km$cluster, D_PM)[, "sil_width"])
}
K_km_silhouette <- if (any(!is.na(sil_km_vec))) which.max(sil_km_vec) else NA_integer_
K_km_elbow      <- {drops <- -diff(wss_vec); which.max(drops[seq_len(min(K_max - 1, 7))])} + 1L

cat(sprintf("\nGap statistic on B_PM (B = %d) ... ", gap_B))
gap_res <- clusGap(B_PM, FUN = kmeans, K.max = K_max, B = gap_B, nstart = 25, iter.max = 100)
gap_tab <- gap_res$Tab
cat("done.\n")
K_km_gap_Tibs <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "Tibs2001SEmax")
K_km_gap_first <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "firstSEmax")
K_km_gap_global <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"], method = "globalmax")

pam_results <- vector("list", K_max)
sil_pam_vec <- numeric(K_max); sil_pam_vec[] <- NA
for (K in 2:K_max) {
  pm <- pam(D_bar_dist, k = K, diss = TRUE)
  pam_results[[K]] <- pm
  sil_pam_vec[K] <- mean(silhouette(pm$clustering, D_bar_dist)[, "sil_width"])
}
K_pam_silhouette <- which.max(sil_pam_vec)

km_summary <- data.frame(
  K = seq_len(K_max), wss_BPM = wss_vec,
  sil_kmeans = sil_km_vec, sil_pam_Dbar = sil_pam_vec,
  gap = gap_tab[, "gap"], gap_SE = gap_tab[, "SE.sim"])
cat("\n--- K-means / PAM cluster validation ---\n")
print(round(km_summary, 4))

cat(sprintf("\nOptimal K via:\n"))
cat(sprintf("  k-means/B_PM silhouette : K = %s\n", K_km_silhouette))
cat(sprintf("  k-means/B_PM elbow      : K = %s\n", K_km_elbow))
cat(sprintf("  k-means gap (Tibs)      : K = %s\n", K_km_gap_Tibs))
cat(sprintf("  k-means gap (first)     : K = %s\n", K_km_gap_first))
cat(sprintf("  k-means gap (global)    : K = %s\n", K_km_gap_global))
cat(sprintf("  PAM/D_bar silhouette    : K = %s\n", K_pam_silhouette))

cat("\nOptimal K per hclust linkage (silhouette):\n")
print(diag_tbl[, c("input", "linkage", "best_silhouette_K", "best_silhouette_val")],
      row.names = FALSE)

# Range check
in_range <- function(k) !is.na(k) && k >= 2 && k <= 8
report_in_range <- function(label, k) {
  cat(sprintf("  %-40s K = %s  %s\n", label, k,
              if (in_range(k)) "[in 2..8]" else "[OUTSIDE]"))
}
cat("\n--- Optimal K within [2, 8]? ---\n")
report_in_range("k-means/B_PM (silhouette)", K_km_silhouette)
report_in_range("k-means/B_PM (elbow)",      K_km_elbow)
report_in_range("k-means/B_PM (gap Tibs)",   K_km_gap_Tibs)
report_in_range("PAM/D_bar    (silhouette)", K_pam_silhouette)
for (i in seq_len(nrow(diag_tbl))) {
  report_in_range(sprintf("hclust/%s/%s", diag_tbl$input[i], diag_tbl$linkage[i]),
                  diag_tbl$best_silhouette_K[i])
}

################################################################################
# 5. Cross-method ARI + vs in-model PSM
################################################################################
ari_one <- function(a, b) {
  tab <- table(a, b); N <- sum(tab); sc <- function(x) sum(choose(x,2))
  idx <- sc(tab); exp <- sc(rowSums(tab)) * sc(colSums(tab)) / choose(N, 2)
  mx  <- 0.5 * (sc(rowSums(tab)) + sc(colSums(tab)))
  if (isTRUE(all.equal(mx, exp))) return(0)
  (idx - exp) / (mx - exp)
}

parts <- list()
for (i in seq_along(linkages)) {
  K_opt <- diag_tbl$best_silhouette_K[diag_tbl$input == "B_PM" &
                                      diag_tbl$linkage == linkages[i]]
  parts[[paste0("hc_BPM_", linkages[i])]] <- cutree(hc_PM[[i]], k = K_opt)
  K_opt <- diag_tbl$best_silhouette_K[diag_tbl$input == "D_bar" &
                                      diag_tbl$linkage == linkages[i]]
  parts[[paste0("hc_Dbar_", linkages[i])]] <- cutree(hc_Db[[i]], k = K_opt)
}
parts[["kmeans_BPM"]] <- km_results[[K_km_silhouette]]$cluster
parts[["pam_Dbar"]]   <- pam_results[[K_pam_silhouette]]$clustering

ari_mat <- outer(seq_along(parts), seq_along(parts),
                 Vectorize(function(i, j) ari_one(parts[[i]], parts[[j]])))
dimnames(ari_mat) <- list(names(parts), names(parts))
cat("\n--- Cross-method ARI ---\n")
print(round(ari_mat, 3))

# Compare vs v9 in-model PSM partition
ext_compare <- data.frame()
v9_part_csv <- file.path(v9_dir, paste0(run_label, "_fmc_item_clusters.csv"))
if (file.exists(v9_part_csv)) {
  v9_df <- read.csv(v9_part_csv, stringsAsFactors = FALSE)
  if (nrow(v9_df) == P_total) {
    for (nm in names(parts)) {
      ext_compare <- rbind(ext_compare,
        data.frame(bj_method = nm,
                   external = "v9_in_model_PSM",
                   ARI = round(ari_one(parts[[nm]], v9_df$fmc_partition), 3)))
    }
  }
}
# Also compare with v8 in-model PSM (well-known 34/34 split)
v8_part_csv <- file.path(plot_root, "case1_all_v8_fmc_r5_K10_e0.1",
                         "case1_all_v8_fmc_r5_K10_e0.1_fmc_item_clusters.csv")
if (file.exists(v8_part_csv)) {
  v8_df <- read.csv(v8_part_csv, stringsAsFactors = FALSE)
  m8 <- merge(data.frame(item = item_names, idx = seq_along(item_names)),
              v8_df[, c("item", "fmc_partition")], by = "item")
  if (nrow(m8) == P_total) {
    m8 <- m8[order(m8$idx), ]
    for (nm in names(parts)) {
      ext_compare <- rbind(ext_compare,
        data.frame(bj_method = nm,
                   external = "v8_in_model_PSM (34/34)",
                   ARI = round(ari_one(parts[[nm]], m8$fmc_partition), 3)))
    }
  }
}
if (nrow(ext_compare) > 0) {
  cat("\n--- ARI vs in-model partitions ---\n")
  print(ext_compare, row.names = FALSE)
}

################################################################################
# 6. Plots
################################################################################
plot_dendro_page <- function(hc_list, diag_sub, tag, file_path) {
  pdf(file_path, width = 14, height = 10)
  par(mfrow = c(2, 2), mar = c(8, 4, 3, 1))
  for (i in seq_along(linkages)) {
    K_opt <- diag_sub$best_silhouette_K[i]
    plot(hc_list[[i]], labels = item_names, cex = 0.5,
         main = sprintf("[%s] %s  coph=%.2f AC=%.2f K*=%d sil=%.2f",
                        tag, linkages[i],
                        diag_sub$cophenetic_corr[i], diag_sub$agglom_coef[i],
                        K_opt, diag_sub$best_silhouette_val[i]),
         xlab = "", sub = "", hang = -1)
    rect.hclust(hc_list[[i]], k = K_opt, border = 2:(K_opt + 1))
  }
  dev.off()
}
plot_dendro_page(hc_PM, PM$diag, "B_PM",
                 file.path(out_dir, "bj_clust_dendrograms_BPM.pdf"))
plot_dendro_page(hc_Db, Db$diag, "D_bar",
                 file.path(out_dir, "bj_clust_dendrograms_Dbar.pdf"))

# Silhouette curves
pdf(file.path(out_dir, "bj_clust_silhouette.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
matplot(seq_len(K_max), sil_PM, type = "b", pch = 1:length(linkages),
        col = seq_along(linkages), lty = 1,
        xlab = "K", ylab = "Avg silhouette", main = "hclust on B_PM")
lines(seq_len(K_max), sil_km_vec, type = "b", pch = 4, col = "black", lty = 2)
legend("bottomright", c(linkages, "kmeans"),
       col = c(seq_along(linkages), "black"),
       pch = c(1:length(linkages), 4),
       lty = c(rep(1, length(linkages)), 2), bty = "n", cex = 0.8)
matplot(seq_len(K_max), sil_Db, type = "b", pch = 1:length(linkages),
        col = seq_along(linkages), lty = 1,
        xlab = "K", ylab = "Avg silhouette", main = "hclust on D_bar (and PAM)")
lines(seq_len(K_max), sil_pam_vec, type = "b", pch = 4, col = "black", lty = 2)
legend("bottomright", c(linkages, "PAM"),
       col = c(seq_along(linkages), "black"),
       pch = c(1:length(linkages), 4),
       lty = c(rep(1, length(linkages)), 2), bty = "n", cex = 0.8)
dev.off()

# Items in LSIRM 2D space — colored by best partition
pdf(file.path(out_dir, "bj_clust_BPM_scatter.pdf"), width = 10, height = 8)
par(mar = c(4, 4, 3, 1))
K_pick <- if (!is.na(K_km_silhouette)) K_km_silhouette else K_km_elbow
km_pick <- km_results[[K_pick]]
plot(B_PM[, 1], B_PM[, 2], pch = 19, cex = 1.4,
     col = km_pick$cluster,
     xlab = "b dim 1", ylab = "b dim 2",
     main = sprintf("Items in LSIRM space (kmeans K=%d, sil=%.3f)",
                    K_pick, sil_km_vec[K_pick]))
text(B_PM[, 1], B_PM[, 2], labels = item_names, pos = 4, cex = 0.55)
abline(h = 0, v = 0, lty = 3, col = "gray70")
# Unit-circle for reference if positions are ~unit-disk constrained
theta <- seq(0, 2*pi, length.out = 200)
lines(cos(theta), sin(theta), col = "gray70", lty = 2)
dev.off()

# D_bar heatmap
pdf(file.path(out_dir, "bj_clust_Dbar_heatmap_ward.pdf"), width = 10, height = 9)
ord <- hc_Db[["ward.D2"]]$order
par(mar = c(7, 7, 3, 1))
image(seq_len(P_total), seq_len(P_total), D_bar[ord, ord],
      col = colorRampPalette(c("steelblue", "white"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("D_bar heatmap, ward.D2 reorder (K* = %d)",
                     Db$diag$best_silhouette_K[1]))
axis(1, at = seq_len(P_total), labels = item_names[ord], las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P_total), labels = item_names[ord], las = 2, cex.axis = 0.55)
box()
cl_ord <- cutree(hc_Db[["ward.D2"]], k = Db$diag$best_silhouette_K[1])[ord]
brks <- which(diff(cl_ord) != 0) + 0.5
abline(h = brks, v = brks, col = "red", lwd = 1.2)
dev.off()

################################################################################
# 7. Save partitions and summary
################################################################################
write.csv(diag_tbl, file.path(out_dir, "bj_clust_hclust_diagnostics.csv"), row.names = FALSE)
write.csv(round(sil_PM, 4), file.path(out_dir, "bj_clust_silhouette_BPM.csv"))
write.csv(round(sil_Db, 4), file.path(out_dir, "bj_clust_silhouette_Dbar.csv"))
write.csv(km_summary, file.path(out_dir, "bj_clust_kmeans_pam_validation.csv"), row.names = FALSE)
write.csv(round(ari_mat, 3), file.path(out_dir, "bj_clust_cross_method_ARI.csv"))
if (nrow(ext_compare) > 0)
  write.csv(ext_compare, file.path(out_dir, "bj_clust_vs_external_ARI.csv"), row.names = FALSE)

save_partition <- function(cluster_vec, filename) {
  df <- data.frame(item = item_names, cluster = cluster_vec, B_PM)
  write.csv(df, file.path(out_dir, filename), row.names = FALSE)
}
save_partition(km_results[[K_km_silhouette]]$cluster,
               sprintf("bj_clust_kmeans_BPM_K%d.csv", K_km_silhouette))
save_partition(pam_results[[K_pam_silhouette]]$clustering,
               sprintf("bj_clust_pam_Dbar_K%d.csv", K_pam_silhouette))
ward_PM_K <- PM$diag$best_silhouette_K[PM$diag$linkage == "ward.D2"]
save_partition(cutree(hc_PM[["ward.D2"]], k = ward_PM_K),
               sprintf("bj_clust_ward_BPM_K%d.csv", ward_PM_K))
ward_Db_K <- Db$diag$best_silhouette_K[Db$diag$linkage == "ward.D2"]
save_partition(cutree(hc_Db[["ward.D2"]], k = ward_Db_K),
               sprintf("bj_clust_ward_Dbar_K%d.csv", ward_Db_K))

sink(file.path(out_dir, "bj_clust_summary.txt"))
cat(sprintf("Clustering on LSIRM b_j MCMC samples (Option C)\n"))
cat(sprintf("Source: %s\n", result_rds))
cat(sprintf("MCMC samples : T = %d  (b: %d items x %d dim)\n", T_mc, P_total, d))
cat(sprintf("Cluster grid : K = 2..%d\n", K_max))
cat("Distance     : Euclidean in LSIRM 2D latent space\n\n")
cat("Hierarchy diagnostics:\n")
print(transform(diag_tbl,
                cophenetic_corr = round(cophenetic_corr, 3),
                agglom_coef = round(agglom_coef, 3),
                best_silhouette_val = round(best_silhouette_val, 3)),
      row.names = FALSE)
cat("\nK-means / PAM validation:\n")
print(round(km_summary, 4))
cat("\nOptimal K per method:\n")
cat(sprintf("  k-means/B_PM silhouette : K = %s\n", K_km_silhouette))
cat(sprintf("  k-means/B_PM elbow      : K = %s\n", K_km_elbow))
cat(sprintf("  k-means/B_PM gap (Tibs) : K = %s\n", K_km_gap_Tibs))
cat(sprintf("  PAM/D_bar    silhouette : K = %s\n", K_pam_silhouette))
cat("  hclust per linkage (silhouette):\n")
for (i in seq_len(nrow(diag_tbl)))
  cat(sprintf("    %-9s %-9s K = %d\n",
              diag_tbl$input[i], diag_tbl$linkage[i], diag_tbl$best_silhouette_K[i]))
all_K <- c(K_km_silhouette, K_km_elbow, K_km_gap_Tibs, K_pam_silhouette,
           diag_tbl$best_silhouette_K)
cat(sprintf("\nAll optimal K within [2, 8]?  %s\n",
            if (all(all_K >= 2 & all_K <= 8, na.rm = TRUE)) "YES" else "NO"))
cat(sprintf("Range of optimal K's: [%d, %d]\n",
            min(all_K, na.rm = TRUE), max(all_K, na.rm = TRUE)))
cat("\nCross-method ARI:\n")
print(round(ari_mat, 3))
if (nrow(ext_compare) > 0) {
  cat("\nARI vs in-model PSM partitions:\n")
  print(ext_compare, row.names = FALSE)
}
sink()

cat(sprintf("\nDone. Outputs prefixed with 'bj_clust_*' under: %s\n", out_dir))
