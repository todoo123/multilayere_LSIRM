################################################################################
# Clustering on Bayesian FMC factor-analysis MCMC samples (v8)
#
# Companion to 01_run_factor_analysis.R, 02_gmm_clustering.R,
# 03_hierarchical_clustering.R.
#
# Difference vs. 01-03:
#   01-03 perform an *external* SVD/PCA factor analysis on the row-centred
#   x_log = log||a_i - b_j|| matrix derived from the LSIRM posterior-mean
#   latent positions. Item factor scores are then a single point estimate.
#
#   This script (04) instead reuses the FA *already performed inside the v8
#   FMC posterior*. The Gibbs sampler stores per-iteration factor scores
#   eta^(t) (P x r) in res$fmc_eta. Each MCMC iteration is one full FA sample.
#
#   Distance metric: items are compared in **respondent-profile space**, i.e.
#   the model-implied profile of item j across the n respondents,
#       hat_x_j = delta + Lambda eta_j   in R^n
#   (delta cancels in differences). The pairwise distance we use is
#       d(j,k) = || Lambda (eta_j - eta_k) ||_2
#              = sqrt( (eta_j - eta_k)^T  M  (eta_j - eta_k) ),  M = Lambda^T Lambda.
#   Equivalently we transform eta with Cholesky factor R of M (R^T R = M):
#       tilde_eta_j = R eta_j   (in r-dim coords with respondent metric),
#   then plain Euclidean on tilde_eta gives the respondent-space distance.
#   Lambda^(t) is not saved per iteration in v8 (save_lambda_full = FALSE),
#   so we plug in Lambda_PM = posterior mean of Lambda; only eta varies in t.
#
#   Two inputs are then constructed:
#     (P1) eta_PM (resp. metric)   = posterior mean of eta in respondent
#                                     metric  (P x r)
#                                  = eta_PM_raw %*% t(R)
#     (P2) D_bar  (resp. metric)   = posterior mean of pairwise distances
#                                     (P x P)
#                                  = (1/T) sum_t || tilde_eta_j^(t) - tilde_eta_k^(t) ||
#
# Pipeline:
#   1. Load v8 result.rds, extract fmc_eta (P x r x T) MCMC samples.
#   2. Build eta_PM (P1) and D_bar (P2).
#   3. Hierarchical clustering -- 4 linkages -- on each input,
#      K = 2..8 silhouette sweep, find optimal K.
#   4. K-means on eta_PM and PAM on D_bar, K = 2..8, silhouette/WSS/gap.
#   5. Check whether the chosen K falls in [2, 8].
#   6. Cross-method ARI: hclust(eta_PM) vs hclust(D_bar) vs kmeans vs PAM.
#   7. Plots + CSVs + summary, all saved with prefix "fmc_clust_*".
################################################################################

suppressMessages({
  library(cluster)   # silhouette, agnes, pam, clusGap
})

################################################################################
# 0. Locate the latest run -- match the convention of 01_run_factor_analysis.R
#
# Source priority:
#   (i)  if `res` already exists in the global workspace, reuse it (no reload)
#   (ii) otherwise load from the latest output dir's matching result.rds.
################################################################################
project_root <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
out_root     <- file.path(project_root, "factor_analysis_v8", "output")
plot_root    <- file.path(project_root, "data", "plot")

run_dirs <- list.dirs(out_root, recursive = FALSE)
stopifnot(length(run_dirs) > 0)
out_dir   <- run_dirs[which.max(file.info(run_dirs)$mtime)]
run_label <- basename(out_dir)
cat(sprintf("Reusing output directory: %s\n", out_dir))

K_max <- 8L      # user-specified cluster grid: 2..8
gap_B <- 100L
seed  <- 20260429
set.seed(seed)

################################################################################
# 1. Load FMC factor-analysis MCMC samples (prefer in-memory `res`)
################################################################################
have_res <- exists("res", envir = .GlobalEnv, inherits = FALSE) &&
            is.list(get("res", envir = .GlobalEnv)) &&
            !is.null(get("res", envir = .GlobalEnv)$fmc_eta)

v8_dir     <- file.path(plot_root, run_label)
result_rds <- file.path(v8_dir, paste0(run_label, "_result.rds"))

if (have_res) {
  res <- get("res", envir = .GlobalEnv)
  cat("Using `res` already present in the workspace (no .rds reload).\n")
} else {
  stopifnot(file.exists(result_rds))
  cat(sprintf("Loading v8 result:        %s\n", result_rds))
  res <- readRDS(result_rds)
}
info <- res$info
P    <- info$P_total
r    <- info$r_fac
n    <- info$n

cat(sprintf("\nDimensions: n = %d  P = %d  r = %d\n", n, P, r))

eta_arr <- res$fmc_eta                # P x r x T
stopifnot(!is.null(eta_arr),
          length(dim(eta_arr)) == 3,
          dim(eta_arr)[1] == P,
          dim(eta_arr)[2] == r)
T_mc <- dim(eta_arr)[3]

cat(sprintf("FMC FA MCMC samples available: T = %d  (eta: %d x %d x %d)\n",
            T_mc, P, r, T_mc))

# Recover item names (consistent with 01)
item_names <- tryCatch({
  ic <- read.csv(file.path(v8_dir, paste0(run_label, "_fmc_item_clusters.csv")),
                 stringsAsFactors = FALSE)
  ic$item
}, error = function(e) paste0("item_", seq_len(P)))
stopifnot(length(item_names) == P)

################################################################################
# 2. Build two inputs in RESPONDENT-PROFILE METRIC
#
#    Distance: d(j,k) = ||Lambda (eta_j - eta_k)||  (Euclidean in R^n)
#    Implementation: transform eta -> tilde_eta = eta %*% t(R),
#                    where R is the Cholesky of M = Lambda^T Lambda
#                    (R^T R = M, R upper triangular, r x r).
#                    Then plain Euclidean on tilde_eta equals the desired
#                    distance in respondent-profile space.
#    Plug-in: Lambda is fixed at posterior mean Lambda_PM (per-iteration
#             not saved by v8; save_lambda_full = FALSE).
################################################################################
Lambda_PM <- res$fmc_Lambda_postmean
stopifnot(!is.null(Lambda_PM), all(dim(Lambda_PM) == c(n, r)))

M <- crossprod(Lambda_PM)              # r x r
R <- chol(M)                           # upper triangular, t(R) %*% R = M
cat(sprintf("\nLambda_PM: %d x %d   ||Lambda||_F = %.3f\n",
            nrow(Lambda_PM), ncol(Lambda_PM),
            sqrt(sum(Lambda_PM^2))))
cat("Eigenvalues of M = Lambda_PM^T Lambda_PM (per-factor variance contribution):\n")
print(round(eigen(M, symmetric = TRUE, only.values = TRUE)$values, 3))

# (P1) Posterior-mean factor scores -- raw and respondent-metric versions
if (!is.null(res$fmc_eta_postmean) &&
    all(dim(res$fmc_eta_postmean) == c(P, r))) {
  eta_PM_raw <- res$fmc_eta_postmean
} else {
  eta_PM_raw <- apply(eta_arr, c(1, 2), mean)
}
rownames(eta_PM_raw) <- item_names
colnames(eta_PM_raw) <- paste0("eta", seq_len(r))

# Transform to respondent-profile metric (used for clustering / kmeans / dist)
eta_PM <- eta_PM_raw %*% t(R)
rownames(eta_PM) <- item_names
colnames(eta_PM) <- paste0("eta_resp", seq_len(r))

# (P2) Posterior-mean pairwise distance in respondent-profile space
cat("\nComputing posterior-mean respondent-profile distance from MCMC samples ...\n")
D_bar <- matrix(0, P, P)
Rt <- t(R)
for (t in seq_len(T_mc)) {
  Et <- eta_arr[, , t] %*% Rt          # P x r in respondent metric
  Dt <- as.matrix(dist(Et, method = "euclidean"))
  D_bar <- D_bar + Dt
}
D_bar <- D_bar / T_mc
rownames(D_bar) <- colnames(D_bar) <- item_names
D_bar_dist <- as.dist(D_bar)
cat(sprintf("D_bar (resp metric): range = [%.3f, %.3f], mean off-diag = %.3f\n",
            min(D_bar[upper.tri(D_bar)]),
            max(D_bar[upper.tri(D_bar)]),
            mean(D_bar[upper.tri(D_bar)])))

# Distance on eta_PM (also in respondent metric, by construction)
D_PM <- dist(eta_PM, method = "euclidean")

# Persist
saveRDS(list(eta_PM      = eta_PM,
             eta_PM_raw  = eta_PM_raw,
             Lambda_PM   = Lambda_PM,
             R           = R,
             D_bar       = D_bar,
             item_names  = item_names),
        file.path(out_dir, "fmc_clust_inputs.rds"))
write.csv(round(eta_PM, 4),
          file.path(out_dir, "fmc_clust_eta_postmean_respmetric.csv"))
write.csv(round(eta_PM_raw, 4),
          file.path(out_dir, "fmc_clust_eta_postmean_raw.csv"))
write.csv(round(D_bar, 4),
          file.path(out_dir, "fmc_clust_Dbar.csv"))

################################################################################
# 3. Hierarchical clustering -- 4 linkages -- on each input
################################################################################
linkages <- c("ward.D2", "average", "complete", "single")

fit_hclust <- function(d) {
  setNames(lapply(linkages, function(m) hclust(d, method = m)), linkages)
}
hc_PM  <- fit_hclust(D_PM)
hc_Db  <- fit_hclust(D_bar_dist)

# Per-linkage diagnostics + silhouette sweep K = 2..K_max
hclust_diag <- function(hc_list, d_input, tag) {
  out <- data.frame(
    input               = tag,
    linkage             = linkages,
    cophenetic_corr     = NA_real_,
    agglom_coef         = NA_real_,
    num_inversions      = NA_integer_,
    best_silhouette_K   = NA_integer_,
    best_silhouette_val = NA_real_,
    stringsAsFactors    = FALSE
  )
  sil_mat <- matrix(NA_real_, nrow = K_max, ncol = length(linkages),
                    dimnames = list(K = paste0("K=", 1:K_max), linkages))
  for (i in seq_along(linkages)) {
    coph <- cophenetic(hc_list[[i]])
    out$cophenetic_corr[i] <- cor(d_input, coph)
    out$num_inversions[i]  <- sum(diff(hc_list[[i]]$height) < 0)
    for (K in 2:K_max) {
      cl <- cutree(hc_list[[i]], k = K)
      if (length(unique(cl)) >= 2) {
        sil_mat[K, i] <- mean(silhouette(cl, d_input)[, "sil_width"])
      }
    }
    out$best_silhouette_K[i]   <- which.max(sil_mat[, i])
    out$best_silhouette_val[i] <- max(sil_mat[, i], na.rm = TRUE)
  }
  list(diag = out, sil = sil_mat)
}

# Agglomerative coefficient via cluster::agnes:
# - eta_PM:   refit identically from eta_PM coordinates
# - D_bar:    refit from the distance matrix directly
agnes_ac <- function(input, is_dist) {
  vapply(linkages, function(lk) {
    method <- if (lk == "ward.D2") "ward" else lk
    if (is_dist) {
      ag <- agnes(input, method = method, diss = TRUE)
    } else {
      ag <- agnes(input, method = method, metric = "euclidean")
    }
    ag$ac
  }, numeric(1))
}

PM <- hclust_diag(hc_PM, D_PM, tag = "eta_PM")
Db <- hclust_diag(hc_Db, D_bar_dist, tag = "D_bar")
PM$diag$agglom_coef <- agnes_ac(eta_PM, FALSE)
Db$diag$agglom_coef <- agnes_ac(D_bar_dist, TRUE)

diag_tbl <- rbind(PM$diag, Db$diag)
sil_PM   <- PM$sil
sil_Db   <- Db$sil

cat("\n--- Hierarchy validity diagnostics ---\n")
print(transform(diag_tbl,
                cophenetic_corr     = round(cophenetic_corr, 3),
                agglom_coef         = round(agglom_coef, 3),
                best_silhouette_val = round(best_silhouette_val, 3)),
      row.names = FALSE)

cat("\n--- hclust silhouette per K (eta_PM input) ---\n")
print(round(sil_PM, 3))
cat("\n--- hclust silhouette per K (D_bar input) ---\n")
print(round(sil_Db, 3))

################################################################################
# 4. K-means on eta_PM, PAM on D_bar, K = 2..8
################################################################################
# K-means on eta_PM (proper kmeans needs coordinates)
km_results <- vector("list", K_max)
wss_vec    <- numeric(K_max); wss_vec[]    <- NA
sil_km_vec <- numeric(K_max); sil_km_vec[] <- NA
for (K in 1:K_max) {
  set.seed(seed + K)
  km <- kmeans(eta_PM, centers = K, nstart = 50, iter.max = 100)
  km_results[[K]] <- km
  wss_vec[K] <- km$tot.withinss
  if (K >= 2) {
    sil_km_vec[K] <- mean(silhouette(km$cluster, D_PM)[, "sil_width"])
  }
}
K_km_silhouette <- if (any(!is.na(sil_km_vec))) which.max(sil_km_vec) else NA_integer_
K_km_elbow      <- {
  drops <- -diff(wss_vec)
  which.max(drops[seq_len(min(K_max - 1, 7))])
} + 1L

# Gap statistic on eta_PM (to mirror 01)
cat(sprintf("\nGap statistic on eta_PM (B = %d) ... ", gap_B))
gap_res <- clusGap(eta_PM, FUN = kmeans, K.max = K_max, B = gap_B,
                   nstart = 25, iter.max = 100)
gap_tab <- gap_res$Tab
cat("done.\n")
K_km_gap_Tibs    <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"],
                          method = "Tibs2001SEmax")
K_km_gap_first   <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"],
                          method = "firstSEmax")
K_km_gap_global  <- maxSE(gap_tab[, "gap"], gap_tab[, "SE.sim"],
                          method = "globalmax")

# PAM on D_bar -- the natural distance-based analogue of kmeans
pam_results <- vector("list", K_max)
sil_pam_vec <- numeric(K_max); sil_pam_vec[] <- NA
for (K in 2:K_max) {
  pm <- pam(D_bar_dist, k = K, diss = TRUE)
  pam_results[[K]] <- pm
  sil_pam_vec[K] <- mean(silhouette(pm$clustering, D_bar_dist)[, "sil_width"])
}
K_pam_silhouette <- which.max(sil_pam_vec)

km_summary <- data.frame(
  K              = seq_len(K_max),
  wss_etaPM      = wss_vec,
  sil_kmeans     = sil_km_vec,
  sil_pam_Dbar   = sil_pam_vec,
  gap            = gap_tab[, "gap"],
  gap_SE         = gap_tab[, "SE.sim"]
)
cat("\n--- K-means / PAM cluster validation table ---\n")
print(round(km_summary, 4))
cat(sprintf("\nOptimal K via:\n"))
cat(sprintf("  k-means on eta_PM, silhouette : %s\n", K_km_silhouette))
cat(sprintf("  k-means on eta_PM, elbow      : %s\n", K_km_elbow))
cat(sprintf("  k-means gap (Tibs2001SEmax)   : %s\n", K_km_gap_Tibs))
cat(sprintf("  k-means gap (firstSEmax)      : %s\n", K_km_gap_first))
cat(sprintf("  k-means gap (globalmax)       : %s\n", K_km_gap_global))
cat(sprintf("  PAM on D_bar,    silhouette   : %s\n", K_pam_silhouette))

# Per-linkage best-K for hclust (already in diag_tbl) -- summarise:
cat("\nOptimal K per hclust linkage (silhouette):\n")
print(diag_tbl[, c("input", "linkage",
                   "best_silhouette_K", "best_silhouette_val")],
      row.names = FALSE)

# Assert the user-requested check: optimal K \in [2, 8]
in_range <- function(k) !is.na(k) && k >= 2 && k <= 8
report_in_range <- function(label, k) {
  cat(sprintf("  %-40s K = %s  %s\n", label, k,
              if (in_range(k)) "[in range 2..8]" else "[OUTSIDE 2..8]"))
}
cat("\n--- Are optimal K's within [2, 8]? ---\n")
report_in_range("k-means/eta_PM (silhouette)",  K_km_silhouette)
report_in_range("k-means/eta_PM (elbow)",        K_km_elbow)
report_in_range("k-means/eta_PM (gap Tibs)",    K_km_gap_Tibs)
report_in_range("PAM/D_bar      (silhouette)",  K_pam_silhouette)
for (i in seq_len(nrow(diag_tbl))) {
  report_in_range(sprintf("hclust/%s/%s", diag_tbl$input[i], diag_tbl$linkage[i]),
                  diag_tbl$best_silhouette_K[i])
}

################################################################################
# 5. Cross-method ARI
################################################################################
ari_one <- function(a, b) {
  tab <- table(a, b)
  N   <- sum(tab)
  sum_choose <- function(x) sum(choose(x, 2))
  index    <- sum_choose(tab)
  expected <- sum_choose(rowSums(tab)) * sum_choose(colSums(tab)) / choose(N, 2)
  maxIndex <- 0.5 * (sum_choose(rowSums(tab)) + sum_choose(colSums(tab)))
  if (isTRUE(all.equal(maxIndex, expected))) return(0)
  (index - expected) / (maxIndex - expected)
}

# Reference partitions per method (using each method's silhouette-optimal K)
parts <- list()
for (i in seq_along(linkages)) {
  K_opt <- diag_tbl$best_silhouette_K[diag_tbl$input == "eta_PM" &
                                      diag_tbl$linkage == linkages[i]]
  parts[[paste0("hc_etaPM_", linkages[i])]] <- cutree(hc_PM[[i]],  k = K_opt)
  K_opt <- diag_tbl$best_silhouette_K[diag_tbl$input == "D_bar" &
                                      diag_tbl$linkage == linkages[i]]
  parts[[paste0("hc_Dbar_",  linkages[i])]] <- cutree(hc_Db[[i]],  k = K_opt)
}
parts[["kmeans_etaPM"]] <- km_results[[K_km_silhouette]]$cluster
parts[["pam_Dbar"]]     <- pam_results[[K_pam_silhouette]]$clustering

ari_mat <- outer(seq_along(parts), seq_along(parts),
                 Vectorize(function(i, j) ari_one(parts[[i]], parts[[j]])))
dimnames(ari_mat) <- list(names(parts), names(parts))
cat("\n--- Cross-method ARI (each at its silhouette-optimal K) ---\n")
print(round(ari_mat, 3))

# Compare against earlier scripts' partitions if present
ext_compare <- data.frame()
km_files  <- list.files(out_dir,
                        pattern = "^partition_K\\d+_silhouette\\.csv$",
                        full.names = TRUE)
gmm_files <- list.files(out_dir,
                        pattern = "^gmm_partition_G\\d+_.+\\.csv$",
                        full.names = TRUE)
hc_files  <- list.files(out_dir,
                        pattern = "^hclust_ward_K\\d+\\.csv$",
                        full.names = TRUE)
for (nm in names(parts)) {
  for (f in c(km_files, gmm_files, hc_files)) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    if ("cluster" %in% names(df) && length(df$cluster) == P) {
      ext_compare <- rbind(ext_compare,
        data.frame(fmc_method = nm,
                   external   = basename(f),
                   ARI        = round(ari_one(parts[[nm]], df$cluster), 3)))
    }
  }
}
if (nrow(ext_compare) > 0) {
  cat("\n--- ARI vs. earlier scripts' partitions ---\n")
  print(ext_compare, row.names = FALSE)
}

################################################################################
# 6. Plots
################################################################################
# (A) Dendrograms -- two pages, one per input
plot_dendro_page <- function(hc_list, diag_sub, tag, file_path) {
  pdf(file_path, width = 14, height = 10)
  par(mfrow = c(2, 2), mar = c(8, 4, 3, 1))
  for (i in seq_along(linkages)) {
    K_opt <- diag_sub$best_silhouette_K[i]
    plot(hc_list[[i]], labels = item_names, cex = 0.5,
         main = sprintf("[%s]  %s   coph=%.2f  AC=%.2f  K*=%d  sil=%.2f",
                        tag, linkages[i],
                        diag_sub$cophenetic_corr[i],
                        diag_sub$agglom_coef[i],
                        K_opt,
                        diag_sub$best_silhouette_val[i]),
         xlab = "", sub = "", hang = -1)
    rect.hclust(hc_list[[i]], k = K_opt, border = 2:(K_opt + 1))
  }
  dev.off()
}
plot_dendro_page(hc_PM, PM$diag, "eta_PM",
                 file.path(out_dir, "fmc_clust_dendrograms_etaPM.pdf"))
plot_dendro_page(hc_Db, Db$diag, "D_bar",
                 file.path(out_dir, "fmc_clust_dendrograms_Dbar.pdf"))

# (B) Silhouette-vs-K curves: one panel per input
pdf(file.path(out_dir, "fmc_clust_silhouette.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
matplot(seq_len(K_max), sil_PM, type = "b", pch = 1:length(linkages),
        col = seq_along(linkages), lty = 1,
        xlab = "K", ylab = "Avg silhouette width",
        main = "hclust on eta_PM")
lines(seq_len(K_max), sil_km_vec, type = "b", pch = 4, col = "black", lty = 2)
legend("bottomright", legend = c(linkages, "kmeans"),
       col = c(seq_along(linkages), "black"),
       pch = c(1:length(linkages), 4),
       lty = c(rep(1, length(linkages)), 2), bty = "n", cex = 0.8)
abline(h = 0.5, col = "gray70", lty = 3)

matplot(seq_len(K_max), sil_Db, type = "b", pch = 1:length(linkages),
        col = seq_along(linkages), lty = 1,
        xlab = "K", ylab = "Avg silhouette width",
        main = "hclust on D_bar (and PAM)")
lines(seq_len(K_max), sil_pam_vec, type = "b", pch = 4, col = "black", lty = 2)
legend("bottomright", legend = c(linkages, "PAM"),
       col = c(seq_along(linkages), "black"),
       pch = c(1:length(linkages), 4),
       lty = c(rep(1, length(linkages)), 2), bty = "n", cex = 0.8)
abline(h = 0.5, col = "gray70", lty = 3)
dev.off()

# (C) K-means validation panel on eta_PM (mirrors 01's layout, K = 2..K_max)
pdf(file.path(out_dir, "fmc_clust_kmeans_validation.pdf"),
    width = 10, height = 8)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(seq_len(K_max), wss_vec, type = "b", pch = 19,
     xlab = "K", ylab = "Total within-cluster SS",
     main = sprintf("Elbow (eta_PM): K=%d", K_km_elbow))
abline(v = K_km_elbow, col = "red", lty = 2)
plot(seq_len(K_max), sil_km_vec, type = "b", pch = 19,
     xlab = "K", ylab = "Avg silhouette",
     main = sprintf("Silhouette (kmeans/eta_PM): K=%s", K_km_silhouette))
if (!is.na(K_km_silhouette)) abline(v = K_km_silhouette, col = "red", lty = 2)
plot(gap_tab[, "gap"], type = "b", pch = 19,
     xlab = "K", ylab = "Gap(K)",
     main = sprintf("Gap (Tibs2001SEmax): K=%s", K_km_gap_Tibs))
arrows(seq_len(K_max), gap_tab[, "gap"] - gap_tab[, "SE.sim"],
       seq_len(K_max), gap_tab[, "gap"] + gap_tab[, "SE.sim"],
       angle = 90, code = 3, length = 0.05, col = "gray60")
abline(v = K_km_gap_Tibs, col = "red", lty = 2)

K_pick <- if (!is.na(K_km_silhouette)) K_km_silhouette else K_km_elbow
km_pick <- km_results[[K_pick]]
plot(eta_PM[, 1], eta_PM[, 2], pch = 19, cex = 1.3,
     col = km_pick$cluster,
     xlab = colnames(eta_PM)[1], ylab = colnames(eta_PM)[2],
     main = sprintf("Items in eta_PM space (kmeans, K=%d)", K_pick))
text(eta_PM[, 1], eta_PM[, 2], labels = item_names, pos = 4, cex = 0.5)
dev.off()

# (D) Heatmap of D_bar reordered by ward.D2
pdf(file.path(out_dir, "fmc_clust_Dbar_heatmap_ward.pdf"),
    width = 10, height = 9)
ord <- hc_Db[["ward.D2"]]$order
par(mar = c(7, 7, 3, 1))
image(seq_len(P), seq_len(P), D_bar[ord, ord],
      col = colorRampPalette(c("steelblue", "white"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf("D_bar heatmap reordered by ward.D2  (K* = %d)",
                     Db$diag$best_silhouette_K[1]))
axis(1, at = seq_len(P), labels = item_names[ord], las = 2, cex.axis = 0.55)
axis(2, at = seq_len(P), labels = item_names[ord], las = 2, cex.axis = 0.55)
box()
cl_ord <- cutree(hc_Db[["ward.D2"]],
                 k = Db$diag$best_silhouette_K[1])[ord]
brks <- which(diff(cl_ord) != 0) + 0.5
abline(h = brks, v = brks, col = "red", lwd = 1.2)
dev.off()

################################################################################
# 7. Save partitions, diagnostics, summary
################################################################################
write.csv(diag_tbl,
          file.path(out_dir, "fmc_clust_hclust_diagnostics.csv"),
          row.names = FALSE)
write.csv(round(sil_PM, 4),
          file.path(out_dir, "fmc_clust_hclust_silhouette_etaPM.csv"))
write.csv(round(sil_Db, 4),
          file.path(out_dir, "fmc_clust_hclust_silhouette_Dbar.csv"))
write.csv(km_summary,
          file.path(out_dir, "fmc_clust_kmeans_pam_validation.csv"),
          row.names = FALSE)
write.csv(round(ari_mat, 3),
          file.path(out_dir, "fmc_clust_cross_method_ARI.csv"))
if (nrow(ext_compare) > 0) {
  write.csv(ext_compare,
            file.path(out_dir, "fmc_clust_vs_external_ARI.csv"),
            row.names = FALSE)
}

# Save the silhouette-optimal partitions (kmeans, PAM, ward on each input)
save_partition <- function(cluster_vec, filename, extra = NULL) {
  df <- data.frame(item = item_names, cluster = cluster_vec,
                   eta_PM_raw, eta_PM)
  if (!is.null(extra)) df <- cbind(df, extra)
  write.csv(df, file.path(out_dir, filename), row.names = FALSE)
}
save_partition(km_results[[K_km_silhouette]]$cluster,
               sprintf("fmc_clust_kmeans_etaPM_K%d.csv", K_km_silhouette))
save_partition(pam_results[[K_pam_silhouette]]$clustering,
               sprintf("fmc_clust_pam_Dbar_K%d.csv", K_pam_silhouette))
ward_PM_K <- PM$diag$best_silhouette_K[PM$diag$linkage == "ward.D2"]
save_partition(cutree(hc_PM[["ward.D2"]], k = ward_PM_K),
               sprintf("fmc_clust_ward_etaPM_K%d.csv", ward_PM_K))
ward_Db_K <- Db$diag$best_silhouette_K[Db$diag$linkage == "ward.D2"]
save_partition(cutree(hc_Db[["ward.D2"]], k = ward_Db_K),
               sprintf("fmc_clust_ward_Dbar_K%d.csv", ward_Db_K))

# Text summary
sink(file.path(out_dir, "fmc_clust_summary.txt"))
cat(sprintf("Clustering on Bayesian FMC factor-analysis MCMC samples\n"))
cat(sprintf("Source result: %s\n", result_rds))
cat(sprintf("MCMC samples : T = %d   (eta: %d x %d x %d)\n", T_mc, P, r, T_mc))
cat(sprintf("Cluster grid : K = 2 .. %d\n", K_max))
cat(sprintf("Distance     : respondent-profile metric, ||Lambda_PM (eta_j - eta_k)||\n"))
cat(sprintf("               (eta transformed via R = chol(Lambda_PM^T Lambda_PM))\n\n"))
cat("Hierarchy diagnostics:\n")
print(transform(diag_tbl,
                cophenetic_corr     = round(cophenetic_corr, 3),
                agglom_coef         = round(agglom_coef, 3),
                best_silhouette_val = round(best_silhouette_val, 3)),
      row.names = FALSE)
cat("\nK-means / PAM validation:\n")
print(round(km_summary, 4))
cat("\nOptimal K per method:\n")
cat(sprintf("  k-means/eta_PM silhouette : K = %s\n", K_km_silhouette))
cat(sprintf("  k-means/eta_PM elbow      : K = %s\n", K_km_elbow))
cat(sprintf("  k-means/eta_PM gap (Tibs) : K = %s\n", K_km_gap_Tibs))
cat(sprintf("  PAM/D_bar      silhouette : K = %s\n", K_pam_silhouette))
cat("  hclust per linkage (silhouette):\n")
for (i in seq_len(nrow(diag_tbl))) {
  cat(sprintf("    %-9s %-9s K = %d\n",
              diag_tbl$input[i], diag_tbl$linkage[i],
              diag_tbl$best_silhouette_K[i]))
}
all_K <- c(K_km_silhouette, K_km_elbow, K_km_gap_Tibs, K_pam_silhouette,
           diag_tbl$best_silhouette_K)
cat(sprintf("\nAll optimal K within [2, 8]?  %s\n",
            if (all(all_K >= 2 & all_K <= 8, na.rm = TRUE)) "YES" else "NO"))
cat(sprintf("Range of optimal K's: [%d, %d]\n",
            min(all_K, na.rm = TRUE), max(all_K, na.rm = TRUE)))
cat("\nCross-method ARI (silhouette-optimal partitions):\n")
print(round(ari_mat, 3))
if (nrow(ext_compare) > 0) {
  cat("\nARI vs earlier scripts' partitions:\n")
  print(ext_compare, row.names = FALSE)
}
sink()

cat(sprintf("\nAll outputs written under: %s\n", out_dir))
cat(sprintf("Files prefixed with 'fmc_clust_' identify this script's outputs.\n"))
