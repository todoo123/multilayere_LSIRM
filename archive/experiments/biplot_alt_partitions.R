################################################################################
# biplot_alt_partitions.R
#
# Post-hoc biplot generator for v11/v11t/v11tm runs.
#
# For a given case_plot_dir (output directory of MIDUS_5layered_result_*.R),
# this script loads:
#   - the chain RDS:           <prefix>_result.rds
#   - the alt-partitions CSV:  <fmc_prefix>_alt_partitions.csv
# and emits ONE biplot PDF per partition column (P0/P1/P3/P4/P5):
#   <fmc_prefix>_biplot_<PARTITION>.pdf
#
# Usage (from anywhere):
#   Rscript biplot_alt_partitions.R [case_plot_dir]
#
# If no argument is supplied the script auto-picks the most recent
# v11tm_MARGB run directory under data/plot/.  Pass an explicit
# directory to target a different chain.
################################################################################
suppressPackageStartupMessages({
  library(grDevices)
})

args      <- commandArgs(trailingOnly = TRUE)
data_dir  <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM/data"
plot_root <- file.path(data_dir, "plot")

# ----- 1. Resolve case_plot_dir -----
if (length(args) >= 1 && nzchar(args[1])) {
  case_plot_dir <- normalizePath(args[1], mustWork = TRUE)
} else {
  cands <- list.dirs(plot_root, recursive = FALSE, full.names = TRUE)
  cands <- cands[grepl("v11tm_MARGB", basename(cands))]
  if (length(cands) == 0)
    stop("No v11tm_MARGB run dirs found under ", plot_root,
         "; pass case_plot_dir explicitly.")
  mtimes <- file.info(cands)$mtime
  case_plot_dir <- cands[which.max(mtimes)]
  cat(sprintf("[auto] using most recent v11tm dir:\n  %s\n", case_plot_dir))
}
case_name <- basename(case_plot_dir)

# Conventional file names (mirrors driver):
#   prefix     = "<case>_<run_label>"
#   fmc_prefix = paste0(prefix, "_fmc")
prefix     <- case_name
fmc_prefix <- paste0(prefix, "_fmc")

rds_path <- file.path(case_plot_dir, paste0(prefix, "_result.rds"))
alt_path <- file.path(case_plot_dir, paste0(fmc_prefix, "_alt_partitions.csv"))
if (!file.exists(rds_path)) stop("RDS not found: ", rds_path)
if (!file.exists(alt_path))
  stop("alt_partitions.csv not found (post-processing block did not run?): ",
       alt_path)

cat(sprintf("[load] %s\n", rds_path))
result   <- readRDS(rds_path)
alt_part <- read.csv(alt_path, stringsAsFactors = FALSE)
item_names_full <- alt_part$item
P_total <- length(item_names_full)
d <- if (!is.null(result$info$d)) result$info$d else dim(result$a)[3]

# ----- 2. Posterior means of a (respondents) and b (items, stacked) -----
A_hat_pm <- apply(result$a, c(2, 3), mean)
b_arrs   <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
B_hat_pm <- do.call(rbind, lapply(b_arrs, function(arr) {
  if (is.null(arr) || length(dim(arr)) != 3 || dim(arr)[2] == 0)
    return(matrix(0, 0, d))
  apply(arr, c(2, 3), mean)
}))
if (nrow(B_hat_pm) != P_total)
  stop(sprintf("B_hat row count %d != alt_partitions row count %d",
               nrow(B_hat_pm), P_total))
rownames(B_hat_pm) <- item_names_full

cluster_pal <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                 "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
                 "#999999", "#66C2A5")

# ----- 3. Biplot helper (verbatim from MIDUS_5layered_result_v11tm.R) -----
make_fmc_biplot <- function(A_hat, B_hat, item_partition, item_names,
                            title, filename, plot_dir, pal) {
  k_max <- max(item_partition)
  pal_use <- if (k_max > length(pal))
    colorRampPalette(pal)(k_max) else pal[seq_len(k_max)]
  pdf(file.path(plot_dir, filename), width = 10, height = 8)
  par(mar = c(4, 4, 3, 1))
  xr <- range(A_hat[, 1], B_hat[, 1])
  yr <- range(A_hat[, 2], B_hat[, 2])
  plot(A_hat[, 1], A_hat[, 2], pch = 21,
       bg = adjustcolor("gray60", alpha.f = 0.30),
       col = "gray40", cex = 0.7,
       xlab = "Dim1", ylab = "Dim2", main = title,
       xlim = xr + c(-1, 1) * 0.1 * diff(xr),
       ylim = yr + c(-1, 1) * 0.1 * diff(yr))
  points(B_hat[, 1], B_hat[, 2], pch = 22,
         bg = pal_use[item_partition], col = "black", cex = 1.7)
  text(B_hat[, 1], B_hat[, 2], labels = item_names, pos = 4, cex = 0.55)
  uq_part <- sort(unique(item_partition))
  legend("topright",
         legend = c("Respondents (a_i)",
                    sprintf("Item partition %d", uq_part)),
         pch    = c(21, rep(22, length(uq_part))),
         pt.bg  = c(adjustcolor("gray60", alpha.f = 0.30),
                    pal_use[uq_part]),
         col    = c("gray40", rep("black", length(uq_part))),
         bty = "n", cex = 0.75)
  dev.off()
}

# ----- 4. Emit one biplot per partition column -----
part_cols <- setdiff(colnames(alt_part), "item")
short_lbl <- c(
  P0_prev_avgcut             = "P0 avg-link cut",
  P1_minBinder_unconstrained = "P1 minBinder (unconstrained)",
  P3_minBinder_drawsOnly     = "P3 minBinder (draws only)",
  P4_minVI_unconstrained     = "P4 minVI (unconstrained)",
  P5_Dahl_drawsOnly          = "P5 Dahl (draws only)"
)

mean_K_plus <- if (!is.null(result$fmc_K_plus)) mean(result$fmc_K_plus) else NA_real_
sm_split <- if (!is.null(result$fmc_split_merge)) result$fmc_split_merge$split_rate else NA_real_
nu_t_used <- if (!is.null(result$info$nu_t)) result$info$nu_t else NA_real_
e0_used   <- if (!is.null(result$info$e0))   result$info$e0   else NA_real_
K_star    <- if (!is.null(result$info$K_star)) result$info$K_star else NA_integer_

for (pc in part_cols) {
  part <- as.integer(alt_part[[pc]])
  k_unique <- length(unique(part))
  pretty   <- if (pc %in% names(short_lbl)) short_lbl[[pc]] else pc
  filename <- paste0(fmc_prefix, "_biplot_", pc, ".pdf")
  title <- sprintf("v11tm biplot — %s  (K=%d; d=%d, K*=%d, e0=%g, nu_t=%g, mean K_+=%.1f, split=%.2f)",
                   pretty, k_unique, d, K_star, e0_used, nu_t_used,
                   mean_K_plus, sm_split)
  cat(sprintf("  -> %s  (K=%d)\n", filename, k_unique))
  make_fmc_biplot(
    A_hat = A_hat_pm, B_hat = B_hat_pm,
    item_partition = part,
    item_names     = item_names_full,
    title          = title,
    filename       = filename,
    plot_dir       = case_plot_dir,
    pal            = cluster_pal
  )
}

cat(sprintf("\n[done] %d biplots written to:\n  %s\n",
            length(part_cols), case_plot_dir))
