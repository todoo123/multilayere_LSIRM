# Direct factor analysis on v8 LSIRM-induced item profiles

Independent benchmark for the FMC clustering result in v8.

## Pipeline

1. Load the most recent `case1_all_v8_fmc_*` result (`*_result.rds`).
2. Compute posterior-mean LSIRM positions: respondents `a` (n × d) and items `b` (P × d, stacked across all 5 layers).
3. Build the **item profile matrix** `x[i, j] = log ||a_i − b_j||` (n × P), then row-centre (matches v8's preprocessing).
4. **Factor analysis** (PCA / SVD, items as observations):
   - Choose `r` (number of factors) via Kaiser cutoff, parallel analysis (recommended), 80%/90% cumulative variance, and scree elbow.
5. **K-means** on item factor scores at chosen `r`, with K = 1..10:
   - Optimal K via WSS elbow, average silhouette width, and gap statistic (3 variants: `Tibs2001SEmax`, `firstSEmax`, `globalmax`).
6. Write all diagnostics + plots to `output/<run_label>/`.

## Run

```sh
cd /Users/hyunseokyoon/Desktop/학교/대학원/Research/joint_LSIRM
Rscript factor_analysis_v8/01_run_factor_analysis.R
```

The script auto-selects the most recently modified v8 result directory under `data/plot/`. To analyse a specific run, edit the `v8_dir` line near the top.

## Outputs

```
factor_analysis_v8/output/<run_label>/
  x_log.rds                  # x_log, x_rc, A_pm, B_pm, item_names
  scree_plot.pdf             # eigenvalues + parallel-analysis q95
  factor_summary.csv         # per-factor eigenvalues, % var, parallel q95
  item_factor_scores.csv     # P x r factor scores at chosen r
  kmeans_validation.pdf      # elbow + silhouette + gap + 2D scatter
  kmeans_validation.csv      # per-K WSS / silhouette / gap / SE
  partition_K{K}_silhouette.csv   # final partition (silhouette-optimal K)
  summary.txt                # console summary written to disk
```

## Dependencies

Base R + `cluster` (already installed). Optional: `ggplot2` (used only if available).

## Notes

- Items are observations (P ≈ 60), respondents are variables (n ≈ 200): `P × n` SVD is fine.
- Row-centring of `x` matches the v8 model preprocessing — eliminates respondent-level baseline variance, leaves item-level structure.
- The script reports multiple criteria for both factor count and cluster count rather than picking one — different criteria favor different choices, and the user should weigh them.
