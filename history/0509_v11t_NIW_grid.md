# 0509 v11t NIW prior grid search (S0 × nu0 × kappa0)

## 1. Motivation

Building on the same-day v13 detour (`0509_v13_history.md`), we returned
to v11t and ran the MIDUS pipeline with the stabilization defaults
(`K_star=6`, `kappa0=1.0`, `nu_t=3`).  Two issues were observed:

1. **Item-position trace multimodality** persisted even after Procrustes
   alignment (mixture-induced, not rotation-related).
2. **Item positions concentrated near the origin** in the biplot —
   suggesting cluster centers were anchored too tightly to `m_0 = 0` and
   within-cluster spread was too small.

The fix candidates were the three NIW prior hyperparameters:

| Knob | Direct effect | Stabilization vs spread |
|---|---|---|
| `S0` | Within-cluster Sigma_l prior mean | Larger → cluster spreads more |
| `nu0` | Concentration of Sigma_l prior | Lower → looser, more spread + more wandering |
| `kappa0` | mu_l prior precision (given Sigma_l) | Lower → cluster centers spread more, more wandering |

We ran a **3 × 3 × 3 = 27-cell grid** (script: `exp_v11t_NIW_grid.R`):

```r
S0_grid     <- c(1.0, 2.0, 3.0)         # current value 1.0
nu0_grid    <- c(12, 8, 6)               # current value 12
kappa0_grid <- c(1.0, 0.5, 0.1)          # current value 1.0
```

Constants per cell: `K_star=6`, `e0=0.05`, `nu_t=3`, `nu2=4`,
`n_split_merge=100`, MCMC `15000 / 5000 / thin=5` (n_save=2000).

## 2. Outputs (artefacts on disk)

Master directory: `plot/exp_v11t_NIW_grid/`

- `exp_v11t_NIW_grid_summary.csv` (27 rows × 23 metric columns)
- `exp_v11t_NIW_grid_REPORT.txt` (sorted-by-silhouette + best-by-* recommendations)
- `exp_v11t_NIW_grid_silhouette_heatmap.pdf`  — S0 × nu0 heatmap, kappa0 facets
- `exp_v11t_NIW_grid_Kplus_heatmap.pdf`
- `exp_v11t_NIW_grid_split_rate_heatmap.pdf` / `*_merge_rate_heatmap.pdf`
- `exp_v11t_NIW_grid_b_origin_dist_heatmap.pdf`  (geometry-compression diagnostic)
- `exp_v11t_NIW_grid_pear_heatmap.pdf`           (chain partition stability)

Per-cell `cell_<S0>_<nu0>_<kap>/`:
- `trace_K_plus.pdf` (chain-level K_+ trace)
- `biplot_hclust.pdf` (cluster-coloured biplot)
- `co_cluster_hclust_order.pdf` (PSM heatmap)
- `samps_compact.rds` (R-readable compact RDS for interactive analysis)
- `cell_summary.txt` (text dump)

Log: `exp_v11t_NIW_grid.log`.  Total runtime: 116.3 minutes.

## 3. Three primary trade-off frontiers

Each row picks the cell that wins on the named criterion.

### Best silhouette (b-based cluster separation)

| Rank | S0 | nu0 | kappa0 | sil  | hclust K | Dahl K | PEAR  | b_origin |
|------|----|-----|--------|------|----------|--------|-------|----------|
| 🥇   | 1  | 12  | 0.5    | 0.755| 2        | 2      | 0.505 | 0.396    |
| 🥈   | 1  | 6   | **0.1**| 0.753| 2        | 2      | 0.351 | **0.879**|
| 🥉   | 1  | 6   | 1.0    | 0.731| 2        | 2      | 0.497 | 0.491    |

### Best PEAR (posterior partition concentration / chain stability)

| Rank | S0 | nu0 | kappa0 | PEAR  | sil  | b_origin |
|------|----|-----|--------|-------|------|----------|
| 🥇   | 3  | 6   | 0.1    | 0.600 | 0.490| 1.119    |
| 🥈   | 1  | 12  | 0.5    | 0.505 | 0.755| 0.396    |
| 🥉   | 1  | 6   | 1.0    | 0.497 | 0.731| 0.491    |

### Largest b origin distance (least geometry compression)

| Rank | S0 | nu0 | kappa0 | b_origin | sil  | hclust K |
|------|----|-----|--------|----------|------|----------|
| 🥇   | 2  | 8   | 0.1    | 1.707    | 0.453| 3        |
| 🥈   | 3  | 8   | 0.1    | 1.652    | 0.391| 2        |
| 🥉   | 1  | 8   | 0.1    | 1.246    | 0.342| 3        |

## 4. Pattern reading

1. **kappa0 ↓ ⇒ cluster centres spread ⇒ b_origin ↑** (geometry compression
   relieved).  Lowest kappa0=0.1 cells dominate the b_origin column.
2. But **kappa0=0.1 also raises K_+ posterior variance and lowers
   silhouette** in many cells — mu_l wanders too much when very loose.
3. **S0=1 dominates the silhouette frontier**; S0=2 and S0=3 inflate
   within-cluster Sigma_l enough that the marginal cluster predictive
   becomes diffuse → merges accepted too easily → silhouette drops.
4. **nu0=6 (loosest concentration) gives the most stable silhouettes
   across S0 values**.  nu0=8 is a noisy middle ground (sometimes
   collapses to single Dahl cluster).
5. The high-PEAR cells (cell 27) trade silhouette for trace stability —
   suggests genuine multi-mode in MIDUS partitions that no setting fully
   eliminates.

## 5. Recommended next setting

**Cell 25**: `S0_init_scale = 1.0, nu0 = 6, kappa0 = 0.1`

Rationale: silhouette 0.753 (≈ best cell 10), b_origin 0.879 (2.2x cell
10's compression-prone 0.396), Dahl/hclust agree at K=2.  Best
overall trade-off between cluster recovery, partition stability, and
geometry expansion.

If chain stability is the top concern over silhouette, prefer cell 27
(S0=3, nu0=6, kappa0=0.1, PEAR=0.600 — highest of any cell).

## 6. Open issues

- All cells settle on K=2 or 3 (mostly K=2).  No setting recovered a
  larger cluster count even when priors were extremely loose.  This is
  consistent with the MIDUS data structurally supporting only 2–3
  meaningful biomarker/cognition factors.
- PEAR remains below 0.6 in every cell — partition is fundamentally
  multi-modal in posterior; conditional-on-Dahl summaries (which we
  already use) remain the right inference target.
- Item-position trace multimodality is reduced but not eliminated by
  any single setting — multiple chains + R-hat is the next step if
  per-iter trace stability is critical.

## 7. Reproducibility

Master script: `exp_v11t_NIW_grid.R`
- Random seed: `set.seed(0502)` (k-means init)
- v11t cpp/wrapper: `data/my_LSIRM_FMC_v11t.cpp`, `data/my_LSIRM_FMC_cpp_v11t.R`
- Data: `MIDUS_preprocess_v4.R` + `MIDUS_preprocess_refresher_v4.R`
  (combined, case1_all subset)
