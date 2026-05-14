# 0511 v11tm — Collapsed-label (marginalized) b update

## 1. Idea / motivation

In v11/v11t the item-position MH update uses the **conditional** Gaussian
mixture prior, anchoring `b_q` to `μ_{c_q}` with precision `Σ_{c_q}^{-1}`:

```
log R_b = ΔlogL  -  0.5·w_q·[(b'-μ_c)' Σ_c^{-1} (b'-μ_c) - (b-μ_c)' Σ_c^{-1} (b-μ_c)]
```

`Σ_{c_q}` plays a dual role: defines cluster shape AND anchors `b_q`.  Tight
`Σ_l` compresses within-cluster pairwise distances → biplot S-curve distortion.
The `b_prior_inflation` knob (`α≥1`) in v11 is a sampler-level remedy for this
that just tempers the prior contribution by `1/α` — non-coherent.

**v11tm idea**: integrate `(c_q, ρ, μ, Σ)` out of the b MH prior factor
entirely, so the prior becomes the **leave-one-out NIW Student-t mixture
predictive** (same machinery as the c-update):

```
A(b; q, w_q) = log Σ_l (n_{l,-q} + e0) · t_{ν_n - d + 1}(b; m_n, V_pred(w_q))
log R_b = ΔlogL  +  A(b'; q, w_q) - A(b; q, w_q)
```

Same joint posterior as v11t conditional kernel — this is a **partially
collapsed Gibbs move** (van Dyk & Park 2008).  Sweep ordering already
satisfies the validity rule because `(c, ρ, μ, Σ)` are refreshed
immediately after the b block.

Documented in `modeling_paper/model_v11_latent_position_mixture.tex`
eqs.(b-marginal-prior), (b-marginal-A), (b-mh-ratio-marginal), and the
"Validity as a partially collapsed Gibbs move" / "Comparison with
α-inflation" paragraphs (added 0511).

## 2. Implementation files

| File | Purpose |
|---|---|
| `data/my_LSIRM_FMC_v11tm.cpp` | v11t cpp + `marginal_b` flag.  Helper `log_marginal_b_prior(b_val, q, w_q, ...)` computes leave-one-out NIW mixture predictive with log-sum-exp.  Each of the 5 b-loops branches on `marginal_b`; when on, `z_all.row(q)` is incrementally synced after each accepted MH step.  Function renamed to `run_lsirm_fmc_v11tm_cpp`. |
| `data/my_LSIRM_FMC_cpp_v11tm.R` | R wrapper.  New argument `marginal_b = TRUE` (default).  Absolute path to cpp source (avoids `getwd()` shifts during preprocess sourcing). |
| `data/MIDUS_5layered_result_v11tm.R` | Full MIDUS driver (clone of `v11t_stab.R`).  Adds: `case_plot_dir` creation guard with stop-on-fail, silhouette tryCatch + `is.matrix` check, per-partition biplot loop (P0/P1/P3/P4/P5) inline. |
| `biplot_alt_partitions.R` | Standalone post-hoc biplot generator from saved RDS (alternative to inline loop; auto-picks latest v11tm dir). |
| `exp_v11tm_marginal_b.R` | First-pass lean comparison script (e0=0.05).  Saves `exp_v11tm_marginal_b_summary.rds` with co-cluster MAE / ARI vs v11t baseline. |

Modeling paper: `modeling_paper/model_v11_latent_position_mixture.tex`
gained the marginalized-update subsection (~70 lines) plus an Algorithm 1
update enumerating the three b-update kernel options.

## 3. MIDUS results — e0 sweep

All chains: same v11t_stab lock-in setting except `e0`.  K*=6, kappa0=1.0,
nu0=12, S0_init_scale=1, nu_t=3, nu2=20, n_split_merge=100, MCMC
30000/10000/thin=5, marginal_b=TRUE.  MIDUS Wave2 + Refresher1, n=221,
P=68 (20/5/15/28/0).  Comparison to `v11t_STAB` baseline (same setting,
marginal_b=FALSE).

| e0 | split rate | merge rate | net K_+ | mean K_+ | median K_+ | P0 (avg-link cut at median K_+) | P5 Dahl | b-silhouette | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.127 | **0.607** | -105 | 1.79 | 2 | 2 (45/23 forced cut) | **K=1 (collapse)** | n/a | merge cascade → all items co-clustered |
| 0.10 | 0.211 | 0.542 | 0 | 2.47 | 2 | **K=2 (clean)** | K=1 | **+0.360** | sweet spot — geometry separation OK, partition still K=1 in collapsed estimators |
| 1.00 | **0.422** | 0.182 | +22 | 5.5 | 6 | K=6 (forced) | K=6 | -0.02 ~ -0.12 (negative) | over-split, cluster geometry blurred |
| v11t (cond) | 0.100 | n/a | n/a | n/a | 2 | K=2 (61/7) | K=2 | n/a | reference |

**Comparison vs v11t baseline (e0=0.05 v11tm)**:
- co-cluster MAE = 0.1733
- ARI(Dahl) = 0.000
- ARI(hclust avg) = -0.0615
- Geometry within-cluster sd: 0.3563 (v11tm) vs 0.3750 (v11t) — marginal 5% reduction.

**Cluster-content disagreement at e0=0.1 vs v11t baseline**:
- v11t baseline 7-item small cluster: B4BIL6, B4BFGN, B4BCRP (inflammation),
  sgst_reverse_*, sgst_mixed_*_incorrect (cognition).  Biological/cognitive
  outlier axis.
- v11tm e0=0.1 P0 K=2 small cluster: B4Q3L/H/P, B4Q1K/N/Z (binary survey
  items).  Survey-item axis.
- → Two distinct posterior modes; same data, different sampler picks
  different mode.  Not a "wrong vs right" — both are local optima of the
  joint posterior.

## 4. Why marginal_b underperforms here

Mechanism (consistent with all three e0 settings):
- Conditional kernel: `b_q` pulled toward `μ_{c_q}` keeps the cluster
  identifiable (anchor stabilizes assignments).
- Marginal kernel: `b_q` feels the full mixture predictive → drifts
  toward the dominant cluster's centre → c-update's NIW predictive for
  the dominant cluster outperforms competing clusters → positive
  feedback → merge cascade.
- e0 ↑ partially counteracts (gives empty clusters non-trivial weight),
  but the trade-off is over-splitting before the cluster-anchoring
  benefit kicks in.

The α=1 conditional kernel is, on this dataset, more useful than the
α-marginalized kernel — exactly the opposite of the modeling-paper
prediction that marginalization would be the "more principled" remedy
for the geometry-compression failure mode.

## 5. Post-processing / artefact issues encountered (and fixed)

- `getwd()` inside `my_LSIRM_FMC_cpp_v11tm.R::sourceCpp` shifted to
  `MIDUS_refresher_1/` during preprocess sourcing; replaced with absolute
  path.
- `cluster::silhouette()` returns a scalar `NA` (not a matrix) for some
  degenerate partitions (singleton clusters); wrapped in tryCatch +
  `is.matrix(s)` guard.  R `if/else` syntax cleaned up to use explicit
  braces (parser was rejecting bare `else` after multi-line tryCatch).
- `dir.create(case_plot_dir)` silently failed in one run (root cause
  unclear; possibly stale FS state or external cleanup).  Added
  `stop()`-on-failure check + `cat()` echo of `case_plot_dir` exists
  status.
- Removed/restored `saveRDS(result)` line: removed at user request, then
  restored after second e0=1 attempt failed mid-post-processing leaving
  no chain on disk.

## 6. Outputs on disk

Per-run directory: `data/plot/case1_all_v11tm_MARGB_d2_K6_e<E>_S0init_1_nu12_kap1_M100_nut3_nu220/`

Files (e0=0.1 has full set; e0=1.0 missing post-silhouette CSVs from
4-H/4-H-bis/4-H-ter/4-I; e0=0.05 was lean exp script with no plots):

- `*_result.rds` — full chain
- `*_trace_a.pdf`, `*_trace_b{1..5}.pdf`, `*_trace_alpha{1..5}.pdf`,
  `*_trace_beta{1..5}.pdf`, `*_trace_beta4_thr.pdf`, `*_trace_beta5_thr.pdf`,
  `*_trace_lsirm_kappa_per_item.pdf`, `*_trace_lsirm_extra.pdf`
- `*_fmc_trace_K_plus.pdf`, `*_fmc_trace_rho.pdf`, `*_fmc_trace_mu.pdf`,
  `*_fmc_trace_logdetSigma.pdf`, `*_fmc_trace_cluster_sizes.pdf`
- `*_fmc_membership_heatmap_c.pdf`, `*_fmc_co_cluster.pdf`,
  `*_fmc_b_pm_scatter.pdf`
- `*_fmc_biplot.pdf` (default partition) + 5 alt biplots
  (`*_biplot_P0_prev_avgcut.pdf`, `*_biplot_P1_minBinder_unconstrained.pdf`,
  `*_biplot_P3_minBinder_drawsOnly.pdf`, `*_biplot_P4_minVI_unconstrained.pdf`,
  `*_biplot_P5_Dahl_drawsOnly.pdf`)
- `*_fmc_silhouette.pdf` + `*_fmc_silhouette_per_item.csv`
- `*_v11t_ws_post_mean.pdf`, `*_v11t_ws_hist.pdf` + per-item CSV
- `*_fmc_alt_partitions.csv`, `*_fmc_alt_partition_summary.csv`,
  `*_fmc_K_plus_summary.csv`, etc.

Logs in project root:
- `exp_v11tm_marginal_b.log` (e0=0.05 lean run)
- `midus_v11tm_e0_1.0.log` (e0=1.0 attempts 1-3, all failed during post-processing
  but with progressively more output preserved)
- `midus_v11tm_e0_0.1.log` (e0=0.1, exit 0)

Comparison summary RDS: `exp_v11tm_marginal_b_summary.rds` (e0=0.05 only).

## 7. Status / verdict

- **Code works**: marginal_b implemented correctly, partially collapsed
  Gibbs validity is sound, cpp/R chain runs cleanly, post-processing
  pipeline survives all three e0 values.
- **MIDUS empirics**: marginal_b kernel does NOT outperform v11t
  conditional kernel on this dataset.  Best e0 (0.1) gives positive
  silhouette but a different partition mode than v11t baseline; smaller
  e0 collapses; larger e0 over-splits.
- **No clear win** on the original motivating problem (LSIRM-geometry
  compression).  Within-cluster sd reduction at e0=0.05 was 5% — not
  enough to justify losing cluster identification.
- v11t conditional kernel + α-inflation remains the operational baseline
  for MIDUS.

## 8. Memory entries written

- `project_v11_marginalized_b_update_0511.md` — modeling paper §"b update"
  now lists three kernel options (conditional, α-inflated, marginalized).
- `project_v11tm_midus_collapse_0511.md` — MIDUS empirical e0 sweep results
  + sweet-spot at e0=0.1.

## 9. Next-step candidates (NOT executed)

1. Simulation (`simulation_4_layered_v13.R` style) with K_true=4 ground
   truth — does marginal_b beat conditional in ARI when the cluster
   structure is genuinely strong?  MIDUS may simply have weak cluster
   structure (data-driven failure, not sampler failure).
2. Hybrid kernel: marginal_b during burn-in (mode exploration), switch
   to conditional post-burnin (mode locking).  Implementation cost: ~10
   lines in cpp.
3. Document the collapse failure mode in `model_v11_latent_position_mixture.tex`
   with a "Caveat" subsection so future readers don't blindly use marginal_b.
4. e0 scan finer (0.07, 0.15, 0.2) to map the bifurcation more precisely.
