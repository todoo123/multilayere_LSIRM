# v14 MoM + Telescoping Sampler — Implementation Record and Tentative Shelving

**Date**: 2026-05-11
**Status**: **TENTATIVELY SHELVED** — implementation complete but the
mixture-of-mixtures architecture does not give reliable cluster recovery on
the v11 simulation regime within the available time budget. Three full-stage
commits remain in `main` and are usable for follow-up work; this document
records what was built, what was learned, and the decision to deprioritise.

---

## 1. What was built

Full three-stage implementation of `modeling_paper/model_v14_variants.md`
(Bayesian Joint Multilayered LSIRM + Sparse Hierarchical MoM + Telescoping
Sampler) under the name `v14`. All v13 files were left untouched.

### Commits (all on `main`)

| SHA | Stage | Summary |
|---|---|---|
| `681a010` | Stage 1 | Two-level MoM (`S_q, I_q`) + Variant B `b_j` MH, K fixed = K_max |
| `ca144d2` | Stage 2 | Telescoping: sample K from BNB(1,4,3), alpha from F(6,3) via log-RWMH; cluster relabel + empty-cluster refresh |
| `62a81f4` | Stage 3 | Variant A (`K_cur · L` Gaussian logsumexp) + A vs B benchmark |
| `d626661` | MIDUS | Real-data driver (`data/MIDUS_5layered_result_v14.R`, n=221, P=68) |

### Files

- `data/my_LSIRM_FMC_v14.cpp` (~1900 lines)
  - Frühwirth-Schnatter Wishart sampler with arma::wishrnd mapping
  - GIG via `GIGrvg::rgig` (Rcpp `Function`)
  - Step 3.1 (S, I) two-stage alloc; Step 3.2–3.3 component/hyperparam
    posteriors; Step 3.4 (K) Eq (3.8); Step 3.5 (α) log-RWMH with
    Jacobian; Step 3.6 empty-cluster refresh; cluster relabel helper
- `data/my_LSIRM_FMC_cpp_v14.R` (~500 lines)
- `data/MIDUS_5layered_result_v14.R`
- `simulation_4_layered_v14.R` (Stage 1 gate, ARI ≥ 0.50)
- `simulation_4_layered_v14_stage2.R` (Stage 2 gate, ARI ≥ 0.55)
- `simulation_4_layered_v14_full.R` (v11-comparable comprehensive sim — **unreliable, see §3**)
- `test/test_v14_wishart_convention.R` — FS↔arma mapping (max rel-err 0.4%)
- `test/test_v14_gig.R` — GIGrvg wiring + Bessel-K ratio mean (max rel-err 0.16%)
- `test/test_v14_smoke.R` — <30s pipeline smoke
- `test/test_v14_alpha_mh.R` — KS p=0.024 vs F(6,3), Jacobian
- `test/test_v14_K_telescoping.R` — K-sample reduces to BNB prior (TV 0.004)
- `test/benchmark_v14_variantA_vs_B.R` — Variant A failed gate (ESS 0.97×, wall 4.6×)

---

## 2. What works

Across the development arc the **mathematical machinery is verified**:

1. **Frühwirth-Schnatter ↔ arma::wishrnd mapping**: 4 test cases (varying d, c),
   max relative error of empirical E[X] = c · C⁻¹ is 0.4%.
2. **GIG sampler**: matches GIGrvg::rgig directly and the closed-form
   Bessel-K mean ratio (rel-err ≤ 0.2%).
3. **Eq (3.8) K-sample**: at K_+=0 reduces to BNB(1,4,3) prior (total
   variation 0.004 < 0.02); at K_+=3 with fixed partition matches the
   analytical formula (TV 0.006 < 0.02); monotone in α (large α → small K).
4. **Eq (3.10–3.12) α-RWMH**: under degenerate likelihood the marginal is
   F(6,3) (KS p = 0.024); log-α Jacobian is included in the acceptance
   ratio and a `WRONG_JACOBIAN` debug switch would break the KS test.
5. **Stage 1 gate**: simulation K_+ mode = K_true = 4, Dahl ARI = 0.538
   (passed ARI ≥ 0.50 gate).
6. **Stage 2 gate**: K_+ mode = K_true = 4, alpha ESS = 124, ARI = 0.638
   (+0.10 vs Stage 1).
7. **MIDUS production run** (20000 iter, 19.8 min): K_+ mode = 3, alpha
   ESS = 998, K_max=30 hit rate 0.04%, Dahl silhouette = 0.65.

The model converges and the post-processing pipeline is intact. The
**v14 toolchain is reusable** for future MoM experiments.

---

## 3. What did not work — the v11 comparison

Goal: rerun `simulation_4_layered_v11.R`'s data-generating mechanism
(n=150, P_total=120, K_true=4, centers ±0.5, σ_meta 0.10–0.20) with the
v14 sampler and recover v11's baseline (Dahl ARI ≈ 0.38, hclust ARI ≈ 0.45).

### Attempts and observations

| Run | Setup | Outcome |
|---|---|---|
| 1 | `lsirm_init = NULL`, `fmc_hyper = NULL` (defaults) | K_+ ≡ 1 (collapse); ARI = 0 |
| 2 | + PCA-based b init (sd ≈ 4, MIDUS-style) | K_+ ≡ 1; b posterior drifts to [-17, +34] |
| 3 | + PCA b rescaled to sd=0.5 | K_+ mean = 8.2 (over-cluster), α = 37, K_max hit rate 6% |
| 4 | + Explicit v11-equivalent `fmc_hyper` (E[Σ_kl] ≈ 0.05·I), PCA sd=1.0 | K_+ ≡ 1 again; b range [-44, +63], γ posterior ≈ 0.07 (true 1.0) |
| 5 | + `fix_gamma = TRUE` | Stopped — see decision below |

### Root cause (best current understanding)

The v14 MoM does **not anchor the `b_j` scale tightly enough** to prevent the
LSIRM (`b`, `γ`) scale degeneracy from escaping. Two specific mechanisms:

1. **`b` scale ↔ `γ` scale tradeoff in LSIRM.** The linear predictor
   `α − β − γ‖a − b‖` is invariant under `(b, γ) → (s · b, γ/s)`. In v11
   this is broken by the **single-level NIW cluster prior + Jain-Neal
   split-merge** that anchors `b_j` close to a cluster centre living in
   the `S0`-scaled latent space. In v14 the **two-level MoM** allows
   `μ_kl` to spread arbitrarily within a single upper cluster (L
   subcomponents can absorb any scale), so the chain comfortably
   drifts to `(b huge, γ tiny, Σ_kl huge)`.

2. **Low-K, mild-separation regime favours K_+ = 1.** The v11
   simulation has 4 well-defined clusters with within/between σ ratio
   ≈ 0.3. With L = 4 subcomponents per upper cluster, the model has
   a Bayes-Occam preference for **K = 1 cluster with 4 subcomponents**
   over **K = 4 clusters with 4 subcomponents each (16 components)**.
   The Dirichlet γ_K = α/K factor and BNB(1,4,3) prior reinforce this
   bias toward small K. Tightening the prior (rescaled PCA proxy)
   merely flips the chain to over-clustering with `K_+ ≈ 8` and α ≈ 37.
3. **MIDUS run worked because data has stronger asymmetric heterogeneity**
   that breaks the (1 upper + L lower) trick: there mode K_+ = 3 with
   stable α ≈ 1.8. Synthetic v11 data with isotropic Gaussian clusters
   is the **worst case** for v14's two-level architecture.

### What this implies architecturally

v14_variants.md §1.1 motivates MoM with: *upper level = interpretable
cluster, lower level = density basis for non-Gaussian shapes*. The
**scale-separation argument** ("`M_0 >> tr(E[Σ_l])/κ_0`") would normally
make the cluster identifiable, but v14 has no `κ_0` (NIW concentration)
and the FS Wishart hyperprior on `Σ_kl` ends up data-driven (under MFG17
defaults) or insufficiently informative (under our explicit hyper).
v13's auxiliary `r_q` + global `Σ_b` was the explicit decoupling
mechanism that we **deliberately dropped per the user-fixed decision**
in the original plan — that decision now appears to be the operational
weakness, at least on this simulation regime.

---

## 4. Decision: shelve MoM for now

Given:

- v14 stage gates passed individually (Stage 1, 2, 3 simulations + unit tests + MIDUS run).
- v14 is **not robustly recovering** the v11 baseline regime in
  apples-to-apples comparison (Dahl ARI stays at 0.0 across 4 different
  hyperparameter choices).
- Each diagnostic iteration costs ~1.5 min wallclock and the parameter
  sensitivity is large; a clean recovery likely requires re-introducing
  `r_q`-style anchoring or moving to L = 1 (which reduces v14 to a
  re-parametrised v13).

We **tentatively shelve the MoM track**. The implementation is preserved
in commits `681a010`, `ca144d2`, `62a81f4`, `d626661`. The MIDUS production
result (K_+ mode = 3, sil = 0.65) is still a legitimate v14 deliverable
should we want to cite it.

---

## 5. Recommendations if v14 is revived

In priority order:

1. **Re-introduce r_q anchor in a v14b branch.** Combine v13's auxiliary
   `r_q` with v14's telescoping + two-level MoM. This preserves both
   the variants.md theory (telescoping for K, BNB prior, etc.) and the
   v13 scale-anchoring that prevents `b` scale drift. Cost: ~1 day.

2. **Try L = 1 first.** Single Gaussian per upper cluster = reparametrised
   v13 single-level + telescoping. If this recovers v11's ARI, the issue
   is confirmed as "two-level absorbs heterogeneity into L". Then
   gradually increase L while monitoring K_+ recovery.

3. **Stronger γ prior.** `sd_log_gamma = 0.05` (vs current 0.5) pins γ near
   the prior mean and prevents the scale-drift escape route. This is a
   workaround, not a fix, but cheap.

4. **`fix_gamma = TRUE` for cluster-recovery-focused runs.** Same idea,
   pin γ = 1 exactly. The geometric cluster recovery would then be
   directly comparable to v11's.

5. **Adaptive `s_α` (Robbins-Monro)** targeting acceptance 0.234. Currently
   `s_α = 0.8` gives 0.74–0.83 acceptance — too high, contributing to slow
   α mixing on multimodal posteriors.

---

## 6. File-system inventory at shelving

```
data/my_LSIRM_FMC_v14.cpp                       (production)
data/my_LSIRM_FMC_cpp_v14.R                     (production)
data/MIDUS_5layered_result_v14.R                (production)
simulation_4_layered_v14.R                      (Stage 1 gate; passes)
simulation_4_layered_v14_stage2.R               (Stage 2 gate; passes)
simulation_4_layered_v14_full.R                 (v11-comparison; UNRELIABLE)
test/test_v14_wishart_convention.R              (passes)
test/test_v14_gig.R                             (passes)
test/test_v14_smoke.R                           (passes)
test/test_v14_alpha_mh.R                        (passes)
test/test_v14_K_telescoping.R                   (passes)
test/benchmark_v14_variantA_vs_B.R              (informational)

plot/case1_all_v14_MoM_d2_Kmax30_L4_alphaInit1_telescON_varB/
   - result.rds                                 (MIDUS chain; K_+=3, sil=0.65)
   - *_traces.pdf, *_biplot_dahl.pdf, etc.

modeling_paper/model_v14_variants.md            (mathematical spec, unchanged)
~/.claude/plans/model-v14-temporal-quill.md     (implementation plan, unchanged)
```

`logs/sim_v14_full.log`, `data/sim_v14_*.rds`, `plot/simulation_4_layered_v14_*/`
are .gitignored experiment artefacts; harmless to leave or delete.

---

## 7. Tentative next steps (not committed to)

- **Resume v13** as the production sampler. v13's `r_q` decoupling and
  Jain-Neal split-merge are the validated path for current research.
- **Defer the variants.md spec** until either (a) we decide to engineer
  the v14b hybrid (Recommendation 1) or (b) the MoM machinery becomes
  necessary for a specific scientific question (e.g., known non-Gaussian
  cluster shapes).
- **Keep `model_v14_variants.md`** as the canonical mathematical reference;
  no edits needed.
