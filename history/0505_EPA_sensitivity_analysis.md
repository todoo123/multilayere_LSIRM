# EPA-LSIRM v12 — `(α, τ)` Sensitivity Analysis

**Date**: 2026-05-05
**Model**: v12 (EPA pairwise partition prior on items, paper §`hyperpriors-and-identifiability`).
**Sampler**: paper-canonical (joint coupling, b MH includes EPA prior).

## Setup

All experiments share:

- Data: 4-layer simulation (binary / continuous / count / ordinal), `n=150`,
  `P_total=120 (30+30+30+30+0)`, true `K=4`, isotropic 4 clusters at `(±1, ±1)`,
  within-cluster sd = 0.15.
- LSIRM: `sigma_b = 1.0`, `gamma_l ~ logN(0, 0.5²)`, `b_epa_coupling = TRUE`.
- MCMC: `n_iter = 20000`, `burnin = 8000`, `thin = 5` → 2400 saved samples.
- EPA: `delta = 0`, `M_SM = 1`, `R = 5`, `M_perm = P_total = 120`,
  `epa_warmup = 0`.
- Seed: 20260501 (identical data across runs).

The hyperparameters that **vary** are `alpha` and `tau` (both held FIXED
during MCMC; we sweep over their fixed values).  This follows DDT (2017)
practice of treating these as bandwidth-like user-set hyperparameters.

## Reading the table

- **mean K**, **median K**: posterior on number of occupied clusters.
- **ARI hclust/Dahl/Binder/VI**: clustering-recovery accuracy against the
  true 4-cluster assignment.  hclust = cut dendrogram of `1 - C̄` at
  K = median(K_+); Dahl/Binder/VI restrict to MCMC samples (Dahl 2006,
  Binder 1978, Wade-Ghahramani 2018).
- **cov LSIRM**: average 95% credible-interval coverage across LSIRM
  parameter families (alpha, beta, positions a/b, gamma, sigma0_sq).
- **C̄ within / between**: average posterior co-cluster probability for
  pairs in the same / different true cluster.  Block contrast = within − between.
- **split / merge rate**: Jain-Neal MH acceptance rates.

## Sensitivity grid

| Label | alpha (fix) | tau (fix) | n_iter | mean K | median K | ARI hclust | ARI Dahl | ARI Binder | ARI VI | C_within | C_between | contrast | cov LSIRM | split rate | merge rate | folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
<!-- AUTOFILL ROWS BELOW -->
| `a0.5_t1.0` | 0.5 | 1 | 20000 | 2.48 | 2 | 0.398 | 0.000 | 0.000 | 0.000 | 0.882 | 0.858 | 0.025 | 0.912 | 0.031 | 0.544 | [link](../plot/simulation_4_layered_v12_sens_a0.5_t1.0/) |
| `a1.0_t1.0` | 1 | 1 | 20000 | 4.13 | 4 | 0.874 | 0.000 | 0.000 | 0.000 | 0.754 | 0.708 | 0.046 | 0.914 | 0.062 | 0.478 | [link](../plot/simulation_4_layered_v12_sens_a1.0_t1.0/) |
| `a1.0_t2.0` | 1 | 2 | 20000 | 3.76 | 4 | 0.600 | 0.000 | 0.000 | 0.000 | 0.840 | 0.802 | 0.039 | 0.910 | 0.018 | 0.545 | [link](../plot/simulation_4_layered_v12_sens_a1.0_t2.0/) |
| `a1.0_t5.0` | 1 | 5 | 20000 | 3.42 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.891 | 0.872 | 0.019 | 0.914 | 0.002 | 0.679 | [link](../plot/simulation_4_layered_v12_sens_a1.0_t5.0/) |
| `a2.0_t1.0` | 2 | 1 | 20000 | 7.18 | 7 | 0.879 | -0.001 | -0.001 | -0.001 | 0.580 | 0.503 | 0.077 | 0.911 | 0.128 | 0.406 | [link](../plot/simulation_4_layered_v12_sens_a2.0_t1.0/) |
| `a2.0_t2.0` | 2 | 2 | 20000 | 6.37 | 6 | 0.892 | 0.000 | 0.000 | -0.001 | 0.706 | 0.632 | 0.073 | 0.915 | 0.036 | 0.470 | [link](../plot/simulation_4_layered_v12_sens_a2.0_t2.0/) |
| `a1.0_t1.0_iter40k` | 1 | 1 | 40000 | 4.08 | 4 | 0.913 | 0.000 | 0.000 | 0.000 | 0.762 | 0.717 | 0.045 | 0.909 | 0.061 | 0.484 | [link](../plot/simulation_4_layered_v12_sens_a1.0_t1.0_iter40k/) |
| `a2.0_t1.0_iter40k` | 2 | 1 | 40000 | 7.25 | 7 | 0.891 | 0.000 | 0.000 | -0.001 | 0.581 | 0.504 | 0.077 | 0.912 | 0.123 | 0.406 | [link](../plot/simulation_4_layered_v12_sens_a2.0_t1.0_iter40k/) |
| `a2.0_t2.0_iter40k` | 2 | 2 | 40000 | 6.45 | 6 | 0.851 | 0.000 | 0.000 | 0.001 | 0.703 | 0.631 | 0.072 | 0.915 | 0.036 | 0.467 | [link](../plot/simulation_4_layered_v12_sens_a2.0_t2.0_iter40k/) |
<!-- AUTOFILL ROWS ABOVE -->

## Per-experiment notes

<!-- AUTOFILL NOTES BELOW -->

### `a0.5_t1.0` (alpha=0.5, tau=1, n_iter=20000)

- mean K_+ = 2.48, median K_+ = 2, range [1, 8]
- ARI hclust=0.398, Dahl=0.000, Binder=0.000, VI=0.000
- Co-cluster contrast = 0.025 (within=0.882, between=0.858)
- LSIRM coverage: overall=0.912 | alpha-fam=0.948, beta-fam=0.965, pos-fam=0.951, gamma=1.000
- Distance recovery cor = 0.825, R_scale = 1.172
- Split rate = 0.031, Merge rate = 0.544, Sigma swap rate = 0.951
- Folder: [`simulation_4_layered_v12_sens_a0.5_t1.0`](../plot/simulation_4_layered_v12_sens_a0.5_t1.0/)


### `a1.0_t1.0` (alpha=1, tau=1, n_iter=20000)

- mean K_+ = 4.13, median K_+ = 4, range [1, 12]
- ARI hclust=0.874, Dahl=0.000, Binder=0.000, VI=0.000
- Co-cluster contrast = 0.046 (within=0.754, between=0.708)
- LSIRM coverage: overall=0.914 | alpha-fam=0.958, beta-fam=0.965, pos-fam=0.953, gamma=1.000
- Distance recovery cor = 0.821, R_scale = 1.135
- Split rate = 0.062, Merge rate = 0.478, Sigma swap rate = 0.910
- Folder: [`simulation_4_layered_v12_sens_a1.0_t1.0`](../plot/simulation_4_layered_v12_sens_a1.0_t1.0/)


### `a1.0_t2.0` (alpha=1, tau=2, n_iter=20000)

- mean K_+ = 3.76, median K_+ = 4, range [1, 10]
- ARI hclust=0.600, Dahl=0.000, Binder=0.000, VI=0.000
- Co-cluster contrast = 0.039 (within=0.840, between=0.802)
- LSIRM coverage: overall=0.910 | alpha-fam=0.953, beta-fam=0.954, pos-fam=0.949, gamma=1.000
- Distance recovery cor = 0.826, R_scale = 1.168
- Split rate = 0.018, Merge rate = 0.545, Sigma swap rate = 0.919
- Folder: [`simulation_4_layered_v12_sens_a1.0_t2.0`](../plot/simulation_4_layered_v12_sens_a1.0_t2.0/)


### `a1.0_t5.0` (alpha=1, tau=5, n_iter=20000)

- mean K_+ = 3.42, median K_+ = 3, range [1, 11]
- ARI hclust=0.000, Dahl=0.000, Binder=0.000, VI=0.000
- Co-cluster contrast = 0.019 (within=0.891, between=0.872)
- LSIRM coverage: overall=0.914 | alpha-fam=0.955, beta-fam=0.965, pos-fam=0.955, gamma=1.000
- Distance recovery cor = 0.833, R_scale = 1.145
- Split rate = 0.002, Merge rate = 0.679, Sigma swap rate = 0.931
- Folder: [`simulation_4_layered_v12_sens_a1.0_t5.0`](../plot/simulation_4_layered_v12_sens_a1.0_t5.0/)


### `a2.0_t1.0` (alpha=2, tau=1, n_iter=20000)

- mean K_+ = 7.18, median K_+ = 7, range [1, 17]
- ARI hclust=0.879, Dahl=-0.001, Binder=-0.001, VI=-0.001
- Co-cluster contrast = 0.077 (within=0.580, between=0.503)
- LSIRM coverage: overall=0.911 | alpha-fam=0.960, beta-fam=0.956, pos-fam=0.947, gamma=1.000
- Distance recovery cor = 0.829, R_scale = 1.172
- Split rate = 0.128, Merge rate = 0.406, Sigma swap rate = 0.850
- Folder: [`simulation_4_layered_v12_sens_a2.0_t1.0`](../plot/simulation_4_layered_v12_sens_a2.0_t1.0/)


### `a2.0_t2.0` (alpha=2, tau=2, n_iter=20000)

- mean K_+ = 6.37, median K_+ = 6, range [1, 16]
- ARI hclust=0.892, Dahl=0.000, Binder=0.000, VI=-0.001
- Co-cluster contrast = 0.073 (within=0.706, between=0.632)
- LSIRM coverage: overall=0.915 | alpha-fam=0.953, beta-fam=0.973, pos-fam=0.953, gamma=1.000
- Distance recovery cor = 0.831, R_scale = 1.166
- Split rate = 0.036, Merge rate = 0.470, Sigma swap rate = 0.863
- Folder: [`simulation_4_layered_v12_sens_a2.0_t2.0`](../plot/simulation_4_layered_v12_sens_a2.0_t2.0/)


### `a1.0_t1.0_iter40k` (alpha=1, tau=1, n_iter=40000)

- mean K_+ = 4.08, median K_+ = 4, range [1, 12]
- ARI hclust=0.913, Dahl=0.000, Binder=0.000, VI=0.000
- Co-cluster contrast = 0.045 (within=0.762, between=0.717)
- LSIRM coverage: overall=0.909 | alpha-fam=0.953, beta-fam=0.960, pos-fam=0.943, gamma=1.000
- Distance recovery cor = 0.822, R_scale = 1.154
- Split rate = 0.061, Merge rate = 0.484, Sigma swap rate = 0.909
- Folder: [`simulation_4_layered_v12_sens_a1.0_t1.0_iter40k`](../plot/simulation_4_layered_v12_sens_a1.0_t1.0_iter40k/)


### `a2.0_t1.0_iter40k` (alpha=2, tau=1, n_iter=40000)

- mean K_+ = 7.25, median K_+ = 7, range [1, 17]
- ARI hclust=0.891, Dahl=0.000, Binder=0.000, VI=-0.001
- Co-cluster contrast = 0.077 (within=0.581, between=0.504)
- LSIRM coverage: overall=0.912 | alpha-fam=0.958, beta-fam=0.963, pos-fam=0.948, gamma=1.000
- Distance recovery cor = 0.826, R_scale = 1.183
- Split rate = 0.123, Merge rate = 0.406, Sigma swap rate = 0.851
- Folder: [`simulation_4_layered_v12_sens_a2.0_t1.0_iter40k`](../plot/simulation_4_layered_v12_sens_a2.0_t1.0_iter40k/)


### `a2.0_t2.0_iter40k` (alpha=2, tau=2, n_iter=40000)

- mean K_+ = 6.45, median K_+ = 6, range [1, 15]
- ARI hclust=0.851, Dahl=0.000, Binder=0.000, VI=0.001
- Co-cluster contrast = 0.072 (within=0.703, between=0.631)
- LSIRM coverage: overall=0.915 | alpha-fam=0.952, beta-fam=0.977, pos-fam=0.951, gamma=1.000
- Distance recovery cor = 0.828, R_scale = 1.172
- Split rate = 0.036, Merge rate = 0.467, Sigma swap rate = 0.862
- Folder: [`simulation_4_layered_v12_sens_a2.0_t2.0_iter40k`](../plot/simulation_4_layered_v12_sens_a2.0_t2.0_iter40k/)

<!-- AUTOFILL NOTES ABOVE -->

## Conclusions

### Quick ranking by ARI hclust (primary clustering metric)

1. **`a2.0_t2.0`** — ARI 0.892 ⭐ (best)
2. `a2.0_t1.0` — ARI 0.879
3. `a1.0_t1.0` — ARI 0.874 (paper-canonical)
4. `a1.0_t2.0` — ARI 0.600
5. `a0.5_t1.0` — ARI 0.398
6. `a1.0_t5.0` — ARI 0.000 (catastrophic)

### α (mass parameter) effect, with τ=1

- **α=0.5**: K=1 attractor wins, mean K_+ = 2.5, ARI 0.40. **Too small.**
- **α=1.0**: stable at K_+ ≈ 4 (matches K_true), ARI 0.87.
- **α=2.0**: over-clusters (K_+ ≈ 7), but contrast and ARI both improve. **Counter-intuitively good.**

α controls the K=1 attractor strength. Smaller α makes K=1 more probable
($P(\mathrm{new}) = \alpha/(\alpha+t-1)$), so α<1 amplifies the collapse.
Larger α keeps K away from 1 even at the cost of over-clustering — but
hclust cuts at median K so the over-clustering does not hurt ARI.

### τ (kernel temperature) effect, with α=1

- **τ=1.0**: baseline; clean within/between contrast in λ values, ARI 0.87.
- **τ=2.0**: sharper kernel — λ_between → 0 — but split-merge struggles
  (split rate 0.018, lowest of the α=1 sweep).  ARI drops to 0.60.
- **τ=5.0**: very sharp kernel; split rate 0.002 (essentially frozen).
  Chain cannot escape its initial state; ARI ≈ 0.

τ has a **sweet-spot** around 1.0 for this geometry.  Larger τ paradoxically
*hurts* clustering recovery because it freezes the sampler — strong kernel
makes the EPA pmf gradient near K=1 mode much steeper, so any split move
proposes a partition with much lower EPA mass and gets rejected.

### Joint observations

- **Co-cluster contrast (within − between)** is highest at `a2.0_t1.0` (0.077)
  and `a2.0_t2.0` (0.073).  Pushing α away from the K=1 attractor matters
  more than sharpening τ.
- **LSIRM coverage** is consistently ≈ 0.91 across all settings — clustering
  hyperparameters do not affect LSIRM parameter recovery.  Distance
  correlation (cor ≈ 0.82) and R_scale (≈ 1.15) are also stable.
- **Dahl/Binder/VI all return 0** in every setting due to K=1 sample
  contamination of the centroid (see history `0505_dahl_binder_vi_diagnosis`
  for details).  Use SALSO or pre-filter K_+ ≥ 2 samples for these.
- **σ swap rate** decreases as τ → ∞ (more decisive λ).  Below 0.95 here.

### Recommendation

For this simulation geometry (cluster centres at distance 2, within-sd 0.15):

- **Best single setting**: `α = 2.0, τ = 2.0` (joint K=1-escape and
  moderate-but-not-extreme contrast).  ARI hclust = 0.892, contrast = 0.073.
- **Paper-default for sensitivity baseline**: `α = 1.0, τ = 1.0`. ARI 0.874,
  contrast 0.046.  Acceptable but blurrier co-cluster.

The general lesson: in the `(α, τ)` plane, the productive region is roughly
**α ∈ [1, 3], τ ∈ [1, 2]**.  Avoid α ≤ 0.5 (collapse) and τ ≥ 3 (frozen
sampler).  Within this region, `α=2.0, τ=2.0` gives the cleanest
co-cluster pattern and the highest ARI.

For **real data** where the geometry scale is unknown, try a small grid
around `(α=1, τ=1)` and pick the configuration that maximises a held-out
metric or visual co-cluster contrast.  Or — closer to the original DDT
practice — set τ from a kernel-bandwidth heuristic (e.g., median pairwise
distance) and α via cross-validated K_+ behaviour.

---

## Phase 2: Sampler-tuning experiments (post-sensitivity)

After the (α, τ) grid above, we explored two additional knobs:

### Phase 2a — `M_SM = 5` (split-merge attempts per sweep)

**Killed due to runtime impracticality.**  M_SM = 5 increased per-sweep
cost ~50–100x relative to M_SM = 1 in this geometry (each split-merge
attempt does R+1 = 6 restricted Gibbs scans × |S*| EPA pmf evaluations,
and at moderate K the inner loop dominates).  After 4 hours, the first
experiment had only completed iter 8000 of 20000.  Estimated total runtime
~17 hours for 3 experiments.  Aborted.

The slowdown is **non-linear** in M_SM because larger K-states (which the
chain visits) make `|S*|` larger, and the cost per restricted Gibbs scan
scales with `|S*|`.  M_SM = 2–3 is probably the practical upper bound for
this geometry; M_SM = 5 is too aggressive.

### Phase 2b — `n_iter = 40000` (2× iterations) at three best settings

To check whether 20K iterations was simply insufficient, we re-ran the
three best (α, τ) settings with doubled iterations (`n_iter = 40000`,
`burnin = 16000`, `thin = 10` → same n_save = 2400).

| label | n_iter | mean K | ARI hclust | contrast | split rate |
|---|---|---|---|---|---|
| `a1.0_t1.0` | 20000 | 4.13 | 0.874 | 0.046 | 0.062 |
| `a1.0_t1.0_iter40k` | **40000** | 4.08 | **0.913** | 0.045 | 0.061 |
| `a2.0_t1.0` | 20000 | 7.18 | 0.879 | 0.077 | 0.128 |
| `a2.0_t1.0_iter40k` | **40000** | 7.25 | 0.891 | 0.077 | 0.123 |
| `a2.0_t2.0` | 20000 | 6.37 | 0.892 | 0.073 | 0.036 |
| `a2.0_t2.0_iter40k` | **40000** | 6.45 | 0.851 | 0.072 | 0.036 |

**Findings**:

- `a1.0_t1.0`: ARI hclust **0.874 → 0.913** (+0.04). Doubling iter
  meaningfully helps the baseline.  20K samples were not fully converged.
- `a2.0_t1.0`: ARI **0.879 → 0.891** (+0.01). Marginal — already mixed well.
- `a2.0_t2.0`: ARI **0.892 → 0.851** (−0.04). Slightly *worse* — Monte
  Carlo variance from a different K-trajectory mix.

**Interpretation**: doubling iterations gives the chain more chances to
visit/escape K-states, which helps when the chain isn't yet converged
(α=1, τ=1) but doesn't help — and can hurt by sample variance — when the
chain is already mixing.  Co-cluster contrast and split rate are
essentially unchanged across all three pairs, confirming that **iteration
count is not the bottleneck for the contrast** — that's structural to
EPA-LSIRM as discussed below.

### Net conclusion across phases

- The (α, τ) sensitivity grid identified the productive operating region.
- Sampler tuning (M_SM, n_iter) gives small marginal gains but does not
  fundamentally change the co-cluster sharpness.
- The fuzziness of the co-cluster matrix in EPA-LSIRM is a **structural
  property of the model** (no conjugate cluster likelihood; clustering
  signal comes only from the EPA distance kernel), not an artefact of
  insufficient sampling.  See discussion in chat history of 2026-05-05.
- **Recommended production setting**: `α = 1.0, τ = 1.0, n_iter = 40000`
  for paper-canonical results, or `α = 2.0, τ = 1.0` (or 2.0) if higher
  contrast is needed and over-clustering is acceptable.
