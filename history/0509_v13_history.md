# 0509 v13 history — Hierarchical latent-position mixture (LSIRM + r_q + Σ_b)

## 1. Model design

v13 introduces an auxiliary cluster-aligned position `r_q` between the
LSIRM item position `z_q = b_j` and the cluster mean `μ_{c_q}`:

```
z_q | r_q, Σ_b           ~ N_d(r_q, Σ_b)
r_q | c_q = l, μ_l, Σ_l  ~ N_d(μ_l, Σ_l)
```

Marginal: `z_q | c_q = l ~ N_d(μ_l, Σ_l + Σ_b)`.

**Motivation**: in v11 the cluster covariance Σ_l plays two roles
simultaneously — defining cluster shape AND setting the prior precision
pulling b_j toward the cluster centre in the b MH update.  Tight
clusters (small Σ_l) compress within-cluster pairwise z-distances,
distorting the LSIRM geometry.  v13 splits the role into a cluster-shape
covariance Σ_l (on r_q) and a global anchor covariance Σ_b (between
z_q and r_q), so cluster shape heterogeneity is preserved while anchor
strength is pooled across all P items via Σ_b.

**Sampler changes vs v11**:
- New per-item conjugate Gibbs draw: `r_q | z_q, c_q, μ, Σ, Σ_b ~ N(...)`
- New conjugate update: `Σ_b | z, r ~ IW` (full) or `σ_b² | z, r ~ IG`
  (isotropic).  Default isotropic.
- b MH ratio prior term changes from `N(b; μ_{c_q}, Σ_{c_q})` to
  `N(b; r_q, Σ_b)`.
- All FMC clustering (collapsed c_q allocation, split-merge, NIW
  posterior on (μ_l, Σ_l)) operates on r, not z.
- `b_prior_inflation` knob from v11 is REMOVED (decoupling now handled
  by Σ_b's data-driven scale).

Reference: `modeling_paper/model_v13.tex` (Sections 1–14).

## 2. Implementation files

| File | Purpose |
|---|---|
| `data/my_LSIRM_FMC_v13.cpp` | C++ MCMC kernel (1820 lines).  LSIRM block byte-identical to v11/v10; FMC block + b MH modified per v13 spec. |
| `data/my_LSIRM_FMC_cpp_v13.R` | R wrapper.  Adds Σ_b prior controls (`a_b`/`b_b` for isotropic, `nu_b`/`S_b` for full IW), `sigma_b_fixed` flag, Procrustes co-rotation of r and Σ_b. |
| `simulation_4_layered_v13.R` | 4-layer (bin/con/cnt/ord) simulation runner.  Currently set to v11-easy data-generating regime (centers ±1, equal sizes, varied σ_meta). |
| `exp_v13_sigma_b_sweep.R` | Sweep over fixed σ_b² ∈ {0.05, 0.10, 0.15, 0.20, 0.30}. |
| `data/MIDUS_5layered_result_v13.R` | Real-data application script.  Mirror of v11t MIDUS pipeline with v13 model swap and v13-specific anchor diagnostics. |

## 3. Simulation results timeline (2026-05-09)

All simulations: P_total=120 (30/30/30/30), n=150, K_true=4, K*=10,
e0=0.1, n_split_merge=100, MCMC 20k/8k/5 (n_save=2400).

### 3.1 Asymmetric A1+B2 setting + free σ_b² (initial test)

Data: centers (±0.5)², sizes 12/9/6/3 per layer (asymmetric),
σ_meta varied 0.10-0.20.  Hyper: kappa0=1, nu0=12, S0=0.434·I,
**σ_b² prior IG(3, 0.05) → E[σ_b²]=0.025**.

**Outcome — FAIL**:
- σ_b² posterior mean = **0.935** (37x prior runaway)
- mean K_+ = 4.10 (median 4, sd 1.33), wandering 1..9
- ARI: hclust 0.067, Dahl/Binder/VI = 0.000
- split rate = 0.309, merge rate = 0.391 (extreme — partition mixing too aggressive)
- All point-estimators collapsed to K=1

**Diagnosis**: prior pseudo-count (a_b=3 → 6) overwhelmed by likelihood
pseudo-count (P·d = 240).  σ_b² absorbed LSIRM-scale noise (~1).  In
positive-feedback equilibrium: large σ_b weakens b anchor → b drifts
to LSIRM-natural positions → |z-r| grows → σ_b² IG posterior grows
further.  Resulting `r ≈ μ_{c_q}` collapse made NIW marginal on r
extremely sharp, accepting both splits and merges promiscuously.

### 3.2 Option B — strong prior to dominate likelihood

Same data as 3.1.  **σ_b² prior IG(200, 10) → E[σ_b²]=0.0503, sd=0.004**
(prior pseudo-count = 400, dominates likelihood 240).

**Outcome — partial improvement**:
- σ_b² posterior mean = **0.0505** (locked at prior, polished prior pinning)
- mean K_+ = 4.94 (mode 5, sd 1.17), still wandering
- ARI: hclust 0.519, Dahl/Binder 0.332, VI 0.155
- split rate = 0.229, merge rate = 0.116 (5x lower than 3.1, still high vs v11)

**Diagnosis**: σ_b² no longer runaway, but partition still unstable.
True 1 ↔ True 4 (vertical neighbours) frequently merged.  Still 5x
worse split/merge rates than v11 lock-in (0.05/0.005).

### 3.3 σ_b² fixed sweep — `exp_v13_sigma_b_sweep.R`

Same data as 3.1/3.2.  σ_b² FIXED at {0.05, 0.10, 0.15, 0.20, 0.30}
(no Gibbs update via new `sigma_b_fixed=TRUE` flag).

| σ_b² | ARI(hclust) | ARI(Dahl) | mean K_+ | split | merge | dist cor |
|---|---|---|---|---|---|---|
| **0.05** | **0.519** | 0.363 | 4.91 | 0.227 | 0.116 | 0.674 |
| 0.10 | 0.477 | **0.374** | 4.86 | 0.225 | 0.121 | 0.672 |
| 0.15 | 0.468 | 0.242 | 4.84 | 0.236 | 0.130 | 0.680 |
| 0.20 | 0.403 | 0.263 | 4.73 | 0.284 | 0.181 | 0.667 |
| 0.30 | 0.416 | 0.350 | 4.59 | 0.304 | 0.252 | 0.671 |

**Outcome — SWEET SPOT IDENTIFIED**: σ_b² ≈ 0.05–0.10 best.
Larger σ_b² ↑ → split/merge rate ↑, ARI ↓.  None reach v11 lock-in
(ARI 0.86–0.88).

**Diagnosis**: realistic-overlap data (center/sd ≈ 1.4) is itself hard
for any model.  Need to test v13 on a regime where v11 is known to
work cleanly to verify v13 is functional.

Outputs saved to `plot/exp_v13_sigma_b_sweep/`.

### 3.4 v11-easy setting + free σ_b² (validation)

Data: **centers (±1)² (clean separation), equal sizes 8/8/7/7, varied
σ_meta 0.10-0.20**.  Hyper: same as 3.1 except S0_scale=0.3 and
σ_b² prior IG(3, 0.10) → E=0.05.

**Outcome — SUCCESS, v13 validated**:

| Metric | v11 lock-in (memory) | **v13 easy** |
|---|---|---|
| ARI(hclust) | 0.86–0.88 | **0.925** ⭐ |
| ARI(Dahl) | — | **0.891** |
| ARI(Binder) | — | **0.891** |
| ARI(VI) | — | **0.871** |
| mean K_+ | ~4 | 4.70 (sd 0.82) |
| Distance recovery cor | 0.9+ | 0.836 |
| split rate | 0.005–0.05 | **0.045** |
| merge rate | 0.001–0.01 | **0.015** |
| σ_b² posterior | — | **0.034** (95% CI [0.013, 0.084]) |

v13 slightly **outperforms** v11 on the same regime.  σ_b² ends below
prior mean (0.05) — data informs it downward when cluster separation
is clean.  Split/merge rates collapse to v11-lock-in healthy regime
once data is informative enough to discriminate.

Outputs saved to `plot/simulation_4_layered_v13_d2_S0_0.3_easy/`.

## 4. Conclusion

**v13 works** in regimes where data has clean cluster structure
(center/sd ≥ 5), validating the hierarchy implementation.  In
realistic-overlap regimes (center/sd ≈ 1.4) all candidate models
(v11/v11t/v13) struggle; v13 is no better but no worse.

**Recommended operating point** for real applications:
- `kappa0 = 1.0`, `nu0 = d+10`, `S0 = 0.434·I`
- `sigma_b_isotropic = TRUE`, `a_b = 3`, `b_b = 0.10` (E[σ_b²]=0.05)
- `sigma_b_fixed = FALSE` (free Gibbs); switch to TRUE only if posterior
  shows runaway (mean σ_b² > 0.3 or split/merge rate > 0.20).
- `n_split_merge = 100`

## 5. Next steps

- **Real MIDUS application** via `data/MIDUS_5layered_result_v13.R`
  (created today, v13 model + v11t pipeline structure).  Compare ARI /
  silhouette / PEAR / σ_b² recovery against v11t locked-in MIDUS results.
- If MIDUS shows runaway σ_b², fall back to fixed σ_b² ≈ 0.05–0.10.
- Future modelling extensions noted in `model_v13.tex` Section 14:
  Wishart hyperprior on S_b (full IW form only), tighter joint Procrustes
  alignment, etc.

## 6. Logs / artefacts on disk

| Path | Contents |
|---|---|
| `sim_v13_optionB_20k.log` | 3.2 (Option B run, 20k iter) |
| `sim_v13_easy_20k.log` | 3.4 (easy v11-friendly run, 20k iter) |
| `exp_v13_sigma_b_sweep.log` | 3.3 (5-value σ_b² sweep) |
| `plot/simulation_4_layered_v13_d2_S0_0.434_imbal_close/` | 3.1 + 3.2 outputs (overwritten) |
| `plot/simulation_4_layered_v13_d2_S0_0.3_easy/` | 3.4 outputs (success) |
| `plot/exp_v13_sigma_b_sweep/` | 3.3 outputs incl. `samps_compact.rds` per σ_b² |
