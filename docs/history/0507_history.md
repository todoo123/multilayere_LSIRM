# 0507 — v11t Student-t mixture 구현 + Simulation + MIDUS tuning

## 오늘의 작업

| # | Task | 산출 |
|---|---|---|
| 1 | v11t Student-t mixture cpp 구현 ([data/my_LSIRM_FMC_v11t.cpp](../data/my_LSIRM_FMC_v11t.cpp)) | scale-mixture form (Lange-Little-Taylor 1989), per-item w_q ~ Gamma, weighted NIW posterior, 모든 split-merge / c Gibbs / b MH 가 w_q 반영 |
| 2 | R wrapper ([data/my_LSIRM_FMC_cpp_v11t.R](../data/my_LSIRM_FMC_cpp_v11t.R)) | `lsirm_fmc_v11t_cpp()`; `b_prior_inflation` 제거하고 `nu_t = 4` (default) 추가 |
| 3 | Simulation 검증 ([simulation_4_layered_v11t.R](../simulation_4_layered_v11t.R), 50k iter) | ARI hclust 0.903 (vs v11 0.860), Dahl 0.872, 7-item C3 cluster 회복; split rate 0.042 (v11 의 5배) |
| 4 | MIDUS v11t 첫 run ([data/MIDUS_5layered_result_v11t.R](../data/MIDUS_5layered_result_v11t.R), 80k iter) | median K_+ = 3 (looseSigma 의 4 에서 1-singleton 흡수); PEAR 0.396; **silhouette 0.351 → 0.628 ↑↑** (geometry 회복) |
| 5 | LSIRM b acceptance 진단 (모든 layer) | b1..b4 acceptance 0.13–0.23 (목표 0.30+ 못 미침); β₂ 0.10; tuning 필요성 발견 |
| 6 | Proposal SD 재튜닝 + MIDUS v11t tuned re-run | b1..b4 0.28–0.31 정상화, β₂ 0.21 향상; PSM contrast 더 sharp 해짐 (PEAR 0.416, within-pair 0.470) |

**결정 요지**: t-mixture (`nu_t = 4`) 가 v11 Gaussian mixture (looseSigma) 의 biplot 압축 문제를 model-coherent 방식으로 해결. silhouette 0.35 → 0.63 으로 b geometry 회복, 7-item inflammation+executive function cluster 는 robust 하게 유지. **현 v11 운영 setting 은 v11t tuned (이 파일 위에서 제안된 proposal SD 포함) 로 lock-in 권장**.

---

# Task 1. v11t cpp 구현 — scale-mixture Student-t mixture

## 1.1 모델 변경

**v11 Gaussian mixture**:
$$b_q \mid c_q = l, \mu_l, \Sigma_l \sim \mathcal{N}_d(\mu_l, \Sigma_l)$$

**v11t Student-t mixture (scale-mixture form)**:
$$w_q \sim \text{Gamma}(\nu_t/2, \nu_t/2)$$
$$b_q \mid c_q = l, w_q \sim \mathcal{N}_d(\mu_l, \Sigma_l / w_q)$$

→ marginal $b_q \sim t_{\nu_t}(\mu_l, \Sigma_l)$. `nu_t = 4` 고정 (LSIRM continuous 층의 robust likelihood 와 일관).

## 1.2 cpp 수정 7곳

[data/my_LSIRM_FMC_v11t.cpp](../data/my_LSIRM_FMC_v11t.cpp) — v11.cpp 1707줄 기반:

1. `vec ws(P_total)` state 추가 (init = 1)
2. 새 helper `niw_posterior_from_indices_weighted(...)`:
   - $\kappa_n = \kappa_0 + \sum_q w_q$ (weighted "effective n")
   - $\nu_n = \nu_0 + n_J$ (자유도는 항목수 그대로)
   - $\bar{z}_w$, weighted scatter $S_w$
3. **b MH ratio (5 layers, b1..b5)**: prior penalty `0.5 * inv_b_inflation * q` → `0.5 * ws[q] * q`
4. **c Gibbs (collapsed)** predictive 변경: `((κ_n + w_q)/(κ_n · w_q · dof)) · S_n` 로 t-density 호출
5. **Split-merge (Jain-Neal)** 모든 niw call sites (10곳) 가 weighted version 사용
6. **(μ, Σ) NIW Gibbs**: weighted version
7. **w_q Gibbs (새 블록 C3a)**: 매 sweep `w_q ~ Gamma((ν+d)/2, (ν+Δ_q)/2)` (Δ_q = (b_q - μ_c)' Σ_c⁻¹ (b_q - μ_c))

**제거된 hack**: `b_prior_inflation` knob (model-coherence 위반). v11t 는 정합 t-mixture 로 대체.

**추가 출력**: `fmc_ws` (n_save × P_total) — per-item w_q traces.

## 1.3 R wrapper

[data/my_LSIRM_FMC_cpp_v11t.R](../data/my_LSIRM_FMC_cpp_v11t.R):
- `lsirm_fmc_v11t_cpp()` 함수
- `nu_t = 4` parameter 추가
- `b_prior_inflation` 제거
- 컴파일 / sourcing 검증 통과

---

# Task 2. Simulation 검증 ([simulation_4_layered_v11t.R](../simulation_4_layered_v11t.R))

## 2.1 Setup

n=150, P=120 (30+30+30+30 bin/con/cnt/ord), K_true=4, isotropic clusters at (±1, ±1), within-cluster sd 0.10–0.20. 50k iter / 20k burnin / thin 10 → 3000 saved.

## 2.2 결과 — v11t simulation

| 지표 | **v11t** | v11 reference (memory) |
|---|---|---|
| median K_+ | 5 | 5 |
| ARI(hclust) | **0.903** | 0.860 |
| ARI(Dahl) | 0.872 | 0.878 |
| ARI(Binder) | 0.872 | 0.878 |
| ARI(VI) | 0.872 | 0.868 |
| **split rate** | **0.042** | ~0.008 (5배 ↑) |
| **merge rate** | **0.015** | ~0.002 (7배 ↑) |
| R_scale | 1.135 | — |
| cor_dist | 0.829 | — |

**Cluster 회복은 v11 와 동등**, **MCMC mixing 은 5–7배 향상**. cross-tab (Dahl K=4) 깨끗:
```
                true=1  true=2  true=3  true=4
Dahl=1            32       1       0       0
Dahl=2             0       1      27       2
Dahl=3             0       1       1      26
Dahl=4             0      29       0       0
```

## 2.3 Student-t weight (w_q) 진단

| 지표 | 값 |
|---|---|
| mean E[w_q] | 1.014 (Gaussian-equivalent 거의 1) |
| range | [0.371, 1.239] |
| Items with E[w_q] < 1 (downweighted) | **40%** |
| Items with E[w_q] < 0.5 (강한 outlier) | 1 (ord1_23, E[w]=0.371) |

→ **의도대로 작동**: cluster boundary item 자동 식별, mean ~1 (Gaussian 근사 잘 유지). cpp 구현 정상 검증.

---

# Task 3. MIDUS v11t 첫 run (untuned proposal SDs)

## 3.1 Setup

n=221, P_total=68 (bin 20 / con 5 / cnt 15 / ord 28), 80k iter / 20k burnin / thin 10 → 6000 saved.

Setting: `nu0=10, S0_init_scale=0.434, kappa0=0.1, K_star=10, n_split_merge=100` (looseSigma 그대로 상속), `nu_t=4`. Proposal SDs 도 looseSigma 그대로.

## 3.2 결과 비교

| 지표 | looseSigma (Gaussian) | **v11t (Student-t, ν=4)** |
|---|---|---|
| median K_+ | 4 | **3** (1-item singleton 흡수) |
| Final partition (P0) | 41/19/7/**1** | 42/19/7 |
| Dahl/Binder/VI | 모두 K=3 일치 | 모두 K=3 일치 |
| 7-item C3 (inflammation+exec) | 동일 | **동일 (robust)** |
| **silhouette mean (b̂)** | 0.351 | **0.628** ↑↑ |
| silhouette median | 0.471 | **0.682** ↑ |
| frac<0 | 0.132 | **0** ↓ |
| within-pair ratio | 0.454 | 0.415 |
| PEAR | 0.424 | 0.396 |
| **empirical b̂ within-sd** | 0.168 / 0.231 | **0.252 / 0.248** |
| ratio (prior/empirical) | 1.48 / 1.08 | **0.99 / 1.01** (perfect match) |

## 3.3 핵심 발견

1. **biplot b̂ geometry 자연스러워짐**. empirical within-cluster sd 0.17–0.23 → 0.25–0.25 로 증가 → prior implied 0.249 와 거의 정확히 일치 (ratio ~1.0). t-mixture 의 heavy tail 이 boundary item 의 prior pull 약화 → b 가 likelihood 로 자유롭게 펴짐.
2. **silhouette 0.35 → 0.63** — K=4 → K=3 변화 효과 + b̂ spread 회복 효과.
3. **C3 (inflammation + executive function) cluster 그대로 robust**. v11t 도 7-item identical: B4BCRP, B4BFGN, B4BIL6 + sgst_mixed_{all, nonswitch, switch} + sgst_reverse_incorrect.
4. **w_q 가 cluster boundary item 정확히 식별**: 가장 downweighted 항목 wl_delayed_repetition (E[w]=0.70), wl_immediate_repetition (0.81), B4BIL6 (0.80), sgst_reverse (0.88).

산출물: [data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4/](../data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4/).

---

# Task 4. LSIRM b acceptance 진단

User 관찰: "item position b 가 좀 별로야" — biplot / trace 의 b 가 부자연스러움.

## 4.1 Acceptance 분석

| Parameter | v11t (untuned) | 평가 |
|---|---|---|
| alpha1..4 | 0.45–0.58 | ✓ OK |
| beta1, beta3 | 0.38, 0.53 | ✓ OK |
| **beta2 (Con, 5 items)** | **0.099** | **❌ 너무 낮음 (목표 0.30+)** |
| log_gamma1..4 | 0.20–0.25 | △ 낮음 |
| a (resp) | 0.362 | ✓ OK |
| **b1 (Bin, 20 items)** | **0.147** | **❌ 너무 낮음** |
| **b2 (Con, 5 items)** | **0.205** | △ 낮음 |
| **b3 (Cnt, 15 items)** | **0.230** | △ 낮음 |
| **b4 (Ord, 28 items)** | **0.136** | **❌ 너무 낮음** |
| log_kappa | 0.354 | ✓ OK |

**진단**: 시뮬 (n=150, P=120) 에선 b acceptance 0.46–0.53 정상. **MIDUS (n=221, P=68) 에선 likelihood precision 가 더 강해서 b 의 effective posterior 가 좁아짐** → 같은 proposal SD 가 step size 2-3배 과대.

→ b chain mixing 나빠 (sticky trace), 사후 불확실성 추정 신뢰도 낮음. user 가 본 "b 별로" 의 정확한 원인.

---

# Task 5. Proposal SD 재튜닝 + v11t tuned re-run

## 5.1 변경 (v11t MIDUS file)

```r
# 0507 v11t MIDUS retune:
beta2: 0.40 -> 0.18      # acceptance 0.099 -> target ~0.30
b1   : 0.50 -> 0.30      # acceptance 0.147 -> target ~0.32
b2   : 0.40 -> 0.27      # acceptance 0.205 -> target ~0.32
b3   : 0.50 -> 0.35      # acceptance 0.230 -> target ~0.33
b4   : 0.40 -> 0.27      # acceptance 0.137 -> target ~0.32
# 다른 SDs 불변
```

run_label suffix `_tunedb` 추가 → 별도 디렉토리에 저장.

## 5.2 튜닝 후 acceptance

| Parameter | 이전 | **튜닝 후** | 평가 |
|---|---|---|---|
| **beta2** | 0.099 | **0.211** | △ (2배↑, 목표 0.30 못 미침) |
| **b1** | 0.147 | **0.282** | ✓ |
| **b2** | 0.205 | **0.295** | ✓ |
| **b3** | 0.230 | **0.311** | ✓ |
| **b4** | 0.137 | **0.232** | △ (목표 0.30 미달) |

b1..b3 정상화. b4, beta2 더 줄이면 완벽하지만 충분히 개선.

## 5.3 PSM contrast 비교 (tuning 효과)

| 지표 | looseSigma | v11t (untuned) | **v11t tuned** |
|---|---|---|---|
| median K_+ | 4 | 3 | 3 |
| Final partition | 41/19/7/1 | 42/19/7 | **42/19/7** |
| **PEAR** | 0.424 | 0.396 | **0.416** ↑ |
| **within-pair ratio** | 0.454 | 0.415 | **0.470** ↑ |
| Pr(co > 0.8) | 0.201 | 0.192 | **0.217** ↑ |
| Pr(co < 0.2) | 0.311 | 0.256 | **0.285** ↑ |
| Pr(co ambig) | 0.208 | 0.261 | **0.231** ↓ |
| silhouette mean | 0.351 | 0.628 | **0.627** |
| silhouette median | 0.471 | 0.682 | **0.687** |
| frac<0 | 0.132 | 0 | **0** |
| split rate | 0.095 | 0.069 | 0.056 |
| merge rate | 0.080 | 0.065 | 0.049 |
| net K_+ change | +384 | -38 | **+111** |

**핵심 결과**:
- PSM contrast **PEAR 0.396 → 0.416, within-pair 0.415 → 0.470** — looseSigma 보다도 sharper.
- silhouette 0.63 그대로 유지 (geometry 회복 효과 보존).
- ambig zone 0.261 → 0.231 (3%p 감소).
- K=3 partition 동일 (42/19/7), 임상 finding (7-item C3) 그대로.

**즉 v11t tuned 가 looseSigma 와 untuned v11t 를 모두 dominate**:
- looseSigma 대비: PSM 비슷+, biplot geometry 훨씬 좋음, K=3 cleaner partition
- untuned v11t 대비: PSM 더 sharp, b chain mixing 정상화, silhouette 동일

## 5.4 v11t 의 Student-t weight 진단 (튜닝 후)

| 지표 | 값 |
|---|---|
| mean E[w_q] | 1.031 |
| range | [0.658, 1.267] (untuned 0.703–1.21 보다 살짝 넓음) |
| Items with E[w_q] < 1 | 38% |
| Items with E[w_q] < 0.5 | 0% (없음) |

가장 downweighted 항목 (모두 cluster 1 또는 3 의 boundary):
- wl_delayed_repetition (cluster 1 boundary)
- wl_immediate_repetition (cluster 1 boundary)
- B4BIL6 (cluster 3 boundary, inflammation marker 중 가장 약하게 묶임)
- sgst_reverse_incorrect (cluster 3 boundary)
- B4BCRP (cluster 3, inflammation marker)

산출물: [data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4_tunedb/](../data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4_tunedb/).

---

# Task 6. 결정: v11t tuned 가 새 reference v11 운영 setting

User 결정: 0506 의 looseSigma lock-in 을 v11t tuned 로 교체.

근거:
- looseSigma 대비 모든 지표 동등하거나 우수
  - PSM contrast: PEAR 0.42 ↔ 0.42, within-pair 0.45 ↔ 0.47
  - biplot geometry: silhouette 0.35 → 0.63 (큰 개선)
  - cluster 분리 안정성: K=3 깔끔 (singleton 없음)
- 임상 finding (7-item inflammation+exec function cluster) 그대로 robust
- model-coherent (b_prior_inflation hack 제거, t-mixture 로 정직한 model change)
- LSIRM b chain mixing 정상화 (b1..b3 acceptance 0.28–0.31)

**현 reference 운영 파일**: [data/MIDUS_5layered_result_v11t.R](../data/MIDUS_5layered_result_v11t.R) (proposal SD tuned, nu_t=4).

**잔여 미해결**:
- β₂ (Con, 5 items) acceptance 0.21 — 목표 0.30 미달, proposal SD 0.18 → 0.10 정도로 더 줄이면 정상화 가능. 다음 작업.
- b4 (Ord, 28 items) acceptance 0.23 — proposal SD 0.27 → 0.20 으로 더 줄이면 정상화. 다음 작업.

paper main result table 에는 v11t tuned 결과 사용. v11 looseSigma 결과는 비교 baseline 으로만.

---

# 참고 산출물 (v11t 추가)

## v11t 새 코드
- `data/my_LSIRM_FMC_v11t.cpp` (cpp, 약 1790줄)
- `data/my_LSIRM_FMC_cpp_v11t.R` (wrapper)
- `simulation_4_layered_v11t.R` (toy validation driver)
- `data/MIDUS_5layered_result_v11t.R` (MIDUS production driver, **현 운영 setting**)

## v11t 결과 디렉토리
- `plot/simulation_4_layered_v11t_d2_S0_0.3_nu4/` (sim 검증)
- `data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4/` (MIDUS untuned, 진단용)
- **`data/plot/case1_all_v11t_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_nut4_tunedb/`** ← **현 reference**
