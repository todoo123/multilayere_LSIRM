# 0507 — v12 EPA PSM uniformity 진단: σ vs α prior 식별

## 오늘의 작업

| # | Task | 산출 |
|---|---|---|
| 1 | v12 MIDUS 의 "z 잘 분리되는데 co-cluster matrix uniform" 보고에 대한 코드 검토 | EPA pmf의 σ-sequential 구조를 [my_LSIRM_FMC_v12.cpp:126-170](../data/my_LSIRM_FMC_v12.cpp#L126-L170) 에서 재확인; sigma swap MH / single-item Gibbs / split-merge 모두 σ-conditional EPA pmf 의존 |
| 2 | 기존 v12_2 simulation log 에서 진단 지표 추출 | `sigma_swap_rate=0.917`, `split=0.057`, `merge=0.498` (α=1, τ=1 fixed long, 120k iter); σ는 stuck 이 아니라 *너무 자유로움* |
| 3 | v12_2 sim 의 PSM 직접 측정 ([epa_co_cluster.csv](../plot/simulation_4_layered_v12_2_sens_a1.0_t1.0_fix_long/epa_co_cluster.csv)) | within true cluster PSM mean=0.743, between=0.745, **ratio=1.00** — co-cluster matrix가 정확히 uniform 으로 확인 |
| 4 | σ-freeze 가설 실험 ([exp_sigma_freeze.R](../exp_sigma_freeze.R), 20k iter × 3 regime) | X1 cluster-aligned freeze: ARI(hclust)=−0.001 (악화); X2 random freeze: 0.978; X3 slow update: 0.932; PSM ratio 1.06–1.09 로만 미미 개선; **σ-freeze 는 해결책 아님** |
| 5 | α prior penalty 진단 (수치 분석) | α=1, δ=0 에서 K=4 partition prior / K=1 prior ≈ 1/162,000; cluster-aligned σ 에서 cluster-2 첫 item 의 new-cluster 확률 1/33 → over-merging 의 정확한 메커니즘 확정 |
| 6 | α-sweep 실험 ([exp_alpha_sweep.R](../exp_alpha_sweep.R), α ∈ {1, 5, 10}, free σ, τ=1) | α 증대 → PSM contrast 회복 (1.08→1.66→2.58) ✓; **그러나 mean K_+ 가 4→15→26 으로 폭발** ✗; α=1 의 K-recovery 가 가장 정확 — α 단독으로는 sweet spot 없음 |
| 7 | Cluster size 분포 직접 측정 ([exp_cluster_sizes.R](../exp_cluster_sizes.R), α=1, 2400 saved iters) | **K_+ trace 가 거짓말함**. mean K_+=3.99 정확하지만 median largest-cluster share=0.89; 28% iters 가 사실상 K=1 (largest > 0.95); **EPA stationary 가 K=1 mode 에 정착**. PSM uniformity 의 진짜 메커니즘 확정 |

**결정 요지 (Task 7 추가 후 갱신)**: PSM uniformity 의 진짜 메커니즘은 **EPA stationary 가 K=1 mode 에 정착하는 것**이다. K_+ trace mean=3.99 는 misleading — 실제 cluster sizes 는 median {largest=89%, 2nd=7%} 로 압도적 unbalanced; 28% iters 가 사실상 K=1. 두 메커니즘 결합 결과: (a) new-cluster prior penalty 1/(α+t) (K=4/K=1 ratio ≈ 1/162,000), (b) existing-cluster sum_S/sum_all 의 rich-get-richer dynamics. b 는 LSIRM likelihood 가 dominant 라 4-cluster geometry 유지하지만, c 는 EPA prior 가 dominant 라 K=1 으로 collapse. b 와 c 의 *분리* 가 핵심. α 를 건드려도 prior structure 자체가 안 바뀜 → **PSM 기반 분석이 필요한 모든 use case 는 v11t 회귀**, v12 EPA 는 hclust(b̂) 단독 estimator 로만 사용.

---

# Task 1. 사용자 보고 검토 — v12 MIDUS 의 PSM uniformity

## 1.1 보고 내용

τ=1 고정, z의 사후 추정이 ground truth 와 잘 일치 (cluster 별로 시각적 분리), LSIRM 모든 parameter 정상 추정, **그런데도 co-cluster matrix 가 거의 uniform**.

이 조합은 매우 specific 한 진단 가능: "모델 문제도 아니고, z 학습의 문제도 아니고, EPA pipeline 이 분리된 z 를 못 활용하고 있다." 사용자가 세 가지 가설 제시:
- 가설 1: σ가 cluster 를 가로지르며 stuck → q 의 prefix 가 multi-cluster mixture → attraction 신호 흐려짐
- 가설 2: single-item Gibbs 의 EPA conditional 평가에서 q 의 σ-position 에 따라 신호 강도 다름 → σ 후반부 item 들은 random reassign 됨
- 가설 3: split-merge 가 거의 reject 되어 single-item Gibbs 만 작동 → cluster 형성 느림

## 1.2 코드 매칭

[my_LSIRM_FMC_v12.cpp](../data/my_LSIRM_FMC_v12.cpp) 의 EPA 블록 모두 검토:

- **`log_epa_pmf` (line 126-170)**: sequential allocation; q 가 σ 의 t 번째에 등장할 때 prefix `σ[1..t-1]` 와의 attraction 만 사용. cached `lambda_mat` 으로 lookup-only.
- **single-item Gibbs (line 1289-1331)**: 각 candidate cluster label 에 대해 *전체 EPA pmf* 평가. q 의 σ-position 은 fixed.
- **σ random-swap MH (line 1438-1458)**: M_perm = P_total = 120 swap 시도/sweep.
- **Jain-Neal nonconjugate split-merge (line 1336-1436)**: M_SM=1, R=5 launch scans, EPA-decoupled (LSIRM cancels).

가설 1, 2 는 코드와 정확히 매칭됨 — sequential 구조 때문에 q 의 σ-position 이 conditional 강도에 직접 영향. 가설 3 는 코드상 split-merge 가 정상 구현됨.

## 1.3 결정

코드 실행 (sim 으로) 으로 가설 검증.

---

# Task 2. v12_2 sim 의 진단 지표 발굴

## 2.1 기존 log 검색

[v12_2_run.log](../v12_2_run.log) (free α, free τ, 100k iter, exp A):
```
sigma swaps: 2316142 / 2400000  (rate 0.965)
split: 987 / 16685  (rate 0.059)
merge: 1767 / 3315  (rate 0.533)
median K_+ : 2   (true K = 4)
ARI(hclust): 0.342
ARI(Dahl/Binder/VI): 0.000
```

[v12_2_t1_long.log](../v12_2_t1_long.log) (α=1, τ=1 *fixed*, 120k iter):
```
sigma swaps: 11003208 / 12000000  (rate 0.917)
split: 4200 / 73079  (rate 0.057)
merge: 13419 / 26921  (rate 0.498)
median K_+ : 4   (true K = 4)
ARI(hclust): 0.978
ARI(Dahl/Binder/VI): 0.000
```

## 2.2 결정적 관찰

- **σ swap rate ~0.92**: σ 가 거의 모든 swap 을 accept. 가설 1 의 "σ stuck" 은 *반대 방향*으로 강력함. σ 는 sweep 마다 거의 prior(uniform on Perm(P)) 에서 다시 뽑히는 상태.
- **K_+ posterior 정확** (median=4, mean=3.95). 하지만 sd=1.71 로 K=2~6 사이 진동 가능성.
- **hclust 작동, Dahl/Binder/VI K=1 collapse**: PSM 의 absolute level 은 높지만 contrast 가 매우 작음.

---

# Task 3. PSM 의 within/between 직접 측정

## 3.1 측정

t1_long 결과 폴더의 [epa_co_cluster.csv](../plot/simulation_4_layered_v12_2_sens_a1.0_t1.0_fix_long/epa_co_cluster.csv) 에서:

```
PSM diag mean = 1.000   (정상)
Within true cluster off-diag PSM: mean=0.743 sd=0.028 q25=0.723 q75=0.761
Between true cluster PSM:        mean=0.745 sd=0.028 q25=0.724 q75=0.768
Within/Between ratio = 1.00
PSM off-diag overall range: [0.682, 0.815]
Row-mean PSM (excl diag): mean=0.744 sd=0.005
```

## 3.2 해석

- 사용자가 본 "co-cluster matrix uniform" 의 정량 확인. ratio=1.00 은 within 이 between 보다 미세하게 *낮을 수도 있을* 정도로 cluster 신호 부재.
- PSM absolute 가 0.7 로 매우 높음. 4-cluster {30,30,30,30} 의 random-co-cluster 기댓값은 0.25. 0.7 은 chain 이 K=2 with sizes {~100, ~20} 같은 *over-merged state* 도 자주 visit 한다는 증거 (e.g. {100,20}: (100/120)² + (20/120)² = 0.722).
- median K_+=4 인데 chain 이 K=2 도 자주 방문 → mean K_+=3.95, sd K_+=1.71.

---

# Task 4. σ-freeze 가설 실험

## 4.1 실험 설계

[exp_sigma_freeze.R](../exp_sigma_freeze.R), 같은 데이터 (seed 20260501), 20k iter, 3 regime:

- X1: cluster-aligned σ + `n_perm_swaps = 0` (σ frozen at GT order)
- X2: random σ + `n_perm_swaps = 0` (σ frozen at random order)
- X3: cluster-aligned σ + `n_perm_swaps = 10` (slow updates)

α=1, τ=1 fixed 공통.

## 4.2 결과

| Regime | σ_swap | split | merge | mean K_+ | within | between | ratio | hclust ARI | Dahl ARI |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (free σ, M_perm=120, 120k) | 0.917 | 0.057 | 0.498 | 3.95 | 0.743 | 0.745 | 1.00 | 0.978 | 0 |
| **X1: cluster-aligned freeze** | 0.000 | 0.088 | 0.429 | 4.46 | 0.711 | 0.674 | 1.06 | **−0.001** | 0 |
| X2: random freeze | 0.000 | 0.064 | 0.487 | 3.94 | 0.775 | 0.710 | 1.09 | 0.978 | 0 |
| X3: cluster-aligned slow | 0.915 | 0.064 | 0.492 | 4.10 | 0.773 | 0.710 | 1.09 | 0.932 | 0 |

## 4.3 해석 — 가설 *반박*

- **σ freeze 는 PSM contrast 를 거의 못 살림** (1.00 → 1.06–1.09). Dahl/Binder/VI 여전히 K=1 collapse.
- **Cluster-aligned σ 는 *오히려 해롭다*** (X1 hclust ARI=−0.001!). 이유는 Task 5 에서 분석.
- σ identity 보다 α prior 자체가 dominant.

---

# Task 5. α prior penalty 의 정량 분석

## 5.1 EPA new-cluster factor

`log_epa_pmf` 에서 q 가 σ 의 step t 에 *처음 나타나는* cluster label 일 때:
$$P(\text{new cluster at step } t) \propto \frac{\alpha + \delta \cdot q_{\text{prev}}}{\alpha + t}$$

δ=0, α=1 에서: factor = 1/(1 + t).

## 5.2 Cluster-aligned σ 에서의 over-merging (X1 의 메커니즘)

cluster-aligned σ 라면 cluster 1 의 32 items 가 σ 의 첫 32 자리, cluster 2 의 첫 item (true c=2) 은 step t=33 에 등장.

step 33 에서 single-item Gibbs candidate:
- "Existing cluster 1": factor = (32/33) × (sum_S/sum_all). prefix 가 cluster 1 only 라 sum_S=sum_all → ratio=1 → factor = **32/33 ≈ 0.97**.
- "New cluster 2": factor = 1/33 ≈ **0.03**.

→ item 33 이 cluster 1 으로 흡수될 확률 97%. **EPA prior 가 cluster 2 의 첫 item 의 new-cluster 생성을 강하게 막는다.**

cluster-aligned σ 에서 K=4 가 자연스럽게 형성되려면 step 33, 65, 97 에서 모두 새 cluster 생성:
$$P(\text{K=4 path}) \approx \frac{1}{33} \cdot \frac{1}{65} \cdot \frac{1}{97} \approx 4.8 \times 10^{-6}$$

거의 불가능. → X1 이 ARI=0 인 이유.

## 5.3 Random σ 에서의 EPA prior K-편향

K=4 partition vs K=1 partition 의 prior ratio 를 random σ 에서 계산:

- existing-cluster factors: z 가 잘 cluster 되어 있을 때 (within λ ≈ 0.9, between λ ≈ exp(-8) ≈ 0.0003), K=1 과 K=4 모두 sum_S/sum_all ≈ 1 → 동일.
- new-cluster factors 만 차이:
  - K=1: 1 (첫 item 만)
  - K=4: 1 × (1/30) × (1/60) × (1/90) ≈ 6.2 × 10⁻⁶ (random σ 에서 새 cluster 가 균등 분포되었다고 근사)
- **K=1 / K=4 prior ratio ≈ 162,000**

LSIRM b-likelihood 가 K=4 를 선호하는 정도와 EPA prior 가 K=1 을 선호하는 정도가 균형을 이뤄 chain 이 K=2 와 K=4 mode 를 모두 visit. **PSM 은 두 mode 의 평균이라 contrast 가 사라짐**.

## 5.4 결론

PSM uniformity 의 근본 원인은:

1. EPA prior 가 α=1, δ=0 setting 에서 K=1 을 K=4 대비 5 자리수 penalize.
2. LSIRM b-likelihood 가 K=4 를 선호 → chain 이 K=2 와 K=4 mode 를 모두 visit.
3. PSM = 두 mode 평균 → uniform 으로 collapse.

σ randomness 는 부수적 — σ-freeze 해도 prior penalty 자체가 안 사라짐.

---

# Task 6. α-sweep (완료)

## 6.1 가설

α 가 크면 new-cluster factor (α+δq_prev)/(α+t) → α/(α+t) 가 커지므로 K=4 의 prior penalty 완화. K=2 mode 에 머무는 시간 감소 → PSM contrast 회복 예상.

## 6.2 실험 결과

[exp_alpha_sweep.R](../exp_alpha_sweep.R), 20k iter, 3 regime (free σ, τ=1 fixed, α ∈ {1, 5, 10}):

| α | σ_swap | split | merge | mean K_+ | sd K_+ | within | between | ratio | hclust ARI | Dahl ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.914 | 0.059 | 0.482 | **3.99** | 1.72 | 0.782 | 0.723 | 1.08 | 0.978 (K=4) | 0 (K=1) |
| 5 | 0.785 | 0.374 | 0.281 | **15.42** | 3.59 | 0.327 | 0.197 | **1.66** | 0.823 (K=15) | 0.149 (K=27) |
| 10 | 0.765 | 0.744 | 0.154 | **25.69** | 4.08 | 0.186 | 0.072 | **2.58** | 0.727 (K=26) | 0.139 (K=35) |

## 6.3 해석 — 가설 *부분 확정, 그러나 실용적으로 폐기*

- **PSM contrast 는 가설대로 회복** (1.08 → 1.66 → 2.58).
- **그러나 mean K_+ 가 4 → 15 → 26 으로 폭발** — true K=4 대비 over-splitting.
- split rate 가 α 와 함께 0.06 → 0.37 → 0.74 로 증가, merge rate 는 0.48 → 0.28 → 0.15 로 감소. detailed balance 가 split-uphill 로 이동.
- α=1 의 mean K_+ = 3.99 가 가장 정확. α 를 키우면 K=2 mode 진동은 사라지지만 cluster 가 fragmented 됨 → Dahl ARI 0.149 (K=27) 같은 잘못된 partition.
- **EPA prior 단독으로는 sweet spot 없음**: α=1 에서 under-clustering (K=2 mode 진동), α≥5 에서 over-clustering (K=15–26). z 의 정보가 EPA pmf 의 existing-cluster factor 에 충분히 sharply 들어가지 않아서, prior 가 K 를 결정.

## 6.4 결론

사용자의 v12_2 lock-in (α=1, τ=1 fixed) 은 K-recovery 측면에서 옳은 선택. PSM uniformity 는 α 조정으로 못 고친다 — α 를 움직이면 K_+ 가 통째로 옮겨감. **MIDUS v12 의 α=1 lock-in 유지 권장**.

---

# 권장 액션 (α-sweep 결과 반영 후 수정)

## A1 (단기, 안전 fallback): hclust(b̂) 를 official estimator 로

PSM-based estimators (Dahl/Binder/VI) 는 v12 EPA 의 K-mode 진동 때문에 본질적으로 unreliable. α-sweep 모든 regime 에서 hclust(b̂) 는 ARI 0.73–0.98 로 robust (α=1 에서 0.978). MIDUS 보고에서 main estimator 로 채택, PSM-based 는 secondary 로 명시.

## A2 (중기, post-hoc): mode-conditional PSM

`epa_K_plus` 의 mode (K_+ = 4) 근처 iter 만 필터링한 c samples 로 PSM 재계산:

```r
mask <- abs(samps$epa_K_plus - 4) <= 1   # K_+ ∈ {3,4,5}
c_filt <- samps$epa_c[mask, ]
co_filt <- crossprod(model.matrix(~ factor(c_filt[1,])-1)) > 0  # 등등 — Dahl 식 재계산
```

K=2 over-merged sweep 을 제거하면 within/between PSM contrast 가 살아남을 것으로 기대. 코드 변경 minimal (post-hoc), MIDUS 분석 파이프라인에 추가.

## A3 (장기): prior 자체 변경

α-sweep 결과는 EPA prior 자체의 mode-shifting 한계를 드러냄. PSM-based estimator 가 필요한 분석이라면:

- **v11 NIW mixture 회귀** (사용자 메모리 v11 looseSigma PEAR 0.42 — v12 의 0.01 대비 압도적).
- **v11t Student-t mixture** (오늘 0507_history.md 의 v11t tuned, silhouette 0.628, ARI 0.903).
- 또는 σ 없는 prior (PPMx, distance-based partition prior).

## ❌ A0 (폐기): α 키우기

α=5/10 은 K_+ 폭발로 partition 망가짐. 검토 시도 결과 명시적으로 폐기.

---

# Task 7. Cluster size 분포 — K_+ trace 가 거짓말함을 증명

## 7.1 동기

PSM mean off-diag = 0.744 ↔ HHI = 0.746 ↔ K_eff = 1.34. K_+ trace 의 mean=3.99 와 모순. 진짜 cluster size 분포 직접 측정.

## 7.2 측정 결과 ([exp_cluster_sizes.R](../exp_cluster_sizes.R), α=1, free σ, 20k iter, 2400 saved)

| Metric | mean | median | sd | range |
|---|---|---|---|---|
| K_+ (raw count) | 3.99 | 4 | 1.72 | [1, 13] |
| HHI (size dispersion) | 0.740 | 0.799 | 0.209 | [0.179, 1.000] |
| K_eff (=1/HHI) | 1.52 | 1.25 | — | [1.00, 5.58] |
| Largest cluster share | 0.82 | 0.89 | 0.17 | [0.25, 1.00] |
| 2nd largest share | 0.12 | 0.07 | 0.12 | [0.00, 0.50] |

Largest-cluster share 히스토그램: 28% 이상이 share > 0.95 (사실상 K=1), 58% 가 share > 0.85.

샘플 (첫 10 saved iters):
```
iter 1:  K=1  sizes=120
iter 2:  K=2  sizes=118,2
iter 5:  K=4  sizes=114,4,1,1   <- K_+=4 지만 K_eff=1.05
iter 9:  K=5  sizes=110,3,3,2,2  <- K_+=5 지만 K_eff=1.16
iter 10: K=7  sizes=47,40,28,2,1,1,1
```

## 7.3 결론 — chain 은 K=1 mode 에 정착

K_+ trace 의 4 는 misleading. 진짜로는 **한 거대 cluster (mean 82%) + 떨어져나갔다 합쳐졌다 하는 singleton 들의 카운트**. 28% iter 는 완전히 K=1, 추가 30% 도 거대 cluster 가 85%+ 차지. EPA stationary distribution 의 mode 가 K=1 임이 직접 확인됨.

## 7.4 메커니즘 — b/c 분리

- **b update** ([cpp:990-1006](../data/my_LSIRM_FMC_v12.cpp#L990-L1006)): N(0, σ_b²) Gaussian prior + LSIRM likelihood + EPA prior 변화. LSIRM likelihood 가 dominant → b 는 4-cluster geometry 유지.
- **c update** (single-item Gibbs + split-merge): EPA pmf 만 평가. New-cluster penalty 1/(α+t) + existing-cluster rich-get-richer. EPA prior 가 dominant → c 는 K=1 mode 로 collapse.

→ **b 와 c 가 분리 (decoupled)**. z 는 잘 cluster 되어 있는데 c 는 그 정보를 못 쓰는 (혹은 prior 가 너무 강해서 못 따라가는) 상태. 사용자가 보고한 "z 는 잘 추정되는데 PSM uniform" 의 정확한 메커니즘.

---

# 산출 파일 (최종)

- [exp_sigma_freeze.R](../exp_sigma_freeze.R) — σ-freeze 가설 실험 (반박)
- [exp_sigma_freeze.log](../exp_sigma_freeze.log) — X1/X2/X3 결과
- [exp_sigma_freeze_results.rds](../exp_sigma_freeze_results.rds)
- [exp_alpha_sweep.R](../exp_alpha_sweep.R) — α-sweep 실험 (가설 부분 확정, 실용적 폐기)
- [exp_alpha_sweep.log](../exp_alpha_sweep.log) — α=1/5/10 결과
- [exp_alpha_sweep_results.rds](../exp_alpha_sweep_results.rds)
- [exp_cluster_sizes.R](../exp_cluster_sizes.R) — Cluster size 분포 직접 측정
- [exp_cluster_sizes.log](../exp_cluster_sizes.log) — HHI/largest-share/sample profiles

---

# 산출 파일

- [exp_sigma_freeze.R](../exp_sigma_freeze.R) — σ-freeze 가설 실험
- [exp_sigma_freeze.log](../exp_sigma_freeze.log) — X1/X2/X3 결과
- [exp_sigma_freeze_results.rds](../exp_sigma_freeze_results.rds) — saved diagnostics list
- [exp_alpha_sweep.R](../exp_alpha_sweep.R) — α-sweep 실험
- [exp_alpha_sweep.log](../exp_alpha_sweep.log) — α=1/5/10 결과 (진행 중)
- [exp_alpha_sweep_results.rds](../exp_alpha_sweep_results.rds) — saved diagnostics list (진행 중)
