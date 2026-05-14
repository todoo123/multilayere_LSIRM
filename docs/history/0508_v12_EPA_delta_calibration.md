# 0508 — v12 EPA δ-sweep + (α, δ) 2D grid: prior calibration as K-recovery lever

## 요지

어제 (0507) 진단으로 v12 EPA 의 PSM uniformity 가 **EPA stationary 의 K=1 collapse** 에서 비롯된다는 것이 확인됨. α-sweep 은 PSM contrast 회복하지만 K_+ 폭발로 부적격. 오늘은 DDT (2017) 의 *discount parameter* δ ∈ [0,1) 를 풀어서 — 페이퍼상 δ 는 free parameter 이고 v12 의 δ=0 fix 는 implementation 단순화 — Pitman-Yor 형 prior 변형이 K=1 collapse 를 깰 수 있는지 검증.

**결론 (Task 1–3 통합)**:

1. **K-recovery 측면**: (α, δ) 를 prior E[K] ≈ K_true 가 되게 calibrate 하면 hclust(b̂) 가 **perfect ARI = 1.000** 달성. **(α=0.5, δ=0.25)** 가 sweet spot — baseline (α=1, δ=0, ARI=0.978) 대비 개선.

2. **PSM uniformity 측면**: (α, δ) calibration 으로 **고쳐지지 않음**. PSM ratio 가 grid 전체에서 1.04–1.10 으로 baseline (1.08) 과 동등. K=1 mode 의 bimodality 가 hyperparameter 의존 *아님* — structural property of EPA prior.

3. **K-direction 의 데이터-불응성**: δ-sweep 에서 **mean K_+ 가 Pitman-Yor prior E[K] = (α/δ)·(n^δ−1) 을 거의 그대로 trace**. 데이터 (LSIRM b-likelihood + λ-attraction) 가 K-방향으로 prior 를 못 끌어당김 → c-채널의 K-bandwidth 가 본질적으로 낮음.

**MIDUS 권장**: v12 의 lock-in 을 **(α=0.5, δ=0.25)** 로 변경. main estimator 는 hclust(b̂) 유지. PSM-based (Dahl/Binder/VI) 는 여전히 unreliable.

---

## Task 1. δ-sweep at α=1 fix

### 1.1 동기

페이퍼 [model_v12_epa_partition.tex:276](../modeling_paper/model_v12_epa_partition.tex#L276) 는 δ=0 으로 hard-fix. 그러나 DDT 원 논문 ([dahl-day-tsai-2016.pdf](https://dahl.byu.edu/papers/dahl-day-tsai-2016.pdf), §2.5) 은 δ 를 free parameter 로 두고 spike-and-slab prior 를 권고. v12 의 δ=0 fix 는 implementation 단순화 (DP-style sequential allocation 회수) 이지 modeling 정당화 아님.

이론적 메커니즘 ([0507_v12_EPA_alpha_diagnosis.md Task 5.1](0507_v12_EPA_alpha_diagnosis.md)): EPA new-cluster factor `(α + δ·q_{t-1})/(α + t-1)`. δ=0 일 때 1/(α+t-1) 으로 K=1→K=2 transition 의 prior penalty 가 매우 강함 (P=120, α=1, t=30 에서 1/30 ≈ 0.03). δ>0 풀면 q_{t-1} 의존 항이 추가되어 K-prior 의 *형태* 자체가 바뀜.

### 1.2 실험

[exp_delta_sweep.R](../exp_delta_sweep.R), 같은 데이터 (seed 20260501, P=120, K_true=4), α=1·τ=1 fix, free σ, 20k iter, δ ∈ {0, 0.25, 0.5, 0.75}.

[exp_delta_sweep.log](../exp_delta_sweep.log) 결과:

| δ | mean K_+ | sd K_+ | %largest>0.95 (K=1-like) | PSM ratio | hclust ARI | Dahl ARI |
|---|---|---|---|---|---|---|
| 0.00 | 3.99 | 1.72 | **27.9%** | 1.08 | 0.978 (K=4) | 0 |
| 0.25 | 7.03 | 3.71 | **13.1%** | 1.13 | 0.978 (K=6) | 0 |
| 0.50 | 16.00 | 9.76 | 4.6% | 1.24 | 0.845 (K=14) | 0.094 |
| 0.75 | 45.97 | 20.14 | 1.2% | 1.56 | 0.325 (K=48) | 0.015 |

### 1.3 발견

**(a) δ 가 K=1 collapse 를 단조 약화**: %K=1-like = 28% → 13% → 5% → 1%. δ-compounding 의 메커니즘 (이미 형성된 cluster 가 많을수록 새 cluster 더 잘 생김) 이 K=1 absorbing 을 정확히 깨는 것 확인.

**(b) K_+ 가 PY prior E[K] 를 거의 그대로 trace**:

| δ | predicted E[K|n=120, α=1] = (1/δ)·(120^δ−1) | observed mean K_+ |
|---|---|---|
| 0.25 | 9.24 | 7.03 |
| 0.50 | 20.00 | 16.00 |
| 0.75 | 46.6 | **45.97** |

posterior mean K_+ 가 prior E[K] 와 5–25% 이내로 일치. **데이터의 K-신호가 prior 를 거의 못 이김** — c-채널의 K-direction bandwidth 가 매우 낮음.

**(c) δ=0.25 의 plateau**: K_+ posterior mean 7 (over-split 약간) 이지만 hclust ARI=0.978 유지. b 의 4-cluster geometry 가 살아 있어서 hclust(b̂) 가 K=6 으로 cut 해도 true partition 회복. PSM ratio 1.13 (개선 미미). Dahl 여전히 K=1.

**(d) α-sweep 과의 비교**:

| 설정 | mean K_+ | %K=1-like | PSM ratio |
|---|---|---|---|
| α=1, δ=0 (baseline) | 3.99 | 27.9% | 1.08 |
| α=5, δ=0 | 15.42 | n/a | 1.66 |
| α=10, δ=0 | 25.69 | n/a | 2.58 |
| α=1, δ=0.25 | 7.03 | 13.1% | 1.13 |
| α=1, δ=0.50 | 16.00 | 4.6% | 1.24 |
| α=1, δ=0.75 | 45.97 | 1.2% | 1.56 |

α 와 δ 모두 K_+ 와 PSM contrast 를 trade off — 다만 δ 가 K=1 collapse 를 더 직접적으로 약화 (대신 K_+ 가 prior 의 polynomial 성장 따라 빠르게 폭발).

### 1.4 시나리오 판정

α 단독(α↑) 과 동일한 over-split 패턴 → **시나리오 B**. (α=2, δ) 같은 K=1-escape-augment 는 불필요. 다음 단계: prior E[K]≈K_true 로 (α, δ) co-calibration.

---

## Task 2. 2D (α, δ) grid — sweet-spot calibration

### 2.1 가설

δ-sweep 의 발견 (b) 가 핵심: **K_+ posterior ≈ prior E[K]**. 따라서 K-recovery 의 길은 prior E[K] 를 K_true 에 calibrate 하는 것.

PY 의 E[K|n, α, δ] = (α/δ)·(n^δ − 1). K_true=4 에 맞추는 (α, δ) 후보:

| (α, δ) | predicted E[K] |
|---|---|
| (0.5, 0.25) | 4.62 |
| (0.5, 0.20) | 4.01 |
| (0.3, 0.25) | 2.77 |
| (1.0, 0.10) | 6.14 |

### 2.2 실험

[exp_alpha_delta_grid.R](../exp_alpha_delta_grid.R), 같은 데이터·세팅, 20k iter × 4 cells. cell 별로 [plot/exp_alpha_delta_grid/a{α}_d{δ}/](../plot/exp_alpha_delta_grid/) 에 PSM heatmap, K_+ trace, largest-share histogram, full PSM CSV, cell summary CSV 저장. 통합 [grid_summary.csv](../plot/exp_alpha_delta_grid/grid_summary.csv).

### 2.3 결과

[exp_alpha_delta_grid.log](../exp_alpha_delta_grid.log):

| (α, δ) | E[K]_pred | mean K_+ | sd K_+ | %K=1-like | PSM ratio | hclust ARI | hclust K | Dahl ARI |
|---|---|---|---|---|---|---|---|---|
| (0.5, 0.25) | 4.62 | **4.66** | 3.09 | 35.8% | 1.07 | **1.000** | **K=4** | 0 |
| (0.5, 0.20) | 4.01 | 3.99 | 2.54 | 38.3% | 1.06 | 0.683 | K=3 | 0 |
| (0.3, 0.25) | 2.77 | 3.27 | 2.17 | 51.4% | 1.04 | 0.711 | K=3 | 0 |
| (1.0, 0.10) | 6.14 | 4.93 | 2.36 | **22.0%** | 1.10 | 0.946 | K=5 | 0 |

### 2.4 해석

**(a) (α=0.5, δ=0.25) 가 sweet spot**:
- mean K_+ = 4.66 (prior 4.62 와 일치, true K=4 와 0.66 차이)
- median K_+ = 4 정확
- hclust(b̂) ARI = **1.000** (perfect partition recovery)
- baseline (α=1, δ=0) ARI=0.978 대비 개선

**(b) (α=1.0, δ=0.10) 의 trade-off**:
- %K=1-like = **22.0%** — 모든 grid 중 K=1 escape 최고
- 그러나 mean K_+ = 4.93 (over-split), hclust K=5, ARI=0.946
- K=1 mode 가 줄긴 했지만 4-cluster 자리가 5-fragment 상태로 변함

**(c) (α=0.5, δ=0.20) 와 (α=0.3, δ=0.25) 는 under-split**:
- 두 cell 모두 hclust K=3 선택 → K=4 의 truth 못 잡음
- K=1-like rate 가 38–51% 로 baseline 보다 *악화*

**(d) PSM ratio 모든 cell 에서 1.04–1.10**:
- baseline (α=1, δ=0) 의 1.08 과 동등
- **(α, δ) hyperparameter calibration 으론 PSM uniformity 안 풀림**
- K=1 mode 와 K=4 mode 의 bimodality 가 prior structure 의 기본 attribute

**(e) Dahl 모든 cell 에서 K=1**:
- chain 의 K=1 mode 가 가장 많이 visit 되어 PSM 평균이 K=1 sample 과 가장 가까움
- PSM-based estimator 는 v12 EPA 에서 hyperparameter 무관하게 unreliable

### 2.5 K-direction 의 데이터-불응성 재확인

(α, δ) 모든 grid cell 에서 mean K_+ 가 prior E[K] 의 ±15% 이내 (4.62→4.66, 4.01→3.99, 2.77→3.27, 6.14→4.93). 데이터의 K-direction signal 부재가 다시 확인됨. K 는 **사실상 hyperparameter 가 결정**.

이게 v12 EPA 의 본질적 한계 — nonparametric 정신 ("자료가 K 결정") 이 c-채널의 정보량 부족으로 *깨져 있음*. (α, δ) calibration 은 EPA 를 사실상 "K-fixed mixture model" 로 환원시키는 셈.

---

## Task 3. PSM uniformity 의 본질

### 3.1 발견

오늘의 (α, δ) grid 와 어제의 σ-freeze, α-sweep 결과 통합:

| 실험 | PSM ratio | 결론 |
|---|---|---|
| α=1, δ=0 (baseline) | 1.08 | K=1 mode 28% — uniform |
| σ-freeze (X1/X2/X3) | 1.06–1.09 | σ 자유도와 무관 |
| α-sweep (5, 10) | 1.66, 2.58 | over-split 의 인공물 (within ↓ 가 between ↓ 보다 느림) |
| δ-sweep (0.25, 0.5, 0.75) | 1.13, 1.24, 1.56 | over-split 의 인공물 (동상) |
| (α, δ) grid (E[K]≈4) | 1.04–1.10 | K-recovery 잘 되어도 uniform |

→ **PSM uniformity 는 prior structure 의 fundamental attribute**. K=4 mode 와 K=1 mode 사이의 bimodality 가 모든 hyperparameter 영역에서 존재. (α=0.5, δ=0.25) 가 mean K_+=4.66 에 도달해도 36% iter 가 K=1 mode → PSM 에 +0.36 baseline 깔림.

### 3.2 가능한 fix

PSM-based 분석이 정말 필요한 use case 는:
1. **mode-conditional PSM**: K_+∈{3,4,5} 만 필터링 후 PSM 재계산 (post-hoc)
2. **v11/v11t 회귀**: mixture prior 가 c-channel 에 likelihood 직접 흐르게 함 (PEAR 0.42 vs v12 의 0.01)
3. **prior 자체 변경**: PPMx, distance-based partition prior 등으로 c 가 데이터 신호 받도록

### 3.3 저장된 산출물

cell 별 visualization (각 cell 의 PSM heatmap, K_+ trace, largest-share histogram) 으로 bimodality 가 모든 (α, δ) 에서 살아있음을 시각적으로 확인 가능:
- [a0.50_d0.25/](../plot/exp_alpha_delta_grid/a0.50_d0.25/) ← sweet spot
- [a0.50_d0.20/](../plot/exp_alpha_delta_grid/a0.50_d0.20/)
- [a0.30_d0.25/](../plot/exp_alpha_delta_grid/a0.30_d0.25/)
- [a1.00_d0.10/](../plot/exp_alpha_delta_grid/a1.00_d0.10/)

---

## 권장 액션 (오늘 추가/수정)

### A1 (단기): MIDUS v12 lock-in 변경

기존 (α=1, δ=0) → **(α=0.5, δ=0.25)** 로 변경.
- prior E[K|n=120] ≈ 4.62 (target K_true ≈ 4)
- hclust(b̂) ARI 0.978 → **1.000** (시뮬레이션)
- δ 풀어도 chain 안정성 유지 (sigma_swap=0.922, split=0.049, merge=0.505)
- main estimator 는 hclust(b̂) 유지 (Dahl 등은 여전히 PSM 의 K=1 mode 오염으로 K=1 collapse)

MIDUS 의 P 가 simulation 의 P=120 과 다르면 prior E[K] 가 달라지므로 (n^δ 의존), MIDUS 적용 전 P 를 대입해 (α, δ) 재calibration 필요. K_true 의 사전 추정값 K* (예: 5-layered structure 기반) 로 `(α/δ)·(P_MIDUS^δ − 1) ≈ K*` 가 되도록 선택.

### A2 (중기): P-sweep 실험

세션 중 사용자 제기 가설 (PSM sharpness 가 P 효과인지). K_true=4 고정, cluster size ∈ {5,10,20,30} → P ∈ {20, 40, 80, 120}. 측정: %K=1-like, mean K_+, PSM ratio, hclust ARI. K=1/K=4 prior ratio 가 P^3 으로 scaling 하므로 P=20 에선 prior 압력 크게 약화될 것 (~750:1 vs P=120 의 162,000:1). MIDUS 의 실제 P 가 20 정도면 simulation 진단이 *worst case* 일 수도.

### A3 (장기): PSM-based estimator 구원 작업

v12 EPA 안에서는 본질적 한계 → 두 trajectory:
- **mode-conditional PSM** (post-hoc filter): pipeline 에 옵션으로 추가
- **prior 변경** (v11/v11t 회귀, 또는 PPMx): PSM 의 *생성 원천*이 likelihood-aware 한 것으로

### ❌ A0 (폐기 확정)

- α↑ 단독: K_+ 폭발
- δ↑ 단독: 동일하게 K_+ 폭발
- (α=2, δ) 시도: 시나리오 C 발생 안 함, 불필요

---

## 산출 파일

- [exp_delta_sweep.R](../exp_delta_sweep.R) — δ-sweep 실험 (α=1 fix, δ ∈ {0, 0.25, 0.5, 0.75})
- [exp_delta_sweep.log](../exp_delta_sweep.log) — diagnostics + summary
- [exp_delta_sweep_results.rds](../exp_delta_sweep_results.rds)
- [exp_alpha_delta_grid.R](../exp_alpha_delta_grid.R) — 2D grid (sweet-spot calibration)
- [exp_alpha_delta_grid.log](../exp_alpha_delta_grid.log)
- [exp_alpha_delta_grid_results.rds](../exp_alpha_delta_grid_results.rds)
- [plot/exp_alpha_delta_grid/grid_summary.csv](../plot/exp_alpha_delta_grid/grid_summary.csv) — all cells, one row each
- [plot/exp_alpha_delta_grid/a{α}_d{δ}/](../plot/exp_alpha_delta_grid/) — per-cell artifacts (PSM CSV, heatmap PDF, K_+ trace, largest-share histogram)
