# 0502 — v10 MIDUS 적용, 진단 메트릭 보강, $S_0$/$e_0$/$r$ 스윕, v9 ablation

## 오늘의 작업 (task 단위 요약)

| # | Task | 산출 |
|---|---|---|
| 1 | v10 sampler 의 첫 MIDUS 적용 (`MIDUS_5layered_result_v10.R` 신규) | 3-run $S_0$ 스윕 + 안정 cluster 발견 |
| 2 | Silhouette 측정 방법 정정 ($1-C$ → η factor space) | 이전 결론 ("$S_0=0.01$ best") 정정 → $S_0=0.005$ 가 silhouette 기준 best |
| 3 | Co-clustering 품질 진단 도입 (within-pair ratio, PEAR) | 두 metric 모두 weak → **partition 자체가 fuzzy** 진단 |
| 4 | $r_{\rm fac}$ + $e_0$ 추가 스윕 (r=2/e0=0.05, r=5/e0=0.05) | **모두 악화**. 안정 cluster 깨짐 |
| 5 | V9 ablation (NIW/split-merge 미장착, 동일 $S_0$) | mean K_+ = **1.00 / sd 0** — V10 sampler 없으면 collapse |
| 6 | Modeling paper verification checklist 확장 | η-silhouette, within-pair ratio, PEAR, $S_0$ elicitation 추가 |
| 7 | MIDUS v10 스크립트 자동 진단 절 추가 | §4-G ($S_0$ elicit), §4-H (silhouette), §4-H-bis (co-cluster quality) |

**최종 best operating point (MIDUS v10): $r_{\rm fac}=5$, $e_0=0.10$, $S_0=0.01\,I_r$.**
**가장 robust 한 발견: 9-item "신경염증 + 집행기능 결함" cluster (silhouette $+0.30$).**

---

# Task 1. v10 sampler 의 첫 MIDUS 적용 + $S_0$ 스윕

## 1.1 신규 파일: `data/MIDUS_5layered_result_v10.R`

V9 mirror 기반으로 v10 sampler (NIW collapsed Gibbs + Jain–Neal split–merge) 적용. 핵심 차이:
- `lsirm_fmc_v10_cpp()` 호출, `n_split_merge=5`, `row_center=TRUE`
- `fmc_hyper$V0 → kappa0` 교체 (`kappa0=1e-3`, `nu0=r+10`, `S0=0.01*I_r`)
- run_label 에 $S_0$, $M_{\rm SM}$ 인코딩 → 여러 hyperparameter 비교 시 자동 폴더 분리

## 1.2 $S_0$ 스윕 (3 runs)

각 run 모두 LSIRM block 동일 (v9 byte-identical), FMC block 의 prior tightness 만 변경. 120k iter / 8000 saved.

| Metric | $S_0=0.01$ | $S_0=0.005$ | $S_0=0.001$ |
|---|---|---|---|
| median K_+ | 3 | 4 | 6 |
| K_+ range | [2, 6] | [2, 7] | [2, 10] |
| split rate | 3.4% | 4.6% | 9.6% |
| Pr(co > 0.8) | 3.9% | 0.8% | 0.5% |
| Pr(co [0.3, 0.7]) (ambig.) | 50.4% | 42.7% | 27.0% |
| within-pair ratio | 0.094 | 0.031 | 0.019 |
| **mean silhouette (η-based)** | **0.151** | **0.163** | **−0.034** |
| frac silhouette < 0 | 30.9% | 25.0% | 52.9% |
| **mean PEAR** | **0.220** | 0.151 | 0.157 |

## 1.3 Partition 구성

- $S_0=0.01$: 3 partitions, **38 / 9 / 21**
- $S_0=0.005$: 4 partitions, 12 / 13 / 22 / 21
- $S_0=0.001$: 6 partitions, 26 / 8 / 20 / 4 / 9 / 1 (singleton — over-fragmented)

## 1.4 안정 cluster 발견 (세 run 공통)

**"신경염증 + 집행기능 결함"** cluster 가 세 run 모두에서 동일 핵심 멤버로 출현:
- 핵심: B4BIL6, B4BFGN, B4BCRP (염증마커) + sgst_reverse_incorrect, sgst_mixed_*_incorrect (집행기능)
- $S_0=0.01$ 의 Partition 2 (n=9): silhouette = **+0.30** (깔끔)
- $S_0=0.005$ 의 Partition 2 (n=13, anxiety 일부 추가): sil = −0.14 (오염)
- $S_0=0.001$ 의 Partition 2 (n=8): sil = −0.06

→ **9-item 깔끔 버전 ($S_0=0.01$) 이 가장 신뢰**.

---

# Task 2. Silhouette 측정 방법 정정

## 2.1 잘못된 방식 (이전)

$1 - C$ (co-cluster matrix 의 1 보수) 를 dissimilarity 로 두고 silhouette 계산.

**문제**: $C$ 는 그 자체가 "η 위에서 어떻게 cluster 됐는지" 의 smoothed function. 즉 silhouette 이 "co-clustering 일관성" 을 재측정하는 것일 뿐 — circular.

## 2.2 올바른 방식

```r
cluster::silhouette(partition, dist(eta_pm))
```
η posterior mean 위 Euclidean distance. partition 의 item 들이 자기 cluster centroid 에 더 가까운가 vs 다른 cluster centroid 에 더 가까운가를 representation space 에서 측정.

## 2.3 결론 정정

이전 결론 "$S_0=0.01$ 의 3-cluster 가 가장 깔끔" 정정:
- mean silhouette: $S_0=0.005$ (0.163) > $S_0=0.01$ (0.151)
- frac sil < 0: $S_0=0.005$ (25%) < $S_0=0.01$ (31%)
- → silhouette metric 으로는 $S_0=0.005$ 가 약간 더 나음

단 둘 다 mean sil < 0.25 (Kaufman & Rousseeuw threshold) 로 **두 run 모두 weak**.

---

# Task 3. Co-clustering 품질 진단 도입

## 3.1 두 metric 의 의도

**Within-pair confidence ratio**:
$$
r_{\rm wp} = \frac{\Pr(C_{jk} > 0.8)}{\text{frac. within-cluster pairs in final partition}}
$$
final partition 의 within-cluster pair 중 posterior 가 자신 있게 (>0.8) 같이 묶는 비율.
- > 0.7 strong / 0.3–0.7 moderate / < 0.3 weak

**PEAR** (Posterior Expected Adjusted Rand, Fritsch & Ickstadt 2009):
$$
\text{PEAR} = E_{s,t}[\text{ARI}(c^{(s)}, c^{(t)})]
$$
MCMC sampled partition 들 사이의 평균 ARI. 라벨 스위칭 invariant.
- > 0.6 strong (posterior concentrated) / 0.3–0.6 moderate / < 0.3 weak

## 3.2 두 metric disagree 시 해석

- **$r_{wp}$ 낮음 + PEAR 높음**: 개별 pair $C_{jk}$ 는 0.4–0.6 정도로 모호하지만 partition shape 자체는 일관됨 → **partition 보고 가능**.
- **둘 다 낮음**: posterior 가 진짜로 partition 결정 못 함. point partition 보고 ✗, **PSM heatmap + 안정 subset cluster 만 보고 ✓**.

## 3.3 MIDUS 결과 — 후자에 해당

세 run 모두 두 metric 모두 weak:
- within-pair ratio: 0.019–0.094
- PEAR: 0.151–0.220

→ MIDUS 의 cluster 분할은 본질적으로 fuzzy. partition 전체가 아니라 **안정 subset cluster (염증+집행기능)** 만 보고하는 것이 정직.

## 3.4 사용자 질문에 대한 정량적 답

> "co-clustering 이 낮더라도 같은 cluster 에 속할 posterior 확률이 가장 높은 cluster 분류가 계속 유지된다면 유의미한가?"

**원리적으로 YES, MIDUS 는 NO**. PEAR 가 그 정확한 정량화. PEAR > 0.6 이면 partition shape 안정 → 의미 있음. MIDUS 는 PEAR 0.15–0.22 로 chain 이 매 iteration 진짜로 다른 partition 을 visit.

---

# Task 4. $r_{\rm fac}$ + $e_0$ 추가 스윕 — 모두 악화

## 4.1 r=2 / e0=0.05 / S0=0.005

**가설**: η posterior PC variance ratio (0.85, 0.09, ...) 가 사실상 1–2D 신호 → r=5 의 나머지 3 차원이 noise. r=2 로 줄이면 신호 집중.

**결과**:

| Metric | r=5 / e0=0.10 / S0=0.01 (best) | **r=2 / e0=0.05 / S0=0.005** |
|---|---|---|
| mean silhouette | **0.151** | **0.066** ↓ |
| PEAR | **0.220** | 0.099 ↓ |
| within-pair ratio | 0.094 | 0.012 ↓ |
| frac sil < 0 | 31% | 46% ↑ |

**결정타**: 안정 cluster (염증+집행기능, 이전 sil +0.30) **무너짐**. B4BIL6 sil −0.115, B4BCRP −0.040, sgst_reverse −0.091, sgst_mixed_* −0.14 이상 — 모두 자기 cluster 보다 다른 cluster 에 더 가까움 (negative sil).

**진단**: r=5 → r=2 가 단순 projection 이 아니라 fitting 새로 됨 (empirical η scale 0.002 → 0.012, 6× 증가). r=5 의 PC2 (9% var) 에 살던 inflammation 신호가 r=2 의 PC1 (depression dominant) 와 합쳐져 분리력 상실.

## 4.2 r=5 / e0=0.05 / S0=0.01 (Option A: e0 효과 격리)

**가설**: r=5 는 그대로 두고 e0 만 줄이면 Rousseau–Mengersen emptying 이 강해져 sharper PSM.

**결과**:

| Metric | r=5 / e0=0.10 (best) | **r=5 / e0=0.05** |
|---|---|---|
| mean silhouette | **0.151** | **0.030** ↓↓ |
| PEAR | **0.220** | 0.145 ↓ |
| within-pair ratio | 0.094 | 0.044 ↓ |
| split rate | 3.4% | **2.7%** ↓ |

**또 안정 cluster 깨짐**: e0=0.10 best 의 Partition 2 (n=9, sil +0.30) 가 e0=0.05 에서 Partition 3 (n=9, sil **−0.064**) 으로 무너지고, B4BFGN/B4BCRP 모두 negative sil.

**진단**:
1. e0 작아지면 빈 component reactivation 어려워짐 ($(n_{l,-j}+e_0)$ factor 의 $e_0$ 부분 작아짐) → split rate 감소 확인됨
2. 작은 cluster (9-item) 가 emptying pressure 에 흔들려 응집 풀어짐

## 4.3 결론

**세 변수 모두 swept** ($r \in \{2, 5\}$, $e_0 \in \{0.05, 0.10\}$, $S_0 \in \{0.001, 0.005, 0.01\}$):

| Run | sil | PEAR | within-pair r | 안정 cluster |
|---|---|---|---|---|
| **r=5 / e0=0.10 / S0=0.01** | **0.151** | **0.220** | **0.094** | ✓ inflam+exec (sil +0.30) |
| r=5 / e0=0.10 / S0=0.005 | 0.163 | 0.151 | 0.031 | 약함 |
| r=5 / e0=0.10 / S0=0.001 | −0.034 | 0.157 | 0.019 | over-fragmented |
| r=5 / e0=0.05 / S0=0.01 | 0.030 | 0.145 | 0.044 | ✗ broken |
| r=2 / e0=0.05 / S0=0.005 | 0.066 | 0.099 | 0.012 | ✗ broken |

→ **MIDUS 최종 best: r=5, e0=0.10, S0=0.01** (lock-in).

---

# Task 5. V9 ablation — NIW/split-merge step 의 marginal contribution

## 5.1 실험 설계

V10 의 NIW + split-merge 가 정말로 essential 한지, 아니면 prior tightening 만으로도 같은 효과가 나는지 검증.

**v9** (NIW 없음, split-merge 없음, c_j ∝ ρ_l · φ_r(η_j; μ_l, Σ_l)) 를 **v10 best 와 동일한 hyperparameter** 로 돌림: r=5, e0=0.10, S0=0.01.

[`data/MIDUS_5layered_result_v9.R`](../data/MIDUS_5layered_result_v9.R) 의 `S0 = 0.5` → `S0 = 0.01` 만 변경.

## 5.2 결과

| Metric | **v9** r=5 e0=0.10 S0=0.01 | **v10** r=5 e0=0.10 S0=0.01 |
|---|---|---|
| **mean K_+** | **1.00** | 3.17 |
| **median K_+** | **1** | 3 |
| **sd K_+** | **0.00** | 0.63 |
| K_+ range | [1, 1] | [2, 6] |
| Final partition | 67 / 1 (single + 1 outlier) | 38 / 9 / 21 |
| Mode label diversity | label 10 만 사용 | 다수 라벨 활동 |

→ **v9 는 chain 전체 iteration 에서 K_+ = 1**. 단일 cluster 로 collapse.

## 5.3 해석

**V10 의 sampler innovation 이 essential** 함이 직접 증명됨:

- v9 의 c_j update: $P(c_j = l \mid \cdot) \propto \rho_l \cdot \phi_r(\eta_j; \mu_l, \Sigma_l)$
- 빈 component 의 $(\mu_l, \Sigma_l)$ 는 prior 에서 한 번 drawn 된 임의 위치 → 데이터 likelihood 가 그 위치에서 sample 된 $\eta_j$ 와 매칭될 확률 ≈ 0 → **한 번 비면 영영 reactivation 안 됨**
- $S_0$ 작게 줘도 도움 안 됨: $\Sigma$ 가 작아져 빈 component 영역이 더 좁아짐 → 오히려 reactivation 더 어려움
- **prior tightening 으로는 chicken-and-egg 못 우회**

V10 의 구조적 변경:
1. Collapsed update 의 $(n_{l,-j}+e_0)$ → 빈 component 에 항상 prior weight $e_0$ 부여 → 큰 데이터 영역 cover
2. Split-merge → 큰 partition 변화 직접 제안

→ V10 best 의 "weak silhouette 0.151" 은 **v9 의 0 → 의미 있는 partition 으로 가는 qualitative jump**. paper ablation study 의 핵심 figure / table 로 활용 가능.

---

# Task 6. Modeling paper verification checklist 확장

[`modeling_paper/model_v10_niw_split_merge.tex`](../modeling_paper/model_v10_niw_split_merge.tex) §14 (Verification checklist) 에 추가:

## 6.1 η-기반 silhouette 사용 명시
$1-C$ 기반은 circular 임을 설명. Per-cluster silhouette mean / median / min, overall mean, frac<0 보고. mean sil < 0.25 인 cluster 는 weak 으로 flag.

## 6.2 Within-pair confidence ratio 정의 + threshold
$r_{wp} = \Pr(C > 0.8) / \text{frac. within pairs}$. 0.7 / 0.3 threshold.

## 6.3 PEAR 정의 + threshold
0.6 / 0.3 threshold. 라벨 스위칭 invariance 의 의미 명시.

## 6.4 두 measure disagree 시 해석 가이드
- $r_{wp}$ low + PEAR high → partition shape 안정, 보고 가능
- 둘 다 low → PSM heatmap + 안정 subset 만 보고

## 6.5 §12 Cluster inference 절
η-space silhouette 사용 권장 paragraph 추가.

## 6.6 $S_0$ elicitation 절 (§5.1) — 0501 작업 연장
디폴트 $S_0$ 0.05 → 0.01 권장. Pilot chain → empirical within-cluster sd 측정 → prior implied sd 와 비교 (5–10× 차이 시 tightening). Concrete example 으로 시뮬레이션 수치 인용.

---

# Task 7. MIDUS v10 스크립트 자동 진단 절 추가

[`data/MIDUS_5layered_result_v10.R`](../data/MIDUS_5layered_result_v10.R) 에 자동 진단 절 신설:

## 7.1 §4-G — $S_0$ elicitation diagnostic
- Empirical within-cluster sd (modal label 기반) 자동 계산
- Prior implied sd 와 ratio 출력, 5× 초과 시 WARNING + suggested $S_0$ 출력
- `_S0_elicitation.csv` 저장

## 7.2 §4-H — Silhouette diagnostic (η-based)
- `cluster::silhouette(final_partition, dist(eta_pm))` 호출
- per-cluster silhouette + 전체 mean / frac<0 출력
- Negative silhouette item 자동 flag
- `_silhouette.pdf` + `_silhouette_per_item.csv` 저장

## 7.3 §4-H-bis — Co-clustering quality
- off-diag mean vs uniform-K baseline
- $\Pr(co>0.8), \Pr(co<0.2), \Pr(co \in [0.3, 0.7])$
- Within-pair confidence ratio (자동 strong/moderate/weak 판정)
- PEAR (500 random pair sampling, 자동 판정)
- `_coclustering_quality.csv` 저장

## 7.4 §4-I — K_+ summary CSV 확장
기존 split-merge counter 외에 silhouette / within-pair ratio / ambiguous fraction / PEAR 추가 → cross-run 비교를 한 CSV 에서 가능.

---

# 신규 / 수정 파일 (오늘)

| 파일 | 종류 |
|---|---|
| `data/MIDUS_5layered_result_v10.R` | **신규** (오늘 작성, $S_0$/$r$/$e_0$ 스윕 + 자동 진단 절) |
| `data/MIDUS_5layered_result_v9.R` | 수정 ($S_0$ 0.5 → 0.01, ablation 용) |
| `data/my_LSIRM_FMC_cpp_v10.R` | 디폴트 $S_0$ 0.05 → 0.01 갱신, elicitation 가이드 주석 |
| `modeling_paper/model_v10_niw_split_merge.tex` | verification checklist 확장 (η-silhouette, $r_{wp}$, PEAR, $S_0$ elicitation 가이드, disagreement 해석) |
| `history/0502_history.md` | 본 문서 |

## 결과 디렉토리 (오늘 생성)

| 디렉토리 | 내용 |
|---|---|
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.01_M5/` | **★ 최종 best** |
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.005_M5/` | sil 약간 더 좋지만 within-pair ratio 떨어짐 |
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.001_M5/` | over-fragmented |
| `data/plot/case1_all_v10_fmc_r5_K10_e0.05_S0_0.01_M5/` | Option A — 악화 |
| `data/plot/case1_all_v10_fmc_r2_K10_e0.05_S0_0.005_M5/` | r=2 — 안정 cluster 깨짐 |
| `data/plot/case1_all_v9_fmc_r5_K10_e0.1_S0_0.01/` | **v9 ablation — K_+=1 collapse** |

## 진단 / 분석 스크립트 (영구 보관 아님)

| 파일 | 위치 |
|---|---|
| analyze_v10_midus.R (S0=0.01) | `/tmp/analyze_v10_midus.R` |
| analyze_v10_midus_S0001.R | `/tmp/analyze_v10_midus_S0001.R` |
| analyze_v10_midus_S0005.R | `/tmp/analyze_v10_midus_S0005.R` |
| silhouette_eta.R | `/tmp/silhouette_eta.R` |

---

# 다음 단계

1. **Paper / report 작성** — 안정 발견 (염증+집행기능 9-item cluster) 중심으로
2. **Ablation table** — Task 5 의 v9 vs v10 비교를 paper figure / table 로
3. **Sensitivity analysis 보고** — $S_0$ 스윕 결과를 supplementary 로
4. **(선택) Binder/VI loss partition** — `mcclust::minbinder()` 또는 `mcclust.ext::minVI()` 로 hclust 대체. 현재 PSM 위 hclust cut 보다 formal 한 single best partition.

---

# 0502 (오후 세션) — Simulation 진단 & 모델 회복 정합성 추적

## 오후의 작업 (task 단위 요약)

| # | Task | 산출 |
|---|---|---|
| 8 | `simulation_4_layered_v10.R` proposal SD 그룹-평균 튜닝 (5 iter) | 모든 acc group-mean ∈ [0.3, 0.6] |
| 9 | Element-level acceptance + recovery 진단 | 78% element 만 in target, 거리 cor 0.749 / ARI 0.867 |
| 10 | β 의 +0.4–0.6 location bias 진단 | shrinkage 단독으로는 설명 불가, ridge identifiability |
| 11 | Procrustes 정렬 보강 (`data/my_LSIRM_FMC_cpp_v10.R`) | iterative-mean refine + 단일 rigid anchor; `procrustes_target` 인자 신설 |
| 12 | P 늘려서 1/√P 가설 검증 (P=10→30) | **반증**: bias 거의 변화 없음 → mean(β_true) 가 원인 아님 |
| 13 | `fix_gamma=FALSE` 로 γ–D ridge 풀기 | β1/β3 bias **+0.6→+0.05**, cov 80%→100% |
| 14 | fix_gamma=FALSE 환경 proposal SD 재튜닝 (2 iter) | log_gamma1/2, a, b2 추가 보정 |
| 15 | **위치 truth scale ↔ prior N(0,I) match** (centers×0.7, sd 0.5) | γ posterior 가 truth=1 처음 cover, 모든 cov95 **93–100%** |
| 16 | `simulation_FMC_LSIRM_v10.R` 로 multi vs uni 비교 (N_REP=1) | multi 압승 (DICE 0.76 vs 0.41) |
| 17 | uni 의 β segment 분리 + uni proposal SD 튜닝 | uni 도 acc 통과; β bin/cnt-dich 양호, con-dich 13% (정보 손실) |

**최종 well-specified 시뮬 결과: 모든 parameter cov95 ≈ 95% 회복.** 보고서 [history/simulation_adjustment_report.md](simulation_adjustment_report.md) 에 iteration 단위 자세히.

---

## Task 8. proposal SD 그룹-평균 튜닝 (5 iter)

### 8.1 목표
`simulation_4_layered_v10.R` 의 `common_lsirm_prop_sd` 를 조정해 `result$accept` 의 group mean 을 모두 [0.3, 0.6] 에 넣기.

### 8.2 절차
- 튜닝 효율 위해 `n_iter`/`burnin`/`thin` 을 50000/20000/10 → **5000/2000/5** 로 임시 축소
- 곱셈 heuristic: acc<0.1 → ×0.3, [0.1,0.3) → ×0.6, [0.6,0.9] → ×1.6, >0.9 → ×3.0
- `fix_gamma=TRUE` 로 `log_gamma*` 자동 acc=1 → 튜닝 대상 제외

### 8.3 결과: 5 iter 만에 13/13 진입
| iter | 진입 수 | 주요 조정 |
|---|---|---|
| 01 | 6/13 | alpha1/3/4 ×1.6, beta3 ×1.6 등 |
| 02 | 7/13 | alpha2 ×1.2 등 |
| 03 | 10/13 | alpha1/3 ×1.2 |
| 04 | 12/13 | alpha4 ×1.05 |
| 05 | **13/13 ✓** | converged |

최종 SD: `alpha=(1.15, 0.48, 1.15, 0.92), beta=(0.44, 0.13, 0.37), a=0.33, b=(0.33, 0.20, 0.30, 0.26), log_kappa=0.20`.

---

## Task 9. Element-level 진단

### 9.1 발견
group mean 이 [0.3, 0.6] 안이라도 **element 별** acc 분포는 더 넓음:
- 870 개 element 중 **678 (77.9%) 만 in target**, 21.8% > 0.6 (특히 alpha3 42%, alpha4 32%, beta4_thr 의 spread 0.260–0.868)
- 단일 SD 로는 그룹 내 모든 element 를 [0.3, 0.6] 에 넣는 것이 본질적으로 어려움

### 9.2 사후 회복은 양호
- distance cor 0.749, ARI 0.867, median K_+ = 4 (= K_true)
- 모형 자체는 정상 — 이슈는 element별 SD heterogeneity

---

## Task 10. β 의 location bias 진단

### 10.1 측정
| | mean(β_true) | mean(β_post) | bias | raw cov95 | **centered cov95** |
|---|---|---|---|---|---|
| beta1 | -0.327 | +0.287 | **+0.614** | 80% | **100%** |
| beta2 | -0.064 | +0.380 | **+0.444** | 70% | 90% |
| beta3 | +0.226 | +0.766 | **+0.539** | 80% | **100%** |

centering 후 coverage 거의 100% — **상대 패턴은 정확**, layer 별 단일 location scalar 만 어긋남.

### 10.2 가설 검증
- **단순 prior shrinkage**: beta3 의 mean(β_true)>0 인데 bias 도 +0.5 → 부호 안 맞음, **기각**
- **(α+c, β+c) ridge**: alpha shift 거의 0, beta shift 0.5 → 함께 안 움직임, **기각**
- **γ–D ridge 로 추정**: γ=1 강제로 D 가 shrunk → ETA 보존 위해 β 가 흡수 (Task 13 에서 확인)

---

## Task 11. Procrustes 정렬 보강

### 11.1 진단 — 원래 wrapper 의 문제
`data/my_LSIRM_FMC_cpp_v10.R` 구버전 line 265: `ref_idx <- n_save` — 마지막 iter 를 reference 로 정렬. 임의의 한 sample 의 frame 에 chain 전체가 anchored → 위치 element-level cov95 15–50% (a 35.7%, b1 50%, b2 15%, b3 35%, b4 35%).

### 11.2 해결
1. `procrustes_target = NULL` 인자 추가 (default; 실데이터에선 자동 fallback)
2. 두 단계 정렬:
   - **Step 1**: iterative posterior-mean refinement (3 pass) — chain 을 self-consistent 한 frame 으로 수렴
   - **Step 2 (옵션)**: `procrustes_target` 주어지면, posterior mean → target 의 **단일 rigid transform** 을 모든 iter 에 일괄 적용 (rotation+reflection+translation, no scaling — `vegan::procrustes(scale=FALSE)`)
3. 시뮬에서 `procrustes_target = list(a=A_true, b1=B1_true, ...)` 명시 전달

### 11.3 효과 (raw / 옛 정렬 → P1 후)
| param | cov95 전 → 후 |
|---|---|
| a | 35.7% → **73.7%** |
| b1 | 50% → **100%** |
| b2 | 15% → **90%** |
| b3 | 35% → **100%** |
| b4 | 35% → **100%** |

per-iter Procrustes-to-truth 는 **부적절** (within-chain 분산 축소 → CI 좁아짐). 단일 rigid anchor 가 정공.

---

## Task 12. P=30 으로 1/√P 가설 검증 — 반증

### 12.1 가설
mean(β_true) 의 변동이 1/√P 로 줄어들면 prior shrinkage bias 도 비례 감소.

### 12.2 결과
| | mean(β_true) P=10 → P=30 | bias P=10 → P=30 |
|---|---|---|
| beta1 | -0.327 → -0.140 | +0.614 → +0.400 |
| beta2 | -0.064 → +0.005 | +0.444 → +0.395 |
| beta3 | +0.226 → +0.063 | +0.539 → +0.324 |

mean(β_true) 가 0 에 가까워졌어도 **bias 는 거의 그대로**. 가설 **반증**. → β bias 는 mean(β_true) 의 함수가 아니라 다른 메커니즘.

---

## Task 13. fix_gamma=FALSE 로 γ–D ridge 풀기

### 13.1 메커니즘
- positions (a, b) 가 prior N(0, I) 로 origin 쪽 shrunk → D shrunk
- `fix_gamma=TRUE` 면 γ=1 고정 → 잔여 γ·D bias 가 (α-β) 에 흡수
- α 는 n=150 데이터에 anchored 못 움직임 → **β 가 흡수**

### 13.2 해법
`fix_gamma = FALSE` 로 풀기. γ 가 상승하며 D shrinkage 보정.

### 13.3 결과
| | bias fix=T → fix=F | cov95 fix=T → fix=F |
|---|---|---|
| beta1 | +0.400 → **+0.002** | 90% → **100%** |
| beta3 | +0.324 → **+0.061** | 93% → **100%** |
| γ·D L1 | -0.441 → +0.129 | 88% → **95.9%** |
| γ·D L3 | -0.427 → -0.024 | 88% → **96.4%** |

γ posterior 는 ~1.3–1.5 로 anchored (γ=1 cover 못 함) — γ 가 D shrinkage 흡수 중 → identifiability 한계.

---

## Task 14. fix_gamma=FALSE 환경 proposal SD 재튜닝

새 환경에서 일부 acc 외 구간:
- log_gamma1 0.244 → SD 0.10→0.07
- log_gamma2 0.135 → SD 0.05→0.018
- a 0.280 → SD 0.33→0.26
- b2 0.224 → SD 0.20→0.13

**2 iter** 만에 모두 [0.3, 0.6] 진입.

---

## Task 15. 위치 prior 와 truth scale match — well-specified 검증

### 15.1 본질 진단
LSIRM 모델 ([`data/my_LSIRM_FMC_v10.cpp:984-985`](../data/my_LSIRM_FMC_v10.cpp#L984-L985)):
```cpp
double lp_curr = -0.5 * dot(a_old, a_old);  // a ~ N(0, I_d)
```
**위치 prior 가 sd=1 로 hardcoded** (latent identifiability 보호용). 그러나 truth 는 ±1.5 corner — prior 와 **scale 불일치**.

### 15.2 해결 (model 손대지 않고 truth scale 줄임)
- `centers_resp = centers_meta * 1.5` → ` * 0.7`
- `sd_cluster_resp = 0.80` → `0.5`
- per-coord var: 2.89 → 0.74 (≈ prior 1)

### 15.3 결과 — 모든 parameter cov95 ≈ 95% 도달
| param | cov95 | bias |
|---|---|---|
| alpha1..4 | 95.3 / 96.7 / 95.3 / 96.7% | ≤ |0.06| |
| beta1..3 | 100 / 96.7 / 96.7% | ≤ |0.05| |
| beta4_thr | 100% | +0.06 |
| a / b1..b4 | 96.7 / 98.3 / 95.0 / 96.7 / 93.3% | ≈ 0 |
| **γ·D L1..L4** | **95.8 / 96.0 / 96.5 / 97.3%** | ≤ |0.08| |
| **γ posterior** | **모두 truth=1.0 cover** ✓ | |

→ well-specified 시뮬에서 모든 회복 정상. β bias 의 root cause 는 **위치 prior 와 truth scale 불일치** 였음.

---

## Task 16. Multilayer vs Unilayer (dichotomized) 비교 — N_REP=1

### 16.1 setup
[`simulation_FMC_LSIRM_v10/simulation_FMC_LSIRM_v10.R`](../simulation_FMC_LSIRM_v10/simulation_FMC_LSIRM_v10.R) 동일 설정 적용 (P=30, prior-match, fix_gamma=FALSE, 튜닝된 SD, procrustes_target=truth). uni 의 b1 는 4-layer truth 를 stack 해 anchor.

### 16.2 Coverage 비교
| param | multi cov95 | uni cov95 |
|---|---|---|
| alpha | 93–97% (4 layer 각각) | **82%** (3-layer mean 비교) |
| beta | 92–100% | **62.2%** |
| dist | 95–97% | **79.6%** |

### 16.3 Cluster recovery
| | DICE | ARI |
|---|---|---|
| multi | **0.755** | **0.675** |
| uni | 0.474 | 0.287 |

→ multilayered 압승.

---

## Task 17. uni β segment 분리 + uni proposal SD 튜닝

### 17.1 β coverage 의 segment 분해 (uni)
| segment | 30 items | cov95 |
|---|---|---|
| **bin** (1-30, 진짜 binary) | 30 | **76.7%** |
| **con_dich** (31-60, 연속→이진) | 30 | **13.3%** |
| **cnt_dich** (61-90, 카운트→이진) | 30 | **96.7%** |

→ con_dich 가 60% 평균을 끌어내림. continuous 의 dichotomization 이 β scale 자체를 misspecify. 본질적 information loss.

### 17.2 uni proposal SD 튜닝
multi 환경에서 uni 는 alpha1=0.228, log_gamma1=0.162 (외 구간):
- alpha1 SD 1.15 → **0.69** (×0.6)
- log_gamma1 SD 0.07 → **0.035** (×0.5)

`uni_lsirm_prop_sd` 별도 list 생성. **1 iter** 만에 모두 [0.3, 0.6] 진입.

### 17.3 Procrustes alignment 책임 가설 기각
양쪽 fit 모두 truth target 으로 anchor 적용 (`scale=FALSE` 로 rotation+reflection+translation 만, scaling 없음). uni 의 낮은 성능은 alignment 가 아닌 **dichotomization 정보 손실 + 단일 α/γ 강제** 의 산물.

---

## 산출물

### 변경된 파일
| 파일 | 변경 |
|---|---|
| `data/my_LSIRM_FMC_cpp_v10.R` | `procrustes_target` 인자, iterative-mean + rigid anchor (140줄 변경) |
| `simulation_4_layered_v10.R` | P=30, prior-match, fix_gamma=FALSE, 튜닝된 SD, truth target |
| `simulation_FMC_LSIRM_v10/simulation_FMC_LSIRM_v10.R` | 동상, uni 별도 SD list, β segment 분리, fit 저장 추가 (65줄 변경) |
| `history/simulation_adjustment_report.md` | iteration 단위 튜닝 기록 (신규) |

### 미수정 (서버에 그대로 있어야)
- `data/my_LSIRM_FMC_v10.cpp` (C++ — 위치 prior sd=1 hardcoded, 변경 안 함)

### 데이터 산출물
- `simulation_FMC_LSIRM_v10/coverage_per_rep_v10.csv`, `cluster_dice_v10.csv`, `rep01_fits_v10.rds`
- `plot/simulation_4_layered_v10_rfac2_norowctr_S0_0.01/` (final full MCMC plots)
- `logs/sim_v10_*.log` (iteration 단위 로그)

---

## 핵심 finding

1. **Procrustes 정렬은 시뮬레이션 평가 시 truth-anchored 로 가야 함** (실데이터 fallback 은 iterative posterior mean). per-iter to truth 는 분산 축소 → 잘못된 결과.
2. **β location bias 의 root cause 는 (α–β) ridge 가 아니라 γ–D ridge**. fix_gamma=FALSE 로 γ 가 D shrinkage 흡수해 풀림.
3. **위치 truth scale 을 prior N(0, I) 와 맞추는 것이 본질적 well-specified test**. 모델 prior 의 sd=1 은 latent identifiability 보호용이라 풀 수 없음.
4. **dichotomization 의 정보 손실은 본질적 한계**. continuous→binary 가 특히 심각 (β cov 13%).
5. **proposal SD 튜닝은 environment-specific** (fix_gamma 옵션, n/P, 데이터 종류에 따라 다름). multi 와 uni 도 별도 SD 필요.

## 다음 단계 (서버 측)

1. [`simulation_FMC_LSIRM_v10/simulation_FMC_LSIRM_v10.R`](../simulation_FMC_LSIRM_v10/simulation_FMC_LSIRM_v10.R) 의 `BASE_DIR`/`MODEL_DIR` 경로 서버용으로 수정
2. `N_REP = 1L` → 본격 비교용 (예: 10 또는 그 이상) 으로 복원
3. 수정된 두 파일 (`data/my_LSIRM_FMC_cpp_v10.R`, `simulation_FMC_LSIRM_v10.R`) 함께 deploy
4. 결과 받아서 N_REP 평균 cov95 / DICE / ARI 표 작성

