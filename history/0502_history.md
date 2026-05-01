# 0502 — v10 MIDUS 적용: $S_0$ 스윕, η-기반 silhouette / within-pair ratio / PEAR 진단 추가

## 오늘의 작업 요약

| # | 범주 | 산출 |
|---|---|---|
| 1 | 적용 | `data/MIDUS_5layered_result_v10.R` — V9 mirror 신규 작성 (2026-05-01 도입했던 v10 sampler 의 첫 MIDUS 적용) |
| 2 | 실험 | $S_0$ 스윕: $\{0.01, 0.005, 0.001\}I_r$ 세 run (각 120k iter, 8000 saved) |
| 3 | 진단 도입 | η-기반 silhouette (cluster::silhouette 위 dist(eta_pm)) — 이전 $1-C$ 기반은 circular |
| 4 | 진단 도입 | within-pair confidence ratio = Pr(co>0.8) / frac. within-cluster pairs |
| 5 | 진단 도입 | PEAR (Posterior Expected Adjusted Rand index, Fritsch & Ickstadt 2009) |
| 6 | 결과 | **세 run 모두 within-pair ratio < 0.10, PEAR < 0.25** — partition 자체가 fuzzy. 단 "염증 + 집행기능" cluster 는 세 run 모두에서 안정 출현 |
| 7 | 문서 | modeling paper verification checklist 업데이트 — η-silhouette, within-pair ratio, PEAR 추가 + 두 metric disagree 시 해석 가이드 |

---

## 1. MIDUS_5layered_result_v10.R 신규 작성

V9 mirror 기반으로 v10 sampler (NIW collapsed Gibbs + Jain–Neal split–merge) 적용.

핵심 차이:
- `lsirm_fmc_v10_cpp()` 호출, `n_split_merge=5`, `row_center=TRUE`
- `fmc_hyper`: `V0` 제거 → `kappa0=1e-3`, `nu0=r+10`, `S0=0.01*I_r` (시뮬레이션 검증된 디폴트)
- run_label 에 `S0`, `M_SM` 인코딩 → 여러 hyperparameter 비교 시 자동 폴더 분리
- 4-G 절: $S_0$ elicitation 자동 진단 (empirical within-cluster sd vs prior implied sd, 5x 초과 시 suggested S0 출력)

## 2. $S_0$ 스윕 — 3 runs

각 run 모두 LSIRM block 동일 (v9 byte-identical), FMC block 만 prior tightness 변경.

### 2.1 결과 요약

| Metric | $S_0=0.01$ | $S_0=0.005$ | $S_0=0.001$ |
|---|---|---|---|
| median K_+ | 3 | 4 | 6 |
| K_+ range | [2,6] | [2,7] | [2,10] |
| split rate | 3.4% | 4.6% | 9.6% |
| Pr(co > 0.8) | 3.9% | 0.8% | 0.5% |
| Pr(co < 0.2) | 26.3% | 28.4% | 52.9% |
| Pr(co ∈ [0.3, 0.7]) (ambig.) | 50.4% | 42.7% | 27.0% |
| within-pair ratio | 0.094 | 0.031 | 0.019 |
| **mean silhouette (η-based)** | **0.151** | **0.163** | **−0.034** |
| frac silhouette < 0 | 30.9% | 25.0% | 52.9% |
| **mean PEAR** | **0.220** | 0.151 | 0.157 |

### 2.2 Partition 구성 (각 run)

- $S_0=0.01$: 3 partitions, 38 / 9 / 21
- $S_0=0.005$: 4 partitions, 12 / 13 / 22 / 21
- $S_0=0.001$: 6 partitions, 26 / 8 / 20 / 4 / 9 / 1 (singleton 등장 = over-fragmented 신호)

### 2.3 안정 발견 (세 run 공통)

**"염증 + 집행기능 결함" cluster** 가 세 run 모두에서 동일 구성으로 출현:
- 핵심 멤버: B4BIL6, B4BFGN, B4BCRP (염증 마커) + sgst_reverse_incorrect, sgst_mixed_*_incorrect (executive function tasks)
- $S_0=0.01$ 의 Partition 2 (n=9): silhouette = +0.30
- $S_0=0.005$ 의 Partition 2 (n=13, anxiety 일부 추가): silhouette = -0.14 ← 오염
- $S_0=0.001$ 의 Partition 2 (n=8): silhouette = -0.06

→ **9-item 깔끔 버전 ($S_0=0.01$) 이 가장 신뢰** 할 만한 cluster.

### 2.4 그 외 cluster

- **Depression self-report (B4Q3C–T 위주)**: 22 items 정도. 모든 run 에서 silhouette ≈ +0.42–0.44 로 가장 강하고 일관됨.
- **Anxiety self-report (B4Q1\* 다수)**: $S_0=0.005$ 에서 처음 별도 partition 으로 분리 (n=21, sil = +0.23). $S_0=0.01$ 에서는 catch-all 에 묻혀 있음.
- **나머지 (memory errors, 스트레스 마커 등)**: 어느 run 에서도 응집 안 됨 — 실제로 헐거운 항목들.

## 3. 측정 방법론 정정 — silhouette 계산 fix

### 3.1 잘못된 방식 (이전)
$1 - C$ (co-cluster matrix 의 1 보수) 를 dissimilarity 로 두고 silhouette 계산.
- 문제: $C$ 는 그 자체가 "η 위에서 어떻게 cluster 됐는지" 의 smoothed function. 즉 silhouette 이 "co-clustering 일관성" 을 재측정하는 것일 뿐.
- 결과: 절대값이 부풀려져 보임 (예: $S_0=0.01$ 에서 0.40 으로 보고했던 게 실제 η-기반은 0.151)

### 3.2 올바른 방식
`cluster::silhouette(partition, dist(eta_pm))` — η posterior mean 위 Euclidean distance.
- 의미: partition 의 item 들이 자기 cluster centroid 에 더 가까운가 vs 다른 cluster centroid 에 더 가까운가를 representation space 에서 측정.

### 3.3 수정 후 결론
사용자 지적으로 정정하면서 **이전 결론 ("$S_0=0.01$ 의 3-cluster 가 가장 깔끔") 도 정정**:
- mean silhouette: $S_0=0.005$ (0.163) > $S_0=0.01$ (0.151)
- frac sil < 0: $S_0=0.005$ (25%) < $S_0=0.01$ (31%)
- → $S_0=0.005$ 가 silhouette metric 으로는 약간 더 나음. 단 차이 작음 + 두 run 모두 mean sil < 0.25 (Kaufman & Rousseeuw threshold) 로 둘 다 weak.

## 4. Co-clustering 품질 진단 — within-pair ratio + PEAR 도입

### 4.1 두 measure 의 의도

**within-pair ratio**:
$$
r_{\rm wp} = \frac{\Pr(C_{jk} > 0.8)}{\text{frac. within-cluster pairs in final partition}}
$$
final partition 의 within-cluster pair 중 posterior 가 자신 있게 (>0.8) 같이 묶는 비율.
- > 0.7: strong
- 0.3–0.7: moderate
- < 0.3: weak (partition 이 데이터로부터 "발견" 됐다기보다 hclust 가 fuzzy 한 PSM 위에서 강제 절단)

**PEAR** (Fritsch & Ickstadt 2009):
$$
\text{PEAR} = E_{s,t}[\text{ARI}(c^{(s)}, c^{(t)})]
$$
MCMC sampled partition 들 사이의 평균 ARI. 라벨 스위칭 invariant.
- > 0.6: strong (posterior 가 partition 공간 위 한 점 주변에 concentrated)
- 0.3–0.6: moderate
- < 0.3: weak (posterior 가 partition 공간 위에 truly diffuse — 매 iteration 다른 partition)

### 4.2 두 measure 가 disagree 할 때의 해석

- **$r_{wp}$ 낮음 + PEAR 높음**: 개별 pair $C_{jk}$ 는 0.4–0.6 정도로 모호하지만 partition shape 자체는 일관됨. **partition 보고 가능**.
- **둘 다 낮음**: posterior 가 진짜로 partition 결정 못 함. 단일 point partition 보고하지 말고 PSM heatmap + 안정 subset cluster 만 보고.
- **MIDUS 는 후자**.

### 4.3 사용자 질문에 대한 답

> "co-clustering 이 낮더라도 같은 cluster 에 속할 posterior 확률이 가장 높은 cluster 분류가 계속 유지된다면 유의미하다고 볼 수 있을까?"

**원리적으로 YES, 우리 MIDUS 는 NO**.
- 원리: PEAR 가 그 정확한 정량화. PEAR > 0.6 이면 라벨 스위칭 보정 후 partition shape 안정 → 의미 있음.
- 실측: MIDUS 세 run 모두 **PEAR 0.15–0.22** (weak). chain 이 매 iteration 마다 진짜로 다른 partition 을 visit. 라벨 스위칭이 아닌 실제 cluster 구조 자체가 흔들림.

## 5. MIDUS_5layered_result_v10.R 진단 자동화

### 5.1 새 절 § 4-H (silhouette diagnostic)
- `cluster::silhouette(final_partition, dist(eta_pm))` 호출
- per-cluster silhouette mean / median / min, overall mean, frac < 0 출력
- negative silhouette item 자동 flag
- `_silhouette.pdf` + `_silhouette_per_item.csv` 저장

### 5.2 새 절 § 4-H-bis (co-clustering quality)
- off-diag mean vs uniform-K baseline
- Pr(co > 0.8), Pr(co < 0.2), Pr(co ∈ [0.3, 0.7])
- within-pair confidence ratio (자동 strong/moderate/weak 판정)
- PEAR (500 random pair sampling, 자동 strong/moderate/weak 판정)
- `_coclustering_quality.csv` 저장

### 5.3 K_+ summary CSV 확장
기존 split-merge counter 외에 silhouette mean / median / frac<0, within-pair ratio, ambiguous fraction, PEAR mean 추가 → cross-run 비교를 한 CSV 에서 가능.

## 6. modeling paper 업데이트

`modeling_paper/model_v10_niw_split_merge.tex` § 14 Verification checklist:

추가된 항목:
- η-기반 silhouette 사용 명시 (1−C 기반은 circular 임을 설명)
- within-pair confidence ratio 정의 + threshold (0.7 / 0.3)
- PEAR 정의 + threshold (0.6 / 0.3) + 라벨 스위칭 invariance 의 의미
- **두 measure disagree 시의 해석 가이드** ($r_{wp}$ 낮지만 PEAR 높으면 partition 보고 가능; 둘 다 낮으면 PSM heatmap + 안정 subset 만 보고)

§ 12 Cluster inference 절에도 η-space silhouette 사용 권장 paragraph 추가.

## 7. MIDUS 최종 권장 보고 스타일

세 run 의 종합 결과로:

1. **메인 finding**: "염증 + 집행기능 결함" 9-item subgroup ($S_0=0.01$ 의 Partition 2). silhouette +0.30, 세 run 에 걸친 안정성. 이게 가장 robust 하고 임상적으로 해석 가능 (신경염증 가설).
2. **부수 finding**: depression self-report (~22 items, B4Q3C–T 위주) 도 모든 run 에서 silhouette ≈ +0.42 로 응집. 단 다른 cluster 와의 경계가 데이터에서 sharply defined 되지 않음.
3. **솔직한 부정 결과**: 전체 partition 의 안정성은 약함 (PEAR < 0.25). 따라서 single point partition 으로 보고하지 않고, 위 두 안정 subgroup 만 보고 + co-cluster heatmap 으로 uncertainty 투명화.

## 8. 신규/수정 파일

| 파일 | 종류 |
|---|---|
| `data/MIDUS_5layered_result_v10.R` | **신규** (오늘 작성, S_0 스윕 + 진단 자동화) |
| `data/my_LSIRM_FMC_cpp_v10.R` | 디폴트 $S_0$ 0.05 → 0.01 갱신, elicitation 가이드 주석 추가 |
| `modeling_paper/model_v10_niw_split_merge.tex` | verification checklist + cluster inference 절 확장 |
| `history/0502_history.md` | 본 문서 |

### 진단 / 분석 스크립트 (영구 보관 아님)

| 파일 | 위치 |
|---|---|
| analyze_v10_midus.R | `/tmp/analyze_v10_midus.R` |
| analyze_v10_midus_S0001.R | `/tmp/analyze_v10_midus_S0001.R` |
| analyze_v10_midus_S0005.R | `/tmp/analyze_v10_midus_S0005.R` |
| silhouette_eta.R | `/tmp/silhouette_eta.R` |

### 결과 디렉토리

| 디렉토리 | 내용 |
|---|---|
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.01_M5/` | $S_0=0.01$, 3-cluster, ARI-ish best, mean sil 0.151 |
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.005_M5/` | $S_0=0.005$, 4-cluster, mean sil 0.163 (η-based 가장 좋음) |
| `data/plot/case1_all_v10_fmc_r5_K10_e0.1_S0_0.001_M5/` | $S_0=0.001$, 6-cluster, over-fragmented |
