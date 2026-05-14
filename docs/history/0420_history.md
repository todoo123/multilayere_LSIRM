# 0420 — MIDUS 전처리 v4 + per-item κⱼ LSIRM v6 + unilayered 비교

## 오늘의 작업 (요약)

| # | 범주 | 산출 |
|---|---|---|
| 1 | 전처리 | `MIDUS_preprocess_v4.R`, `MIDUS_preprocess_refresher_v4.R` 신규 |
| 2 | 경로 정리 | `MIDUS_5layered_result_v5.R`, `MIDUS_5layered_result_v6.R` → v4 preprocess source |
| 3 | 모델 | `my_LSIRM_5layered_nonhierarchical_v6.cpp` (+ R wrapper) — **per-item kappa_j** |
| 4 | result pipeline | `MIDUS_5layered_result_v6.R` — v5 구조 기반 재작성, v4 preprocess + v6 모델 호출 |
| 5 | 공통 함수화 | `utils.R`에 `kmeans_cluster_b`, `binarize_by_colthreshold`, `make_binarized_multilayer_for_lsirm` 추가 |
| 6 | 비교 분석 | multilayered vs unilayered(dichotomized) LSIRM → k-means 비교 플로팅 |

---

## 1. v4 전처리 (variable selection 교체)

### 1-1. 신규 파일
- `data/MIDUS_preprocess_v4.R` (Wave 2)
- `data/MIDUS_preprocess_refresher_v4.R` (Refresher 1)

기존 `MIDUS_preprocess_2_v3.R` 네이밍에서 `_2_` 제거하여 `MIDUS_preprocess_v~~` 형태로 통일.

### 1-2. Layer 재구성 (v3 → v4)
| Layer | v3 | v4 |
|---|---|---|
| `Y_bin`  | CESD (이진화) + 기타 | **bio_cesd_vars** (20) |
| `Y_con`  | 염증/HPA/코그 Z-score | **bio_continuous_vars** (5) |
| `Y_cnt`  | 수면/음주/cog accuracy | **새 cognitive score count** (15) |
| `Y_ord1` | MASQ (28) | **bio_ord5_vars** (28, MASQ) |
| `Y_ord2` | PSQI B4S5 | **EMPTY** |

ord2가 빈 matrix라도 v5/v6 cpp의 `P5=0` 분기를 타기 때문에 모델 코드 수정 불필요.

### 1-3. 새 cognitive score count 변수 (15개)
원변수 → count 변환식 (높을수록 인지↓):

| 변수 | Wave 2 원변수 | 변환 | Refresher 1 |
|---|---|---|---|
| `wl_immediate_omit`            | B3TWLITU | `15 - correct`  | RA3TWLITU |
| `wl_immediate_repetition`      | B3TWLITR | raw             | RA3TWLITR |
| `wl_immediate_intrusion`       | B3TWLITI | raw             | RA3TWLITI |
| `wl_delayed_omit`              | B3TWLDTU | `15 - correct`  | RA3TWLDTU |
| `wl_delayed_repetition`        | B3TWLDTR | raw             | RA3TWLDTR |
| `wl_delayed_intrusion`         | B3TWLDTI | raw             | RA3TWLDTI |
| `catflu_repetition`            | B3TCTFLR | raw             | RA3TCTFLR |
| `catflu_intrusion`             | B3TCTFLI | raw             | RA3TCTFLI |
| `numseries_incorrect`          | B3TNSTOT | `5 - correct`   | RA3TNSTOT |
| `backcount_error`              | B3TBKERR | raw             | RA3TBKERR |
| `sgst_normal_incorrect`        | B3TSTN   | `20 - correct`  | RA3TSTN   |
| `sgst_reverse_incorrect`       | B3TSTR   | `20 - correct`  | RA3TSTR   |
| `sgst_mixed_nonswitch_incorrect` | B3TSTXBO | `23 - correct`| RA3TSTXBO |
| `sgst_mixed_switch_incorrect`  | B3TSTXBS | `6 - correct`   | RA3TSTXBS |
| `sgst_mixed_all_incorrect`     | B3TSTXBB | `29 - correct`  | RA3TSTXBB |

### 1-4. 최종 dimension
| 데이터 | n | Y_bin | Y_cnt | Y_ord1 | Y_ord2 | Y_con |
|---|---|---|---|---|---|---|
| Wave 2 | 125 | 20 | 15 | 28 | 0 | 5 |
| Refresher 1 | 96 | 20 | 15 | 28 | 0 | 5 |
| **Combined** | **221** | 20 | 15 | 28 | 0 | 5 |

### 1-5. 경로 연결 (v5 / v6 result 모두 v4 preprocess 사용)
```diff
- source(... "MIDUS_preprocess_2_v3.R" ...)
+ source(... "MIDUS_preprocess_v4.R" ...)
- source(... "MIDUS_preprocess_refresher_v3.R" ...)
+ source(... "MIDUS_preprocess_refresher_v4.R" ...)
```
- `MIDUS_5layered_result_v5.R`의 `cnt_cognition_vars`, `cognition_vars`도 v4 컬럼명에 맞춰 교체.
- 구 result 파일(v4, `MIDUS_5layered_result.R`, `MIDUS_EDA.R`, test 파일)은 재현성을 위해 v3 preprocess 유지.

---

## 2. Per-item κⱼ Nonhierarchical LSIRM (v6)

### 2-1. 동기
v5 count layer는 **공유 kappa**로 NB2를 모델링하는데, v4 count 문항 15개는 상한·zero-inflation 정도가 이질적:

| 유형 | 예시 | 요구 κ |
|---|---|---|
| 상한 작고 zero-inflated | `wl_delayed_repetition`, `catflu_intrusion` | 큼 |
| 상한 넓고 밀도 높음 | `wl_*_omit` (0~15, 중앙값 7) | 작음 |
| 상한 넓고 long-tail | `backcount_error` (0~90) | 중간~큼 |

공유 κ는 분산 이질성을 β3 하나로 흡수할 수 없어서 latent position 추정이 왜곡됨 → **per-item kappa_j** 도입.

### 2-2. 신규 파일
- `data/my_LSIRM_5layered_nonhierarchical_v6.cpp`
- `data/my_LSIRM_5layered_nonhierarchical_cpp_v6.R` (wrapper: `lsirm_sharedpos_layer5_grm_v6_cpp`)

### 2-3. 변경점 (v5 → v6 cpp)
| 항목 | v5 | v6 |
|---|---|---|
| `log_kappa` | `double` | `vec(P3)` |
| 업데이트 | 한 번에 전체 | 항목별 MH loop (P3회) |
| prior | `N(μ, σ)` 공유 | **동일** (공유 hyperparameter, per-item 파라미터) |
| 저장 | `vec(n_save)` | `mat(n_save × P3)` |
| 우도 | `size = 1/kappa` | `size_j = 1/kappa(j)` — α3, β3, γ3, a_i, b3, κ 업데이트 모든 곳에 반영 |

### 2-4. 검증 (MIDUS v4 데이터 n=221, 짧은 MCMC 500 iter)
| 항목 | posterior mean κⱼ |
|---|---|
| `wl_immediate_omit` (dense)          | 0.47 |
| `wl_delayed_repetition` (zero-inflated) | 1.06 |
| `backcount_error` (long-tail)         | 1.02 |
| `sgst_mixed_nonswitch_incorrect`      | 1.22 |

→ 기대대로 문항별 dispersion 이질성이 κⱼ에 잘 흡수됨.

P3=0 case도 무문제 통과 (log_kappa가 0-length vec로 처리).

---

## 3. MIDUS_5layered_result_v6.R 재작성

v5 result 구조(nonhierarchical)를 기반으로 전면 재작성:
- v4 preprocess 두 개 + refresher 결합
- v6 nonhierarchical 모델 로드
- 기존 v6 result의 hierarchical(z+a1~a5) 블록 제거
- Traceplot helper에 **per-item κⱼ traceplot + posterior summary CSV** 추가
- Variable subsets (con/cnt/ord)는 v5와 동일한 switch 구조 유지

---

## 4. Unilayered LSIRM 비교 & 공통 함수화

### 4-1. `utils.R`에 추가된 함수
| 함수 | 용도 |
|---|---|
| `kmeans_cluster_b(b, b_layer, K_range, K, plot_dir, file_prefix, seed)` | item position matrix b에 대해 silhouette/elbow로 K 선택 → kmeans 실행 → 3종 plot + CSV |
| `binarize_by_colthreshold(X, method, strict)` | 컬럼별 mean 또는 Q1~Q4 기준 이진화 |
| `make_binarized_multilayer_for_lsirm(Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2, ...)` | 5-layer 입력을 하나의 binary matrix로 묶어 반환 (layer labels 포함) |

### 4-2. result_v6.R 파이프라인 확장
섹션 8~11 추가:

| § | 내용 | 산출 |
|---|---|---|
| 8 | Multilayered `b` → k-means | `case_plot_dir/kmeans_multilayer_{diagnostic, clusters_K*, silhouette_K*, *.csv}.pdf` |
| 9 | Dichotomize (con/cnt/ord1 → binary) + `lsirm_basic` (from `my_LSIRM_cpp.R`) | `uni_plot_dir/unilayered_result.rds` |
| 10 | Unilayered traceplots (a/b/alpha/beta/scalars) + biplot | `uni_plot_dir/uni_trace_*.pdf`, `uni_biplot.pdf` |
| 11 | Unilayered `b` → k-means + Multi vs Uni 비교 | confusion matrix, `cluster_comparison_multi_vs_uni.csv`, `biplot_compare_multi_vs_uni.pdf`, ARI (mclust 있을 때) |

### 4-3. 기본 설정
- `dichot_method <- "mean"` (다른 선택: `"Q1"`, `"Q2"`, `"Q3"`, `"Q4"`)
- `bin_method = "none"` — binary layer는 그대로 유지
- `ord_input = "raw"` — ordinal을 원 범주(1~5) 그대로 threshold
- unilayered MCMC 설정은 multilayered와 동일한 d/n_iter/burnin/thin

### 4-4. 경로 처리
`my_LSIRM_cpp.R`는 project root에 있고 내부에서 `sourceCpp("my_LSIRM.cpp")`를 상대경로로 호출하므로, 해당 source 및 `lsirm_basic()` 실행 시 일시적으로 `setwd(proj_root)` 후 복귀.

---

## 파일 변경 요약

### 신규
- `data/MIDUS_preprocess_v4.R`
- `data/MIDUS_preprocess_refresher_v4.R`
- `data/my_LSIRM_5layered_nonhierarchical_v6.cpp`
- `data/my_LSIRM_5layered_nonhierarchical_cpp_v6.R`

### 수정
- `data/MIDUS_5layered_result_v5.R` — source 경로 v4로, 변수 subset 이름 교체
- `data/MIDUS_5layered_result_v6.R` — 전면 재작성 (v4 preprocess + v6 모델 + 섹션 8~11)
- `data/utils.R` — `kmeans_cluster_b` + `binarize_*` 2개 함수 추가

### 문서
- `history/0420_history.md` — 본 문서
