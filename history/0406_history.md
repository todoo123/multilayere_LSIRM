# 2026-04-06 작업 이력

## Task 1: MIDUS Refresher 1 전처리 코드 생성

MIDUS Wave 2 전처리 코드(`MIDUS_preprocess_2_v3.R`)를 기반으로, Refresher 1 데이터에 맞는 전처리 코드를 신규 작성.
Excel 변수명세(`MIDUS_wave_2_변수명세.xlsx`의 `refresher_1_변수list` 시트)를 참조하여 변수명 매핑.

### 생성/수정 파일

1. **`data/MIDUS_preprocess_refresher_v3.R`** (신규)
2. **`data/MIDUS_preprocess_2_v3.R`** (수정)

### 주요 내용

| 항목 | Wave 2 | Refresher 1 |
|------|--------|-------------|
| 데이터 | ICPSR 04652, 29282, 25281 | ICPSR 36532, 36901, 37081 |
| ID 변수 | `M2ID` | `MRID` |
| Survey 변수 prefix | `B1PA*`, `B1SA*` | `RA1PA*`, `RA1SA*` |
| Biomarker 변수 prefix | `B4*` | `RA4*` |
| Cognitive 변수 prefix | `B3*` | `RA3*` |
| tobacco/alcohol | `B4H26`, `B4H33`, `B4H40` | `RA4H38`, `RA4H49`, `RA4H56` |
| age/sex | `B1PAGE_M2`, `B1PRSEX` | `RA1PRAGE`, `RA1PRSEX` |
| cortisol final avg | `B4BSCL4A` → 주석 처리 | `RA4BSCL4A` — 데이터에 없어 제외 |
| cortisol all avg | `B4BSCL14` | `RA4BSCLAV` |

- `B4BSCL4A`를 wave_2 코드에서도 주석 처리하여 두 데이터셋의 continuous 변수 구조 통일 (7개)


## Task 2: Wave 2 + Refresher 1 결합 분석 코드 수정

기존 wave_2 단독 분석이던 `MIDUS_5layered_result_v4.R`을 wave_2 + refresher_1 결합 분석으로 수정.

### 수정 파일

- **`data/MIDUS_5layered_result_v4.R`**

### 주요 변경 사항

| 항목 | 기존 | 변경 |
|------|------|------|
| 데이터 로드 | `source("MIDUS_preprocess_2_v3.R")` 단독 | wave_2, refresher_1 각각 `new.env()` 격리 환경에서 `source()` |
| 데이터 결합 | — | `combine_lsirm()` 함수로 Y 행렬 rbind (변수명은 wave_2 기준 통일) |
| 출처 추적 | — | `source` 벡터 추가 (`"wave2"` / `"refresher1"`) |
| 결합 대상 | — | `lsirm_all`, `lsirm_p4`, `lsirm_p3` 모두 결합 |
| plot prefix | `M2_ALL_v4` | `M2R1_ALL_v4` |
| biplot 제목 | `MIDUS_2: P1-P3-P4` | `MIDUS W2+R1: P1-P3-P4` |


## Task 3: Standard vs Robust LSIRM Coverage 비교 시뮬레이션

4-layered LSIRM의 standard (v4) vs robust (v5) 모델 성능을 coverage 기준으로 비교하는 시뮬레이션 스크립트 작성.

### 생성 파일

- **`simulation_4_layered_robust__comparison.R`**

### 시뮬레이션 설계

| 항목 | 내용 |
|------|------|
| 비교 모델 | v4 (standard, Gaussian continuous) vs v5 (robust, Student-t continuous, ν=5) |
| Scenario 1 | Standard: `simulation_4_layered_v5.R` 세팅 동일 (Student-t errors, ν=5) |
| Scenario 2 | Outlier: 10% respondent의 latent position을 N(0, 8) 에서 샘플 (cluster에서 극단적으로 이탈) |
| 반복 횟수 | 100회 |
| 실행 방식 | `parallel::mclapply` 병렬 실행 |
| MCMC 설정 | n_iter=50000, burnin=10000, thin=10 (4000 saved samples) |

### Coverage 평가 대상 (simulation_4_layered_v3_coverage.R 기반)

- **Parameter coverage**: alpha, beta1, beta2, beta3, u, delta
- **Distance coverage**: dist_bin (L1), dist_con (L2), dist_cnt (L3), dist_ord (L4) — per-layer gamma 반영
- **OVERALL**: 전체 parameter 통합 coverage

### 출력

- `simulation_results/robust_comparison_raw.csv`: rep별 상세 결과
- `simulation_results/robust_comparison_summary.csv`: (scenario × model × param)별 mean/sd coverage, CI width
- 콘솔: standard vs robust side-by-side 비교표


## Task 4: Ordinal layer를 GRM (Graded Response Model) 으로 교체 (v5)

기존 v4의 ordinal layer (L4, L5)에서 사용하던 cumulative link 모델 (`u + cumsum(softplus(delta))` ascending threshold) 을 lsirm12pl 패키지의 GRM 구현 방식으로 교체.

### 핵심 변경: 모델 parameterization

| 항목 | v4 (Cumulative Link) | v5 (GRM) |
|------|---------------------|----------|
| Threshold 구조 | `u + cumsum(softplus(delta))`, ascending | `beta4/beta5`, descending order |
| 파라미터 | `u` (fixed=0), `delta` (P×(K-2)) | `beta4` (P4×(K1-1)), `beta5` (P5×(K2-1)) |
| 확률 모델 | `P(Y≤k) = logistic(threshold_k - eta)` | `P(Y≥k) = logistic(eta + beta_k)` |
| Threshold 업데이트 | Joint MH (delta 전체) + Jacobian 보정 | Individual MH per threshold + ordering constraint check |
| 데이터 인코딩 | 1-based (1..K) | 내부 0-based 변환 (0..K-1) |

### 생성 파일

1. **`data/my_LSIRM_5layered_nonhierarchical_v5.cpp`** — C++ MCMC 본체
2. **`data/my_LSIRM_5layered_nonhierarchical_cpp_v5.R`** — R wrapper (Procrustes 포함)
3. **`data/MIDUS_5layered_result_v5.R`** — 결합 데이터 분석 runner + 진단 plot

### 주요 변경 상세 (C++)

- **Helper 교체**: `build_thresholds_cpp`, `log_p_ordinal_single_cpp` 제거 → `log_grm_prob_y0based` (template) 추가
- **Hyperparameter**: `mu_u/sd_u/mu_delta/sd_delta` → `mu_beta4/sd_beta4/mu_beta5/sd_beta5`
- **Proposal SD**: `u/delta/delta2` → `beta4/beta5`
- **Threshold 업데이트 (6a/6b)**: GRM 방식 — threshold별 individual MH, descending order constraint check
- **Acceptance 추적**: `acc_thr/acc_thr2` (vec) → `acc_beta4_thr/acc_beta5_thr` (mat: item × threshold)
- **Output**: `u/delta/thr/u2/delta2/thr2` → `beta4/beta5`

### 주요 변경 상세 (R wrapper)

- 함수명: `lsirm_sharedpos_layer5_grm_cpp`
- Init: `init_grm_beta()` — per-item descending 초기값 생성
- Post-process: `aperm(res$beta4, c(3,1,2))` → (n_save, P4, K1-1)

### 주요 변경 상세 (Result)

- Source: v5 파일 로드
- Hyper: `mu_beta4=0, sd_beta4=2, mu_beta5=0, sd_beta5=2`
- Traceplot: `delta/delta2` → `beta4_thr/beta5_thr` (item명 + threshold index 표시)
- Plot prefix: `M2R1_ALL_v5`


## Task 5: CESD reverse items 이진화 순서 버그 수정

### 버그 내용

CESD reverse 문항(B4Q3D, B4Q3H, B4Q3L, B4Q3P)의 이진화 순서가 잘못되어 있었음.

| 단계 | 기존 (잘못된 순서) | 수정 (올바른 순서) |
|------|-------------------|-------------------|
| 1 | 전체 이진화 (1→0, 2/3/4→1) | **reverse coding (5-x: 1↔4, 2↔3)** |
| 2 | reverse items만 flip (0↔1) | **전체 이진화 (1→0, 2/3/4→1)** |

### 영향받는 값

ICPSR 값 2, 3인 경우 결과가 반대로 코딩됨:
- 값 2: 기존 0 → 수정 후 1
- 값 3: 기존 0 → 수정 후 1
- 값 1, 4는 동일

### 수정 파일

1. **`data/MIDUS_preprocess_2_v3.R`** — Wave 2
2. **`data/MIDUS_preprocess_refresher_v3.R`** — Refresher 1


## Task 6: Tobacco/Alcohol binary 변수 주석처리

B4H26 (흡연 여부), B4H33 (음주 여부)를 제외하고 fitting 해보기 위해 주석처리.

### 주석처리 내용

| 파일 | 변수 | 내용 |
|------|------|------|
| `MIDUS_preprocess_2_v3.R` | `B4H26`, `B4H33` | `bio_binary_vars`에서 주석처리 |
| `MIDUS_preprocess_2_v3.R` | skip pattern | `B4H40 = ifelse(B4H33 == 0 & ...)` 비활성 |
| `MIDUS_preprocess_refresher_v3.R` | `RA4H38`, `RA4H49` | `bio_binary_vars`에서 주석처리 |
| `MIDUS_preprocess_refresher_v3.R` | skip pattern | `RA4H56 = ifelse(RA4H49 == 0 & ...)` 비활성 |

- `bio_binary_vars`가 비어 있게 되므로 Y_bin에서 CESD 20문항만 남음
- skip pattern 로직도 B4H33/RA4H49 참조 불가하므로 함께 주석처리


## Task 7: Cognitive 변수 확장 및 B3TCOMPZ3 제외

기존 6개 cognitive 변수를 12개로 확장하고, 하위지표와 중복되는 global composite (B3TCOMPZ3)를 제외.

### 변경 원칙

- **높을수록 인지력 나쁨** (우울 방향과 일치) 으로 통일
- cognitive 변수를 `cog_con_vars` (continuous)와 `cog_cnt_vars` (count)로 분리하여 LSIRM 5-layer 구조에 맞게 배치

### 제거/주석처리

| 변수 | 사유 |
|------|------|
| `B3TCOMPZ3` / `RA3TCOMPZ3` | 전체 인지 종합 점수 — 하위 지표(B3TEMZ3, B3TEFZ3 등)와 중복 |
| `B3TSMXNS` / `RA3TSMXNS` | B3TSMXBS (switch 통합 지표)로 대체 |
| `B3TSMXRS` / `RA3TSMXRS` | B3TSMXBS (switch 통합 지표)로 대체 |

### 추가 변수 (9개 신규)

**Continuous (RT 변수, 높을수록 느림=인지↓)**

| Wave 2 | Refresher 1 | 영역 | 설명 |
|--------|-------------|------|------|
| `B3TSMN` | `RA3TSMN` | Processing Speed | Normal single-task median RT (초) |
| `B3TSMR` | `RA3TSMR` | Processing Speed | Reverse single-task median RT (초) |
| `B3TSMXBS` | `RA3TSMXBS` | Task Switching | Mixed block all switch median RT (초) |
| `B3TSMXNO` | `RA3TSMXNO` | Task Switching | Mixed block normal non-switch median RT (초) |
| `B3TSMXRO` | `RA3TSMXRO` | Task Switching | Mixed block reverse non-switch median RT (초) |

**Count (오답/오류 수, 높을수록 인지↓)**

| Wave 2 | Refresher 1 | 영역 | 설명 | 전처리 |
|--------|-------------|------|------|--------|
| `B3TSPN` | `RA3TSPN` | EF/Speed | Normal single-task 정확도 → 오답 수 | `abs(round(x/0.05) - 20)` → 0~20 integer |
| `B3TSPR` | `RA3TSPR` | EF/Speed | Reverse single-task 정확도 → 오답 수 | `abs(round(x/0.05) - 20)` → 0~20 integer |
| `B3TCTFLR` | `RA3TCTFLR` | Verbal Fluency | Category fluency 반복 수 | `floor()` → integer |
| `B3TCTFLI` | `RA3TCTFLI` | Verbal Fluency | Category fluency 침입 오류 수 | `floor()` → integer |

### 최종 cognitive 변수 구성 (12개)

| 타입 | 변수 | 영역 |
|------|------|------|
| continuous | B3TEMZ3 (reverse) | Episodic Memory |
| continuous | B3TEFZ3 (reverse) | Executive Function |
| continuous | B3TWLF | Memory Retention |
| continuous | B3TSMN | Processing Speed |
| continuous | B3TSMR | Processing Speed |
| continuous | B3TSMXBS | Task Switching |
| continuous | B3TSMXNO | Task Switching |
| continuous | B3TSMXRO | Task Switching |
| count | B3TSPN → 오답 수 | EF/Speed |
| count | B3TSPR → 오답 수 | EF/Speed |
| count | B3TCTFLR | Verbal Fluency |
| count | B3TCTFLI | Verbal Fluency |

### split_for_lsirm 호출 변경

- `cog_con_vars` → `con_vars`에 배치
- `cog_cnt_vars` → `cnt_vars`에 배치
- P3, P1-P3-P4 모두 반영

### 수정 파일

1. **`data/MIDUS_preprocess_2_v3.R`** — Wave 2
2. **`data/MIDUS_preprocess_refresher_v3.R`** — Refresher 1
