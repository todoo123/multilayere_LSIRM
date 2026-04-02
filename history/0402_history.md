# 2026-04-02 작업 이력

## Task 1: 파일 이름 변경 (EDA → preprocess)

기존 EDA 파일들이 실제로는 전처리(preprocessing) 역할을 하고 있어 이름을 변경함.

- `data/MIDUS_EDA_2.R` → `data/MIDUS_preprocess_2.R`
- `data/MIDUS_EDA_2_v2.R` → `data/MIDUS_preprocess_2_v2.R`
- `data/MIDUS_EDA_2_v3.R` → `data/MIDUS_preprocess_2_v3.R`

### 참조 파일 업데이트

이름 변경에 따라 `source()` 호출부 수정:

- `data/MIDUS_5layered_result.R` (line 7)
- `data/MIDUS_5layered_result_v4.R` (line 7)


## Task 2: MIDUS_EDA.R 신규 생성

`data/MIDUS_EDA.R` — raw data 특성 탐색을 위한 EDA 스크립트.

### 분석 내용

`MIDUS_preprocess_2_v3.R`의 `lsirm_all` (P1-P3-P4 통합, screen_case==1) 5개 layer matrix 사용:

| Layer | 설명 |
|-------|------|
| Y_bin | Binary (CESD + Biomarker) |
| Y_cnt | Count |
| Y_ord1 | Ordinal-5 (MASQ) |
| Y_ord2 | Ordinal-4 (PSQI) |
| Y_con | Continuous |

### 계산

각 layer별로:
- `alpha_i` = i번째 응답자의 행 평균 (row mean)
- `beta_j` = j번째 문항의 열 평균 (column mean)
- Residual = `Y_{ij} - (alpha_i + beta_j)`

### 시각화

1. **alpha_i bar plot** — 5개 layer를 facet으로 응답자별 bar plot → `data/plot/MIDUS_EDA_alpha_barplots.pdf`
2. **Residual histogram** — 5개 layer별 density histogram → `data/plot/MIDUS_EDA_residual_histograms.pdf`


## Task 3: 저장 경로 수정

- MIDUS_2 폴더가 `.gitignore` 대상이므로, 작업 디렉토리를 `data/`로 유지
- PDF 저장 경로를 `data/plot/`으로 지정


## Task 4: alpha를 layer별로 분리 추정

기존: 하나의 공유 `alpha` 벡터 (length n) 가 5개 layer에서 공유됨
변경: `alpha1`~`alpha5` 각각 독립적으로 추정 (layer-specific respondent effect)

### 수정 파일

1. **`data/my_LSIRM_5layered_nonhierarchical_v3.cpp`**
   - `vec alpha` → `vec alpha1, alpha2, alpha3, alpha4, alpha5`
   - Alpha update: 기존 하나의 MH block (모든 layer의 likelihood 합산) → 5개 독립 MH block (각 layer의 likelihood만 사용)
   - Prior/proposal: 동일한 형태 유지 (`N(0, sigma_alpha_sq)`, proposal sd = `sd_alpha`)
   - `sigma_alpha_sq` Gibbs update: 5개 alpha의 sum of squares를 pooling하여 업데이트
   - Beta, gamma, position(a_i), item position(b_j), threshold, kappa, sigma0 업데이트에서 `alpha(i)` → 해당 layer의 `alphaL(i)`로 변경
   - Storage/return: `store_alpha` → `store_alpha1`~`store_alpha5`, acceptance rate도 동일하게 분리

2. **`data/my_LSIRM_5layered_nonhierarchical_cpp.R`**
   - Init: `alpha = rnorm(n)` → `alpha1`~`alpha5` 각각 `rnorm(n, 0, 0.1)`


## Task 5: alpha별 proposal SD 분리 + result 파일 업데이트

### prop_sd 분리

기존: `prop_sd$alpha` 하나로 5개 alpha의 proposal SD를 공유
변경: `prop_sd$alpha1`~`prop_sd$alpha5` 각각 독립 지정 가능

### 수정 파일

1. **`data/my_LSIRM_5layered_nonhierarchical_v3.cpp`**
   - `sd_alpha = prop_sd["alpha"]` → `sd_alpha1`~`sd_alpha5` 각각 unpack
   - 각 alpha MH block에서 해당 layer의 `sd_alphaL` 사용

2. **`data/my_LSIRM_5layered_nonhierarchical_cpp.R`**
   - `prop_sd` default: `alpha=0.1` → `alpha1=0.1, ..., alpha5=0.1`

3. **`data/MIDUS_5layered_result.R`**
   - `common_prop_sd`: `alpha=0.5` → `alpha1=0.5, ..., alpha5=0.5`
   - Alpha traceplot: `res$samples$alpha` 단일 → `res$samples$alpha1`~`alpha5` 루프로 각각 PDF 생성


## Task 6: sigma_alpha_sq를 layer별로 분리 (독립 Gibbs sampling)

기존: pooled `sigma_alpha_sq` 하나로 5개 alpha의 prior variance를 공유
문제: alpha를 layer별로 독립 추정하면서 prior variance는 pooling하면 각 layer의 scale 차이를 반영하지 못함
변경: `sigma_alpha1_sq`~`sigma_alpha5_sq` 각각 독립 inverse-gamma Gibbs sampling

### 구조

- Prior: `alpha_l_i ~ N(0, sigma_alpha_l_sq)`
- Hyperprior: `sigma_alpha_l_sq ~ InvGamma(a_sigma, b_sigma)` (l = 1,...,5)
- Gibbs update: `sigma_alpha_l_sq | alpha_l ~ InvGamma(a_sigma + n/2, b_sigma + 0.5 * sum(alpha_l^2))`

### 수정 파일

1. **`data/my_LSIRM_5layered_nonhierarchical_v3.cpp`**
   - Init: `sigma_alpha_sq` → `sigma_alpha1_sq`~`sigma_alpha5_sq`
   - Storage: `store_sigma_alpha_sq` → `store_sigma_alpha1_sq`~`store_sigma_alpha5_sq`
   - Alpha prior: 각 MH block에서 해당 layer의 `sigma_alphaL_sq` 사용
   - Gibbs update: pooled 단일 draw → 5개 독립 inverse-gamma draw
   - Return: 5개 개별 반환

2. **`data/my_LSIRM_5layered_nonhierarchical_cpp.R`**
   - Init: `sigma_alpha_sq = 1` → `sigma_alpha1_sq = 1, ..., sigma_alpha5_sq = 1`

3. **`data/MIDUS_5layered_result.R`**
   - Traceplot: `sigma_alpha_sq` 단일 → 5개 루프로 개별 traceplot 생성
