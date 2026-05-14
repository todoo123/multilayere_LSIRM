# 2026-04-05 작업 이력

## Task 1: 모델 명세 LaTeX 문서 생성

기존 v3 코드 기반으로 5-layered nonhierarchical LSIRM의 모델 명세를 LaTeX 문서로 작성.

### 생성 파일

- `model.tex` — Notation, Likelihood (5 layers), Prior Distributions, MCMC Algorithm (9 steps) 포함

### 내용

| Section | 설명 |
|---------|------|
| Notation | 인덱스($i,j,l,k$), 차원($n, P_l, d, K_1, K_2$), 파라미터 정의 |
| Likelihood | L1(Bernoulli-Logit), L2(Normal), L3(NegBin), L4/L5(Cumulative Logit) |
| Priors | $\alpha, \beta, a_i, b_j, \log\gamma, \log\kappa, \sigma^2, \delta$ |
| MCMC | Step 1~9 순차 기술, MH vs Gibbs 구분, algorithm2e 환경 사용 |


## Task 2: Robust Continuous Layer 구현 (v4)

Layer 2 (Continuous)를 Normal에서 Student-$t$ distribution으로 변경하여 outlier에 robust한 모델 구현.
Normal--Gamma mixture representation을 사용: $\lambda_{ij}^{(2)} \sim \text{Gamma}(\nu_2/2, \nu_2/2)$를 도입하여 $Y_{ij}^{(2)} \mid \lambda_{ij}^{(2)} \sim N(\mu_{ij}^{(2)}, \sigma_0^2 / \lambda_{ij}^{(2)})$.

### 생성 파일

1. **`data/my_LSIRM_5layered_nonhierarchical_v4.cpp`**
2. **`data/my_LSIRM_5layered_nonhierarchical_cpp_v4.R`**

### v3 대비 변경 사항

#### C++ (`v4.cpp`)

- 함수명: `run_lsirm_cpp` → `run_lsirm_v4_cpp`
- 새 인자: `double nu2` (자유도, 고정값으로 전달)
- 새 변수: `mat lambda2(n, P2)` — observation-specific latent scale 변수
- **Step 0 추가 (Gibbs)**: 매 iteration 시작 시 $\lambda_{ij}^{(2)} \sim \text{Gamma}((\nu_2+1)/2, (\nu_2 + e_{ij}^2/\sigma_0^2)/2)$ 샘플링
- **Layer 2 likelihood 변경**: 모든 Layer 2 관련 MH step에서 `sqrt(sigma0_sq)` → `sqrt(sigma0_sq / lambda2(i,j))`
  - Alpha 2 update
  - Beta 2 update
  - Gamma 2 update
  - Shared position $a_i$ update (L2 부분)
  - Item position $b_2$ update
- **sigma0_sq Gibbs update 변경**: `SSE` → `wSSE` (weighted sum of squares: $\sum \lambda_{ij} \cdot e_{ij}^2$)
- **Storage 추가**: `store_lambda2_mean` — iteration별 $\lambda$ 평균값 (진단용)
- **Return 추가**: `lambda2_mean` 벡터 반환

#### R wrapper (`cpp_v4.R`)

- `sourceCpp` 경로: `v3.cpp` → `v4.cpp`
- 함수명: `lsirm_sharedpos_layer5_lsgrm_cpp` → `lsirm_sharedpos_layer5_robust_cpp`
- 새 인자: `nu2 = 5` (default 자유도 5)
- C++ 함수 호출: `run_lsirm_cpp(...)` → `run_lsirm_v4_cpp(..., nu2)`
- `res$info`에 `nu2` 추가


## Task 3: Result 스크립트 v4 생성

`data/MIDUS_5layered_result.R`를 v4 robust 모델에 맞게 수정하여 `data/MIDUS_5layered_result_v4.R` 생성.

### 생성 파일

- **`data/MIDUS_5layered_result_v4.R`**

### 기존 `MIDUS_5layered_result.R` 대비 변경 사항

| 항목 | v3 (기존) | v4 (변경) |
|------|-----------|-----------|
| source | `my_LSIRM_5layered_nonhierarchical_cpp.R` | `my_LSIRM_5layered_nonhierarchical_cpp_v4.R` |
| 함수 호출 | `lsirm_sharedpos_layer5_lsgrm_cpp(...)` | `lsirm_sharedpos_layer5_robust_cpp(..., nu2=nu2)` |
| 새 변수 | — | `nu2 <- 5` (자유도 설정) |
| trace extra | `par(mfrow=c(6,2))` | `par(mfrow=c(7,2))`, `lambda2_mean` traceplot 추가 |
| prefix | `M2_ALL` | `M2_ALL_v4` |
| biplot 파일명 | `M2_ALL_biplot.pdf` | `M2_ALL_v4_biplot.pdf` |
| biplot 제목 | `MIDUS_2: P1-P3-P4 (All)` | `MIDUS_2: P1-P3-P4 (Robust v4)` |


## Task 4: Lambda2 per-edge 진단 기능 추가

Robust Layer 2의 latent scale 변수 $\lambda_{ij}^{(2)}$를 edge 단위로 저장·리포팅할 수 있도록 변경.

### 수정 파일

#### C++ (`v4.cpp`)

- **Storage 추가**: `cube store_lambda2(n, P2, n_save)` — 매 저장 iteration마다 전체 $\lambda$ 행렬 스냅샷
- **Accumulator 추가**: `mat sum_lambda2(n, P2)` — posterior mean 계산용 running sum
- **Save section**: `store_lambda2.slice(save_idx) = lambda2` 및 `sum_lambda2 += lambda2`
- **Return 추가**: `lambda2` (full cube), `lambda2_postmean` ($= \text{sum\_lambda2} / n_{\text{save}}$, n×P2 행렬)

#### R wrapper (`cpp_v4.R`)

- `aperm(res$lambda2, c(3,1,2))` 추가 — C++ cube (n×P2×n_save) → R array (n_save×n×P2)

#### Result (`MIDUS_5layered_result_v4.R`)

`make_traceplots` 함수에 3개 진단 플롯 추가:

1. **Per-edge traceplot** (`_trace_lambda2_edges.pdf`): 무작위 12개 (i,j) edge의 $\lambda_{ij}^{(2)}$ traceplot, $\lambda=1$ 기준선 표시
2. **Heatmap** (`_lambda2_postmean_heatmap.pdf`): 전체 (n×P2) posterior mean heatmap, 낮은 $\lambda$ (빨강) = 잠재적 outlier
3. **Per-item boxplot** (`_lambda2_postmean_boxplot.pdf`): item별 $\lambda$ 분포 boxplot, $\lambda=1$ 기준선
4. **콘솔 출력**: overall mean, 5-number summary, $\lambda < 0.5$ 인 edge 수/비율


## Task 5: 4-layered 모델에 Robust Continuous Layer 이식 (v5)

5-layered v4의 robust regression (Student-$t$ via Normal-Gamma mixture)을 4-layered v4 모델에 이식.
4-layered 모델은 shared alpha (layer 간 공유), 단일 sigma_alpha_sq 구조를 유지.

### 생성 파일

1. **`my_LSIRM_4layered_nonhierarchical_v5.cpp`**
2. **`my_LSIRM_4layered_nonhierarchical_cpp_v5.R`**

### 기존 4-layered v4 대비 변경 사항

#### C++ (`v5.cpp`)

- 함수명: `run_lsirm_cpp` → `run_lsirm_v5_cpp`
- 새 인자: `double nu2` (자유도, 고정값)
- 새 변수: `mat lambda2(n, P2)` — observation-specific latent scale
- **Step 0 추가 (Gibbs)**: $\lambda_{ij}^{(2)} \sim \text{Gamma}((\nu_2+1)/2, (\nu_2 + e_{ij}^2/\sigma_0^2)/2)$
- **Layer 2 likelihood 변경**: alpha, beta2, gamma2, $a_i$, $b_2$ MH step에서 `sqrt(sigma0_sq)` → `sqrt(sigma0_sq / lambda2(i,j))`
- **sigma0_sq Gibbs**: SSE → weighted SSE ($\sum \lambda_{ij} \cdot e_{ij}^2$)
- **Storage**: `store_lambda2` (full cube), `store_lambda2_mean` (iteration별 평균), `sum_lambda2` (posterior mean용)
- **Return**: `lambda2`, `lambda2_mean`, `lambda2_postmean` 추가

#### R wrapper (`cpp_v5.R`)

- `sourceCpp` 경로: `v4.cpp` → `v5.cpp`
- 함수명: `lsirm_sharedpos_layer4_lsgrm_cpp` → `lsirm_sharedpos_layer4_robust_cpp`
- 새 인자: `nu2 = 5`
- C++ 호출: `run_lsirm_cpp(...)` → `run_lsirm_v5_cpp(..., nu2)`
- `aperm(res$lambda2, c(3,1,2))` 추가
- `res$info`에 `nu2` 추가

### 4-layered vs 5-layered 구조 차이 (유지됨)

| 항목 | 4-layered (v5) | 5-layered (v4) |
|------|----------------|----------------|
| Alpha | 공유 `alpha` 1개 | `alpha1`~`alpha5` 독립 |
| sigma_alpha | `sigma_alpha_sq` 1개 | `sigma_alpha1_sq`~`sigma_alpha5_sq` |
| Layers | bin/con/cnt/ord (4개) | bin/con/cnt/ord1/ord2 (5개) |


## Task 6: Simulation 스크립트 v5 생성

`simulation_4_layered_v3.R`를 v5 robust 모델에 맞게 수정하여 `simulation_4_layered_v5.R` 생성.

### 생성 파일

- **`simulation_4_layered_v5.R`**

### v3 대비 주요 변경 사항

| 항목 | v3 | v5 |
|------|----|----|
| source | `_cpp.R` (shared gamma) / `_cpp_v4.R` (per-layer gamma) | `_cpp_v5.R` (per-layer gamma + robust) |
| 함수 호출 | `lsirm_sharedpos_layer4_lsgrm_cpp(...)` | `lsirm_sharedpos_layer4_robust_cpp(..., nu2=nu2_fit)` |
| Y_con 생성 | `ETA2 + N(0, sigma)` (Gaussian errors) | `ETA2 + N(0, sigma)/sqrt(lambda)`, `lambda ~ Gamma(nu2/2, nu2/2)` (Student-t errors) |
| 새 변수 | — | `nu2_true=5` (데이터 생성), `nu2_fit=5` (모델 피팅), `lambda_true` (true latent scales) |
| gamma traceplot | 단일 `log_gamma` | `log_gamma1`~`log_gamma4` (per-layer) |
| distance recovery | 단일 `gamma_post` | `gamma1_post`~`gamma4_post` (per-layer) |
| lambda2 진단 | — | traceplot (12 edges), heatmap, boxplot, true vs estimated scatter |
| plot prefix | `4layered_` | `4layered_v5_` |
| 하단 LSIRM 비교 | 포함 (binarized unilayer 비교) | 제거 (robust 모델 자체 진단에 집중) |
