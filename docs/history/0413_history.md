# 0413 History: V6 - 5-Layered Hierarchical LSIRM

## Task Summary
V5 (nonhierarchical) 모델에 hierarchical respondent latent position 구조를 추가하여 V6 모델을 구현.

## 변경 사항

### 모델 구조 변경: Nonhierarchical → Hierarchical
- **V5**: 단일 shared respondent position `a[i]` (n x d)를 5개 layer에서 공유
- **V6**: Global position `z[i]` + Layer-specific local positions `a1..a5[i]` (각 n x d)

### 수학적 구조
```
z[i,k] ~ N(0, 1)                              (global prior)
a_l[i,k] | z[i,k] ~ N(z[i,k], sigma1_sq)     (local | global, l=1,...,5)
sigma1_sq ~ InvGamma(a_sigma_global, b_sigma_global)
```
- z는 likelihood에 직접 등장하지 않음 → local position의 prior로만 작용
- sigma1_sq → 0: 모든 a_l이 z로 수렴 (V5와 동일)
- sigma1_sq → ∞: 각 layer 완전히 독립

### MCMC Sampling 변경
1. **Step 4 (respondent position)**: Joint 5-layer MH → 5개 독립 MH
   - 각 a_l: single-layer likelihood + N(z, sigma1_sq) prior
2. **새 Gibbs step 추가**:
   - z sampling (closed form): `var_z = 1/(1 + 5/sigma1_sq)`, `mean_z = var_z * sum(a_l)/sigma1_sq`
   - sigma1_sq sampling (InvGamma): `shape = a_sg + n*5*d/2`, `rate = b_sg + SSE/2`

### 참고 모델
- `legacy/my_LSIRM_mixture_3layered.cpp`: 3-layer hierarchical 구조의 z/sigma1_sq Gibbs step 참고

## 생성 파일
| 파일 | 설명 |
|------|------|
| `data/my_LSIRM_5layered_hierarchical_v6.cpp` | V6 C++ MCMC 구현 |
| `data/my_LSIRM_5layered_hierarchical_cpp_v6.R` | V6 R wrapper (init, Procrustes, 후처리) |

## 기존 V5 대비 주요 차이점
| 항목 | V5 | V6 |
|------|----|----|
| Respondent position | `a` (n x d, shared) | `z` (global) + `a1..a5` (local) |
| Position prior | `a ~ N(0, I)` | `a_l ~ N(z, sigma1_sq * I)` |
| 새 파라미터 | - | `z`, `sigma1_sq` |
| Hyperparameter | - | `a_sigma_global`, `b_sigma_global` |
| Proposal SD | `a` | `a1, a2, a3, a4, a5` |
| Acceptance tracking | `acc_a` | `acc_a1..acc_a5` |
| Procrustes stack | `[a; b1..b5]` | `[z; a1..a5; b1..b5]` |

## 유지된 V5 기능
- 5 layers: binary, continuous (robust t-dist), count (NB), ordinal1 (GRM), ordinal2 (GRM)
- Robust continuous layer (lambda2 latent scales, nu2 df)
- GRM ordinal thresholds (descending order constraint)
- Layer-specific gamma parameters
- fix_gamma 옵션

---

## Task 2: MIDUS 데이터 적용 스크립트 (V6)

### 생성 파일
| 파일 | 설명 |
|------|------|
| `data/MIDUS_5layered_result_v6.R` | V6 모델의 MIDUS 데이터 적용 스크립트 |

### V5 result 스크립트 대비 변경 사항
- **모델 소스**: `my_LSIRM_5layered_nonhierarchical_cpp_v5.R` → `my_LSIRM_5layered_hierarchical_cpp_v6.R`
- **함수 호출**: `lsirm_sharedpos_layer5_grm_cpp()` → `lsirm_hierarchical_layer5_grm_cpp()`
- **Hyperparameters**: `a_sigma_global=1, b_sigma_global=0.5` 추가
- **Proposal SD**: `a=0.3` → `a1=0.3, a2=0.3, a3=0.3, a4=0.3, a5=0.3`
- **Traceplot 추가**:
  - `z` (global position) traceplot
  - `a1..a5` (layer-specific local position) traceplots
  - `sigma1_sq` (global-local coupling) traceplot
- **Biplot 변경**: `A_hat = apply(res$a, ...)` → `Z_hat = apply(res$z, ...)` (global position 사용)
- **3D Biplot**: 동일하게 z (global) 기준으로 변경
- **Prefix**: `v5_` → `v6_`
- **Result 저장**: `{case}_result.rds` → `{case}_v6_result.rds`

---

## Task 3: LaTeX 모델 설명 문서 (V6)

### 생성 파일
| 파일 | 설명 |
|------|------|
| `paper/model_v6.tex` | V6 hierarchical 5-layered LSIRM 모델 설명 LaTeX 문서 |

### 문서 구조
1. **Notation**: 모든 파라미터의 차원, 인덱스, 정의 명시
   - 새 notation: `z_i \in \mathbb{R}^d` (global), `a_i^{(l)} \in \mathbb{R}^d` (local), `\sigma_1^2` (coupling variance)
2. **Hierarchical Respondent Position Structure**: 수식 (1)-(3)
   - `z_i ~ N(0, I_d)`, `a_i^{(l)} | z_i ~ N(z_i, sigma1^2 I_d)`, `sigma1^2 ~ InvGamma`
3. **Data & Likelihood**: 5개 layer별 likelihood (V5와 동일하되 `a_i` → `a_i^{(l)}`)
4. **Prior Distributions**: 모든 파라미터의 prior 명시
5. **MCMC Algorithm**: Steps 0-10
   - Step 4: 5개 독립 MH (layer별 local position)
   - Step 9 (NEW): z Gibbs (closed-form Normal) + sigma1_sq Gibbs (conjugate InvGamma) with 유도 포함
6. **Procrustes Matching**: Joint stacked matrix `[z; a1..a5; b1..b5]`
7. **Remarks on Hierarchical Extension**: V5 → V6 변경 요약

### V5 model_v2.tex 대비 변경 사항
- Hierarchical structure section 신규 추가
- 모든 linear predictor에서 `a_i` → `a_i^{(l)}` (layer-specific)
- MCMC Step 4: joint MH → 5개 독립 MH + hierarchical prior
- MCMC Step 9: z, sigma1_sq Gibbs sampling 신규 추가 (full derivation 포함)
- Procrustes stack: `[a; b1..b5]` → `[z; a1..a5; b1..b5]`

---

## Task 4: V6 Simulation 코드

### 생성 파일
| 파일 | 설명 |
|------|------|
| `simulation_5_layered_v6.R` | V6 5-layered hierarchical LSIRM 시뮬레이션 스크립트 |

### V5 simulation (4-layered) 대비 주요 변경 사항
| 항목 | V5 (4-layered) | V6 (5-layered hierarchical) |
|------|----------------|---------------------------|
| Layers | 4 (bin, con, cnt, ord) | 5 (bin, con, cnt, ord1, ord2) |
| Respondent position | Shared `A_true` (n x d) | Global `Z_true` + Local `A1..A5_true` (hierarchical) |
| Position generation | `A ~ cluster` | `Z ~ cluster`, `A_l | Z ~ N(Z, sigma1_sq * I)` |
| Alpha | Single shared `alpha_true` | Layer-specific `alpha1..alpha5_true` |
| Ordinal model | `u/delta` parameterization (1 layer) | GRM descending thresholds (2 layers, K1=5, K2=4) |
| Distance computation | `dist(A, B_l)` 공통 | `dist(A_l, B_l)` layer별 local position 사용 |
| New true parameter | - | `sigma1_sq_true = 0.3` |
| Model function | `lsirm_sharedpos_layer4_robust_cpp` | `lsirm_hierarchical_layer5_grm_cpp` |

### Simulation 구조 (23 sections)
1. Settings (n=150, P1-P5, d=2, K1=5, K2=4)
2. Cluster centers (respondent 3-cluster, item 5-layer 각각)
3. Position generation (global z → local a1-a5 with sigma1_sq)
4. True intercepts (alpha1-5, beta1-3)
5. GRM thresholds (beta4: P4 x 4, beta5: P5 x 3, descending order)
6. Distances & linear predictors (layer-specific a_l)
7. Data generation (binary, robust-t continuous, NB count, GRM ordinal x 2)
8. TRUE position plots (global + local 6-panel)
9. Model fitting (V6 hierarchical)
10. Acceptance rates
11-16. Traceplots (z, a1-5, b1-5, alpha1-5, beta1-3, beta4/5 thresholds, scalars, lambda2)
17. Boxplots with true values
18. Posterior mean positions
19. TRUE vs estimated position comparison (12-panel: 6 positions x 2)
20. Distance recovery per layer (5 layers)
21. Global-local distance histograms (TRUE vs estimated, density overlay)
22. Position recovery correlation
23. sigma1_sq recovery summary

### 추가된 V6-specific 진단
- `sigma1_sq` traceplot + recovery (true vs posterior)
- Global-local distance: TRUE vs estimated 비교 histogram/density
- Density plot에 이론적 기대값 `E[||z-a||] = sqrt(2*sigma1_sq)` 표시
- 6-panel position plot (z + a1-a5): TRUE vs estimated side-by-side

---

## Task 5: Prior Hyperparameter 조정 (sigma0_sq, sigma1_sq)

### 목적
Continuous layer의 latent position distance가 지나치게 커지는 문제를 prior를 통해 제어.

### 문제 진단
- 기존 InvGamma(1, b) prior는 mean이 존재하지 않고 (a=1일 때 mean=∞) heavy tail을 가짐
- sigma0_sq, sigma1_sq가 큰 값을 취할 확률이 높아 position이 과도하게 퍼짐

### 변경 내용 (`data/MIDUS_5layered_result_v6.R`)

| 파라미터 | 의미 | 변경 전 | 변경 후 |
|----------|------|---------|---------|
| `sigma0_sq` | Continuous layer 잔차분산 | InvGamma(1, 1): mode=0.5, mean=∞ | InvGamma(3, 2): mode=0.5, mean=1.0 |
| `sigma1_sq` | Global-local coupling 분산 | InvGamma(1, 0.5): mode=0.25, mean=∞ | InvGamma(3, 1): mode=0.25, mean=0.5 |

### Prior 변경 근거
- **Mode 유지**: 변경 전후 mode가 동일하여 posterior peak 위치는 비슷
- **Tail 제어**: a=1 → a=3으로 변경하여 mean이 유한해지고 (b/(a-1)) heavy tail이 크게 줄어듦
- InvGamma(a,b) 성질: mode = b/(a+1), mean = b/(a-1) (a>1일 때만 존재)

### 적용된 코드 변경
```r
# Before
hyper = list(..., a_sigma0=1, b_sigma0=1, ..., a_sigma_global=1, b_sigma_global=0.5)

# After
hyper = list(..., a_sigma0=3, b_sigma0=2, ..., a_sigma_global=3, b_sigma_global=1)
```

### 참고
- R wrapper 기본값 (`my_LSIRM_5layered_hierarchical_cpp_v6.R`)은 변경하지 않음 (MIDUS 스크립트에서만 적용)
- Item position prior `b ~ N(0, I)`는 C++ 하드코딩이므로 변경하지 않음
