# 0503 — v10 MIDUS 최종 운영 setting 확정 (proposal 재튜닝 + gamma prior 좁힘)

## 오늘의 작업

| # | Task | 산출 |
|---|---|---|
| 1 | v10 traceplot 흔들림 진단 (multimodal-like drift) | 비식별성 + sticky proposal 합작 진단 — 진짜 multimodality 가 아니라 random-walk 표류 |
| 2 | v9 → v10 proposal SD 비교 → 누락된 proposal 재조정 | `log_gamma`, `a`, `beta`, `b` proposal 전반 키움 |
| 3 | `log_gamma_k` prior 좁힘 (sd 0.5 → 0.4) | LSIRM 의 global scale 비식별 자유도를 prior 가 더 강하게 잡도록 |
| 4 | burnin 40k → 20k 축소 + seed 단순화 (`0502`) | warmup 의 4분의 1 (5k) 후로 FMC block 이 충분히 안정. 길게 남길 필요 X |
| 5 | 운영 setting 전체 lock-in | **현재 [data/MIDUS_5layered_result_v10.R](../data/MIDUS_5layered_result_v10.R) 가 MIDUS v10 의 reference setting** |

**결정 요지**: 0502 의 `r=5 / e0=0.10 / S0=0.01` operating point 는 유지하면서, sampler 거동 (proposal SD + gamma prior) 만 재튜닝. cluster 결과의 정량적 metric (silhouette, PEAR, within-pair ratio) 가 본질적으로 약한 데이터 신호를 반영한다는 점이 0502 와 동일하게 확인됨 — 데이터 자체의 fact 로 받아들이고 sampler 안정성에 집중.

---

# Task 1. Trace plot 흔들림 진단 결과

## 1.1 증상
- LSIRM 의 element-level trace (`a`, `b_k`, `gamma_k`) 가 multimodal 처럼 보임
- acceptance rate 은 정상 범위 (대부분 0.3 – 0.5)
- v9 에서는 이 정도 표류가 없었음

## 1.2 원인 분리 (3 갈래)
1. **회전·반사·평행이동 비식별성**: LSIRM likelihood 가 `||a_i − b_j||` 에만 의존 — element-level coordinate 는 본질적으로 추정 불가능. 사후 Procrustes 가 잡아주지만 chain 내부에서는 자유.
2. **PPCA 회전 비식별성**: `D = Λ·η + δ + ε` 가 `(Λ Q, Qᵀ η)` 에 invariant. `mu`, `eta`, `Lambda` element-level trace 는 정착 못함.
3. **잉여 컴포넌트 잡음**: K* = 10 vs K_+ ≈ 3 – 5 → 빈 컴포넌트의 `rho_l`, `mu_l` 은 prior 에서 거의 자유 추출.

## 1.3 식별 가능한 양만 보면 multimodality 없음 (확인 항목)
- pairwise distance `||a_i − b_j||` (회전 invariant)
- co-cluster matrix entry (label invariant)
- `K_+`, `sigma0_sq`, `gamma_k`, `sigma_eps_sq` (스칼라)
- `log|Sigma_l|` (회전 invariant)

> 결론: **비식별성으로 인한 표류이지 진짜 multimodality 가 아님**. element-level 해석 대신 식별 가능한 양 + co-cluster matrix 로 결과 보고.

---

# Task 2. v9 → v10 proposal SD 재튜닝 (현재 setting)

## 2.1 v9 vs 이전 v10 vs 현재 v10 비교

| 파라미터 | v9 | v10 (이전) | v10 (현재 0503) | 변화 |
|---|---|---|---|---|
| alpha1 | 0.6 | 0.78 | **0.78** | 유지 |
| alpha2 | 0.4 | 0.4 | **0.4** | 유지 |
| alpha3 | 0.5 | 0.5 | **0.5** | 유지 |
| alpha4 | 0.5 | 0.60 | **0.60** | 유지 |
| alpha5 | 0.6 | 0.6 | **0.6** | 유지 |
| log_gamma1 | 0.10 | 0.07 | **0.10** | v9 복귀 |
| log_gamma2 | 0.05 | 0.055 | **0.15** | ↑ 3× |
| log_gamma3 | 0.05 | 0.06 | **0.15** | ↑ 2.5× |
| log_gamma4 | 0.05 | 0.04 | **0.05** | v9 복귀 |
| log_gamma5 | 0.20 | 0.20 | **0.20** | 유지 |
| a | 0.30 | 0.35 | **0.5** | ↑ |
| beta1 | 0.50 | 0.50 | **0.50** | 유지 |
| beta2 | 0.1 | 0.1 | **0.4** | ↑ 4× |
| beta3 | 0.20 | 0.20 | **0.20** | 유지 |
| beta4 | 0.30 | 0.30 | **0.50** | ↑ |
| beta5 | 0.30 | 0.30 | **0.30** | 유지 |
| b1 | 0.35 | 0.35 | **0.5** | ↑ |
| b2 | 0.20 | 0.24 | **0.4** | ↑ |
| b3 | 0.20 | 0.32 | **0.5** | ↑ |
| b4 | 0.10 | 0.165 | **0.4** | ↑ 2.4× |
| b5 | 0.50 | 0.50 | **0.50** | 유지 |
| log_kappa | 0.30 | 0.30 | **0.30** | 유지 |

## 2.2 핵심 변경 의도
- **gamma 의 mixing 회복**: 이전 v10 에서 `log_gamma1`, `log_gamma4` proposal 을 줄였더니 acceptance 는 좋아졌지만 chain 이 sticky 해짐 → trace 가 표류처럼 보임. v9 수준 또는 그 이상으로 키워서 한 step 당 거리를 늘림.
- **`b_k` 전반 확장**: b 가 LSIRM 의 item position 인데 좁은 proposal 로 매 iteration 짧게만 움직이면 아래 PPCA 입력 인 `D = log||a_i − b_j||` 가 천천히 변하고, eta 도 천천히 변함. 결국 cluster 가 lock-in 되지 못함.
- **acceptance 0.20 – 0.30 까지 떨어지는 것 허용** (Roberts–Rosenthal 0.234 result; global scale 류는 0.15 – 0.25 가 ESS 더 좋음).

---

# Task 3. `log_gamma_k` prior 좁힘 (sd 0.5 → 0.4)

## 3.1 이론적 근거
- LSIRM 의 global scale 자유도: 5 layer 가 `(a, b)` 를 공유하므로 **gamma_k 의 상대비는 data 가 식별** 하지만, **전체 scale 한 자유도는 prior 가 잡아야** 함.
- 이전 prior `N(0, 0.5²)` 는 gamma 95% CI ≈ `[0.37, 2.7]` — chain 이 이 범위를 자유롭게 표류 가능.
- 새 prior `N(0, 0.4²)` 는 95% CI ≈ `[0.45, 2.2]` — 약 25% 좁아짐. 데이터 신호를 죽이지 않으면서 표류 범위만 축소.

## 3.2 왜 0.4 인가 (0.2 가 아닌)
- 0.2 까지 좁히면 (95% CI `[0.67, 1.49]`) MIDUS 처럼 layer 간 신호 강도가 다른 (특히 ord1 / con) 데이터에서 prior pile-up 위험.
- 0.4 는 boundary case — 데이터가 prior 와 양립하지 않을 때 posterior 가 prior boundary 까지 가더라도 신호가 살아있음을 확인할 수 있는 수준.
- 추후 v9 와 비교 시 `log_gamma_k` posterior median 이 1 에서 크게 벗어나지 않으면 0.3 까지 더 좁혀볼 여지 있음.

---

# Task 4. burnin · seed · iter 정책

## 4.1 변경
- `n_iter`: 120000 (v9) → **200000** (v10 유지)
- `burnin`: 40000 (이전 v10) → **20000** (현재)
- `thin`: 10 (변경 없음)
- `nu2`: 4 (변경 없음)
- `set.seed`: `20260502` → **`0502`** (단순화)
- saved iterations: **18000**

## 4.2 burnin 단축의 근거
- `fmc_warmup = max(1000, burnin/4) = 5000` — FMC block 이 LSIRM warmup 후 5k iter 으로 진입.
- 0502 sweep 결과 K_+ 가 burnin 의 절반 시점에 이미 stationary. 20k 충분.
- saved 18k 가 PEAR / silhouette MC 추정에 충분한 sample size.

---

# Task 5. 최종 운영 hyperparameter (lock-in)

## 5.1 LSIRM block

```r
common_lsirm_hyper <- list(
  a_sigma = 1, b_sigma = 1,
  a_tau1 = 1, b_tau1 = 1, a_tau2 = 1, b_tau2 = 1, a_tau3 = 1, b_tau3 = 1,
  a_sigma0 = 1, b_sigma0 = 1,
  mu_log_gamma1 = 0, sd_log_gamma1 = 0.4,   # 0.5 → 0.4
  mu_log_gamma2 = 0, sd_log_gamma2 = 0.4,
  mu_log_gamma3 = 0, sd_log_gamma3 = 0.4,
  mu_log_gamma4 = 0, sd_log_gamma4 = 0.4,
  mu_log_gamma5 = 0, sd_log_gamma5 = 0.4,
  mu_log_kappa = 0, sd_log_kappa = 0.1,
  mu_beta4 = 0, sd_beta4 = 2,
  mu_beta5 = 0, sd_beta5 = 2
)

common_lsirm_prop_sd <- list(
  alpha1 = 0.78, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.60, alpha5 = 0.6,
  log_gamma1 = 0.10, log_gamma2 = 0.15, log_gamma3 = 0.15,
  log_gamma4 = 0.05, log_gamma5 = 0.20,
  a = 0.5,
  beta1 = 0.50, beta2 = 0.4, beta3 = 0.20, beta4 = 0.50, beta5 = 0.30,
  b1 = 0.5, b2 = 0.4, b3 = 0.5, b4 = 0.4, b5 = 0.50,
  log_kappa = 0.30
)
```

## 5.2 FMC block

```r
r_fac  <- 5L
K_star <- 10L
e0     <- 0.1
n_split_merge <- 5L

common_fmc_hyper <- list(
  e0            = e0,
  m0            = rep(0, r_fac),
  kappa0        = 1e-3,
  nu0           = r_fac + 10,        # = 15
  S0            = 0.01 * diag(r_fac),
  tau_lambda_sq = 4.0,
  a_eps         = 5, b_eps = 0.1,
  a_delta       = 2, b_delta = 1
)
```

## 5.3 MCMC

```r
common_mcmc <- list(d = 2, n_iter = 200000, burnin = 20000, thin = 10)
nu2 <- 4
set.seed(0502)
```

## 5.4 init
- `eta_init`: PCA on row-stacked Y blocks (자세한 내용 [data/MIDUS_5layered_result_v10.R:280-294](../data/MIDUS_5layered_result_v10.R#L280-L294))
- `c_init`: k-means(K* = 10) on PCA scores
- `Lambda_init`: random N(0, 0.5²)
- `lsirm_init`: NULL (C++ 내부 random)

## 5.5 진단 절 (코드에 포함됨)
- §4-G: `S0` elicitation diagnostic (`prior_implied_sd / empirical_within_sd`, ratio > 5 → tighten 권장)
- §4-H: η-space silhouette (Kaufman & Rousseeuw 1990 thresholds)
- §4-H-bis: within-pair ratio + PEAR (Fritsch & Ickstadt 2009)
- §4-E-bis: minBinder / minVI alternative partitions (P0 vs P1 vs P3 vs P4)
- §4-F: posterior summaries (co-cluster CSV, eta_postmean CSV, Lambda_postmean CSV)

---

# 결과 메트릭의 해석 (변경 없음, 0502 의 진단과 일관)

| Metric | 값 (0502 best run) | 해석 |
|---|---|---|
| silhouette mean | 0.151 | weak (Kaufman–Rousseeuw < 0.25) |
| PEAR | 0.220 | weak (posterior diffuse) |
| within-pair ratio | 0.094 | weak (< 0.3) |
| Pr(co > 0.8) | 3.9% | 낮음 |
| Pr(co [0.3, 0.7]) | 50% | 높은 ambiguous zone |
| 9-item 신경염증+집행기능 sub-cluster silhouette | +0.30 | reasonable structure (강한 발견) |

→ **MIDUS 데이터는 본질적으로 fuzzy partition** 을 가지며, 부분적으로 robust 한 9-item cluster (염증 5 + 집행기능 4) 가 가장 신뢰할 수 있는 발견. 보고는 partition 점추정 대신 **co-cluster matrix + sub-cluster silhouette** 로 진행.

---

# 다음 단계 (필요 시)

1. **multi-chain run** (4 chains, 다른 seed) — partition mode hopping 진단. PEAR 가 chain 간 동일 mode 면 sampler OK, 다른 mode 면 tier-2 처방 고려.
2. **MDS-based LSIRM init** — chain 별 layout 일관성 확보.
3. (옵션) `log_gamma` prior 0.4 → 0.3 추가 좁힘 — posterior 가 prior 와 충돌 안 하면 안전.
