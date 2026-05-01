# 0501 — v9 LSIRM `tau_beta^2` prior loosening (1.0 → 4.0)

## 오늘의 작업 (요약)

| # | 범주 | 산출 |
|---|---|---|
| 1 | 진단 | v4 ~ v9 LSIRM hyperparameter setting 비교 — v6 부터 안정화, v7 = v8 = v9 (LSIRM 블록) |
| 2 | 진단 | v9 prior 코드 검증 — `a` (latent position), `b1/b2/b3` (item position) 의 prior 가 cpp 에 하드코딩됨, R 쪽에서 노출 안 됨 |
| 3 | 진단 | alpha = random effect (제대로 구현), beta = fixed effect (구현은 fixed 이지만 `tau_beta_l_sq = 1.0` 으로 너무 타이트) 발견 |
| 4 | 문헌 | Jeon et al. (LSIRM 원논문) prior 컨벤션 확인 — 기본값 $\tau_\beta^2 = 4$, 극단 endorsement item 있을 때만 1 로 죈다고 명시 |
| 5 | 진단 | MIDUS binary endorsement 분포 — 범위 [0.20, 0.77], 극단 item 0개. 논문이 1 로 죄는 조건에 해당 안 됨 |
| 6 | 진단 | 저장된 v9 chain 의 beta posterior 검사 — beta1/beta3 에서 prior bound binding 확인 |
| 7 | 수정 | `data/my_LSIRM_FMC_v9.cpp` L1027-1029 — `tau_beta{1,2,3}_sq = 1.0 → 4.0` |

---

## 1. 시작 상태

- v9 chain 의 traceplot 에서 latent position `a` 수렴이 부진. 사용자 보고.
- v9 LSIRM hyperparameter 설정 (`MIDUS_5layered_result_v9.R` L113-141) 점검 요청.

## 2. v4 ~ v9 LSIRM hyperparameter 비교

같은 항목 대부분 동일. 주요 차이:

| 항목 | v4 | v5 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|---|
| `sd_log_kappa` | 0.5 | 0.5 | **0.1** | 0.1 | 0.1 | 0.1 |
| `(mu_beta4, sd_beta4)` | — | (0,2) | (0,2) | (0,2) | (0,2) | (0,2) |
| `(mu_u, sd_u, mu_delta, sd_delta)` | (0,2,0,1) | — | — | — | — | — |
| `prop_sd$alpha1, alpha2, alpha5` | 0.5 | 0.5 | 0.5 | (0.6, 0.4, 0.6) | same | same |
| `prop_sd$log_gamma3, log_gamma4` | 0.10, 0.20 | same | same | 0.05, 0.05 | same | same |
| `prop_sd$beta1, beta2, b4, log_kappa` | (0.6, 0.25, 0.5, 0.4) | same | same | (0.50, 0.10, 0.10, 0.30) | same | same |
| `n_iter / burnin` | 100k/20k | same | same | same | same | **120k/40k** |
| `nu2` | **5** | 4 | 4 | 4 | 4 | 4 |

→ v4: ordinal 이 GRM 이전 (delta + u parametrization). v5: GRM 도입 → `u/delta/delta2` 제거, `beta4/beta5` 추가. v6: `sd_log_kappa` 0.5→0.1 강화. v7: prop_sd 광범위 재튜닝. v8 = v9 (LSIRM 블록 동일). v9 만 burnin 확장.

## 3. cpp 코드 점검 — latent position prior 는 R 에서 못 바꿈

### 3.1 `a` (respondent latent position)
[my_LSIRM_FMC_v9.cpp:823-825](../data/my_LSIRM_FMC_v9.cpp#L823-L825):
```cpp
double lp_curr = -0.5 * dot(a_old, a_old);
double lp_next = -0.5 * dot(a_prop, a_prop);
```
→ **prior 가 $N(0, I_d)$ 로 cpp 에 하드코딩**. `lsirm_hyper` 의 어떤 항목도 이 prior 를 제어하지 않음. (`a_sigma0/b_sigma0` 는 continuous noise scale `sigma0_sq` 용)

### 3.2 `b1, b2, b3` (item latent position)
[my_LSIRM_FMC_v9.cpp:1027-1029](../data/my_LSIRM_FMC_v9.cpp#L1027-L1029) (수정 전):
```cpp
tau_beta1_sq = 1.0;
tau_beta2_sq = 1.0;
tau_beta3_sq = 1.0;
```
→ 매 sweep 마다 1.0 으로 강제 reset. `a_tau1, b_tau1, ...` 는 R 쪽에서 받아오지만 사실상 dead code. `b` 자체의 prior 는 별개로 [my_LSIRM_FMC_v9.cpp:609-610](../data/my_LSIRM_FMC_v9.cpp#L609-L610) 에서 `tau_beta_l_sq` 를 prior variance 로 사용 → **fixed effect $N(0, 1)$**.

## 4. alpha vs beta — random / fixed effect 분류

| 파라미터 | prior | variance 처리 | 분류 |
|---|---|---|---|
| `alpha1..5` (per-respondent intercept) | $N(0, \sigma_{\alpha_l}^2)$ | $\sigma_{\alpha_l}^2 \sim IG(a_\sigma, b_\sigma)$, 매 sweep Gibbs update ([cpp L1022-1026](../data/my_LSIRM_FMC_v9.cpp#L1022-L1026)) | **random effect** ✅ |
| `beta1, beta2, beta3` (item difficulty for bin/con/cnt) | $N(0, \tau_\beta^2)$ | $\tau_\beta^2 = 1.0$ 강제 (Gibbs disabled) | **fixed effect** (의도는 OK, 분산만 너무 좁음) |
| `beta4, beta5` (GRM threshold for ord1/ord2) | $N(0, \mathrm{sd}_\beta^2 = 4)$ | hyperparameter 로 직접 4 지정 | fixed effect ✅ |

## 5. Jeon et al. 논문 (LSIRM 원논문) prior 컨벤션 확인

`reference/LSIRM.pdf` Section 3.1 명시:

$$
\alpha_j \mid \sigma^2 \sim N(0, \sigma^2), \quad \sigma^2 \sim \mathrm{IG}(a_\sigma, b_\sigma)
$$

$$
\beta_i \mid \tau_\beta^2 \sim N(0, \tau_\beta^2), \quad \tau_\beta^2 \text{ FIXED}
$$

$$
a_j, b_i \sim \mathrm{MVN}_p(\mathbf{0}, I_p)
$$

**기본 hyperparameter (논문이 throughout 사용)**: $\tau_\beta^2 = 4, a_\sigma = b_\sigma = 1, \mu_\gamma = 0.5, \tau_\gamma^2 = 1$.

**왜 4 인가** (논문 인용):
> "values of $\alpha$ outside of the interval $[-5, +5]$ correspond to probabilities close to 0 or 1, which are unrealistic. ... using priors that place most probability mass on $[-5, +5]$ are reasonable"

→ logit/probit scale 의 effective range 가 $[-5, 5]$ 이므로 sd=2 인 $N(0, 4)$ 가 weakly informative.

**Section 4 의 abortion 분석에서는 $\tau_\beta^2 = 1$**:
> "For β, we chose **a stronger prior with $\tau_\beta^2 = 1$** because otherwise the MCMC did not converge well due to the boundary effects of probability for Items 5–7 (the positive answer probability was too close to 1)."

→ endorsement 가 0/1 boundary 에 닿는 극단 item 이 있을 때 한정적 선택. v9 의 현재 `tau = 1.0` 은 이 컨벤션 차용.

## 6. MIDUS data 진단 (1) — Binary endorsement 분포

`/tmp/diagnose_beta.R` 실행 (preprocess W2 + R1):

```
N = 221, P_bin = 20
Per-item endorsement proportion (sorted):
  B4Q3Q B4Q3O B4Q3I B4Q3S B4Q3J B4Q3B B4Q3C B4Q3A B4Q3N B4Q3M
  0.204 0.267 0.271 0.299 0.308 0.326 0.380 0.403 0.466 0.493
  B4Q3F B4Q3D B4Q3T B4Q3R B4Q3G B4Q3E B4Q3P B4Q3L B4Q3K B4Q3H
  0.525 0.552 0.575 0.597 0.615 0.620 0.661 0.692 0.733 0.765

p < 0.05: 0 items
p > 0.95: 0 items
range of p_endorse: [0.204, 0.765]
```

→ **모든 binary item 이 [0.20, 0.77] 범위 내. 극단 endorsement 없음.** 논문이 $\tau_\beta^2 = 1$ 로 죈 조건에 해당 안 됨.

## 7. v9 chain 진단 (2) — beta posterior 가 prior bound binding

`data/plot/case1_all_v9_fmc_r5_K10_e0.1/case1_all_v9_fmc_r5_K10_e0.1_result.rds` 로딩 → 분석.

prior sd = 1.0 ($\tau_\beta^2 = 1$), 95% prior bound ≈ ±1.96.

| 계층 | postmean 범위 | 95% CI 끝값 | bound 근접 (`|pm| > 1.76`) | 진단 |
|---|---|---|---|---|
| **beta1 (binary)** | [-2.465, 0.989] | -3.217, 1.571 | 3 / 20 | ⚠️ **binding** (postmean 자체가 prior 95% 밖) |
| beta2 (continuous) | [-0.454, -0.138] | -0.906, 0.075 | 0 / 5 | ✅ 여유 |
| **beta3 (count)** | [-2.930, 1.287] | -3.154, 2.006 | 2 / 15 | ⚠️ **binding** (postmean -2.93) |

대조 — alpha (random effect) 의 분산:
```
sigma_alpha1_sq postmean = 1.96  (binary)
sigma_alpha2_sq postmean = 0.082 (continuous, 표준화 영향)
sigma_alpha3_sq postmean = 0.138 (count)
sigma_alpha4_sq postmean = 1.24  (ord1 GRM)
sigma_alpha5_sq postmean = 1.93  (ord2)
```
→ alpha 의 분산은 데이터에서 자유롭게 1~2 까지 추정. **오직 beta 만 1.0 에 묶임.**

`a` (latent position) postmean:
```
n=221, d=2
Dim1 range [-2.131, 1.949], Dim2 range [-1.618, 1.545]
sd per dim: 0.750, 0.714
```

## 8. 진단 결론 — 비대칭이 latent position 비식별을 만든다

LSIRM linear predictor:
$$
\eta_{ij} = \alpha_{li} - \beta_{lj} - \gamma_l \|a_i - b_{lj}\|
$$

beta 가 0 근처에 묶이면:
1. item 평균 effect 를 $\beta_{lj}$ 가 충분히 흡수 못 함
2. 잔여 신호가 $\alpha_{li}$ (random, 자유) 와 $\gamma_l \|a_i - b_{lj}\|$ (거리) 로 새어 들어감
3. → $a_i, b_{lj}, \alpha_{li}$ 사이에 ridge 형태 비식별성
4. → MCMC chain 이 ridge 위를 떠다니며 latent position trace 수렴 부진

beta1/beta3 의 postmean 이 prior 95% bound 밖에 있다는 사실 = **데이터가 더 큰 |β| 를 원하지만 prior 가 잡아당기고 있음** 의 직접 증거.

## 9. 수정 — v9 만 적용

**파일**: `data/my_LSIRM_FMC_v9.cpp` L1027-1029

**변경 전**:
```cpp
tau_beta1_sq = 1.0;
tau_beta2_sq = 1.0;
tau_beta3_sq = 1.0;
```

**변경 후**:
```cpp
// 0501: 1.0 -> 4.0 (Jeon et al. paper default tau_beta^2 = 4).
// beta1/beta3 posterior was binding the prior bound on MIDUS data
// (postmean reaching -2.5 ~ -2.9 vs 95% prior bound = +/-1.96),
// pulling residual signal into a_i and breaking latent-position identifiability.
tau_beta1_sq = 4.0;
tau_beta2_sq = 4.0;
tau_beta3_sq = 4.0;
```

새로운 95% prior bound: $\pm 2 \cdot 1.96 \approx \pm 3.92$. beta1 (-2.47, -2.23, -2.02) 와 beta3 (-2.66, -2.93) 모두 새로운 bound 안에 여유롭게 들어옴.

**v6/v7/v8 cpp 는 동일 패턴이지만 사용자 요청대로 v9 만 수정.** 비교 실험을 다시 할 경우 일관성 확인 필요.

## 10. 다음 단계

1. v9 cpp recompile (다음 실행 시 wrapper 가 자동으로 sourceCpp 호출)
2. `MIDUS_5layered_result_v9.R` 재실행
3. 기대 효과:
   - beta1/beta3 postmean 이 ±3 까지 자유롭게 이동
   - alpha-beta-distance 사이 ridge 약화
   - latent position `a` 의 traceplot 수렴 개선
   - **부수 효과 가능성**: beta 가 자유로워지면 그동안 beta 가 못 흡수한 신호가 alpha 에서 빠져나가 `sigma_alpha_l_sq` postmean 도 변할 수 있음. 첫 진단 항목으로 비교할 것.
4. 만약 수렴이 여전히 부진하면:
   - `prop_sd$a` 튜닝 (현재 0.30) — acceptance rate 기반
   - Procrustes alignment 로 회전 비식별 검사
   - cpp 의 `a` prior 를 hyperparameter 로 노출 (현재 N(0, I_d) 하드코딩)

### 신규/수정 파일

| 파일 | 종류 |
|---|---|
| `data/my_LSIRM_FMC_v9.cpp` | 수정 (L1027-1029, tau_beta_l_sq 1.0 → 4.0) |
| `history/0501_history.md` | 신규 (이 문서) |

### 관련 (진단 스크립트, 영구 보관 아님)

| 파일 | 위치 |
|---|---|
| diagnose_beta.R | `/tmp/diagnose_beta.R` |
| diagnose 출력 | `/tmp/diag.out` |

---

# 0501 (오후) — v10 도입: NIW prior + collapsed Gibbs + Jain–Neal split–merge + S_0 권장값 갱신

## 오후 작업 요약

| # | 범주 | 산출 |
|---|---|---|
| 1 | 검토 | `modeling_paper/model_v10_niw_split_merge.tex` 수학 검증 — NIW marginal $2\pi → \pi$, `\frac{1}{2}` 깨진 LaTeX 수정, V0→κ0 매칭 공식 명시 |
| 2 | 구현 | `data/my_LSIRM_FMC_v10.cpp` 신규 — v9 LSIRM 블록 그대로 + NIW collapsed c-update + Jain–Neal restricted-Gibbs split–merge, `n_split_merge`/`row_center` 옵션 추가 |
| 3 | 구현 | `data/my_LSIRM_FMC_cpp_v10.R` wrapper 신규 — `fmc_hyper$V0 → kappa0` 으로 교체 |
| 4 | 시뮬레이션 | `simulation_4_layered_v10.R` 신규 — v9 와 동일 데이터 생성 메커니즘 |
| 5 | 진단 | r_fac=5 / row_ctr=T / S0=0.05: K_+=2, ARI 0.471 — V9 의 K_+=1 collapse 탈출 |
| 6 | 진단 | r_fac=2 / row_ctr=T / S0=0.05: K_+=3 (median), ARI 0.463, split rate 10.5% |
| 7 | 진단 | r_fac=2 / row_ctr=F / S0=0.05: ARI 0.463 (동일) — row-centering 은 병목 아님 확인 |
| 8 | 진단 | r_fac=2 / row_ctr=F / S0=**0.01**: K_+ median=5, **ARI 0.894**, split rate 14.9% — 진짜 병목은 NIW prior scale |
| 9 | 권장값 갱신 | modeling paper $S_0$ 디폴트 $0.05 I_r → 0.01 I_r$ + prior elicitation 절 추가 |

---

## 1. modeling paper 수정

### 1.1 NIW marginal likelihood 상수항
`modeling_paper/model_v10_niw_split_merge.tex` eq. (15):

**변경 전**: $-\frac{n_A r}{2}\log(2\pi)$
**변경 후**: $-\frac{n_A r}{2}\log\pi$

이유: Murphy (2007) eq. (232) — Gaussian density 의 $2$ 가 IW normalizing constant 의 $2^{\nu d/2}$ 와 정확히 상쇄되어 $\pi^{-n_A r/2}$ 만 남음. 이 항은 $c$ 에 의존하지 않는 상수이므로 split–merge MH ratio 결과에는 영향 없지만 절대값 (예: log evidence) 보고 시 틀림.

### 1.2 LaTeX form-feed 깨짐
eq. (29) 와 (30) 에서 `\frac{1}{2}` 가 form-feed (\x0c) + `rac12` 로 손상되어 있던 것을 `\frac{1}{2}` 로 정상화.

### 1.3 $\kappa_0$ 디폴트 + V9 매칭 공식
$E[\Sigma_l/\kappa_0] = S_0/((\nu_0 - r - 1)\kappa_0) \stackrel{!}{=} V_0$ 로 풀어 V9 의 $V_0 = 9I_r$ 와 매칭 시 $\kappa_0 \approx 6 \times 10^{-4}$ → 디폴트 $10^{-3}$ 명시.

### 1.4 (오후 추가) $S_0$ 디폴트 갱신 + prior elicitation 절
시뮬레이션 진단 결과 (3.5 절) 반영해 디폴트 표 업데이트:

| | 변경 전 | 변경 후 |
|---|---|---|
| $S_0$ | $0.05 I_r$ | **$0.01 I_r$** |

prior elicitation 가이드 절 추가: "η_pm posterior 의 within-cluster sd 를 진단해 $S_0$ scale 을 그 분산의 5–10x 정도가 되도록 잡는다" 원칙 + 실패 모드 (S_0 너무 크면 split Bayes factor 가 약화되어 K_+ 가 낮은 곳에 갇힘) 명시.

## 2. v10 sampler 구현

### 2.1 cpp 변경점 (`my_LSIRM_FMC_v10.cpp`)
v9 LSIRM 블록은 **byte-identical**. PPCA-mixture 블록만 교체:

| 단계 | v9 | v10 |
|---|---|---|
| (1) η_j | $N_r(m_\eta, V_\eta)$ | 동일 |
| (2) c_j | $\propto \rho_l \phi_r(\eta_j; \mu_l, \Sigma_l)$ | $\propto (n_{l,-j}+e_0) \cdot t_{\nu_n - r + 1}(\eta_j; m_n, V_{\rm pred})$ (collapsed) |
| (3) split–merge | — | Jain–Neal 1-scan restricted Gibbs, 빈 라벨 균등 선택, deterministic merge |
| (4) (μ_l, Σ_l) | independent N + IW posterior | NIW posterior draw (Σ → IW(ν_n, S_n), μ\|Σ → N(m_n, Σ/κ_n)) |
| rho | label update 전 sample | label update 후 storage 용으로만 sample |

### 2.2 새 옵션
- `n_split_merge` (int, default 1): sweep 당 split-merge 시도 횟수
- `row_center` (bool, default TRUE): row-centering 토글 (진단용)
- 출력에 `fmc_split_merge` list 추가 (split/merge attempts, accepts, rates)

### 2.3 R wrapper (`my_LSIRM_FMC_cpp_v10.R`)
`fmc_hyper$V0` → `fmc_hyper$kappa0` 교체. 디폴트 `kappa0=1e-3`, `nu0=r+10`, `S0=0.05*I_r` (이후 0.01 권장으로 갱신 예정).

## 3. 시뮬레이션 진단 사이클

v9 와 동일한 4-layer 데이터 (n=150, P=10/10/10/10, K_true=4 diamond clusters, 일부 회전 ellipse) 에서 50k iteration 으로 차례로 실험.

### 3.1 r_fac=5 / row_ctr=T / S0=0.05 (V10 디폴트)
- median K_+ = 2, mean 1.88, ARI = 0.471
- split rate 3.0%, merge rate 4.1%
- η posterior PC variance: 0.85, 0.09, 0.04, 0.01, 0.01 — 실질 1-D
- partition: top half {1,2} vs bottom half {3,4}
- **결론**: V9 의 K_+=1 collapse 탈출은 했지만 K_+=2 에 갇힘

### 3.2 r_fac=2 / row_ctr=T / S0=0.05
- median K_+ = 3, mean 2.63, ARI = 0.463
- split rate 10.5% (3.5x 증가)
- K_+ 가 1~5 까지 이동, 그러나 partition 은 여전히 {1,2} vs {3,4} + 1 singleton
- **결론**: r_fac 축소는 K_+ 탐색 폭은 넓혔으나 ARI 는 변화 없음

### 3.3 r_fac=2 / row_ctr=F / S0=0.05
- median K_+ = 3, mean 2.68, ARI = 0.463 (3.2 와 거의 동일)
- δ posterior mean vs row-mean log-dist 상관 = 0.909 → δ 가 row-centering 을 정상적으로 흡수
- σ_δ² mean 0.37 (vs row_ctr=T 의 0.016)
- **결론**: row-centering 은 병목이 아니다. δ 가 같은 일을 한다.

### 3.4 진단: η_pm cluster 평균
r_fac=2 / row_ctr=F 의 η_pm 은 4 클러스터 차이를 **실제로 담고 있음**:
```
cluster 1 (top-left  -1,+1): dim1=+0.016  dim2=+0.003
cluster 2 (top-right +1,+1): dim1=+0.022  dim2=+0.051   ← 가로 ellipse
cluster 3 (btm-right +1,-1): dim1=-0.036  dim2=-0.014
cluster 4 (btm-left  -1,-1): dim1=-0.035  dim2=-0.052   ← 회전 ellipse
```
dim1 = top vs bottom, dim2 = elongated vs round → 2D 구조 보존.
`kmeans(eta_pm, k=4)` 직접 ARI = **0.629** (모델 partition 0.463 보다 훨씬 높음).
→ 정보는 η 안에 있으나 sampler 가 못 뽑아냄. NIW prior $E[\Sigma_l] \approx 0.0056 I_r$, sd ≈ 0.075. 실제 within-cluster sd = 0.01–0.02. **prior 가 5–10x 느슨**.

### 3.5 r_fac=2 / row_ctr=F / S0=**0.01** (병목 해결)
- median K_+ = 5, mean 4.59, max 7
- **ARI partition = 0.894** (40 items 중 38 정확)
- split rate 14.9%, merge rate 4.2%
- Cross-tab: 11/12, 12+1/12, 7/8, 8/8 + 1 singleton

```
              true_item_cluster
final_partition  1  2  3  4
              1 11  0  0  0
              2  1 12  0  0
              3  0  0  7  0
              4  0  0  0  8
              5  0  0  1  0
```

→ **사실상 perfect recovery**.

## 4. 정리

V9 → V10 발전 경로 분해:

| 단계 | ARI | 핵심 변화 |
|---|---|---|
| V9 | -0.006 | (collapsed at K_+=1) |
| V10 r5 ctr=T S=0.05 | 0.471 | NIW + collapsed Gibbs + split–merge → K_+=1 lock-in 풀림 |
| V10 r2 ctr=T S=0.05 | 0.463 | r_fac 축소 (변화 없음) |
| V10 r2 ctr=F S=0.05 | 0.463 | row_center 제거 (변화 없음) |
| **V10 r2 ctr=F S=0.01** | **0.894** | **S_0 tightening — 진짜 병목** |

## 5. 신규/수정 파일

| 파일 | 종류 |
|---|---|
| `modeling_paper/model_v10_niw_split_merge.tex` | 수정 (NIW marginal $2\pi→\pi$, LaTeX 깨짐 복구, $\kappa_0$ 매칭 공식, $S_0$ 디폴트 0.05→0.01, prior elicitation 절 추가) |
| `data/my_LSIRM_FMC_v10.cpp` | **신규** (NIW collapsed Gibbs + Jain–Neal split–merge) |
| `data/my_LSIRM_FMC_cpp_v10.R` | **신규** (wrapper, V0→κ0) |
| `simulation_4_layered_v10.R` | **신규** (v9 와 동일 시뮬레이션 setup) |
| `data/MIDUS_5layered_result_v10.R` | **신규** (v9 mirror, $S_0=0.01 I_r$, `n_split_merge=5`) |
| `history/0501_history.md` | 본 섹션 추가 |

### 진단 스크립트 (영구 보관 아님)
| 파일 | 위치 |
|---|---|
| analyze_v10.R | `/tmp/analyze_v10.R` |
| analyze_v10_rfac2.R | `/tmp/analyze_v10_rfac2.R` |
| analyze_v10_norowctr.R | `/tmp/analyze_v10_norowctr.R` |

---

# 0501 (저녁) — v10 coverage + cluster recovery 시뮬레이션 (서버용 단일 파일)

## 작업 요약

| # | 범주 | 산출 |
|---|---|---|
| 1 | 신규 폴더 | `simulation_FMC_LSIR_v10/` — 시뮬레이션 코드 + 모든 CSV/RDS 산출물을 한 곳에 모음 (서버 이식성) |
| 2 | 신규 스크립트 | `simulation_FMC_LSIR_v10/simulation_FMC_LSIR_v10.R` — multilayer vs unilayer (이진화) v10 모델 비교 시뮬레이션 |
| 3 | 평가 metric 정의 | parameter coverage (95% CrI 포함률 + 폭) + 클러스터 복원 DICE (label-invariant pair-counting Sørensen index) |

## 1. 디자인 결정

### 1.1 단일 파일 + 단일 폴더 정책
사용자 요청: "서버에서 구동될 것이기 때문에 하나의 파일 안에서 전부 수행되어야 해. 나중에 복사해서 서버에 붙여넣을 수 있도록."

→ 모든 시뮬레이션 로직 (데이터 생성, multilayer/unilayer fitting, coverage, DICE, aggregation) 을 [simulation_FMC_LSIR_v10.R](../simulation_FMC_LSIR_v10/simulation_FMC_LSIR_v10.R) 단일 파일로 작성.
→ 파일 상단 `BASE_DIR` / `OUTPUT_DIR` / `MODEL_DIR` 만 바꾸면 서버 이식 가능.
→ 모든 출력 파일명 끝에 `_v10` suffix.

### 1.2 multilayer vs unilayer 비교 디자인
- multilayer: 4개 layer (binary/continuous/count/ordinal) 그대로 v10 모델에 투입.
- unilayer: 각 layer 를 column 평균 기준 이진화 → 단일 binary matrix 로 합쳐 v10 모델의 `Y_bin` 에만 투입 (다른 layer 는 빈 matrix).
- 두 모델 모두 v10 sampler 사용 → FMC 클러스터링 비교 가능. cluster 가정은 데이터 생성 단계에 내장.

### 1.3 parameter coverage 범위 선택
**multilayer**: alpha1..4, beta1..3, beta4 (GRM thresholds), distance(a,b_l)*gamma_l (각 layer), gamma1..4, sigma0_sq.
**unilayer**: distance(a, b_all)*gamma1, gamma1 만. 이유 — 이진화 모델의 alpha/beta 는 multilayer truth 와 1-to-1 대응이 없음 (random-effect 4개 → 1개로 collapse). 거리는 회전·이진화에 invariant 한 기하 quantity 이므로 비교 가능.

### 1.4 cluster 복원 DICE
$$
\mathrm{DICE} = \frac{2A}{2A + B + C}, \quad
A = |\{(i,j): I^{\rm true}_{ij} = 1 \wedge I^{\rm est}_{ij} = 1\}|, \ldots
$$
- pair-counting Sørensen index. label switching 에 invariant. ARI 도 함께 보고하지만 primary metric 은 DICE.
- est partition: posterior similarity matrix 의 average-linkage cutree(K = K_true).

## 2. 산출 CSV 구조

| 파일 | 포맷 | 단위 |
|---|---|---|
| `coverage_per_rep_v10.csv` | long format (rep × model × param) | rep 별 raw |
| `coverage_summary_v10.csv` | wide (model × param 별 mean ± sd) | rep 평균 |
| `cluster_dice_v10.csv` | wide (rep 별 multi/uni 양쪽 metric) | rep 별 raw |
| `cluster_dice_summary_v10.csv` | model 별 mean ± sd | rep 평균 |
| `sim_settings_v10.rds` | 설정 snapshot | 1 회 |

각 rep 마다 incremental save (`coverage_per_rep_v10.csv`, `cluster_dice_v10.csv`) — 서버 interrupt 시 부분 결과 보존.

## 3. MCMC 설정

`n_iter=30k, burnin=10k, thin=10` (per fit). 기본 `N_REP=10` → 총 20 fits. r_fac=2, K_star=10, K_true=4, S0=0.01 I_r (modeling paper 권장값), `n_split_merge=5`, `row_center=FALSE`. 모두 0501 오후의 v10 r2 ctr=F S=0.01 setup 과 동일.

## 4. 신규 파일

| 파일 | 종류 |
|---|---|
| `simulation_FMC_LSIR_v10/simulation_FMC_LSIR_v10.R` | **신규** (자체완결 시뮬레이션 + coverage + DICE) |
| `history/0501_history.md` | 본 섹션 추가 |
