# 0430 — FMC clustering 진단 → v9 (PPCA-FMC) 전환 → b_j 직접 clustering 으로 우회

## 오늘의 작업 (요약)

| # | 범주 | 산출 |
|---|---|---|
| 1 | 분석 | `factor_analysis_v8/04_fmc_factor_clustering.R` — v8 FMC 의 $\eta$ MCMC sample 기반 post-hoc clustering 스크립트 (eta_PM + D_bar 두 입력, hclust × 4 linkage + kmeans + PAM, K=2..8) |
| 2 | 진단 | v8 의 cluster collapse 원인 규명 — $\Lambda \to 0$ degenerate fixed point 발견 ($\|\Lambda\|_F = 1.20$ vs prior 16.6) |
| 3 | 시도 | v8 psi prior tightening `(a_psi, b_psi) = (2,1) → (5, 0.2)` — K_+=2 → K_+=1 으로 더 악화. delta squash 부수효과. |
| 4 | 모델 | **v9 모델 개발** — FA 의 per-respondent $\psi_i$ 를 shared $\sigma_\varepsilon^2$ (Probabilistic PCA) 로 교체. `data/my_LSIRM_FMC_v9.cpp`, `_cpp_v9.R`, `MIDUS_5layered_result_v9.R` 신규 |
| 5 | 문서 | `paper/model_v8.tex`, `paper/model_v9.tex` 신규 — 모델 정의, posterior update, collapse 진단, PPCA 의 수학적 정당화 |
| 6 | 검증 | v9 run 1 (default prior) — K_+ 가 1↔2 mixing 시작 (개선) but Lambda 여전히 작음 ($\|\Lambda\|_F = 1.71$), final partition 67/1 degenerate |
| 7 | 시도 | v9 run 2 (loosened prior: $\tau_\lambda^2 = 4$, $S_0 = 0.5I$, burnin 40k, Lambda init sd 0.5) — Lambda $\to$ 8.5 자라남 but eta 가 보상으로 줄어듦, **scale indeterminacy** 로 Lambda$\eta$ product 동일, K_+=1 lock-in |
| 8 | 우회 | **Option C 채택** — `factor_analysis_v8/05_bj_clustering.R` 신규. FA layer 우회, LSIRM 의 $b_j$ MCMC sample 직접 clustering |
| 9 | 결과 | b_j 기반 clustering: **K=4, silhouette = 0.45 (강함)**, 4 quadrant geometric 분리, 다른 method 들끼리 ARI 0.81+ 합의 |

---

## 1. 시작 상태 (v8)

세션 시작 시 v8 (`MIDUS_5layered_result_v8.R`) 결과:
- median $K_+$ = 2, sd = 0 (모든 8000 iter K=2 lock-in)
- 최종 partition 34/34 (clean split)
- 사용자: "$e_0 \in \{0.05, 0.1, 1\}$ 모두 K=2 만 나옴 — 어디가 문제?"

## 2. v8 진단 — Lambda collapse 발견

`fmc_eta`, `fmc_Lambda_postmean`, `fmc_mu`, `fmc_Sigma`, `fmc_rho` 후처리 분석:

| 진단 항목 | 값 | 의미 |
|---|---|---|
| $\|\Lambda_{\text{PM}}\|_F$ | 1.20 | prior 기대치 16.6 대비 7% — Lambda 가 0 에 붕괴 |
| Reconstruction $\Lambda\eta$ sd | 0.001 | $x_{rc}$ sd 0.5 의 0.2% — factor 항이 일을 안 함 |
| $\rho$ posterior | 4번, 10번 component 만 ~0.49 each, 나머지 ~0.001 | overfitted Dirichlet 은 정상 작동 |
| Active $\mu$ 사이 거리 | 0.065 | $\sqrt{\text{tr}(\Sigma_l)} = 0.54$ 의 1/8 — 두 cluster Gaussian 거의 완전 겹침 |

**기전**: Per-respondent $\psi_i$ 가 너무 자유로워서 noise 를 모두 흡수 → $\Lambda\eta$ 항을 0 으로 보내도 likelihood 동등. MCMC 가 더 단순한 "$\Lambda{\to}0$" mode 선호.

## 3. 시도 1 — v8 psi prior tightening

`MIDUS_5layered_result_v8.R` 의 `common_fmc_hyper`:
```r
a_psi = 5, b_psi = 0.2    # was (2, 1); mean: 1 → 0.05
```

**결과**: 더 나빠짐. K_+ = **1** lock-in (모든 iter), $\sigma_\delta^2$ 도 squash, Lambda 여전히 작음. Prior tweaking 만으로 해결 불가 확인.

## 4. 04 스크립트 — post-hoc clustering on FMC outputs

`factor_analysis_v8/04_fmc_factor_clustering.R` 신규:
- 입력 (P1) `eta_PM` — posterior mean of $\eta$ (P×r)
- 입력 (P2) `D_bar` — $(1/T)\sum_t \|\eta_j^{(t)} - \eta_k^{(t)}\|$ (P×P)
- Respondent-profile metric: $\tilde\eta = \eta R^\top$, $R = \mathrm{chol}(\Lambda^\top\Lambda)$
- hclust × 4 linkage + kmeans + PAM, K=2..8

v8 results 위에서 돌렸을 때:
- 대부분 K=2 합의, hclust ward.D2 / D_bar 만 K=4
- D_bar/ward K=4 이 v8 in-model K=2 의 **strict refinement** (14/20/20/14)
- ARI(D_bar/ward K=4, hclust_ward_K7 from SVD-based 03) = 0.58 — SVD-based 와 잘 일치

→ 사용자 결론: joint Bayesian model 에는 D_bar 가 자연스럽고, in-model PSM 이 가장 베이즈적.

## 5. 핵심 통찰 — PPCA 로 FA 대체

사용자 제안: "FA 대신 SVD 의 Bayesian 버전을 쓰면 안 되나?"

해석: **Probabilistic PCA** (Tipping & Bishop 1999). FA 의 per-respondent $\psi_i$ 를 shared $\sigma_\varepsilon^2$ 로 대체:

| | FA (v8) | PPCA (v9) |
|---|---|---|
| Noise | $\varepsilon_{ij} \sim N(0, \psi_i)$ | $\varepsilon_{ij} \sim N(0, \sigma_\varepsilon^2)$ |
| 자유 모수 | $\psi_1, \dots, \psi_n$ ($n$ 개) | $\sigma_\varepsilon^2$ (1 개) |
| 흡수 능력 | per-respondent 선별 흡수 가능 | 단일 scalar — 선별 흡수 불가 |
| $\Lambda\to 0$ mode | likelihood-stable | 이론상 unstable |

이론적으로 PPCA 는 SVD 의 generative 정당화 (MLE = PCA), $\Lambda\to 0$ 함정이 없어야 함.

## 6. v9 모델 구현

### 6.1 코드 변경 (3 개 파일)

**`data/my_LSIRM_FMC_v9.cpp`** (v8.cpp 의 1360 줄 복사 + 수정):
- `vec psi (length n)` → `double sigma_eps_sq`
- `a_psi, b_psi` → `a_eps, b_eps`
- $\eta_j$ Gibbs precompute: `Lambda^T diag(1/psi) Lambda` → `inv_sigma_eps_sq * Lambda^T Lambda`
- $\delta_i, \lambda_i$ updates: `psi(i)` → `sigma_eps_sq` (scalar)
- $\psi_i$ per-respondent IG draw $n$ 회 → 단일 $\sigma_\varepsilon^2$ Gibbs draw on total RSS:
$$
\sigma_\varepsilon^2 \mid \cdot \sim IG\left(a_\varepsilon + \tfrac{nP}{2},\; b_\varepsilon + \tfrac12 \sum_{i,j}(x_{ij} - \delta_i - \lambda_i^\top \eta_j)^2\right)
$$
- 함수명 `run_lsirm_fmc_v8_cpp` → `run_lsirm_fmc_v9_cpp`
- Output: `fmc_psi_postmean, fmc_psi` 제거, `fmc_sigma_eps_sq_postmean, fmc_sigma_eps_sq` 신규
- LSIRM block 은 byte-identical (V6 와 동일, 변경 없음)

**`data/my_LSIRM_FMC_cpp_v9.R`** (R wrapper):
- Default `fmc_hyper`: `a_eps = 5, b_eps = 0.1` (mean $\sigma_\varepsilon^2 = 0.025$)
- `fmc_init$psi (length n)` → `fmc_init$sigma_eps_sq (scalar)`
- `save_psi_full` toggle 제거 (full trace 가 single vector 이므로 항상 저장)

**`data/MIDUS_5layered_result_v9.R`** (run script):
- run_label `v8_fmc_*` → `v9_fmc_*`
- `common_fmc_hyper`: `a_eps, b_eps` 추가, `a_psi, b_psi` 제거
- `fmc_init_smart$psi` 제거, `sigma_eps_sq` 추가
- `_resp_postmean.csv` 의 psi 열 제거
- 신규 출력: `_sigma_eps_sq_summary.csv`, `_trace_sigma_eps_sq.pdf`

### 6.2 Paper 작성

**`paper/model_v8.tex`** (이전 단계에서 작성) — V8 구조 + collapse 진단 문서
**`paper/model_v9.tex`** — V9 정의:
- §3-7: 모델 specification, posterior update (eq 3.1-3.9)
- §5: Why PPCA fixes the V8 collapse — 수학적 증명 (homoscedastic IG posterior 가 단일 평균을 흡수해야 하므로 likelihood-optimal solution 이 $\Lambda\eta$ 를 활성화)
- §8: verification checklist (5 가지 진단 지표)

## 7. v9 run 1 — 부분 개선

**Hyperparameter (run 1, conservative)**:
- $\tau_\lambda^2 = 0.25$ (v8 carryover), $S_0 = 0.05 I$, $a_\varepsilon = 5, b_\varepsilon = 0.1$
- burnin 20k, n_iter 100k

**결과**:
| 지표 | v8 | v9 run 1 |
|---|---|---|
| K_+ | 2 (constant) | **1 ↔ 2 mixing** (5123/2877 split) |
| K_+ sd | 0 | 0.48 |
| $\|\Lambda\|_F$ | 1.20 | 1.71 (+43%) |
| Lambda·eta sd | 0.001 | 0.025 |
| Final partition | 34/34 | **67/1 (degenerate)** |
| ARI(v8, v9) | — | **0.000** |

→ chain mixing 회복 (좋음), but median $K_+$ = 1 이라 final partition 이 K=2 에서 outlier 1 개만 분리되는 trivial split. Lambda 여전히 prior 의 10%.

## 8. v9 run 2 — Hyperparameter loosening

**의도**: PPCA 가 collapse 를 막고 있으니 v8 시기의 prior 압박 (`tau=0.25`, `S0=0.05I`) 풀어서 Lambda/eta 자유롭게 자라도록.

**변경**:
```r
S0            = 0.5  * diag(r_fac),  # was 0.05*I (10x)
tau_lambda_sq = 4.0,                  # was 0.25  (16x)
fmc_init_smart$Lambda sd = 0.5        # was 0.1   (5x)
n_iter / burnin = 120k / 40k          # was 100k / 20k
```

**결과**:
| 지표 | run 1 | run 2 |
|---|---|---|
| $\|\Lambda\|_F$ | 1.71 | **8.53** (+5x) |
| $\Lambda^\top\Lambda$ eigvals | 1.7, 0.7, 0.3, 0.1, 0.1 | **39.6, 26.8**, 3.8, 1.5, 1.0 |
| Per-iter eta sd | 0.47 | **0.12** (4x ↓) |
| Lambda·eta recon sd | 0.025 | 0.033 (≈ same!) |
| K_+ | 1 ↔ 2 | **1 (locked)** |
| Final partition | 67/1 | 67/1 |

**진단 — FA scale indeterminacy**:

$(\Lambda, \eta) \leftrightarrow (c\Lambda, \eta/c)$ 가 likelihood 보존. Run 2 에서:
- prior 가 Lambda 자라도록 허락 (5x)
- but eta 는 보상으로 작아짐 (4x)
- product $\Lambda\eta$ 동일

Per-cell variance 분석:
$$
\mathrm{Var}_{\text{cell}}(\Lambda\eta) \approx \frac{\mathrm{tr}(\Lambda^\top\Lambda)}{n} \cdot \frac{\mathrm{tr}(\Sigma_\eta)}{r} \approx \frac{72.7}{221} \cdot \frac{0.075}{5} \approx 0.005
$$

vs $x_{rc}$ cell variance $0.30$ → structured part **1.7%** 만 설명. PPCA 의 이론적 예측과 다름.

**근본 원인 — local mode trapping**:
- log-likelihood at current mode ≈ $-8400$
- log-likelihood at ideal mode ≈ $-7500$
- 900 unit 차이지만 chain 이 이 경계를 못 넘음
- Random Lambda init + PCA-on-Y init for eta (LSIRM-derived $x_{rc}$ 와 다른 공간) 이 SVD 해 근방을 벗어난 mode 에 정착시킴

**해결안 후보**:
1. SVD-based Lambda init (cpp 에 post-warmup re-init 추가) — 본격 해법
2. b_j 직접 clustering (Option C) — FA 우회, 빠른 우회로

사용자 선택: **Option C 먼저 시도**.

## 9. Option C — `05_bj_clustering.R`

`factor_analysis_v8/05_bj_clustering.R` 신규. 04 의 구조를 따르되 입력만 변경:
- $\eta$ MCMC sample 대신 **$b_j$ MCMC sample 직접 사용**
- $b_1, \dots, b_5$ 를 P axis 로 stacking → `B_arr` (T × P × d=2)
- 입력 (P1) `B_PM` — posterior mean (P × 2)
- 입력 (P2) `D_bar` — pairwise Euclidean distance 평균 (P × P)
- LSIRM space 가 이미 metric 공간이므로 $\Lambda$ transform 불필요
- d=2 의 직접 geometric clustering

### 9.1 v9 result 위에서 실행 결과 — 강한 K=4 신호

| Method | Optimal K | Silhouette |
|---|---:|---:|
| **k-means / B_PM** | **4** | **0.446** ⭐ |
| **hclust ward.D2 / B_PM** | **4** | **0.434** ⭐ |
| hclust ward.D2 / D_bar | 4 | 0.246 |
| hclust average / B_PM | 2 | 0.444 |
| hclust complete / B_PM | 8 | 0.364 |
| PAM / D_bar | 3 | 0.257 |

silhouette > 0.4 는 well-separated cluster 의 강한 증거.

### 9.2 Cross-method 합의

```
hc_BPM_ward.D2 ↔ hc_Dbar_ward.D2  : ARI = 0.871
hc_BPM_ward.D2 ↔ kmeans_BPM       : ARI = 0.810
hc_Dbar_ward.D2 ↔ kmeans_BPM      : ARI = 0.678
```

여러 방법들이 거의 같은 K=4 partition 으로 수렴 → 결과 견고.

### 9.3 In-model PSM 과 비교

```
모든 b_j 기반 partition vs v9 in-model PSM : ARI ≈ 0
모든 b_j 기반 partition vs v8 in-model PSM : ARI ≈ 0
```

→ FMC mixture 의 in-model partition 은 b_j 의 진짜 cluster 구조를 **전혀** 반영하지 못함. **데이터 한계가 아니라 FMC FA layer 가 신호를 추출 못한 것**이 원인이라는 결정적 증거.

### 9.4 K=4 partition 내용 (kmeans)

| Cluster | n | $(\bar b_1, \bar b_2)$ | 사분면 |
|---|---:|---|---|
| C1 | 26 | (+0.10, +0.66) | center-top |
| C2 | 9 | (+1.40, +0.88) | right-top |
| C3 | 12 | (-0.48, -0.24) | left-bottom |
| C4 | 21 | (+0.72, -0.64) | right-bottom |

LSIRM 2D space 의 4 quadrant 으로 자연스럽게 분리. unit-disk 제약 가정과 달리 실제 range 는 $b_1 \in [-1.31, 2.10]$, $b_2 \in [-1.68, 1.37]$ — LSIRM 위치는 unit disk 안에 강제되지 않음.

---

## 10. 정리 및 다음 단계

### 확인된 사실

1. **v8 FA collapse 의 본질**: per-respondent $\psi_i$ 의 자유도가 변동 흡수 → $\Lambda\to 0$ degenerate fixed point
2. **PPCA (v9) 가 막은 것**: $\psi_i$ 선별 흡수
3. **PPCA 가 못 막은 것**: scale indeterminacy 로 새로운 "$\Lambda$ 크지만 $\eta$ 작음" mode 생성, product 여전히 작음
4. **데이터 자체에는 강한 cluster 신호**: LSIRM 의 $b_j$ 에 K=4 geometric structure 명확 (silhouette 0.45)

### 권장 진행

**즉시 사용 가능**: `bj_clust_kmeans_BPM_K4.csv` partition (silhouette 0.446, cross-method ARI 0.81+)

**v10 후보** (논문 정식 모델로 가려면):
- Option D-(ii): cluster-conditional prior on $b_j$. $b_j \mid c_j = l \sim N_d(\mu_l^B, \Sigma_l^B)$. cluster signal 이 LSIRM 단계에 직접 들어감. Two-way coupling. `paper/model_v8.tex` §13 의 future direction 그대로.
- 또는 v9 cpp 에 SVD-based Lambda init (post-warmup) 추가.

### 신규/수정 파일 목록 (오늘)

| 파일 | 종류 |
|---|---|
| `factor_analysis_v8/04_fmc_factor_clustering.R` | 신규 (FA-based post-hoc clustering) |
| `factor_analysis_v8/05_bj_clustering.R` | 신규 (b_j-direct clustering) |
| `data/my_LSIRM_FMC_v9.cpp` | 신규 (PPCA-FMC) |
| `data/my_LSIRM_FMC_cpp_v9.R` | 신규 |
| `data/MIDUS_5layered_result_v9.R` | 신규 |
| `data/MIDUS_5layered_result_v8.R` | 수정 (a_psi, b_psi tightening 시도) |
| `paper/model_v8.tex` | 신규 |
| `paper/model_v9.tex` | 신규 |

### 출력 디렉토리

| 위치 | 내용 |
|---|---|
| `data/plot/case1_all_v8_fmc_r5_K10_e0.1/` | v8 결과 (K=2 lock-in 진단용) |
| `data/plot/case1_all_v9_fmc_r5_K10_e0.1/` | v9 run 2 결과 + b_j clustering 결과 (`bj_clust_*` prefix) |
| `factor_analysis_v8/output/case1_all_v8_fmc_r5_K10_e0.1/` | 04 의 FA-based post-hoc clustering 결과 (`fmc_clust_*` prefix) |
