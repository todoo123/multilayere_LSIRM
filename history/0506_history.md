# 0506 — v12 EPA simulation 정밀화, MIDUS v12 적용, PSM ambiguity 진단

## 오늘의 작업

| # | Task | 산출 |
|---|---|---|
| 1 | v12 simulation 의 cluster contrast 강화 — `simulation_4_layered_v12_2.R` 신규 | `sigma_meta` 1/3, `sd_cluster_resp` 0.3 — within-cluster sd 0.32–0.45 → 0.17–0.26 |
| 2 | v12_2 simulation `(α, τ)` 운영점 탐색 (sweet spot 확정) | `(α=1, τ=1) fixed` 만 K=4 회복; `(α, τ)` free 또는 `τ≥2` 는 collapse |
| 3 | v12_2 simulation 의 long-chain 검증 (100k iter) | 20k 결과와 본질 동일 (median K=4, hclust ARI 0.978–1.0); chain 길이는 contrast 를 직접 향상 안 함 |
| 4 | MIDUS v12 신규 파일 — [data/MIDUS_5layered_result_v12.R](../data/MIDUS_5layered_result_v12.R) | v11 구조 유지하면서 NIW mixture → EPA pairwise partition 으로 교체 |
| 5 | MIDUS v12 첫 운영 (`α=1, τ=1` 100k iter) | median K=4, hclust 깔끔한 4-cluster, 그러나 PEAR 0.01 / ambig zone 99.7% |
| 6 | MIDUS τ-sweep `(τ ∈ {1.3, 1.5})` | ambig zone 99.7% → 93.9% → 80.9%, 단 cluster 내용 사실상 불변 |
| 7 | PSM ambiguity 의 근본 원인 진단 | EPA prior 식별성 한계 — 데이터가 c 직접 관찰 안 하고 prior 만으로 c 식별 |
| 8 | 결정: v11 NIW mixture 재검토 (PSM contrast 가 v12 보다 sharper 가능성) | 다음 작업 — `(nu0, S0)` 재고려 필요 (현재 100 / 6 가 큰 값으로 의심) |
| 9 | v11 simulation best-setting 소급 기록 (`S0=0.3, kappa0=1, nu0=12, e0=0.1, K*=10`) | ARI hclust/Dahl/Binder/VI = 0.860 / 0.878 / 0.878 / 0.868 (true K=4 vs median K_+=5) |

**결정 요지**: v12 의 PSM ambiguity 는 sampler bug 가 아니라 **EPA prior architecture 의 inherent 식별성 한계**임을 simulation + MIDUS 양쪽에서 확인. 더 나은 PSM contrast 가 필요하면 v12 EPA 가 아닌 v11 NIW mixture 로 돌아가야 함. v11 의 `(nu0=100, S0=6)` lock-in 의 정당성을 재검토하고 운영점 재탐색.

---

# Task 1. simulation_4_layered_v12_2 — cluster contrast 강화

## 1.1 동기

기존 [simulation_4_layered_v12.R](../simulation_4_layered_v12.R) 에서 EPA partition sampler 가 K=4 회복 못 하는 (PEAR≈0, Dahl/Binder/VI K=1 collapse) 문제 — 데이터 어려움 때문인지 sampler 한계인지 분리하기 위해 **easier mode** 시뮬을 따로 만듦.

## 1.2 변경점

[simulation_4_layered_v12_2.R](../simulation_4_layered_v12_2.R):

```r
# v12: sigma_meta = list(diag(c(0.15, 0.15)), ..., diag(c(0.18, 0.18)))
# v12_2:
sigma_meta <- list(
  diag(c(0.05, 0.05)),
  diag(c(0.07, 0.07)),
  diag(c(0.03, 0.03)),
  diag(c(0.06, 0.06))
)
sd_cluster_resp <- 0.3   # v12: 0.5
```

cluster centers (±1, ±1) 불변. within-cluster sd: 0.32–0.45 → **0.17–0.26**. inter-center 거리 / within-sd ≈ 10 SD 분리 (v12 는 ~5 SD).

## 1.3 plot 디렉토리 분리

`plot_dir` 이름을 `simulation_4_layered_v12_2_*` 로 변경해 v12 output 과 충돌 방지.

---

# Task 2. v12_2 simulation `(α, τ)` 운영점 탐색

## 2.1 default expA — `(α, τ)` free, 20k iter

`V12_EXPERIMENT=A` (update_alpha=TRUE, update_tau=TRUE).

| 지표 | 결과 |
|---|---|
| mean K_+ | **2.55** |
| median K_+ | 2 |
| ARI(hclust) | 0.342 |
| ARI(Dahl/Binder/VI) | 0/0/0 |
| posterior mean α | 0.472 (prior mean 1 보다 작음) |
| posterior mean τ | 0.911 |
| co_within − co_between | **0.017** |
| split rate | 0.059 |
| merge rate | 0.533 |

**해석**: α 가 0.47 로 끌려와 partition prior 가 K-작음 쪽 mass 강화 → over-merge. **데이터 자체에는 K=4 신호가 분명히 있으나 (cluster 분리 ~10 SD), free `(α, τ)` 가 collapse 유도**.

## 2.2 fix `(α=1, τ=1)`, 20k iter — sweet spot 발견

| 지표 | 결과 |
|---|---|
| **median K_+** | **4 ✓** |
| mean K_+ | 4.03 |
| sd K_+ | 1.73 |
| **ARI(hclust)** | **1.000 ✓** (완벽 회복) |
| ARI(Dahl/Binder/VI) | 0/0/0 (여전히 collapse) |
| co_within − co_between | 0.061 |
| split rate | 0.062, merge rate 0.491 |

**핵심**: posterior mean b̂ 가 4-cluster 로 완벽 회복. **그러나 Dahl/Binder/VI 는 K=1 으로 collapse** — 이는 per-iter assignment jitter 가 PSM 을 평탄화시키기 때문 (이슈가 정점추정 함수에 있지, chain 자체는 정상).

## 2.3 τ-sweep `(τ ∈ {2, 3})`, 100k iter

| 지표 | τ=1 long | τ=2 long | τ=3 long |
|---|---|---|---|
| median K_+ | 4 ✓ | 3 | 3 |
| ARI(hclust) | 0.978 | 0.711 | 0.711 |
| split rate | 0.057 | 0.0165 | **0.0062** |
| co_contrast | 0.055 | 0.046 | 0.034 |

**τ↑ → split rate exponential 감소 → chain 이 한 번 merge 되면 못 빠져나옴 → K=3 collapse**. τ 키우는 것은 직관과 반대로 **악화**.

## 2.4 τ=1 long (100k vs 20k 비교)

| 지표 | τ=1 short | τ=1 long |
|---|---|---|
| median K_+ | 4 | 4 |
| mean K_+ | 4.03 | 3.95 |
| max K_+ | 13 | 11 (tail 줄음) |
| ARI(hclust) | 1.000 | 0.978 |
| co_contrast | 0.061 | 0.055 |

**iter 5배 늘려도 본질 동일**. chain 은 20k 에서 이미 수렴.

## 2.5 결론

`(α=1, τ=1) fixed` 가 v12 EPA sampler 의 sweet spot. paper main result table 에 사용.

---

# Task 3. MIDUS v12 신규 파일

## 3.1 [data/MIDUS_5layered_result_v12.R](../data/MIDUS_5layered_result_v12.R) 작성

v11 의 NIW mixture 블록을 EPA pairwise partition 으로 교체. 주요 차이:

| | v11 | **v12** |
|---|---|---|
| 모델 | NIW finite mixture (b_j ~ N(mu_l, Sigma_l)) | EPA pairwise (lambda_qr = exp(−τ‖b_q−b_r‖²)) |
| 구조 hyperparameter | K_star=10, e0=0.05, NIW(kappa0, m0, nu0, S0) | sigma_b=1, alpha, tau, delta=0 |
| Sampler | n_split_merge=100, fmc_warmup=20% | M_SM=1, R=5, M_perm=P_total, epa_warmup=0 |
| MCMC | 80k / 20k / 10 | 100k / 20k / 10 |

운영 setting: `α=1, τ=1` 고정 (simulation 결과 따라).

## 3.2 첫 run 결과 (`α=1, τ=1`, 100k iter)

데이터: n=221, P_total=68 (bin 20 / con 5 / cnt 15 / ord 28)

| 지표 | 값 |
|---|---|
| mean K_+ | 4.08 |
| **median K_+** | **4** |
| sd K_+ | 1.73 |
| split rate | 0.170 (sim 보다 3배 ↑) |
| merge rate | 0.508 |
| sigma swap rate | 0.880 |
| **silhouette (b̂)** | **0.420** (moderate) |
| **PEAR** | **0.010** (weak) |
| Pr(co ∈ [0.3, 0.7]) | **99.7%** (ambig zone 거의 전부) |
| within-pair ratio | 0.000 |
| Dahl/Binder/VI | K=1 collapse |
| Final partition (P0) | 43 / 8 / 16 / 1 (median K=4 avg-link cut) |

**임상적으로 의미있는 발견**: P0 의 **C2 (8 items)** 가 **inflammation marker 3종 (B4BCRP, B4BFGN, B4BIL6) + executive function set-shifting tasks (sgst_mixed_*) + B4Q3B** 를 묶음. 이게 v12 가 v11/v10 과 다르게 추출한 새 sub-cluster.

**LSIRM acceptance**: β₂ = 0.097 (5개짜리 continuous layer 의 acceptance 너무 낮음 — proposal SD 0.4 → 0.15 로 튜닝 필요. 별도 작업).

산출물: [data/plot/case1_all_v12_epa_d2_sigb_1_a1_t1_uAF_uTF_M1_R5_iter100k/](../data/plot/case1_all_v12_epa_d2_sigb_1_a1_t1_uAF_uTF_M1_R5_iter100k/).

---

# Task 4. MIDUS τ-sweep `(τ ∈ {1.3, 1.5})`

`update_alpha`, `update_tau`, `alpha_fix`, `tau_fix` 를 환경변수로 override 가능하게 수정.

## 4.1 결과

| 지표 | τ=1.0 | τ=1.3 | τ=1.5 |
|---|---|---|---|
| **median K_+** | 4 | 4 | 4 |
| mean K_+ | 4.08 | 3.97 | 3.92 |
| max K_+ | 13 | 12 | 11 |
| split rate | 0.170 | 0.135 | 0.120 |
| **Pr(co ∈ [0.3, 0.7])** | **99.7%** | 93.9% | **80.9%** ↓ |
| within-pair ratio | 0.000 | 0.001 | **0.004** |
| PEAR | 0.010 | 0.007 | 0.010 |
| silhouette (b̂) | 0.420 | 0.425 | 0.416 |
| **P0 cluster sizes** | 43/8/16/1 | 44/8/15/1 | 42/9/16/1 |

## 4.2 핵심 관찰

1. **ambig zone 19%p 감소** (99.7% → 80.9%): τ↑ 가 PSM 분포를 점차 bimodal 쪽으로 밀어줌.
2. **median K_+ = 4 견고**: τ=1.5 까지 안전. simulation 의 τ=2 collapse 까지는 여유 있음.
3. **Cluster 내용 사실상 불변** (43/8/16/1 ↔ 42/9/16/1): τ 미세 조정은 PSM sharpness 만 조절. 임상 finding (8-item C2 inflammation+exec-function) 은 **τ 와 무관하게 robust**.
4. silhouette ~0.42 도 τ 와 무관 — b̂ 위치 자체는 LSIRM 측이 결정.

## 4.3 결론

τ=1.5 가 셋 중 가장 나음 (ambig zone 줄고 K=4 유지) 이지만, **τ 만 조정해서는 PSM contrast 를 본질적으로 못 올림**. paper 에 τ=1.0 또는 1.5 둘 중 하나 쓰면 됨.

---

# Task 5. PSM ambiguity 의 근본 원인 진단

User question: "K_+ posterior 가 안정적이고 다른 parameter 도 안정적인데, 왜 co-clustering 분포만 ambiguous 한가?"

## 5.1 K_+ stability ≠ partition stability

P=68 항목을 4 군집으로 나누는 partition 수: Stirling S(68, 4) ≈ 10⁴⁰. 체인이 1000 개 다른 4-cluster partition 을 균등 visit 해도 K_+ 는 항상 4 stable. 그러나 PSM Π[i,j] 는 **여러 partition 들의 평균이라 boundary 항목 쌍에서 0.4–0.7 로 떨어짐**. K_+ 는 *coarse statistic* 일 뿐.

## 5.2 진짜 원인: 데이터가 c 를 직접 관찰하지 않음

v12 likelihood 는 b 에 의존, c 에 무관:

```
Y_{ij} ~ f(η_{ij}),  η_{ij} = α_i − β_j − γ_l ‖a_i − b_j‖
```

c 는 **prior 만으로 식별**:

```
p_EPA(c | σ, b, α, τ) ∝ ∏_q [ Σ_r λ_{qr}(b, τ) · 1{c_r = c_q} / (q−1+α) + ... ]
```

τ=1, MIDUS b̂ silhouette 0.42 (인접 cluster 거리 ~1) 에서:

| 항목 쌍 | similarity λ |
|---|---|
| 같은 cluster (거리 ≈ 0.3) | exp(−0.09) ≈ 0.91 |
| 인접 cluster (거리 ≈ 1.0) | exp(−1.0) ≈ 0.37 |
| 먼 cluster (거리 ≈ 2.0) | exp(−4.0) ≈ 0.018 |

**within vs adjacent ratio = 2.5 배** — 단일 항목 Gibbs 가 boundary 항목을 swap 할 때 prior 가 충분히 강하게 막지 못함.

## 5.3 이것이 simulation 에서도 재현되는 이유

simulation v12_2 fix(1,1) long: cluster 거리 2 / within-sd 0.2 → 10 SD 분리. λ ratio 45배. 그런데도 co_contrast 0.055. **EPA prior 의 식별력이 본질적으로 soft** — 거리 1 이상의 항목에 대해 even modest similarity 를 부여하므로 boundary 항목 oscillation 을 prior 가 차단 못 함.

## 5.4 LSIRM 다른 parameter 는 왜 안정적인가

| Parameter | likelihood 와의 관계 | 사후 분포 |
|---|---|---|
| α, β, γ, b, a | likelihood 에 직접 진입 | 데이터로 직접 식별 → 안정 |
| **c** | **likelihood 와 무관, prior 만으로** | **prior bandwidth 가 결정 → 약함** |
| K_+ | c 의 coarse statistic | 안정 (mode 가 잡혀있는 통계) |

likelihood 가 직접 식별하는 모든 parameter 는 안정 수렴. **likelihood 가 간접적으로만 식별하는 c 는 prior 식별력 한계로 약하게 수렴**. v12 EPA architecture 의 inherent trade-off.

## 5.5 σ swap 이 PSM blur 의 원인이 *아닌* 이유

PSM 은 marginal posterior 를 추정 — σ 는 nuisance 로 적분됨. 빠른 σ mixing 은 estimator variance 줄이지 bias 안 만듦. swap rate 0.88 은 건강한 신호. → **PSM blur 는 사후분포 자체가 diffuse 하다는 뜻이지 sampler bug 아님**.

## 5.6 따라서

Stationary 진입과 sharp PSM 은 **별개**의 개념. stationary 는 도달했고, 그 stationary distribution 자체가 boundary 항목에 대해 의도적 불확실성을 담고 있는 것. sampler 정상.

**개선 방향은 둘 중 하나**:
- **(A)** 모델 architecture 변경 — c 에게 likelihood-type contribution 을 주는 prior 로 (= v11 NIW mixture 회귀)
- **(B)** 현 v12 결과 받아들이고 paper 에 PSM ambiguity 를 정직하게 framing

---

# Task 6. 다음 작업: v11 NIW mixture 재검토

## 6.1 동기

v11 NIW finite mixture 에서는 c_j 가 b_j 의 generative model 에 들어감:

```
b_j | c_j = l, μ_l, Σ_l ~ N_d(μ_l, Σ_l)
```

→ c_j 가 likelihood-type contribution 을 받음. 어떤 b_j 에 대해 "어떤 cluster 의 Gaussian density 가 가장 큰가" 가 직접 평가됨. **PSM contrast 가 v12 EPA 보다 sharper 할 잠재력**.

## 6.2 v11 lock-in 운영점 (현재 [data/MIDUS_5layered_result_v11.R](../data/MIDUS_5layered_result_v11.R))

```r
# Mixture 구조
d      = 2L
K_star = 10L          # overfitted prior
e0     = 0.05         # Dirichlet 농도

# NIW prior on (μ_l, Σ_l)
common_fmc_hyper <- list(
  e0     = 0.05,
  m0     = c(0, 0),
  kappa0 = 0.1,         # μ_l 정밀도
  nu0    = 100,         # Wishart df (Σ_l 강제고정)
  S0     = 6 * I_2,     # Wishart scale
  nu_S0  = NA           # S0 hyperprior 비활성
)

# Sampler / MCMC
n_split_merge     = 100L
b_prior_inflation = 1.0
common_mcmc <- list(d = 2, n_iter = 80000, burnin = 20000, thin = 10)
```

## 6.3 의문점 (user 제기)

**`nu0=100`, `S0=6I` 가 너무 큰가?** 0506 design log 의 정당화는:

- **target**: E[Σ_l] = S0 / (nu0 − d − 1) = 6/97 ≈ **0.062 · I** (per-dim sd ≈ 0.25)
- **rationale**: v10 biplot 의 cluster 가시반경 ~0.5 = 2σ 로 읽음
- **anti-feedback**: data-driven Σ_l 가 b 를 cluster 중심으로 piling 시키는 runaway shrinkage 차단

→ nu0=100 의 본질은 **"Σ_l 를 거의 고정시켜 data-feedback 차단"** 이라는 design choice. 이 choice 의 부작용 / 대안 검토는 다음 작업.

## 6.4 다음 단계 (TBD)

(A) 현재 v11 lock-in 으로 MIDUS 다시 돌려서 PSM contrast 가 v12 보다 정말 sharper 한지 확인  
(B) `(nu0, S0)` 재 calibration — 0.062·I target 이 v11 b-space 에 적합한지 재확인  
(C) Σ_l 고정 풀고 hyperprior 복귀 시 runaway shrinkage 가 다시 발생하는지 점검

---

# Task 7. v11 NIW mixture (nu0, S0) sweep 실행

User 결정: 옵션 (A) → (B) 순차 진행.

## 7.1 v11 LooseSigma — `nu0=10, S0=0.434` (target E[Σ_l]=0.062·I 보존)

[data/MIDUS_5layered_result_v11_looseSigma.R](../data/MIDUS_5layered_result_v11_looseSigma.R) 신규.

변경:
- `nu0`: 100 → **10** (concentration 풀어 data 가 Σ_l 끌고 가게)
- `S0`: 6·I → **0.434·I** (E[Σ_l] = S0/(nu0−d−1) 보존)
- `kappa0`: 0.1 (불변)
- `nu_S0`: NA (S0 hyperprior 비활성, 불변)

Sampler 안전장치 추가:
- `mcclust.ext::minbinder.ext` 의 `include.greedy = FALSE` (numerical edge case 우회)
- alt-partition 블록에 tryCatch 씌움
- `saveRDS(result, ...)` 를 alt-partition 블록 *이전* 으로 이동 (chain 손실 방지)

(첫 실행은 mcclust.ext greedy 의 `Error in d_mat[2, colind] : subscript out of bounds` 로 후처리 단계에서 fail 했고, 위 fix 후 재실행 성공.)

### 결과 (80k iter / 20k burnin / 10 thin → 6000 saved)

| 지표 | v12 EPA | **v11 LooseSigma** | 개선 |
|---|---|---|---|
| mean K_+ | 4.08 | 3.67 | 비슷 |
| **median K_+** | 4 | **4** | 동일 |
| sd(K_+) | 1.73 | **0.84** | 2배 tight |
| max K_+ | 13 | **7** | tail 짧음 |
| silhouette mean | 0.420 | 0.351 | 약간 ↓ |
| silhouette median | 0.468 | 0.471 | 동일 |
| **within-pair ratio** | **0.000** | **0.454** ✓ | 0 → moderate |
| Pr(co > 0.8) | 0.000 | **0.201** ✓ | 0 → 20% |
| Pr(co < 0.2) | 0.000 | **0.311** ✓ | 0 → 31% |
| Pr(co ∈ ambig) | **99.7%** | **20.8%** ↓ | 5배 감소 |
| **PEAR** | 0.010 | **0.424 (moderate)** ✓ | 40배 향상 |
| Dahl/Binder/VI | K=1 collapse | **K=3 일치** | collapse 해결 |

**무엇이 검증됨**:
- v11 NIW mixture 의 likelihood-type c-identification (`b_j ~ N(μ_l, Σ_l)`) 이 v12 EPA pairwise prior 보다 압도적으로 sharp 한 PSM 산출.
- runaway shrinkage 발생 안 함: empirical b̂ within-cluster sd 0.17/0.23 vs prior implied 0.25 → ratio 1.07–1.48 (data 와 prior 정합).

**Dahl/Binder/VI 가 한결같이 K=3** (loss 452.3) 으로 수렴 — v12 의 K=1 collapse 와 정반대. P0 (avg-link cut at K=4) 와 P1/P3/P5 (K=3) 의 ARI 0.91–0.96 로 한 cluster 만 split 차이.

### 임상적으로 의미있는 partition (Dahl K=3)

```
Cluster A (7):  B4BCRP, B4BFGN, B4BIL6  +  sgst_mixed_{all,nonswitch,switch}, sgst_reverse
                ─ inflammation 마커 3종 + executive function (set-shifting) 4종

Cluster B (19): B4Q1{CCC,D,FF,K,N,Z}, B4Q3{C,D,F,H,I,J,L,N,P,R,T}, catflu_intrusion
                ─ B4Q1/B4Q3 일부 (positive items 추정)

Cluster C (42): 나머지 B4Q1/B4Q3, B4BNE12, B4BSCL14, word-list cognition,
                backcount, numseries, sgst_normal, catflu_repetition
                ─ 우울/심리 + simple cognition + 잔여 inflammation 2종
```

**Cluster A** 는 v12 의 C2 (8-item) 와 거의 동일 — B4Q3B 빠지고 7개. v12 에서는 PEAR 0.01 로 신뢰도 매우 낮았는데, **v11 LooseSigma 에서는 PEAR 0.42 / within-pair 0.45 로 robust**.

**관찰된 단점**: biplot 에서 b̂ position 이 LSIRM-only 결과보다 살짝 압축됨 — mixture prior pull 의 부작용.

## 7.2 v11 LooseB — `nu0=10, S0=1.75` (target 0.25·I)

[data/MIDUS_5layered_result_v11_looseB.R](../data/MIDUS_5layered_result_v11_looseB.R) 신규.

동기: looseSigma 의 biplot 압축 해결 시도. Σ_l target 을 v12 EPA 에서 관찰된 LSIRM-natural within-cluster spread (~0.5 per dim) 에 맞춤.

변경: looseSigma 에서 `S0`: 0.434·I → **1.75·I** (target E[Σ_l] = 0.25·I).

### 결과 — K_+ 가 4 → 2 로 collapse

| 지표 | LooseSigma (target 0.062) | **LooseB (target 0.25)** |
|---|---|---|
| **median K_+** | **4** | **2** ← collapse |
| mean K_+ | 3.67 | 2.52 |
| sd K_+ | 0.84 | 0.68 |
| silhouette mean | 0.35 (K=4) | 0.79 (K=2 라 의미 다름) |
| within-pair ratio | 0.454 | 0.985 |
| PEAR | 0.42 | 0.70 |
| Pr(co ambig) | 20.8% | 5.8% |
| empirical b̂ within-sd | 0.17, 0.23 | **0.41, 0.40** |

**진단**: target Σ_l 이 0.25·I 면 cluster Gaussian 이 충분히 넓어서 *서로 다른 진짜 군집까지 흡수* 가능. data 가 prior 의 큰 target 을 받아들이고 K 를 줄여 cost minimize.

**S0 diagnostic**: prior implied sd 0.5 vs empirical 0.40, ratio 1.21–1.26 — runaway 는 안 났음. data 가 prior 와 매칭되었으나, K=4 구조가 K=2 의 wider Gaussian 안으로 흡수.

**시사점**: target 0.062 (looseSigma) 은 LSIRM-natural empirical 보다 작아 b 들을 끌어당기지만, target 0.25 는 너무 커서 cluster 정체성을 잃음. **사이 영역 (0.10–0.15) 에 진짜 sweet spot 있을 수 있음**. 다만 검증되지 않음.

## 7.3 결정: LooseSigma 를 v11 운영 setting 으로 lock-in

User 결정 (롤백): **`case1_all_v11_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_alpha1`** (LooseSigma) 가 현 시점의 가장 favorable setting.

근거:
- v12 EPA 대비 PSM contrast 극적 개선 (PEAR 0.01 → 0.42).
- median K_+ = 4 안정.
- runaway shrinkage 없음.
- 임상적으로 robust 한 7-item inflammation+exec-function cluster 추출.
- biplot 약간 왜곡되지만 LooseB (K=2 collapse) 보다 훨씬 나은 trade-off.

**현 reference 운영 파일**: [data/MIDUS_5layered_result_v11_looseSigma.R](../data/MIDUS_5layered_result_v11_looseSigma.R)

---

# Task 8. b geometry 왜곡 원인 분석 (LooseSigma 후속)

User 관찰: LooseSigma biplot 이 LSIRM-only 결과 대비 살짝 왜곡됨.

## 8.1 정량적 원인

LSIRM b 업데이트 MH ratio:

```
log r = log L(Y | b') - log L(Y | b) + log N(b'; μ_c, Σ_c) - log N(b; μ_c, Σ_c)
```

mixture prior precision = Σ_l⁻¹. LooseSigma 의 empirical Σ_l ≈ 0.17 → prior precision ≈ **6 per dim**. 비교: pure LSIRM 의 N(0, σ_b²=1) prior precision = **1 per dim**. → mixture prior 가 LSIRM-only 대비 **6배 강하게 b 를 cluster 중심으로 끌어당김**.

## 8.2 검토했지만 기각한 해결책

### `b_prior_inflation` (model coherence 위반)

v11 코드에 존재하는 `b_prior_inflation = α` knob (b MH 안에서만 Σ_l → α·Σ_l 로 부풀림). α > 1 이면 b geometry 회복 — **그러나 c, μ, Σ 업데이트는 여전히 원래 Σ_l 사용**.

→ 각 업데이트가 서로 다른 generative model 의 conditional 을 target. **joint posterior 가 정의되지 않음** (Bayesian 모델 정합성 위반). v11 lock-in design log 도 `sampler-level, NOT a model change` 로 명시하고 0506 에 1.0 으로 되돌림.

paper 결과로 안 씀.

## 8.3 검토한 model-coherent 옵션

| 옵션 | 변경 | 기대 효과 | 결과 |
|---|---|---|---|
| (A) LooseSigma | `nu0`: 100→10, `S0`: 6→0.434 | data-driven Σ_l, target 0.062 보존 | ✓ K=4, PEAR 0.42, biplot 약간 왜곡 |
| (B) LooseB | (A) 에서 `S0`: 0.434→1.75 (target 0.25) | b geometry 자연 회복 | ✗ K=2 collapse |
| (B') 사이 sweep | target 0.10–0.15 | (A) 와 (B) 사이 sweet spot | TBD |
| (C) Wishart hyperprior on S0 활성화 | `nu_S0`: NA → 5, Lambda_S0 추가 | data 가 S_0 자체를 calibrate | TBD |
| (D) `kappa0` 만 풀기 | (A) 에서 `kappa0`: 0.1 → 0.025 | μ_l prior 폭 ↑ (cluster center spread ±3) | TBD |

User 결정: 추가 sweep 보다 LooseSigma (A) 를 운영 setting 으로 받아들임. 향후 paper review 단계에서 필요하면 (B'), (C), (D) 재검토 가능.

---

# 참고 산출물 (v11 sweep 추가)

## v11 NIW mixture sweep
- `data/MIDUS_5layered_result_v11.R` (lock-in baseline, nu0=100, S0=6 — 0506 운영점에서 lock-in 해제)
- **`data/MIDUS_5layered_result_v11_looseSigma.R` ← 현 운영 setting**
- `data/MIDUS_5layered_result_v11_looseB.R` (collapse failure case)

## v11 결과 디렉토리
- `data/plot/case1_all_v11_fmc_d2_K10_e0.05_S0init_0.434_nu10_M100_alpha1/` ← **현 reference**
- `data/plot/case1_all_v11_fmc_d2_K10_e0.05_S0init_1.75_nu10_M100_alpha1/` (looseB, K=2 collapse)

---

# 참고 산출물

## 시뮬레이션
- `simulation_4_layered_v12_2.R` (신규)
- `plot/simulation_4_layered_v12_2_d2_sigb_1_expA/` (free α, τ)
- `plot/simulation_4_layered_v12_2_sens_a1.0_t1.0_fix/` (20k)
- `plot/simulation_4_layered_v12_2_sens_a1.0_t1.0_fix_long/` (100k)
- `plot/simulation_4_layered_v12_2_sens_a1.0_t2.0_fix_long/`
- `plot/simulation_4_layered_v12_2_sens_a1.0_t3.0_fix_long/`

## MIDUS
- `data/MIDUS_5layered_result_v12.R` (신규)
- `data/plot/case1_all_v12_epa_d2_sigb_1_a1_t1_uAF_uTF_M1_R5_iter100k/`
- `data/plot/case1_all_v12_epa_d2_sigb_1_a1_t1.3_uAF_uTF_M1_R5_iter100k/`
- `data/plot/case1_all_v12_epa_d2_sigb_1_a1_t1.5_uAF_uTF_M1_R5_iter100k/`

---

# Task 9. v11 simulation best-setting 정리 (소급 기록)

## 9.1 동기

MIDUS v11 LooseSigma lock-in (Task 7) 과 별개로, **simulation 4-layered v11** 의 운영점도 정리해 두기 위함. 0505 에 이미 튜닝 완료된 상태이나 별도 history 항목으로 기록되지 않아서 본 entry 로 소급 기록.

## 9.2 Simulation 환경

[simulation_4_layered_v11.R](../simulation_4_layered_v11.R):

- **데이터**: 4-layer (binary / continuous / count / ordinal), `n=150`, `P_total=120 (30+30+30+30)`
- **True K = 4**, isotropic clusters at `(±1, ±1)`
- **True within-cluster sd**: 0.10–0.20 (per-cluster: diag(0.15), diag(0.20), diag(0.10), diag(0.18))
- **Centers/sd ratio ≈ 5** — Gaussian-mixture-friendly separation, 단 size variation 은 남겨둠

## 9.3 Best 운영점: `S0_scale = 0.3`

결과 디렉토리: [plot/simulation_4_layered_v11_d2_S0_0.3/](../plot/simulation_4_layered_v11_d2_S0_0.3/)

**Hyperparameters** ([simulation_4_layered_v11.R:402-410](../simulation_4_layered_v11.R)):

```r
common_fmc_hyper <- list(
  e0     = 0.1,
  m0     = c(0, 0),
  kappa0 = 1.0,
  nu0    = d + 10,         # = 12
  S0     = 0.3 * diag(d)   # E[Σ_l] = 0.033, per-dim sd ≈ 0.18 (true ≈ 0.18 매칭)
)
common_mcmc     <- list(d = 2, n_iter = 50000, burnin = 20000, thin = 10)
fmc_warmup_iter <- burnin / 4    # = 5000
n_split_merge   <- 100
K_star          <- 10
```

LSIRM proposal SD 는 0503 v10 lock-in 그대로 사용.

## 9.4 성능 비교 (S0_scale sweep)

`fmc_recovery_summary.csv` 기준:

| 지표 | **S0=0.3 (best)** | S0=1.0 |
|---|---|---|
| median K_+ | 5 | 5 |
| ARI hclust | **0.860** | 0.805 |
| ARI Dahl | **0.878** | 0.824 |
| ARI Binder | **0.878** | 0.824 |
| ARI VI | **0.868** | 0.836 |
| Split rate | **3.89%** | 0.95% |
| Merge rate | **1.45%** | 0.29% |
| cor_dist_all | 0.502 | 0.497 |
| R_scale | 0.838 | 0.821 |

S0 를 truth 에 맞춤으로써 ARI ~0.05–0.06 pt 향상 + split/merge mixing 4배 개선.

## 9.5 튜닝 history (0505 시점 결정 사유, 코드 주석 + 본 history 통합)

1. **`S0_scale: 1.0 → 0.3`**
   - S0=1.0 → per-dim sd 0.33 (true 0.18 의 ~2배). NIW marginal 이 split 선호 → K_+ inflate.
   - S0=0.3 → per-dim sd 0.18 ≈ true 와 일치 → over-cluster 페널티 정확화.

2. **`kappa0: 1e-3 → 1.0`**
   - kappa0=1e-3 → empty cluster predictive sd `√(1/κ₀) ≈ 32` per dim → μ trace ±30~60 spike, split move likelihood inflate → over-cluster.
   - kappa0=1.0 → μ 를 m0=0 에 적당히 anchor (sd ≈ √(0.3/9) ≈ 0.18).

3. **`nu0: 4 시도 → d+10=12 복귀`**
   - nu0=4 → `E[Σ_l|S0=I] = I` (true 0.18·I 보다 ~6배 큼) → mega-cluster merge, K_+ 2–3 collapse.
   - nu0=12 → E[Σ_l] ≈ S0/9 로 적정 scale.

4. **`n_split_merge: 5 → 100`**
   - v11 은 b 가 random-walk MH 로 mixing 느림 → split-merge 가 partition 재배치 dominant mechanism.
   - per-attempt acceptance 가 v10 (12%/3%) 보다 훨씬 낮음 (0.8%/0.2%). 100회로 늘려서 sweep 당 accept 수 보강 (runtime 2-3배 비용 감수).

## 9.6 시뮬레이션 vs MIDUS lock-in 비교

| | Simulation (best) | MIDUS LooseSigma (Task 7 lock-in) |
|---|---|---|
| `S0` | 0.3·I | 0.434·I |
| `nu0` | d+10 = 12 | 10 |
| `kappa0` | 1.0 | 0.1 |
| `e0` | 0.1 | 0.05 |
| `K_star` | 10 | 10 |
| target `E[Σ_l]` | 0.033·I (sd 0.18) | 0.062·I (sd 0.25) |
| `n_iter / burnin / thin` | 50000 / 20000 / 10 | 80000 / 20000 / 10 |

차이 정당성:
- MIDUS 는 ground truth 모르고 b spread 가 더 큼 → prior 더 풀어둠 (kappa0↓, nu0↓, e0↓).
- Simulation 은 truth 가 isotropic 0.10–0.20 sd 로 잘 정의됨 → tight 하게 맞춤.

## 9.7 한계 (정직한 reporting)

- **Dahl/Binder/VI 모두 K_+=5**, true K=4 대비 +1 over-clustering. ARI 0.87대로 매우 높지만 K=4 정확 회복은 아님.
- `fmc_co_cluster_true_order.pdf` 에서 어느 true cluster 가 split 되었는지 확인 가능.
- **v11t** (Student-t likelihood, S0=0.3, ν_t=4) 실험은 [plot/simulation_4_layered_v11t_d2_S0_0.3_nu4/](../plot/simulation_4_layered_v11t_d2_S0_0.3_nu4/) 에 결과 미완 (true_positions.pdf 만 존재).

## 9.8 산출물

- [simulation_4_layered_v11.R](../simulation_4_layered_v11.R) — best 운영점 (S0=0.3)
- [plot/simulation_4_layered_v11_d2_S0_0.3/](../plot/simulation_4_layered_v11_d2_S0_0.3/) ← **현 reference**
- [plot/simulation_4_layered_v11_d2_S0_1/](../plot/simulation_4_layered_v11_d2_S0_1/) (이전 운영점, S0=1.0)
- [plot/simulation_4_layered_v11t_d2_S0_0.3_nu4/](../plot/simulation_4_layered_v11t_d2_S0_0.3_nu4/) (v11t 미완)
