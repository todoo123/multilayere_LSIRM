# 0427 — Bipartite SBM 구현 및 multilayered LSIRM v6 결과와 연결

## 오늘의 작업 (요약)

| # | 범주 | 산출 |
|---|---|---|
| 1 | 모델 | `bipartite_SBM.cpp` 신규 — Gamma block likelihood + Dirichlet/Cat/Gamma Gibbs + log-MH for κ |
| 2 | 래퍼 | `bipartite_SBM_cpp.R` 신규 — `bipartite_sbm()`, `build_distance_cube()`, `compute_sbm_icl()` |
| 3 | 파이프라인 | `MIDUS_5layered_result_v6.R` — 섹션 8b 추가 (LSIRM posterior distance trajectory → bipartite SBM) |
| 4 | 인터페이스 | LSIRM의 `result$a`, `result$b1..b5` 그대로 → 거리 cube `(n × p × m)` → SBM |
| 5 | 출력 | item co-clustering probability heatmap + cluster CSV + Λ posterior mean + `log κ` trace + **SBM biplot** |
| 6 | 모델 선택 | **`(Q, L)` grid search + ICL** — `Q_sbm` / `L_sbm`이 vector면 자동 grid search, ICL CSV/heatmap/line plot 산출 |

---

## 1. 목표

`paper/bipartite_SBM.tex` §1.2 (post-LSIRM bipartite SBM)을 실제 코드로 옮긴다. multilayered LSIRM v6의 **MCMC sample 전체**(평균이 아닌 trajectory)를 입력으로 받아 row/column joint clustering을 수행한다. 본 라운드에서는 LSIRM 결과를 한 번에 통째로 SBM에 흘려보내는 단방향 구조이며, 향후 LSIRM + SBM joint sampler로 확장할 수 있도록 한 sample 당 한 sweep이라는 인터페이스를 유지한다.

> 사용자 요청에 따라 **iter / burnin / thinning 옵션은 노출하지 않음** — LSIRM 단계에서 이미 thinning된 m개 sample을 그대로 m sweep로 소비.

---

## 2. 신규 파일

| 파일 | 역할 |
|---|---|
| `data/bipartite_SBM.cpp` | RcppArmadillo MCMC kernel — `compute_distance_cube`, `run_bipartite_sbm_cpp` |
| `data/bipartite_SBM_cpp.R` | R wrapper — `sourceCpp` 후 `bipartite_sbm(D_arr, Q, L, ...)` 와 `build_distance_cube(a_samps, b_samps_list)` 노출 |

`my_LSIRM_5layered_nonhierarchical_v6.cpp` / `_cpp_v6.R`와 동일한 컨벤션:
- `[[Rcpp::depends(RcppArmadillo)]]`, `using namespace arma`
- 래퍼는 `getwd()` 기준 `sourceCpp(...)` 호출 (data/에서 source)
- log-scale MH는 `R::dnorm(., μ, σ, 1)` (Gaussian on ξ-space, no Jacobian) — v6 cpp의 line 824-828 패턴과 동일

---

## 3. 모델 요약

### 3-1. Likelihood / Prior

| 모수 | 분포 | 비고 |
|---|---|---|
| $X^{(s)}_{ij} \mid z_i=q, w_j=l, \kappa, \lambda_{ql}$ | $\text{Gamma}(\kappa, \lambda_{ql})$ | $X^{(s)} = D^{(s)}$ — LSIRM s번째 sample의 거리행렬 |
| $\lambda_{ql} \mid \kappa$ | $\text{Gamma}(\kappa r,\; r \bar X^{(s)})$ | $r = 0.1$, $\bar X^{(s)}$는 slice별 평균 |
| $\pi$ | $\text{Dirichlet}(1, \ldots, 1)$ | $Q$차원 |
| $\rho$ | $\text{Dirichlet}(1, \ldots, 1)$ | $L$차원 |
| $z_i \mid \pi$ | $\text{Categorical}(\pi)$ | 0-based 내부 |
| $w_j \mid \rho$ | $\text{Categorical}(\rho)$ | 0-based 내부 |
| $\log \kappa$ | $\mathcal{N}(0, 4)$ | weakly informative |

### 3-2. 한 sweep (LSIRM sample s마다 1회)

| 단계 | 업데이트 | 규칙 |
|---|---|---|
| 0 | $\bar X^{(s)}$ | `mean(vectorise(D.slice(s)))` |
| 1 | $\pi \mid z$ | `Dirichlet(1 + n_q)` (independent Gamma + normalize) |
| 2 | $\rho \mid w$ | `Dirichlet(1 + m_l)` |
| 3 | $\lambda_{ql} \mid \cdot$ | `Gamma(κ(r + N_ql),  rX̄ + S_ql)` (Appendix §A.3 step 3) |
| 4 | $z_i \mid \cdot$ | log-weight $\log \pi_q + \sum_j [\kappa \log \lambda_{q,w_j} - \lambda_{q,w_j} X_{ij}]$ → log-sum-exp 정규화 |
| 5 | $w_j \mid \cdot$ | symmetric across columns |
| 6 | $\log \kappa$ | RW-MH (Gaussian prior on ξ, no Jacobian) |
| 7 | store | $(z, w, \pi, \rho, \Lambda, \log \kappa)$ → s번째 슬롯 |

### 3-3. 수치 안정성

- 모든 `log()` 호출은 `+EPS (1e-12)` floor 적용
- categorical 정규화: log-sum-exp 후 `R::runif(0,1)` cumulative draw
- Gibbs Λ draw가 0 근처면 EPS로 floor (다음 iteration의 `log Lambda`에서 NaN 방지)
- `D[i,j,s] <= 0`이면 R 래퍼에서 `1e-10` 으로 floor (LSIRM 위치는 연속이라 사실상 발생 X, 안전장치)

### 3-4. ICL 계산 (`compute_sbm_icl()` in `bipartite_SBM_cpp.R`)

paper §1.2.7 Algorithm 식을 그대로 구현.

$$
\mathrm{ICL}(Q, L) = \log p(D, \hat z, \hat w \mid \hat \Theta_{Q,L}, Q, L) - \frac{\nu_{Q,L}}{2} \log(mnp), \qquad \nu_{Q,L} = (Q-1) + (L-1) + QL + 1
$$

- $(\hat z^{(s)}, \hat w^{(s)})$: 각 LSIRM sample $s$의 SBM trajectory를 representative partition으로 사용 (per-s)
- $\hat \Theta = (\hat \pi, \hat \rho, \hat \Lambda, \hat \kappa)$: trajectory 사후 평균
- penalty의 sample size는 `m * n * p` (paper line 1033)

### 3-5. 미구현 (의도적)

- Label switching post-processing: 사용자 요청대로 사후 relabeling은 별도 처리 (현재는 raw posterior sample 저장; cluster mode는 trajectory 빈도 1위 라벨)
- iter / burnin / thinning: LSIRM 단계에 위임. 추후 joint LSIRM+SBM Gibbs scheme에서 통합 예정

---

## 4. MIDUS 파이프라인 통합

### 4-1. 위치
`MIDUS_5layered_result_v6.R`의 기존 §8 (multilayered $b$ → k-means) 직후에 **§8b** 신규 삽입. §9 (unilayered LSIRM 비교)은 그대로 유지.

### 4-2. 흐름

```r
# 1. distance trajectory 빌드 (한 번만)
b_samps_list <- list(result$b1, result$b2, result$b3, result$b4, result$b5)
b_samps_list <- b_samps_list[ sapply(b_samps_list, function(x)
                              !is.null(x) && length(dim(x))==3 && dim(x)[2]>0) ]
D_cube <- build_distance_cube(result$a, b_samps_list)   # (n × p × m)

# 2. (Q, L) 설정 — scalar = 단일 fit, vector = grid search
Q_sbm <- 3                # or e.g. 2:5
L_sbm <- 3                # or e.g. 2:5

# 3. expand.grid 기반 자동 분기
grid_QL <- expand.grid(Q = Q_sbm, L = L_sbm)
for (gi in seq_len(nrow(grid_QL))) {
  fit_g <- bipartite_sbm(D_cube, Q = grid_QL$Q[gi], L = grid_QL$L[gi], ...)
  icl_g <- compute_sbm_icl(fit_g, D_cube)
  # save: result.rds, item/respondent_clusters.csv, co_cluster_w.csv,
  #       Lambda_postmean.csv, co_cluster_w.pdf, trace_log_kappa.pdf,
  #       biplot.pdf  (모두 _Q{Q}_L{L} suffix)
}

# 4. grid 종합 — ICL CSV + plot, best (Q, L) 보고
```

### 4-3. 산출 (`case_plot_dir` 아래)

per-(Q, L) artifacts (총 7종, suffix `_Q{Q}_L{L}`):

| 파일 | 내용 |
|---|---|
| `*_sbm_Q{Q}_L{L}_result.rds` | `fit_g` 전체 (z, w, π, ρ, Λ, log κ trajectory) |
| `*_sbm_Q{Q}_L{L}_item_clusters.csv` | `item × sbm_cluster` mode |
| `*_sbm_Q{Q}_L{L}_respondent_clusters.csv` | `respondent × sbm_cluster` mode |
| `*_sbm_Q{Q}_L{L}_co_cluster_w.csv` | item × item co-cluster 확률 행렬 |
| `*_sbm_Q{Q}_L{L}_Lambda_postmean.csv` | Λ 사후 평균 (Q × L) |
| `*_sbm_Q{Q}_L{L}_co_cluster_w.pdf` | cluster-순으로 reorder된 heatmap |
| `*_sbm_Q{Q}_L{L}_trace_log_kappa.pdf` | `log κ` trace |
| `*_sbm_Q{Q}_L{L}_biplot.pdf` | **respondent (circle, alpha) + item (square)** 두 cluster 색으로 표시한 latent space biplot |

grid-level artifacts (1종 + ICL plot):

| 파일 | 내용 |
|---|---|
| `*_sbm_grid_ICL.csv` | `Q, L, icl, cll, penalty, nu, acc_log_kappa, kappa_postmean` 한 행/조합 |
| `*_sbm_ICL.pdf` | grid 길이 ≥ 2일 때만 — Q∧L 모두 vector면 heatmap (ICL 값 셀에 표기, best (Q,L)에 검은 박스), 한쪽만 vector면 line plot |

### 4-4. 메모리

`n ≈ 220`, `p ≈ 68` (case1_all: 20 + 5 + 15 + 28 = 68), `m ≈ 8000` 기준 `D_cube`는 약 1 GB. SBM 종료 후 `rm(D_cube); gc()`로 §9 unilayered 블록 진입 전 해제. `n` 또는 `p`가 더 큰 시나리오에서는 향후 streaming 변형(`a_arr`, `b_arr`을 cpp에 직접 전달, slice별 즉석 거리 계산)으로 교체 예정.

---

## 5. 검증 (smoke tests)

| # | 항목 | 결과 |
|---|---|---|
| 1 | `Rcpp::sourceCpp("data/bipartite_SBM.cpp")` | clean compile (RcppArmadillo) |
| 2 | toy `D ~ array(rgamma(8*6*30, 2, 1), c(8,6,30))`, `bipartite_sbm(D, 2, 2)` | finite samples, `acc_log_kappa ∈ [0.2, 0.7]` |
| 3 | `co_w` symmetry | `all(co_w == t(co_w))` TRUE |
| 4 | dim consistency | `dim(D_cube) == c(n, p, m)`, `dim(fit$Lambda) == c(Q, L, m)` |
| 5 | **planted 2×2 block 회복** (n=30, p=20, m=200, true κ=3, λ_diag=2 / λ_off=0.5) | within-block co-cluster prob = **1.000**, between = **0.000**, posterior κ̂ = 3.13 |
| 6 | **ICL 모형 선택**: 같은 planted 데이터에서 `Q ∈ {1,2,3} × L ∈ {1,2,3}` grid 검색 | argmax ICL = **(2, 2)** ✅ — true (2, 2) 정확 회복 |
| 7 | scalar `(Q_sbm=3, L_sbm=3)` → `expand.grid` nrow=1 → 단일 fit 경로 | OK (분기 없이 동일 코드 경로) |
| 8 | biplot helper (`make_sbm_biplot`) toy 입력 | PDF 정상 생성 (≥6KB) |

end-to-end run (전체 MIDUS 파이프라인)은 LSIRM MCMC가 수 시간 소요되므로 사용자 환경에서 별도 실행. §8b 단독 부분만 toy 입력으로 사전 검증 완료.

---

## 파일 변경 요약

### 신규
- `data/bipartite_SBM.cpp`
- `data/bipartite_SBM_cpp.R`

### 수정
- `data/MIDUS_5layered_result_v6.R` — §8b 신규 삽입 (`bipartite_SBM_cpp.R` source, `build_distance_cube`, `bipartite_sbm`, item co-clustering / Λ 사후 요약 / heatmap 저장)

### 문서
- `history/0427_history.md` — 본 문서

---

## 6. v7 — LSIRM + Bipartite SBM joint MCMC

### 6-1. 동기

§5에서 post-LSIRM SBM이 LSIRM sample 1개 = SBM sweep 1번이라 mixing이 부족했음. v7에서는 **하나의 MCMC chain** 안에서 LSIRM과 SBM을 함께 돌려, 표준 burnin / thin이 두 모듈에 동시에 적용되도록 한다. 향후 fully joint LSIRM+SBM model로 확장 가능한 구조 (현재는 LSIRM이 SBM에 forward-only로 정보 공급).

### 6-2. 신규 파일

| 파일 | 역할 |
|---|---|
| `data/my_LSIRM_SBM_v7.cpp` | RcppArmadillo MCMC kernel — `run_lsirm_sbm_v7_cpp`. LSIRM v6 update 루틴을 그대로 가져온 뒤, 각 iteration 끝에 **SBM 한 sweep**(π → ρ → Λ → z → w → log κ_SBM) 추가. SBM은 현재 `D_ij = ‖a_i − b_concat_j‖`를 입력으로 사용 (b_concat = rbind(b1..b5)) |
| `data/my_LSIRM_SBM_cpp_v7.R` | R wrapper `lsirm_sbm_v7_cpp(...)`. **LSIRM/SBM 하이퍼파라미터·proposal SD·init을 별도 리스트로 분리** (`lsirm_hyper`, `sbm_hyper`, `lsirm_prop_sd`, `sbm_prop_sd`, `lsirm_init`, `sbm_init`). MCMC 종료 후 LSIRM 위치에만 Procrustes 매칭 적용 (orthogonal+translation은 거리 보존 → SBM cluster에 영향 없음) |
| `data/MIDUS_5layered_result_v7.R` | MIDUS Wave 2 + Refresher 1 데이터로 v7 joint chain을 돌리는 executor |

### 6-3. 모델 구조

한 iteration 내 업데이트 순서:

```
[A] LSIRM (v6 그대로)
    lambda2 → α1..5 → β1..3 → log γ1..5
    → a → b1..b5 → β4_thr → β5_thr
    → log κ_j (per-item NB) → Gibbs scalars (σ², τ²)

[B] SBM (현재 a, b1..b5로부터 D 즉석 계산)
    π → ρ → Λ → z → w → log κ_SBM
```

`burnin`, `thin`이 LSIRM trajectory와 SBM trajectory에 **공통 적용**되므로, 사용자가 `n_iter=100k, burnin=20k, thin=10`으로 주면 LSIRM과 SBM 모두 8000개 후처리 sample을 얻는다. mixing 문제 (§5 1순위 원인)가 자연 해결.

### 6-4. 출력 구조 (단일 List, flat with `sbm_` prefix)

LSIRM 측 키는 v6과 동일 — 기존 plotting 코드 재사용 가능:
`alpha1..5, beta1..3, beta4, beta5, log_gamma1..5, log_kappa, a, b1..b5, sigma_alpha*_sq, tau_beta*_sq, sigma0_sq, lambda2, lambda2_mean, lambda2_postmean, accept`.

SBM 측 키 (신규):
| 키 | 형태 | 의미 |
|---|---|---|
| `sbm_z` | umat (n_save × n) | respondent cluster trajectory (1-based) |
| `sbm_w` | umat (n_save × P_total) | item cluster trajectory (1-based) |
| `sbm_pi` | mat (n_save × Q) | row cluster proportions |
| `sbm_rho` | mat (n_save × L) | column cluster proportions |
| `sbm_Lambda` | cube (Q × L × n_save) | block intensity λ_ql |
| `sbm_log_kappa` | vec (n_save) | SBM Gamma shape |
| `sbm_Xbar` | vec (n_save) | iteration-별 평균 거리 |
| `sbm_Q`, `sbm_L` | int | cluster 개수 |
| `accept$sbm_log_kappa` | double | SBM log κ MH acceptance |

### 6-5. Traceplot 카테고리 (`MIDUS_5layered_result_v7.R`)

LSIRM 쪽 (v6와 동일 set):
- `*_trace_a.pdf`, `*_trace_b{1..5}.pdf`, `*_trace_alpha{1..5}.pdf`, `*_trace_beta{1..3}.pdf`
- `*_trace_beta{4,5}_thr.pdf`, `*_trace_lsirm_kappa_per_item.pdf`
- `*_trace_lsirm_extra.pdf` — sigma0_sq, gamma1..5, sigma_alpha*, lambda2_mean

**SBM 쪽 (신규)**:

| 파일 | 내용 |
|---|---|
| `*_sbm_trace_pi.pdf` | π[1..Q] 각 trajectory, mean ± 95% CI |
| `*_sbm_trace_rho.pdf` | ρ[1..L] 각 trajectory |
| `*_sbm_trace_Lambda.pdf` | Λ[q, l] Q × L 패널 |
| `*_sbm_trace_kappa_Xbar.pdf` | κ_SBM trace + Xbar trace |
| `*_sbm_trace_cluster_sizes.pdf` | n_q^(s), m_l^(s) 시간 추이 (overplot) |
| `*_sbm_membership_heatmap_w.pdf` | iter × item heatmap, color = cluster id (label switching 즉시 가시화) |
| `*_sbm_membership_heatmap_z.pdf` | iter × respondent heatmap (n>80이면 80명 무작위 sample) |
| `*_sbm_trace_w_individual.pdf` | item 12개 step trace |
| `*_sbm_trace_z_individual.pdf` | respondent 12명 step trace |
| `*_sbm_co_cluster_w.pdf` | item 사후 co-clustering 확률 heatmap |
| `*_sbm_biplot.pdf` | A_hat / B_hat (Procrustes-aligned posterior mean) + cluster mode 색칠 |
| `*_sbm_ICL.csv` | (Q, L, cll, penalty, ICL, ν) — 단일 fit 기준 |

### 6-6. Hyperparameter 분리 예시 (v7 wrapper signature)

```r
lsirm_sbm_v7_cpp(
  Y_bin, Y_con, Y_cnt, Y_ord1, Y_ord2,
  Q = 3, L = 3,
  d = 2, n_iter = 100000, burnin = 20000, thin = 10, nu2 = 4,

  lsirm_hyper   = list(a_sigma=1, b_sigma=1, ..., mu_log_kappa=0, sd_log_kappa=0.1, ...),
  sbm_hyper     = list(r=0.1, mu_log_kappa=0, sd_log_kappa=2),

  lsirm_prop_sd = list(alpha1=0.5, ..., log_kappa=0.4),
  sbm_prop_sd   = list(log_kappa=0.1),

  lsirm_init    = NULL,   # default: rnorm(...) per v6 spec
  sbm_init      = NULL    # default: random labels, uniform π/ρ, Λ=1, log κ=0
)
```

### 6-7. 검증 (smoke test)

| # | 항목 | 결과 |
|---|---|---|
| 1 | `Rcpp::sourceCpp("data/my_LSIRM_SBM_v7.cpp")` | clean compile |
| 2 | tiny synthetic (n=30, P_total=19, Q=L=2, n_iter=600, burnin=200, thin=4) end-to-end | 0.4초 완료, NaN 없음 |
| 3 | dim 일관성 | `a` (100×30×2), `b1..b5` (100×P_l×2), `sbm_z` (100×30), `sbm_w` (100×19), `sbm_Lambda` (2×2×100), `sbm_log_kappa` length 100 |
| 4 | label range | `sbm_z ∈ {1,2}`, `sbm_w ∈ {1,2}` (1-based 변환 OK) |
| 5 | acceptance | LSIRM α/β/a/b 모두 0.85+ (toy 단위라 큼), SBM log κ ≈ 0.25 (정상 범위) |
| 6 | plotting paths | `trace_pi`, `cluster_sizes`, `membership_w`, biplot 모두 PDF 정상 생성 |
| 7 | `compute_sbm_icl()` 호환 | v7 결과를 list로 감싸서 그대로 호출 가능 (`fit_for_icl <- list(z=..., w=..., pi=..., rho=..., Lambda=..., log_kappa=..., Q=..., L=...)`) |

### 6-8. v6 ↔ v7 호환

- v7의 LSIRM 부분 keys는 v6 wrapper와 100% 동일 → v6의 `make_traceplots`, `make_biplot` 같은 plotting helper를 그대로 재사용 가능
- v6에서 사용하는 `bipartite_SBM_cpp.R`의 `build_distance_cube`, `compute_sbm_icl`은 v7 결과에도 동일하게 동작 → ICL 계산은 외부 helper로 통일

### 6-9. 미구현 (의도적)

- (Q, L) grid search: v7 자체는 단일 (Q, L) 만 받음. grid는 executor R 단계에서 outer loop으로 처리 (chain 한 번 = 비싸므로 사용자 명시적 선택)
- Joint structure에서 cluster membership이 LSIRM positions에 영향을 주는 fully-joint prior: 본 라운드 범위 밖

---

## 파일 변경 요약 (v7 추가)

### 신규
- `data/my_LSIRM_SBM_v7.cpp`
- `data/my_LSIRM_SBM_cpp_v7.R`
- `data/MIDUS_5layered_result_v7.R`
