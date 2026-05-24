# 0525 v5 — per-item kappa + MCMC 튜닝 + MIDUS 실데이터 재분석 + manuscript 작성

(작업 기간 0522–0525)

## 1. Motivation

v5(`src/lsirm_5layered.cpp`)의 카운트(음이항) 레이어가 **단일 전역 $\kappa$** 만 추정하던 것을
**문항별 $\kappa_j$** 로 확장하고, ordinal 레이어 수렴 문제를 잡은 뒤 MIDUS(Wave2+Refresher1)
실데이터를 재적합하여 manuscript 실데이터 절을 완성하는 것이 목표.

## 2. 작업 흐름

### 2.1 per-item kappa 구현 — **완료**
- 출처: `archive/model_versions/my_LSIRM_FMC_v14.cpp` (line 14 "per-item kappa") 로직 이식.
- `src/lsirm_5layered.cpp`:
  - `log_kappa`/`kappa` → 길이 $P_3$ 벡터, `store_log_kappa` → `(n_save × P3)` 행렬,
    `acc_log_kappa` → 길이 $P_3$ 벡터.
  - 음이항 우도 5곳(α₃·β₃·γ₃·공유 $a_i$·b₃)에서 `size = 1/kappa(j)` 로 문항별 적용.
  - kappa MH 를 문항별 루프(`for j<P3`)로 교체(해당 문항 열만 평가).
- `R/02_model_lsirm.R`: 기본 init `log_kappa = rep(0, P3)`.
- `R/03_run_and_cluster.R`: 문항별 κ traceplot PDF(`*_trace_kappa_per_item.pdf`).
- 컴파일/실행 환경: **시스템 R 4.4.1** (`/Library/Frameworks`) + `/opt/homebrew/bin` g++-14.
  conda `mlpractice` R 은 sourceCpp 시 hang(커널 I/O 대기) → 사용 금지.

### 2.2 자유 κ 1차 run → 사용자 결정: κ 를 1 근방으로 정규화
- 자유 산포 run 에서 $\hat\kappa_j$ 가 0.034–5.50 (164배 차이)로 문항별 과대산포 차이가 큼을 확인.
- 그러나 사용자는 카운트 분산을 $\kappa$ 만으로 설명하는 것을 막기 위해 **prior 로 κ 를 1 근방에 강하게 정규화**하기로 결정.

### 2.3 κ prior/proposal 튜닝 — **완료**
- 사용자가 `sd_log_kappa=0.001` 로 설정했으나 proposal(0.4)이 prior 폭의 400배라
  **MH 채택률 0.003 으로 붕괴**, κ 가 init=1 에 갇힘(사실상 per-item κ 무력화).
- 튜닝: `common_hyper$sd_log_kappa` 0.001→**0.01**, `common_prop_sd$log_kappa` 0.4→**0.02**.
- 결과: κ 채택률 **~0.50**, $\hat\kappa_j$ 0.99–1.00(문항 간 sd 0.003) — 의도대로 1 근방 정규화 + 정상 혼합.

### 2.4 ordinal 수렴 fix
- `common_prop_sd$log_gamma4` 0.2→**0.1** (순서형 거리가중 γ₄ random-walk step 축소).
- GRM 역치 β₄ 는 기존부터 **성분별(component-wise) MH** (v5 이후 전 구현체 동일 확인).

### 2.5 실데이터 재적합(n_iter=100k) + 결과 추출

## 3. 핵심 결과 (최종 튜닝 run)

데이터: **N=221** (Wave2 125 + Refresher1 96), **P=68** (bin 20 + con 5 + cnt 15 + ord 28).
MCMC 100,000 / burnin 20,000 / thin 10 → 8,000 표본.

### 수렴
- κ 채택률 0.50, $\hat\kappa_j$ 0.99–1.00 (sd 0.003)
- ordinal β₄ threshold ESS 중앙값 363(채택 0.59); γ₄ ESS ~90(채택 0.16)
- 거리가중 γ 사후평균: **순서형 1.31 > 카운트 0.54 ≈ 이진 0.48 > 연속 0.20**

### 군집 (BC + NJW 스펙트럴, 문항 68개)
- 평균 silhouette: K2 0.360 / K3 0.332 / **K4 0.404 / K5 0.423**
- nesting(refinement index): 인접 K 쌍 모두 0.84–0.91; **K2→4 0.91, K4→5 0.91** 가장 깔끔(nested refinement)
- **K=4 해석**:
  - C3 (n=19, sil 0.48): 핵심 정동(CES-D 우울 + MASQ 일반고통), 생체지표 0개 — 가장 깨끗
  - C4 (n=24, sil 0.44): 불안 각성(MASQ AA) + 염증(NE12·CRP)
  - C1 (n=10, sil 0.35): 실행·전환 인지(SGST) + 염증(IL-6·피브리노겐)
  - C2 (n=15, sil 0.28): 언어·일화기억(단어목록·유창성) + 코르티솔 — 경계 무름

### 문헌 정합성 (실제 검증 + 인용 확보)
- 핵심 정동–염증 분리: symptom-specificity \citep{Milaneschi2021}
- CRP–불안/스트레스(C4): GAD 메타분석 \citep{Costello2019}, UKB·NESDA \citep{Kennedy2021,Vogelzangs2013}
  (단 BMI·우울 보정 시 약화 → 불안 특이적 아님)
- IL-6·피브리노겐–실행기능/처리속도(C1): Edinburgh \citep{Rafnsson2007}, IL-6 메타 \citep{Bradburn2018},
  영역특이성 \citep{Lin2018}, MIDUS \citep{Marsland2015}, allostatic \citep{Perlman2022}
- IL-6–피로·무기력(anergia) \citep{Foley2024}; 면역-대사형 우울 \citep{Penninx2024,Lamers2010,Lamers2013}

## 4. manuscript / 산출물 작업
- **부록 B(Variable descriptions)**: 68문항 표(변수명·설명·자료형)를 부록으로 이동; 부록 A/B 표제.
- **데이터 인용**: ICPSR 6개 데이터셋 + MIDUS 프로젝트 개요(Brim2004, Radler2014).
- **시뮬레이션 표**: CI coverage+width 를 한 표로 병합(셀 = 커버리지(폭)); 캡션 전역 가운데 정렬.
- **레이어3 수식**: 우도·prior(#10)·업데이트 규칙을 per-item $\kappa_j$ 로 수정.
- **\section{분석 결과}**: `docs/분석결과.md` 내용을 LaTeX 로 병합(\citep), 전부 **습니다체**.
  - 표: silhouette(`tab:rda-sil`), **nesting/refinement(`tab:rda-nest`)**, K=4 구성(`tab:rda-k4`)
  - 그림: 군집 biplot + BC 유사도 히트맵(`fig:bc-cluster`), K=2–5 grid(`fig:bc-grid`)
    — 이미지 파일은 사용자가 직접 업로드(`KK4_bc_cluster_biplot.pdf`, `KK4_bc_heatmap.pdf`, `bc_clusters_grid.pdf`)
  - refinement index 정의/해석 보강: $\text{refinement}=\frac{1}{P}\sum_c \max_r n_{rc}$, 완전 nesting=1.0
- **BibTeX 13건**(Overleaf .bib 용) 별도 제공 — 키가 menuscript \citep 와 일치.
  ※ 분석결과.md ref[4]는 1저자가 Milaneschi 가 아니라 **Penninx** 임을 확인해 `Penninx2024` 로 정정.

## 5. Git / 저장소
- `.gitignore`: `results/**/*_result.rds`(172MB, GitHub 100MB 초과) + `results/**/.DS_Store` 제외.
  기존 `!results/**` 가 `*.rds` 를 덮어써서 대용량 rds 가 추적되던 문제 해결.
- 미push placeholder 커밋(".") 정리 후 깨끗한 커밋으로 재구성, **origin/main push 완료**(89949f5).
- 대용량 `result.rds` 는 로컬에만 보존(공유하려면 Git LFS 필요).

## 6. Follow-ups
- κ 자유 산포(사전분산 완화) 민감도 분석 — 잠재기하 영향 확인.
- γ₄ ESS(~90) 보강: 적응적 제안 또는 추가 thinning.
- 응답자 잠재위치 $a_i$ 기반 표현형(응답자 수준) 군집화 — 향후 확장.
- manuscript figure 이미지 업로드 후 컴파일 확인.
