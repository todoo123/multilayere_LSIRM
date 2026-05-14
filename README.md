# joint_LSIRM — Multilayered Latent Space Item Response Model for MIDUS

다층(5-layer) LSIRM 으로 MIDUS 데이터(Wave 2 + Refresher 1)의 혼합형 문항
(binary / count / ordinal / continuous)을 공통 잠재공간에 임베딩하고,
문항별 사후분포 간 Bhattacharyya 거리에 spectral clustering 을 적용해
문항을 군집화한다.

최종 모델은 **v5** (`R/` 디렉토리의 파이프라인)로 고정되었다. v6~v14 및
각종 실험·시뮬레이션은 `archive/` 에 보존되어 있다.

## 디렉토리 구조

```
R/        최종 v5 파이프라인 (00 → 03 순서로 실행)
src/      Rcpp/Armadillo C++ 소스 (LSIRM Gibbs sampler)
data/
  raw/        MIDUS 원자료 (ICPSR .rda, git 미포함)
  processed/  전처리 산출물 (git 미포함)
results/v5/   최종 모델 fit + clustering 산출물
docs/
  history/        개발 일지
  meetings/       미팅 자료
  modeling_paper/ 모델 수식 문서 (.tex)
reference/    참고 논문 PDF
archive/      비최종 버전 (v3~v4, v6~v14), 실험, 시뮬레이션, legacy
```

## 파이프라인

`R/` 의 스크립트는 번호 순서대로 의존한다. 진입점은 `R/03_run_and_cluster.R`.

| 파일 | 역할 |
|------|------|
| `R/00_preprocess_wave2.R`     | MIDUS Wave 2 전처리 (`data/raw/MIDUS_2/`) |
| `R/01_preprocess_refresher.R` | MIDUS Refresher 1 전처리 (`data/raw/MIDUS_refresher_1/`) |
| `R/02_model_lsirm.R`          | 5-layer LSIRM Gibbs sampler R 래퍼 (`src/lsirm_5layered.cpp` 컴파일) |
| `R/utils.R`                   | 공통 헬퍼 |
| `R/03_run_and_cluster.R`      | **메인** — 데이터 결합 → MCMC → traceplot/biplot → BC spectral clustering |

## 재현 방법

1. `data/raw/` 아래에 MIDUS ICPSR 원자료를 배치한다
   (`MIDUS_2/`, `MIDUS_refresher_1/` — 각 `ICPSR_*/DS0001/*.rda`).
2. **저장소 루트에서** 메인 스크립트를 실행한다:
   ```r
   setwd("/path/to/joint_LSIRM")
   source("R/03_run_and_cluster.R")
   ```
   모든 경로는 `getwd()` 기준 상대경로로 해석된다.
3. 산출물(traceplot, biplot, BC heatmap, 군집 membership CSV, `*_result.rds`)은
   `results/v5/<case>_<run_label>/` 에 저장된다.

필요 R 패키지: `dplyr`, `purrr`, `Rcpp`, `RcppArmadillo`, `vegan`, `MASS`,
`cluster`, `grDevices`.
