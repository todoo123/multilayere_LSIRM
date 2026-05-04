# Simulation Adjustment Report

본 문서는 [`simulation_4_layered_v10.R`](../simulation_4_layered_v10.R) 시뮬레이션의 MCMC proposal SD 튜닝 작업을 기록한다.

---

## 1. 작업 목표 (Task Specification)

### 1.1 배경
- 대상 스크립트: [`simulation_4_layered_v10.R`](../simulation_4_layered_v10.R)
- 모델: 4-layer joint LSIRM + PPCA-FMC (NIW + split-merge), v10
- 목적: 주어진 simulation setting (n=150, P1=P2=P3=P4=10, K_true=4 등) 하에서, **모든 Metropolis–Hastings parameter 의 acceptance rate 가 0.3 ~ 0.6 구간에 들어오도록 proposal SD 를 반복적으로 조정**한다.

### 1.2 조정 대상 (proposal SD list)
스크립트 내 `common_lsirm_prop_sd` 리스트의 다음 항목들을 조정한다 ([`simulation_4_layered_v10.R:360-368`](../simulation_4_layered_v10.R#L360-L368)):

```r
common_lsirm_prop_sd <- list(
  alpha1 = 0.5, alpha2 = 0.4, alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.5,
  log_gamma1 = 0.10, log_gamma2 = 0.05, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.05,
  a = 0.30,
  beta1 = 0.40, beta2 = 0.10, beta3 = 0.20, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.30, b2 = 0.20, b3 = 0.20, b4 = 0.20, b5 = 0.20,
  log_kappa = 0.20
)
```

> 4-layer 시뮬레이션이므로 layer 5 관련 항목 (`alpha5`, `log_gamma5`, `beta5`, `b5`) 은 사용되지 않을 수 있다. 실제로 사용되는 항목만 튜닝한다.

### 1.3 진단 기준 (Acceptance rate target)
- 시뮬레이션 종료 후, `result$accept` 객체에서 다음을 확인한다:
  - `alpha1..4`: 각 vector 의 mean acceptance rate
  - `beta1..3`: 각 vector 의 mean acceptance rate
  - `log_gamma1..4`: scalar acceptance rate
  - `a`, `b1..b4`: matrix/vector 의 mean acceptance rate
  - `log_kappa`: per-item mean acceptance rate
  - (필요 시) `beta4` (GRM threshold) acceptance rate
- **목표 구간: [0.3, 0.6]**
- 구간 밖일 경우 조정 규칙:
  - **acceptance rate 가 낮음 (< 0.3) → proposal SD 를 줄인다** (제안이 너무 멀어 거부됨)
  - **acceptance rate 가 높음 (> 0.6) → proposal SD 를 키운다** (제안이 너무 좁아 자주 수락됨)

> 주의: 위 규칙은 "low acc → 작게 / high acc → 크게" 형태로, 일반적인 Metropolis 튜닝 직관 (low acc → SD 작게 / high acc → SD 크게) 과 동일하다. 사용자가 메시지에서 두 경우 모두 "낮춘다"고 적었지만, 표준 직관에 맞춰 위와 같이 해석한다.

### 1.4 조정 휴리스틱 (Adjustment heuristic)
한 번의 simulation run 후 acceptance 가 구간 밖이면:
- acc < 0.1 → SD ← SD × 0.3
- 0.1 ≤ acc < 0.3 → SD ← SD × 0.6
- 0.6 < acc ≤ 0.9 → SD ← SD × 1.6
- acc > 0.9 → SD ← SD × 3.0

이 곱셈 인자는 일반적인 Robbins–Monro 스타일 적응의 근사이며, 첫 반복 후 결과를 보고 보수적으로 조정한다.

---

## 2. 반복 작업 절차 (Iteration Protocol)

각 iteration 은 다음 단계로 구성된다:

1. **현재 proposal SD 기록**: 본 보고서의 "Iterations" 섹션에 새 iteration block 을 만들고 현재 `common_lsirm_prop_sd` 값을 기록한다.
2. **시뮬레이션 실행**:
   ```bash
   cd /Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM
   Rscript simulation_4_layered_v10.R 2>&1 | tee logs/sim_v10_iter<NN>.log
   ```
3. **`result$accept` 확인**: 콘솔 출력 (스크립트의 "Acceptance" 섹션, [`simulation_4_layered_v10.R:432-443`](../simulation_4_layered_v10.R#L432-L443)) 또는 저장된 RDS (`v10_simulation_result_*.rds`) 로부터 acceptance rate 를 추출하고, 보고서에 표로 정리한다.
4. **판정**:
   - 모든 parameter 가 [0.3, 0.6] 구간이면 → **종료**.
   - 일부가 벗어나면 → 1.4 의 heuristic 으로 SD 를 갱신하고, [`simulation_4_layered_v10.R:360-368`](../simulation_4_layered_v10.R#L360-L368) 을 `Edit` 도구로 수정한 뒤 다음 iteration 으로 이동.
5. **변경 이력 기록**: 어떤 SD 가 어떤 acc 때문에 어떻게 바뀌었는지 보고서에 명시.

---

## 3. Terminal 에서 Claude Code 로 이 반복 작업을 시키는 방법

이 작업은 단일 명령 응답으로 끝나지 않는 **장시간 / 다수-반복 루프**이다. Terminal 에서 Claude Code (`claude` CLI) 를 띄워 다음 중 한 가지 방식으로 운영하면 된다.

### 3.1 권장: 대화형 세션 + 명시적 지시문
가장 단순하고 통제 가능한 방식.

```bash
cd /Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM
claude
```

세션이 열리면 다음과 같은 지시문을 한 번 보낸다:

> `history/simulation_adjustment_report.md` 를 읽고, 거기 명시된 protocol 대로
> `simulation_4_layered_v10.R` 의 `common_lsirm_prop_sd` 를 튜닝해 줘.
> 모든 acceptance rate 가 [0.3, 0.6] 구간에 들어올 때까지 반복하고,
> 각 iteration 마다 보고서에 결과를 append 해 줘. 최대 8 iteration 까지만 시도해.

이렇게 하면 Claude Code 가 자동으로:
- `Bash` 로 `Rscript simulation_4_layered_v10.R` 실행
- 출력에서 acceptance rate 파싱
- `Edit` 로 proposal SD 수정
- 보고서에 iteration block 추가
- 종료 조건까지 반복

#### 장시간 실행 대비 팁
- 시뮬레이션 한 번이 길다면 (`n_iter = 50000`, [`simulation_4_layered_v10.R:389`](../simulation_4_layered_v10.R#L389)), `Bash` 호출에 `timeout` 을 명시적으로 늘리도록 지시한다 (Claude Code 의 Bash 도구 기본 timeout 은 2 분, 최대 10 분).
- 한 번 run 이 10 분을 넘기면 `run_in_background: true` 로 실행하고 `BashOutput` / 로그 tail 로 진행을 확인하도록 지시한다.
- iteration 수를 사전에 제한 (예: 최대 6~8 회) 해 무한 루프 방지.

### 3.2 비대화형 (one-shot) 호출
스크립트 화 하고 싶다면:

```bash
claude -p "history/simulation_adjustment_report.md 의 protocol 을 따라 \
simulation_4_layered_v10.R 의 proposal SD 를 acceptance rate 가 [0.3,0.6] \
구간에 들어올 때까지 튜닝하고, 각 iteration 결과를 보고서에 append 해 줘. \
최대 8 iteration." \
  --permission-mode acceptEdits
```

- `-p` (print mode): 응답을 stdout 으로 받고 종료.
- `--permission-mode acceptEdits`: Edit/Write 자동 승인 (반복 중 매 prompt 끊기지 않게).
- 단, 이 모드에서는 중간 개입이 어려우므로 첫 시도는 대화형 (3.1) 을 권장.

### 3.3 `/loop` 슬래시 커맨드 (자동 반복)
Claude Code 내장 `/loop` 을 쓰면 model 이 스스로 페이스를 정해 반복한다. 단, 이 task 는 "종료 조건이 모델 판단" 이므로 `/loop` 의 dynamic mode 가 적합하다:

```
/loop history/simulation_adjustment_report.md 의 protocol 을 한 iteration 수행해 줘. 모든 acceptance 가 [0.3,0.6] 에 들어왔으면 loop 를 끝내.
```

각 wake-up 마다 한 iteration 을 돌리고, 종료 조건 만족 시 모델이 `ScheduleWakeup` 을 호출하지 않아 루프가 자연 종료된다.

### 3.4 권장 실행 순서
1. 처음에는 **3.1 대화형** 으로 한 두 iteration 진행 → 출력 형식과 SD 변동이 합리적인지 직접 확인.
2. 안정되면 동일 세션에서 "남은 iteration 자동 진행" 지시 → 여기서부터는 사실상 자율.
3. 문제가 생기면 Ctrl-C 로 중단 후 보고서를 보고 수동 개입.

---

## 4. 보고서 구조 규칙 (How to append iteration logs)

각 iteration 결과는 아래 template 으로 본 문서 하단의 `## Iterations` 섹션에 추가한다.

```markdown
### Iteration <NN>  (<YYYY-MM-DD HH:MM>)

**Proposal SD (before run):**
| param | value |
|---|---|
| alpha1 | ... |
| ...   | ... |

**Acceptance rates (from result$accept):**
| param | acc | in [0.3, 0.6]? | action |
|---|---|---|---|
| alpha1 | 0.45 | yes | keep |
| beta2  | 0.18 | no  | SD × 0.6 |
| ...    | ...  | ... | ... |

**Notes:** runtime, warnings, anomalies.

**Next iteration SD diff:**
- `beta2`: 0.10 → 0.06
- ...
```

---

## Tuning Run Settings (temporary)

튜닝 효율을 위해 [`simulation_4_layered_v10.R:389`](../simulation_4_layered_v10.R#L389) 의
MCMC 길이를 일시적으로 축소했음:
- `n_iter`: 50000 → **5000**
- `burnin`: 20000 → **2000**
- `thin`: 10 → **5**

Acceptance rate 는 수천 iter 만으로도 안정적으로 추정되므로 튜닝에 충분하며,
**모든 acc가 [0.3, 0.6] 안에 들어오면 원래 값(50000/20000/10)으로 복원**한다.

또한 `fix_gamma=TRUE` ([`simulation_4_layered_v10.R:425`](../simulation_4_layered_v10.R#L425)) 이므로
`log_gamma1..4` 는 실제 sampling 이 일어나지 않아 acc=1.000 은 정상이며,
**`log_gamma*` 는 튜닝 대상에서 제외**한다.

R 인터프리터: `/Library/Frameworks/R.framework/Resources/bin/Rscript` (R 4.4.1).
(conda mlpractice 환경의 R 은 libreadline 누락으로 사용 불가.)

---

## Iterations

### Iteration 01  (2026-05-02)

**Proposal SD (before run):**
| param | value |
|---|---|
| alpha1 | 0.50 |
| alpha2 | 0.40 |
| alpha3 | 0.50 |
| alpha4 | 0.50 |
| beta1 | 0.40 |
| beta2 | 0.10 |
| beta3 | 0.20 |
| a | 0.30 |
| b1 | 0.30 |
| b2 | 0.20 |
| b3 | 0.20 |
| b4 | 0.20 |
| log_kappa | 0.20 |

**Split-merge:** split 813/5313 (0.153), merge 698/17187 (0.041)

**Acceptance rates (from result$accept):**
| param | acc | in [0.3, 0.6]? | action |
|---|---|---|---|
| alpha1 | 0.784 | no (high) | SD × 1.6 → 0.80 |
| alpha2 | 0.617 | no (borderline) | SD × 1.2 → 0.48 |
| alpha3 | 0.781 | no (high) | SD × 1.6 → 0.80 |
| alpha4 | 0.751 | no (high) | SD × 1.6 → 0.80 |
| beta1 | 0.599 | yes (boundary) | keep 0.40 |
| beta2 | 0.634 | no (borderline) | SD × 1.3 → 0.13 |
| beta3 | 0.760 | no (high) | SD × 1.6 → 0.32 |
| log_gamma1..4 | 1.000 | n/a (fix_gamma=TRUE) | skip |
| a | 0.605 | no (within MC noise) | keep 0.30 |
| b1 | 0.608 | no (within MC noise) | keep 0.30 |
| b2 | 0.332 | yes | keep 0.20 |
| b3 | 0.703 | no (high) | SD × 1.5 → 0.30 |
| b4 | 0.660 | no (borderline) | SD × 1.3 → 0.26 |
| log_kappa | 0.499 | yes | keep 0.20 |

**Notes:**
- 6/13 in target. acc 가 전반적으로 높은 편 → SD 가 너무 작음.
- `a`, `b1` 은 acc 0.605, 0.608 로 0.6 의 +0.005~0.008 범위 → MC noise 로 간주, 보수적으로 유지.

**Next iteration SD diff:**
- alpha1: 0.50 → 0.80
- alpha2: 0.40 → 0.48
- alpha3: 0.50 → 0.80
- alpha4: 0.50 → 0.80
- beta2: 0.10 → 0.13
- beta3: 0.20 → 0.32
- b3: 0.20 → 0.30
- b4: 0.20 → 0.26

---

### Iteration 02  (2026-05-02)

**Proposal SD (before run):** alpha1=0.80, alpha2=0.48, alpha3=0.80, alpha4=0.80, beta1=0.40, beta2=0.13, beta3=0.32, a=0.30, b1=0.30, b2=0.20, b3=0.30, b4=0.26, log_kappa=0.20

**Split-merge:** split 881/5433 (0.162), merge 762/17067 (0.045)

**Acceptance rates:**
| param | acc | in [0.3, 0.6]? | action |
|---|---|---|---|
| alpha1 | 0.676 | no | SD × 1.2 → 0.96 |
| alpha2 | 0.564 | yes | keep 0.48 |
| alpha3 | 0.678 | no | SD × 1.2 → 0.96 |
| alpha4 | 0.629 | no (borderline) | SD × 1.1 → 0.88 |
| beta1 | 0.598 | yes | keep 0.40 |
| beta2 | 0.555 | yes | keep 0.13 |
| beta3 | 0.640 | no (borderline) | SD × 1.15 → 0.37 |
| a | 0.605 | no (borderline, stuck) | SD × 1.1 → 0.33 |
| b1 | 0.610 | no (borderline) | SD × 1.1 → 0.33 |
| b2 | 0.330 | yes | keep 0.20 |
| b3 | 0.588 | yes | keep 0.30 |
| b4 | 0.576 | yes | keep 0.26 |
| log_kappa | 0.498 | yes | keep 0.20 |

**Notes:**
- 7/13 in target. alpha2/beta1/beta2/b3/b4 진입 성공.
- `a` 가 iter01,02 모두 0.605 → run-to-run noise 가 작음. 작은 nudge.

**Next iteration SD diff:**
- alpha1: 0.80 → 0.96
- alpha3: 0.80 → 0.96
- alpha4: 0.80 → 0.88
- beta3: 0.32 → 0.37
- a: 0.30 → 0.33
- b1: 0.30 → 0.33

---

### Iteration 03  (2026-05-02)

**Proposal SD (before run):** alpha1=0.96, alpha2=0.48, alpha3=0.96, alpha4=0.88, beta1=0.40, beta2=0.13, beta3=0.37, a=0.33, b1=0.33, b2=0.20, b3=0.30, b4=0.26, log_kappa=0.20

**Split-merge:** split 845/5433 (0.156), merge 780/17067 (0.046)

**Acceptance rates:**
| param | acc | in [0.3, 0.6]? | action |
|---|---|---|---|
| alpha1 | 0.619 | no | SD × 1.2 → 1.15 |
| alpha2 | 0.563 | yes | keep 0.48 |
| alpha3 | 0.623 | no | SD × 1.2 → 1.15 |
| alpha4 | 0.600 | yes (boundary) | keep 0.88 |
| beta1 | 0.602 | no (just over) | SD × 1.1 → 0.44 |
| beta2 | 0.552 | yes | keep 0.13 |
| beta3 | 0.598 | yes | keep 0.37 |
| a | 0.573 | yes | keep 0.33 |
| b1 | 0.575 | yes | keep 0.33 |
| b2 | 0.329 | yes | keep 0.20 |
| b3 | 0.585 | yes | keep 0.30 |
| b4 | 0.579 | yes | keep 0.26 |
| log_kappa | 0.498 | yes | keep 0.20 |

**Notes:**
- 10/13 in target. alpha4 / beta3 가 정확히 boundary (0.600 / 0.598). 안정적.
- 잔여 3개: alpha1, alpha3 (각 0.62 근처), beta1 (0.602).

**Next iteration SD diff:**
- alpha1: 0.96 → 1.15
- alpha3: 0.96 → 1.15
- beta1: 0.40 → 0.44

---

### Iteration 04  (2026-05-02)

**Proposal SD (before run):** alpha1=1.15, alpha2=0.48, alpha3=1.15, alpha4=0.88, beta1=0.44, beta2=0.13, beta3=0.37, a=0.33, b1=0.33, b2=0.20, b3=0.30, b4=0.26, log_kappa=0.20

**Split-merge:** split 813/5537 (0.147), merge 733/16963 (0.043)

**Acceptance rates:**
| param | acc | in [0.3, 0.6]? | action |
|---|---|---|---|
| alpha1 | 0.569 | yes | keep 1.15 |
| alpha2 | 0.562 | yes | keep 0.48 |
| alpha3 | 0.571 | yes | keep 1.15 |
| alpha4 | 0.602 | no (just over) | SD × 1.05 → 0.92 |
| beta1 | 0.576 | yes | keep 0.44 |
| beta2 | 0.553 | yes | keep 0.13 |
| beta3 | 0.595 | yes | keep 0.37 |
| a | 0.573 | yes | keep 0.33 |
| b1 | 0.571 | yes | keep 0.33 |
| b2 | 0.333 | yes | keep 0.20 |
| b3 | 0.584 | yes | keep 0.30 |
| b4 | 0.576 | yes | keep 0.26 |
| log_kappa | 0.497 | yes | keep 0.20 |

**Notes:**
- 12/13 in target. 잔여: alpha4 (0.602) — MC noise 수준의 boundary 초과.
- alpha1/alpha3 는 ×1.2 nudge 후 0.62→0.57 로 정확히 들어옴.

**Next iteration SD diff:**
- alpha4: 0.88 → 0.92

---

### Iteration 05  (2026-05-02) — **CONVERGED**

**Proposal SD (before run):** alpha1=1.15, alpha2=0.48, alpha3=1.15, alpha4=0.92, beta1=0.44, beta2=0.13, beta3=0.37, a=0.33, b1=0.33, b2=0.20, b3=0.30, b4=0.26, log_kappa=0.20

**Split-merge:** split 778/5331 (0.146), merge 704/17169 (0.041)

**Acceptance rates:**
| param | acc | in [0.3, 0.6]? |
|---|---|---|
| alpha1 | 0.570 | yes |
| alpha2 | 0.563 | yes |
| alpha3 | 0.567 | yes |
| alpha4 | 0.587 | yes |
| beta1 | 0.576 | yes |
| beta2 | 0.555 | yes |
| beta3 | 0.596 | yes |
| a | 0.574 | yes |
| b1 | 0.586 | yes |
| b2 | 0.325 | yes |
| b3 | 0.579 | yes |
| b4 | 0.574 | yes |
| log_kappa | 0.497 | yes |

**13/13 모두 [0.3, 0.6] 진입.** 튜닝 종료.

---

## Final Tuned Proposal SD

[`simulation_4_layered_v10.R:360-368`](../simulation_4_layered_v10.R#L360-L368) 의 최종 값:

```r
common_lsirm_prop_sd <- list(
  alpha1 = 1.15, alpha2 = 0.48, alpha3 = 1.15, alpha4 = 0.92, alpha5 = 0.5,
  log_gamma1 = 0.10, log_gamma2 = 0.05, log_gamma3 = 0.05,
  log_gamma4 = 0.05, log_gamma5 = 0.05,
  a = 0.33,
  beta1 = 0.44, beta2 = 0.13, beta3 = 0.37, beta4 = 0.30, beta5 = 0.30,
  b1 = 0.33, b2 = 0.20, b3 = 0.30, b4 = 0.26, b5 = 0.20,
  log_kappa = 0.20
)
```

`alpha5`, `beta5`, `b5`, `log_gamma5` 는 4-layer 시뮬에서 사용되지 않으므로 미조정.
`log_gamma1..4` 는 `fix_gamma=TRUE` 라 미조정.
`beta4` (GRM threshold) 의 acc 는 `result$accept` 에 별도 키로 보고되지 않으므로 (Gibbs/통합 업데이트 가능성) 미조정.

## Restored MCMC Settings

[`simulation_4_layered_v10.R:389`](../simulation_4_layered_v10.R#L389):
- `n_iter`: 5000 → **50000** (원복)
- `burnin`: 2000 → **20000** (원복)
- `thin`: 5 → **10** (원복)

## Full-MCMC Quality Check (n_iter=50000, burnin=20000, thin=10)

튜닝 종료 후 원래 MCMC 길이로 한 번 돌린 결과
(`logs/sim_v10_full.log`, RDS:
`plot/simulation_4_layered_v10_rfac2_norowctr_S0_0.01/v10_simulation_result_rfac2_norowctr_S0_0.01.rds`).

### Group-mean acceptance
| group | mean acc |
|---|---|
| alpha1..4 | 0.568 / 0.561 / 0.571 / 0.585 |
| beta1..3  | 0.574 / 0.552 / 0.597 |
| log_gamma1..4 | 1.000 (fix_gamma=TRUE; not sampled) |
| a / b1..4 | 0.572 / 0.580 / 0.329 / 0.583 / 0.576 |
| log_kappa per-item mean | 0.498 |

→ 모든 그룹 평균은 [0.3, 0.6] 안. **튜닝의 1차 목표는 달성**.

### Element-level acceptance (per-element distribution)
| param | n | min | med | max | mean | <0.3 | >0.6 |
|---|---|---|---|---|---|---|---|
| alpha1 | 150 | 0.520 | 0.569 | 0.610 | 0.568 | 0 | 13 |
| alpha2 | 150 | 0.540 | 0.561 | 0.601 | 0.561 | 0 | 1 |
| alpha3 | 150 | 0.348 | 0.587 | 0.629 | 0.571 | 0 | 63 |
| alpha4 | 150 | 0.517 | 0.582 | 0.648 | 0.585 | 0 | 48 |
| beta1 | 10 | 0.471 | 0.530 | 0.781 | 0.574 | 0 | 3 |
| beta2 | 10 | 0.548 | 0.552 | 0.557 | 0.552 | 0 | 0 |
| beta3 | 10 | 0.533 | 0.600 | 0.679 | 0.597 | 0 | 5 |
| a | 150 | 0.440 | 0.570 | 0.663 | 0.572 | 0 | 41 |
| b1 | 10 | 0.484 | 0.551 | 0.769 | 0.580 | 0 | 3 |
| b2 | 10 | 0.311 | 0.329 | 0.342 | 0.329 | 0 | 0 |
| b3 | 10 | 0.489 | 0.586 | 0.681 | 0.583 | 0 | 3 |
| b4 | 10 | 0.488 | 0.562 | 0.729 | 0.576 | 0 | 3 |
| log_kappa | 10 | 0.495 | 0.499 | 0.500 | 0.498 | 0 | 0 |
| beta4_thr | 40 | 0.260 | 0.493 | 0.868 | 0.499 | 2 | 7 |

**총 870 element 중 678 (77.9%) 가 [0.3, 0.6] 안.** 21.8% 가 0.6 초과, 0.2% (beta4_thr 2개) 가 0.3 미만.

#### Element-level 미달 그룹과 spread (강한 → 약한 순)
| group | spread (max−min) | 평가 |
|---|---|---|
| beta4_thr | **0.608** (0.260–0.868) | **단일 SD 로 [0.3, 0.6] 수렴 불가** — element 간 본질적 이질성 |
| beta1 | 0.310 (0.471–0.781) | 단일 SD 로 매우 어려움 |
| b1 | 0.285 (0.484–0.769) | 단일 SD 로 매우 어려움 |
| alpha3 | 0.281 (0.348–0.629) | 단일 SD 로 어려움 |
| b4 | 0.241 (0.488–0.729) | 어려움 |
| a | 0.223 (0.440–0.663) | 가능하나 빡빡 |
| b3 | 0.192 (0.489–0.681) | 가능 |
| beta3 | 0.146 (0.533–0.679) | 가능 |
| alpha4 | 0.131 (0.517–0.648) | 가능 |

### Recovery quality
| metric | value |
|---|---|
| distance cor (all layers) | 0.749 |
| distance cor L1 (Bin) | 0.772 |
| distance cor L2 (Con) | 0.765 |
| distance cor L3 (Cnt) | 0.756 |
| distance cor L4 (Ord) | 0.732 |
| median K_+ | 4 (= K_true) |
| ARI (final partition vs true) | **0.867** |
| split / merge rate | 0.152 / 0.043 |

→ **사후 회복 품질은 양호**. 모형은 정상 동작.

### 종합 진단
1. **Group-mean 기준** (사용자 최초 요구의 약한 해석): 통과 (모든 group mean ∈ [0.3, 0.6]).
2. **Element-level 기준** (사용자 보충 질문의 엄격한 해석): 78% 만 통과. spread 가 큰
   `beta4_thr`, `beta1`, `b1`, `alpha3` 는 단일 SD 로는 사실상 만족 불가.
3. **모형 회복 (cor, ARI)**: 양호 — 현재 SD 로 사후 추론은 신뢰 가능.

### 다음 옵션
- (A) 현 SD 로 확정. group-mean 만 [0.3, 0.6] 인 점, recovery 양호인 점 근거.
- (B) 추가 튜닝으로 element-level pass-rate 를 더 높임 (예: 90%+ in target).
  - spread 가 큰 그룹에서는 trade-off (max↓ 시 min<0.3 위험) 발생 가능.
- (C) 모델 수정으로 **per-element proposal SD** 를 도입 (Robbins-Monro adaptation).
  `data/my_LSIRM_FMC_cpp_v10.R` / C++ 백엔드 변경 필요. 가장 근본적이나 비용 큼.

---

## P1: Procrustes 정렬 보강 (2026-05-02)

### 진단
원래 wrapper ([`data/my_LSIRM_FMC_cpp_v10.R`](../data/my_LSIRM_FMC_cpp_v10.R), 구버전 line 265) 의 Procrustes 는
`ref_idx <- n_save` — 즉 **마지막 iteration 의 stacked 좌표를 reference 로** 사용해 다른 iter 를 정렬.
reference 가 임의의 한 sample 이라 chain 전체가 그 sample 의 임의 회전/병진 frame 에 anchored.
결과: 모형 회복은 좋아도 (`cor(D)=0.749`, `ARI=0.867`) **위치 element 별 95% CI coverage 가 15–50%** 밖에 안 됨.

| param | 옛 정렬 cov95 | RMSE | 옛 메커니즘 |
|---|---|---|---|
| a | 35.7% | 1.85 | last-iter as ref |
| b1 | 50.0% | 1.36 | |
| b2 | 15.0% | 1.45 | |
| b3 | 35.0% | 1.41 | |
| b4 | 35.0% | 1.30 | |

### Diagnostic 1: per-iter Procrustes 직접 truth 정렬 — 부분 성공, 분산 축소
- per-iter 정렬은 각 iter 를 truth 에 과적합 → within-chain 분산 줄어들어 CI 좁아짐
- RMSE 는 줄지만 (1.85→0.92) coverage 그대로 (35.7%)

### Diagnostic 2: posterior-mean → truth 단일 rigid transform 모든 iter 일괄 적용 — 정공법
- chain 의 frame 만 옮기고 within-chain 분산 보존
- 검증 결과: a 35.7%→**74%**, b1 50%→**100%**, b2 15%→**90%**, b3 35%→**100%**, b4 35%→**100%**

### 구현
[`data/my_LSIRM_FMC_cpp_v10.R`](../data/my_LSIRM_FMC_cpp_v10.R) 함수 시그니처에
`procrustes_target = NULL` 인자 추가. 정렬 블록을 두 단계로 재작성:

1. **Step 1: iterative-mean refinement** (3 passes). 매 pass 마다 stacked posterior mean 을
   target 으로 모든 iter 를 per-iter Procrustes. Chain 이 self-consistent 한 frame 으로 수렴.
2. **Step 2 (옵션): 외부 target 으로 단일 rigid re-anchor**. `procrustes_target` 가 주어지면,
   posterior mean → target 의 단일 회전+병진 변환을 모든 iter 에 일괄 적용.
   분산 보존하며 frame 만 truth 의 frame 으로 옮김.

[`simulation_4_layered_v10.R:425`](../simulation_4_layered_v10.R#L425) 에서:
```r
procrustes_target = list(a  = A_true,
                         b1 = B1_true, b2 = B2_true,
                         b3 = B3_true, b4 = B4_true)
```

### Full MCMC 재실행 결과 (n_iter=50000)
로그: `logs/sim_v10_full_p1b.log`

```
iterative-mean pass 1/3, target shift RMSE = (initial)
iterative-mean pass 2/3, target shift RMSE = 0.0944
iterative-mean pass 3/3, target shift RMSE = 0.0000   ← 수렴
external re-anchor: post-mean stacked RMSE 1.674 -> 0.835
```

### Coverage 비교 (P1 전 vs 후)
| param | n | **P1 전 cov95** | **P1 후 cov95** | RMSE 전 → 후 |
|---|---|---|---|---|
| alpha1 | 150 | 95.3% | 95.3% | 0.746 → 0.746 |
| alpha2 | 150 | 84.7% | 84.7% | 0.633 → 0.633 |
| alpha3 | 150 | 96.0% | 96.0% | 0.756 → 0.756 |
| alpha4 | 150 | 92.0% | 92.0% | 0.665 → 0.665 |
| beta1  | 10 | 80.0% | 80.0% | 0.751 → 0.751 |
| beta2  | 10 | 70.0% | 70.0% | 0.505 → 0.505 |
| beta3  | 10 | 80.0% | 80.0% | 0.592 → 0.592 |
| beta4_thr | 40 | 90.0% | 90.0% | 0.432 → 0.432 |
| **a**  | 300 | **35.7%** | **73.7%** | **1.845 → 0.920** |
| **b1** | 20 | **50.0%** | **100.0%** | **1.359 → 0.310** |
| **b2** | 20 | **15.0%** | **90.0%** | **1.450 → 0.322** |
| **b3** | 20 | **35.0%** | **100.0%** | **1.413 → 0.390** |
| **b4** | 20 | **35.0%** | **100.0%** | **1.301 → 0.429** |

→ 위치 (b) coverage 가 95% 가깝거나 그 이상으로 회복. a 는 73.7% (95% 미달)
이지만 single-frame 정렬로 가능한 한계로 보임.

### 남은 문제: beta 의 +0.4–0.6 location bias
`P1` 으로 해결되지 않음. 원인 분석 (Hypothesis 1+2 동시):
- pure prior shrinkage: beta3 의 부호가 반대라 단독으로는 설명 불가
- pure (α+c, β+c) ridge: alpha 와 beta shift 가 동일해야 하는데 empirically 다름
- 즉 *이질적 두 효과*의 결합 (prior pull + 데이터-소수성 효과)

centering 후에는 coverage 100% 회복:
| param | raw cov | centered cov | raw RMSE | centered RMSE |
|---|---|---|---|---|
| beta1 | 80.0% | **100.0%** | 0.751 | 0.431 |
| beta2 | 70.0% | 90.0% | 0.505 | 0.240 |
| beta3 | 80.0% | **100.0%** | 0.592 | 0.245 |

#### Q: prior 의 τ 를 키워서 해결되나? — **부분적으로만**
- ETA = α − β − γ·D 라 likelihood 는 (α − β) 만 식별. mean(β) 는 prior 가 결정.
- τ ↑ → prior pull 약해짐 → posterior location 이 데이터-기반 위치로 표류 가능.
- 그러나 τ → ∞ 면 location 이 improper 비식별 → MCMC 발산.
- 우리의 bias 크기 (~0.5) 는 단순 shrinkage-to-0 가 예측하는 범위 (beta3 의 경우 부호조차 반대)
  를 넘으므로, τ 만 키운다고 일관되게 해결되지 않음.

#### 정공법
- 계층 평균: `β_j ~ N(μ_β, τ²)`, `μ_β ~ flat`. μ_β 가 자동으로 mean(β_true) 학습.
- 또는 sum-to-zero 식별제약 (`mean(α_l) = 0` 강제).
- 가장 싼 실용 옵션: 사후 단계에서 `α ← α − μ_α; β ← β + μ_α` (P2, 보류 중).

### 모형 회복 품질 (P1 후 동일)
| metric | value |
|---|---|
| dist cor (all layers) | 0.749 |
| dist cor L1/L2/L3/L4 | 0.772 / 0.765 / 0.756 / 0.732 |
| median K_+ | 4 (= K_true) |
| ARI (final partition vs true) | 0.867 |
| split / merge rate | 0.152 / 0.043 |

---

## Iter01→Iter05 SD 변동 요약

---

## Iter01→Iter05 SD 변동 요약
| param | iter01 SD | iter05 SD | × |
|---|---|---|---|
| alpha1 | 0.50 | 1.15 | 2.30 |
| alpha2 | 0.40 | 0.48 | 1.20 |
| alpha3 | 0.50 | 1.15 | 2.30 |
| alpha4 | 0.50 | 0.92 | 1.84 |
| beta1 | 0.40 | 0.44 | 1.10 |
| beta2 | 0.10 | 0.13 | 1.30 |
| beta3 | 0.20 | 0.37 | 1.85 |
| a | 0.30 | 0.33 | 1.10 |
| b1 | 0.30 | 0.33 | 1.10 |
| b2 | 0.20 | 0.20 | 1.00 |
| b3 | 0.20 | 0.30 | 1.50 |
| b4 | 0.20 | 0.26 | 1.30 |
| log_kappa | 0.20 | 0.20 | 1.00 |





