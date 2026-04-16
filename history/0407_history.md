# 0407 Variable Specification (MIDUS 2 기준, v3 preprocess)

## 변경 이력
- **0407**: B4P2A (Waist cm), B4PWHR (Waist-Hip Ratio) 주석처리 — 임시 제외하여 결과 확인

---

## Layer 구성 (screen_case == 1 기준)

### 1. Binary (Y_bin)
CESD 20문항 이진화 (원래 1~4 ordinal → 1=0, 2/3/4=1). Reverse items (D, H, L, P)는 역코딩 후 이진화.

| # | Wave 2 변수 | 설명 |
|---|------------|------|
| 1 | B4Q3A | CESD: I was bothered by things that usually don't bother me |
| 2 | B4Q3B | CESD: I did not feel like eating; my appetite was poor |
| 3 | B4Q3C | CESD: I felt that I could not shake off the blues |
| 4 | B4Q3D | CESD: I felt that I was just as good as other people (reverse) |
| 5 | B4Q3E | CESD: I had trouble keeping my mind on what I was doing |
| 6 | B4Q3F | CESD: I felt depressed |
| 7 | B4Q3G | CESD: I felt that everything I did was an effort |
| 8 | B4Q3H | CESD: I felt hopeful about the future (reverse) |
| 9 | B4Q3I | CESD: I thought my life had been a failure |
| 10 | B4Q3J | CESD: I felt fearful |
| 11 | B4Q3K | CESD: My sleep was restless |
| 12 | B4Q3L | CESD: I was happy (reverse) |
| 13 | B4Q3M | CESD: I talked less than usual |
| 14 | B4Q3N | CESD: I felt lonely |
| 15 | B4Q3O | CESD: People were unfriendly |
| 16 | B4Q3P | CESD: I enjoyed life (reverse) |
| 17 | B4Q3Q | CESD: I had crying spells |
| 18 | B4Q3R | CESD: I felt sad |
| 19 | B4Q3S | CESD: I felt that people dislike me |
| 20 | B4Q3T | CESD: I could not get going |

**Total: 20 items**

---

### 2. Count (Y_cnt)

| # | Wave 2 변수 | 설명 | 변환 |
|---|------------|------|------|
| 1 | B4S4 | Sleep hours (actual sleep at night) | to_numeric + floor |
| 2 | B4H40 | Alcohol: drinks per drinking day (period drank most) | to_numeric + floor |
| 3 | sleep_trouble_count | Sleep trouble count (B4S11A~J 합산) | 10 items 이진화(>=2→1, 1→0) 후 합산, 0~10 |

**Total: 3 items**

---

### 3. Ordinal-5 (Y_ord1) — MASQ 5-point Likert (1~5)

#### General Distress-Anxious (GDA, 11 items)

| # | Wave 2 변수 | 설명 |
|---|------------|------|
| 1 | B4Q1D | MASQ: Felt afraid |
| 2 | B4Q1H | MASQ: Had diarrhea |
| 3 | B4Q1K | MASQ: Felt nervous |
| 4 | B4Q1N | MASQ: Felt uneasy |
| 5 | B4Q1P | MASQ: Had a lump in my throat |
| 6 | B4Q1T | MASQ: Had an upset stomach |
| 7 | B4Q1Z | MASQ: Felt keyed up, on edge |
| 8 | B4Q1FF | MASQ: Was unable to relax |
| 9 | B4Q1II | MASQ: Felt nauseous |
| 10 | B4Q1CCC | MASQ: Felt tense or high-strung |
| 11 | B4Q1GGG | MASQ: Muscles were tense or sore |

#### Anxious Arousal (AA, 17 items)

| # | Wave 2 변수 | 설명 |
|---|------------|------|
| 12 | B4Q1B | MASQ: Startled easily |
| 13 | B4Q1F | MASQ: Hands were shaky |
| 14 | B4Q1M | MASQ: Was short of breath |
| 15 | B4Q1Q | MASQ: Felt faint |
| 16 | B4Q1S | MASQ: Had hot or cold spells |
| 17 | B4Q1X | MASQ: Hands were cold or sweaty |
| 18 | B4Q1BB | MASQ: Was trembling or shaking |
| 19 | B4Q1DD | MASQ: Had trouble swallowing |
| 20 | B4Q1KK | MASQ: Felt dizzy or lightheaded |
| 21 | B4Q1NN | MASQ: Had pain in my chest |
| 22 | B4Q1PP | MASQ: Felt like I was choking |
| 23 | B4Q1RR | MASQ: Muscles twitched or trembled |
| 24 | B4Q1TT | MASQ: Had a very dry mouth |
| 25 | B4Q1VV | MASQ: Was afraid I was going to die |
| 26 | B4Q1ZZ | MASQ: Heart was racing or pounding |
| 27 | B4Q1BBB | MASQ: Felt numbness or tingling in body |
| 28 | B4Q1JJJ | MASQ: Had to urinate frequently |

**Total: 28 items (GDA 11 + AA 17)**

---

### 4. Ordinal-4 (Y_ord2) — PSQI Sleep Quality (1~4)

| # | Wave 2 변수 | 설명 |
|---|------------|------|
| 1 | B4S5 | Rate sleep quality overall (1=Very good ~ 4=Very bad) |

**Total: 1 item**

---

### 5. Continuous (Y_con)

| # | Wave 2 변수 | 설명 | 비고 |
|---|------------|------|------|
| 1 | B4BSCL14 | HPA axis: Saliva cortisol all sample average | |
| 2 | B4BNE12 | Autonomic arousal: Urine Norepinephrine 12 hour | |
| ~~3~~ | ~~B4P2A~~ | ~~Obesity: Waist in centimeters~~ | 0407 주석처리 |
| ~~4~~ | ~~B4PWHR~~ | ~~Central adiposity: Waist-Hip Ratio~~ | 0407 주석처리 |
| 3 | B4BIL6 | Inflammation: Blood Serum IL6 (pg/mL) | |
| 4 | B4BFGN | Inflammation: Blood Fibrinogen (mg/dL) | |
| 5 | B4BCRP | Inflammation: Blood C-Reactive Protein (ug/mL) | |

**Total: 5 items (B4P2A, B4PWHR 제외)**

---

## Screening

| Gate | Duration | Frequency | 설명 |
|------|----------|-----------|------|
| B1PA60 | B1PA61 | B1PA62 | Sadness branch |
| B1PA72 | B1PA73 | B1PA74 | Anhedonia branch |

- gate == 1 & duration <= 2 & frequency <= 3 → pass
- sadness_pass == 1 OR anhedonia_pass == 1 → screen_case = 1

## Covariates
- age: B1PAGE_M2
- sex: B1PRSEX

---

## P1-P3 분석 (Cognitive) 추가 변수

### Continuous (cog)

| # | Wave 2 변수 | 설명 | 변환 |
|---|------------|------|------|
| 1 | B3TEMZ3 | Episodic Memory Z-score | 부호 반전 (높=나쁨) |
| 2 | B3TEFZ3 | Executive Functioning Z-score | 부호 반전 (높=나쁨) |
| 3 | B3TWLF | Word List Proportion forgot | 높=많이 잊음 |
| 4 | B3TSMN | SGST Normal single-task median RT (초) | |
| 5 | B3TSMR | SGST Reverse single-task median RT (초) | |
| 6 | B3TSMXBS | SGST Mixed block all switch median RT (초) | |
| 7 | B3TSMXNO | SGST Mixed block normal non-switch median RT (초) | |
| 8 | B3TSMXRO | SGST Mixed block reverse non-switch median RT (초) | |

### Count (cog)

| # | Wave 2 변수 | 설명 | 변환 |
|---|------------|------|------|
| 1 | B3TSPN | SGST Normal single-task 오답 수 | 정확도 → 20 - round(x/0.05), 0~20 |
| 2 | B3TSPR | SGST Reverse single-task 오답 수 | 정확도 → 20 - round(x/0.05), 0~20 |
| 3 | B3TCTFLR | Category fluency 반복 수 | zero-inflated, floor |
| 4 | B3TCTFLI | Category fluency 침입 오류 수 | zero-inflated, floor |

---

## 전체 요약 (P1-P3-P4, 0407 기준)

| Layer | Items |
|-------|-------|
| Binary (CESD) | 20 |
| Count (bio + cog) | 7 (bio 3 + cog 4) |
| Ordinal-5 (MASQ) | 28 |
| Ordinal-4 (PSQI) | 1 |
| Continuous (bio + cog) | 13 (bio 5 + cog 8) |
| **Total** | **69** |
