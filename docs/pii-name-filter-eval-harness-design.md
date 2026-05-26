# PII 이름 필터 개선 — 내부 eval harness + 룰 튜닝 설계

> 상태: **설계 (DESIGN)**. 실행/구현은 별도 승인 후. (2026-05-26)
> 안전 계약 유지: migration 076 (`pii_candidates`) — 원문 PII / matched_text / char offset 외부 export 금지.
> DB write 없음, annotation 미수정, prod 미반영.

## 0. 배경 — 왜 학습이 아니라 룰 튜닝인가

PII 학습 export(`uncounted-api/scripts/analysis/export_pii_training.mjs`, read-only) 결과:

| 버킷 | count | type 분포 |
|---|---|---|
| positive (confirmed/corrected) | 53 | 이름 51, IP주소 2 |
| negative (rejected, hard neg) | 72 | 이름 71, 전화번호 1 |
| skipped | 1 | — |

검수 데이터는 **사실상 전부 "이름"**(positive 51 / negative 71)이고, export 가능한 구조 피처가
confirmed/rejected에서 **완전히 동일**하다:

- confidence: 전부 `0.70` (`pii_confidence._CONFIDENCE_AMBIGUOUS`)
- high_precision_pattern: 전부 `false`
- span_len: positive {3:50, 2:1}, negative {3:71}

→ 어떤 threshold로도 precision은 **0.42 고정**. 구조 피처 학습/threshold 튜닝 모두 무의미.
confirmed/rejected를 가르는 유일한 신호는 **원문 텍스트**인데, 076 계약상 외부 export 금지.

**결론**: 이건 모델 성능 문제가 아니라 voice-api bootstrap detector의 **룰 품질 문제**다.

## 1. 근본 원인 (코드 확정)

`app/pii_masker.py::_is_likely_name_with_context` (L298–321):

```
2글자 (성+1글자): 뒤에 _HONORIFICS 가 와야만 이름  → 고정밀
3글자 (성+2글자): denylist(_NAME_EXCLUDE_PREFIX) + 어두 경계 통과 시 → return True (무조건 emit)
```

검수된 122건이 전부 span_len=3 → **3글자 경로가 0.42 precision의 원인**. 이 경로는
denylist에 없는 "성+2글자 일반어/활용형"을 전부 이름으로 방출한다.
`pii_confidence`는 이름을 항상 `confidence=0.70 / needs_human_decision`으로 보내므로
detector 단계에는 graded 신호가 없다.

## 2. 내부 eval harness 설계 (step 1)

### 목적
원문을 **export하지 않고**, 검수 라벨(confirmed=정답 PII, rejected=오탐)을 기준으로
현 이름 필터의 baseline precision/recall을 측정하고, 룰 변경 전후 회귀를 측정한다.

### 데이터 결합 (restricted join, in-memory only)
1. `uncounted-api` DB에서 `pii_candidates` 중 `predicted_type='이름' AND admin_decision IS NOT NULL`
   행을 읽는다 → `utterance_id, char_start, char_end, admin_decision`.
2. 같은 프로세스 내에서 `utterances.transcript_text`를 join하여 `text[char_start:char_end]`와
   앞뒤 문맥 window(±N자)를 **메모리에서만** 재구성한다.
3. 이 텍스트는 **디스크에 쓰지 않고, 출력에 포함하지 않는다.** 평가 계산에만 사용.

### baseline 측정
- 각 라벨 발화에 `detect_pii_spans(text, enable_name_masking=True)`를 재실행.
- 검수된 (utterance_id, char_start, char_end) 위치에서 detector가 이름 span을 emit했는지 대조:
  - confirmed 위치에 emit → TP, emit 안 함 → FN
  - rejected 위치에 emit → FP
- 산출: precision = TP/(TP+FP), recall = TP/(TP+FN). (예상 baseline ≈ precision 0.42 / recall 1.0)

### 회귀 측정
- 룰 변경 후 동일 harness 재실행. **게이트: precision↑ AND recall ≥ floor**(예: 0.95).
- Red-Green: 변경 전(FAIL 기준선) → 변경 후(개선) 비교로 회귀 검증.

### 출력 (안전)
- precision / recall / TP·FP·FN count, FP의 **패턴 분류 집계만**(예: "활용형 어미", "일반명사").
- 개별 원문 token·offset은 출력하지 않는다. report는 집계 수치 + 룰 카테고리만.
- 위치: `.research/` (gitignore). 076 계약 위반 0.

### 비고
- harness는 `uncounted-api`(DB 접근) + `uncounted-voice-api`(detector) 양쪽을 참조.
  voice-api detect_pii_spans를 import하거나 detect-batch HTTP로 재실행 둘 다 가능.
  → 권장: detect-batch HTTP 재사용(이미 backfill에서 검증된 계약, 원문 미응답).
  단 HTTP 응답엔 offset만 오므로, 라벨 대조는 harness가 보유한 검수 offset과 매칭.

## 3. 이름 필터 룰 튜닝 아이디어 (step 2)

> 전부 harness로 측정 가능. denylist 보강은 **사람 검토 후 수동 추가**(자동 보강 금지).

1. **denylist(`_NAME_EXCLUDE_PREFIX`) 보강** — harness가 분류한 71개 FP의 "성+2글자 일반어/활용형"
   카테고리를 사람이 검토해 추가. (예: 성씨로 시작하는 활용형/합성어 중 누락분)
2. **3글자 경로 graded confidence 도입** — 현재 무조건 `return True` 대신:
   - 호칭/문맥 동반 3글자 → 높은 confidence
   - 문맥 없는 bare 3글자 → 낮은 confidence(또는 weak tier)
   → 이래야 향후 threshold/auto_rejected 경로가 의미를 가지고, 재수집 시 피처가 분리된다.
3. **활용형/조사 후행 제외** — 성+2글자 직후가 용언 어미/조사로 이어져 비-이름 토큰을 이루면 제외
   (`after` 문맥 분석). (예: "…했다" 류 conjugation)
4. **(선택, 무거움)** 형태소 분석(kiwipiepy/mecab) 게이트를 3글자 경로에 적용 — 고유명사 태그 확인.

권장 착수: **(1) denylist 보강 + (2) graded confidence** — 둘 다 harness로 측정되고 위험 낮음.

## 4. 순서 & 게이트

```
step 1  내부 eval harness  → baseline precision/recall 측정 (별도 실행 승인 필요)
step 2  룰 튜닝 (denylist + graded conf)  → harness 회귀 (precision↑, recall floor)
step 3  데이터 재수집  → graded conf로 피처 분리 생기는지 확인 → 재 export
step 4  그 후에야 학습/threshold/calibration 논의
```

### 금지/안전 (전 단계 공통)
- 원문/offset/스니펫 export·출력 금지 (076 계약 유지).
- denylist 자동 보강 금지 — 사람 검토 후 수동.
- DB write·annotation 수정·prod 반영 금지.
- harness/룰 변경이 live detector로 머지되려면 harness가 **precision 개선 + recall 무회귀** 입증 후.
- IP주소(positive 2)/전화번호(negative 1)는 표본 부족 → 평가만, 룰/학습 대상 아님.
