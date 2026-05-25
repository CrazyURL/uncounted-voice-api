# Segmenter v2 Worker Wiring 계획서

- 작성일: 2026-05-26
- 상태: **계획 (승인 대기)** — 코드 미적용. 본 문서 승인 후 별도 PR로 wiring.
- 선행: 진단·v2 순수함수·테스트 완료 (commit `e1f87a4`, branch `feat/stt-segmenter-v2`)
- 근거: [stt_segmentation_quality_audit_20260525.md](../scripts/analysis/stt_segmentation_quality_audit_20260525.md)

---

## 0. 목표와 비목표

- 목표: 신규 STT 처리에서 같은-화자 짧은 발화 과분할을 줄인다 (bidirectional, 추정 발화 −3.3% / 짧은단편 −11.0%).
- 비목표(이번 wiring 범위 아님): 기존 utterance row 재처리, DB 백필, hotwords/INITIAL_PROMPT prod 반영, 모델 교체.

---

## 1. 어디에 붙이는가

v2는 `segment()`가 생성한 발화 경계를 **병합**한다. 호출 지점은 두 곳:

| 경로 | 파일:라인 | 호출 |
|---|---|---|
| 일반 | `app/stt_processor.py:785` | `utterance_boundaries = segment_utterances(all_words, total_dur)` |
| 청크(>1h) | `app/services/chunk_utterance_emitter.py:69` | `chunk_local_utts = segment_utterances(chunk_local_words, ...)` |

두 경로 모두 `segment()` **직후, 발화별 오디오 추출 루프 이전**에 동일하게 병합이 일어나야 한다.

### 권장 구현: `segment()` 파이프라인 내부에 최종 단계로 삽입
`app/services/utterance_segmenter.py`의 기존 파이프라인:
```
raw → _fix_hanging_words → _merge_short_utterances → _split_long_utterances → _apply_padding
```
여기서 `_split_long_utterances` 다음, `_apply_padding` **이전**에 `_merge_v2_step`을 추가한다.
- 장점: 두 호출 지점(일반/청크)을 한 번에 커버, 호출부 수정 없음, 패딩·오디오 추출과 자연 정합.
- v2 순수함수(`merge_v2`)는 dict 단위 API이므로 `_RawUtterance ↔ dict` 어댑터를 둔다
  (`{start_sec,end_sec,speaker_id,transcript_text,word_count}` — words 결합 포함).

### ⚠️ 제약: 이 시점에는 PII interval이 없다
PII는 하류(`worker.persist_results`, voice API job 결과 병합)에서 채워진다. 따라서 `segment()`
시점 v2에는 `pii_intervals=[]`이고 **straddle 보호 로직은 동작하지 않는다**. 진단상 straddle
관측 0건 + 병합은 결코 span을 절단하지 않으므로 안전. (PII-aware re-merge가 필요하면 worker
하류에서의 row 재병합이 되어 오디오 재추출을 수반 → 별도 과제.)

---

## 2. 기본값 / Feature Flag

`app/config.py`에 추가 (현 PR의 `MERGE_V2_*` 임계값에 더해):

| 키 | 기본값 | 설명 |
|---|---|---|
| `MERGE_V2_ENABLED` | **false** | 마스터 스위치. false면 v1 동작 그대로(무변경) |
| `MERGE_V2_BIDIRECTIONAL` | **true** | enabled일 때 bidirectional 사용(권장). false면 forward-only |
| `MERGE_V2_GAP_SEC` / `_MAX_SEC` / `_SHORT_SEC` / `_SHORT_WORDS` | 0.8 / 13 / 2.0 / 5 | (이미 추가됨) |

- 기본 OFF로 머지 → 코드가 들어가도 prod 동작 불변. 환경변수로만 켠다.
- `_merge_v2_step`은 `if not config.MERGE_V2_ENABLED: return utterances`로 시작.

---

## 3. Rollback

- **즉시 롤백**: `MERGE_V2_ENABLED=false` 설정 후 worker/uvicorn 재시작. 코드 revert 불필요.
- DB 마이그레이션·스키마 변경 없음. 기존 row 불변.
- v2는 **신규 처리에만** 영향 → 켠 이후 처리분만 달라지고, 끄면 그 시점부터 다시 v1.
- 코드 롤백이 필요하면 wiring PR만 revert (순수함수/진단/테스트는 독립 유지).

---

## 4. short-heavy 세션 한정 검증 방식

1. **오프라인 게이트(이미 가능)**: `scripts/analysis/stt_segmentation_audit.py --dry-run-v2 --v2-bidirectional`
   로 대상 세션의 before/after 추정. short-heavy(짧은단편≥50%) 세션에서 감소율 확인.
2. **dev/staging 실처리**: short-heavy 대표 오디오 소수를 flag ON으로 신규 처리 → 다음 가드 확인:
   - 병합 후 어떤 발화도 `MAX_UTTERANCE_SEC` 초과 안 함 (max_merged + padding 상한).
   - 화자 순도 유지(구조상 교차-화자 병합 없음 — 회귀 테스트로 보장).
   - 종결어미 끝 비율 상승 / 짧은단편 비율 하락 방향 확인.
   - 수동 QA: 경계 몇 건 청취해 과병합(다른 문장 붙음) 없는지.
3. **카나리**: flag를 특정 신규 세션군에만(예: 내부 테스트 계정) 적용 후 검수 피드백 → 전체 확대.

---

## 5. 기존 발화 row 재처리 — 금지

- 이번 wiring은 **신규 처리 전용**. 기존 utterance row는 **재처리/백필하지 않는다**.
- 기존 row에 v2를 소급 적용하려면 (a) 저장된 `transcript_words`로 재병합 + (b) 병합 발화의
  오디오 재추출·재업로드 + (c) PII/품질/라벨 재계산 → 무거운 별도 과제이며 **별도 승인 게이트**.
- 따라서 본 계획의 효과는 "앞으로 처리되는 통화"에만 점진 반영된다.

---

## 6. 작업 순서 (승인 후 별도 PR)

1. `_merge_v2_step` + `_RawUtterance↔dict` 어댑터 추가 (`utterance_segmenter.py`).
2. `MERGE_V2_ENABLED` / `MERGE_V2_BIDIRECTIONAL` config 추가.
3. 회귀 테스트: flag OFF면 기존 21개 그대로 통과 / ON이면 병합 + max·화자순도 가드 테스트.
4. dev 검증(4절) → 카나리 → 확대.
5. (분리 트랙) hotwords/INITIAL_PROMPT, 기존 row 재처리는 각각 별도 승인.
