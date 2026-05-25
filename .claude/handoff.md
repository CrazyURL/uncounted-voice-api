# Handoff Document
생성일시: 2026-05-26 KST
effort: high
track: STT 발화분리 품질 (Segmenter v2)

## 0. 상태 INDEX (현재 트랙)

- ✅ STT 발화분리 **진단 완료** (read-only, live 60세션/5,643발화)
- ✅ Segmenter v2 **설계·구현·테스트 완료** (순수함수, forward+bidirectional)
- ⛔ worker **wiring 미적용** (계획서만 작성: `docs/stt-segmenter-v2-wiring-plan.md`)
- ⛔ **DB write / prod 반영 / 재처리 없음** — 전부 미실행
- 커밋: `e1f87a4` (branch `feat/stt-segmenter-v2`, 6파일만, PII 변경 미포함, 미푸시)

## 1. 완료한 작업

- **진단**: 1차 원인 = postprocess 병합 한계(v1의 5초 가드). VAD/WhisperX/diarization 아님.
  근거: <0.5s 단편 4%, 종결어미 끝 15%, 같은-화자 gap≤0.8 병합가능 257쌍.
- **Segmenter v2** (`app/services/utterance_segmenter_v2.py`): `merge_v2` 순수함수.
  같은 화자·gap≤0.8·짧음(2s/5단어)·문장종결 중단·max 13s·PII straddle 보존. forward + bidirectional.
- **종결어미 휴리스틱** (`app/services/korean_sentence_ending.py`): 종결/연결/조사 판정.
- **진단 스크립트** (`scripts/analysis/stt_segmentation_audit.py`): read-only Supabase + dry-run.
- **리포트** (`scripts/analysis/stt_segmentation_quality_audit_20260525.md`).
- **config** (`app/config.py`): `MERGE_V2_*` 임계값 4개.
- 테스트: v2 15 + v1 회귀 21 = **36 passed**.

## 2. 핵심 수치 (dry-run, DB 미반영)

| 모드 | 발화 감소 | 짧은 단편 감소 |
|---|---|---|
| forward (명세) | −1.1% | −3.8% |
| **bidirectional (권장)** | **−3.3%** | **−11.0%** |

## 3. 다음 단계 (각각 별도 승인 게이트)

- [ ] **worker wiring**: `docs/stt-segmenter-v2-wiring-plan.md` 승인 → 별도 PR. flag OFF 기본.
- [ ] hotwords/INITIAL_PROMPT IT보안 도메인 + 수석님 교정 (분리 트랙).
- [ ] short-heavy 세션 한정 단계적 재처리 (분리 트랙).

## 4. 주의사항

- working tree에 **무관한 PII 트랙 변경**(pii_masker.py 등) 존재 — 본 커밋에 미포함, 섞지 말 것.
- `test_v3_pii_regression.py` 1건 실패는 **PII 트랙 소관**, 본 트랙과 무관(기록만).
- v2 wiring 시 PII interval은 `segment()` 시점에 없음 → straddle 보호는 신규 처리에서 무의미(병합은 절단 안 함). 계획서 1절 참조.
- 기존 utterance row 재처리는 금지(신규 처리 전용). 백필은 별도 과제.
