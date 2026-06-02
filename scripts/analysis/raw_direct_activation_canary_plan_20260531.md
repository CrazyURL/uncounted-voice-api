# raw_direct activation canary plan (read-only 계획)

작성: 2026-05-31
대상 코드: b3c338e (#21 raw_direct) + #22 None hardening 포함
전제: 본 문서는 **계획만**. activation/restart/canary/재처리 미실행.

## 목적
SPEAKER_MAPPING_MODE=raw_direct 활성 시 whisperx 대비 화자매핑 품질(back-channel/overlap 보존)이 개선되는지, None corruption·회귀 없는지 단건 canary로 검증.

## activation 절차 (canary 실행 시)
1. `voice-api@dev.service.d/speaker-mapping.conf` 작성: `Environment=SPEAKER_MAPPING_MODE=raw_direct`
2. daemon-reload
3. voice-api restart (worker는 선택)
4. health 확인 (model large-v3, loaded, 미설정/건너뜀 경고 0)
5. canary 3세션 단건 재처리 (whisperx 기준값과 비교)

## canary 대상 3세션 + 적합성
| seq | id | status | uc | speakers | canary 적합성 |
|-----|-----|--------|-----|----------|--------------|
| #192136 | d0212e244f08e0 | done | 3 | 2 | ✅ 적합 (2화자, back-channel/overlap 관찰 가능) |
| #191312 | ba059bf0f5b874 | done | **0** | **0** | ⚠️ **부적합** (빈 통화, uc0/spk0 — 화자매핑 비교 대상 없음) |
| #192163 | 5e0185212bd409 | done | 1 | 1 | △ 약함 (단일화자, mixed/overlap 관찰 약함) |

→ **#191312는 빈 세션이라 화자매핑 검증 불가**. 대체 후보(2화자 이상, back-channel 가능성 있는 세션) 권장: 예) pii_hit 454dd82f(spk2,uc70), long 0f2dba3d(spk2,uc359) 등 multi-speaker 통화.

## 측정 항목 (whisperx baseline vs raw_direct after)
| 항목 | 측정 방법 | 기대(raw_direct) |
|------|-----------|------------------|
| mixed speaker ratio | 발화 내 speaker_id 혼재 비율 | 변화 관찰 |
| overlap preservation | 겹친 발화 보존 수 | whisperx보다 ↑ (목적) |
| back-channel recall | 짧은 맞장구 발화 포착 수 | ↑ (목적) |
| speaker turn count | session_speakers 턴 수 | 비교 |
| WER/CER | (전사 텍스트 자체는 모델 동일 large-v3 → 변화 미미 예상) | ≈ |
| hallucination | 비정상 반복/환각 발화 | 0 유지 |
| None speaker contamination | speaker_id="None"/"SPEAKER_UNKNOWN" 수 | **0** (PR#22 coerce) |
| orphan delta | 재처리 전후 orphan 변화 | 자동 cleanup |

## 중단 조건 (canary 중 즉시 HOLD/rollback)
- failed > 0
- NRestarts 증가 (api 0 / worker 1 기준)
- speaker_id == "None" (raw 문자열) DB 적재
- orphan 급증
- queue stall
- VRAM ≥ 7.6GB 지속

## rollback
- drop-in 제거 또는 =whisperx → daemon-reload + voice-api restart → 즉시 whisperx 복귀 (~15s)
- canary 세션이 raw_direct로 재처리됐어도 whisperx로 재재처리하면 원복 (단 in-place 덮어쓰기 주의 — before 스냅샷 필수)

## 측정 한계 (baseline doc과 동일)
- 모델구분/매핑모드 필드 없음 → whisperx vs raw_direct 행단위 diff는 before 스냅샷 보존 필수
- in-place 재처리라 whisperx 원본은 재처리 직전 캡처해야 비교 가능

## 산출 예정 (canary 승인·실행 시)
- whisperx vs raw_direct 세션별 측정표
- None contamination 0 검증
- GO/HOLD 판정
