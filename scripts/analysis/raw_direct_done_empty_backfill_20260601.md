# raw_direct done-empty 대규모 backfill — 실측 결과 (완료)

작성: 2026-06-01
코드: baed23b, model=large-v3, SPEAKER_MAPPING_MODE=raw_direct ACTIVE (.env.dev:66)
대상: done-empty 잔여 81세션 (done + utterance_count=0 + utterance rows=0 + raw_audio 보유, 이미 처리한 45세션[초기3+확대canary22+PhaseA20] 제외). multi_turn 일반세션 제외, diar_fail 별도확장 없음.
원칙: concurrency=1, env 추가변경 없음, worker/api 불필요 restart 없음, 20건 단위 체크포인트, 중단조건 자동 ABORT, 전부 DB 실측.

> 주: 승인 시 "잔여 99건"으로 표기됐으나 실제 모집단은 81건(PhaseA 18 처리분 반영 후 실측). 99는 PhaseA 처리 전 추정치.

## 최종 KPI (DB 실측, 81 대상 전수)
| 항목 | 값 |
|---|---|
| **복원률** | **81 / 81 = 100%** (rows>0) |
| **복원 utterance 총량** | **5,123** (min 5 / max 264) |
| **speaker count 분포** | **전부 2명 {2: 81}** (1명·None배정 0) |
| **None corruption (raw "None")** | **0** |
| none_all (None/null/UNKNOWN) | **0** |
| back-channel (≤0.7s recall) | 371개 |
| global failed | **0** |
| done_empty(uc=0) 전역 잔량 | **0** (110 → 0) |

## 진행 경과 (2-pass + worker 자동복구)

### 1차 (57시도 → 56복원, ABORT)
- 처리 56/56 = 100% 복원, 복원 utterance 1,512, 전부 2-speaker, None 0.
- latency: min 40.4 / max 131.6 / **avg 61.8s**, VRAM peak 6,336 MiB.
- **ABORT 발생**: seq191247 (dur 617s 장문) → `Server disconnected` (worker↔api HTTP 단발 끊김, **데이터 corruption/코드결함 아님**) → status=failed → 설계된 중단조건(global failed 0→1) 정확 작동 → 자동 정지.
- ABORT 시점 안전성 전 GREEN (None 0, 기존 56건 무영향, OOM 0, restart 0).

### seq191247 — worker 자동복구
- voice-worker@dev (Restart=always) 정책으로 failed 건을 자동 재시도 → **status=done, utterance_count=83, retry=1**.
- 단발 네트워크 이벤트였음이 입증됨(재처리로 정상 복원). 이 복구는 운영 worker의 자동 동작이며 backfill 러너가 의도한 것이 아님(사실 그대로 기록).

### 2차 재개 (사용자 승인: "seq191247 제외 재개")
- 멱등 재선정(현재 done & uc=0 & rows=0 만) → 24건 대상, seq191247 EXCL.
- **처리 24/24 = 100% 복원**, 복원 utterance 3,528, 전부 2-speaker, None 0.
- CHECKPOINT@20: 20/20 복원 100%, VRAM peak 6,462 MiB, latency avg 185.4s.
- FINAL: latency min 101.4 / max 344.6 / **avg 207.8s** (막바지 dur 1,000s+ 장문 집중 구간 — dur 오름차순 정렬상 뒤쪽이 장문이라 건당 latency 상승, 정상).
- ABORT=None.

## 안전성 — 전 항목 GREEN (1차+2차 통산)
- **None corruption 0 / none_all 0** (81세션 전수, speaker_mapping.py overlap→None 케이스 미발생)
- **데이터 corruption 0** — 기존 복원분 무영향, 멱등 재처리로 중복 rows 없음
- global failed 최종 **0** (1차 중 1건 발생했으나 worker 자동복구)
- VRAM peak **6,462 MiB** / 8,188 (< 7.6GB 임계, OOM 0)
- api NRestarts 0 (base 0), worker NRestarts 1 (base 1) — restart anomaly 없음
- false_split / runaway **0** — max 264발화도 장문(dur 비례) 정상치, 과분할 패턴 없음
- multi_turn 과병합 **재현 안 됨** (done-empty는 wb=0 복원형이라 병합 리스크 구조적으로 없음)
- concurrency=1 유지(running 항상 ≤1), raw_direct ACTIVE 유지, env 무변경

## 잔여 판정
- **done_empty(uc=0) 전역 잔량 = 0.** 오디오 보유 done-empty 전량 복원 완료.
- 진짜 빈통화(오디오 없는 done-empty) 별도 잔존 없음 — 전역 0 도달로 확인.

## 누적 (전체 raw_direct done-empty 복원 트랙)
- 확대 canary 8 + Phase A 18 + 본 backfill 81(1차56+2차24+자동복구1) = **done-empty 복원 통산 107세션**
- done_empty(uc=0): 110 → **0**
- 통산 None corruption 0, failed 최종 0, OOM 0, restart anomaly 0

## 결론: **raw_direct done-empty backfill 성공 (GO 입증 완료)**
1. **복원률 100% (81/81), 전역 done_empty 0 도달.** whisperx가 통째 누락시킨 통화를 raw_direct가 예외 없이 복원.
2. **안전성 완전 통과** — None 0, corruption 0, OOM 0, restart 0, 과분할 0.
3. 유일 이벤트(seq191247 Server disconnected)는 인프라성 단발 오류였고 worker 자동복구로 해소 + 중단조건이 정확히 작동함을 실증.
4. **장문 구간 latency 주의**: dur 1,000s+ 통화는 건당 200~345s. 대량 장문 재처리 시 시간 예산 반영 필요(안전성에는 무영향).

## 후속 (미승인, 별개 판단 대상)
- multi_turn(일반 다발화) 대규모 재처리는 과병합 우려로 본 트랙과 **분리** — 별도 승인 필요
- raw_direct 정식 확정 여부는 운영 정책 결정 사항
