# raw_direct Phase A 복원 (20세션) — 실측 결과

작성: 2026-06-01
코드: baed23b (#23 R2a 포함, R2a runtime 미연결 — raw_direct 로직 무관), model=large-v3, SPEAKER_MAPPING_MODE=raw_direct ACTIVE
구성: done-empty 18 + diar_fail 2 = 20 (확대 canary 22 + 초기 3 제외, raw_audio 보유만, multi_turn 제외). 전부 DB 실측.

## 안전성 — 전부 PASS
- **None corruption(raw "None") 0, none_all 0** (20세션 전체)
- failed 증가 0 (global failed 0 유지), running 0 복귀
- api NRestarts 0 (base 0), worker NRestarts 1 (base 1) — restart anomaly 없음
- VRAM peak **6,278 MiB** / 8,188 (< 7.6GB 임계), OOM 0
- false_split 징후 0 (dur≤60 & af_rows>30 케이스 없음 — runaway 없음)
- ABORT=None
- latency: min 20.3 / max 40.7 / avg 35.5s

## KPI
### ★ done-empty 복원률: **18/18 = 100%**
- whisperx uc=0 (전부 18건) → raw_direct af_rows = [1,2,3,3,1,1,1,1,4,5,2,9,5,6,5,8,15,9]
- **18세션 전원 발화 복원.** 화자도 0→1(8건) 또는 0→2(10건) 정상 배정.
- 복원 발화 분포: 1발화 6건 / 2~5발화 7건 / 6~9발화 4건 / 15발화 1건(seq191102). 통화 길이/내용에 비례한 자연스러운 복원.
- back-channel(≤0.7s): whisperx 0 → raw_direct **15개** 포착 (짧은 추임새 recall).

### diar_fail (2건): wb 유지
- wb_rows [3,4] → af_rows [3,4] (변화 없음). 이 2건은 whisperx가 이미 일부 발화를 잡은 케이스라 raw_direct로도 동일. seq192376 back-channel 0→1 소폭 개선.

## false split / merge
- **false_split: 0** (과분할 없음). 최대 복원이 15발화(seq191102, dur 45s)로 합리적 범위.
- false_merge: 해당 없음 (done-empty는 wb=0이라 병합 대상 없음; multi_turn 제외했으므로 과병합 패턴 미발생).
- **중단조건 "multi_turn 과병합 패턴이 done-empty/diar_fail에서 반복" → 발생 안 함** ✓

## 누적 (확대 canary + Phase A)
- done-empty 복원: 확대 8/8 + Phase A 18/18 = **26/26 (100%)**
- diar_fail 복원: 확대 4/4 + Phase A 0/2(이미 처리된 케이스) — 누락형(wb=0)은 전수 복원, 부분처리형은 유지
- None corruption 누적 0, failed 누적 0, OOM 0, restart anomaly 0

## 대규모 backfill GO/HOLD 판단: **GO (조건부)**
근거:
1. **done-empty 복원 26/26 = 100%** (확대+PhaseA 합산). whisperx가 통째 누락시킨 통화를 raw_direct가 예외 없이 복원.
2. **안전성 완전 통과** 2회 연속 (None 0, failed 0, OOM 0, restart 0, false_split 0, runaway 0).
3. 우려였던 multi_turn 과병합은 done-empty/diar_fail 모집단에서 **재현 안 됨** (해당 카테고리는 복원형이라 병합 리스크 구조적으로 없음).

권고:
- **done-empty 99건(잔여) 대규모 backfill GO.** 복원율·안전성 충분 입증.
- diar_fail은 모집단 거의 소진(잔여 0~2). 별도 트랙 불필요.
- ⚠️ multi_turn(일반 다발화) 대규모 재처리는 **별개 판단** — 과병합 우려(확대 canary seq192323)가 있어 done-empty backfill과 분리해야 함.
- 대규모 backfill 시: 배치 모니터(None/failed/VRAM/false_split), worker concurrency=1 유지(VRAM peak 6.3GB+여유 빠듯), 배치 단위 체크포인트 권장.

## 데이터 영향
- 20세션 in-place 재처리(whisperx→raw_direct). global done 583 불변. done-empty 110 → (26건 복원으로 점차 감소, 단 이 집계는 utterance_count 기준이라 backfill 진행 시 갱신).
