# raw_direct 확대 canary (22세션) — 실측 결과

작성: 2026-06-01
코드: b3c338e (#21 raw_direct + #22 None hardening), model=large-v3, SPEAKER_MAPPING_MODE=raw_direct ACTIVE
방식: 카테고리별 22세션 in-place 재처리, whisperx baseline(재처리 직전 캡처) vs raw_direct 비교. 전부 DB 실측.

## 안전성 — 전부 PASS
- **None corruption(raw "None") 0, none_all 0** (22세션 전체) — PR#22 coerce 작동
- failed 증가 0 (global failed 0 유지), running 0 복귀
- api NRestarts 0 (base 0), worker NRestarts 1 (base 1) — restart anomaly 없음
- VRAM peak **6,346 MiB** / 8,188 (< 7.6GB 임계), OOM 0
- speaker collapse 없음, runaway segmentation 없음(false_split 징후 0)
- latency: min 20.3s / max 60.6s / avg 37.7s (large-v3 기준)
- ABORT=None

## KPI (최우선)
### ★ done-empty 복원률: **8/8 = 100%**
- whisperx uc=0 (전부) → raw_direct af_rows = [1,1,3,1,4,1,1,3]
- 8세션 모두 발화 복원. 화자도 0→1 또는 0→2 정상 배정. back-channel 2세션에서 포착(0→1).
- **whisperx.assign_word_speakers가 통째 누락시킨 통화를 raw_direct가 전수 복원** = raw_direct 핵심 가치 입증.

### ★ diar_fail_suspect 복원: **4/4**
- wb_rows [5,5,0,0] → af_rows [6,6,13,8]
- 특히 seq191297 (uc0→13, spk0→2, turns0→12), seq192099 (uc0→8, spk0→2) — whisperx 누락 통화를 큰 폭 복원.

### uc<=2 정상화: **0/6** (해석 주의)
- af_rows [1,1,1,4,1,1] — 대부분 1발화 유지(진짜 1발화 통화로 추정), seq192094는 2→1 감소.
- 이 카테고리는 **실제로 발화가 적은 짧은 통화**라 raw_direct로도 변화 없음이 정상(오류 아님). done-empty/diar_fail과 달리 whisperx가 이미 정상 처리한 케이스.

## false split / false merge
- **false_split 징후: 0** (dur≤30 & af_rows>10 케이스 없음 — 과분할 안 함) ✓
- **false_merge 징후: 1** — seq192323 (multi_turn): wb_rows 26→16, turns 20→12. raw_direct가 발화를 더 합침(<0.7×wb). ⚠️ 이 1건은 raw_direct가 whisperx보다 적게 분할 — back-channel도 3→1 감소. **유일한 부정 신호**(과병합 의심).
- 나머지 multi_turn 3건: seq192008(27→23), seq195955(22→22), seq192219(21→20) — 소폭 감소, turn 대체로 유지.

## 종합 방향 (22세션)
- uc 증가(복원) 12 / 감소 4 / 동일 6
- 증가 12건은 대부분 done-empty·diar_fail(whisperx 누락 복원)
- 감소 4건은 multi_turn(raw_direct가 과분할 완화 or 과병합) — seq192323만 과병합 우려
- back-channel 총: whisperx 12 → raw_direct **16** (+4, 짧은 추임새 recall 개선)
- speaker count: done-empty/diar_fail 10건에서 0→1·0→2 정상 배정

## overlap preservation
- multi_turn 4건이 overlap 가능 후보였으나, timeline상 명확한 동시발화 케이스는 본 표본에서 분리 측정 안 됨(speaker_id 단일 라벨 기준). raw_direct가 turn을 대체로 보존(192219 16→16, 192008 21→18)하나 overlap 정량화는 별도 word-level 분석 필요.

## 운영 ON 유지 추천: **조건부 YES**
근거:
1. **안전성 완전 통과** (None 0, failed 0, OOM 0, restart 0, false_split 0) — 운영 리스크 없음.
2. **done-empty/diar_fail 복원 12/12** = whisperx 사각지대(누락 통화)를 raw_direct가 메우는 명확한 ROI. done-empty 109건·diar_fail 85건 모집단에 적용 시 대규모 데이터 복원 기대.
3. 우려 1건: seq192323 과병합(false_merge) — multi_turn(긴 다발화)에서 raw_direct가 발화를 합치는 경향 일부. 단 1/4건, 치명적 아님.

권고:
- **raw_direct ON 유지 권장** (whisperx 누락 복원 가치가 과병합 리스크보다 큼).
- 단 **multi_turn 과병합**은 후속 관찰 필요 — SPEAKER_MAP_TOLERANCE/OVERLAP_MIN 파라미터 튜닝 여지.
- done-empty 109 + diar_fail 85 모집단 backfill은 별도 승인 게이트(대규모 재처리).

## 데이터 영향
- 22세션 in-place 재처리됨(whisperx→raw_direct 결과로 갱신). global done 583 불변. orphan 영향은 cleanup 운영화로 자동 처리.
