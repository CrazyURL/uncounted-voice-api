# raw_direct activation report (Phase 3 canary) — 실측 정정본

작성: 2026-06-01 (DB 실측, 단일 실행 캡처)
대상: b3c338e (#21 raw_direct + #22 None hardening), model=large-v3
활성화: `.env.dev`에 `SPEAKER_MAPPING_MODE=raw_direct` + voice-api restart (worker 무접촉)

## Activation (실측)
- `.env.dev` 백업(.env.dev.bak_p3_*) 후 `SPEAKER_MAPPING_MODE=raw_direct` 1줄 추가
- voice-api restart: api MainPID 3253778(NR 0), worker MainPID 3166906 **불변**(NR 1) ✓
- proc env + .env.dev 양쪽 raw_direct 반영 확인 ✓

## Canary 3세션 (whisperx baseline 캡처 → raw_direct 재처리 → 비교, 단일 실행 실측)
처리시간 STEP2 ~30초(15:48:38→15:49:08, 3세션 done).

| seq | whisperx (uc/spk/turns/backch) | raw_direct (uc/spk/turns/backch/none_raw) | 해석 |
|-----|--------------------------------|--------------------------------------------|------|
| #192136 | 3 / 2 / 2 / 0 | 4 / 2 / 3 / 0 / 0 (dist 00:3,01:1) | 발화 +1, 흡수 감소 |
| **#191312** | **0 / 0 / 0 / 0** | **10 / 2 / 8 / 1 / 0** (dist 00:6,01:4) | ★ whisperx 전체누락 → raw_direct 복원 |
| #192163 | 1 / 1 / 1 / 0 | 1 / 1 / 1 / 0 / 0 | 단일화자, 동일 |

## ★ 핵심 발견 — #191312
- **whisperx: uc=0** (62초 2화자 통화인데 화자배정 실패로 발화 0개 = 데이터 완전 누락)
- **raw_direct: uc=10, 2화자, 8 turns, back-channel(≤0.7s) 1개** (정상 복원)
- whisperx.assign_word_speakers가 통째로 떨어뜨린 통화를 raw pyannote direct mapping이 **복원** → raw_direct 설계 목적(whisperx 화자배정 사각지대 우회)의 **직접 입증 사례**.

## 중단조건 — 전부 미해당 (안전 PASS, 실측)
- failed 0 · **speaker_id="None"(raw) 0, none_all 0** (PR#22 coerce 작동, 3세션 합산) · api NRestarts 0 · worker NR 1 불변 · VRAM 4,668 MiB(<7.6GB) · queue idle · orphan 불변 · global done 583 불변

## 판정: 안전성 GO + 품질 긍정 신호 (단 표본 소)
- **안전**: None corruption 0, failed 0, 무중단, OOM 없음.
- **품질**: #191312에서 whisperx 누락 통화 복원(uc 0→10) = raw_direct 가치 입증. #192136 발화 흡수 감소(+1). #192163 단일화자 동일.
- ⚠️ 단 3세션 표본이라 일반화는 확대 canary 필요. 특히 #191312 같은 "whisperx 누락→raw_direct 복원" 패턴이 얼마나 흔한지(전체 done-empty/저utterance 세션 분포) 확인 가치 큼.

## 현재 상태
- **raw_direct ACTIVE 유지중** (.env.dev+restart). 품질 긍정 신호 있으나 표본 소 → 정식 ON 유지 vs 확대검증 결정은 사용자 판정.

## Rollback (~15s)
```
# .env.dev SPEAKER_MAPPING_MODE 줄 제거(또는 =whisperx); 백업 .env.dev.bak_p3_*
sudo systemctl restart voice-api@dev.service   # worker 무접촉
```
