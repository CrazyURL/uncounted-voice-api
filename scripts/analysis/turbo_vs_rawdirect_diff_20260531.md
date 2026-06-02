# turbo vs raw_direct diff (화자매핑) — 실측 정정본

작성: 2026-06-01
주의: 시기상 두 축 변경 — (a) turbo→large-v3 (PR#20), (b) whisperx→raw_direct (PR#21). 주 비교는 화자매핑 축(둘 다 large-v3). turbo 참고.
한계: 모델/매핑모드 구분 필드 없음 + in-place 재처리 → 행단위 transcript diff 불가. whisperx baseline은 raw_direct 재처리 직전 단일실행 캡처(실측).

## 화자매핑 축: whisperx vs raw_direct (large-v3, DB 실측)
| seq | whisperx uc/spk/turns/backch | raw_direct uc/spk/turns/backch | dist 변화 |
|-----|------------------------------|--------------------------------|-----------|
| #192136 | 3/2/2/0 | 4/2/3/0 | 00:2,01:1 → 00:3,01:1 |
| #191312 | **0/0/0/0** | **10/2/8/1** | {} → 00:6,01:4 |
| #192163 | 1/1/1/0 | 1/1/1/0 | 00:1 → 00:1 |

## 관찰
1. **#191312 (결정적)**: whisperx가 화자배정 실패로 **발화 0개**(62초 2화자 통화 완전 누락) → raw_direct가 **10발화/2화자/8턴/back-channel 1개** 복원. raw pyannote direct가 whisperx.assign_word_speakers 사각지대를 메움.
2. **#192136**: 발화 3→4, 흡수 감소(SPEAKER_00 분리 증가).
3. **#192163**: 단일화자, 변화 없음.

## back-channel / overlap
- back-channel(≤0.7s): whisperx 전 세션 0 → raw_direct #191312에서 **1개 포착**. raw_direct가 짧은 추임새를 잡는 첫 실측 신호.
- overlap(동시발화): 본 3세션엔 명확한 overlap 케이스 부족(timeline상 화자 구간 비중첩) → overlap preservation은 여전히 미검증.

## 결론
- raw_direct가 (a) whisperx 누락 통화 복원(#191312 uc0→10), (b) back-channel 1개 포착, (c) 발화 흡수 감소(#192136) — **3개 긍정 신호 실측**.
- 단 표본 3건(유효 2건)이라 통계 결론 불가. 확대 canary 필요.
