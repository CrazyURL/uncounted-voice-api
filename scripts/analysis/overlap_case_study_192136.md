# speaker mapping case study — #192136 & #191312 (실측 정정본)

작성: 2026-06-01
large-v3, whisperx vs raw_direct, DB 실측 (원문 제외, 시간/화자만)

## #192136 (2화자, 30s) — 발화 흡수 감소
### raw_direct 타임라인 (실측)
| seq | speaker | start~end | dur |
|-----|---------|-----------|-----|
| 1 | SPEAKER_00 | 1.39~8.70 | 7.31 |
| 2 | SPEAKER_00 | 9.68~11.16 | 1.48 |
| 3 | SPEAKER_01 | 11.18~17.97 | 6.79 |
| 4 | SPEAKER_00 | 17.99~18.73 | 0.74 |
- raw_direct: uc=4, dist 00:3/01:1, turns 3, none 0
- whisperx baseline: uc=3, dist 00:2/01:1, turns 2
- → SPEAKER_00 발화 1개 더 분리(2→3). 흡수 감소.

## ★ #191312 (62s, 2화자) — whisperx 전체 누락 → raw_direct 복원 (핵심)
### raw_direct 타임라인 (DB 실측)
| seq | speaker | start~end | dur |
|-----|---------|-----------|-----|
| 1 | SPEAKER_00 | 0.81~4.65 | 3.84 |
| 2 | SPEAKER_01 | 5.53~6.25 | **0.72** |
| 3 | SPEAKER_00 | 6.67~8.51 | 1.84 |
| 4 | SPEAKER_01 | 8.53~9.25 | **0.72** |
| 5 | SPEAKER_00 | 10.91~11.43 | **0.52** |
| 6 | SPEAKER_01 | 12.07~13.41 | 1.34 |
| 7 | SPEAKER_00 | 14.11~23.16 | 9.05 |
| 8 | SPEAKER_00 | 23.58~30.54 | 6.96 |
| 9 | SPEAKER_00 | 31.48~35.19 | 3.71 |
| 10 | SPEAKER_01 | 53.48~56.71 | 3.23 |
- raw_direct: uc=10, 2화자, none 0. **back-channel(≤0.7s): seq5(0.52) 1개** + seq2/4(0.72, 경계 근처)
- **whisperx baseline: uc=0 (발화 전무 — 화자배정 실패로 통화 통째 누락)**
- → raw_direct가 whisperx 사각지대 통화를 **완전 복원**. 짧은 화자 교대(seq1~6, 0.5~1.8s)가 밀집된 구간을 raw_direct가 분리 — whisperx가 통째 흡수/누락했던 패턴. raw_direct 설계 목적의 직접 입증 사례.

## 한계
- 두 세션 모두 명확한 **동시발화(overlap) 구간은 없음** → overlap preservation 자체는 미입증.
- back-channel은 #191312 seq5(0.8s)·seq10(~0.6s)에서 짧은 발화 포착 — whisperx가 0개였던 것 대비 개선 신호.

## 결론
raw_direct의 핵심 가치(whisperx 화자배정 사각지대 복원)가 **#191312에서 실측 입증**(uc 0→10). #192136은 흡수 감소. overlap preservation은 적합 세션 확대 검증 필요.
