# 평가 하베스트 (frozen gold set)

"모든 통화를 테스트"하는 게 아니라 **대표 샘플 30통화에 한 번 정답을 달아 얼리고(frozen)**, 모든 파이프라인 버전을 같은 잣대로 채점한다. (수능: 시험지 고정 → 버전 간 비교 가능)

## 워크플로
```
1. select_sample.py        # 대표 30통화 층화선정(길이×품질) → sample_manifest.json
2. bootstrap_annotations.py # DB의 STT/화자/PII후보 미리채움 → gold/<seq>.json
3. [사람] 교정              # text_gold(STT오류)·speaker_gold·pii_missed 수정 + reviewed=true
4. grade.py                # 고정 gold에 현재 파이프라인 채점 → WER/DER/PII recall 점수표
```

## 핵심 원칙
- **사람은 "교정"만** — 빈손 전사 X. 기계가 80% 맞춘 걸 고치므로 통화당 ~15-20분.
- **gold 는 버전 독립** — 사람이 검증한 진실. 어떤 파이프라인 버전을 채점해도 같은 정답.
- **1회 투자(사람 ~8-10h) → 영원히 측정.** 파이프라인 바뀔 때마다 grade.py 만 재실행.

## 메트릭 & 텔레포니 바
| 메트릭 | 도구 | 바(텔레포니/de-id 표준) |
|---|---|---|
| WER | jiwer (PII 정규화) | 10-20% |
| DER | pyannote.metrics | 5-15% |
| PII recall | gold pii_missed | ≥98% + review queue |

## 주의
- bootstrap 은 재처리된 최종 DB 기준 권장(STT 시드가 최신 파이프라인). 재처리 완료 후 실행.
- DER 은 발화 시간구간 기준(화자 라벨 혼동 중심). 경계까지 보는 완전 DER 은 독립 turn-time 주석 필요(후속).
