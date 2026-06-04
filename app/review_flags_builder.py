# -*- coding: utf-8 -*-
"""검수 소프트플래그 빌더 — 발화 텍스트에서 사람 검수 신호를 수집한다.

PII 자동마스킹([PII_*])과 구분: 여기 플래그는 **마스킹이 아니라 검수 대기 신호**다.
review_flags(JSONB 배열) + review_priority_score(정렬키)로 DB에 적재되어
어드민 레드큐로 surfacing 된다. 설계: docs/design_review_panel_redesign_20260603.md §4,§6.

1차 탐지기(이미 보유): 호격(이름+아/야, 저우선) + Nim-Guard(사물+님, 확신형 환각).
저신뢰(word.probability)는 백엔드 적재(P1-a/b) 후 추가. 반복/루프 환각은 이미
text_quality 가 *수정*하므로 중복 플래그하지 않는다.
"""
from app.ner_guard import detect_inanimate_honorific, review_flags

# severity → 우선순위 가중치. priority_score = Σ(가중치).
SEVERITY_WEIGHT = {"low": 1, "med": 3, "high": 5}


def build_utterance_review_flags(text: str) -> tuple[list[dict], int]:
    """발화 텍스트에서 검수 플래그 배열과 우선순위 점수를 산출한다. immutable.

    반환: (flags, priority_score)
      flags = [{"type","severity","detail","span"?}, ...]
      priority_score = Σ severity 가중치 (0 = 검수불요).
    """
    flags: list[dict] = []

    # 호격(이름+아/야) — 자동마스킹 불가(소리야류 FP 혼재), 저우선 검수.
    for hit in review_flags(text or ""):
        flags.append({
            "type": "vocative",
            "severity": "low",
            "detail": f"호격 추정: {hit.text}",
            "span": [hit.start, hit.end],
        })

    # 사물+님(Nim-Guard) — 통계신호 못잡는 확신형 환각(공인인증서님).
    for token in detect_inanimate_honorific(text or ""):
        flags.append({
            "type": "object_nim",
            "severity": "med",
            "detail": f"사물+님 확신형환각 추정: {token}",
        })

    score = sum(SEVERITY_WEIGHT.get(f["severity"], 0) for f in flags)
    return flags, score
