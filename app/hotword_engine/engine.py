# -*- coding: utf-8 -*-
"""B+D 핫워드 엔진 코어.

- B: build_domain_prompt — 발음페어링을 initial_prompt에 부착
- 문맥게이트: detect_domain — 통화 전체 텍스트의 도메인 키워드 빈도 판정
- D: correct_confusions — 문맥+Token Guard+세션사전 충족 시에만 혼동쌍 교정

family-safe: 문맥 미충족(키워드<min) → 무변경. immutable: 입력 미변형, 신규 반환.
정본 설계: docs/design_review_panel_redesign_20260603.md §5.
"""
import re

from app.hotword_engine.guard import is_correctable
from app.hotword_engine.profiles import DomainProfile


def build_domain_prompt(base_prompt: str | None, profile: DomainProfile | None) -> str | None:
    """B: base_prompt 뒤에 도메인 발음페어링 부착. 멱등(중복 부착 방지).

    profile 없으면 base 그대로 반환 (엔진 OFF = byte-identical).
    """
    if profile is None or not profile.phonetic_pairs:
        return base_prompt
    terms = ", ".join(f"{a}({b})" if b else a for a, b in profile.phonetic_pairs)
    addition = f" 보안 IT 용어: {terms}."
    base = base_prompt or ""
    if addition.strip() in base:
        return base or None
    return (base + addition).strip()


def detect_domain(full_text: str, profile: DomainProfile | None) -> bool:
    """문맥 prior: 도메인 키워드가 min_keywords개 이상이면 True."""
    if profile is None:
        return False
    text = full_text or ""
    hits = sum(1 for kw in profile.context_keywords if kw in text)
    return hits >= profile.min_keywords


def _correct_text(text: str, pairs: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    """좌경계만 적용(우경계 제거 → 조사 허용: 선생님이→수석님이). 신규 문자열 반환."""
    out = text or ""
    total = 0
    for wrong, right in pairs:
        pattern = re.compile(rf"(?<![가-힣]){re.escape(wrong)}")
        out, n = pattern.subn(right, out)
        total += n
    return out, total


def correct_confusions(
    segments: list[dict],
    profile: DomainProfile | None,
    session_dict: frozenset | set | None = None,
) -> tuple[list[dict], int]:
    """D: 세그먼트 텍스트의 혼동쌍 교정. (신규 segments, 치환건수) 반환.

    게이트(전부 통과해야 발동):
      1. profile 존재 + 혼동쌍 존재
      2. detect_domain(통화 전체) — family-safe 핵심
      3. Token Guard(is_correctable) — 고빈도/단일어 컷
      4. 세션사전: session_dict 제공 시 정답어가 사전에 존재할 때만
         (None = v1, 프로파일 자체를 승인 권위로 간주)

    immutable: 입력 segments 미변형, 변경분만 신규 dict로 복사.
    """
    if profile is None or not profile.confusion_pairs:
        return list(segments), 0

    full_text = " ".join(s.get("text", "") for s in segments)
    if not detect_domain(full_text, profile):
        return list(segments), 0

    active = tuple(
        (wrong, right)
        for wrong, right in profile.confusion_pairs
        if is_correctable(wrong) and (session_dict is None or right in session_dict)
    )
    if not active:
        return list(segments), 0

    new_segments: list[dict] = []
    total = 0
    for seg in segments:
        new_text, n = _correct_text(seg.get("text", ""), active)
        total += n
        new_segments.append({**seg, "text": new_text} if n else {**seg})
    return new_segments, total
