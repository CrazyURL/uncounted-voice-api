# -*- coding: utf-8 -*-
"""도메인 프로파일 (immutable 데이터).

각 프로파일 = (B용 발음페어링) + (D용 큐레이트 혼동쌍) + (문맥게이트 키워드).
정본 설계: docs/design_review_panel_redesign_20260603.md §5.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    """도메인 1종의 핫워드/교정 규칙. frozen → immutable."""

    name: str
    # B: initial_prompt 부착용. (약어, 발음). 발음 빈 문자열이면 용어만 표기.
    phonetic_pairs: tuple[tuple[str, str], ...]
    # D: 큐레이트 혼동쌍. (오인식어, 정답어). 일반 자모거리 아님 — 정밀 큐레이션.
    confusion_pairs: tuple[tuple[str, str], ...]
    # 문맥 prior: 아래 키워드가 min_keywords개 이상 등장 → 이 도메인 통화로 판정.
    context_keywords: tuple[str, ...]
    min_keywords: int = 2


# IT 보안 지원 통화 프로파일 (sess3 등 실측 검증 도메인)
IT_SECURITY = DomainProfile(
    name="it_security",
    phonetic_pairs=(
        ("DLP", "디엘피"),
        ("NAC", "엔에이씨"),
        ("EPP", "이피피"),
        ("DRM", "디알엠"),
        ("OA망", ""),
        ("공동인증서", ""),
        ("예외정책", ""),
        ("팝업창", ""),
    ),
    confusion_pairs=(
        ("선생님", "수석님"),
    ),
    context_keywords=(
        "보안", "DLP", "NAC", "공동인증서", "회의실",
        "책임", "정책", "프로그램", "결재", "품의", "수석",
    ),
    min_keywords=2,
)

_PROFILES = {p.name: p for p in (IT_SECURITY,)}


def get_profile(name: str | None) -> DomainProfile | None:
    """프로파일명으로 조회. 없거나 빈 이름이면 None (= 엔진 비활성)."""
    if not name:
        return None
    return _PROFILES.get(name)
