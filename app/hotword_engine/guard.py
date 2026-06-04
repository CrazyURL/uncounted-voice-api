# -*- coding: utf-8 -*-
"""Token Guard — 오교정 차단 안전규칙.

플랫 글로벌 리스트가 야기하는 무차별 환각 오염(고빈도어 강제 치환)을
상치 단계에서 컷한다. 정본 설계: docs/design_review_panel_redesign_20260603.md §5.
"""

# 고빈도 공통어 — 어떤 경우에도 교정 대상이 되면 안 됨.
COMMON_WORDS = frozenset(
    "네 응 어 그 거 저 음 예 아 좀 뭐 이 저기 그냥 근데 그래서 그러니까 그렇죠".split()
)

# 최소 교정 길이 (단일/초단 글자 교정 금지).
MIN_CORRECT_LEN = 2


def is_correctable(
    wrong: str,
    *,
    common_words: frozenset = COMMON_WORDS,
    min_len: int = MIN_CORRECT_LEN,
) -> bool:
    """교정 대상어(wrong)가 자동 치환해도 안전한가.

    - 최소길이 미만 → 금지 (그·네 등)
    - 공통어 블록리스트 → 금지
    """
    w = (wrong or "").strip()
    if len(w) < min_len:
        return False
    if w in common_words:
        return False
    return True
