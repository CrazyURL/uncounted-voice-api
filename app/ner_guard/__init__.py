# -*- coding: utf-8 -*-
"""정적 사전 기반 PII 이름 가드 (하이브리드).

설계: docs/design_review_panel_redesign_20260603.md §6.
- 풀네임(성+이름) → 자동 마스킹 (FP 0 실측)
- 호격(이름+아/야) → 검수 플래그
- 사물+님(Nim-Guard) → 검수 플래그 (확신형 환각)
"""
from app.ner_guard.detector import (
    NameHit,
    auto_mask_names,
    detect_name_hits,
    review_flags,
)
from app.ner_guard.honorific_guard import detect_inanimate_honorific

__all__ = [
    "NameHit",
    "auto_mask_names",
    "detect_name_hits",
    "review_flags",
    "detect_inanimate_honorific",
]
