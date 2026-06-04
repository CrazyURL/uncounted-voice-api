# -*- coding: utf-8 -*-
"""B+D 도메인 핫워드 엔진 (env-gate, 기본 OFF).

정본 설계: docs/design_review_panel_redesign_20260603.md §5.
"""
from app.hotword_engine.engine import (
    build_domain_prompt,
    correct_confusions,
    detect_domain,
)
from app.hotword_engine.profiles import DomainProfile, get_profile

__all__ = [
    "build_domain_prompt",
    "correct_confusions",
    "detect_domain",
    "DomainProfile",
    "get_profile",
]
