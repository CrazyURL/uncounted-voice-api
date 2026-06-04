# -*- coding: utf-8 -*-
"""ner_guard 유닛테스트 — 풀네임 자동마스킹 + 호격/사물님 검수플래그.

설계: docs/design_review_panel_redesign_20260603.md §6. 실측 12/12 케이스 회귀잠금.
"""
import pytest

from app.ner_guard import (
    auto_mask_names,
    detect_inanimate_honorific,
    detect_name_hits,
    review_flags,
)


class TestFullNameAutoMask:
    def test_masks_real_full_name(self):
        out, hits = auto_mask_names("김지민 책임님이 요청주신거거든요")
        assert "[PII_이름]" in out
        assert "김지민" not in out
        assert any(h.kind == "full" for h in hits)

    def test_preserves_particle(self):
        # 토큰 내 조사는 보존 — "김현정이" → "[PII_이름]이"
        out, _ = auto_mask_names("김현정이 검색해볼까")
        assert "[PII_이름]이" in out

    @pytest.mark.parametrize("noun", ["공인인증서님이", "공영주차장에서", "영수증 발급", "이유식 먹였어", "김해시청입니다"])
    def test_no_false_positive_on_common_nouns(self, noun):
        # 음절겹침 명사 → 마스킹 안 됨 (FP 0 실측)
        out, hits = auto_mask_names(noun)
        assert out == noun
        assert not [h for h in hits if h.kind == "full"]

    def test_multiple_names(self):
        out, hits = auto_mask_names("이영희 고객님과 박서연 사원")
        assert out.count("[PII_이름]") == 2


class TestVocativeReviewFlag:
    def test_vocative_flagged_not_masked(self):
        # 호격은 자동마스킹 X, 검수 플래그 O
        out, hits = auto_mask_names("민준아 이리 와")
        assert out == "민준아 이리 와"   # 마스킹 안 함
        assert any(h.kind == "vocative" for h in hits)

    def test_review_flags_returns_vocative_only(self):
        flags = review_flags("소리야 무슨 소리야")  # 소리야=겹침이지만 플래그(저우선)
        assert all(h.kind == "vocative" for h in flags)


class TestNimGuard:
    @pytest.mark.skipif(not detect_inanimate_honorific("공인인증서님 테스트 보안 정책"), reason="kiwipiepy 미설치")
    def test_flags_inanimate_honorific(self):
        flags = detect_inanimate_honorific("공인인증서님이 처리해달라고")
        assert "공인인증서님" in flags

    def test_no_crash_without_text(self):
        assert detect_inanimate_honorific("") == []


def test_detect_name_hits_immutable():
    text = "김지민 책임님"
    hits = detect_name_hits(text)
    assert text == "김지민 책임님"  # 입력 미변형
    assert hits[0].text == "김지민"
