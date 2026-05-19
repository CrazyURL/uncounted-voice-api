"""049 v5: 통화 등급별 PII 마스킹 분기 테스트.

약관 v1.2 제5조의2 3항 — STANDARD 등급은 직무 발화자 측 실명·소속·직책 등
식별 가능 단어의 강화 마스킹을 의무로 한다.

본 테스트는 pii_masker.py의 grade 인자 분기를 직접 검증한다.
"""
import pytest

from app.pii_masker import detect_pii_spans, mask_pii, mask_segments


class TestGradeArgumentBackwardCompat:
    """grade 인자 default=None — 기존 호출은 영향 없어야 한다."""

    def test_mask_pii_no_grade_no_name_masking(self):
        text = "김철수씨가 010-1234-5678로 전화"
        result = mask_pii(text)
        # 기본 enable_name_masking=False → 이름 마스킹 X
        assert "김철수" in result["masked_text"]
        assert "010-****-5678" in result["masked_text"]

    def test_detect_spans_no_grade_no_name_masking(self):
        text = "홍길동님 전화 010-1234-5678"
        spans = detect_pii_spans(text)
        types = {s["type"] for s in spans}
        assert "전화번호" in types
        assert "이름" not in types

    def test_mask_pii_premium_explicit(self):
        """grade='premium' + enable_name_masking=False → 이름 보존 (양측 개인 통화)."""
        text = "김철수씨가 왔습니다"
        result = mask_pii(text, enable_name_masking=False, grade="premium")
        assert "김철수" in result["masked_text"]


class TestStandardGradeForcesNameMasking:
    """STANDARD 등급은 enable_name_masking을 자동 True로 강제."""

    def test_mask_pii_standard_forces_name_mask(self):
        text = "김철수씨가 왔습니다"
        # 호출자가 enable_name_masking=False를 박아도 standard는 강제 True
        result = mask_pii(text, enable_name_masking=False, grade="standard")
        assert "김OO" in result["masked_text"]

    def test_detect_spans_standard_emits_name(self):
        text = "변호사 김철수입니다"
        spans = detect_pii_spans(text, enable_name_masking=False, grade="standard")
        types = {s["type"] for s in spans}
        assert "이름" in types

    def test_mask_segments_standard_forces_name_mask(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "안녕하세요 김철수입니다"},
            {"start": 1.0, "end": 2.0, "text": "010-1234-5678로 연락"},
        ]
        summary = mask_segments(segments, enable_name_masking=False, grade="standard")
        # 첫 segment 이름 마스킹 적용
        assert "김OO" in segments[0]["text"]
        # 둘째 segment 전화 마스킹
        assert "010-****-5678" in segments[1]["text"]
        # 요약에 이름 + 전화번호 모두 박힘
        types = {item["type"] for item in summary}
        assert "이름" in types
        assert "전화번호" in types


class TestExcludedGradeAlsoForcesMasking:
    """EXCLUDED 등급은 거래 불가지만 호출 시 STANDARD와 동일 처리."""

    def test_mask_pii_excluded_forces_name_mask(self):
        text = "이영희가 와서"
        result = mask_pii(text, enable_name_masking=False, grade="excluded")
        # excluded도 강화 마스킹 적용
        assert "이OO" not in result["masked_text"] or "이영" in result["masked_text"] or "이영희" not in result["masked_text"]
        # 정확히는 enable_name_masking이 True가 되어야 한다
        result2 = mask_pii(text, enable_name_masking=True)
        assert result["masked_text"] == result2["masked_text"]


class TestGradeOverridesEnableNameMaskingOnlyForRestricted:
    """premium은 enable_name_masking 호출자 값을 따른다 (강제 X)."""

    def test_premium_respects_enable_name_masking_true(self):
        text = "김철수씨"
        result = mask_pii(text, enable_name_masking=True, grade="premium")
        assert "김OO" in result["masked_text"]

    def test_premium_respects_enable_name_masking_false(self):
        text = "김철수씨"
        result = mask_pii(text, enable_name_masking=False, grade="premium")
        # premium은 호출자 값(False) 그대로 → 이름 보존
        assert "김철수" in result["masked_text"]
