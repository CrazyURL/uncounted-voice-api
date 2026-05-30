"""PR-B: 한국어 PII 사각지대 보강 (app/pii_extended.py + pii_masker.detect_pii_spans 통합).

5 카테고리 × (정상 hit / false positive 방지 / 한국어 컨텍스트) + flag + 9fa79d3c 회귀.

전체 30+ cases. 기존 PII 테스트(`test_pii_spans.py`, `test_pii_grade.py`)는 별도 회귀.
"""

from __future__ import annotations

import pytest

from app.pii_extended import (
    LABEL_CREDENTIAL_LIKE,
    LABEL_FOREIGN_ID_LIKE,
    LABEL_KOREAN_NAME_CANDIDATE,
    LABEL_NUMERIC_SENSITIVE_LIKE,
    LABEL_PAYMENT_LIKE,
    category_labels,
    detect_extended_spans,
    is_candidate_type,
    resolve_pii_detector_extended_enabled,
    suggested_confidence,
)


def types(spans: list[dict]) -> list[str]:
    return [s["type"] for s in spans]


def count(spans: list[dict], label: str) -> int:
    return sum(1 for s in spans if s["type"] == label)


# ─────────────────────────────────────────────────────────────────────────
# Feature flag
# ─────────────────────────────────────────────────────────────────────────

class TestFeatureFlag:
    def test_default_true(self):
        assert resolve_pii_detector_extended_enabled({}) is True

    @pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
    def test_truthy(self, raw):
        assert resolve_pii_detector_extended_enabled({"WORKER_PII_DETECTOR_EXTENDED": raw}) is True

    @pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
    def test_falsy(self, raw):
        assert resolve_pii_detector_extended_enabled({"WORKER_PII_DETECTOR_EXTENDED": raw}) is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            resolve_pii_detector_extended_enabled({"WORKER_PII_DETECTOR_EXTENDED": "maybe"})


# ─────────────────────────────────────────────────────────────────────────
# 빈 / 형식 가드
# ─────────────────────────────────────────────────────────────────────────

class TestEmptyAndShape:
    def test_empty_text(self):
        assert detect_extended_spans("") == []

    def test_none_safely(self):
        assert detect_extended_spans(None) == []  # type: ignore[arg-type]

    def test_normal_korean_no_hit(self):
        assert detect_extended_spans("안녕하세요 오늘 날씨가 좋네요") == []

    def test_span_dict_shape(self):
        # credential_like 은 enable_name_masking 무관 — 항상 활성
        spans = detect_extended_spans("비밀번호 Abc123")
        assert len(spans) >= 1
        s = spans[0]
        assert set(s.keys()) == {"type", "char_start", "char_end", "matched_text"}
        assert isinstance(s["char_start"], int)
        assert isinstance(s["char_end"], int)
        assert s["char_start"] < s["char_end"]


# ─────────────────────────────────────────────────────────────────────────
# credential_like
# ─────────────────────────────────────────────────────────────────────────

class TestCredentialLike:
    def test_korean_password_keyword(self):
        spans = detect_extended_spans("비밀번호는 Abc1234 입니다")
        assert count(spans, LABEL_CREDENTIAL_LIKE) >= 1

    def test_english_password_keyword(self):
        spans = detect_extended_spans("the password is Hello9 right")
        assert count(spans, LABEL_CREDENTIAL_LIKE) >= 1

    def test_keyword_absent_no_hit(self):
        # 비밀번호 키워드 없이 영숫자만 → false positive 방지
        spans = detect_extended_spans("자동차 모델 ABC123 입니다")
        assert count(spans, LABEL_CREDENTIAL_LIKE) == 0

    def test_keyword_present_korean_digits_only_no_hit(self):
        # 영문 없이 한글+숫자만 → credential 룰 미적용 (정규식 영숫자 혼합 필수)
        spans = detect_extended_spans("비밀번호 12345 입력")
        assert count(spans, LABEL_CREDENTIAL_LIKE) == 0


# ─────────────────────────────────────────────────────────────────────────
# foreign_id_like
# ─────────────────────────────────────────────────────────────────────────

class TestForeignIdLike:
    def test_foreign_id_5_prefix(self):
        # 외국인등록증: 두번째 자리 5 시작
        spans = detect_extended_spans("외국인등록증 850515-5876543")
        assert count(spans, LABEL_FOREIGN_ID_LIKE) >= 1

    def test_foreign_id_6_prefix(self):
        spans = detect_extended_spans("등록증 900101-6234567")
        assert count(spans, LABEL_FOREIGN_ID_LIKE) >= 1

    def test_no_hit_for_korean_resident_1_prefix(self):
        # 한국인 주민번호(1xxxxxx)는 기존 PII_PATTERNS 가 처리, 본 룰은 5-8 만
        spans = detect_extended_spans("주민번호 900101-1234567")
        assert count(spans, LABEL_FOREIGN_ID_LIKE) == 0

    def test_no_separator_no_hit(self):
        # 구분자 없는 13자리 연속 숫자 → no foreign_id (false positive 방지)
        spans = detect_extended_spans("9001015234567")
        assert count(spans, LABEL_FOREIGN_ID_LIKE) == 0


# ─────────────────────────────────────────────────────────────────────────
# payment_like
# ─────────────────────────────────────────────────────────────────────────

class TestPaymentLike:
    def test_payment_context_with_digits(self):
        spans = detect_extended_spans("이체 1234567890 으로 했어요")
        assert count(spans, LABEL_PAYMENT_LIKE) >= 1

    def test_song_geum_keyword(self):
        spans = detect_extended_spans("송금 234567 완료")
        assert count(spans, LABEL_PAYMENT_LIKE) >= 1

    def test_cvc_keyword(self):
        spans = detect_extended_spans("CVC 123456")
        assert count(spans, LABEL_PAYMENT_LIKE) >= 1

    def test_no_keyword_no_hit(self):
        # 결제 키워드 없으면 payment_like 미발생
        spans = detect_extended_spans("그냥 234567 숫자")
        assert count(spans, LABEL_PAYMENT_LIKE) == 0


# ─────────────────────────────────────────────────────────────────────────
# numeric_sensitive_like
# ─────────────────────────────────────────────────────────────────────────

class TestNumericSensitiveLike:
    def test_six_digit_hit(self):
        spans = detect_extended_spans("번호 123456")
        assert count(spans, LABEL_NUMERIC_SENSITIVE_LIKE) >= 1

    def test_ten_digit_phone(self):
        spans = detect_extended_spans("0212345678")
        assert count(spans, LABEL_NUMERIC_SENSITIVE_LIKE) >= 1

    def test_five_digit_no_hit(self):
        # 5자리 미만 → no hit
        spans = detect_extended_spans("숫자 12345")
        assert count(spans, LABEL_NUMERIC_SENSITIVE_LIKE) == 0

    def test_decimal_no_hit(self):
        # 소수점 부분 제외
        spans = detect_extended_spans("값은 3.141592 입니다")
        assert count(spans, LABEL_NUMERIC_SENSITIVE_LIKE) == 0


# ─────────────────────────────────────────────────────────────────────────
# korean_name_like_candidate
# ─────────────────────────────────────────────────────────────────────────

class TestKoreanNameCandidate:
    def test_kim_chulsoo_sajang(self):
        spans = detect_extended_spans("김철수 사장님이 오셨어요", enable_name_masking=True)
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) >= 1

    def test_lee_younghee_daeri(self):
        spans = detect_extended_spans("이영희 대리에게 전달했어요", enable_name_masking=True)
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) >= 1

    def test_park_jieun_ssi(self):
        spans = detect_extended_spans("박지은씨한테 말했어요", enable_name_masking=True)
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) >= 1

    def test_no_surname_no_hit(self):
        # 성씨 없는 호칭만 → no hit
        spans = detect_extended_spans("우리 사장님이 말씀하셨다", enable_name_masking=True)
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) == 0

    def test_english_name_no_hit(self):
        spans = detect_extended_spans("John 대리님이 출장", enable_name_masking=True)
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) == 0

    def test_two_names_two_hits(self):
        spans = detect_extended_spans(
            "김철수 사장님과 박영희 대리님이 회의", enable_name_masking=True,
        )
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) == 2

    def test_premium_grade_default_no_name_candidate(self):
        # enable_name_masking=False (premium 기본) → korean_name 후보 미발생.
        # 기존 _SURNAME_PATTERN 의 premium 정책 동형.
        spans = detect_extended_spans("김철수 사장님이 오셨어요")
        assert count(spans, LABEL_KOREAN_NAME_CANDIDATE) == 0


# ─────────────────────────────────────────────────────────────────────────
# 분류 helper
# ─────────────────────────────────────────────────────────────────────────

class TestClassification:
    def test_is_candidate_credential_and_name(self):
        assert is_candidate_type(LABEL_CREDENTIAL_LIKE)
        assert is_candidate_type(LABEL_KOREAN_NAME_CANDIDATE)

    def test_is_not_candidate_id_payment_numeric(self):
        # 핵심 확정 PII (외국인등록증 / 결제 / numeric_sensitive) 는 candidate 마킹 X
        # — confirmed PII 측면에서는 high/medium confidence 로 처리 가능
        assert not is_candidate_type(LABEL_FOREIGN_ID_LIKE)
        assert not is_candidate_type(LABEL_PAYMENT_LIKE)
        assert not is_candidate_type(LABEL_NUMERIC_SENSITIVE_LIKE)

    def test_suggested_confidence(self):
        assert suggested_confidence(LABEL_CREDENTIAL_LIKE) == "high"
        assert suggested_confidence(LABEL_FOREIGN_ID_LIKE) == "high"
        assert suggested_confidence(LABEL_PAYMENT_LIKE) == "high"
        assert suggested_confidence(LABEL_NUMERIC_SENSITIVE_LIKE) == "medium"
        assert suggested_confidence(LABEL_KOREAN_NAME_CANDIDATE) == "medium"
        assert suggested_confidence("unknown") == "low"

    def test_category_labels_all_five(self):
        assert set(category_labels()) == {
            LABEL_CREDENTIAL_LIKE,
            LABEL_FOREIGN_ID_LIKE,
            LABEL_PAYMENT_LIKE,
            LABEL_NUMERIC_SENSITIVE_LIKE,
            LABEL_KOREAN_NAME_CANDIDATE,
        }


# ─────────────────────────────────────────────────────────────────────────
# 9fa79d3c 회귀 fixture (재현)
# ─────────────────────────────────────────────────────────────────────────

class TestSession9fa79d3cRegression:
    """세션 9fa79d3cbf1fb2c9 export QA 결과 동등 패턴 (실 transcript 미포함)."""

    def test_multi_category_hits(self):
        text = (
            "오늘 시스템 로그인 정보를 알려드릴게요\n"
            "비밀번호는 Abc12345 입니다\n"
            "외국인등록증 800101-5234567\n"
            "이체 1234567890 완료\n"
            "김철수 사장님이 결재하셨어요"
        )
        # 9fa79d3c 동등 = 운영자 검수용 fixture, name_masking 활성으로 5 카테고리 검증.
        spans = detect_extended_spans(text, enable_name_masking=True)
        ts = types(spans)
        assert LABEL_CREDENTIAL_LIKE in ts
        assert LABEL_FOREIGN_ID_LIKE in ts
        assert LABEL_PAYMENT_LIKE in ts
        assert LABEL_KOREAN_NAME_CANDIDATE in ts
        # numeric_sensitive_like 는 위 문장에서 6+자리 숫자 (Abc12345 제외) 가 있어 발생
        assert LABEL_NUMERIC_SENSITIVE_LIKE in ts


# ─────────────────────────────────────────────────────────────────────────
# detect_pii_spans 통합 (PII_PATTERNS + extended)
# ─────────────────────────────────────────────────────────────────────────

class TestIntegrationWithDetectPiiSpans:
    """기존 detect_pii_spans 가 PR-B 신규 룰을 자동 호출하는지 검증.

    feature flag default true 라 별도 monkeypatch 없이 동작.
    """

    def test_extended_called_by_detect_pii_spans(self):
        from app.pii_masker import detect_pii_spans

        spans = detect_pii_spans("비밀번호 Abc1234 외국인등록증 850515-5876543")
        ts = [s["type"] for s in spans]
        assert LABEL_CREDENTIAL_LIKE in ts
        assert LABEL_FOREIGN_ID_LIKE in ts

    def test_extended_disabled_via_env(self, monkeypatch):
        from app.pii_masker import detect_pii_spans

        monkeypatch.setenv("WORKER_PII_DETECTOR_EXTENDED", "false")
        spans = detect_pii_spans("비밀번호 Abc1234")
        ts = [s["type"] for s in spans]
        assert LABEL_CREDENTIAL_LIKE not in ts

    def test_existing_patterns_still_active(self):
        # 기존 주민등록번호([1-4]) 는 PII_PATTERNS 가 처리, 본 PR-B 영향 0
        from app.pii_masker import detect_pii_spans

        spans = detect_pii_spans("주민번호 900101-1234567")
        ts = [s["type"] for s in spans]
        assert "주민등록번호" in ts
