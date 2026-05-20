"""음성 전사형(한글 숫자어) 전화번호 탐지 — PII-1A Category 1.

배경: 한국어 통화 STT 는 전화번호를 "공일공 일이삼사 오육칠팔" 처럼 한글 숫자어로
전사한다. 기존 detect_pii_spans 의 \\d 정규식은 이를 한 건도 못 잡는다.

이 파일은 include_spoken_pii=True 경로만 검증한다. 기본 경로(mask_pii / 오디오 마스킹)는
변경하지 않음(freeze)을 함께 고정한다.
"""

import pytest

from app.pii_masker import detect_pii_spans, mask_pii
from app.pii_confidence import score_candidates


def _phone_spans(text):
    spans = detect_pii_spans(text, include_spoken_pii=True)
    return [s for s in spans if s["type"] == "전화번호"]


# ── positive ────────────────────────────────────────────────────────────
def test_sino_mobile_basic():
    text = "제 번호 공일공 일이삼사 오육칠팔 입니다"
    spans = _phone_spans(text)
    assert len(spans) == 1
    # offset 이 원문 run 을 정확히 가리킨다
    s = spans[0]
    assert text[s["char_start"]:s["char_end"]] == "공일공 일이삼사 오육칠팔"


def test_sino_mobile_no_spaces():
    text = "공일공일이삼사오육칠팔"
    spans = _phone_spans(text)
    assert len(spans) == 1
    assert spans[0]["char_start"] == 0


def test_native_digits_inside_run():
    # 공일공 하나둘삼사 오육칠팔 → 0 1 0 1 2 3 4 5 6 7 8
    text = "번호는 공일공 하나둘삼사 오육칠팔 이에요"
    spans = _phone_spans(text)
    assert len(spans) == 1


def test_eleven_digit_010_run():
    text = "공일공 구팔칠육 오사삼이"  # 010 9876 5432
    spans = _phone_spans(text)
    assert len(spans) == 1


# ── tier / 안전 ───────────────────────────────────────────────────────────
def test_spoken_phone_scores_auto_confirmed():
    spans = detect_pii_spans("공일공 일이삼사 오육칠팔", include_spoken_pii=True)
    scored = score_candidates(spans)
    phone = [c for c in scored if c["type"] == "전화번호"]
    assert len(phone) == 1
    assert phone[0]["high_precision_pattern"] is True
    assert phone[0]["confidence_tier"] == "auto_confirmed"


def test_score_output_has_no_raw_text():
    spans = detect_pii_spans("공일공 일이삼사 오육칠팔", include_spoken_pii=True)
    scored = score_candidates(spans)
    for c in scored:
        assert "matched_text" not in c
        assert set(c.keys()) == {
            "type", "char_start", "char_end",
            "confidence", "high_precision_pattern", "confidence_tier",
        }


# ── negative (오탐 방지) ──────────────────────────────────────────────────
def test_counting_native_not_phone():
    # 세는 말 — 0 으로 시작하지 않음 → 전화번호 아님
    text = "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 개"
    assert _phone_spans(text) == []


def test_scattered_common_syllables_not_phone():
    text = "이거 사 가지고 삼일 동안 일 하면서 사 먹었어"
    assert _phone_spans(text) == []


def test_too_short_run_not_phone():
    text = "공일공"  # 3자리만
    assert _phone_spans(text) == []


def test_run_not_starting_with_zero_rejected():
    # 11 단어 run 이지만 01 로 시작하지 않음
    text = "일이삼사오육칠팔구공일"
    assert _phone_spans(text) == []


# ── freeze: 기본 경로 불변 ────────────────────────────────────────────────
def test_default_path_ignores_spoken_phone():
    # include_spoken_pii 기본 False → 음성 전화번호 미탐지(마스킹 경로 보호)
    spans = detect_pii_spans("공일공 일이삼사 오육칠팔")
    assert [s for s in spans if s["type"] == "전화번호"] == []


def test_mask_pii_unchanged_by_spoken_phone():
    # mask_pii 는 include_spoken_pii 를 켜지 않으므로 음성 전화번호를 마스킹하지 않는다
    text = "공일공 일이삼사 오육칠팔 라고 했어요"
    result = mask_pii(text)
    assert result["total_masked"] == 0
    assert result["masked_text"] == text


def test_arabic_phone_still_detected_in_spoken_mode():
    # 음성 모드에서도 기존 아라비아 전화번호는 그대로 잡힌다 (회귀)
    text = "전화 010-1234-5678 그리고 공일공 일이삼사 오육칠팔"
    spans = _phone_spans(text)
    assert len(spans) == 2
