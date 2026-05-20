"""PII confidence tier 합성 단위 테스트 (PII-1A).

pii_confidence 는 detect_pii_spans 결과(type/offset)를 받아
confidence / high_precision_pattern / confidence_tier 를 합성한다.
ML 확률이 아니라 정규식 pattern class 기반 부트스트랩 휴리스틱이다.

핵심 안전 계약: 출력에 matched_text(원문 span) 가 절대 포함되지 않는다.
"""

import pytest

from app.pii_confidence import classify_tier, score_candidates


# ── classify_tier 경계값 ────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize(
    "confidence,high_precision,ambiguous,expected",
    [
        (0.90, True, False, "auto_confirmed"),     # ≥0.90 & high_precision
        (0.95, True, False, "auto_confirmed"),
        (0.90, False, False, "needs_human_decision"),  # 0.90 이지만 high_precision 아님
        (0.89, True, False, "needs_human_decision"),   # high_precision 이지만 <0.90
        (0.50, False, False, "needs_human_decision"),  # 하한 경계
        (0.49, False, False, "auto_rejected"),         # <0.50 & !high_precision
        (0.49, True, False, "needs_human_decision"),   # <0.50 이지만 high_precision → 큐 유지
        (0.10, False, True, "needs_human_decision"),   # 약신호여도 ambiguous → 사람 판단
    ],
)
def test_classify_tier_boundaries(confidence, high_precision, ambiguous, expected):
    assert classify_tier(confidence, high_precision, ambiguous) == expected


# ── score_candidates: type → tier 매핑 ──────────────────────────────
@pytest.mark.unit
def test_phone_is_auto_confirmed():
    spans = [{"type": "전화번호", "char_start": 5, "char_end": 18, "matched_text": "010-1234-5678"}]
    out = score_candidates(spans)
    assert len(out) == 1
    assert out[0]["confidence_tier"] == "auto_confirmed"
    assert out[0]["high_precision_pattern"] is True
    assert out[0]["confidence"] >= 0.90


@pytest.mark.unit
def test_name_is_needs_human_decision():
    spans = [{"type": "이름", "char_start": 0, "char_end": 3, "matched_text": "홍길동"}]
    out = score_candidates(spans)
    assert out[0]["confidence_tier"] == "needs_human_decision"
    assert out[0]["high_precision_pattern"] is False


@pytest.mark.unit
@pytest.mark.parametrize("pii_type", ["주민등록번호", "이메일", "카드번호", "계좌번호", "여권번호", "IP주소"])
def test_high_precision_types_auto_confirmed(pii_type):
    spans = [{"type": pii_type, "char_start": 0, "char_end": 4, "matched_text": "xxxx"}]
    out = score_candidates(spans)
    assert out[0]["confidence_tier"] == "auto_confirmed"
    assert out[0]["high_precision_pattern"] is True


# ── 안전 계약: 원문 미포함 ───────────────────────────────────────────
@pytest.mark.unit
def test_score_candidates_strips_matched_text():
    spans = [
        {"type": "전화번호", "char_start": 5, "char_end": 18, "matched_text": "010-1234-5678"},
        {"type": "이름", "char_start": 0, "char_end": 3, "matched_text": "홍길동"},
    ]
    out = score_candidates(spans)
    for c in out:
        assert "matched_text" not in c
        # 키는 offset/type/confidence/tier/high_precision 만
        assert set(c.keys()) == {
            "type",
            "char_start",
            "char_end",
            "confidence",
            "high_precision_pattern",
            "confidence_tier",
        }


@pytest.mark.unit
def test_score_candidates_preserves_offsets():
    spans = [{"type": "이메일", "char_start": 4, "char_end": 20, "matched_text": "a@example.com"}]
    out = score_candidates(spans)
    assert out[0]["char_start"] == 4
    assert out[0]["char_end"] == 20


@pytest.mark.unit
def test_empty_input():
    assert score_candidates([]) == []
