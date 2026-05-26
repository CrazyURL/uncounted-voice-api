"""이름 graded confidence (name_context) 테스트.

detect_pii_spans 가 이름 span 에 name_context(honorific/mid/weak_trailing) 를 부착하고,
pii_confidence 가 이를 confidence/tier 로 매핑한다.

핵심 안전 속성:
  - 마스킹(emit)은 문맥과 무관하게 유지된다 (프라이버시 무손실).
  - weak_trailing(조사/어미 후행)만 auto_rejected 로 검수 큐에서 내려간다.
  - name_context 는 score_candidates 출력에 누출되지 않는다.
  - name_context 가 없는 span 은 기존 동작(0.70/needs_human_decision) 을 유지한다.
"""

import pytest

from app.pii_confidence import score_candidates
from app.pii_masker import detect_pii_spans, mask_pii


def _name_span(text: str) -> dict:
    return next(
        s for s in detect_pii_spans(text, enable_name_masking=True) if s["type"] == "이름"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected_ctx,expected_conf,expected_tier",
    [
        ("김민수 씨가 왔어요", "honorific", 0.85, "needs_human_decision"),
        ("김민수 왔어요", "mid", 0.70, "needs_human_decision"),
        ("김민수가 왔어요", "weak_trailing", 0.40, "auto_rejected"),
    ],
)
def test_name_context_grading(text, expected_ctx, expected_conf, expected_tier):
    span = _name_span(text)
    assert span["name_context"] == expected_ctx
    scored = score_candidates([span])[0]
    assert scored["confidence"] == expected_conf
    assert scored["confidence_tier"] == expected_tier


@pytest.mark.unit
def test_weak_trailing_name_is_still_masked():
    """weak_trailing(auto_rejected)이어도 마스킹은 유지 — 프라이버시 안전망 불변."""
    result = mask_pii("김민수가 왔어요", enable_name_masking=True)
    assert "김민수" not in result["masked_text"]


@pytest.mark.unit
def test_name_context_not_leaked_in_output():
    scored = score_candidates([_name_span("김민수가 왔어요")])[0]
    assert "name_context" not in scored
    assert "matched_text" not in scored


@pytest.mark.unit
def test_name_without_context_keeps_legacy_behavior():
    """name_context 없는 이름 span(수동 구성) 은 기존 0.70/needs_human 유지 (하위호환)."""
    scored = score_candidates(
        [{"type": "이름", "char_start": 0, "char_end": 3, "matched_text": "홍길동"}]
    )[0]
    assert scored["confidence"] == 0.70
    assert scored["confidence_tier"] == "needs_human_decision"
