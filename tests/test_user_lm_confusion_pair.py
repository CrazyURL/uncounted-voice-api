"""Unit tests — app/user_lm/confusion_pair_engine.py"""
from __future__ import annotations

from app.user_lm.confusion_pair_engine import (
    BASE_MARGIN_PER_CONFIRM,
    CONFIRM_COUNT_MIN,
    CONTEXT_BONUS,
    REJECT_PENALTY_PER_REJECT,
    WHISPER_WORD_SCORE_MIN,
    score_confusion_pair,
)
from app.user_lm.types import ConfusionPair


def _pair(
    *,
    user_id: str = "u1",
    confirm: int = 2,
    reject: int = 0,
    contexts: tuple[str, ...] = (),
    weight: float = 2.0,
) -> ConfusionPair:
    return ConfusionPair(
        from_word="본격",
        to_word="원격",
        user_id=user_id,
        confirm_count=confirm,
        reject_count=reject,
        contexts=contexts,
        weight=weight,
    )


class TestCrossUserContamination:
    def test_pair_from_other_user_blocked(self):
        p = _pair(user_id="u_owner")
        s = score_confusion_pair(pair=p, user_id="u_caller")
        assert s.applied is False
        assert "cross_user_blocked" in s.reason
        assert s.confidence_delta == 0.0
        assert s.needs_review is False

    def test_same_user_passes_basic_guards(self):
        p = _pair(user_id="u1", confirm=2)
        s = score_confusion_pair(pair=p, user_id="u1")
        assert s.applied is True
        assert s.confidence_delta > 0
        assert "applied" in s.reason


class TestConfirmCountGuard:
    def test_confirm_count_0_blocked(self):
        s = score_confusion_pair(pair=_pair(confirm=0), user_id="u1")
        assert s.applied is False
        assert "insufficient_confirm" in s.reason

    def test_confirm_count_1_blocked(self):
        s = score_confusion_pair(pair=_pair(confirm=1), user_id="u1")
        assert s.applied is False
        assert s.needs_review is False

    def test_confirm_count_min_passes(self):
        s = score_confusion_pair(pair=_pair(confirm=CONFIRM_COUNT_MIN), user_id="u1")
        assert s.applied is True

    def test_high_confirm_count_higher_margin(self):
        low = score_confusion_pair(pair=_pair(confirm=2), user_id="u1")
        high = score_confusion_pair(pair=_pair(confirm=10), user_id="u1")
        assert high.confidence_delta > low.confidence_delta


class TestRejectSuppress:
    def test_reject_equal_confirm_suppressed(self):
        s = score_confusion_pair(pair=_pair(confirm=3, reject=3), user_id="u1")
        assert s.applied is False
        assert "rejected_suppress" in s.reason

    def test_reject_greater_than_confirm_suppressed(self):
        s = score_confusion_pair(pair=_pair(confirm=2, reject=5), user_id="u1")
        assert s.applied is False
        assert "rejected_suppress" in s.reason

    def test_reject_less_than_confirm_lowers_margin(self):
        no_reject = score_confusion_pair(pair=_pair(confirm=5, reject=0), user_id="u1")
        with_reject = score_confusion_pair(pair=_pair(confirm=5, reject=2), user_id="u1")
        assert with_reject.confidence_delta < no_reject.confidence_delta
        assert with_reject.applied is True


class TestWhisperWordScoreGuard:
    def test_low_whisper_score_needs_review(self):
        s = score_confusion_pair(
            pair=_pair(confirm=3),
            user_id="u1",
            whisper_word_score=0.3,
        )
        assert s.needs_review is True
        assert s.applied is False
        assert "low_whisper_score" in s.reason

    def test_boundary_whisper_score_min_passes(self):
        s = score_confusion_pair(
            pair=_pair(confirm=3),
            user_id="u1",
            whisper_word_score=WHISPER_WORD_SCORE_MIN,
        )
        assert s.needs_review is False
        assert s.applied is True


class TestContextBonus:
    def test_context_match_adds_bonus(self):
        plain = score_confusion_pair(pair=_pair(confirm=2), user_id="u1")
        with_ctx = score_confusion_pair(
            pair=_pair(confirm=2, contexts=("종료", "접속")),
            user_id="u1",
            context_after="종료하면",
        )
        assert with_ctx.confidence_delta == plain.confidence_delta + CONTEXT_BONUS
        assert "context_match" in with_ctx.reason

    def test_context_miss_no_bonus(self):
        plain = score_confusion_pair(pair=_pair(confirm=2), user_id="u1")
        with_ctx = score_confusion_pair(
            pair=_pair(confirm=2, contexts=("종료",)),
            user_id="u1",
            context_after="회의실",
        )
        assert with_ctx.confidence_delta == plain.confidence_delta
        assert "context_match" not in with_ctx.reason

    def test_context_before_also_matched(self):
        s = score_confusion_pair(
            pair=_pair(confirm=2, contexts=("PC",)),
            user_id="u1",
            context_before="고객 PC에",
        )
        assert "context_match" in s.reason


class TestNoRawPiiLeak:
    def test_reason_string_no_raw_pii(self):
        """confusion pair engine 자체는 PII detection 안 함, reason 에 from_word 가 들어가지 않는지만 확인."""
        s = score_confusion_pair(
            pair=ConfusionPair(
                from_word="01012345678",  # 숫자 — 호출자가 protected_keyword 로 차단해야 함
                to_word="masked",
                user_id="u1",
                confirm_count=3,
                weight=3.0,
            ),
            user_id="u1",
        )
        # engine 자체는 from_word/to_word 를 reason 에 노출 안 함
        assert "01012345678" not in s.reason
        assert "masked" not in s.reason
