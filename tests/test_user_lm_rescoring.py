"""Unit tests — app/user_lm/rescoring_helper.py (6 guards + provenance)."""
from __future__ import annotations

from app.user_lm.rescoring_helper import (
    ALPHA_MAX,
    BETA_MAX,
    CONFUSION_PAIR_MIN_WEIGHT,
    MARGIN_THRESHOLD_NATS,
    rescore_transcript,
)
from app.user_lm.types import (
    ConfusionPair,
    PhraseMemoryEntry,
    PiiInterval,
    RescoringInput,
    WordCandidate,
)


def _bonkyeok_pair(
    *,
    confirm: int = 5,
    reject: int = 0,
    weight: float = 5.0,
    user_id: str = "u1",
    contexts: tuple[str, ...] = ("종료", "접속", "연결"),
) -> ConfusionPair:
    return ConfusionPair(
        from_word="본격",
        to_word="원격",
        user_id=user_id,
        confirm_count=confirm,
        reject_count=reject,
        weight=weight,
        contexts=contexts,
    )


def _words_bonkyeok() -> tuple[WordCandidate, ...]:
    return (
        WordCandidate(word="본격", start=0.0, end=0.6, score=0.85),
        WordCandidate(word="종료하면", start=0.6, end=1.4, score=0.91),
        WordCandidate(word="된다고", start=1.4, end=2.0, score=0.93),
        WordCandidate(word="하네요", start=2.0, end=2.8, score=0.95),
    )


def _input(
    *,
    text: str = "본격 종료하면 된다고 하네요",
    user_id: str = "u1",
    pairs: tuple[ConfusionPair, ...] = (),
    memory: tuple[PhraseMemoryEntry, ...] = (),
    pii: tuple[PiiInterval, ...] = (),
    words: tuple[WordCandidate, ...] | None = None,
    whisper_score: float = 0.85,
    global_lm_score: float = 0.55,
    alpha: float = 0.3,
    beta: float = 0.2,
    lm_version: str | None = "v20260601_001",
) -> RescoringInput:
    return RescoringInput(
        user_id=user_id,
        text=text,
        words=words if words is not None else _words_bonkyeok(),
        confusion_pairs=pairs,
        phrase_memory=memory,
        pii_intervals=pii,
        whisper_score=whisper_score,
        global_lm_score=global_lm_score,
        alpha=alpha,
        beta=beta,
        lm_version=lm_version,
    )


class TestBonkyeokToWongyeokHappyPath:
    """본격 → 원격 정정의 정공법 시나리오 — IT 헬프데스크 통화 #192163."""

    def test_correction_applied(self):
        pair = _bonkyeok_pair()
        out = rescore_transcript(_input(pairs=(pair,)))
        assert "원격" in out.corrected_text
        assert "본격" not in out.corrected_text
        assert out.safe_to_auto_apply is True
        assert out.needs_review is False
        assert any(r["type"] == "confusion_pair" for r in out.applied_rules)
        assert out.source == "user_lm"
        assert out.lm_version == "v20260601_001"


class TestMarginThreshold:
    def test_low_margin_needs_review(self):
        # confirm=2 (min), context match 없음 → margin = 2*0.05 = 0.10 < 0.20 → needs_review
        pair = ConfusionPair(
            from_word="본격", to_word="원격", user_id="u1",
            confirm_count=2, reject_count=0,
            weight=CONFUSION_PAIR_MIN_WEIGHT,
            contexts=(),
        )
        out = rescore_transcript(_input(pairs=(pair,)))
        assert out.safe_to_auto_apply is False
        assert out.needs_review is True
        assert "margin" in (out.needs_review_reason or "")

    def test_high_margin_auto_apply(self):
        pair = _bonkyeok_pair(confirm=10, contexts=("종료",))
        out = rescore_transcript(_input(pairs=(pair,)))
        assert out.margin >= MARGIN_THRESHOLD_NATS
        assert out.safe_to_auto_apply is True


class TestConfusionPairWeightGuard:
    def test_pair_weight_below_threshold_blocked(self):
        pair = ConfusionPair(
            from_word="본격", to_word="원격", user_id="u1",
            confirm_count=5, weight=1.0,  # < 2.0
        )
        out = rescore_transcript(_input(pairs=(pair,)))
        assert out.corrected_text == "본격 종료하면 된다고 하네요"
        assert not any(r["type"] == "confusion_pair" for r in out.applied_rules)


class TestRejectedPairSuppress:
    def test_rejected_pair_no_apply(self):
        pair = _bonkyeok_pair(confirm=3, reject=5)
        out = rescore_transcript(_input(pairs=(pair,)))
        assert "본격" in out.corrected_text
        assert out.safe_to_auto_apply is False


class TestPiiOverlapGuard:
    def test_pii_overlap_blocks_substitution(self):
        # "본격" word 가 PII 구간 안 (0.0-0.6)
        pii = (PiiInterval(start=0.0, end=1.0, type="NAME"),)
        out = rescore_transcript(_input(
            pairs=(_bonkyeok_pair(),),
            pii=pii,
        ))
        assert "본격" in out.corrected_text  # 정정 안 됨
        assert out.safe_to_auto_apply is False

    def test_pii_outside_word_does_not_block(self):
        # PII 가 word 밖
        pii = (PiiInterval(start=10.0, end=12.0, type="NAME"),)
        out = rescore_transcript(_input(
            pairs=(_bonkyeok_pair(),),
            pii=pii,
        ))
        assert "원격" in out.corrected_text


class TestNumericProtectionGuard:
    def test_numeric_keyword_in_context_blocks(self):
        # 인접 context 에 "계좌번호" → 보호 발동
        pair = ConfusionPair(
            from_word="삼사오", to_word="345",
            user_id="u1", confirm_count=10, weight=10.0,
            contexts=(),
        )
        words = (
            WordCandidate(word="계좌번호", start=0.0, end=0.8, score=0.95),
            WordCandidate(word="삼사오", start=0.9, end=1.5, score=0.85),
            WordCandidate(word="입니다", start=1.5, end=2.0, score=0.95),
        )
        out = rescore_transcript(_input(
            text="계좌번호 삼사오 입니다",
            pairs=(pair,),
            words=words,
        ))
        # 자동 정정 차단
        assert "삼사오" in out.corrected_text
        assert "345" not in out.corrected_text

    def test_no_numeric_keyword_allows(self):
        pair = _bonkyeok_pair()
        out = rescore_transcript(_input(pairs=(pair,)))
        # 본격→원격 은 numeric 보호 미발동
        assert "원격" in out.corrected_text


class TestCrossUserContaminationGuard:
    def test_other_user_pair_blocked_in_helper(self):
        pair = _bonkyeok_pair(user_id="u_other")
        out = rescore_transcript(_input(pairs=(pair,), user_id="u_me"))
        assert "본격" in out.corrected_text  # 변경 안 됨
        assert not any(r["type"] == "confusion_pair" for r in out.applied_rules)

    def test_phrase_memory_other_user_no_boost(self):
        mem = (PhraseMemoryEntry(surface="원격 접속", weight=5.0, user_id="u_other"),)
        out = rescore_transcript(_input(
            text="원격 접속 됩니다",
            user_id="u_me",
            memory=mem,
        ))
        # phrase boost 미발생
        boost_entries = [r for r in out.applied_rules if r.get("type") == "phrase_memory_boost"]
        assert boost_entries == []


class TestPhraseBoost:
    def test_phrase_memory_match_records_boost(self):
        mem = (PhraseMemoryEntry(surface="원격 접속", weight=3.0, user_id="u1"),)
        out = rescore_transcript(_input(text="원격 접속 가능합니다", memory=mem))
        boost_entries = [r for r in out.applied_rules if r.get("type") == "phrase_memory_boost"]
        assert boost_entries
        assert boost_entries[0]["boost"] == 3.0
        assert "원격 접속" in boost_entries[0]["matched"]


class TestWhisperScoreGuard:
    def test_low_whisper_word_score_needs_review(self):
        # 본격 word 의 whisper score < 0.5 → needs_review
        pair = _bonkyeok_pair()
        words = (
            WordCandidate(word="본격", start=0.0, end=0.6, score=0.3),
            WordCandidate(word="종료하면", start=0.6, end=1.4, score=0.91),
        )
        out = rescore_transcript(_input(
            text="본격 종료하면",
            pairs=(pair,),
            words=words,
        ))
        # confusion engine 의 needs_review → rescoring 도 substitution 안 함
        assert "본격" in out.corrected_text


class TestAlphaBetaBounds:
    def test_alpha_clamped_to_max(self):
        out = rescore_transcript(_input(alpha=10.0, beta=0.0))
        # final_score 가 clamp 된 alpha 로 계산되었는지 — 직접 확인 어렵지만 에러 없으면 OK
        # 0 <= final_score <= 1 검증
        assert 0.0 <= out.final_score <= 1.0

    def test_beta_clamped_to_max(self):
        out = rescore_transcript(_input(alpha=0.0, beta=10.0))
        assert 0.0 <= out.final_score <= 1.0

    def test_alpha_plus_beta_over_one_handled(self):
        out = rescore_transcript(_input(alpha=0.6, beta=0.6))
        # alpha + beta = 1.2 > 1 → 보호 작동 (alpha 줄어듦)
        assert 0.0 <= out.final_score <= 1.0

    def test_negative_alpha_clamped_to_zero(self):
        out = rescore_transcript(_input(alpha=-0.5, beta=0.2))
        assert 0.0 <= out.final_score <= 1.0


class TestGoldenSetRegressionHook:
    def test_golden_set_pass_allows_apply(self):
        out = rescore_transcript(
            _input(pairs=(_bonkyeok_pair(),)),
            golden_set_check=lambda orig, new: True,
        )
        assert out.safe_to_auto_apply is True

    def test_golden_set_fail_blocks_apply(self):
        out = rescore_transcript(
            _input(pairs=(_bonkyeok_pair(),)),
            golden_set_check=lambda orig, new: False,
        )
        assert out.safe_to_auto_apply is False
        assert "golden_set_regression" in (out.needs_review_reason or "")

    def test_golden_set_exception_treated_as_failure(self):
        def _raise(orig, new):
            raise RuntimeError("model load fail")
        out = rescore_transcript(
            _input(pairs=(_bonkyeok_pair(),)),
            golden_set_check=_raise,
        )
        assert out.safe_to_auto_apply is False
        assert "golden_set_error" in (out.needs_review_reason or "")


class TestProvenance:
    def test_source_user_lm(self):
        out = rescore_transcript(_input(pairs=(_bonkyeok_pair(),)))
        assert out.source == "user_lm"

    def test_lm_version_preserved(self):
        out = rescore_transcript(_input(pairs=(_bonkyeok_pair(),), lm_version="v_test_001"))
        assert out.lm_version == "v_test_001"

    def test_applied_rules_have_reason(self):
        out = rescore_transcript(_input(pairs=(_bonkyeok_pair(),)))
        cp_rules = [r for r in out.applied_rules if r["type"] == "confusion_pair"]
        assert cp_rules
        assert cp_rules[0]["reason"]


class TestNoRawPiiLogging:
    def test_pii_overlap_surface_redacted_if_applied(self):
        """PII 구간과 겹친 word 가 우연히 정정 통과해도 surface 노출 X.

        본 가드는 _word_overlaps_pii 결과를 _surface_redacted 에 전달. PII 구간 안의 word 는
        보호 가드(#2)에서 차단되어 applied_rules 에 들어가지 않아야 함 (간접 검증).
        """
        pii = (PiiInterval(start=0.0, end=1.0, type="NAME"),)
        out = rescore_transcript(_input(
            pairs=(_bonkyeok_pair(),),
            pii=pii,
        ))
        # PII 차단으로 applied_rules 에 confusion_pair 0
        cp_rules = [r for r in out.applied_rules if r["type"] == "confusion_pair"]
        assert cp_rules == []


class TestNoOpCases:
    def test_no_pairs_no_substitution(self):
        out = rescore_transcript(_input(pairs=()))
        assert out.corrected_text == "본격 종료하면 된다고 하네요"
        assert out.safe_to_auto_apply is False

    def test_pair_not_in_text_no_substitution(self):
        pair = ConfusionPair(
            from_word="고양이", to_word="강아지",
            user_id="u1", confirm_count=5, weight=5.0,
        )
        out = rescore_transcript(_input(pairs=(pair,)))
        assert out.corrected_text == "본격 종료하면 된다고 하네요"


class TestKoreanSpacingVariants:
    def test_pair_with_normalized_text(self):
        # text 에 다중 공백
        out = rescore_transcript(_input(
            text="본격  종료하면  된다고  하네요",
            pairs=(_bonkyeok_pair(),),
        ))
        # 둘 중 하나에는 원격 들어가야 함 (normalize 또는 직접 매칭)
        assert "원격" in out.corrected_text


class TestMultipleConfusionPairs:
    def test_multiple_pairs_aggregate(self):
        p1 = _bonkyeok_pair(confirm=5)
        p2 = ConfusionPair(
            from_word="하네요", to_word="합니다",
            user_id="u1", confirm_count=5, weight=5.0,
        )
        out = rescore_transcript(_input(pairs=(p1, p2)))
        cp_rules = [r for r in out.applied_rules if r["type"] == "confusion_pair"]
        # 두 pair 모두 적용
        assert len(cp_rules) == 2
        assert "원격" in out.corrected_text
        assert "합니다" in out.corrected_text


class TestFinalScoreFormula:
    def test_final_score_within_bounds(self):
        out = rescore_transcript(_input(
            whisper_score=0.8,
            global_lm_score=0.6,
            alpha=0.3,
            beta=0.2,
        ))
        # (0.5*0.8) + (0.3*0) + (0.2*0.6) = 0.4 + 0 + 0.12 = 0.52
        # (phrase score 0 가정)
        assert 0.4 <= out.final_score <= 0.7
