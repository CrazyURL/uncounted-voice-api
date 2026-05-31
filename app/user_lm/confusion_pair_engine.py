"""R2 confusion pair engine — pure scoring for user-specific 1:1 substitutions.

본격 ↔ 원격 류 사용자별 반복 오인식 보정의 점수 단위 계산.
DB 접근 없음, 외부 호출 없음, frozen data → frozen output.
"""
from __future__ import annotations

from typing import Optional

from app.user_lm.types import ConfusionPair, ConfusionScore


# 가드 임계값 (rescoring_helper 의 final guard 와 일관)
CONFIRM_COUNT_MIN: int = 2
WHISPER_WORD_SCORE_MIN: float = 0.5
CONTEXT_BONUS: float = 0.08          # nats — context 매칭 시 추가 margin
BASE_MARGIN_PER_CONFIRM: float = 0.05  # nats — 한 번 confirm 당 base margin
REJECT_PENALTY_PER_REJECT: float = 0.10  # nats — 한 번 reject 당 margin 차감


def _normalize_korean(text: str) -> str:
    """양끝 공백 제거 + 중간 공백 단일화. NFC 등은 호출자 책임."""
    return " ".join((text or "").strip().split())


def score_confusion_pair(
    *,
    pair: ConfusionPair,
    user_id: str,
    context_before: str = "",
    context_after: str = "",
    whisper_word_score: float = 1.0,
) -> ConfusionScore:
    """confusion pair 적용 시의 margin/applied 판정.

    가드:
      1. cross-user contamination 차단 (pair.user_id != user_id)
      2. confirm_count >= CONFIRM_COUNT_MIN
      3. reject_count < confirm_count
      4. whisper_word_score >= WHISPER_WORD_SCORE_MIN (낮으면 needs_review)
      5. context 매칭 시 bonus (정확도 가중)

    반환:
      ConfusionScore(score_delta, confidence_delta, reason, applied, needs_review)
        - applied=True 면 호출자가 substitution 실행 가능
        - needs_review=True 는 자동 적용 금지, 관리자 검수 큐
    """
    # Guard 1: cross-user
    if pair.user_id != user_id:
        return ConfusionScore(
            score_delta=0.0,
            confidence_delta=0.0,
            reason=f"cross_user_blocked:pair_owner={pair.user_id}",
            applied=False,
            needs_review=False,
        )

    # Guard 2: confirm count
    if pair.confirm_count < CONFIRM_COUNT_MIN:
        return ConfusionScore(
            score_delta=0.0,
            confidence_delta=0.0,
            reason=f"insufficient_confirm:{pair.confirm_count}<{CONFIRM_COUNT_MIN}",
            applied=False,
            needs_review=False,
        )

    # Guard 3: rejected suppress
    if pair.reject_count >= pair.confirm_count:
        return ConfusionScore(
            score_delta=0.0,
            confidence_delta=0.0,
            reason=f"rejected_suppress:reject={pair.reject_count}>=confirm={pair.confirm_count}",
            applied=False,
            needs_review=False,
        )

    # Margin 계산
    margin = pair.confirm_count * BASE_MARGIN_PER_CONFIRM
    margin -= pair.reject_count * REJECT_PENALTY_PER_REJECT

    # context bonus
    ctx_before_norm = _normalize_korean(context_before)
    ctx_after_norm = _normalize_korean(context_after)
    context_match = False
    if pair.contexts:
        for ctx in pair.contexts:
            ctx_norm = _normalize_korean(ctx)
            if ctx_norm and (ctx_norm in ctx_before_norm or ctx_norm in ctx_after_norm):
                margin += CONTEXT_BONUS
                context_match = True
                break

    # Guard 4: whisper word score
    if whisper_word_score < WHISPER_WORD_SCORE_MIN:
        return ConfusionScore(
            score_delta=margin,
            confidence_delta=margin,
            reason=(
                f"low_whisper_score:{whisper_word_score:.2f}<{WHISPER_WORD_SCORE_MIN}"
                + (";context_match" if context_match else "")
            ),
            applied=False,
            needs_review=True,
        )

    return ConfusionScore(
        score_delta=margin,
        confidence_delta=margin,
        reason=(
            f"applied:confirm={pair.confirm_count},reject={pair.reject_count}"
            + (";context_match" if context_match else "")
        ),
        applied=True,
        needs_review=False,
    )
