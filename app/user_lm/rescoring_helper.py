"""R2 rescoring helper — confusion pair + phrase memory + 6 guards 통합.

순수 함수. DB 없음. voice-api 호출 없음.
호출자가 user의 confusion_pairs/phrase_memory/pii_intervals 를 인자로 전달.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.user_lm.confusion_pair_engine import score_confusion_pair
from app.user_lm.phrase_memory_engine import score_phrase_match
from app.user_lm.types import (
    ConfusionPair,
    PhraseMemoryEntry,
    PiiInterval,
    RescoringInput,
    RescoringOutput,
    WordCandidate,
)


# 가드 임계 (사용자 명시)
MARGIN_THRESHOLD_NATS: float = 0.20
WHISPER_WORD_SCORE_MIN: float = 0.50
CONFUSION_PAIR_MIN_WEIGHT: float = 2.0

# α/β bounds (sanitize 시 clamp)
ALPHA_MAX: float = 0.6
BETA_MAX: float = 0.4

# 숫자/계좌/인증 보호 키워드 (Tier S/A HOTWORDS 의 부분집합)
NUMERIC_PROTECTION_KEYWORDS: tuple[str, ...] = (
    "주민등록번호", "주민번호",
    "외국인등록번호",
    "운전면허번호",
    "여권번호",
    "계좌번호",
    "카드번호",
    "인증번호",
    "비밀번호",
    "전화번호", "휴대전화번호",
    "사업자등록번호", "법인등록번호",
)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _word_overlaps_pii(
    word_start: float,
    word_end: float,
    pii_intervals: tuple[PiiInterval, ...],
) -> bool:
    for iv in pii_intervals:
        if not (word_end < iv.start or word_start > iv.end):
            return True
    return False


def _surface_redacted(surface: str) -> str:
    """provenance log 의 surface 노출 시 raw PII 출력 방지 — 길이만 유지한 placeholder."""
    if not surface:
        return ""
    return f"[REDACTED:{len(surface)}]"


def _numeric_protection_triggered(
    *,
    surface: str,
    context_before: str,
    context_after: str,
) -> bool:
    """surface 또는 인접 context 에 보호 키워드가 있으면 자동 정정 금지."""
    haystack = " ".join((surface, context_before, context_after))
    haystack_norm = _normalize(haystack)
    for kw in NUMERIC_PROTECTION_KEYWORDS:
        if kw in haystack_norm:
            return True
    return False


def _find_word_position(
    *,
    words: tuple[WordCandidate, ...],
    surface: str,
) -> Optional[WordCandidate]:
    """words 중 surface 와 매칭하는 WordCandidate 1개 (앞쪽 우선)."""
    if not surface:
        return None
    norm = _normalize(surface)
    for w in words:
        if _normalize(w.word) == norm:
            return w
    # 부분 매칭 (다음 token 이 norm 인 경우 등 — 보수적 skip)
    return None


def _gather_context(
    *,
    words: tuple[WordCandidate, ...],
    target: WordCandidate,
    n: int = 2,
) -> tuple[str, str]:
    """target word 의 앞/뒤 n token 을 context_before / context_after 로 반환."""
    if not words:
        return ("", "")
    try:
        idx = words.index(target)
    except ValueError:
        return ("", "")
    before = " ".join(w.word for w in words[max(0, idx - n):idx])
    after = " ".join(w.word for w in words[idx + 1:idx + 1 + n])
    return (before, after)


def rescore_transcript(
    input_: RescoringInput,
    *,
    golden_set_check: Optional[Callable[[str, str], bool]] = None,
) -> RescoringOutput:
    """6 guard 통합 rescoring.

    Args:
        input_: RescoringInput frozen 구조체
        golden_set_check: (original_text, corrected_text) -> bool
            False 반환 시 회귀 의심 → safe_to_auto_apply=False.
            None 이면 회귀 가드 비활성 (PR-R2c 의 ETL 단계에서 주입 예정).

    Returns:
        RescoringOutput frozen, applied_rules 의 surface 는 PII overlap 시 redacted.
    """
    # alpha/beta clamp
    alpha = max(0.0, min(ALPHA_MAX, float(input_.alpha)))
    beta = max(0.0, min(BETA_MAX, float(input_.beta)))
    # 합산이 1 초과 시 보호 — beta 우선 보존
    if alpha + beta > 1.0:
        alpha = max(0.0, 1.0 - beta)

    # 1) phrase memory score (cross-user 격리는 engine 내부)
    phrase_score, matched_phrases = score_phrase_match(
        text=input_.text,
        memory=input_.phrase_memory,
        user_id=input_.user_id,
    )

    # 2) confusion pair 시도 — 적용 가능한 pair 만 수집
    new_text = input_.text
    applied_rules: list[dict] = []
    margin_acc: float = 0.0
    reasons_block: list[str] = []

    for pair in input_.confusion_pairs:
        # weight 가드 (R2a 가 사용자 명시) — confusion pair 의 weight 가 2 미만이면 needs_review
        if pair.weight < CONFUSION_PAIR_MIN_WEIGHT:
            reasons_block.append(
                f"pair_weight<{CONFUSION_PAIR_MIN_WEIGHT}:from={_surface_redacted(pair.from_word) if input_.pii_intervals else pair.from_word}"
            )
            continue

        # context 수집 + word position
        word = _find_word_position(words=input_.words, surface=pair.from_word)
        ctx_before, ctx_after = ("", "")
        whisper_score = 1.0
        word_in_pii = False
        if word is not None:
            ctx_before, ctx_after = _gather_context(words=input_.words, target=word, n=2)
            whisper_score = word.score
            word_in_pii = _word_overlaps_pii(word.start, word.end, input_.pii_intervals)

        # Guard #2: PII span overlap
        if word_in_pii:
            reasons_block.append("pii_overlap")
            continue

        # Guard #3: numeric / 계좌 / 인증 보호
        if _numeric_protection_triggered(
            surface=pair.from_word,
            context_before=ctx_before,
            context_after=ctx_after,
        ):
            reasons_block.append("numeric_protection")
            continue

        # confusion pair 자체 scoring (cross-user / confirm count / reject / whisper word score)
        cs = score_confusion_pair(
            pair=pair,
            user_id=input_.user_id,
            context_before=ctx_before,
            context_after=ctx_after,
            whisper_word_score=whisper_score,
        )

        if cs.needs_review:
            reasons_block.append(f"confusion_pair_needs_review:{cs.reason}")
            continue

        if not cs.applied:
            reasons_block.append(f"confusion_pair_blocked:{cs.reason}")
            continue

        # text substitution (한 번만 — 첫 occurrence)
        norm_text = _normalize(new_text)
        norm_from = _normalize(pair.from_word)
        if norm_from not in norm_text:
            continue

        # surface 치환 — 정규화된 text 와 원본 text 모두 시도
        substituted = new_text.replace(pair.from_word, pair.to_word, 1)
        if substituted == new_text:
            # normalized 일치만 (공백 변종)
            substituted = norm_text.replace(norm_from, _normalize(pair.to_word), 1)

        if substituted == new_text:
            continue

        new_text = substituted
        margin_acc += cs.confidence_delta
        applied_rules.append({
            "type": "confusion_pair",
            "surface": _surface_redacted(pair.from_word) if word_in_pii else pair.from_word,
            "corrected": _surface_redacted(pair.to_word) if word_in_pii else pair.to_word,
            "reason": cs.reason,
            "weight": pair.weight,
        })

    # phrase memory 도 applied_rules 에 기록 (substitution 은 안 함, boost 만)
    if matched_phrases:
        applied_rules.append({
            "type": "phrase_memory_boost",
            "matched": matched_phrases,
            "boost": phrase_score,
        })

    # 3) final score 결합
    # phrase_score 정규화 (sigmoid-like: weight 합산 → 0~1 boost)
    phrase_score_norm = min(1.0, phrase_score / 10.0)  # weight 합 10 이상이면 max boost
    final_score = (
        (1.0 - alpha - beta) * float(input_.whisper_score)
        + alpha * phrase_score_norm
        + beta * float(input_.global_lm_score)
    )

    # margin: confusion pair 의 confidence_delta 합산 + (phrase boost 의 일부)
    margin = margin_acc + (alpha * phrase_score_norm * 0.5)

    # 4) guard #1: margin threshold
    safe = True
    needs_review_reasons: list[str] = []
    if margin < MARGIN_THRESHOLD_NATS and applied_rules and any(r.get("type") == "confusion_pair" for r in applied_rules):
        safe = False
        needs_review_reasons.append(f"margin {margin:.3f} < {MARGIN_THRESHOLD_NATS}")

    # 5) guard #6: golden set regression
    if golden_set_check is not None:
        try:
            if not golden_set_check(input_.text, new_text):
                safe = False
                needs_review_reasons.append("golden_set_regression")
        except Exception as e:  # noqa: BLE001
            safe = False
            needs_review_reasons.append(f"golden_set_error:{type(e).__name__}")

    # 6) applied 0 → safe_to_auto_apply 는 False (substitution 안 한 경우)
    has_confusion_substitution = any(r.get("type") == "confusion_pair" for r in applied_rules)

    return RescoringOutput(
        corrected_text=new_text,
        applied_rules=tuple(applied_rules),
        final_score=final_score,
        margin=margin,
        confidence_delta=margin,
        safe_to_auto_apply=(safe and has_confusion_substitution),
        needs_review=(not safe) and has_confusion_substitution,
        needs_review_reason=(
            "; ".join(needs_review_reasons) if needs_review_reasons else None
        ),
        source="user_lm",
        lm_version=input_.lm_version,
    )
