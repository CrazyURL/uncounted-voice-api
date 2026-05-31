"""R2 phrase memory engine — user-specific phrase boost, decay, cap.

DB 없음. 호출자가 phrase memory list 를 인자로 전달.
"""
from __future__ import annotations

import math
from typing import Iterable

from app.user_lm.types import ConfusionPair, PhraseMemoryEntry


# 격리/cap 한도
MAX_PAIRS_PER_USER: int = 500
MAX_PHRASES_PER_USER: int = 500
DECAY_HALF_LIFE_DAYS_DEFAULT: float = 30.0
SECONDS_PER_DAY: float = 86400.0


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def score_phrase_match(
    *,
    text: str,
    memory: Iterable[PhraseMemoryEntry],
    user_id: str,
) -> tuple[float, list[str]]:
    """text 가 user 의 phrase memory 와 매칭되면 weight 합산 boost.

    cross-user contamination 차단: user_id 불일치 entry 는 skip.

    반환: (boost_score, matched_surfaces)
    """
    text_norm = _normalize(text)
    if not text_norm:
        return (0.0, [])
    boost = 0.0
    matched: list[str] = []
    for entry in memory:
        if entry.user_id != user_id:
            continue
        surface = _normalize(entry.surface)
        if not surface:
            continue
        if surface in text_norm:
            boost += max(0.0, float(entry.weight))
            matched.append(surface)
    return (boost, matched)


def decay_phrase_memory(
    *,
    memory: Iterable[PhraseMemoryEntry],
    now_ts: float,
    half_life_days: float = DECAY_HALF_LIFE_DAYS_DEFAULT,
) -> tuple[PhraseMemoryEntry, ...]:
    """last_used_at 기반 weight 감쇠.

    weight_new = weight * 0.5 ** (elapsed_days / half_life_days)
    last_used_at=None 인 경우 변경 없음 (cold entry).
    half_life_days <= 0 무시 → 변경 없음.
    """
    if half_life_days <= 0:
        return tuple(memory)
    half_seconds = half_life_days * SECONDS_PER_DAY
    out: list[PhraseMemoryEntry] = []
    for entry in memory:
        if entry.last_used_at is None:
            out.append(entry)
            continue
        elapsed = max(0.0, now_ts - entry.last_used_at)
        factor = math.pow(0.5, elapsed / half_seconds)
        new_weight = max(0.0, entry.weight * factor)
        out.append(PhraseMemoryEntry(
            surface=entry.surface,
            weight=new_weight,
            user_id=entry.user_id,
            last_used_at=entry.last_used_at,
        ))
    return tuple(out)


def decay_confusion_pairs(
    *,
    pairs: Iterable[ConfusionPair],
    now_ts: float,
    half_life_days: float = DECAY_HALF_LIFE_DAYS_DEFAULT,
) -> tuple[ConfusionPair, ...]:
    """confusion pair weight 의 last_used_at 기반 감쇠 (phrase 와 같은 규칙)."""
    if half_life_days <= 0:
        return tuple(pairs)
    half_seconds = half_life_days * SECONDS_PER_DAY
    out: list[ConfusionPair] = []
    for p in pairs:
        if p.last_used_at is None:
            out.append(p)
            continue
        elapsed = max(0.0, now_ts - p.last_used_at)
        factor = math.pow(0.5, elapsed / half_seconds)
        new_weight = max(0.0, p.weight * factor)
        out.append(ConfusionPair(
            from_word=p.from_word,
            to_word=p.to_word,
            user_id=p.user_id,
            confirm_count=p.confirm_count,
            reject_count=p.reject_count,
            contexts=p.contexts,
            last_used_at=p.last_used_at,
            weight=new_weight,
        ))
    return tuple(out)


def cap_phrases(
    *,
    memory: Iterable[PhraseMemoryEntry],
    max_phrases: int = MAX_PHRASES_PER_USER,
) -> tuple[PhraseMemoryEntry, ...]:
    """user 당 최대 N phrase. weight desc → last_used_at desc → surface 순.

    cap 후 항상 user 별로 그룹화 보존 (user_id 다르면 별도 cap).
    """
    if max_phrases <= 0:
        return tuple()
    by_user: dict[str, list[PhraseMemoryEntry]] = {}
    for e in memory:
        by_user.setdefault(e.user_id, []).append(e)
    out: list[PhraseMemoryEntry] = []
    for uid, entries in by_user.items():
        # weight desc, last_used_at desc (None → -inf), surface asc
        entries.sort(
            key=lambda x: (
                -x.weight,
                -(x.last_used_at if x.last_used_at is not None else float("-inf")),
                x.surface,
            )
        )
        out.extend(entries[:max_phrases])
    return tuple(out)


def cap_pairs(
    *,
    pairs: Iterable[ConfusionPair],
    max_pairs: int = MAX_PAIRS_PER_USER,
) -> tuple[ConfusionPair, ...]:
    """user 당 최대 N pair. weight desc → confirm_count desc → last_used_at desc."""
    if max_pairs <= 0:
        return tuple()
    by_user: dict[str, list[ConfusionPair]] = {}
    for p in pairs:
        by_user.setdefault(p.user_id, []).append(p)
    out: list[ConfusionPair] = []
    for uid, plist in by_user.items():
        plist.sort(
            key=lambda x: (
                -x.weight,
                -x.confirm_count,
                -(x.last_used_at if x.last_used_at is not None else float("-inf")),
                x.from_word,
            )
        )
        out.extend(plist[:max_pairs])
    return tuple(out)
