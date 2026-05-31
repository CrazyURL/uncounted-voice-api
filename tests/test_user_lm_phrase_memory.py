"""Unit tests — app/user_lm/phrase_memory_engine.py"""
from __future__ import annotations

import math

from app.user_lm.phrase_memory_engine import (
    MAX_PAIRS_PER_USER,
    MAX_PHRASES_PER_USER,
    cap_pairs,
    cap_phrases,
    decay_confusion_pairs,
    decay_phrase_memory,
    score_phrase_match,
)
from app.user_lm.types import ConfusionPair, PhraseMemoryEntry


def _phrase(
    *,
    surface: str = "원격 접속",
    weight: float = 1.0,
    user_id: str = "u1",
    last_used_at: float | None = None,
) -> PhraseMemoryEntry:
    return PhraseMemoryEntry(
        surface=surface,
        weight=weight,
        user_id=user_id,
        last_used_at=last_used_at,
    )


class TestPhraseMatch:
    def test_match_boosts_score(self):
        mem = [_phrase(surface="원격 접속", weight=3.5)]
        boost, matched = score_phrase_match(text="원격 접속 부탁드립니다", memory=mem, user_id="u1")
        assert boost == 3.5
        assert matched == ["원격 접속"]

    def test_no_match_zero_boost(self):
        mem = [_phrase(surface="원격 접속", weight=3.5)]
        boost, matched = score_phrase_match(text="회의실에서 인사", memory=mem, user_id="u1")
        assert boost == 0.0
        assert matched == []

    def test_cross_user_phrase_blocked(self):
        mem = [_phrase(surface="원격 접속", weight=3.5, user_id="other")]
        boost, matched = score_phrase_match(text="원격 접속 부탁드립니다", memory=mem, user_id="u1")
        assert boost == 0.0
        assert matched == []

    def test_korean_spacing_variants_normalized(self):
        mem = [_phrase(surface="원격  접속", weight=2.0)]  # 다중 공백
        boost, matched = score_phrase_match(text="네 원격 접속 됩니다", memory=mem, user_id="u1")
        assert boost == 2.0
        assert matched == ["원격 접속"]

    def test_multiple_phrases_sum(self):
        mem = [
            _phrase(surface="원격 접속", weight=2.0),
            _phrase(surface="회의실 예약", weight=1.5),
        ]
        boost, matched = score_phrase_match(
            text="원격 접속 후 회의실 예약 부탁드립니다", memory=mem, user_id="u1",
        )
        assert boost == 3.5
        assert set(matched) == {"원격 접속", "회의실 예약"}

    def test_negative_weight_clamped_to_zero(self):
        mem = [_phrase(surface="원격 접속", weight=-1.0)]
        boost, _ = score_phrase_match(text="원격 접속", memory=mem, user_id="u1")
        assert boost == 0.0

    def test_empty_text_zero_boost(self):
        mem = [_phrase(surface="원격 접속", weight=2.0)]
        boost, matched = score_phrase_match(text="", memory=mem, user_id="u1")
        assert boost == 0.0
        assert matched == []


class TestDecayPhraseMemory:
    def test_fresh_phrase_no_change(self):
        mem = [_phrase(surface="원격 접속", weight=2.0, last_used_at=100.0)]
        out = decay_phrase_memory(memory=mem, now_ts=100.0, half_life_days=30.0)
        assert out[0].weight == 2.0

    def test_half_life_exact(self):
        last = 0.0
        now = 30 * 86400.0  # 30 days
        mem = [_phrase(weight=4.0, last_used_at=last)]
        out = decay_phrase_memory(memory=mem, now_ts=now, half_life_days=30.0)
        assert math.isclose(out[0].weight, 2.0, rel_tol=1e-6)

    def test_no_last_used_unchanged(self):
        mem = [_phrase(weight=5.0, last_used_at=None)]
        out = decay_phrase_memory(memory=mem, now_ts=100.0, half_life_days=30.0)
        assert out[0].weight == 5.0
        assert out[0].last_used_at is None

    def test_half_life_zero_or_negative_no_decay(self):
        mem = [_phrase(weight=2.0, last_used_at=0.0)]
        out = decay_phrase_memory(memory=mem, now_ts=1e9, half_life_days=0)
        assert out[0].weight == 2.0
        out2 = decay_phrase_memory(memory=mem, now_ts=1e9, half_life_days=-1)
        assert out2[0].weight == 2.0


class TestDecayConfusionPairs:
    def test_pair_decay_same_rule(self):
        last = 0.0
        now = 60 * 86400.0  # 60 days
        pair = ConfusionPair(
            from_word="본격", to_word="원격", user_id="u1",
            confirm_count=3, weight=8.0, last_used_at=last,
        )
        out = decay_confusion_pairs(pairs=[pair], now_ts=now, half_life_days=30.0)
        # 2 half-lives → 8 → 4 → 2
        assert math.isclose(out[0].weight, 2.0, rel_tol=1e-6)
        # confirm_count 는 decay 영향 없음
        assert out[0].confirm_count == 3


class TestCapPhrases:
    def test_cap_default_max(self):
        # 600 entries → user 당 500 만 보존
        mem = [_phrase(surface=f"phrase{i}", weight=float(i)) for i in range(600)]
        out = cap_phrases(memory=mem)
        u1_count = sum(1 for e in out if e.user_id == "u1")
        assert u1_count == MAX_PHRASES_PER_USER

    def test_cap_weight_desc_priority(self):
        mem = [_phrase(surface="low", weight=1.0), _phrase(surface="high", weight=10.0)]
        out = cap_phrases(memory=mem, max_phrases=1)
        assert len(out) == 1
        assert out[0].surface == "high"

    def test_cap_per_user_isolated(self):
        mem = [
            _phrase(surface=f"p{i}", weight=1.0, user_id="u1") for i in range(3)
        ] + [
            _phrase(surface=f"q{i}", weight=1.0, user_id="u2") for i in range(3)
        ]
        out = cap_phrases(memory=mem, max_phrases=2)
        u1 = [e for e in out if e.user_id == "u1"]
        u2 = [e for e in out if e.user_id == "u2"]
        assert len(u1) == 2
        assert len(u2) == 2


class TestCapPairs:
    def test_cap_weight_desc(self):
        pairs = [
            ConfusionPair(from_word="a", to_word="b", user_id="u1", confirm_count=1, weight=1.0),
            ConfusionPair(from_word="x", to_word="y", user_id="u1", confirm_count=5, weight=5.0),
        ]
        out = cap_pairs(pairs=pairs, max_pairs=1)
        assert len(out) == 1
        assert out[0].from_word == "x"

    def test_cap_zero_returns_empty(self):
        pairs = [ConfusionPair(from_word="a", to_word="b", user_id="u1", weight=1.0)]
        assert cap_pairs(pairs=pairs, max_pairs=0) == tuple()
