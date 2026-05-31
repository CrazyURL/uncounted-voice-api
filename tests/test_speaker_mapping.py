"""Unit tests — app/speaker_mapping.py raw pyannote direct mapping."""
from __future__ import annotations

import pytest

from app.speaker_mapping import (
    _assign_one,
    _compute_overlap_ranges,
    _split_on_speaker_change,
    assign_speakers,
)


# ───── helpers ─────

def _word(w: str, s: float, e: float) -> dict:
    return {"word": w, "start": s, "end": e}


def _seg(start: float, end: float, text: str, words: list[dict]) -> dict:
    return {"start": start, "end": end, "text": text, "words": words}


def _diar(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


# ───── _assign_one ─────

class TestAssignOne:
    def test_exact_match_returns_speaker_with_exact_source(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        sp, src = _assign_one(1.0, 2.0, diar, 0.15, 0.30, 0.7)
        assert sp == "SPEAKER_00"
        assert src == "exact"

    def test_no_diar_segments(self):
        sp, src = _assign_one(1.0, 2.0, [], 0.15, 0.30, 0.7)
        assert sp is None
        assert src == "no_diar_segments"

    def test_no_timestamp(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        sp, src = _assign_one(None, 2.0, diar, 0.15, 0.30, 0.7)
        assert sp is None
        assert src == "no_timestamp"

    def test_overlap_two_speakers(self):
        diar = [
            _diar("SPEAKER_00", 0.0, 5.0),
            _diar("SPEAKER_01", 2.0, 7.0),  # overlap 2.0~5.0
        ]
        sp, src = _assign_one(3.0, 4.0, diar, 0.15, 0.30, 0.7)
        assert sp is None
        assert src.startswith("overlap_")
        assert "SPEAKER_00" in src and "SPEAKER_01" in src

    def test_tolerance_default_150ms(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        # mid=5.1 (밖, 100ms) → tolerance 적용
        sp, src = _assign_one(5.05, 5.15, diar, 0.15, 0.30, 0.7)
        assert sp == "SPEAKER_00"
        assert src.startswith("tolerance_")

    def test_backchannel_short_word_300ms(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        # mid=5.25 (밖, 250ms), word_dur=0.1 (짧음) → back-channel
        sp, src = _assign_one(5.2, 5.3, diar, 0.15, 0.30, 0.7)
        assert sp == "SPEAKER_00"
        assert src.startswith("backchannel_")

    def test_backchannel_long_word_rejected(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        # mid=5.25 (밖, 250ms), word_dur=1.0 (>0.7) → ambiguous
        sp, src = _assign_one(4.75, 5.75, diar, 0.15, 0.30, 0.7)
        # mid=5.25, segment end=5.0, dist=0.25 → tolerance default (0.15) 초과
        # word_dur=1.0 > 0.7 → back-channel mode 거부 → ambiguous
        assert sp is None
        assert src == "ambiguous"

    def test_ambiguous_beyond_max_tolerance(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        # mid=5.5 (밖, 500ms) → tolerance_max(300) 초과 → ambiguous
        sp, src = _assign_one(5.45, 5.55, diar, 0.15, 0.30, 0.7)
        assert sp is None
        assert src == "ambiguous"


# ───── _split_on_speaker_change ─────

class TestSplitOnSpeakerChange:
    def test_no_change_single_segment(self):
        words = [
            {"word": "a", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"word": "b", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]
        out = _split_on_speaker_change(words, "a b")
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[0]["text"] == "a b"

    def test_speaker_change_splits(self):
        words = [
            {"word": "a", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"word": "b", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ]
        out = _split_on_speaker_change(words, "a b")
        assert len(out) == 2
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[1]["speaker"] == "SPEAKER_01"

    def test_none_to_speaker_splits(self):
        words = [
            {"word": "a", "start": 0.0, "end": 1.0, "speaker": None},
            {"word": "b", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]
        out = _split_on_speaker_change(words, "a b")
        assert len(out) == 2
        assert out[0]["speaker"] is None
        assert out[1]["speaker"] == "SPEAKER_00"

    def test_empty_words(self):
        out = _split_on_speaker_change([], "")
        assert out == []


# ───── _compute_overlap_ranges ─────

class TestComputeOverlapRanges:
    def test_no_overlap(self):
        diar = [
            _diar("SPEAKER_00", 0.0, 1.0),
            _diar("SPEAKER_01", 2.0, 3.0),
        ]
        out = _compute_overlap_ranges(diar)
        assert out == []

    def test_clear_overlap(self):
        diar = [
            _diar("SPEAKER_00", 0.0, 3.0),
            _diar("SPEAKER_01", 2.0, 5.0),
        ]
        out = _compute_overlap_ranges(diar)
        assert len(out) == 1
        assert out[0]["start"] == 2.0
        assert out[0]["end"] == 3.0
        assert out[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    def test_overlap_below_threshold_filtered(self):
        diar = [
            _diar("SPEAKER_00", 0.0, 3.0),
            _diar("SPEAKER_01", 2.99, 5.0),  # 10ms overlap
        ]
        out = _compute_overlap_ranges(diar, overlap_min_s=0.05)
        assert out == []


# ───── assign_speakers (end-to-end) ─────

class TestAssignSpeakers:
    def test_basic_two_speakers_no_mixed(self):
        """본인 → 상대 분할 보장. mixed segment 0."""
        diar = [
            _diar("SPEAKER_00", 0.0, 5.0),
            _diar("SPEAKER_01", 5.5, 10.0),
        ]
        result = {
            "segments": [
                _seg(0.0, 10.0, "안녕 잘 지내", [
                    _word("안녕", 1.0, 2.0),
                    _word("잘", 6.0, 6.5),
                    _word("지내", 7.0, 8.0),
                ]),
            ],
        }
        out = assign_speakers(diar, result)
        segs = out["segments"]
        # 1 input segment, but speaker change → 2 sub-segments
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPEAKER_00"
        assert segs[1]["speaker"] == "SPEAKER_01"
        # mixed (한 segment 안 2 speaker word) = 0
        for seg in segs:
            sps = {w.get("speaker") for w in seg["words"] if w.get("speaker")}
            assert len(sps) <= 1

    def test_short_backchannel_label_preserved(self):
        """cell F (tolerance=0) 에서 None 처리되던 짧은 back-channel 가
        Phase 3 (tolerance ±300ms back-channel mode) 에서는 라벨."""
        diar = [
            _diar("SPEAKER_00", 0.0, 5.0),
            _diar("SPEAKER_01", 5.5, 10.0),
        ]
        result = {
            "segments": [
                _seg(5.1, 5.4, "네", [_word("네", 5.1, 5.3)]),
            ],
        }
        out = assign_speakers(diar, result)
        # word.mid=5.2, segment 안에 없음. 가장 가까운 segment = SPEAKER_00 (end=5.0, dist=0.2) 또는 SPEAKER_01 (start=5.5, dist=0.3)
        # SPEAKER_00 이 더 가까움(0.2s) → tolerance(0.15) 초과 → back-channel mode
        # word_dur=0.2 ≤ 0.7 → 통과 → SPEAKER_00
        assert out["segments"][0]["speaker"] == "SPEAKER_00"
        assert out["segments"][0]["words"][0]["speaker_source"].startswith("backchannel_")

    def test_overlap_speaker_none_preserved(self):
        """overlap 구간 word 는 강제 배정 안 됨 (None 보존)."""
        diar = [
            _diar("SPEAKER_00", 0.0, 5.0),
            _diar("SPEAKER_01", 2.0, 7.0),
        ]
        result = {
            "segments": [
                _seg(0.0, 7.0, "overlap word", [
                    _word("overlap", 3.0, 3.5),  # 두 segment 모두 안 → overlap
                ]),
            ],
        }
        out = assign_speakers(diar, result)
        assert out["segments"][0]["speaker"] is None
        assert out["segments"][0]["words"][0]["speaker_source"].startswith("overlap_")

    def test_overlap_ranges_attached(self):
        # word range 가 sub-segment 로 잘려도 overlap (2~3) 와 겹쳐야 attach
        diar = [
            _diar("SPEAKER_00", 0.0, 3.0),
            _diar("SPEAKER_01", 2.0, 5.0),
        ]
        # word 1.0~2.5 → mid=1.75 SPEAKER_00 단독 안 (< 2.0) → exact, sub-segment 1.0~2.5
        # overlap 2~3 와 겹침 2~2.5 → attach
        result = {
            "segments": [_seg(1.0, 2.5, "x", [_word("x", 1.0, 2.5)])],
        }
        out = assign_speakers(diar, result)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"
        assert len(out["segments"][0]["overlap_ranges"]) == 1
        assert out["segments"][0]["overlap_ranges"][0]["start"] == 2.0
        assert out["segments"][0]["overlap_ranges"][0]["end"] == 3.0

    def test_no_diar_returns_none_speaker(self):
        result = {
            "segments": [_seg(0.0, 5.0, "lonely", [_word("lonely", 1.0, 2.0)])],
        }
        out = assign_speakers([], result)
        assert out["segments"][0]["speaker"] is None
        assert out["segments"][0]["words"][0]["speaker_source"] == "no_diar_segments"

    def test_empty_words_in_segment(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        result = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "no words", "words": []}],
        }
        out = assign_speakers(diar, result)
        assert len(out["segments"]) == 1
        assert out["segments"][0]["speaker"] is None
        assert out["segments"][0]["source_distribution"] == {"no_words": 1}

    def test_source_distribution_summary(self):
        diar = [_diar("SPEAKER_00", 0.0, 10.0)]
        result = {
            "segments": [
                _seg(0.0, 10.0, "exact x2 then backchannel", [
                    _word("exact1", 1.0, 1.5),
                    _word("exact2", 5.0, 5.5),
                    _word("bc", 10.2, 10.3),
                ]),
            ],
        }
        out = assign_speakers(diar, result)
        # 모든 word 가 SPEAKER_00, mixed=0, 1 sub-segment
        seg = out["segments"][0]
        assert seg["speaker"] == "SPEAKER_00"
        dist = seg["source_distribution"]
        assert dist.get("exact") == 2
        assert dist.get("backchannel") == 1

    def test_tolerance_ms_param_override(self):
        diar = [_diar("SPEAKER_00", 0.0, 5.0)]
        result = {
            "segments": [_seg(5.0, 5.5, "x", [_word("x", 5.1, 5.2)])],
        }
        # tolerance_default 0 + tolerance_max 0 → ambiguous (강제 X)
        out = assign_speakers(diar, result, tolerance_default_ms=0, tolerance_max_ms=0)
        assert out["segments"][0]["speaker"] is None
        assert out["segments"][0]["words"][0]["speaker_source"] == "ambiguous"

    def test_speaker_change_recall_3way(self):
        """본인 → 상대 → 본인 → 상대 4 turn. sub-segment 4개."""
        diar = [
            _diar("SPEAKER_00", 0.0, 2.0),
            _diar("SPEAKER_01", 2.5, 4.0),
            _diar("SPEAKER_00", 4.5, 6.0),
            _diar("SPEAKER_01", 6.5, 8.0),
        ]
        result = {
            "segments": [
                _seg(0.0, 8.0, "본인 상대 본인 상대", [
                    _word("본인1", 0.5, 1.0),
                    _word("상대1", 3.0, 3.5),
                    _word("본인2", 5.0, 5.5),
                    _word("상대2", 7.0, 7.5),
                ]),
            ],
        }
        out = assign_speakers(diar, result)
        assert len(out["segments"]) == 4
        speakers_seq = [s["speaker"] for s in out["segments"]]
        assert speakers_seq == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
