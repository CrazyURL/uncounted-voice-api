"""Tests for Segmenter v2 — utterance-level postprocess merge.

v2는 word 단위가 아니라 이미 분리된 발화(utterance) 단위 리스트를 입력받아
같은 화자의 짧은 인접 발화를 문장 종결 단위에 가깝게 병합한다. (read-only 설계,
DB/재처리와 무관한 순수함수)
"""
import pytest

from app.services.utterance_segmenter_v2 import merge_v2


def _utt(text, start, end, speaker="SPEAKER_0", *, word_count=None,
         pii_intervals=None, numeric_patterns=None):
    return {
        "start_sec": start,
        "end_sec": end,
        "speaker_id": speaker,
        "transcript_text": text,
        "word_count": word_count if word_count is not None else len(text.split()),
        "pii_intervals": pii_intervals or [],
        "numeric_patterns": numeric_patterns or [],
    }


# 테스트 공통 임계값 (config 비의존)
KW = dict(gap_sec=0.8, max_merged_sec=13.0, short_sec=2.0, short_words=5)


class TestMergeBasic:
    def test_empty_returns_empty(self):
        assert merge_v2([], **KW) == []

    def test_single_unit_unchanged(self):
        units = [_utt("안녕하세요", 0.0, 1.0)]
        result = merge_v2(units, **KW)
        assert len(result) == 1
        assert result[0]["transcript_text"] == "안녕하세요"


class TestSameSpeakerShortMerge:
    def test_merges_two_short_same_speaker(self):
        # "이렇게 하면"(미완결, 짧음) + "된다"(0.4s gap) → 한 문장으로 병합
        units = [
            _utt("이렇게 하면", 0.0, 1.0),
            _utt("된다", 1.4, 2.0),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 1
        assert result[0]["transcript_text"] == "이렇게 하면 된다"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 2.0

    def test_does_not_merge_when_gap_too_large(self):
        # gap 1.0s > 0.8s → 병합 금지
        units = [
            _utt("그래서", 0.0, 0.6),
            _utt("말이죠", 1.6, 2.2),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 2

    def test_does_not_merge_different_speakers(self):
        units = [
            _utt("그래서", 0.0, 0.6, "SPEAKER_0"),
            _utt("맞아요", 0.8, 1.4, "SPEAKER_1"),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 2


class TestSentenceEndingStop:
    def test_stops_at_sentence_ending(self):
        # 첫 발화가 종결어미(~요)로 끝나면 다음 발화를 흡수하지 않는다
        units = [
            _utt("알겠어요", 0.0, 0.8),
            _utt("그리고요", 1.0, 1.6),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 2

    def test_merges_until_sentence_completes(self):
        # 미완결 조각이 누적되다 종결어미가 완성되면 그 지점에서 닫힌다
        units = [
            _utt("저는", 0.0, 0.4),       # 조사로 끝 (미완결)
            _utt("그래서", 0.6, 1.0),     # 연결 (미완결)
            _utt("갑니다", 1.2, 1.8),     # 종결 → 닫힘
            _utt("다음", 2.0, 2.4),       # 새 문장 시작
        ]
        result = merge_v2(units, **KW)
        assert result[0]["transcript_text"] == "저는 그래서 갑니다"
        assert len(result) == 2


class TestMaxMergedDuration:
    def test_stops_before_exceeding_max(self):
        # 각 0.4s 미완결 조각이 누적되지만 max_merged_sec(2.0s)를 넘기지 않는다
        units = [
            _utt("어", 0.0, 0.4, word_count=1),
            _utt("그", 0.5, 0.9, word_count=1),
            _utt("저", 1.0, 1.4, word_count=1),
            _utt("음", 1.5, 1.9, word_count=1),
            _utt("뭐", 2.0, 2.4, word_count=1),
            _utt("그게", 2.5, 2.9, word_count=1),
        ]
        result = merge_v2(units, gap_sec=0.8, max_merged_sec=2.0,
                          short_sec=2.0, short_words=5)
        for u in result:
            assert (u["end_sec"] - u["start_sec"]) <= 2.0 + 1e-6


class TestShortCandidateGuard:
    def test_long_unit_not_merge_candidate(self):
        # 현재 발화가 2초 이상 AND 5단어 이상이면 병합 후보가 아니다 (독립 유지)
        units = [
            _utt("저는 오늘 회사에 가서 일을 했어", 0.0, 3.0, word_count=6),
            _utt("그리고", 3.2, 3.6, word_count=1),
        ]
        result = merge_v2(units, **KW)
        # 첫 발화는 short 아님(미완결이어도 흡수 안 함) → 독립
        assert result[0]["transcript_text"] == "저는 오늘 회사에 가서 일을 했어"

    def test_short_by_words_even_if_long_duration(self):
        # duration은 길어도 단어<5면 여전히 후보 (OR 조건)
        units = [
            _utt("음 그러니까", 0.0, 2.5, word_count=2),
            _utt("그게 맞아", 2.8, 3.4, word_count=2),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 1


class TestPiiBoundaryProtection:
    def test_does_not_split_pii_spanning_boundary(self):
        # 전화번호 PII가 경계(2.0s)를 가로지름: 앞 발화 우측 끝 + 뒤 발화 좌측 시작에 PII
        # → 종결어미/short 게이트와 무관하게 병합해 PII span을 한 발화에 보존
        units = [
            _utt("번호는 010에", 0.0, 2.0, word_count=2,
                 pii_intervals=[{"startSec": 1.6, "endSec": 2.0, "piiType": "전화번호"}]),
            _utt("1234입니다", 2.1, 3.0, word_count=1,
                 pii_intervals=[{"startSec": 2.1, "endSec": 2.6, "piiType": "전화번호"}]),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 1, "경계를 가로지르는 PII span은 한 발화로 합쳐져야 함"
        # PII 정보는 union으로 보존
        assert len(result[0]["pii_intervals"]) == 2

    def test_merge_unions_pii_metadata(self):
        units = [
            _utt("계좌는", 0.0, 0.6, word_count=1,
                 numeric_patterns=[{"type": "account", "surface_masked": "***"}]),
            _utt("이거예요", 0.8, 1.4, word_count=1),
        ]
        result = merge_v2(units, **KW)
        assert len(result) == 1
        assert len(result[0]["numeric_patterns"]) == 1


class TestBidirectional:
    def test_long_incomplete_left_absorbs_short_continuation_when_enabled(self):
        # 미완결 긴 발화 + 짧은 연속 조각: forward(기본)는 못 잡고 bidirectional은 잡는다
        units = [
            _utt("저는 오늘 회사에 가서 일을 하다가", 0.0, 3.0, word_count=7),  # 미완결(연결어미)
            _utt("말았어요", 3.3, 3.9, word_count=1),                          # 짧은 종결 조각
        ]
        forward = merge_v2(units, **KW)
        assert len(forward) == 2, "forward-only는 긴 좌측을 흡수하지 않음"

        bidir = merge_v2(units, bidirectional=True, **KW)
        assert len(bidir) == 1, "bidirectional은 미완결 긴 발화가 짧은 연속 조각을 흡수"

    def test_bidirectional_keeps_complete_left_separate(self):
        # 종결된 긴 발화 뒤 짧은 조각은 bidirectional이어도 병합 안 함 (별개 문장)
        units = [
            _utt("저는 오늘 회사에 가서 일을 했습니다", 0.0, 3.0, word_count=7),  # 종결
            _utt("그리고", 3.3, 3.7, word_count=1),
        ]
        bidir = merge_v2(units, bidirectional=True, **KW)
        assert len(bidir) == 2, "완결 문장은 짧은 조각을 흡수하지 않음"


class TestImmutability:
    def test_input_not_mutated(self):
        units = [
            _utt("이렇게 하면", 0.0, 1.0),
            _utt("된다", 1.4, 2.0),
        ]
        snapshot = [dict(u) for u in units]
        merge_v2(units, **KW)
        assert units == snapshot, "입력 리스트/딕셔너리가 변경되면 안 됨"
