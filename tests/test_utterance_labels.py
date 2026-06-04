# -*- coding: utf-8 -*-
"""Tier A/B 통계·언어 라벨 단위 테스트."""
from app.utterance_labels import (
    speech_rate_wpm, filler_word_count, honorific_level, question_type,
    language_mix_flag, audio_quality_class, build_utterance_stat_labels,
)


class TestSpeechRate:
    def test_wpm(self):
        # 6 단어 / 30초 = 12 wpm
        assert speech_rate_wpm("하나 둘 셋 넷 다섯 여섯", 30.0) == 12.0

    def test_zero_duration(self):
        assert speech_rate_wpm("하나 둘", 0) is None
        assert speech_rate_wpm("", 10) is None


class TestFiller:
    def test_counts_fillers(self):
        assert filler_word_count("음 그 저기 내일 보자") == 3  # 음/그/저기
        assert filler_word_count("안녕하세요 반갑습니다") == 0


class TestHonorific:
    def test_formal(self):
        assert honorific_level("안녕하세요. 반갑습니다.") == "formal"

    def test_informal(self):
        assert honorific_level("내일 보자. 같이 먹자.") == "informal"

    def test_mixed(self):
        assert honorific_level("안녕하세요. 같이 가자.") == "mixed"

    def test_empty(self):
        assert honorific_level("") is None


class TestQuestion:
    def test_wh(self):
        assert question_type("이거 어디서 났어?") == "wh"

    def test_yes_no(self):
        assert question_type("지금 가요?") == "yes_no"

    def test_none(self):
        assert question_type("지금 갑니다.") is None


class TestLanguageMix:
    def test_latin(self):
        assert language_mix_flag("DLP 팝업창 떴어요") is True
        assert language_mix_flag("팝업창 떴어요") is False


class TestAudioQualityClass:
    def test_map(self):
        assert audio_quality_class("A") == "high"
        assert audio_quality_class("C") == "low"
        assert audio_quality_class(None) is None


class TestBuilder:
    def test_full_labels_and_silence(self):
        utt = {"transcript_text": "지금 어디 가요?", "duration_sec": 10.0,
               "start_sec": 12.0, "quality_grade": "A"}
        prev = {"end_sec": 11.5}
        out = build_utterance_stat_labels(utt, prev)
        assert out["question_type"] == "wh"
        assert out["audio_quality_class"] == "high"
        assert out["silence_before_sec"] == 0.5
        assert "label_source" not in out  # 라벨흐름 소유 — stat 에서 미설정

    def test_no_prev_no_silence(self):
        out = build_utterance_stat_labels({"transcript_text": "네", "duration_sec": 1.0}, None)
        assert "silence_before_sec" not in out
