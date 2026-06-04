# -*- coding: utf-8 -*-
"""검수 소프트플래그 빌더 단위 테스트."""
from app.review_flags_builder import build_utterance_review_flags, SEVERITY_WEIGHT


class TestBuildUtteranceReviewFlags:
    def test_clean_text_no_flags(self):
        flags, score = build_utterance_review_flags("오늘 날씨가 좋네요")
        assert flags == [] and score == 0

    def test_empty(self):
        assert build_utterance_review_flags("") == ([], 0)
        assert build_utterance_review_flags(None) == ([], 0)

    def test_object_nim_flagged_med(self):
        # 사물+님(공인인증서님) → Nim-Guard med
        flags, score = build_utterance_review_flags("공인인증서님 어디 계세요")
        nim = [f for f in flags if f["type"] == "object_nim"]
        assert nim, "Nim-Guard 플래그 기대"
        assert nim[0]["severity"] == "med"
        assert score >= SEVERITY_WEIGHT["med"]

    def test_score_is_sum_of_weights(self):
        flags, score = build_utterance_review_flags("공인인증서님 안녕하세요")
        assert score == sum(SEVERITY_WEIGHT[f["severity"]] for f in flags)

    def test_flag_shape(self):
        flags, _ = build_utterance_review_flags("공인인증서님")
        for f in flags:
            assert {"type", "severity", "detail"} <= set(f)
            assert f["severity"] in SEVERITY_WEIGHT
