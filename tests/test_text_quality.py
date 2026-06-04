# -*- coding: utf-8 -*-
"""반복/루프 환각 축약 단위 테스트."""
from app.text_quality import collapse_repetitions, collapse_segment_repetitions


class TestCollapseRepetitions:
    def test_single_token_loop_collapsed(self):
        out, c = collapse_repetitions("하는지 하는지 하는지 하는지")
        assert out == "하는지" and c == 3

    def test_backchannel_one_char_preserved(self):
        assert collapse_repetitions("네 네 네") == ("네 네 네", 0)
        assert collapse_repetitions("아 아 아 아 아") == ("아 아 아 아 아", 0)

    def test_two_word_phrase_loop_collapsed(self):
        out, c = collapse_repetitions("그 다음 그 다음 그 다음 일이")
        assert out == "그 다음 일이" and c == 2

    def test_below_min_repeat_preserved(self):
        assert collapse_repetitions("진짜 진짜") == ("진짜 진짜", 0)

    def test_normal_text_unchanged(self):
        assert collapse_repetitions("안녕하세요 반갑습니다") == ("안녕하세요 반갑습니다", 0)

    def test_collapse_with_tail(self):
        out, c = collapse_repetitions("맞아 맞아 맞아 그래")
        assert out == "맞아 그래" and c == 2

    def test_empty(self):
        assert collapse_repetitions("") == ("", 0)


class TestCollapseSegmentRepetitions:
    def _seg(self, text, words):
        return {"text": text, "words": [{"word": w, "start": i} for i, w in enumerate(words)]}

    def test_text_and_words_collapsed_together(self):
        segs = [self._seg("하는지 하는지 하는지 하는지 맞아요",
                          ["하는지", "하는지", "하는지", "하는지", "맞아요"])]
        new, n = collapse_segment_repetitions(segs)
        assert n == 3
        assert new[0]["text"] == "하는지 맞아요"
        assert [w["word"] for w in new[0]["words"]] == ["하는지", "맞아요"]

    def test_input_not_mutated(self):
        segs = [self._seg("하는지 하는지 하는지", ["하는지", "하는지", "하는지"])]
        collapse_segment_repetitions(segs)
        assert segs[0]["text"] == "하는지 하는지 하는지"
        assert len(segs[0]["words"]) == 3

    def test_backchannel_segment_preserved(self):
        segs = [self._seg("네 네 네", ["네", "네", "네"])]
        new, n = collapse_segment_repetitions(segs)
        assert n == 0 and [w["word"] for w in new[0]["words"]] == ["네", "네", "네"]

    def test_words_with_leading_space_matched(self):
        segs = [{"text": "응 응 응 응", "words": [
            {"word": " 응"}, {"word": "응"}, {"word": " 응"}, {"word": "응"}]}]
        # 1글자 추임새이므로 보존 (content-len<2)
        _, n = collapse_segment_repetitions(segs)
        assert n == 0
