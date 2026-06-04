# -*- coding: utf-8 -*-
"""regex PII 발화(utterance) 마스킹 — text+words 동기 (Gate-1 근본수정)."""
from app.pii_masker import mask_utterance_pii


def _words(text):
    """공백 분리 word dict 리스트 (start/end 더미)."""
    out = []
    i = 0
    for w in text.split():
        out.append({"word": w, "start": float(i), "end": float(i + 1)})
        i += 1
    return out


class TestMaskUtterancePii:
    def test_no_pii_unchanged(self):
        t = "오늘 날씨가 참 좋네요"
        mt, mw, summ = mask_utterance_pii(t, _words(t))
        assert mt == t and summ == {} and [w["word"] for w in mw] == t.split()

    def test_phone_masked_in_text_and_words(self):
        t = "번호는 010-6390-9878 입니다"
        mt, mw, summ = mask_utterance_pii(t, _words(t))
        assert "010-6390-9878" not in mt
        assert "[PII_전화번호]" in mt
        # words 에도 원시 전화번호가 남으면 안 됨
        joined = " ".join(w["word"] for w in mw)
        assert "6390" not in joined
        assert any("[PII_전화번호]" in w["word"] for w in mw)
        assert summ.get("전화번호", 0) >= 1

    def test_multiword_phone_collapses_words(self):
        # 공백으로 쪼개진 전화번호 — 여러 word 가 하나의 토큰으로
        t = "지금 010 6390 9878 이에요"
        mt, mw, summ = mask_utterance_pii(t, _words(t))
        joined = " ".join(w["word"] for w in mw)
        assert "6390" not in joined and "9878" not in joined
        assert "[PII_전화번호]" in joined

    def test_rrn_masked(self):
        t = "주민번호 871204-1125625 확인"
        mt, _, summ = mask_utterance_pii(t, _words(t))
        assert "871204" not in mt and "[PII_주민등록번호]" in mt

    def test_input_not_mutated(self):
        t = "번호 010-1234-5678"
        w = _words(t)
        mask_utterance_pii(t, w)
        assert t == "번호 010-1234-5678"
        assert [x["word"] for x in w] == t.split()

    def test_token_time_range_spans_pii(self):
        # 토큰 word 의 end 는 마지막 겹친 word 의 end 까지 확장
        t = "콜 010 6390 9878 끝"
        _, mw, _ = mask_utterance_pii(t, _words(t))
        tok = [w for w in mw if "[PII_전화번호]" in w["word"]]
        assert tok and tok[0]["end"] >= tok[0]["start"]
