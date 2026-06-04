# -*- coding: utf-8 -*-
"""STT 평가 정규화 정책 (LOCKED 2026-06-03).

정본: docs/design_review_panel_redesign_20260603.md §0 정규화 정책.
모든 STT WER/CER 측정 스크립트는 반드시 이 모듈을 거쳐 정규화한 뒤 연산해야 한다.
한국어 띄어쓰기/조사 노이즈로 인한 '착시 점수'(단어단위 과소평가)를 통제하기 위함.
"""
import re
import jiwer

# 한글 음절 · 영숫자만 보존, 그 외(공백·문장부호·기호) 전부 제거
_KEEP = re.compile(r"[^0-9A-Za-z가-힣]")


def norm_chars(text: str) -> str:
    """규칙①: 모든 공백/문장부호 제거 + 한글·영숫자만 + 소문자.

    공백제거 Character 일치율(평가/바이어 리포트 정식 지표)의 전처리.
    """
    return _KEEP.sub("", text or "").lower()


def norm_words(text: str) -> str:
    """참고용(단어단위): 토큰별 정규화 + 공백 보존(토큰 경계 유지)."""
    toks = [_KEEP.sub("", t).lower() for t in (text or "").split()]
    return " ".join(t for t in toks if t)


def char_accuracy(reference: str, hypothesis: str) -> float:
    """규칙① 정식 지표: 공백제거 Character 정확도 = 1 - CER. 반환 [0.0, 1.0].

    바이어 납품 단가 산정·모델 평가의 단일 기준 지표.
    """
    ref, hyp = norm_chars(reference), norm_chars(hypothesis)
    if not ref:
        return 1.0 if not hyp else 0.0
    return max(0.0, 1.0 - jiwer.cer(ref, hyp))


def word_accuracy(reference: str, hypothesis: str) -> float:
    """참고 지표: 공백보존 단어 정확도 = 1 - WER. 반환 [0.0, 1.0].

    띄어쓰기 노이즈에 민감 — 단독 품질지표로 쓰지 말 것(규칙① 보조용).
    """
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    if not ref:
        return 1.0 if not hyp else 0.0
    return max(0.0, 1.0 - jiwer.wer(ref, hyp))


if __name__ == "__main__":
    # 자체검증: 띄어쓰기만 다른 두 문장은 규칙①에서 100%, 단어단위에선 폭락해야 함
    a = "어떤 거 되는 거죠"
    b = "어떤거 되는거죠"
    assert norm_chars(a) == norm_chars(b) == "어떤거되는거죠", norm_chars(a)
    assert char_accuracy(a, b) == 1.0, char_accuracy(a, b)
    assert word_accuracy(a, b) < 1.0, word_accuracy(a, b)
    # 진짜 오역은 규칙①에서도 감점
    assert char_accuracy("수석님 한가지", "선생님 한가지") < 1.0
    print("self-check PASS")
    print(f"  띄어쓰기차이: char={char_accuracy(a,b)*100:.0f}% word={word_accuracy(a,b)*100:.0f}%")
    print(f"  진짜오역    : char={char_accuracy('수석님','선생님')*100:.0f}%")
