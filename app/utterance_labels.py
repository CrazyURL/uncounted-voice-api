# -*- coding: utf-8 -*-
"""발화 통계/언어 라벨 (Tier A/B) — GPU 불요, 텍스트+타이밍 기반.

worker payload 에 컬럼만 있고 산출 코드가 없어 100% null 이던 write-orphan 라벨을 채운다:
  speech_rate_wpm / silence_before_sec / filler_word_count / honorific_level /
  question_type / language_mix_flag / audio_quality_class.
confidence_tier 는 word.probability 적재(P1-a/b) 후, label_source 는 라벨흐름 소유라 별도.
전부 결정론적·immutable.
"""
import re

_LATIN = re.compile(r"[A-Za-z]{2,}")            # 2자+ 라틴 = 외국어 혼용(단일자 제외)
_CONTENT = re.compile(r"[0-9A-Za-z가-힣]")
_WH = ("뭐", "무엇", "무슨", "누구", "언제", "어디", "왜", "어떻게", "어떤", "얼마", "몇")
_FILLERS = (
    "음", "어", "그", "저", "저기", "뭐", "막", "인제", "이제",
    "그니까", "그러니까", "아", "에", "거시기", "그게", "뭐랄까",
)
_FORMAL_END = ("요", "니다", "세요", "셔요", "까요", "데요", "구요", "네요", "지요", "시죠", "ㅂ니다")
_INFORMAL_END = ("어", "아", "지", "야", "거든", "자", "냐", "니", "대", "걸", "군")


def _tokens(text):
    return [t for t in (text or "").split() if _CONTENT.search(t)]


def speech_rate_wpm(text, duration_sec):
    """분당 단어 수. duration 0/None 이면 None."""
    if not duration_sec or duration_sec <= 0:
        return None
    n = len(_tokens(text))
    if n == 0:
        return None
    return round(n / (duration_sec / 60.0), 1)


def filler_word_count(text):
    """간투사/추임새 토큰 수 (음/어/그/저기 등 독립 토큰)."""
    cnt = 0
    for t in _tokens(text):
        clean = re.sub(r"[^가-힣]", "", t)
        if clean in _FILLERS:
            cnt += 1
    return cnt


def honorific_level(text):
    """문장 종결 기준 존댓말/반말. DB CHECK 허용값 'honorific'|'casual'|'mixed'|None."""
    toks = _tokens(text)
    if not toks:
        return None
    # 종결 후보 = 문장부호 직전 토큰들(.?! 기준) + 마지막 토큰
    sents = re.split(r"[.?!]", text or "")
    formal = informal = 0
    for s in sents:
        st = _tokens(s)
        if not st:
            continue
        last = re.sub(r"[^가-힣]", "", st[-1])
        if not last:
            continue
        if any(last.endswith(e) for e in _FORMAL_END):
            formal += 1
        elif any(last.endswith(e) for e in _INFORMAL_END):
            informal += 1
    if formal and informal:
        return "mixed"
    if formal:
        return "honorific"
    if informal:
        return "casual"
    return None


def question_type(text):
    """의문 유형. 'wh'|'yes_no'|None."""
    t = text or ""
    has_q = "?" in t or any(re.sub(r"[^가-힣]", "", w).endswith(("까", "나요", "까요", "죠", "니", "냐"))
                            for w in _tokens(t))
    if not has_q:
        return None
    if any(wh in t for wh in _WH):
        return "wh"
    return "yes_no"


_HANGUL_CH = re.compile(r"[가-힣]")


def language_mix_flag(text):
    """언어 구성. DB CHECK 허용값 'korean'|'english'|'mixed'|None.

    한글+라틴(2자+) 공존=mixed, 라틴만=english, 한글만=korean, 내용없음=None.
    """
    if not _CONTENT.search(text or ""):
        return None
    has_latin = bool(_LATIN.search(text or ""))
    has_kor = bool(_HANGUL_CH.search(text or ""))
    if has_latin and has_kor:
        return "mixed"
    if has_latin:
        return "english"
    return "korean"


def audio_quality_class(quality_grade):
    """quality_grade(A/B/C) → DB CHECK 허용값 excellent/good/fair."""
    return {"A": "excellent", "B": "good", "C": "fair"}.get(quality_grade)


def build_utterance_stat_labels(utt, prev_utt=None):
    """발화 1건의 Tier A/B 라벨 dict 산출(즉시 utt 에 병합 가능). immutable.

    utt: {"transcript_text","duration_sec","start_sec","quality_grade",...}
    prev_utt: 직전 발화(silence_before_sec 계산용) 또는 None.
    """
    text = utt.get("transcript_text", "")
    labels = {
        "speech_rate_wpm": speech_rate_wpm(text, utt.get("duration_sec")),
        "filler_word_count": filler_word_count(text),
        "honorific_level": honorific_level(text),
        "question_type": question_type(text),
        "language_mix_flag": language_mix_flag(text),
        "audio_quality_class": audio_quality_class(utt.get("quality_grade")),
    }
    if prev_utt is not None and utt.get("start_sec") is not None and prev_utt.get("end_sec") is not None:
        gap = float(utt["start_sec"]) - float(prev_utt["end_sec"])
        labels["silence_before_sec"] = round(max(0.0, gap), 2)
    return labels
