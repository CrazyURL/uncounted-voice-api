# -*- coding: utf-8 -*-
"""STT 텍스트 품질 사후수정 — Whisper 반복/루프 환각 축약.

Whisper가 "하는지 하는지 하는지 하는지" 처럼 동일 토큰/구를 무한반복하는 루프 환각을
결정론적으로 축약한다(수정 우선, 마킹 아님). 추임새(네/응 등 1글자) 정상 반복은 보존.
설계: docs/design_review_panel_redesign_20260603.md §7.
"""
import re

_CONTENT = re.compile(r"[0-9A-Za-z가-힣]")


def _clen(tok: str) -> int:
    return len(_CONTENT.findall(tok))


def collapse_repetitions(text: str, min_repeat: int = 3, keep: int = 1) -> tuple[str, int]:
    """연속 반복 토큰/2-구를 축약. 반환 (수정문, 축약건수). immutable.

    - 단일 토큰 ≥min_repeat 연속 반복 + 실글자 ≥2 → keep개로 축약 (하는지×4 → 하는지)
    - 2-단어 구 ≥min_repeat 연속 반복 → keep회로 축약 (그 다음×3 → 그 다음)
    - 1글자 추임새(네/응/어) 반복은 보존 (정상 맞장구)
    """
    if not text:
        return text, 0
    words = text.split()
    n = len(words)
    out: list[str] = []
    collapsed = 0
    i = 0
    while i < n:
        # 2-단어 구 반복 우선 체크
        if i + 1 < n:
            phrase = (words[i], words[i + 1])
            reps = 0
            j = i
            while j + 1 < n and (words[j], words[j + 1]) == phrase:
                reps += 1; j += 2
            if reps >= min_repeat and (_clen(words[i]) + _clen(words[i + 1])) >= 2:
                out.extend(list(phrase) * keep)
                collapsed += reps - keep
                i = j
                continue
        # 단일 토큰 반복
        j = i
        while j < n and words[j] == words[i]:
            j += 1
        run = j - i
        if run >= min_repeat and _clen(words[i]) >= 2:
            out.extend([words[i]] * keep)
            collapsed += run - keep
        else:
            out.extend(words[i:j])
        i = j
    return " ".join(out), collapsed


def _collapse_words(words: list, min_repeat: int, keep: int) -> tuple[list, int]:
    """word dict 리스트에서 연속 반복 word를 드롭. 반환 (새 리스트, 드롭수). immutable."""
    n = len(words)
    out: list = []
    dropped = 0
    i = 0
    while i < n:
        wi = words[i]
        key = str(wi.get("word", "")).strip() if isinstance(wi, dict) else str(wi).strip()
        j = i
        while j < n:
            wj = words[j]
            kj = str(wj.get("word", "")).strip() if isinstance(wj, dict) else str(wj).strip()
            if kj != key:
                break
            j += 1
        run = j - i
        if run >= min_repeat and _clen(key) >= 2:
            out.extend(words[i:i + keep])
            dropped += run - keep
        else:
            out.extend(words[i:j])
        i = j
    return out, dropped


def collapse_segment_repetitions(
    segments: list, min_repeat: int = 3, keep: int = 1
) -> tuple[list, int]:
    """세그먼트의 text와 words를 동시에 반복축약(D 교훈: utterance는 words에서 재구성).

    반환 (새 segments, 총 축약건수). immutable — 입력 미변경, 새 객체 반환.
    """
    new_segs: list = []
    total = 0
    for seg in segments:
        if not isinstance(seg, dict):
            new_segs.append(seg)
            continue
        new_seg = dict(seg)
        words = seg.get("words")
        if isinstance(words, list) and words:
            nw, dropped = _collapse_words(words, min_repeat, keep)
            if dropped:
                new_seg["words"] = nw
                total += dropped
        nt, c = collapse_repetitions(str(seg.get("text", "")), min_repeat, keep)
        if c:
            new_seg["text"] = nt
            if not (isinstance(words, list) and words):
                total += c
        new_segs.append(new_seg)
    return new_segs, total
