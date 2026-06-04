# -*- coding: utf-8 -*-
"""정적 사전 기반 PII 이름 탐지 — 원시 윈도우 스캔(Kiwi 불요, O(n)).

하이브리드 정책(설계 §6):
  - 풀네임(성+이름) 매칭 → AUTO_MASK (FP 0 실측, 고확신)
  - 호격(이름+아/야) 매칭 → REVIEW_FLAG (FP 위험, 사람 확정)

Kiwi 토크나이저는 이름을 불안정하게 분리(김지민→김+지+민)하므로 사용하지 않고,
띄어쓰기 단위 독립 3글자 토큰을 원시 매칭한다(연음·음절겹침 오탐 차단).
"""
import re
from dataclasses import dataclass

from app.ner_guard.static_names import FULL_NAMES, VOCATIVES

_TOKEN = re.compile(r"\S+")
_HANGUL = re.compile(r"[^가-힣]")

# 이름 뒤에 붙는 조사/어미 (3글자 이름 + 조사 토큰 처리용: "김현정이"→"김현정")
_JOSA = frozenset(
    "이 가 은 는 을 를 과 와 도 만 의 에 께 요 씨 님 이가 에서 에게 한테 으로 께서 "
    "라고 이라고 입니다 이에요 예요 이고 이라 이지 인데 이야 처럼 부터 까지 마저 조차".split()
)


@dataclass(frozen=True)
class NameHit:
    text: str          # 매칭된 한글(예: "김현정")
    start: int         # 원문 토큰 시작 offset
    end: int           # 원문 토큰 끝 offset
    kind: str          # "full" (자동마스킹) | "vocative" (검수)


def detect_name_hits(text: str) -> list[NameHit]:
    """띄어쓰기 단위 독립 3글자 토큰을 사전 매칭. immutable 결과."""
    hits: list[NameHit] = []
    for m in _TOKEN.finditer(text or ""):
        clean = _HANGUL.sub("", m.group())
        if len(clean) == 3:
            if clean in FULL_NAMES:
                hits.append(NameHit(clean, m.start(), m.end(), "full"))
            elif clean in VOCATIVES:
                hits.append(NameHit(clean, m.start(), m.end(), "vocative"))
        elif len(clean) > 3 and clean[:3] in FULL_NAMES and clean[3:] in _JOSA:
            # 조사 붙은 토큰: "김현정이" → 풀네임 "김현정" + 조사 "이"
            hits.append(NameHit(clean[:3], m.start(), m.end(), "full"))
    return hits


def auto_mask_names(text: str, mask_token: str = "[PII_이름]") -> tuple[str, list[NameHit]]:
    """풀네임(고확신)만 자동 마스킹. 호격은 건드리지 않고 플래그로만 반환.

    반환: (마스킹된 텍스트, 전체 NameHit 목록[full+vocative]).
    immutable: 입력 미변형.
    """
    hits = detect_name_hits(text)
    full = [h for h in hits if h.kind == "full"]
    if not full:
        return text, hits
    # 뒤에서부터 치환해 offset 보존
    out = text
    for h in sorted(full, key=lambda x: x.start, reverse=True):
        # 토큰 내 한글(이름) 부분만 마스킹, 조사/문장부호는 보존
        token = out[h.start:h.end]
        masked = re.sub(re.escape(h.text), mask_token, token, count=1)
        out = out[:h.start] + masked + out[h.end:]
    return out, hits


def review_flags(text: str) -> list[NameHit]:
    """검수 플래그 후보(호격 = 저우선). 풀네임은 자동마스킹되므로 제외."""
    return [h for h in detect_name_hits(text) if h.kind == "vocative"]


def mask_utterance(text: str, words: list[dict], mask_token: str = "[PII_이름]"):
    """utterance용 A형 자동마스킹: transcript_text + words 동기화.

    utterance.transcript_text 가 words 에서 재구성되므로 text·words 둘 다 마스킹해야
    납품 데이터에 실명이 남지 않는다(이우주/김현정 누출 근본원인).
    반환: (masked_text, masked_words, n_masked, all_hits). immutable.
    """
    masked_text, hits = auto_mask_names(text, mask_token)
    full_names = {h.text for h in hits if h.kind == "full"}
    if not full_names:
        return text, list(words), 0, hits
    new_words = []
    for w in words:
        wt = w.get("word", "")
        new_wt = wt
        for nm in full_names:
            if nm in _HANGUL.sub("", new_wt):
                new_wt = re.sub(re.escape(nm), mask_token, new_wt)
        new_words.append({**w, "word": new_wt} if new_wt != wt else w)
    return masked_text, new_words, len(full_names), hits
