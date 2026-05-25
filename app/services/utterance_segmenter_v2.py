"""Segmenter v2 — 발화 단위 후처리 병합 (postprocess merge).

기존 [utterance_segmenter.py]는 word 단위에서 5초 가드·0.5초 silence gap으로 분리하여
긴 발화 뒤에 따라오는 짧은 조각("그래서", "된다.")을 독립 row로 남긴다. v2는 그렇게
저장된 발화 단위 리스트를 입력받아 같은 화자의 짧은 인접 발화를 문장 종결 단위에
가깝게 병합한다.

순수함수 — 입력을 변경하지 않고 새 리스트/딕셔너리를 반환한다. DB write·재처리와 무관하며
진단 dry-run 및 (별도 승인 후) 라이브 파이프라인 후처리 양쪽에서 재사용 가능하다.

병합 정책 (사용자 지정):
  - 같은 화자 인접쌍만 검토 (화자 바뀌면 병합 금지)
  - gap <= MERGE_V2_GAP_SEC (기본 0.8초)
  - 현재 누적 발화가 (duration < MERGE_V2_SHORT_SEC) OR (단어 < MERGE_V2_SHORT_WORDS) 이면 후보
  - 병합 결과 duration이 MERGE_V2_MAX_SEC(기본 13초) 초과 시 중단
  - 현재 발화가 종결어미로 끝나면 중단 (문장 경계 보존)
  - PII span이 발화 경계를 가로지르면 위 게이트와 무관하게 병합해 span을 보존
"""

from app import config
from app.services.korean_sentence_ending import ends_with_sentence_ending

# PII span이 발화 경계에 '닿았다'고 볼 시간 허용오차 (초)
_PII_EDGE_EPS = 0.25


def merge_v2(
    units: list[dict],
    *,
    gap_sec: float | None = None,
    max_merged_sec: float | None = None,
    short_sec: float | None = None,
    short_words: int | None = None,
    bidirectional: bool = False,
) -> list[dict]:
    """짧은 인접 발화를 같은 화자·문장 종결 기준으로 병합한 새 리스트 반환.

    bidirectional=False (기본, 명세 정책): 현재 발화가 짧을 때만 다음을 흡수(forward).
    bidirectional=True: 미완결(종결어미 없음) 긴 발화가 뒤따르는 짧은 연속 조각도 흡수.
        진단 결과 forward-only가 같은-화자 과분할의 31%만 잡아 권장 옵션이다.
    """
    gap_sec = config.MERGE_V2_GAP_SEC if gap_sec is None else gap_sec
    max_merged_sec = config.MERGE_V2_MAX_SEC if max_merged_sec is None else max_merged_sec
    short_sec = config.MERGE_V2_SHORT_SEC if short_sec is None else short_sec
    short_words = config.MERGE_V2_SHORT_WORDS if short_words is None else short_words

    if not units:
        return []

    result: list[dict] = []
    current = _normalize(units[0])

    for raw_next in units[1:]:
        nxt = _normalize(raw_next)
        if _can_merge(current, nxt, gap_sec, max_merged_sec, short_sec, short_words, bidirectional):
            current = _merge(current, nxt)
        else:
            result.append(current)
            current = nxt

    result.append(current)
    return result


# -- Internal --

def _normalize(unit: dict) -> dict:
    """입력 발화를 v2 내부 표준 딕셔너리로 복사 (입력 미변경)."""
    text = unit.get("transcript_text") or ""
    word_count = unit.get("word_count")
    if word_count is None:
        word_count = len(text.split())
    return {
        "start_sec": _to_float(unit.get("start_sec")),
        "end_sec": _to_float(unit.get("end_sec")),
        "speaker_id": str(unit.get("speaker_id", "SPEAKER_00")),
        "transcript_text": text,
        "word_count": int(word_count),
        "pii_intervals": list(unit.get("pii_intervals") or []),
        "numeric_patterns": list(unit.get("numeric_patterns") or []),
        # word 단위 타임스탬프 패스스루 (파이프라인 wiring 시 _RawUtterance 재구성용; DB 경로는 미사용)
        "words": list(unit.get("words") or []),
    }


def _can_merge(
    current: dict,
    nxt: dict,
    gap_sec: float,
    max_merged_sec: float,
    short_sec: float,
    short_words: int,
    bidirectional: bool,
) -> bool:
    # 화자가 바뀌면 절대 병합 금지
    if current["speaker_id"] != nxt["speaker_id"]:
        return False

    gap = nxt["start_sec"] - current["end_sec"]

    # PII span이 경계를 가로지르면 게이트를 우회해 보존 (gap만 확인)
    if _pii_straddles_boundary(current, nxt):
        return gap <= gap_sec

    if gap > gap_sec:
        return False

    # 현재 발화가 문장으로 종결되면 다음 발화를 흡수하지 않음 (문장 경계 보존)
    if ends_with_sentence_ending(current["transcript_text"]):
        return False

    if _too_long(current, nxt, max_merged_sec):
        return False

    # forward: 현재 발화가 짧으면 후보 (duration OR 단어 수)
    if _is_short(current, short_sec, short_words):
        return True

    # bidirectional: 미완결 긴 발화 + 짧은 연속 조각도 흡수
    return bidirectional and _is_short(nxt, short_sec, short_words)


def _is_short(unit: dict, short_sec: float, short_words: int) -> bool:
    duration = unit["end_sec"] - unit["start_sec"]
    return duration < short_sec or unit["word_count"] < short_words


def _too_long(current: dict, nxt: dict, max_merged_sec: float) -> bool:
    return (nxt["end_sec"] - current["start_sec"]) > max_merged_sec


def _merge(current: dict, nxt: dict) -> dict:
    text = " ".join(p for p in (current["transcript_text"], nxt["transcript_text"]) if p)
    return {
        "start_sec": current["start_sec"],
        "end_sec": nxt["end_sec"],
        "speaker_id": current["speaker_id"],
        "transcript_text": text,
        "word_count": current["word_count"] + nxt["word_count"],
        "pii_intervals": current["pii_intervals"] + nxt["pii_intervals"],
        "numeric_patterns": current["numeric_patterns"] + nxt["numeric_patterns"],
        "words": current["words"] + nxt["words"],
    }


def _pii_straddles_boundary(current: dict, nxt: dict) -> bool:
    """PII interval이 current의 우측 끝과 nxt의 좌측 시작에 동시에 닿으면 True."""
    touches_right = any(
        _iv_end(iv) >= current["end_sec"] - _PII_EDGE_EPS
        for iv in current["pii_intervals"]
    )
    touches_left = any(
        _iv_start(iv) <= nxt["start_sec"] + _PII_EDGE_EPS
        for iv in nxt["pii_intervals"]
    )
    return touches_right and touches_left


def _iv_start(iv: dict) -> float:
    return _to_float(iv.get("startSec", iv.get("start")))


def _iv_end(iv: dict) -> float:
    return _to_float(iv.get("endSec", iv.get("end")))


def _to_float(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0
