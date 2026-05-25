"""Utterance boundary segmentation — ported from Android UtteranceSegmenter.java."""

from dataclasses import dataclass

from app import config
from app.services.utterance_segmenter_v2 import merge_v2


@dataclass(frozen=True)
class UtteranceBoundary:
    start_sec: float
    end_sec: float
    duration_sec: float
    padded_start_sec: float
    padded_end_sec: float
    speaker_id: str
    transcript_text: str
    words: tuple


def segment(words: list[dict], total_duration: float) -> list[UtteranceBoundary]:
    """Split words into utterance boundaries by speaker change and silence gaps."""
    if not words:
        return []

    raw = _split_by_boundaries(words)
    fixed = _fix_hanging_words(raw)
    merged = _merge_short_utterances(fixed)
    split = _split_long_utterances(merged)
    final = _merge_v2_step(split)
    return _apply_padding(final, total_duration)


# -- Internal types --

@dataclass
class _RawUtterance:
    speaker_id: str
    words: list[dict]

    @property
    def duration(self) -> float:
        if not self.words:
            return 0.0
        return _to_float(self.words[-1].get("end")) - _to_float(self.words[0].get("start"))

    def merge_with(self, other: "_RawUtterance") -> "_RawUtterance":
        return _RawUtterance(self.speaker_id, list(self.words) + list(other.words))


# -- Step 1 & 2: Split by speaker change + silence --

def _split_by_boundaries(words: list[dict]) -> list[_RawUtterance]:
    utterances: list[_RawUtterance] = []
    current_words: list[dict] = [words[0]]
    current_speaker = _get_speaker_id(words[0])

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        word = words[i]
        speaker = _get_speaker_id(word)
        gap = _to_float(word.get("start")) - _to_float(prev_word.get("end"))

        if speaker != current_speaker or gap >= config.SILENCE_GAP_SEC:
            if current_words:
                utterances.append(_RawUtterance(current_speaker, current_words))
            current_words = []
            current_speaker = speaker

        current_words.append(word)

    if current_words:
        utterances.append(_RawUtterance(current_speaker, current_words))

    return utterances


# -- Step 2: Fix hanging words --

def _fix_hanging_words(utterances: list[_RawUtterance]) -> list[_RawUtterance]:
    """발화 끝에 고립된 단어를 다음 발화 앞으로 이동.

    조건: 발화 마지막 단어의 직전 gap >= HANGING_WORD_GAP_SEC
          AND 현재 발화 단어 수 > 1
          AND 다음 발화 화자가 같음 (화자 보존)
    """
    result = list(utterances)

    for i in range(len(result) - 1):
        u = result[i]
        next_u = result[i + 1]

        if len(u.words) <= 1:
            continue

        last_word = u.words[-1]
        prev_word = u.words[-2]
        gap_before_last = _to_float(last_word.get("start")) - _to_float(prev_word.get("end"))

        if gap_before_last < config.HANGING_WORD_GAP_SEC:
            continue

        if _get_speaker_id(last_word) != next_u.speaker_id:
            continue

        result[i] = _RawUtterance(u.speaker_id, u.words[:-1])
        result[i + 1] = _RawUtterance(next_u.speaker_id, [last_word] + list(next_u.words))

    return result


# -- Step 3 & 5: Merge short, preserve short answers --

def _normalize_word(raw: str) -> str:
    """단어에서 구두점/공백 제거하여 SHORT_ANSWER_WORDS 매칭용으로 정규화."""
    return (raw or "").strip().strip(".,!?\"'~…:;").lower()


def _is_short_answer(u: _RawUtterance) -> bool:
    """utterance_segmenter step에서 '독립 유지할 짧은 답변'으로 판정.

    조건: 2단어 이하 + SHORT_ANSWER_MIN_SEC 이상 + 단어 중 하나라도 SHORT_ANSWER_WORDS.
    """
    if len(u.words) > 2:
        return False
    if u.duration < config.SHORT_ANSWER_MIN_SEC:
        return False
    return any(
        _normalize_word(w.get("word") or w.get("text") or "") in config.SHORT_ANSWER_WORDS
        for w in u.words
    )


def _is_backchannel(u: _RawUtterance) -> bool:
    """진짜 맞장구(backchannel)인지 엄격하게 판정.

    Case C의 과병합을 방지하기 위한 조건:
    - 1초 미만
    - 2단어 이하
    - 모든 단어가 SHORT_ANSWER_WORDS에 속해야 함 (콘텐츠 단어 섞이면 False)
    """
    if u.duration >= 1.0:
        return False
    if len(u.words) > 2:
        return False
    return all(
        _normalize_word(w.get("word") or w.get("text") or "") in config.SHORT_ANSWER_WORDS
        for w in u.words
    )


def _merge_short_utterances(utterances: list[_RawUtterance]) -> list[_RawUtterance]:
    if len(utterances) <= 1:
        return list(utterances)

    working = list(utterances)
    result: list[_RawUtterance] = [working[0]]

    for i in range(1, len(working)):
        curr = working[i]
        if curr.duration >= config.MIN_UTTERANCE_SEC or _is_short_answer(curr):
            result.append(curr)
            continue

        last = result[-1]

        # 같은 화자 → 이전 발화에 병합 (단, last가 이미 충분히 길면 독립 유지)
        if last.speaker_id == curr.speaker_id:
            if last.duration < config.MIN_UTTERANCE_SEC:
                result[-1] = last.merge_with(curr)
            else:
                result.append(curr)
        # 다른 화자 + 다음 발화가 같은 화자 → 다음에 병합
        elif i + 1 < len(working) and working[i + 1].speaker_id == curr.speaker_id:
            working[i + 1] = curr.merge_with(working[i + 1])
        # 진짜 맞장구만 이전 발화에 병합 (Case C — 엄격 조건)
        elif _is_backchannel(curr):
            result[-1] = _RawUtterance(last.speaker_id, list(last.words) + list(curr.words))
        else:
            # 짧지만 콘텐츠가 있는 이종 화자 발화는 독립 유지
            result.append(curr)

    return result


# -- Step 4: Split long utterances --

def _split_long_utterances(utterances: list[_RawUtterance]) -> list[_RawUtterance]:
    result: list[_RawUtterance] = []
    for u in utterances:
        if u.duration <= config.MAX_UTTERANCE_SEC:
            result.append(u)
        else:
            result.extend(_split_at_midpoint(u))
    return result


def _split_at_midpoint(u: _RawUtterance) -> list[_RawUtterance]:
    if u.duration <= config.MAX_UTTERANCE_SEC or len(u.words) < 2:
        return [u]

    u_start = _to_float(u.words[0].get("start"))
    u_end = _to_float(u.words[-1].get("end"))
    target = (u_start + u_end) / 2.0

    best_idx = -1
    best_score = -1.0

    for i in range(len(u.words) - 1):
        gap = _to_float(u.words[i + 1].get("start")) - _to_float(u.words[i].get("end"))
        mid = (_to_float(u.words[i].get("end")) + _to_float(u.words[i + 1].get("start"))) / 2.0
        proximity = 1.0 / (1.0 + abs(mid - target))
        score = gap * 10.0 + proximity
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return [u]

    left = _RawUtterance(u.speaker_id, u.words[:best_idx + 1])
    right = _RawUtterance(u.speaker_id, u.words[best_idx + 1:])

    return _split_at_midpoint(left) + _split_at_midpoint(right)


# -- Step 5.5: Segmenter v2 후처리 병합 (기본 OFF, 신규 처리 전용) --

def _merge_v2_step(utterances: list[_RawUtterance]) -> list[_RawUtterance]:
    """MERGE_V2_ENABLED=true일 때만 같은-화자 짧은 발화를 문장 종결 기준으로 추가 병합.

    OFF(기본)이면 입력을 그대로 반환 → v1 동작 완전 동일. 재처리와 무관(신규 처리 경로).
    """
    if not config.MERGE_V2_ENABLED or len(utterances) <= 1:
        return utterances

    units = [_to_v2_unit(u) for u in utterances]
    merged = merge_v2(units, bidirectional=config.MERGE_V2_BIDIRECTIONAL)
    return [_RawUtterance(m["speaker_id"], list(m["words"])) for m in merged]


def _to_v2_unit(u: _RawUtterance) -> dict:
    """_RawUtterance → merge_v2 입력 dict (words 패스스루 포함)."""
    text = " ".join(
        w.get("word", w.get("text", "")).strip()
        for w in u.words
        if w.get("word") or w.get("text")
    )
    return {
        "start_sec": _to_float(u.words[0].get("start")) if u.words else 0.0,
        "end_sec": _to_float(u.words[-1].get("end")) if u.words else 0.0,
        "speaker_id": u.speaker_id,
        "transcript_text": text,
        "word_count": len(u.words),
        "words": list(u.words),
    }


# -- Step 6: Apply padding --

def _apply_padding(utterances: list[_RawUtterance], total_duration: float) -> list[UtteranceBoundary]:
    result: list[UtteranceBoundary] = []
    for u in utterances:
        if not u.words:
            continue
        start = _to_float(u.words[0].get("start"))
        end = _to_float(u.words[-1].get("end"))
        padded_start = max(0.0, start - config.PADDING_SEC)
        padded_end = min(total_duration, end + config.PADDING_SEC)
        text = " ".join(
            w.get("word", w.get("text", "")).strip()
            for w in u.words
            if w.get("word") or w.get("text")
        )
        result.append(UtteranceBoundary(
            start_sec=_round2(start),
            end_sec=_round2(end),
            # WAV 추출은 padded 구간으로 진행되므로 duration_sec도 padded 길이로 일치시킴.
            # 이렇게 해야 metadata.duration_sec == 실제 WAV 길이.
            duration_sec=_round2(padded_end - padded_start),
            padded_start_sec=_round2(padded_start),
            padded_end_sec=_round2(padded_end),
            speaker_id=u.speaker_id,
            transcript_text=text,
            words=tuple(u.words),
        ))
    return result


# -- Utilities --

def _get_speaker_id(word: dict) -> str:
    # SPEAKER_NN 패턴 (자릿수 패딩) 유지 — 다른 코드 경로와 일관성
    return str(word.get("speakerId", word.get("speaker", "SPEAKER_00")))


def _to_float(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def _round2(val: float) -> float:
    return round(val * 100.0) / 100.0
