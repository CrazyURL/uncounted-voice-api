"""Raw pyannote direct mapping — word midpoint 기준 speaker 라벨 + sub-segment 분할.

`whisperx.assign_word_speakers` 의 overlap 흡수 / 짧은 back-channel 누락 문제를 우회한다.
설계 정본: scripts/analysis/stt_pipeline_raw_pyannote_mapping_plan_20260531.md

알고리즘 (per word):
  1. word.mid = (word.start + word.end) / 2
  2. mid 가 exactly 1개 pyannote segment 안 → 그 speaker (source="exact")
  3. mid 가 ≥2 segment 안 (overlap) → speaker=None, source="overlap_<A>_<B>" (강제 X)
  4. mid 가 segment 밖 ≤ TOLERANCE_DEFAULT_MS → nearest speaker, source="tolerance_<ms>ms"
  5. mid 가 segment 밖 TOLERANCE_DEFAULT_MS < dist ≤ TOLERANCE_MAX_MS
     AND word_dur ≤ BACKCHANNEL_DUR_MAX → nearest speaker, source="backchannel_<ms>ms"
  6. 그 외 → speaker=None, source="ambiguous" (강제 배정 금지)

per segment:
  - word 의 speaker 가 바뀌면 sub-segment 분할 (mixed 금지)
  - speaker change 강제 병합 금지 (None ↔ speaker 변화도 분할 지점)
  - overlap_ranges 메타 attach (pyannote 가 ≥50ms overlap 보고한 구간 보존)

호환: whisperx.assign_word_speakers 와 동일한 schema (result dict, segments list).
  단 신규 메타: word.speaker_source / segment.source_distribution / segment.overlap_ranges.
"""
from __future__ import annotations

from typing import Any, Iterable

# 기본 tolerance — config.py 가 import 후 override 가능
TOLERANCE_DEFAULT_MS = 150
TOLERANCE_MAX_MS = 300
BACKCHANNEL_DUR_MAX = 0.7
OVERLAP_MIN_SEC = 0.05  # 50ms 이상만 overlap 으로 인정


def _diar_to_list(diar_segments: Any) -> list[dict]:
    """pandas DataFrame / list / iterable → list[dict] 변환."""
    if hasattr(diar_segments, "to_dict"):
        return diar_segments.to_dict(orient="records")
    if isinstance(diar_segments, list):
        return diar_segments
    if isinstance(diar_segments, Iterable):
        return list(diar_segments)
    return []


def _assign_one(
    word_start: float | None,
    word_end: float | None,
    diar_list: list[dict],
    tolerance_default_s: float,
    tolerance_max_s: float,
    backchannel_dur_max_s: float,
) -> tuple[str | None, str]:
    """반환: (speaker | None, source string)."""
    if word_start is None or word_end is None:
        return None, "no_timestamp"

    mid = (word_start + word_end) / 2.0
    word_dur = max(0.0, word_end - word_start)

    # Step 1: exact 구간 안 (overlap 포함)
    inside: list[dict] = []
    for d in diar_list:
        ds = d.get("start", 0.0) or 0.0
        de = d.get("end", 0.0) or 0.0
        if ds <= mid <= de:
            inside.append(d)
    if len(inside) == 1:
        return inside[0].get("speaker"), "exact"
    if len(inside) >= 2:
        speakers = sorted({d.get("speaker") for d in inside if d.get("speaker")})
        return None, "overlap_" + "_".join(speakers)

    # Step 2/3: 가까운 segment 찾기
    best: dict | None = None
    best_dist = float("inf")
    for d in diar_list:
        ds = d.get("start", 0.0) or 0.0
        de = d.get("end", 0.0) or 0.0
        dist = min(abs(mid - ds), abs(mid - de))
        if dist < best_dist:
            best_dist = dist
            best = d
    if best is None:
        return None, "no_diar_segments"

    if best_dist <= tolerance_default_s:
        return best.get("speaker"), f"tolerance_{int(best_dist * 1000)}ms"

    if best_dist <= tolerance_max_s and word_dur <= backchannel_dur_max_s:
        return best.get("speaker"), f"backchannel_{int(best_dist * 1000)}ms"

    return None, "ambiguous"


def _split_on_speaker_change(
    words: list[dict],
    parent_text: str,
) -> list[dict]:
    """word.speaker 변화 시 sub-segment 분할.
    None ↔ speaker 변화도 분할 지점 (ambiguous 구간 보존).
    """
    if not words:
        return []

    sub_segs: list[dict] = []
    cur_words: list[dict] = []
    cur_speaker: Any = "__INIT__"

    for w in words:
        sp = w.get("speaker")
        if cur_words and sp != cur_speaker:
            sub_segs.append(_build_segment(cur_words, cur_speaker, parent_text))
            cur_words = []
        cur_words.append(w)
        cur_speaker = sp

    if cur_words:
        sub_segs.append(_build_segment(cur_words, cur_speaker, parent_text))
    return sub_segs


def _build_segment(words: list[dict], speaker: Any, parent_text: str) -> dict:
    starts = [w.get("start") for w in words if w.get("start") is not None]
    ends = [w.get("end") for w in words if w.get("end") is not None]
    text = " ".join((w.get("word") or "").strip() for w in words).strip()
    sources = [w.get("speaker_source") for w in words if w.get("speaker_source")]
    src_dist: dict[str, int] = {}
    for s in sources:
        # 그룹화: exact / overlap_* / tolerance_* / backchannel_* / ambiguous / no_*
        if s.startswith("overlap_"):
            key = "overlap"
        elif s.startswith("tolerance_"):
            key = "tolerance"
        elif s.startswith("backchannel_"):
            key = "backchannel"
        else:
            key = s
        src_dist[key] = src_dist.get(key, 0) + 1
    return {
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "text": text,
        "speaker": speaker if speaker != "__INIT__" else None,
        "words": words,
        "parent_segment_text": parent_text,
        "source_distribution": src_dist,
    }


def _compute_overlap_ranges(
    diar_list: list[dict],
    overlap_min_s: float = OVERLAP_MIN_SEC,
) -> list[dict]:
    overlaps: list[dict] = []
    n = len(diar_list)
    for i in range(n):
        a = diar_list[i]
        for j in range(i + 1, n):
            b = diar_list[j]
            os_ = max(a.get("start", 0.0) or 0.0, b.get("start", 0.0) or 0.0)
            oe = min(a.get("end", 0.0) or 0.0, b.get("end", 0.0) or 0.0)
            if (oe - os_) >= overlap_min_s:
                overlaps.append({
                    "start": os_,
                    "end": oe,
                    "speakers": sorted({a.get("speaker"), b.get("speaker")} - {None}),
                })
    return overlaps


def _attach_overlap_ranges(
    out_segments: list[dict],
    overlaps: list[dict],
) -> None:
    """segment 별로 자기 구간과 겹치는 overlap 영역 메타 부착 (in-place)."""
    for seg in out_segments:
        ss = seg.get("start") or 0.0
        se = seg.get("end") or 0.0
        seg["overlap_ranges"] = [
            ov for ov in overlaps
            if not (se < ov["start"] or ss > ov["end"])
        ]


def assign_speakers(
    diarize_segments: Any,
    result: dict,
    tolerance_default_ms: int = TOLERANCE_DEFAULT_MS,
    tolerance_max_ms: int = TOLERANCE_MAX_MS,
    backchannel_dur_max: float = BACKCHANNEL_DUR_MAX,
    overlap_min_s: float = OVERLAP_MIN_SEC,
) -> dict:
    """raw pyannote timeline 직접 매핑.

    Args:
        diarize_segments: pyannote DataFrame (start/end/speaker) 또는 list[dict]
        result: whisperx.align 출력 (dict with "segments")
        tolerance_default_ms: word mid 가 segment 밖일 때 기본 tolerance (~150ms)
        tolerance_max_ms: back-channel 모드 최대 tolerance (~300ms)
        backchannel_dur_max: back-channel mode 가 적용되는 word duration 상한 (~0.7s)
        overlap_min_s: overlap_ranges 메타에 포함될 최소 overlap 길이 (~50ms)

    Returns:
        whisperx 호환 schema (result dict with updated "segments" list).
        - segments: speaker change 시 sub-segment 분할 (mixed 금지)
        - words: speaker + speaker_source (신규) 라벨
        - segments[].source_distribution: 라벨 출처 통계 (신규)
        - segments[].overlap_ranges: 겹치는 pyannote overlap 구간 (신규)
        - segments[].parent_segment_text: 분할 전 원 segment text (신규)
    """
    diar_list = _diar_to_list(diarize_segments)
    tolerance_default_s = tolerance_default_ms / 1000.0
    tolerance_max_s = tolerance_max_ms / 1000.0

    overlaps = _compute_overlap_ranges(diar_list, overlap_min_s)
    in_segments = result.get("segments") or []
    out_segments: list[dict] = []

    for seg in in_segments:
        parent_text = (seg.get("text") or "").strip()
        words = seg.get("words") or []

        if not words:
            new_seg = dict(seg)
            new_seg["speaker"] = None
            new_seg["words"] = []
            new_seg["parent_segment_text"] = parent_text
            new_seg["source_distribution"] = {"no_words": 1}
            out_segments.append(new_seg)
            continue

        labeled_words: list[dict] = []
        for w in words:
            sp, src = _assign_one(
                w.get("start"),
                w.get("end"),
                diar_list,
                tolerance_default_s,
                tolerance_max_s,
                backchannel_dur_max,
            )
            new_w = dict(w)
            new_w["speaker"] = sp
            new_w["speaker_source"] = src
            labeled_words.append(new_w)

        sub_segs = _split_on_speaker_change(labeled_words, parent_text)
        out_segments.extend(sub_segs)

    _attach_overlap_ranges(out_segments, overlaps)

    out = dict(result)
    out["segments"] = out_segments
    return out
