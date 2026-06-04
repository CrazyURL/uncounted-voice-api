"""NeMo 전체재분리 화자분리 (단기 통화 ≤임계 전용).

동적 스위칭 확정(2026-06-03, 실측 2026-06-05):
  - ≤~200초 통화 → **NeMo 전체재분리**: 전체 통화를 NeMo MSDD 로 분리하면 도입부
    여보세요#1/#2 까지 정확(90~210s 윈도우 sweet spot). VRAM 동거 안전(peak ~7.6GB).
  - >~200초 → anchor (OOM 방어, 도입부 best-effort). [[anchor_diarization]]

anchor 와 달리 임베딩/코사인 없이 **NeMo turn → word.speaker 직접 매핑**(word 시작점이
포함된 turn 의 화자). NeMo 화자 ↔ SPEAKER_label 은 도입부 F0(저음=본인=SP01) 로 결정.

안전:
  - env gate VOICE_NEMO_FULL_DIAR_ENABLED (기본 false).
  - NeMo 미응답 / 2화자 아님 / F0 매핑 불가 → 원본 result 반환(fallback, 무중단).
  - DB·오디오 미변경. word.speaker 메모리 오버라이트만.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.services import hybrid_diarization as hd
from app.services import anchor_diarization as ad

logger = logging.getLogger(__name__)

_INTRO_F0_MIN_DIFF_HZ = 8.0  # 두 화자 F0 차가 이보다 작으면 본인/상대 매핑 불가 → fallback


def is_enabled() -> bool:
    return os.environ.get("VOICE_NEMO_FULL_DIAR_ENABLED", "false").strip().lower() == "true"


def apply_nemo_full_diarization(
    result: dict,
    audio_path: str,
    audio: Any,
    sample_rate: int,
    duration_sec: float,
    task_id: str = "",
) -> dict:
    """전체 통화를 NeMo 로 재분리해 word.speaker 를 직접 매핑한 새 result 반환.

    게이트 OFF / NeMo 실패 / 2화자 아님 / F0 매핑 불가 시 입력 그대로(무변경).
    """
    if not is_enabled():
        return result

    # NeMo 전체 윈도우(통화 길이). 서비스가 window_seconds 만큼 처리.
    nemo = hd._call_nemo(audio_path, float(duration_sec))
    if not nemo or nemo.get("status") != "success" or not nemo.get("turns"):
        logger.info("[%s] nemo_full: NeMo 미응답 — fallback", task_id)
        return result
    turns = nemo["turns"]

    # NeMo 화자별 turn 묶기 → F0 로 저음=SP01 / 고음=SP00 매핑.
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        try:
            by_spk.setdefault(t["nemo_spk"], []).append((float(t["start"]), float(t["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(by_spk) != 2:
        logger.info("[%s] nemo_full: 화자 %d명(2 아님) — fallback", task_id, len(by_spk))
        return result

    f0 = hd._f0_medians_from_segments(audio, sample_rate, by_spk)
    if len(f0) != 2:
        return result
    (lo_spk, lo_f0), (hi_spk, hi_f0) = sorted(f0.items(), key=lambda x: x[1])
    if (hi_f0 - lo_f0) < _INTRO_F0_MIN_DIFF_HZ:
        logger.info("[%s] nemo_full: F0 차 %.1fHz < %.1f — 본인/상대 매핑 불가, fallback",
                    task_id, hi_f0 - lo_f0, _INTRO_F0_MIN_DIFF_HZ)
        return result
    label_of = {lo_spk: ad._LABEL_LOW, hi_spk: ad._LABEL_HIGH}
    logger.info("[%s] nemo_full: F0 %s → %s (저음 %s=%.0fHz)",
                task_id, {k: round(v, 1) for k, v in f0.items()}, label_of, lo_spk, lo_f0)

    # 각 word 를 시작점이 포함된 NeMo turn 의 화자로 배정.
    overwritten = 0
    new_segments = []
    for seg in result.get("segments") or []:
        new_seg = dict(seg)
        new_words = []
        for w in (seg.get("words") or []):
            nw = dict(w)
            s = w.get("start")
            if s is not None:
                nspk = ad._nemo_spk_at_start(turns, float(s))
                lab = label_of.get(nspk)
                if lab is not None and nw.get("speaker") != lab:
                    nw["speaker"] = lab
                    nw["speaker_source"] = "nemo_full"
                    overwritten += 1
            new_words.append(nw)
        if new_words:
            new_seg["words"] = new_words
        new_segments.append(new_seg)

    logger.info("[%s] nemo_full: NeMo 전체재분리 완료 turns=%d overwritten=%d", task_id, len(turns), overwritten)
    out = dict(result)
    out["segments"] = new_segments
    return out
