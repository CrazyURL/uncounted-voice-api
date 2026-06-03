"""Cross-talk (overlapping speech) detection for utterance-level enrichment.

Uses pyannote's overlap-aware diarization Annotation
(``DiarizeOutput.speaker_diarization.get_overlap()``) to find true
simultaneous-speech regions, then attaches per-utterance metadata.

Design decisions (locked 2026-06-02):
- NO source separation. On single-channel call audio it injects phase
  artifacts and, by splitting one turn into two tracks, breaks total-duration
  / timeline integrity. We only DETECT + FLAG. Audio is never modified.
- Region cutoff filters noise at the source: an overlap region must last
  >= ``cutoff_sec`` (default 0.2s) to count — sub-0.2s blips are coughs /
  backchannel artifacts (empirically verified on session f5414ac6).
- Attribution rule: a region is clipped to each utterance's [start, end];
  any positive intersection flags that utterance. We favour recall, because
  ``is_overlapping == False`` is sold as a premium guarantee and a missed
  overlap (false-negative) silently contaminates the premium tier.
"""
from __future__ import annotations

from typing import Any

DEFAULT_CUTOFF_SEC = 0.2
_CLIP_EPS = 1e-3  # ignore float-noise slivers when clipping to utterance bounds


def extract_overlap_regions(
    diarize_output: Any,
    cutoff_sec: float = DEFAULT_CUTOFF_SEC,
) -> list[tuple[float, float]]:
    """Return ``[(start, end), ...]`` regions where >=2 speakers are active.

    Accepts a pyannote 4.x ``DiarizeOutput`` (uses ``.speaker_diarization``) or a
    raw ``Annotation`` exposing ``get_overlap()``. Regions shorter than
    ``cutoff_sec`` are dropped. Returns ``[]`` when overlap info is unavailable
    (e.g. the whisperx wrapper's exclusive dataframe) — callers must treat that
    as "unknown", not "no overlap".
    """
    ann = getattr(diarize_output, "speaker_diarization", diarize_output)
    get_overlap = getattr(ann, "get_overlap", None)
    if get_overlap is None:
        return []
    regions: list[tuple[float, float]] = []
    for seg in get_overlap():
        start = float(seg.start)
        end = float(seg.end)
        if end - start >= cutoff_sec:
            regions.append((start, end))
    regions.sort()
    return regions


def utterance_overlap_features(
    utt_start: float | None,
    utt_end: float | None,
    overlap_regions: list[tuple[float, float]],
) -> dict[str, Any]:
    """Compute overlap features for one utterance ``[utt_start, utt_end]``.

    Each (already noise-filtered) region is clipped to the utterance bounds;
    a clipped piece with positive duration counts. Returns a dict with:
    ``is_overlapping``, ``overlap_count``, ``overlap_total_sec``,
    ``overlap_ratio`` (overlap / utterance duration), ``overlap_intervals``.
    """
    us = float(utt_start or 0.0)
    ue = float(utt_end or 0.0)
    dur = max(0.0, ue - us)

    intervals: list[dict[str, float]] = []
    for os_, oe in overlap_regions:
        cs = max(us, float(os_))
        ce = min(ue, float(oe))
        if ce - cs > _CLIP_EPS:
            intervals.append({"start_sec": round(cs, 3), "end_sec": round(ce, 3)})

    total = sum(iv["end_sec"] - iv["start_sec"] for iv in intervals)
    ratio = round(total / dur, 4) if dur > 0 else 0.0
    return {
        "is_overlapping": len(intervals) > 0,
        "overlap_count": len(intervals),
        "overlap_total_sec": round(total, 3),
        "overlap_ratio": ratio,
        "overlap_intervals": intervals,
    }
