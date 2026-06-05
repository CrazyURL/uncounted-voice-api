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

# IVR/ARS 도입부 전용 화자 라벨. 사람(SPEAKER_00/01)과 구분해 발화를 '삭제'하지 않고
# 별도 라벨로 보존한다(대표님 결정 2026-06-05). renumber_speakers_in_place 는 seg["speaker"]
# 기준으로만 매핑을 만들고 이 라벨은 word/utterance 에만 존재하므로 자동으로 통과(미변경)된다.
_IVR_LABEL = "SPEAKER_IVR"


def is_enabled() -> bool:
    return os.environ.get("VOICE_NEMO_FULL_DIAR_ENABLED", "false").strip().lower() == "true"


def _ivr_exclude_enabled() -> bool:
    """IVR/ARS 도입부 클러스터 제외 게이트(기본 OFF, 카나리 후 활성)."""
    return os.environ.get("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "false").strip().lower() == "true"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _identify_ivr_speakers(
    by_spk: dict[str, list[tuple[float, float]]], duration_sec: float
) -> set[str]:
    """IVR/ARS 클러스터 식별: 통화 후반부에 미등장하는 화자.

    휴리스틱(별도 음향 감지기 불필요, NeMo 클러스터 시간분포만 사용):
      - last_end < duration × IVR_ABSENT_FRAC (통화 후반부에 한 번도 등장 안 함)
    상담사·고객은 본문 전체에 걸쳐 등장하므로 last_end 가 후반까지 → 사람(유지).
    IVR/ARS 안내멘트는 도입부에만 존재하고 본문에 미등장 → 제외.

    first_start 조건을 두지 않는 이유: IVR 이 여러 안내멘트 클러스터로 쪼개지면
    두 번째 멘트가 통화 중간(t>0)에 시작할 수 있다(실측 eb34a6a: IVR 가 0~96s,
    44~111s 두 클러스터). last_end 만으로 판정하면 둘 다 안전하게 제외된다.
    late-joiner(후반 첫 등장)는 last_end 가 높아(first≤last) 자동으로 사람으로 유지된다.
    """
    absent_frac = _env_float("VOICE_NEMO_IVR_ABSENT_FRAC", 0.5)
    absent_before = duration_sec * absent_frac
    ivr: set[str] = set()
    for spk, spans in by_spk.items():
        if not spans:
            continue
        last_end = max(e for _, e in spans)
        if last_end < absent_before:
            ivr.add(spk)
    return ivr


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

    # IVR 제외 ON → NeMo 자동추정(num_speakers=None) 으로 IVR+상담사+고객 분리 유도.
    # OFF → 기존대로 2명 강제(byte-identical).
    ivr_exclude = _ivr_exclude_enabled()
    req_num_speakers = None if ivr_exclude else 2

    # NeMo 전체 윈도우(통화 길이). 서비스가 window_seconds 만큼 처리.
    nemo = hd._call_nemo(audio_path, float(duration_sec), num_speakers=req_num_speakers)
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

    # 진단: 각 NeMo 클러스터의 시간분포(first_start/last_end/총발화/턴수).
    if ivr_exclude:
        spans_dbg = {
            spk: (round(min(s for s, _ in v), 1), round(max(e for _, e in v), 1),
                  round(sum(e - s for s, e in v), 1), len(v))
            for spk, v in by_spk.items()
        }
        logger.info("[%s] nemo_full DEBUG clusters(first,last,dur,n) dur_total=%.0f: %s",
                    task_id, duration_sec, spans_dbg)

    # IVR/ARS 클러스터 제외 → 남은 화자만 사람(상담사·고객)으로 매핑.
    # ivr_end = IVR 클러스터들의 최대 종료시각 = 도입부(ARS 안내멘트)가 끝나는 지점.
    # 이 시점 이전 단어는 사람 클러스터(통화 전체 존속)에 IVR 음성이 흡수된 잔여까지
    # 포함해 모두 null 처리한다(태스크 전략: "도입부를 화자분리 대상에서 분리 제외").
    ivr_spks: set[str] = set()
    ivr_end = 0.0
    if ivr_exclude and len(by_spk) > 2:
        ivr_spks = _identify_ivr_speakers(by_spk, float(duration_sec))
        if ivr_spks:
            ivr_end = max(max(e for _, e in by_spk[spk]) for spk in ivr_spks)
            logger.info("[%s] nemo_full: IVR 클러스터 제외 %s (전체 %d명), 도입부 cutoff=%.1fs",
                        task_id, sorted(ivr_spks), len(by_spk), ivr_end)
            by_spk = {k: v for k, v in by_spk.items() if k not in ivr_spks}

    if len(by_spk) != 2:
        logger.info("[%s] nemo_full: 화자 %d명(IVR 제외 후 2 아님) — fallback", task_id, len(by_spk))
        return result

    f0 = hd._f0_medians_from_segments(audio, sample_rate, by_spk)
    if len(f0) != 2:
        return result
    (lo_spk, lo_f0), (hi_spk, hi_f0) = sorted(f0.items(), key=lambda x: x[1])
    if (hi_f0 - lo_f0) < _INTRO_F0_MIN_DIFF_HZ:
        if ivr_spks:
            # IVR 제외로 2화자를 이미 깨끗이 분리한 경우: F0 가 가까워도 fallback(=붕괴) 대신
            # F0 순서대로 라벨을 배정해 '분리' 자체를 보존한다. 본인/상대 역할은 저신뢰지만
            # 다운스트림(relation/admin)에서 교정 가능 — 붕괴는 복구 불가하므로 분리를 우선한다.
            logger.info("[%s] nemo_full: F0 차 %.1fHz < %.1f 이나 IVR 제외 2화자 — 순서배정(분리 우선)",
                        task_id, hi_f0 - lo_f0, _INTRO_F0_MIN_DIFF_HZ)
        else:
            logger.info("[%s] nemo_full: F0 차 %.1fHz < %.1f — 본인/상대 매핑 불가, fallback",
                        task_id, hi_f0 - lo_f0, _INTRO_F0_MIN_DIFF_HZ)
            return result
    label_of = {lo_spk: ad._LABEL_LOW, hi_spk: ad._LABEL_HIGH}
    logger.info("[%s] nemo_full: F0 %s → %s (저음 %s=%.0fHz)",
                task_id, {k: round(v, 1) for k, v in f0.items()}, label_of, lo_spk, lo_f0)

    # 각 word 를 시작점이 포함된 NeMo turn 의 화자로 배정.
    # IVR turn 에 속한 word → speaker=None(utterance_segmenter 가 필터 → 발화 제외).
    overwritten = 0
    ivr_nulled = 0
    new_segments = []
    for seg in result.get("segments") or []:
        new_seg = dict(seg)
        seg_words = seg.get("words") or []
        # word 타임스탬프 없는 segment 는 word 단위 null 을 우회한다. 도입부(start<cutoff)면
        # segment-level 로 제외 마킹(IVR 안내멘트 꼬리). stt_processor 가 source 를 전파한다.
        if not seg_words:
            ss = seg.get("start")
            if (ss is not None and ivr_end > 0.0 and float(ss) < ivr_end
                    and new_seg.get("speaker") != _IVR_LABEL):
                new_seg["speaker"] = _IVR_LABEL
                new_seg["speaker_source"] = "nemo_full_ivr"
                ivr_nulled += 1
            new_segments.append(new_seg)
            continue
        new_words = []
        for w in seg_words:
            nw = dict(w)
            s = w.get("start")
            if s is not None:
                nspk = ad._nemo_spk_at_start(turns, float(s))
                # IVR 클러스터 소속 OR 도입부 cutoff 이전 → null(도입부 분리 제외).
                # cutoff 은 사람 클러스터에 흡수된 잔여 IVR 음성까지 함께 제거한다.
                if nspk in ivr_spks or (ivr_end > 0.0 and float(s) < ivr_end):
                    # 도입부 단어 → IVR 전용 라벨(삭제 아님·보존). 사람 라벨과 구분된다.
                    if nw.get("speaker") != _IVR_LABEL:
                        nw["speaker"] = _IVR_LABEL
                        nw["speaker_source"] = "nemo_full_ivr"
                        ivr_nulled += 1
                else:
                    lab = label_of.get(nspk)
                    if lab is not None and nw.get("speaker") != lab:
                        nw["speaker"] = lab
                        nw["speaker_source"] = "nemo_full"
                        overwritten += 1
            new_words.append(nw)
        if new_words:
            new_seg["words"] = new_words
        new_segments.append(new_seg)

    logger.info("[%s] nemo_full: NeMo 전체재분리 완료 turns=%d overwritten=%d ivr_nulled=%d",
                task_id, len(turns), overwritten, ivr_nulled)
    out = dict(result)
    out["segments"] = new_segments
    return out
