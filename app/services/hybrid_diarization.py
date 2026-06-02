"""하이브리드 발화분리 — 도입부 N초만 NeMo MSDD 고해상도 재분리 후 오버라이트.

배경(2026-06-02 PoC 측정):
  pyannote 3.1 은 통화 도입부의 짧은 turn(0.3~0.45s 의 "여보세요/네/바쁘세요")을
  단일 화자로 뭉쳐 화자가 뒤바뀐다(GT1 4-turn → 1 화자). NeMo MSDD(telephonic,
  multi-scale)는 같은 구간을 정확히 분리한다. 단 전체에 NeMo 를 돌리면 짧은 turn 이
  전체 통계에 묻혀 다시 실패 + VRAM 7.7GB(임계초과). 따라서 도입부 30s 윈도우만
  NeMo 로 재분리하고 그 결과로 word.speaker 를 오버라이트하는 하이브리드가 최적.
  (측정: 10s/20s/전체=실패, 15s/30s/60s=성공. 30s = 정확도·자원 sweet spot.)

ID 통일:
  NeMo speaker_0/1 ↔ pyannote SPEAKER_00/01 매핑은 **임베딩 코사인이 주력**이다
  (PoC2.5: margin +0.19~0.83). pyannote 도입부가 단일화자로 뭉친 경우 시간-overlap
  매핑은 1:1 보장이 깨질 수 있어, overlap 은 코사인 실패 시 백업으로만 쓴다.

안전:
  - env gate VOICE_HYBRID_DIAR_ENABLED (기본 false) → 꺼지면 호출 자체 안 함(무회귀).
  - NeMo 서비스 미응답/타임아웃/저신뢰 → 원본 word 유지(fallback, 무중단).
  - DB·오디오 원본 미변경. word.speaker 메모리 오버라이트만.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

WINDOW_SEC_DEFAULT = 30.0
_COSINE_MIN_MARGIN = 0.05   # 코사인 1:1 매핑 최소 margin (이보다 작으면 모호 → overlap 백업)


def is_enabled() -> bool:
    return os.environ.get("VOICE_HYBRID_DIAR_ENABLED", "false").strip().lower() == "true"


def _window_sec() -> float:
    try:
        return float(os.environ.get("VOICE_HYBRID_DIAR_WINDOW_SEC", str(WINDOW_SEC_DEFAULT)))
    except ValueError:
        return WINDOW_SEC_DEFAULT


def _nemo_endpoint() -> str:
    return os.environ.get("VOICE_HYBRID_NEMO_ENDPOINT", "http://localhost:8009/api/diarize/intro")


def _cos(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    import numpy as np
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _map_by_cosine(
    pyannote_embeddings: dict[str, list[float]],
    nemo_embeddings: dict[str, list[float]],
) -> dict[str, str] | None:
    """NeMo spk → pyannote spk 코사인 1:1 매핑. 모호하면 None(백업으로 위임)."""
    if not pyannote_embeddings or not nemo_embeddings:
        return None
    mapping: dict[str, str] = {}
    used_py: set[str] = set()
    for n_spk, n_emb in nemo_embeddings.items():
        sims = sorted(
            ((p_spk, _cos(n_emb, p_emb)) for p_spk, p_emb in pyannote_embeddings.items()),
            key=lambda x: (x[1] is not None, x[1] or -1.0),
            reverse=True,
        )
        sims = [(p, s) for p, s in sims if s is not None]
        if not sims:
            return None
        best_p, best_s = sims[0]
        second_s = sims[1][1] if len(sims) > 1 else -1.0
        if best_s - second_s < _COSINE_MIN_MARGIN:
            return None  # 1·2위 차이 모호 → 코사인 포기
        if best_p in used_py:
            return None  # 두 NeMo 가 같은 pyannote 로 → 1:1 깨짐
        mapping[n_spk] = best_p
        used_py.add(best_p)
    return mapping


def _map_by_overlap(
    pyannote_turns: list[tuple[float, float, str]],
    nemo_turns: list[dict],
    window_limit: float,
) -> dict[str, str]:
    """백업: 시간 overlap 면적 기반 NeMo spk → pyannote spk 매핑.

    pyannote 도입부가 단일화자로 뭉친 경우 1:1 이 안 될 수 있으므로 코사인 실패시만.
    """
    overlap: dict[tuple[str, str], float] = {}
    for p_st, p_ed, p_spk in pyannote_turns:
        if p_st >= window_limit:
            continue
        p_ed_clip = min(p_ed, window_limit)
        for n in nemo_turns:
            inter_st = max(p_st, n["start"])
            inter_ed = min(p_ed_clip, n["end"])
            if inter_st < inter_ed:
                key = (p_spk, n["nemo_spk"])
                overlap[key] = overlap.get(key, 0.0) + (inter_ed - inter_st)
    mapping: dict[str, str] = {}
    used_py: set[str] = set()
    for (p_spk, n_spk), _ov in sorted(overlap.items(), key=lambda x: x[1], reverse=True):
        if n_spk not in mapping and p_spk not in used_py:
            mapping[n_spk] = p_spk
            used_py.add(p_spk)
    return mapping


def _call_nemo(audio_path: str, window_sec: float) -> dict | None:
    """NeMo 마이크로서비스 호출. 실패 시 None(fallback)."""
    try:
        import requests
    except ImportError:
        logger.warning("[hybrid_diar] requests 미설치 — fallback")
        return None
    try:
        resp = requests.post(
            _nemo_endpoint(),
            json={"audio_path": audio_path, "window_seconds": window_sec},
            timeout=float(os.environ.get("VOICE_HYBRID_NEMO_TIMEOUT", "60")),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — 무중단 fallback 이 목적
        logger.warning("[hybrid_diar] NeMo 서비스 호출 실패 — fallback: %s", type(exc).__name__)
        return None


def _pyannote_turns_from_segments(segments: list[dict]) -> list[tuple[float, float, str]]:
    """segments → (start,end,speaker) turn 리스트 (overlap 백업 매핑용)."""
    turns: list[tuple[float, float, str]] = []
    for seg in segments:
        spk = seg.get("speaker")
        if spk and seg.get("start") is not None and seg.get("end") is not None:
            turns.append((float(seg["start"]), float(seg["end"]), spk))
    return turns


def apply_hybrid_intro(
    result: dict,
    audio_path: str,
    pyannote_embeddings: dict[str, list[float]] | None = None,
    *,
    window_sec: float | None = None,
) -> dict:
    """도입부 N초 word.speaker 를 NeMo 결과로 오버라이트한 새 result 반환.

    Args:
        result: whisperx/raw_direct 출력 (segments[].words[].speaker 포함).
        audio_path: 원본 오디오 경로 (NeMo 서비스가 직접 읽음).
        pyannote_embeddings: {pyannote_spk: 256-dim} 도입부 화자 임베딩(코사인 매핑용).
                             None 이면 overlap 백업만 사용.
        window_sec: 도입부 윈도우(기본 env/30s).

    Returns:
        새 result dict. 게이트 OFF/NeMo 실패/매핑 실패 시 입력을 그대로 반환(무변경).
    """
    if not is_enabled():
        return result
    win = window_sec if window_sec is not None else _window_sec()

    nemo = _call_nemo(audio_path, win)
    if not nemo or nemo.get("status") != "success" or not nemo.get("turns"):
        return result
    nemo_turns: list[dict] = nemo["turns"]
    nemo_embeddings: dict[str, list[float]] = nemo.get("embeddings") or {}

    segments = result.get("segments") or []
    pyannote_turns = _pyannote_turns_from_segments(segments)

    # ── ID 매핑: 코사인 주력 → 실패 시 overlap 백업 ──
    mapping = _map_by_cosine(pyannote_embeddings or {}, nemo_embeddings)
    map_src = "cosine"
    if mapping is None:
        mapping = _map_by_overlap(pyannote_turns, nemo_turns, win)
        map_src = "overlap"
    if not mapping:
        logger.warning("[hybrid_diar] ID 매핑 실패 — fallback")
        return result

    # ── 하드 오버라이트 (도입부 윈도우 내 word 만) ──
    def nemo_spk_at(t: float) -> str | None:
        for n in nemo_turns:
            if n["start"] <= t <= n["end"]:
                return n["nemo_spk"]
        return None

    overwritten = 0
    new_segments: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        words = seg.get("words") or []
        new_words = []
        for wd in words:
            nw = dict(wd)
            ws = nw.get("start")
            if ws is not None and ws < win:
                n_spk = nemo_spk_at(float(ws))
                if n_spk is not None and n_spk in mapping:
                    if nw.get("speaker") != mapping[n_spk]:
                        nw["speaker"] = mapping[n_spk]
                        nw["speaker_source"] = "hybrid_nemo_intro"
                        overwritten += 1
                # NeMo 가 침묵으로 본 word 는 원본 유지(fallback)
            new_words.append(nw)
        if new_words:
            new_seg["words"] = new_words
        new_segments.append(new_seg)

    logger.info(
        "[hybrid_diar] 도입부 오버라이트 완료 win=%.0fs map=%s overwritten=%d",
        win, map_src, overwritten,
    )
    out = dict(result)
    out["segments"] = new_segments
    return out
